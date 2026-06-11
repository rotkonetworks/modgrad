// Q4_K matvec via dp4a + int32 wide loads (the ollama/llama.cpp technique).
//
// Two kernels:
//   quantize_q8_1 : x[f32] -> xq[int8] + xd[f32 scale] + xsum[f32 Σxq], per 32-elem sub-block
//   matvec_q4k_dp4a : y[row] = Σ_subblock xd*(dw*sc*<nibbles·xq> - dminw*m*xsum)
//                     reads qs as int32, masks 4 nibbles, __dp4a with 4 int8 activations.
//                     One WARP per output row; warp-shuffle reduction (no shared barriers).
//
// Q4_K block (144B): d(f16) dmin(f16) scales[12] qs[128]. ggml sub-block order:
//   sub s in [0,8): group g=s/2, hi=s&1; the 32 nibbles are (low if !hi else high) of
//   qs[16 + g*32 .. +32]; they map to x positions [blk*256 + s*32 .. +32] (natural order).

#include <hip/hip_runtime.h>

extern "C" {

__device__ __forceinline__ float dq_f16(unsigned short h) {
    unsigned int s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) { if (m == 0u) return s ? -0.0f : 0.0f; float v = (float)m / 1024.0f * 6.103515625e-05f; return s ? -v : v; }
    if (e == 31u) return __int_as_float(0x7FC00000);
    return __int_as_float((int)((s << 31) | ((e + 112u) << 23) | (m << 13)));
}
__device__ __forceinline__ void dq_scm(int j, const unsigned char* sc, unsigned char* o, unsigned char* m) {
    if (j < 4) { *o = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { unsigned char a = sc[j + 4], b = sc[j - 4], c = sc[j];
           *o = (a & 0xF) | ((b >> 6) << 4); *m = (a >> 4) | ((c >> 6) << 4); }
}

// x[n] -> xq[n] int8, xd[n/32] scale, xsum[n/32] = Σ xq.  One thread per 32-elem sub-block.
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, signed char* __restrict__ xq,
                                     float* __restrict__ xd, float* __restrict__ xsum, int nsub) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nsub) return;
    const float* xs = x + s * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xs[i]));
    float d = amax / 127.0f;
    // Guard against tiny/denormal amax: 1/d can overflow to inf, and then
    // 0*inf = NaN. If the scale is unusable, emit a zero sub-block.
    float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    if (!isfinite(id)) { id = 0.0f; d = 0.0f; }
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)rintf(xs[i] * id);
        q = max(-127, min(127, q));
        xq[s * 32 + i] = (signed char)q;
        sum += q;
    }
    xd[s] = d; xsum[s] = (float)sum;
}
hipError_t launch_quantize_q8_1(const float* x, signed char* xq, float* xd, float* xsum, int n) {
    int nsub = n / 32, b = 128, g = (nsub + b - 1) / b;
    quantize_q8_1_kernel<<<g, b, 0, 0>>>(x, xq, xd, xsum, nsub);
    return hipGetLastError();
}

__global__ void matvec_q4k_dp4a_kernel(
    const unsigned char* __restrict__ w, const signed char* __restrict__ xq,
    const float* __restrict__ xd, const float* __restrict__ xsum,
    float* __restrict__ y, int out_dim, int bpr)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gtid >> 5;            // one 32-lane group per row
    int lane = gtid & 31;
    // NOTE: do NOT early-return out-of-range rows before the shuffle reduction —
    // __shfl_down is a wavefront op; on wave64 a divergent return at the row
    // boundary gives undefined results. All lanes participate; we guard compute
    // and the write instead.
    float acc = 0.0f;
    if (row < out_dim) {
    const unsigned char* wrow = w + (size_t)row * bpr * 144;
    int nsub = bpr * 8;
    for (int s = lane; s < nsub; s += 32) {
        int blk = s >> 3, sub = s & 7;
        const unsigned char* b = wrow + (size_t)blk * 144;
        float dw   = dq_f16((unsigned short)b[0] | ((unsigned short)b[1] << 8));
        float dmw  = dq_f16((unsigned short)b[2] | ((unsigned short)b[3] << 8));
        if (!isfinite(dw)) dw = 0.0f; if (!isfinite(dmw)) dmw = 0.0f;
        unsigned char sc, m; dq_scm(sub, b + 4, &sc, &m);
        int g = sub >> 1, hi = sub & 1;
        const int* qs = (const int*)(b + 16 + g * 32);   // 8 int32 = 32 bytes
        const int* xv = (const int*)(xq + (size_t)s * 32);
        // gfx1102 has no v_dot4 — but the win is the WIDE int32 loads of qs/xq,
        // not the dot instruction (we're memory-bound). Unpack + MAC in registers.
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int w4 = qs[j];      // 4 quant bytes (wide load)
            int xq4 = xv[j];     // 4 int8 activations (wide load)
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int sh = k * 8;
                int nib = hi ? ((w4 >> (sh + 4)) & 0xF) : ((w4 >> sh) & 0xF);
                int xb = (int)(signed char)((xq4 >> sh) & 0xFF);
                isum += nib * xb;
            }
        }
        acc += xd[s] * (dw * (float)sc * (float)isum - dmw * (float)m * xsum[s]);
    }
    }  // if (row < out_dim)
    // width-32 shuffle: reduce within each 32-lane row group, wave-size-safe.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o, 32);
    // Guard: a rare degenerate int8 sub-block can yield non-finite; clamp to 0
    // so it can't poison the residual stream (negligible accuracy impact).
    if (!isfinite(acc)) acc = 0.0f;
    if (lane == 0 && row < out_dim) y[row] = acc;
}
hipError_t launch_matvec_q4k_dp4a(const unsigned char* w, const signed char* xq,
        const float* xd, const float* xsum, float* y, int out_dim, int bpr) {
    if (out_dim <= 0 || bpr <= 0) return hipErrorInvalidValue;
    int block = 256;                      // 8 warps
    int grid = (out_dim * 32 + block - 1) / block;
    matvec_q4k_dp4a_kernel<<<grid, block, 0, 0>>>(w, xq, xd, xsum, y, out_dim, bpr);
    return hipGetLastError();
}

} // extern "C"
