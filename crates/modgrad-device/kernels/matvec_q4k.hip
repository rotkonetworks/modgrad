// Fused Q4_K_M matvec: y = dequant(W_q4) @ x   (f32 accumulation)
//
// Stable ROCm/HIP replacement for the hand-rolled KFD `matvec_q4k.s`
// (which crashed Xorg on gfx1102 — raw /dev/kfd dispatch is fragile).
// Dispatched via the HIP runtime (hipLaunchKernel), so faults are caught
// by the runtime instead of wedging the display.
//
// One block (256 threads) per output row. Thread `tid` owns element
// position `tid` within every 256-quant block of the row, accumulates
// its contribution across all `blocks_per_row` blocks, then the block
// reduces to y[row].
//
// CORRECT ggml Q4_K element ordering (NOT all-low-then-all-high):
//   for position p in [0,256):
//     is      = p >> 5                  (sub-block / scale index, 0..7)
//     is_high = (p >> 5) & 1            (which nibble)
//     qs_idx  = ((p >> 6) << 5) | (p & 31)   (byte in qs[128])
//   weight = d*sc[is]*nibble - dmin*m[is];  x position = blk*256 + p
//
// Q4_K_M block (144 bytes / 256 elems): d(f16) dmin(f16) scales[12] qs[128]

#include <hip/hip_runtime.h>

extern "C" {

__device__ __forceinline__ float mq4k_f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 0x1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        float m = ((float)mant) / 1024.0f * 6.103515625e-05f;
        return sign ? -m : m;
    }
    if (exp == 31u) return __int_as_float(0x7FC00000);
    unsigned int bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return __int_as_float((int)bits);
}

// 6-bit (sc, m) extractor, mirrors gguf.rs `q4k_get_scale_min_k4`.
__device__ __forceinline__ void mq4k_sc_m(int j, const unsigned char* scales,
                                          unsigned char* sc, unsigned char* m) {
    if (j < 4) {
        *sc = (unsigned char)(scales[j]     & 63);
        *m  = (unsigned char)(scales[j + 4] & 63);
    } else {
        unsigned char a = scales[j + 4];
        unsigned char b = scales[j - 4];
        unsigned char c = scales[j];
        *sc = (unsigned char)((a & 0x0F) | ((b >> 6) << 4));
        *m  = (unsigned char)((a >> 4)   | ((c >> 6) << 4));
    }
}

__global__ void matvec_q4k_kernel(
    const unsigned char* __restrict__ w_q4,  // [out_dim * blocks_per_row * 144]
    const float* __restrict__ x,             // [blocks_per_row * 256]
    float* __restrict__ y,                   // [out_dim]
    int out_dim, int blocks_per_row)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;                    // element position 0..255 within a block

    const unsigned char* wrow = w_q4 + (size_t)row * blocks_per_row * 144;

    // Cache d*sc and dmin*m for every (block, sub-block) in shared memory —
    // decoded once instead of redundantly per thread. (Known-good version.)
    extern __shared__ float sh[];
    float* dsc = sh;                          // [blocks_per_row*8]
    float* dm  = sh + blocks_per_row * 8;     // [blocks_per_row*8]
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x) {
        int blk = i >> 3;
        int sub = i & 7;
        const unsigned char* b = wrow + (size_t)blk * 144;
        unsigned short d_h    = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
        unsigned short dmin_h = (unsigned short)b[2] | ((unsigned short)b[3] << 8);
        float d    = mq4k_f16_to_f32(d_h);    if (!isfinite(d))    d = 0.0f;
        float dmin = mq4k_f16_to_f32(dmin_h); if (!isfinite(dmin)) dmin = 0.0f;
        unsigned char sc, m;
        mq4k_sc_m(sub, b + 4, &sc, &m);
        dsc[i] = d * (float)sc;
        dm[i]  = dmin * (float)m;
    }
    __syncthreads();

    int is      = tid >> 5;
    int is_high = (tid >> 5) & 1;
    int qs_idx  = ((tid >> 6) << 5) | (tid & 31);

    float acc = 0.0f;
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        const unsigned char* b = wrow + (size_t)blk * 144;
        unsigned char qs_byte = b[16 + qs_idx];
        int nibble = is_high ? ((qs_byte >> 4) & 0xF) : (qs_byte & 0xF);
        float w = dsc[blk * 8 + is] * (float)nibble - dm[blk * 8 + is];
        acc += w * x[blk * 256 + tid];
    }

    __shared__ float sred[256];
    sred[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = sred[0];
}

// Host launcher. w_q4/x/y are device pointers. Returns hipError_t (0 = ok).
hipError_t launch_matvec_q4k(
    const unsigned char* w_q4,
    const float* x,
    float* y,
    int out_dim,
    int blocks_per_row)
{
    if (out_dim <= 0 || blocks_per_row <= 0) return hipErrorInvalidValue;
    dim3 grid(out_dim);
    dim3 block(256);
    size_t shmem = (size_t)blocks_per_row * 8 * 2 * sizeof(float);
    matvec_q4k_kernel<<<grid, block, shmem, 0>>>(w_q4, x, y, out_dim, blocks_per_row);
    return hipGetLastError();
}

} // extern "C"
