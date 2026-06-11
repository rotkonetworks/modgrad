// Fused Q5_K matvec: y = dequant(W_q5) @ x   (f32 accumulation)
//
// Same structure as matvec_q4k.hip; Q5_K adds a 5th bit per quant from qh.
//
// Q5_K block (176 bytes / 256 elems):
//   +0x00: d    (f16)
//   +0x02: dmin (f16)
//   +0x04: scales[12]   — 8 sub-blocks of 6-bit (sc,m), canonical layout
//   +0x10: qh[32]       — the high (5th) bit per element
//   +0x30: ql[128]      — 4 low bits per element
//
// ggml element order (interleaved low/high per 32-group), position p in [0,256):
//   is     = p >> 5                       (sub-block / scale, 0..7)
//   is_high= (p >> 5) & 1                 (low or high nibble of ql)
//   ql_idx = ((p >> 6) << 5) | (p & 31)   (byte in ql[128])
//   qh_l   = p & 31                        (byte in qh[32])
//   hbit   = (qh[qh_l] >> is) & 1
//   q5     = nibble + (hbit << 4)          (0..31)
//   weight = d*sc[is]*q5 - dmin*m[is]; x position = blk*256 + p

#include <hip/hip_runtime.h>

extern "C" {

__device__ __forceinline__ float mq5k_f16_to_f32(unsigned short h) {
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

__device__ __forceinline__ void mq5k_sc_m(int j, const unsigned char* scales,
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

#define Q5K_BLOCK_BYTES 176

__global__ void matvec_q5k_kernel(
    const unsigned char* __restrict__ w_q5,
    const float* __restrict__ x,
    float* __restrict__ y,
    int out_dim, int blocks_per_row)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    const unsigned char* wrow = w_q5 + (size_t)row * blocks_per_row * Q5K_BLOCK_BYTES;

    extern __shared__ float sh[];
    float* dsc = sh;
    float* dm  = sh + blocks_per_row * 8;
    for (int i = tid; i < blocks_per_row * 8; i += blockDim.x) {
        int blk = i >> 3;
        int sub = i & 7;
        const unsigned char* b = wrow + (size_t)blk * Q5K_BLOCK_BYTES;
        unsigned short d_h    = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
        unsigned short dmin_h = (unsigned short)b[2] | ((unsigned short)b[3] << 8);
        float d    = mq5k_f16_to_f32(d_h);    if (!isfinite(d))    d = 0.0f;
        float dmin = mq5k_f16_to_f32(dmin_h); if (!isfinite(dmin)) dmin = 0.0f;
        unsigned char sc, m;
        mq5k_sc_m(sub, b + 4, &sc, &m);
        dsc[i] = d * (float)sc;
        dm[i]  = dmin * (float)m;
    }
    __syncthreads();

    int is      = tid >> 5;
    int is_high = (tid >> 5) & 1;
    int ql_idx  = ((tid >> 6) << 5) | (tid & 31);
    int qh_l    = tid & 31;

    float acc = 0.0f;
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        const unsigned char* b = wrow + (size_t)blk * Q5K_BLOCK_BYTES;
        const unsigned char* qh = b + 16;
        const unsigned char* ql = b + 48;
        int nibble = is_high ? ((ql[ql_idx] >> 4) & 0xF) : (ql[ql_idx] & 0xF);
        int hbit = (qh[qh_l] >> is) & 1;
        int q5 = nibble + (hbit << 4);
        float w = dsc[blk * 8 + is] * (float)q5 - dm[blk * 8 + is];
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

hipError_t launch_matvec_q5k(
    const unsigned char* w_q5, const float* x, float* y,
    int out_dim, int blocks_per_row)
{
    if (out_dim <= 0 || blocks_per_row <= 0) return hipErrorInvalidValue;
    dim3 grid(out_dim);
    dim3 block(256);
    size_t shmem = (size_t)blocks_per_row * 8 * 2 * sizeof(float);
    matvec_q5k_kernel<<<grid, block, shmem, 0>>>(w_q5, x, y, out_dim, blocks_per_row);
    return hipGetLastError();
}

} // extern "C"
