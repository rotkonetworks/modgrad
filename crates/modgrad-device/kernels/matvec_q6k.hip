// Fused Q6_K matvec: y = dequant(W_q6) @ x   (f32 accumulation).
// Used for the tied lm_head (token_embd is Q6_K). One block per output row.
//
// Q6_K block (210 bytes / 256 elems):
//   +0x00: ql[128]      — low 4 bits
//   +0x80: qh[64]       — high 2 bits
//   +0xC0: scales[16]   — int8 per sub-block
//   +0xD0: d (f16)
// Signed 6-bit quants (offset -32), no min. ggml element layout per
// 128-group (l in 0..32): out[l]=q1, out[l+32]=q2, out[l+64]=q3, out[l+96]=q4.

#include <hip/hip_runtime.h>

extern "C" {

__device__ __forceinline__ float mq6k_f16_to_f32(unsigned short h) {
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

#define Q6K_BLOCK_BYTES 210

__global__ void matvec_q6k_kernel(
    const unsigned char* __restrict__ w_q6,
    const float* __restrict__ x,
    float* __restrict__ y,
    int out_dim, int blocks_per_row)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;
    int tid = threadIdx.x;
    const unsigned char* wrow = w_q6 + (size_t)row * blocks_per_row * Q6K_BLOCK_BYTES;

    // Cache d * scale[16] per block in shared memory.
    extern __shared__ float dscale[];        // [blocks_per_row * 16]
    for (int i = tid; i < blocks_per_row * 16; i += blockDim.x) {
        int blk = i >> 4;        // i/16
        int k   = i & 15;        // i%16
        const unsigned char* b = wrow + (size_t)blk * Q6K_BLOCK_BYTES;
        unsigned short d_h = (unsigned short)b[208] | ((unsigned short)b[209] << 8);
        float d = mq6k_f16_to_f32(d_h); if (!isfinite(d)) d = 0.0f;
        signed char sc = (signed char)b[192 + k];
        dscale[i] = d * (float)sc;
    }
    __syncthreads();

    // ggml position mapping for this thread's element (tid in 0..255).
    int grp   = tid >> 7;         // /128
    int within = tid & 127;
    int quad  = within >> 5;      // 0..3 (q1..q4)
    int l     = within & 31;
    int qlb = grp * 64, qhb = grp * 32, scb = grp * 8;
    int ql_idx = qlb + l + ((quad & 1) ? 32 : 0);
    int sc_idx = scb + (l >> 4) + quad * 2;

    float acc = 0.0f;
    for (int blk = 0; blk < blocks_per_row; ++blk) {
        const unsigned char* b = wrow + (size_t)blk * Q6K_BLOCK_BYTES;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        int nibble = (quad < 2) ? (ql[ql_idx] & 0xF) : (ql[ql_idx] >> 4);
        int qh_bits = (qh[qhb + l] >> (quad * 2)) & 3;
        int q6 = nibble | (qh_bits << 4);
        float w = dscale[blk * 16 + sc_idx] * (float)(q6 - 32);
        acc += w * x[blk * 256 + tid];
    }

    __shared__ float sred[256];
    sred[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) sred[tid] += sred[tid + s]; __syncthreads(); }
    if (tid == 0) y[row] = sred[0];
}

hipError_t launch_matvec_q6k(
    const unsigned char* w_q6, const float* x, float* y,
    int out_dim, int blocks_per_row)
{
    if (out_dim <= 0 || blocks_per_row <= 0) return hipErrorInvalidValue;
    dim3 grid(out_dim);
    dim3 block(256);
    size_t shmem = (size_t)blocks_per_row * 16 * sizeof(float);
    matvec_q6k_kernel<<<grid, block, shmem, 0>>>(w_q6, x, y, out_dim, blocks_per_row);
    return hipGetLastError();
}

} // extern "C"
