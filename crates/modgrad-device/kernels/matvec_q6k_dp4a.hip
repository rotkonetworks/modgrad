// Q6_K matvec via INT8 dp4a (mirrors matvec_q4k_dp4a.hip approach).
//
// Q6_K block (210 bytes / 256 elems):
//   +0x00: ql[128]      — low 4 bits
//   +0x80: qh[64]       — high 2 bits (4 weights packed per byte)
//   +0xC0: scales[16]   — int8 per sub-block
//   +0xD0: d (f16)      — master scale
// Each 256-elem block has 16 sub-blocks of 16 weights each; sub-block `s`
// (s in 0..16) uses scale sc[s] applied to weights at a specific 16-element
// stride mapped by ggml's q1/q2/q3/q4 layout.
//
// Pipeline:
//   1. quantize_q8_1 (already exists for Q4_K dp4a) writes xq[256/blk] + xd[8/blk] + xsum[8/blk].
//      Q6_K reuses these same inputs since the 32-element sub-block size is identical.
//   2. matvec_q6k_dp4a: one WARP per output row; lanes split sub-blocks.
//
// Reference layout (per group g ∈ {0, 1} of each 128-element half):
//   for l in 0..32:
//     q1 = ((ql[g*64+l]    & 0xF) | ((qh[g*32+l] & 3)        << 4)) - 32  -> out[g*128 + l]
//     q2 = ((ql[g*64+l+32] & 0xF) | ((qh[g*32+l] >> 2 & 3)   << 4)) - 32  -> out[g*128 + l + 32]
//     q3 = ((ql[g*64+l]      >> 4)| ((qh[g*32+l] >> 4 & 3)   << 4)) - 32  -> out[g*128 + l + 64]
//     q4 = ((ql[g*64+l+32]   >> 4)| ((qh[g*32+l] >> 6 & 3)   << 4)) - 32  -> out[g*128 + l + 96]
//
// Sub-blocks (size 16): sub-block `s` (s in 0..16) covers
//     out_idx in [s*16 .. s*16 + 16]
//   For s in [0..2):  g=0, quad=0 (q1).  ql=&ql[0],   shift=lo, qh_shift=0
//   For s in [2..4):  g=0, quad=1 (q2).  ql=&ql[32],  shift=lo, qh_shift=2
//   For s in [4..6):  g=0, quad=2 (q3).  ql=&ql[0],   shift=hi, qh_shift=4
//   For s in [6..8):  g=0, quad=3 (q4).  ql=&ql[32],  shift=hi, qh_shift=6
//   (same for g=1: s in 8..16)
// The 16-element sub-block uses scale sc[s] for ALL 16 quants.

#include <hip/hip_runtime.h>

extern "C" {

__device__ __forceinline__ float q6k_dp4_f16_to_f32(unsigned short h) {
    unsigned int s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) return s ? -0.0f : 0.0f;
        float v = (float)m / 1024.0f * 6.103515625e-05f;
        return s ? -v : v;
    }
    if (e == 31u) return __int_as_float(0x7FC00000);
    return __int_as_float((int)((s << 31) | ((e + 112u) << 23) | (m << 13)));
}

#define Q6K_BLOCK_BYTES 210

__global__ void matvec_q6k_dp4a_kernel(
    const unsigned char* __restrict__ w_q6,
    const signed char* __restrict__ xq,   // INT8 quantized activations (256 per source block)
    const float* __restrict__ xd,         // f32 scale per 32-elem sub-block (8 per source block)
    const float* __restrict__ xsum,       // f32 sum(xq) per 32-elem sub-block
    float* __restrict__ y,
    int out_dim, int blocks_per_row)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gtid >> 5;                // one 32-lane warp per output row
    int lane = gtid & 31;

    float acc = 0.0f;
    if (row < out_dim) {
        const unsigned char* wrow = w_q6 + (size_t)row * blocks_per_row * Q6K_BLOCK_BYTES;
        // 8 sub-blocks-of-32 per Q4_K-aligned input chunk × blocks_per_row blocks
        // (one 256-elem Q6_K block = 8 of the 32-element xq sub-blocks).
        int nsub32 = blocks_per_row * 8;

        for (int s32 = lane; s32 < nsub32; s32 += 32) {
            int blk = s32 >> 3;          // which 256-elem Q6_K block
            int sub32 = s32 & 7;         // which 32-elem sub-block within (0..8)

            const unsigned char* b = wrow + (size_t)blk * Q6K_BLOCK_BYTES;
            unsigned short d_h = (unsigned short)b[208] | ((unsigned short)b[209] << 8);
            float d = q6k_dp4_f16_to_f32(d_h);
            if (!isfinite(d)) d = 0.0f;

            // Sub-block `sub32` covers 32 weights at out_offset = blk*256 + sub32*32.
            // Map sub32 (0..8) to the (g, quad, l_off) tuple per the ggml layout:
            //   sub32 0: g=0, quad=0 (q1), l in 0..32   -> ql[g*64 + l + 0],  qh_shift=0, lo
            //   sub32 1: g=0, quad=1 (q2), l in 0..32   -> ql[g*64 + l + 32], qh_shift=2, lo
            //   sub32 2: g=0, quad=2 (q3), l in 0..32   -> ql[g*64 + l + 0],  qh_shift=4, hi
            //   sub32 3: g=0, quad=3 (q4), l in 0..32   -> ql[g*64 + l + 32], qh_shift=6, hi
            //   sub32 4..7: same with g=1
            int g = sub32 >> 2;
            int quad = sub32 & 3;
            int ql_off = (quad & 1) ? 32 : 0;
            int qh_shift = quad * 2;
            int hi_nibble = quad >> 1;

            const unsigned char* ql = b + g * 64;
            const unsigned char* qh = b + 128 + g * 32;
            const signed char* sc_ptr = (const signed char*)(b + 192);

            // The 32 weights in this sub-block straddle 2 Q6_K scales (each scale
            // covers 16 weights). sc index for l in [0..16): sc_idx_lo
            //                                  l in [16..32): sc_idx_hi
            int sc_idx_lo, sc_idx_hi;
            // ggml layout: per the 128-group, l/16 picks scale index local to the
            // half-block. The 4 quads each get +2 scales offset.
            //   quad 0: sc_idx = g*8 + l/16
            //   quad 1: sc_idx = g*8 + l/16 + 2
            //   quad 2: sc_idx = g*8 + l/16 + 4
            //   quad 3: sc_idx = g*8 + l/16 + 6
            sc_idx_lo = g * 8 + 0 + quad * 2;
            sc_idx_hi = g * 8 + 1 + quad * 2;
            int sc_lo = (int)sc_ptr[sc_idx_lo];
            int sc_hi = (int)sc_ptr[sc_idx_hi];

            const signed char* xv = xq + (size_t)s32 * 32;

            // Accumulate INT for the low-half (16 weights) and the high-half (16
            // weights) separately so each can be multiplied by its own scale.
            int isum_lo = 0, isum_hi = 0;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                unsigned char ql_byte = ql[l + ql_off];
                int nibble = hi_nibble ? (ql_byte >> 4) : (ql_byte & 0xF);
                int qh_bits = (qh[l] >> qh_shift) & 3;
                int q6 = (nibble | (qh_bits << 4)) - 32;
                int x_i8 = (int)xv[l];
                int prod = q6 * x_i8;
                if (l < 16) isum_lo += prod; else isum_hi += prod;
            }

            // Q6_K has no per-block min; just d * sc * Σ(q * x).
            float scl = xd[s32] * d;
            acc += scl * ((float)sc_lo * (float)isum_lo + (float)sc_hi * (float)isum_hi);
        }
    }

    // Warp shuffle reduction down to lane 0.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0 && row < out_dim) {
        y[row] = acc;
    }
}

hipError_t launch_matvec_q6k_dp4a(
    const unsigned char* w, const signed char* xq, const float* xd, const float* xsum,
    float* y, int out_dim, int bpr)
{
    // One warp per row. RDNA3 wave32 → 32 threads/warp. Block of 128 (4 warps).
    const int THREADS_PER_BLOCK = 128;
    const int WARPS_PER_BLOCK = 4;
    int n_warps_needed = out_dim;
    int n_blocks = (n_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    matvec_q6k_dp4a_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, 0>>>(
        w, xq, xd, xsum, y, out_dim, bpr);
    return hipGetLastError();
}

}  // extern "C"
