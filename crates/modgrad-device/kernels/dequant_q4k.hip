// Q4_K_M dequantize — custom hipcc kernel.
//
// Decodes the GGUF Q4_K_M block format directly into fp32 on device.
// One workgroup per block (256 elements ⇒ one block per grid x).
// Threads cooperate so the 16-byte block header + 128-byte qs payload
// is read once per block, with each thread emitting 1-2 fp32 outputs.
//
// Block format (matches `crates/modgrad-device/src/kfd/gguf.rs`):
//   +0x00: d    (fp16) — super-block scale
//   +0x02: dmin (fp16) — super-block min
//   +0x04: scales[12]  — 8 sub-blocks worth of 6-bit (sc, m) packed
//                        in the canonical llama.cpp layout
//   +0x10: qs[128]     — 256 4-bit quants (low nibble of byte j is
//                        the quant for element j; high nibble is the
//                        quant for element j+128)
//
// Per-element math (sub-block index `is`, low nibble path):
//     out = d * sc[is] * (qs[j] & 0xF) - dmin * m[is]
// Per-element math (sub-block index `is+1`, high nibble path):
//     out = d * sc[is+1] * (qs[j] >> 4) - dmin * m[is+1]
//
// The thread-to-element mapping mirrors the Rust reference's
// dispatch order so the on-device output is bit-identical to the
// host dequant. The kernel is the canonical reader; it does NOT need
// to handle any 16-bit/32-bit byte-order swaps because the format is
// little-endian and we run on little-endian hardware.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`. Linked into the final binary as a static
// .o; the Rust-visible symbol is `launch_dequant_q4k` below.

#include <hip/hip_runtime.h>

#define Q4K_BLOCK_BYTES  144
#define Q4K_BLOCK_ELEMS  256

extern "C" {

// fp16 -> f32. Matches the reader in `kfd::gguf::f16_to_f32`. We
// reproduce the bit-twiddling explicitly rather than rely on
// __half2float so the kernel stays free of header-only conversion
// helpers that vary across HIP versions.
__device__ __forceinline__ float q4k_f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 0x1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        // Subnormal: m / 1024 * 2^-14
        float m = ((float)mant) / 1024.0f * 6.103515625e-05f; // 2^-14
        return sign ? -m : m;
    }
    if (exp == 31u) return __int_as_float(0x7FC00000); // NaN
    unsigned int bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return __int_as_float((int)bits);
}

// 6-bit (sc, m) extractor, mirrors `q4k_get_scale_min_k4`.
//   j ∈ [0, 8). `scales` points at the 12-byte header at offset +0x04.
__device__ __forceinline__ void q4k_get_sc_m(int j, const unsigned char* scales,
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

__global__ void dequant_q4k_kernel(
    const unsigned char* __restrict__ q4k,
    float* __restrict__ fp32_out,
    int n_blocks)
{
    int blk = blockIdx.x;
    if (blk >= n_blocks) return;

    const unsigned char* b = q4k + blk * Q4K_BLOCK_BYTES;
    float* out = fp32_out + blk * Q4K_BLOCK_ELEMS;

    // Cache the 16-byte header in shared memory so the 256 threads
    // don't all hammer global memory for the same bytes.
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ unsigned char s_scales[12];
    if (threadIdx.x == 0) {
        unsigned short d_h    = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
        unsigned short dmin_h = (unsigned short)b[2] | ((unsigned short)b[3] << 8);
        float d    = q4k_f16_to_f32(d_h);
        float dmin = q4k_f16_to_f32(dmin_h);
        // Match the host reader's NaN/inf clamp — keeps the kernel
        // bit-identical to the CPU dequant on malformed GGUF blocks.
        if (!isfinite(d))    d    = 0.0f;
        if (!isfinite(dmin)) dmin = 0.0f;
        s_d    = d;
        s_dmin = dmin;
        for (int i = 0; i < 12; ++i) s_scales[i] = b[4 + i];
    }
    __syncthreads();

    // Each thread handles two output elements: one from the low
    // nibble path, one from the high nibble path. Mapping mirrors
    // the Rust dequant exactly:
    //
    //   for j in (0..128).step_by(32) {
    //     for l in 0..32 {
    //       out[j+l]       = d1 * (qs[j+l] & 0xF) - m1
    //       out[j+l+128]   = d2 * (qs[j+l] >> 4) - m2
    //     }
    //   }
    //
    // We pick `tid` ∈ [0, 256) and split into the four (j, l)
    // pairs. tid 0..128 → low nibble path; tid 128..256 → high
    // nibble path. Within each half, tid % 128 == j*32 + l, with
    // is = (j / 32) * 2 (low) or is = (j / 32) * 2 + 1 (high).
    int tid = threadIdx.x;
    int is_high = tid >> 7;          // 0 or 1
    int local   = tid & 0x7F;        // 0..127
    int j       = local & 0xE0;      // 0, 32, 64, 96 (top 3 bits of local)
    int l       = local & 0x1F;      // 0..31

    int sub = (j >> 5) * 2 + is_high;
    unsigned char sc, m;
    q4k_get_sc_m(sub, s_scales, &sc, &m);

    float d_eff = s_d    * (float)sc;
    float m_eff = s_dmin * (float)m;

    int qs_idx = j + l;                // 0..128
    unsigned char qs_byte = b[16 + qs_idx];
    int q_val = is_high
        ? ((int)(qs_byte >> 4) & 0xF)
        : ((int)qs_byte & 0xF);

    int out_idx = j + l + (is_high * 128);
    out[out_idx] = d_eff * (float)q_val - m_eff;
}

// Host-side launcher. Returns hipError_t (0 = success). The Rust FFI
// declaration in `rocm.rs` mirrors this signature literally. The byte
// pointer is opaque so callers can hand in either a hipMalloc'd
// device buffer or — once we wire D2D copy — a host-pinned mirror.
hipError_t launch_dequant_q4k(
    const unsigned char* q4k,
    int n_blocks,
    float* fp32_out)
{
    if (n_blocks <= 0) return hipErrorInvalidValue;
    dim3 grid(n_blocks);
    dim3 block(Q4K_BLOCK_ELEMS);
    dequant_q4k_kernel<<<grid, block, 0, 0>>>(q4k, fp32_out, n_blocks);
    return hipGetLastError();
}

} // extern "C"
