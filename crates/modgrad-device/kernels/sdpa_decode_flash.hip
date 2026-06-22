// Flash-attention-style decode SDPA with warp-shuffle Q·K reduction.
// Reads ~constant shared memory regardless of seq_len.
//
// Layout for head_dim=128, RDNA3 wave32:
//   blockDim.x = 128 = 4 warps × 32 lanes
//   thread tid holds dim[tid] of Q and writes dim[tid] of O.
//
// Per-KV-step cost:
//   1 partial dot product (lane qd * kt[lane])
//   5 in-warp shuffles to reduce 32 → 1 (no sync)
//   1 shared-mem write of each warp's partial (4 writes)
//   1 __syncthreads()
//   thread 0 of each warp reads 4 partials, sums them, broadcasts
//   1 __syncthreads()
//   online softmax update in registers
//   weighted-V add in registers
//
// Total: 2 __syncthreads() per KV step. Old kernel materialized scores into
// shared mem with O(seq) writes + 2 reductions across the whole seq. This
// kernel scales much better for large seq_len.

#include <hip/hip_runtime.h>

extern "C" {

#define WAVE_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

__global__ void sdpa_decode_flash_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_dim,
    float scale, int win_start)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int tid = threadIdx.x;
    int grp = n_heads / n_kv_heads;
    int kv_h = h / grp;
    int warp = tid >> 5;
    int lane = tid & 31;
    int n_warps = blockDim.x >> 5;

    if (tid >= head_dim) return;

    const float* qh = q + (size_t)h * head_dim;
    const float* kbase = k_cache + (size_t)kv_h * head_dim;
    const float* vbase = v_cache + (size_t)kv_h * head_dim;

    float qd = qh[tid];

    // Per-warp partials shared across the block.
    __shared__ float warp_partial[16];   // up to 16 warps (we use n_warps=4)
    __shared__ float dot_full;           // broadcast score after reduction

    // Online softmax state — in registers.
    float o = 0.0f;
    float l = 0.0f;
    float m = -1e30f;

    for (int t = win_start; t < seq_len; ++t) {
        const float* kt = kbase + (size_t)t * kv_dim;
        // Per-thread partial: qd * kt[tid]
        float partial = qd * kt[tid];
        // In-warp reduction → lane 0 holds sum of this warp's 32 partials
        partial = warp_reduce_sum(partial);
        if (lane == 0) warp_partial[warp] = partial;
        __syncthreads();
        // Thread 0 of block sums the warp partials and broadcasts.
        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < n_warps; ++w) s += warp_partial[w];
            dot_full = s * scale;
        }
        __syncthreads();
        float s_t = dot_full;

        // Online softmax update (all threads in lockstep).
        float m_new = fmaxf(m, s_t);
        float exp_old = __expf(m - m_new);
        float exp_new = __expf(s_t - m_new);
        // Each thread updates its own dim of o and the scalar l.
        float vd = vbase[(size_t)t * kv_dim + tid];
        o = o * exp_old + exp_new * vd;
        l = l * exp_old + exp_new;
        m = m_new;
    }

    out[(size_t)h * head_dim + tid] = o / l;
}

hipError_t launch_sdpa_decode_flash(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_dim,
    float scale, int win_start)
{
    if (n_heads <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    if (head_dim > 512) return hipErrorInvalidValue;
    dim3 grid(n_heads);
    // Round head_dim up to a multiple of WAVE_SIZE so we don't have ragged warps.
    int block = ((head_dim + WAVE_SIZE - 1) / WAVE_SIZE) * WAVE_SIZE;
    sdpa_decode_flash_kernel<<<grid, block, 0, 0>>>(
        q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, seq_len, kv_dim, scale, win_start);
    return hipGetLastError();
}

} // extern "C"
