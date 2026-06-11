// Decode-step scaled-dot-product attention (one query token attending to the
// resident KV cache). GQA, causal-by-construction, optional sliding window.
// One block per query head.
//
//   q        : [n_heads * head_dim]            (current token, post-rope)
//   k_cache  : [seq_len * kv_dim]              (kv_dim = n_kv_heads*head_dim)
//   v_cache  : [seq_len * kv_dim]
//   out      : [n_heads * head_dim]
//   scale    : 1/sqrt(head_dim)
//   win_start: first cache position to attend (sliding window) — 0 = global
//
// Dynamic shared memory: seq_len floats (scores).

#include <hip/hip_runtime.h>

extern "C" {

__global__ void sdpa_decode_kernel(
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

    const float* qh = q + (size_t)h * head_dim;
    const float* kbase = k_cache + (size_t)kv_h * head_dim;
    const float* vbase = v_cache + (size_t)kv_h * head_dim;

    extern __shared__ float sc[];        // [seq_len]
    __shared__ float red[256];
    __shared__ float smax, sinv;

    // scores
    for (int t = win_start + tid; t < seq_len; t += blockDim.x) {
        const float* kt = kbase + (size_t)t * kv_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kt[d];
        sc[t] = dot * scale;
    }
    __syncthreads();

    // max
    float m = -1e30f;
    for (int t = win_start + tid; t < seq_len; t += blockDim.x) m = fmaxf(m, sc[t]);
    red[tid] = m; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    if (tid == 0) smax = red[0];
    __syncthreads();

    // exp + sum
    float sum = 0.0f;
    for (int t = win_start + tid; t < seq_len; t += blockDim.x) {
        float e = __expf(sc[t] - smax);
        sc[t] = e;
        sum += e;
    }
    red[tid] = sum; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    if (tid == 0) sinv = 1.0f / red[0];
    __syncthreads();

    // weighted sum of V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = win_start; t < seq_len; ++t) acc += sc[t] * vbase[(size_t)t * kv_dim + d];
        out[(size_t)h * head_dim + d] = acc * sinv;
    }
}

hipError_t launch_sdpa_decode(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int kv_dim,
    float scale, int win_start)
{
    if (n_heads <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    dim3 grid(n_heads);
    dim3 block(256);
    size_t shmem = (size_t)seq_len * sizeof(float);
    sdpa_decode_kernel<<<grid, block, shmem, 0>>>(
        q, k_cache, v_cache, out, n_heads, n_kv_heads, head_dim, seq_len, kv_dim, scale, win_start);
    return hipGetLastError();
}

} // extern "C"
