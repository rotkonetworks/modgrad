// Elementwise / RoPE / GeGLU glue kernels for the resident Gemma forward.
// All operate on device buffers; the engine chains them stream-ordered
// (default stream) with no intermediate sync — the activation never
// leaves VRAM.

#include <hip/hip_runtime.h>

extern "C" {

// h[i] += x[i]   (residual add, length n)
__global__ void k_residual_add(float* __restrict__ h, const float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += x[i];
}
hipError_t launch_residual_add(float* h, const float* x, int n) {
    int b = 256, g = (n + b - 1) / b;
    k_residual_add<<<g, b, 0, 0>>>(h, x, n);
    return hipGetLastError();
}

// h[i] *= s   (scalar multiply in place)
__global__ void k_scale(float* __restrict__ h, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] *= s;
}
hipError_t launch_scale(float* h, float s, int n) {
    int b = 256, g = (n + b - 1) / b;
    k_scale<<<g, b, 0, 0>>>(h, s, n);
    return hipGetLastError();
}

// GeGLU: out[i] = gelu_tanh(gate[i]) * up[i]   (length n)
__global__ void k_geglu(const float* __restrict__ gate, const float* __restrict__ up,
                        float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gate[i];
    float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    float g = 0.5f * v * (1.0f + tanhf(c));
    out[i] = g * up[i];
}
hipError_t launch_geglu(const float* gate, const float* up, float* out, int n) {
    int b = 256, g = (n + b - 1) / b;
    k_geglu<<<g, b, 0, 0>>>(gate, up, out, n);
    return hipGetLastError();
}

// Final logit soft-capping: y[i] = cap * tanh(y[i] / cap)
__global__ void k_softcap(float* __restrict__ y, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && cap > 0.0f) y[i] = cap * tanhf(y[i] / cap);
}
hipError_t launch_softcap(float* y, float cap, int n) {
    int b = 256, g = (n + b - 1) / b;
    k_softcap<<<g, b, 0, 0>>>(y, cap, n);
    return hipGetLastError();
}

// NEOX RoPE in place on q/k. data is [n_heads * head_dim] for ONE token at
// absolute position `pos`. Pairs (i, i+half) per head.
//   freq = base^(-2 i / head_dim) / freq_factors[i],  angle = pos * freq
// freq_factors (nullable): Gemma4 "proportional" RoPE on full-attention layers —
// the GGUF rope_freqs has 1.0 for the rotated dims and ~1e30 for the rest, so
// freq/ff ≈ 0 there → no rotation (partial_rotary_factor 0.25). null => full RoPE.
__global__ void k_rope_neox(float* __restrict__ data, int pos, int n_heads,
                            int head_dim, float base, const float* __restrict__ freq_factors) {
    int half = head_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // over n_heads*half
    if (idx >= n_heads * half) return;
    int h = idx / half;
    int i = idx % half;
    float freq = powf(base, -2.0f * (float)i / (float)head_dim);
    if (freq_factors) freq /= freq_factors[i];
    float ang = (float)pos * freq;
    float c = cosf(ang), s = sinf(ang);
    int off = h * head_dim;
    float x0 = data[off + i];
    float x1 = data[off + i + half];
    data[off + i]        = x0 * c - x1 * s;
    data[off + i + half] = x0 * s + x1 * c;
}
hipError_t launch_rope_neox(float* data, int pos, int n_heads, int head_dim, float base,
                            const float* freq_factors) {
    int total = n_heads * (head_dim >> 1);
    int b = 256, g = (total + b - 1) / b;
    k_rope_neox<<<g, b, 0, 0>>>(data, pos, n_heads, head_dim, base, freq_factors);
    return hipGetLastError();
}

// Copy src -> dst (device-to-device, length n) — used to append K/V into the
// resident cache at the right offset, and to snapshot the residual.
__global__ void k_copy(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
hipError_t launch_copy(float* dst, const float* src, int n) {
    int b = 256, g = (n + b - 1) / b;
    k_copy<<<g, b, 0, 0>>>(dst, src, n);
    return hipGetLastError();
}

} // extern "C"
