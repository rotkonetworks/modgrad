// AdamW (decoupled weight decay) — custom hipcc kernel.
//
// Modern transformer training uses AdamW with bias-corrected first/
// second moments and decoupled weight decay (Loshchilov & Hutter 2017).
// Math (one element):
//
//     m_t   = beta1 * m_{t-1} + (1 - beta1) * g
//     v_t   = beta2 * v_{t-1} + (1 - beta2) * g * g
//     m_hat = m_t / (1 - beta1^t)
//     v_hat = v_t / (1 - beta2^t)
//     w_t   = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_{t-1})
//
// Bias-correction terms `bc1_inv = 1/(1 - beta1^t)` and `bc2_inv =
// 1/(1 - beta2^t)` are computed once per step on the host and passed
// in — saves a `pow()` per element and matches the host CPU path's
// pre-divided form exactly (`crates/modgrad-device/src/backend/cpu.rs::adamw`).
//
// One thread per element. Updates `w`, `m`, `v` in place on device.
// No reductions or shared memory — purely embarrassingly parallel.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`, archived alongside `rms_norm` and
// `dequant_q4k` into `libmodgrad_kernels.a`. The only Rust-visible
// symbol is `launch_adamw` below, declared `extern "C"`.

#include <hip/hip_runtime.h>

#define ADAMW_BLOCK 256

extern "C" {

__global__ void adamw_kernel(
    float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1_inv,
    float bc2_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi * bc1_inv;
    float v_hat = vi * bc2_inv;
    float wi = w[i];
    w[i] = wi - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * wi);
}

// Host-side launcher. Returns hipError_t (0 = success). The Rust FFI
// declaration in `rocm.rs` mirrors this signature literally.
hipError_t launch_adamw(
    float* w,
    const float* g,
    float* m,
    float* v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1_inv,
    float bc2_inv)
{
    if (n <= 0) {
        return hipErrorInvalidValue;
    }
    int blocks = (n + ADAMW_BLOCK - 1) / ADAMW_BLOCK;
    dim3 grid(blocks);
    dim3 block(ADAMW_BLOCK);
    adamw_kernel<<<grid, block, 0, 0>>>(
        w, g, m, v, n,
        lr, beta1, beta2, eps, weight_decay,
        bc1_inv, bc2_inv);
    return hipGetLastError();
}

} // extern "C"
