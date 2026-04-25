// RMSNorm forward — custom hipcc kernel.
//
// Modern LLMs (Gemma, Llama, Qwen, ...) normalise activations by their
// root-mean-square instead of the LayerNorm mean+variance combination.
// MIOpen ships LayerNorm but no RMSNorm primitive, so this file provides
// the missing kernel:
//
//     y[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * weight[c]
//
// One block per row; threads cooperate on a row-wide sum-of-squares
// reduction via shared memory. BLOCK_SIZE is fixed at 256 — covers
// every realistic hidden size (>= 256) in a single warp pass without
// burning enormous LDS for narrow models. Hidden sizes that are not a
// multiple of BLOCK_SIZE simply have idle threads in the tail; this is
// cheap relative to the global memory traffic.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`. Linked into the final binary as a static
// .o; the only Rust-visible symbol is `launch_rms_norm` below, declared
// `extern "C"`.

#include <hip/hip_runtime.h>

#define RMS_NORM_BLOCK 256

extern "C" {

__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int n,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const float* x_row = x + row * hidden;
    float* y_row = y + row * hidden;

    // Per-thread partial sum of squares.
    float partial = 0.0f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = x_row[c];
        partial += v * v;
    }

    // Block-wide reduction via shared memory.
    __shared__ float sdata[RMS_NORM_BLOCK];
    sdata[threadIdx.x] = partial;
    __syncthreads();

    // Standard tree reduction. blockDim.x is BLOCK_SIZE here.
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Lane 0 computes rstd = 1 / sqrt(mean(x^2) + eps); broadcast via LDS.
    __shared__ float rstd_shared;
    if (threadIdx.x == 0) {
        float mean_sq = sdata[0] / (float)hidden;
        rstd_shared = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rstd = rstd_shared;

    // Apply: y = x * rstd * weight.
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        y_row[c] = x_row[c] * rstd * weight[c];
    }
}

// Host-side launcher. Returns hipError_t (0 = success). The Rust FFI
// declaration in `rocm.rs` mirrors this signature literally.
hipError_t launch_rms_norm(
    const float* x,
    const float* weight,
    float* y,
    int n,
    int hidden,
    float eps)
{
    if (n <= 0 || hidden <= 0) {
        return hipErrorInvalidValue;
    }
    dim3 grid(n);
    dim3 block(RMS_NORM_BLOCK);
    rms_norm_kernel<<<grid, block, 0, 0>>>(x, weight, y, n, hidden, eps);
    return hipGetLastError();
}

} // extern "C"
