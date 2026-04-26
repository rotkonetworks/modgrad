// RMSNorm backward — custom hipcc kernel.
//
// Pairs with `rms_norm.hip` to keep the final-RMSNorm gradient on
// device. Eliminates the host-bounce that `GptModelResident::backward`
// previously did (D2H input + H2D grad on every backward step) and
// matches the per-feature learnable `weight[c]` (gamma) used in the
// forward.
//
// Forward (recap):
//     y[r, c] = x[r, c] * rstd[r] * weight[c]
//     rstd[r] = 1 / sqrt(mean(x[r, :]^2) + eps)
//
// Backward — given dy, computes dx and (optionally) dweight:
//     S[r]       = sum_c (dy[r, c] * x[r, c] * weight[c])
//     dx[r, c]   = (weight[c] * dy[r, c]
//                   - x[r, c] * S[r] * rstd[r]^2 / hidden) * rstd[r]
//     dweight[c] = sum_r (dy[r, c] * x[r, c] * rstd[r])
//
// Two kernels because the reductions go in opposite directions:
//   - dx kernel: per-row reduction over `c`. One block per row,
//     `BLOCK = 256` threads cooperate on `S[r]` via LDS, then
//     each thread emits its slice of `dx`.
//   - dweight kernel: per-column reduction over `r`. One block per
//     column, threads sum `dy[r, c] * x[r, c] * rstd[r]` across rows.
//     Accumulates into `dweight[c]` (caller is responsible for zeroing
//     beforehand if a fresh gradient is desired).
//
// `rstd` is recomputed inside both kernels — it's cheap (one pass over
// `hidden`) and avoids carrying a per-row scratch buffer between the
// forward and backward residencies. The forward kernel doesn't write
// rstd today; if we ever switch to a fused fwd+bwd path, lifting
// `rstd_shared` into a global write is mechanical.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`. Linked into the final binary as a static
// .o; only the host launchers below are Rust-visible (`extern "C"`).

#include <hip/hip_runtime.h>

#define RMS_NORM_BWD_BLOCK 256

extern "C" {

// Per-row reduction: compute `dx` for each (r, c) and the row-local
// sum `S[r] = sum_c dy[r, c] * x[r, c] * weight[c]`. One block per
// row, threads stride over `c` and cooperate on the reduction.
__global__ void rms_norm_backward_dx_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ weight,
    float* __restrict__ dx,
    int n,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const float* x_row = x + row * hidden;
    const float* dy_row = dy + row * hidden;
    float* dx_row = dx + row * hidden;

    // Pass 1: per-thread partial sum-of-squares for rstd.
    float partial_sq = 0.0f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = x_row[c];
        partial_sq += v * v;
    }

    __shared__ float sdata[RMS_NORM_BWD_BLOCK];
    sdata[threadIdx.x] = partial_sq;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    __shared__ float rstd_shared;
    if (threadIdx.x == 0) {
        float mean_sq = sdata[0] / (float)hidden;
        rstd_shared = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rstd = rstd_shared;

    // Pass 2: per-thread partial sum for S = Σ dy[c] * x[c] * weight[c].
    float partial_s = 0.0f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        partial_s += dy_row[c] * x_row[c] * weight[c];
    }

    sdata[threadIdx.x] = partial_s;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    __shared__ float s_shared;
    if (threadIdx.x == 0) {
        s_shared = sdata[0];
    }
    __syncthreads();
    float s = s_shared;

    // Pass 3: emit dx.
    //   dx[c] = (weight[c] * dy[c] - x[c] * s * rstd^2 / hidden) * rstd
    float inv_n_rstd2 = (rstd * rstd) / (float)hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float v = weight[c] * dy_row[c] - x_row[c] * s * inv_n_rstd2;
        dx_row[c] = v * rstd;
    }
}

// Per-column reduction: dweight[c] = Σ_r dy[r, c] * x[r, c] * rstd[r].
// One block per column, threads stride over rows and cooperate on the
// row-axis reduction. Accumulates into `dweight` (caller zeros if a
// fresh gradient is desired — same convention as MIOpen LayerNorm
// backward).
//
// `n` rarely exceeds a few thousand for the final-RMSNorm in a decoder
// (one row per token in the active batch). For the n=1 single-token
// path this kernel still launches `hidden` blocks of `BLOCK` threads;
// the per-block reduction is O(1) work but the block-launch overhead
// is fixed-cost. Cheap relative to the global memory traffic.
__global__ void rms_norm_backward_dweight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dweight,
    int n,
    int hidden,
    float eps)
{
    int col = blockIdx.x;
    if (col >= hidden) return;

    // Per-thread partial: sum dy[r, col] * x[r, col] * rstd[r] over r.
    float partial = 0.0f;
    for (int r = threadIdx.x; r < n; r += blockDim.x) {
        const float* x_row = x + r * hidden;
        // Recompute rstd[r] — one pass over the row's `hidden` floats.
        // Done by every thread that owns row r in this column-block;
        // the cost is amortised across the per-column dweight emission
        // because the row sweep is required anyway. Cheaper than
        // staging an n-sized rstd scratch buffer between fwd and bwd.
        float sumsq = 0.0f;
        for (int c = 0; c < hidden; ++c) {
            float v = x_row[c];
            sumsq += v * v;
        }
        float rstd = rsqrtf(sumsq / (float)hidden + eps);
        partial += dy[r * hidden + col] * x_row[col] * rstd;
    }

    __shared__ float sdata[RMS_NORM_BWD_BLOCK];
    sdata[threadIdx.x] = partial;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Accumulate into caller's dweight — match MIOpen LayerNorm
        // backward convention. Caller pre-zeros if a fresh gradient is
        // desired.
        dweight[col] += sdata[0];
    }
}

// Host launcher. Mirrors the Rust FFI declaration in `rocm.rs`.
// `dweight` may be NULL when the caller doesn't need the gamma
// gradient (e.g. inference paths or frozen-norm experiments) — we
// elide the second kernel launch in that case to save the
// one-block-per-column dispatch cost on tall-thin shapes.
hipError_t launch_rms_norm_backward(
    const float* x,
    const float* dy,
    const float* weight,
    float* dx,
    float* dweight,
    int n,
    int hidden,
    float eps)
{
    if (n <= 0 || hidden <= 0) {
        return hipErrorInvalidValue;
    }

    // dx kernel: one block per row.
    {
        dim3 grid(n);
        dim3 block(RMS_NORM_BWD_BLOCK);
        rms_norm_backward_dx_kernel<<<grid, block, 0, 0>>>(
            x, dy, weight, dx, n, hidden, eps);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) return err;
    }

    // dweight kernel: one block per column. Skip when caller passed
    // NULL — frozen-gamma path is non-trivially common (final norm
    // initialised to 1.0 and not learned in some configs).
    if (dweight != nullptr) {
        dim3 grid(hidden);
        dim3 block(RMS_NORM_BWD_BLOCK);
        rms_norm_backward_dweight_kernel<<<grid, block, 0, 0>>>(
            x, dy, dweight, n, hidden, eps);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) return err;
    }

    return hipSuccess;
}

} // extern "C"
