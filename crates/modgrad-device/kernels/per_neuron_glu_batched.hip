// Per-neuron batched GLU forward — custom hipcc kernel.
//
// Replaces an N-call MIOpen `miopenGLUForward` loop in the CTM
// forward path. The cortex's NLM stage iterates `d_model` neurons,
// each with a `[2*half_size]` slab whose first `half_size` floats
// are values and next `half_size` floats are gates:
//
//     y[n, i] = x[n, i] * sigmoid(x[n, half_size + i])
//
// for n in 0..n_neurons, i in 0..half_size.
//
// The previous implementation issued one MIOpen dispatch per
// neuron (`forward.rs:842-851` and `forward.rs:919-928`). For
// d_model = 64 and a depth-2 NLM that is 128 dispatches per CTM
// tick, each carrying ~10 µs of MIOpen launch overhead. This
// kernel collapses the loop into a single grid launch over
// `n_neurons * half_size` threads.
//
// Layout note: the per-neuron interleaved layout (value half then
// gate half WITHIN each neuron) is NOT the same as MIOpen's `dim=0`
// layout (all values then all gates across the batch). MIOpen
// can't batch this in one call — its solver only splits along the
// leading axis (`src/solver/glu/forward_glu.cpp`). The custom
// kernel here is required, not just a perf win.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`. The Rust-visible symbol is
// `launch_per_neuron_glu_batched`, declared `extern "C"`.

#include <hip/hip_runtime.h>

#define PNGB_BLOCK 256

extern "C" {

__global__ void per_neuron_glu_batched_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_neurons,
    int half_size)
{
    // Element-wise: each thread handles one output (n, i) pair.
    // Total output elements = n_neurons * half_size.
    int total = n_neurons * half_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = idx / half_size;
    int i = idx - n * half_size;

    // Each neuron's input slab is `2 * half_size` floats, value
    // half then gate half. The gate sits at `[n*2*half_size + half_size + i]`.
    int base = n * 2 * half_size;
    float v = x[base + i];
    float g = x[base + half_size + i];

    // sigmoid(g) = 1 / (1 + exp(-g)). MIOpen's GLU uses the same
    // formulation — testing confirmed parity within 1e-3 (f32 arith).
    float gate_sig = 1.0f / (1.0f + expf(-g));
    y[idx] = v * gate_sig;
}

// Host-side launcher. Returns hipError_t (0 = success). The Rust FFI
// declaration in `rocm.rs` mirrors this signature literally.
hipError_t launch_per_neuron_glu_batched(
    const float* x,
    float* y,
    int n_neurons,
    int half_size)
{
    if (n_neurons <= 0 || half_size <= 0) {
        return hipErrorInvalidValue;
    }
    int total = n_neurons * half_size;
    int blocks = (total + PNGB_BLOCK - 1) / PNGB_BLOCK;
    dim3 grid(blocks);
    dim3 block(PNGB_BLOCK);
    per_neuron_glu_batched_kernel<<<grid, block, 0, 0>>>(
        x, y, n_neurons, half_size);
    return hipGetLastError();
}

} // extern "C"
