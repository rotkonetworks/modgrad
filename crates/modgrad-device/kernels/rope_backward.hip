// RoPE backward — custom hipcc kernel.
//
// Rotary Position Embeddings (RoPE) split a head vector into pairs
// (x_i, x_{i+half}) and rotate each pair by an angle θ_i that depends on
// the position and the pair index. The forward rotation is:
//
//     x_post[i]        = x_pre[i] * cos[i] - x_pre[i+half] * sin[i]
//     x_post[i+half]   = x_pre[i] * sin[i] + x_pre[i+half] * cos[i]
//
// The rotation matrix is orthogonal, so the adjoint (gradient backward)
// is the same rotation by the negated angle — equivalent to swapping the
// sign of `sin`:
//
//     dx_pre[i]        = dx_post[i] *  cos[i] + dx_post[i+half] * sin[i]
//     dx_pre[i+half]   = dx_post[i] * (-sin[i]) + dx_post[i+half] * cos[i]
//
// The host reference is `rope_backward()` in
// `crates/modgrad-transformer/src/resident.rs`. This kernel reproduces
// that math element-for-element on device so the resident attention
// backward path can avoid the D2H/host-loop/H2D bounce.
//
// Layout: `dx_post` and `dx_pre` are each `[num_heads, head_dim]`
// row-major where `head_dim = 2 * half_dim`. `cos` and `sin` are each
// `[half_dim]` — one per pair, indexed at the position the matched
// forward used. The caller is responsible for slicing the right row out
// of the precomputed `RotaryEmbedding` cos/sin tables.
//
// One block per head, threads cooperate over pairs. BLOCK_SIZE is fixed
// at 256 so the kernel covers any practical `half_dim` (head_dim up to
// 512) in a single warp pass; for smaller `half_dim` the tail threads
// are idle, cheaper than the per-launch overhead of a smaller block.
//
// In-place is supported: `dx_post` and `dx_pre` may alias because the
// kernel reads BOTH lanes of each pair before writing either, so no
// thread observes the other lane's post-write state.
//
// Compiled by `crates/modgrad-device/build.rs` against
// `--offload-arch=gfx1102`. Linked into the final binary as a static
// .o; the only Rust-visible symbol is `launch_rope_backward` below,
// declared `extern "C"`.

#include <hip/hip_runtime.h>

#define ROPE_BWD_BLOCK 256

extern "C" {

__global__ void rope_backward_kernel(
    const float* __restrict__ dx_post,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    float* __restrict__ dx_pre,
    int num_heads,
    int head_dim,
    int half_dim)
{
    int head = blockIdx.x;
    if (head >= num_heads) return;

    const float* head_in = dx_post + head * head_dim;
    float* head_out      = dx_pre  + head * head_dim;

    // Each thread handles one pair (i, i+half_dim). Iterate so a single
    // block can cover any half_dim up to BLOCK_SIZE * inf.
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        // Read BOTH halves before writing — supports in-place
        // dx_pre == dx_post.
        float dl = head_in[i];
        float dr = head_in[i + half_dim];
        float c  = cos_tab[i];
        float s  = sin_tab[i];

        // Adjoint of the forward rotation:
        //   forward: (l, r) -> (l*c - r*s, l*s + r*c)
        //   adjoint: (dl_post, dr_post) -> (dl_post*c + dr_post*s,
        //                                  -dl_post*s + dr_post*c)
        head_out[i]            = dl *  c + dr * s;
        head_out[i + half_dim] = -dl * s + dr * c;
    }
}

// Host-side launcher. Returns hipError_t (0 = success). The Rust FFI
// declaration in `rocm.rs` mirrors this signature literally.
hipError_t launch_rope_backward(
    const float* dx_post,
    const float* cos_tab,
    const float* sin_tab,
    float* dx_pre,
    int num_heads,
    int head_dim)
{
    if (num_heads <= 0 || head_dim <= 0 || (head_dim & 1) != 0) {
        return hipErrorInvalidValue;
    }
    int half_dim = head_dim / 2;
    dim3 grid(num_heads);
    dim3 block(ROPE_BWD_BLOCK);
    rope_backward_kernel<<<grid, block, 0, 0>>>(
        dx_post, cos_tab, sin_tab, dx_pre, num_heads, head_dim, half_dim);
    return hipGetLastError();
}

} // extern "C"
