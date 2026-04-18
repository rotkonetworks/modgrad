//! Vjp (Vector-Jacobian-Product) pairing — forward/backward coupled at the type level.
//!
//! Adapted from JAX's `jax.custom_vjp`, reduced to what a non-tracing SDK
//! actually needs: a trait that *declares forward and backward as a pair*.
//! You can't implement one without the other. If you change the forward's
//! signature, the backward's signature has to change too, and the compiler
//! tells you. That's the invariant.
//!
//! # What this is NOT
//! This is not an autodiff engine. We still hand-write every backward; the
//! trait just ensures the two halves stay in sync. JAX's version integrates
//! with the tracer so `grad()` works automatically — we skipped the tracer
//! (see session notes on the 12-person-year cost), so we also skip
//! auto-derivation. What we gain is *invariant checking*, not free gradients.
//!
//! # Why bother without the tracer?
//! We do: every week someone adds a new op and forgets the backward, or
//! changes a dimension in the forward but not the backward. The Rust type
//! system can refuse to compile that class of mistake if we express
//! "these two belong together" as a trait.
//!
//! # Shape
//! ```ignore
//! pub trait Vjp {
//!     type Primal;   // forward input + parameters
//!     type Output;   // forward output
//!     type Cache;    // activations saved for the backward pass
//!     type Cotangent;// gradient flowing backward (same shape as Output)
//!     type Grad;     // gradient flowing backward (same shape as Primal)
//!
//!     fn forward (primal: &Self::Primal) -> (Self::Output, Self::Cache);
//!     fn backward(primal: &Self::Primal, cache: &Self::Cache,
//!                 cotangent: &Self::Cotangent) -> Self::Grad;
//! }
//! ```
//!
//! Each op implements `Vjp` once; the implementation is one file that
//! owns both directions. Review diffs stay coherent — a PR that changes
//! the forward output shape can't silently leave the backward unchanged.

/// A forward pass paired with its backward (vector-Jacobian product).
///
/// Implementors hand-write both functions. The trait exists to make the
/// pairing explicit at the type level: a new op can't be merged with
/// only the forward half.
pub trait Vjp {
    /// Complete state the forward pass consumes — typically a struct
    /// bundling weights / bias / input. Borrowed so impls don't need
    /// to take ownership.
    type Primal;

    /// Activation-plus-output data the forward pass produces. Owned so
    /// the backward pass can consume it.
    type Output;

    /// Intermediate state the backward pass needs (means, pre-activation
    /// values, etc.). Separate from `Output` because many ops have
    /// large activation caches that aren't part of their "public" output.
    type Cache;

    /// Vector (gradient) arriving at the output of the op. Same shape
    /// as `Output` by convention — but typed distinctly so mistakes
    /// in plumbing are rejected.
    type Cotangent;

    /// Gradient the op produces for its inputs + parameters. Shape
    /// matches `Primal`; production of this is the backward's job.
    type Grad;

    /// Forward pass. Pure function of `primal`; returns the output and
    /// any state the backward will need.
    fn forward(primal: &Self::Primal) -> (Self::Output, Self::Cache);

    /// Backward pass (vector-Jacobian product). Given the original
    /// inputs (`primal`) and the saved forward state (`cache`), plus
    /// the gradient that's arrived at our output (`cotangent`), produce
    /// the gradient for our inputs.
    fn backward(
        primal: &Self::Primal,
        cache: &Self::Cache,
        cotangent: &Self::Cotangent,
    ) -> Self::Grad;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Demonstrator: y = x * x, dx = 2 * x * dy ───
    //
    // Trivial op proving the trait can be implemented. The real consumers
    // (Matmul, LayerNorm, SwiGLU VJP impls) will be wired later — this
    // test only establishes that the shapes work.

    struct Square;

    struct SquarePrimal { x: Vec<f32> }
    struct SquareCache  { x: Vec<f32> }  // save x for backward

    impl Vjp for Square {
        type Primal    = SquarePrimal;
        type Output    = Vec<f32>;
        type Cache     = SquareCache;
        type Cotangent = Vec<f32>;
        type Grad      = Vec<f32>;

        fn forward(p: &Self::Primal) -> (Self::Output, Self::Cache) {
            let y: Vec<f32> = p.x.iter().map(|&v| v * v).collect();
            (y, SquareCache { x: p.x.clone() })
        }

        fn backward(
            _primal: &Self::Primal,
            cache: &Self::Cache,
            cotangent: &Self::Cotangent,
        ) -> Self::Grad {
            cache.x.iter().zip(cotangent.iter())
                .map(|(&x, &dy)| 2.0 * x * dy)
                .collect()
        }
    }

    #[test]
    fn square_forward_matches_manual() {
        let p = SquarePrimal { x: vec![1.0, 2.0, 3.0] };
        let (y, _cache) = <Square as Vjp>::forward(&p);
        assert_eq!(y, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn square_backward_matches_analytic() {
        let p = SquarePrimal { x: vec![1.0, 2.0, 3.0] };
        let (_y, cache) = <Square as Vjp>::forward(&p);
        // dL/dy = 1 everywhere → dL/dx = 2x
        let dy = vec![1.0, 1.0, 1.0];
        let dx = <Square as Vjp>::backward(&p, &cache, &dy);
        assert_eq!(dx, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn square_gradient_check_finite_difference() {
        // Numerical sanity: (f(x+ε) - f(x-ε)) / 2ε ≈ analytic dx
        let p = SquarePrimal { x: vec![2.5, -1.5] };
        let (_y, cache) = <Square as Vjp>::forward(&p);
        let dy = vec![1.0, 1.0];
        let dx_analytic = <Square as Vjp>::backward(&p, &cache, &dy);

        let eps = 1e-3;
        for (i, &expected) in dx_analytic.iter().enumerate() {
            let mut plus  = p.x.clone(); plus[i]  += eps;
            let mut minus = p.x.clone(); minus[i] -= eps;
            let (y_plus,  _) = <Square as Vjp>::forward(&SquarePrimal { x: plus });
            let (y_minus, _) = <Square as Vjp>::forward(&SquarePrimal { x: minus });
            let numeric = (y_plus[i] - y_minus[i]) / (2.0 * eps);
            assert!((numeric - expected).abs() < 1e-2,
                "param {i}: analytic {expected}, numeric {numeric}");
        }
    }
}
