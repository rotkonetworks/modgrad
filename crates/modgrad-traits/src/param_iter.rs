//! Parameter iteration trait — JAX PyTree-style traversal for model weights.
//!
//! An `impl ParamIter` on a weight struct is a *single declaration* of that
//! model's parameter layout, which other machinery (optimizer, VRAM mirror,
//! serialization, norm clipping, parameter counting) then walks instead of
//! each hand-maintaining its own copy.
//!
//! # Why
//! Before this trait, `modgrad-ffn` exposed parameter layout through an
//! index arithmetic (`idx_block(layer, which)`) that both `VramMirror`
//! upload, `FfnAdamW::step_update`, and checkpoint save had to stay in
//! sync with independently. Adding a field to `FfnBlock` meant touching
//! three files; forgetting one produced a silent stale-index bug.
//!
//! After: every consumer calls `walk_params` and gets the tensors in a
//! consistent, name-keyed order. Adding a field requires updating the
//! `impl ParamIter` site and *nothing else*.
//!
//! # Names
//! Parameter names are dot-paths (`"blocks.3.gate.weight"`) so callers
//! can use them as keys for checkpointing or debugging. Names should
//! be stable across runs — downstream checkpoint formats treat them as
//! identifiers.
//!
//! # Relationship to JAX
//! JAX's `jax.tree_util` is generic over any nested structure because
//! Python is duck-typed. In Rust we express the same idea as a trait
//! whose visitor gives you the leaves. We drop the "any tree" generality
//! — neural-network parameters are always slices of scalars, which is
//! what every caller needs.

/// A type whose parameters can be enumerated as `&[T]` / `&mut [T]` slices
/// with stable dot-path names.
///
/// The default `n_params` and `clone_all` methods derive from the walks,
/// so an implementer only has to provide `walk_params` + `walk_params_mut`.
pub trait ParamIter<T = f32> {
    /// Visit every parameter tensor in canonical order. Each call to `f`
    /// receives a stable dot-path name (`"blocks.{i}.gate.weight"`) and
    /// the raw slice.
    fn walk_params(&self, f: &mut dyn FnMut(&str, &[T]));

    /// Mutable variant. Order must match `walk_params` exactly — callers
    /// that zip the two walks rely on this invariant.
    fn walk_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut [T]));

    /// Total element count across every parameter. Default impl walks
    /// once and sums; overriders can cache if they like.
    fn n_params(&self) -> usize {
        let mut total = 0usize;
        self.walk_params(&mut |_, data| total += data.len());
        total
    }

    /// Enumerate parameter names + lengths without exposing the data.
    /// Useful for pre-sizing an optimizer state / VRAM mirror.
    fn param_names_and_sizes(&self) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        self.walk_params(&mut |name, data| out.push((name.to_string(), data.len())));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny model to exercise the trait shape.
    struct Dummy {
        a: Vec<f32>,
        blocks: Vec<DummyBlock>,
        c: Vec<f32>,
    }
    struct DummyBlock { w: Vec<f32>, b: Vec<f32> }

    impl ParamIter for Dummy {
        fn walk_params(&self, f: &mut dyn FnMut(&str, &[f32])) {
            f("a", &self.a);
            for (i, blk) in self.blocks.iter().enumerate() {
                f(&format!("blocks.{i}.w"), &blk.w);
                f(&format!("blocks.{i}.b"), &blk.b);
            }
            f("c", &self.c);
        }
        fn walk_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut [f32])) {
            f("a", &mut self.a);
            for (i, blk) in self.blocks.iter_mut().enumerate() {
                f(&format!("blocks.{i}.w"), &mut blk.w);
                f(&format!("blocks.{i}.b"), &mut blk.b);
            }
            f("c", &mut self.c);
        }
    }

    #[test]
    fn walk_visits_every_parameter_in_order() {
        let m = Dummy {
            a: vec![1.0],
            blocks: vec![
                DummyBlock { w: vec![2.0; 3], b: vec![3.0; 2] },
                DummyBlock { w: vec![4.0; 3], b: vec![5.0; 2] },
            ],
            c: vec![6.0; 4],
        };
        let mut names = Vec::new();
        m.walk_params(&mut |n, _| names.push(n.to_string()));
        assert_eq!(names, vec![
            "a",
            "blocks.0.w", "blocks.0.b",
            "blocks.1.w", "blocks.1.b",
            "c",
        ]);
    }

    #[test]
    fn default_n_params_sums_walks() {
        let m = Dummy {
            a: vec![0.0; 1],
            blocks: vec![DummyBlock { w: vec![0.0; 3], b: vec![0.0; 2] }],
            c: vec![0.0; 4],
        };
        assert_eq!(m.n_params(), 1 + 3 + 2 + 4);
    }

    #[test]
    fn walk_mut_allows_in_place_update() {
        let mut m = Dummy {
            a: vec![0.0],
            blocks: vec![DummyBlock { w: vec![0.0], b: vec![0.0] }],
            c: vec![0.0],
        };
        m.walk_params_mut(&mut |_, data| {
            for x in data.iter_mut() { *x = 7.0; }
        });
        assert_eq!(m.a, vec![7.0]);
        assert_eq!(m.blocks[0].w, vec![7.0]);
        assert_eq!(m.blocks[0].b, vec![7.0]);
        assert_eq!(m.c, vec![7.0]);
    }

    #[test]
    fn param_names_and_sizes_reports_layout() {
        let m = Dummy {
            a: vec![0.0; 1],
            blocks: vec![DummyBlock { w: vec![0.0; 3], b: vec![0.0; 2] }],
            c: vec![0.0; 4],
        };
        let ns = m.param_names_and_sizes();
        assert_eq!(ns.len(), 4);
        assert_eq!(ns[0], ("a".to_string(), 1));
        assert_eq!(ns[3], ("c".to_string(), 4));
    }
}
