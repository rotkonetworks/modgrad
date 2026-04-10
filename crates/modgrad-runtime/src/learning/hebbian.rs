//! Hebbian plasticity: Oja's rule, NMDA-gated, calcium-dependent.
//!
//! STATUS: Never worked. Destroys features every time.
//! Kept as research reference — do not enable in production.
//!
//! The rule: ΔW = lr × (post × pre - post² × W)
//! Converges to first principal component of input distribution.
//! With multiple neurons + lateral inhibition → PCA.
//!
//! Why it fails: Hebbian correlates co-activation, which is an
//! observation of trained networks, not a cause of training.
//! See discussion in project memory.

// Intentionally empty — the LocalHebbian struct lives in session.rs.
// This file exists to document that Hebbian was tried and doesn't work.
// If you want to experiment: enable session.hebbian_enabled = true
// and the inline Hebbian block in forward.rs will activate.
