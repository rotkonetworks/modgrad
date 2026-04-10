//! Research library: learning techniques explored during development.
//!
//! Each module is a self-contained technique that was wired into the brain
//! at some point, tested, and documented. None are active in the default
//! forward pass — they're here as a library to pull from.
//!
//! To use a technique: import it and call it from your training loop.
//! Each function takes (&CtmWeights, &mut CtmSession, &CtmTickState)
//! and returns weight deltas or modifies session state.
//!
//! ## Catalogue
//!
//! | Technique | Best result | Status |
//! |-----------|-------------|--------|
//! | three_factor | 56% on 7×7 maze | Works. Motor-pathway only. |
//! | salience | +26pp with RPE×conflict gating | Works. Prevents catastrophic forgetting. |
//! | cerebellar | Prediction error delta rule | Works. Climbing fiber signal. |
//! | qec_learn | MWPM error localization | Mechanically correct, didn't improve learning. |
//! | hebbian | Oja's rule, NMDA-gated | Destroyed features every time. Historical only. |
//! | cri | Conditioned Reflex Injection | 40% from memory alone, no weight changes. |
//! | spsa | Rank-1 perturbation | 0% improvement on QEC. Too noisy for 100K+ params. |
//! | cholesky_init | LS warm start for synapses | Made motor accuracy worse (8% from 28%). |
//! | titans | Adaptive η, θ, α dynamics | +26pp, stable plateau. Best online method. |

pub mod qec_learn;
