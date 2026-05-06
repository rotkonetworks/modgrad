//! Weights and state for the faithful Ctm CTM.

use serde::{Deserialize, Serialize};
use modgrad_compute::neuron::{Linear, SuperLinear, SimpleRng};
use super::config::CtmConfig;
use super::synapse::SynapseUNet;
use wincode_derive::{SchemaRead, SchemaWrite};

/// All trainable weights for the Ctm CTM.
#[derive(Debug, Clone, Serialize, Deserialize, SchemaRead, SchemaWrite)]
pub struct CtmWeights {
    pub config: CtmConfig,

    // ── Synapse: concat(attn_out, activated_state) → pre-activations ──
    pub synapse: SynapseUNet,

    // ── NLM (trace_processor) ──
    // Deep:    SuperLinear(M → 2H) → GLU → SuperLinear(H → 2) → GLU → [D]
    // Shallow: SuperLinear(M → 2)  → GLU → [D]
    pub nlm_stage1: SuperLinear,
    pub nlm_stage2: Option<SuperLinear>,

    // ── Learnable start states ──
    pub start_activated: Vec<f32>,   // [d_model]
    pub start_trace: Vec<f32>,       // [d_model × memory_length]

    // ── Input feature projection ──
    pub kv_proj: Linear,             // raw_input_dim → d_input
    pub kv_ln_gamma: Vec<f32>,       // [d_input]
    pub kv_ln_beta: Vec<f32>,        // [d_input]

    // ── Attention ──
    pub q_proj: Linear,              // synch_action_size → d_input
    // Multihead attention: packed in_proj and out_proj
    pub mha_in_proj: Linear,         // d_input → 3×d_input (Q,K,V packed)
    pub mha_out_proj: Linear,        // d_input → d_input

    // ── Sync topology (random-pairing indices) ──
    pub sync_out_left: Vec<usize>,
    pub sync_out_right: Vec<usize>,
    pub sync_action_left: Vec<usize>,
    pub sync_action_right: Vec<usize>,

    // ── Learnable decay parameters ──
    pub decay_params_out: Vec<f32>,    // [n_synch_out]
    pub decay_params_action: Vec<f32>, // [n_synch_action]

    // ── Output projector ──
    pub output_proj: Linear,           // synch_out_size → out_dims

    // ── Adaptive exit gate (LoopLM-style) ──
    // Present iff exit_strategy is AdaptiveGate.
    // sync_out → 1 → sigmoid → per-tick halting probability.
    #[serde(default)]
    pub exit_gate: Option<Linear>,     // synch_out_size → 1
}

impl CtmWeights {
    pub fn new(config: CtmConfig, raw_input_dim: usize) -> Self {
        let d = config.d_model;
        let m = config.memory_length;
        let h = config.memory_hidden_dims;

        // NLM stages
        let (nlm1_out, nlm_stage2) = if config.deep_nlms {
            (2 * h, Some(SuperLinear::new(d, h, 2)))
        } else {
            (2, None)
        };
        let nlm_stage1 = SuperLinear::new(d, m, nlm1_out);

        // Start states: uniform init matching PyTorch default
        let mut rng = SimpleRng::new(42);
        let bound1 = 1.0 / (d as f32).sqrt();
        let start_activated: Vec<f32> = (0..d)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound1).collect();
        let bound2 = 1.0 / (m as f32).sqrt();
        let start_trace: Vec<f32> = (0..d * m)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * bound2).collect();

        // Sync indices: random-pairing
        let sync_out_left = random_indices(d, config.n_synch_out, &mut rng);
        let mut sync_out_right = random_indices(d, config.n_synch_out, &mut rng);
        // Self-pairing for first n_random_pairing_self entries
        for i in 0..config.n_random_pairing_self.min(config.n_synch_out) {
            sync_out_right[i] = sync_out_left[i];
        }

        let sync_action_left = random_indices(d, config.n_synch_action, &mut rng);
        let mut sync_action_right = random_indices(d, config.n_synch_action, &mut rng);
        for i in 0..config.n_random_pairing_self.min(config.n_synch_action) {
            sync_action_right[i] = sync_action_left[i];
        }

        // Decay params: init to 1.0 (will be clamped to [0,15], exp(-1) ≈ 0.37 decay)
        let decay_params_out = vec![1.0; config.n_synch_out];
        let decay_params_action = vec![1.0; config.n_synch_action];

        Self {
            synapse: SynapseUNet::new(
                config.synapse_in_dim(), config.d_model,
                config.synapse_depth, config.min_width,
            ),
            nlm_stage1,
            nlm_stage2,
            start_activated,
            start_trace,
            kv_proj: Linear::new(raw_input_dim, config.d_input),
            kv_ln_gamma: vec![1.0; config.d_input],
            kv_ln_beta: vec![0.0; config.d_input],
            q_proj: Linear::new(config.synch_size_action(), config.d_input),
            mha_in_proj: Linear::new(config.d_input, 3 * config.d_input),
            mha_out_proj: Linear::new(config.d_input, config.d_input),
            sync_out_left,
            sync_out_right,
            sync_action_left,
            sync_action_right,
            decay_params_out,
            decay_params_action,
            output_proj: Linear::new(config.synch_size_out(), config.out_dims),
            exit_gate: if config.exit_strategy.has_gate() {
                Some(Linear::new(config.synch_size_out(), 1))
            } else {
                None
            },
            config,
        }
    }

    /// Save weights. Format by extension: `.bin` → bincode, `.json` → JSON.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        modgrad_persist::persist::save(self, path).map_err(|e| e.into())
    }

    /// Load weights. Format by extension, with JSON fallback for legacy files.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        modgrad_persist::persist::load(path).map_err(|e| e.into())
    }

    /// Total trainable parameters.
    pub fn n_params(&self) -> usize {
        let mut n = 0usize;
        n += self.synapse.n_params();
        n += self.nlm_stage1.weights.len() + self.nlm_stage1.biases.len();
        if let Some(ref s2) = self.nlm_stage2 {
            n += s2.weights.len() + s2.biases.len();
        }
        n += self.start_activated.len() + self.start_trace.len();
        n += self.kv_proj.weight.len() + self.kv_proj.bias.len();
        n += self.kv_ln_gamma.len() * 2;
        n += self.q_proj.weight.len() + self.q_proj.bias.len();
        n += self.mha_in_proj.weight.len() + self.mha_in_proj.bias.len();
        n += self.mha_out_proj.weight.len() + self.mha_out_proj.bias.len();
        n += self.decay_params_out.len() + self.decay_params_action.len();
        n += self.output_proj.weight.len() + self.output_proj.bias.len();
        if let Some(ref g) = self.exit_gate { n += g.weight.len() + g.bias.len(); }
        n
    }
}

// ─── CtmWeightsTyped<D> — Path C JAX-style port ──────────────
//
// Device-generic weight container mirroring `CtmWeights` field-by-field.
// Each `Vec<f32>` becomes a `Tensor<D>`; each `Linear` becomes
// `tensor::Linear<D>`; each `SuperLinear` becomes `tensor::SuperLinear<D>`;
// the `SynapseUNet` becomes `SynapseUNetTyped<D>`. Vec<usize> sync
// indices stay on host (small integer LUTs read at op build time).
//
// Constructed via `from_untyped(&CtmWeights)` for migration from
// existing checkpoints, or via `from_host(...)` for fresh init —
// caller passes either the existing untyped CtmWeights or the raw
// host-side parameter slices and we upload to D.
//
// v0 scope: data container + transfer constructors + n_params.
// `forward<D>` lives in forward.rs (separate slice).

use modgrad_device::backend::tensor as tensor_api;
use modgrad_device::backend::tensor::{Device, Tensor};
use modgrad_device::backend::BackendError;
use crate::synapse::SynapseUNetTyped;

/// Path C typed CTM weight container. Same layout as `CtmWeights`,
/// but every f32 buffer is a `Tensor<D>` so the same struct works
/// on Cpu and Rocm (and any future `D: Device`).
pub struct CtmWeightsTyped<D: Device> {
    pub config: CtmConfig,
    pub synapse: SynapseUNetTyped<D>,
    pub nlm_stage1: tensor_api::SuperLinear<D>,
    pub nlm_stage2: Option<tensor_api::SuperLinear<D>>,

    /// `[d_model]`
    pub start_activated: Tensor<D>,
    /// `[d_model × memory_length]`
    pub start_trace: Tensor<D>,

    pub kv_proj: tensor_api::Linear<D>,
    pub kv_ln_gamma: Tensor<D>,
    pub kv_ln_beta: Tensor<D>,

    pub q_proj: tensor_api::Linear<D>,
    pub mha_in_proj: tensor_api::Linear<D>,
    pub mha_out_proj: tensor_api::Linear<D>,

    /// Sync topology — integer LUTs, stay on host.
    pub sync_out_left: Vec<usize>,
    pub sync_out_right: Vec<usize>,
    pub sync_action_left: Vec<usize>,
    pub sync_action_right: Vec<usize>,

    pub decay_params_out: Tensor<D>,
    pub decay_params_action: Tensor<D>,

    pub output_proj: tensor_api::Linear<D>,
    pub exit_gate: Option<tensor_api::Linear<D>>,
}

impl<D: Device> CtmWeightsTyped<D> {
    /// Migration constructor: transfer all buffers from the existing
    /// untyped `CtmWeights` into device-resident `Tensor<D>`s. After
    /// this call, every weight read goes through the typed cascade
    /// — for `D = Rocm`, weights stay on-device for all subsequent
    /// forward/backward calls (no per-call PCIe upload).
    pub fn from_untyped(w: &CtmWeights) -> Result<Self, BackendError> {
        let cfg = &w.config;
        let synapse = SynapseUNetTyped::<D>::from_untyped(&w.synapse)?;

        let nlm_stage1 = tensor_api::SuperLinear::<D>::from_host(
            &w.nlm_stage1.weights, &w.nlm_stage1.biases,
            w.nlm_stage1.n_neurons, w.nlm_stage1.in_per, w.nlm_stage1.out_per,
        )?;
        let nlm_stage2 = match &w.nlm_stage2 {
            Some(s2) => Some(tensor_api::SuperLinear::<D>::from_host(
                &s2.weights, &s2.biases, s2.n_neurons, s2.in_per, s2.out_per,
            )?),
            None => None,
        };

        let kv_proj = tensor_api::Linear::<D>::from_host(
            &w.kv_proj.weight, &w.kv_proj.bias,
            w.kv_proj.in_dim, w.kv_proj.out_dim,
        )?;
        let q_proj = tensor_api::Linear::<D>::from_host(
            &w.q_proj.weight, &w.q_proj.bias,
            w.q_proj.in_dim, w.q_proj.out_dim,
        )?;
        let mha_in_proj = tensor_api::Linear::<D>::from_host(
            &w.mha_in_proj.weight, &w.mha_in_proj.bias,
            w.mha_in_proj.in_dim, w.mha_in_proj.out_dim,
        )?;
        let mha_out_proj = tensor_api::Linear::<D>::from_host(
            &w.mha_out_proj.weight, &w.mha_out_proj.bias,
            w.mha_out_proj.in_dim, w.mha_out_proj.out_dim,
        )?;
        let output_proj = tensor_api::Linear::<D>::from_host(
            &w.output_proj.weight, &w.output_proj.bias,
            w.output_proj.in_dim, w.output_proj.out_dim,
        )?;
        let exit_gate = match &w.exit_gate {
            Some(g) => Some(tensor_api::Linear::<D>::from_host(
                &g.weight, &g.bias, g.in_dim, g.out_dim,
            )?),
            None => None,
        };

        Ok(Self {
            config: cfg.clone(),
            synapse, nlm_stage1, nlm_stage2,
            start_activated: Tensor::<D>::from_slice(&w.start_activated)?,
            start_trace: Tensor::<D>::from_slice(&w.start_trace)?,
            kv_proj,
            kv_ln_gamma: Tensor::<D>::from_slice(&w.kv_ln_gamma)?,
            kv_ln_beta: Tensor::<D>::from_slice(&w.kv_ln_beta)?,
            q_proj, mha_in_proj, mha_out_proj,
            sync_out_left: w.sync_out_left.clone(),
            sync_out_right: w.sync_out_right.clone(),
            sync_action_left: w.sync_action_left.clone(),
            sync_action_right: w.sync_action_right.clone(),
            decay_params_out: Tensor::<D>::from_slice(&w.decay_params_out)?,
            decay_params_action: Tensor::<D>::from_slice(&w.decay_params_action)?,
            output_proj, exit_gate,
        })
    }

    /// Total trainable parameters — matches `CtmWeights::n_params`
    /// element-for-element, since the typed buffers carry the same
    /// number of f32s.
    pub fn n_params(&self) -> usize {
        // Synapse n_params via its host counterpart's formula —
        // the typed struct stores the same layout, so the count
        // matches; we sum directly.
        let mut n = 0usize;
        n += self.synapse.first_projection.linear.weight.len()
            + self.synapse.first_projection.linear.bias.len()
            + self.synapse.first_projection.ln_gamma.len() * 2;
        for i in 0..self.synapse.down_blocks.len() {
            n += self.synapse.down_blocks[i].linear.weight.len()
                + self.synapse.down_blocks[i].linear.bias.len()
                + self.synapse.down_blocks[i].ln_gamma.len() * 2;
            n += self.synapse.up_blocks[i].linear.weight.len()
                + self.synapse.up_blocks[i].linear.bias.len()
                + self.synapse.up_blocks[i].ln_gamma.len() * 2;
            n += self.synapse.skip_ln_gamma[i].len() * 2;
        }
        n += self.nlm_stage1.weights.len() + self.nlm_stage1.biases.len();
        if let Some(ref s2) = self.nlm_stage2 {
            n += s2.weights.len() + s2.biases.len();
        }
        n += self.start_activated.len() + self.start_trace.len();
        n += self.kv_proj.weight.len() + self.kv_proj.bias.len();
        n += self.kv_ln_gamma.len() * 2;
        n += self.q_proj.weight.len() + self.q_proj.bias.len();
        n += self.mha_in_proj.weight.len() + self.mha_in_proj.bias.len();
        n += self.mha_out_proj.weight.len() + self.mha_out_proj.bias.len();
        n += self.decay_params_out.len() + self.decay_params_action.len();
        n += self.output_proj.weight.len() + self.output_proj.bias.len();
        if let Some(ref g) = self.exit_gate { n += g.weight.len() + g.bias.len(); }
        n
    }
}

// ─── CtmGradientsTyped<D> — Path C gradient container ────────
//
// Mirrors the untyped `crate::train::CtmGradients` shape but stores
// every gradient as a `Tensor<D>`. Constructed via `zeros(&typed)`
// from a `CtmWeightsTyped<D>` so each grad buffer has the same
// length as its matching weight buffer. Backward primitives
// (`Linear<D>::backward`, `SuperLinear<D>::backward`,
// `layer_norm_bwd`, etc.) accumulate into these tensors.
//
// Note: the full SynapseUNet backward (UNet skip-conn adjoint)
// requires a typed `UNetGrads` analog, which is non-trivial. v0
// includes a placeholder `synapse_grads` field that follow-up
// commits will flesh out per the existing `UNetGrads` layout.

/// Typed gradient container — same field structure as
/// `crate::train::CtmGradients`, every f32 buffer typed.
pub struct CtmGradientsTyped<D: Device> {
    /// SynapseUNet gradients — placeholder for now. The actual
    /// per-block typed UNetGrads land in the SynapseUNet<D>::backward
    /// commit.
    pub synapse_placeholder: Tensor<D>, // size 0 sentinel; replace with real typed UNetGrads
    pub nlm_s1_w: Tensor<D>,
    pub nlm_s1_b: Tensor<D>,
    pub nlm_s2_w: Option<Tensor<D>>,
    pub nlm_s2_b: Option<Tensor<D>>,
    pub d_start_activated: Tensor<D>,
    pub d_start_trace: Tensor<D>,
    pub kv_proj_w: Tensor<D>,
    pub kv_proj_b: Tensor<D>,
    pub kv_ln_d_gamma: Tensor<D>,
    pub kv_ln_d_beta: Tensor<D>,
    pub q_proj_w: Tensor<D>,
    pub q_proj_b: Tensor<D>,
    pub mha_in_w: Tensor<D>,
    pub mha_in_b: Tensor<D>,
    pub mha_out_w: Tensor<D>,
    pub mha_out_b: Tensor<D>,
    pub d_decay_out: Tensor<D>,
    pub d_decay_action: Tensor<D>,
    pub out_proj_w: Tensor<D>,
    pub out_proj_b: Tensor<D>,
    pub exit_gate_w: Option<Tensor<D>>,
    pub exit_gate_b: Option<Tensor<D>>,
}

impl<D: Device> CtmGradientsTyped<D> {
    /// Allocate zeroed gradient tensors matching the weight shapes.
    pub fn zeros(w: &CtmWeightsTyped<D>) -> Result<Self, BackendError> {
        Ok(Self {
            synapse_placeholder: Tensor::<D>::zeros(0)?,
            nlm_s1_w: Tensor::<D>::zeros(w.nlm_stage1.weights.len())?,
            nlm_s1_b: Tensor::<D>::zeros(w.nlm_stage1.biases.len())?,
            nlm_s2_w: match &w.nlm_stage2 {
                Some(s) => Some(Tensor::<D>::zeros(s.weights.len())?),
                None => None,
            },
            nlm_s2_b: match &w.nlm_stage2 {
                Some(s) => Some(Tensor::<D>::zeros(s.biases.len())?),
                None => None,
            },
            d_start_activated: Tensor::<D>::zeros(w.config.d_model)?,
            d_start_trace: Tensor::<D>::zeros(w.config.d_model * w.config.memory_length)?,
            kv_proj_w: Tensor::<D>::zeros(w.kv_proj.weight.len())?,
            kv_proj_b: Tensor::<D>::zeros(w.kv_proj.bias.len())?,
            kv_ln_d_gamma: Tensor::<D>::zeros(w.kv_ln_gamma.len())?,
            kv_ln_d_beta: Tensor::<D>::zeros(w.kv_ln_beta.len())?,
            q_proj_w: Tensor::<D>::zeros(w.q_proj.weight.len())?,
            q_proj_b: Tensor::<D>::zeros(w.q_proj.bias.len())?,
            mha_in_w: Tensor::<D>::zeros(w.mha_in_proj.weight.len())?,
            mha_in_b: Tensor::<D>::zeros(w.mha_in_proj.bias.len())?,
            mha_out_w: Tensor::<D>::zeros(w.mha_out_proj.weight.len())?,
            mha_out_b: Tensor::<D>::zeros(w.mha_out_proj.bias.len())?,
            d_decay_out: Tensor::<D>::zeros(w.decay_params_out.len())?,
            d_decay_action: Tensor::<D>::zeros(w.decay_params_action.len())?,
            out_proj_w: Tensor::<D>::zeros(w.output_proj.weight.len())?,
            out_proj_b: Tensor::<D>::zeros(w.output_proj.bias.len())?,
            exit_gate_w: match &w.exit_gate {
                Some(g) => Some(Tensor::<D>::zeros(g.weight.len())?),
                None => None,
            },
            exit_gate_b: match &w.exit_gate {
                Some(g) => Some(Tensor::<D>::zeros(g.bias.len())?),
                None => None,
            },
        })
    }

    /// Backward through the output projection only. Given upstream
    /// `d_pred` (= dL/dpred for one tick) and the cached `sync_out`
    /// from the forward pass, accumulate `d_W += d_pred ⊗ sync_out`,
    /// `d_b += d_pred`, and overwrite `d_sync_out = W^T · d_pred`.
    ///
    /// This is the simplest backward stage in the CTM tick — the
    /// other stages (NLM, synapse U-Net, MHA, sync_update reverse,
    /// q_proj, kv_proj) compose on top of this and ride matvec_t +
    /// outer_product_acc the same way `Linear<D>::backward` does.
    pub fn backward_output_proj(
        &mut self,
        w: &CtmWeightsTyped<D>,
        d_pred: &Tensor<D>,
        sync_out: &Tensor<D>,
        d_sync_out: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        w.output_proj.backward(
            d_pred, sync_out,
            &mut self.out_proj_w, &mut self.out_proj_b,
            d_sync_out,
        )
    }

    /// Backward through `q_proj`: given upstream `d_q` and the
    /// cached `sync_action` from forward, accumulate `d_W += d_q ⊗ sync_action`,
    /// `d_b += d_q`, and overwrite `d_sync_action = W^T · d_q`.
    pub fn backward_q_proj(
        &mut self,
        w: &CtmWeightsTyped<D>,
        d_q: &Tensor<D>,
        sync_action: &Tensor<D>,
        d_sync_action: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        w.q_proj.backward(
            d_q, sync_action,
            &mut self.q_proj_w, &mut self.q_proj_b,
            d_sync_action,
        )
    }

    /// Backward through `kv_proj`: given upstream `d_kv_pre_ln`
    /// (gradient flowing back through the LN to kv_proj output)
    /// and the cached observation token, accumulate weight grads
    /// and overwrite `d_obs`.
    pub fn backward_kv_proj(
        &mut self,
        w: &CtmWeightsTyped<D>,
        d_kv_pre_ln: &Tensor<D>,
        obs: &Tensor<D>,
        d_obs: &mut Tensor<D>,
    ) -> Result<(), BackendError> {
        w.kv_proj.backward(
            d_kv_pre_ln, obs,
            &mut self.kv_proj_w, &mut self.kv_proj_b,
            d_obs,
        )
    }

    /// Backward through `kv_ln`: composition of `layer_norm_bwd`.
    /// Given `d_kv_post_ln` (gradient flowing back from MHA into LN
    /// output), `kv_pre_ln_cached` (the LN input from forward), and
    /// `ln_cache` (mean/rstd produced by `layer_norm_train` at forward),
    /// accumulate `d_gamma`, `d_beta` and overwrite `d_kv_pre_ln`.
    pub fn backward_kv_ln(
        &mut self,
        w: &CtmWeightsTyped<D>,
        d_kv_post_ln: &Tensor<D>,
        kv_pre_ln: &Tensor<D>,
        ln_cache: &Tensor<D>,
        d_kv_pre_ln: &mut Tensor<D>,
        d_in: usize,
    ) -> Result<(), BackendError> {
        // `layer_norm_bwd` accumulates into d_gamma/d_beta and
        // overwrites d_x. n_rows = 1 (single token), n_cols = d_in.
        tensor_api::layer_norm_bwd(
            d_kv_post_ln, kv_pre_ln, &w.kv_ln_gamma, ln_cache,
            d_kv_pre_ln,
            &mut self.kv_ln_d_gamma, &mut self.kv_ln_d_beta,
            1, d_in,
        )
    }
}

#[cfg(test)]
mod ctm_weights_typed_tests {
    use super::*;
    use modgrad_device::backend::tensor::Cpu;
    #[cfg(feature = "rocm")]
    use modgrad_device::backend::tensor::Rocm;

    /// Build a small CtmWeights, port to CtmWeightsTyped<Cpu>,
    /// assert n_params matches and that key buffers transferred
    /// without corruption (round-trip through to_vec).
    #[test]
    fn cpu_ctm_weights_typed_round_trip() {
        let cfg = CtmConfig {
            iterations: 2,
            d_model: 4,
            d_input: 6,
            heads: 1,
            n_synch_out: 4,
            n_synch_action: 4,
            synapse_depth: 2,
            memory_length: 4,
            deep_nlms: false,
            memory_hidden_dims: 0,
            out_dims: 3,
            n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg.clone(), 8);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w)
            .expect("from_untyped");

        // n_params must match.
        assert_eq!(typed.n_params(), w.n_params(),
            "typed n_params {} vs untyped {}", typed.n_params(), w.n_params());

        // start_activated round-trip.
        let sa = typed.start_activated.to_vec().expect("start_activated to_vec");
        assert_eq!(sa, w.start_activated);

        // kv_proj weight round-trip.
        let kvw = typed.kv_proj.weight.to_vec().expect("kv_proj weight");
        assert_eq!(kvw, w.kv_proj.weight);

        // sync indices preserved exactly.
        assert_eq!(typed.sync_out_left, w.sync_out_left);
        assert_eq!(typed.sync_action_right, w.sync_action_right);
    }

    #[test]
    fn cpu_ctm_gradients_typed_zeros() {
        let cfg = CtmConfig {
            iterations: 2, d_model: 4, d_input: 6, heads: 1,
            n_synch_out: 4, n_synch_action: 4,
            synapse_depth: 2, memory_length: 4, deep_nlms: false,
            memory_hidden_dims: 0, out_dims: 3, n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg, 8);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let grads = CtmGradientsTyped::<Cpu>::zeros(&typed).expect("grads");
        // Sizes must match weights.
        assert_eq!(grads.kv_proj_w.len(), w.kv_proj.weight.len());
        assert_eq!(grads.q_proj_w.len(), w.q_proj.weight.len());
        assert_eq!(grads.out_proj_w.len(), w.output_proj.weight.len());
        assert_eq!(grads.d_decay_out.len(), w.decay_params_out.len());
        // All zero-initialised.
        let z = grads.kv_proj_w.to_vec().unwrap();
        assert!(z.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cpu_ctm_gradients_backward_q_proj_and_kv_proj_smoke() {
        // Verifies q_proj and kv_proj backward wiring composes cleanly.
        let cfg = CtmConfig {
            iterations: 1, d_model: 2, d_input: 4, heads: 1,
            n_synch_out: 2, n_synch_action: 2,
            synapse_depth: 1, memory_length: 2, deep_nlms: false,
            memory_hidden_dims: 0, out_dims: 3, n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let raw_in = 4;
        let w = CtmWeights::new(cfg, raw_in);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut grads = CtmGradientsTyped::<Cpu>::zeros(&typed).unwrap();

        // q_proj: in_dim = synch_action_size (2), out_dim = d_input (4)
        let d_q = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let sync_action = Tensor::<Cpu>::from_slice(&[1.0, -1.0]).unwrap();
        let mut d_sa = Tensor::<Cpu>::zeros(2).unwrap();
        grads.backward_q_proj(&typed, &d_q, &sync_action, &mut d_sa).unwrap();
        let dq_w = grads.q_proj_w.to_vec().unwrap();
        // d_W shape = [4 × 2] flat = 8. d_q ⊗ sync_action = each row = [d_q[i]*1, d_q[i]*(-1)]
        // = [1, -1, 1, -1, 1, -1, 1, -1].
        assert_eq!(dq_w, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let dq_b = grads.q_proj_b.to_vec().unwrap();
        assert_eq!(dq_b, vec![1.0, 1.0, 1.0, 1.0]);

        // kv_proj: in_dim = raw_in (4), out_dim = d_input (4)
        let d_kv_pre_ln = Tensor::<Cpu>::from_slice(&[1.0, 0.0, 0.0, -1.0]).unwrap();
        let obs = Tensor::<Cpu>::from_slice(&[2.0, 0.0, 1.0, 0.0]).unwrap();
        let mut d_obs = Tensor::<Cpu>::zeros(4).unwrap();
        grads.backward_kv_proj(&typed, &d_kv_pre_ln, &obs, &mut d_obs).unwrap();
        let dkv_b = grads.kv_proj_b.to_vec().unwrap();
        assert_eq!(dkv_b, vec![1.0, 0.0, 0.0, -1.0]);
        // d_W rows: [d[0]*obs[0..4], d[1]*..., d[2]*..., d[3]*...]
        // = row 0: [2,0,1,0]; row 1: [0,0,0,0]; row 2: [0,0,0,0]; row 3: [-2,0,-1,0]
        let dkv_w = grads.kv_proj_w.to_vec().unwrap();
        assert_eq!(dkv_w, vec![
            2.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -2.0, 0.0, -1.0, 0.0,
        ]);
    }

    #[test]
    fn cpu_ctm_gradients_backward_kv_ln_composes_layer_norm_bwd() {
        // Build a small LN forward-with-cache, then run backward
        // through CtmGradientsTyped::backward_kv_ln. Verify d_gamma
        // and d_beta accumulate (sums across rows).
        let cfg = CtmConfig {
            iterations: 1, d_model: 2, d_input: 4, heads: 1,
            n_synch_out: 2, n_synch_action: 2,
            synapse_depth: 1, memory_length: 2, deep_nlms: false,
            memory_hidden_dims: 0, out_dims: 3, n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg.clone(), 4);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut grads = CtmGradientsTyped::<Cpu>::zeros(&typed).unwrap();

        // Hand-cooked kv_pre_ln; build LN cache via layer_norm_train.
        let kv_pre_ln_data = vec![1.0, 2.0, 3.0, 4.0];
        let kv_pre_ln = Tensor::<Cpu>::from_slice(&kv_pre_ln_data).unwrap();
        let mut kv_post_ln = Tensor::<Cpu>::zeros(4).unwrap();
        let mut ln_cache = Tensor::<Cpu>::zeros(2).unwrap(); // 2 * n_rows = 2
        tensor_api::layer_norm_train(
            &kv_pre_ln, &typed.kv_ln_gamma, &typed.kv_ln_beta,
            &mut kv_post_ln, &mut ln_cache, 1, 4,
        ).unwrap();

        // Hand-cooked d_kv_post_ln.
        let d_kv_post_ln = Tensor::<Cpu>::from_slice(&[1.0, 0.0, 0.0, -1.0]).unwrap();
        let mut d_kv_pre_ln = Tensor::<Cpu>::zeros(4).unwrap();
        grads.backward_kv_ln(
            &typed, &d_kv_post_ln, &kv_pre_ln, &ln_cache,
            &mut d_kv_pre_ln, 4,
        ).unwrap();

        // d_kv_pre_ln must have zero mean (LayerNorm invariant).
        let dx = d_kv_pre_ln.to_vec().unwrap();
        let mean = dx.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "d_x mean should be ~0, got {}", mean);

        // d_gamma and d_beta must be non-zero (gradient flowed).
        let dg = grads.kv_ln_d_gamma.to_vec().unwrap();
        let db = grads.kv_ln_d_beta.to_vec().unwrap();
        assert!(dg.iter().any(|&v| v.abs() > 1e-6),
            "d_gamma all zero — backward did not flow");
        assert!(db.iter().any(|&v| v.abs() > 1e-6),
            "d_beta all zero — backward did not flow");
    }

    #[test]
    fn cpu_ctm_gradients_backward_output_proj_smoke() {
        // Output proj backward through the gradient container.
        // Verifies the wiring; correctness vs untyped is a follow-up
        // commit once full ctm_backward_typed lands.
        let cfg = CtmConfig {
            iterations: 1, d_model: 2, d_input: 4, heads: 1,
            n_synch_out: 2, n_synch_action: 2,
            synapse_depth: 1, memory_length: 2, deep_nlms: false,
            memory_hidden_dims: 0, out_dims: 3, n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg, 4);
        let typed = CtmWeightsTyped::<Cpu>::from_untyped(&w).unwrap();
        let mut grads = CtmGradientsTyped::<Cpu>::zeros(&typed).unwrap();

        // Hand-crafted upstream: d_pred = ones[3], sync_out = [1, -1].
        let d_pred = Tensor::<Cpu>::from_slice(&[1.0, 1.0, 1.0]).unwrap();
        let sync_out = Tensor::<Cpu>::from_slice(&[1.0, -1.0]).unwrap();
        let mut d_sync_out = Tensor::<Cpu>::zeros(2).unwrap();

        grads.backward_output_proj(&typed, &d_pred, &sync_out, &mut d_sync_out)
            .expect("backward_output_proj");

        // out_proj_w is row-major [3 × 2]. d_W = d_pred ⊗ sync_out
        // = [[1·1, 1·(-1)], [1·1, 1·(-1)], [1·1, 1·(-1)]]
        // = [[1, -1], [1, -1], [1, -1]] flat = [1, -1, 1, -1, 1, -1].
        let dw = grads.out_proj_w.to_vec().unwrap();
        assert_eq!(dw, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        // out_proj_b += d_pred = [1, 1, 1].
        let db = grads.out_proj_b.to_vec().unwrap();
        assert_eq!(db, vec![1.0, 1.0, 1.0]);
        // d_sync_out = W^T · d_pred. We can't predict the exact value
        // without the random-init weight values, but it should be
        // non-zero (most likely) and length 2.
        let dso = d_sync_out.to_vec().unwrap();
        assert_eq!(dso.len(), 2);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_ctm_weights_typed_constructible() {
        let cfg = CtmConfig {
            iterations: 2, d_model: 4, d_input: 6, heads: 1,
            n_synch_out: 4, n_synch_action: 4,
            synapse_depth: 2, memory_length: 4, deep_nlms: false,
            memory_hidden_dims: 0, out_dims: 3, n_random_pairing_self: 0,
            min_width: 2,
            exit_strategy: crate::config::ExitStrategy::None,
            collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg, 8);
        let typed = match CtmWeightsTyped::<Rocm>::from_untyped(&w) {
            Ok(t) => t,
            Err(BackendError::Runtime(msg)) if msg.contains("hipMalloc") => return,
            Err(e) => panic!("rocm CtmWeightsTyped build failed: {:?}", e),
        };
        assert_eq!(typed.n_params(), w.n_params());
        // Round-trip: download a buffer, compare.
        let sa = typed.start_activated.to_vec().expect("start_activated to_vec");
        for (a, b) in sa.iter().zip(&w.start_activated) {
            assert!((a - b).abs() < 1e-6,
                "start_activated diverged on Rocm round-trip: {} vs {}", a, b);
        }
    }
}

/// Per-forward mutable state (not persisted).
pub struct CtmState {
    /// Pre-activation history: [d_model × memory_length], row-major by neuron.
    pub trace: Vec<f32>,
    /// Post-activation state: [d_model].
    pub activated: Vec<f32>,

    /// Sync accumulators.
    pub alpha_action: Option<Vec<f32>>,
    pub beta_action: Option<Vec<f32>>,
    pub alpha_out: Vec<f32>,
    pub beta_out: Vec<f32>,

    /// Episodic memory: persistent KV buffer for long-range context.
    /// When present, MHA attends over episodic entries in addition to
    /// the current observation. Entries accumulate across forward calls
    /// with bounded memory via hierarchical compression.
    pub episodic: Option<modgrad_compute::kv_buffer::EpisodicMemory>,
}

impl CtmState {
    pub fn new(w: &CtmWeights) -> Self {
        Self {
            trace: w.start_trace.clone(),
            activated: w.start_activated.clone(),
            alpha_action: None,
            beta_action: None,
            alpha_out: vec![0.0; w.config.n_synch_out],
            beta_out: vec![0.0; w.config.n_synch_out],
            episodic: None,
        }
    }

    /// Create state with episodic memory enabled.
    /// `short`, `mid`, `long`: capacity in entries per tier.
    pub fn with_episodic(w: &CtmWeights, short: usize, mid: usize, long: usize) -> Self {
        let mut s = Self::new(w);
        s.episodic = Some(modgrad_compute::kv_buffer::EpisodicMemory::new(
            w.config.d_input, short, mid, long,
        ));
        s
    }
}

fn random_indices(max: usize, count: usize, rng: &mut SimpleRng) -> Vec<usize> {
    (0..count).map(|_| (rng.next_u64() as usize) % max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CtmConfig, ExitStrategy};

    fn small_config() -> CtmConfig {
        CtmConfig {
            iterations: 4,
            d_model: 16,
            d_input: 8,
            heads: 2,
            n_synch_out: 16,
            n_synch_action: 8,
            synapse_depth: 1,
            memory_length: 4,
            deep_nlms: false,
            memory_hidden_dims: 0,
            out_dims: 3,
            n_random_pairing_self: 0,
            min_width: 4,
            exit_strategy: ExitStrategy::None,
            collect_trajectories: false,
        }
    }

    fn small_config_deep() -> CtmConfig {
        CtmConfig {
            deep_nlms: true,
            memory_hidden_dims: 2,
            ..small_config()
        }
    }

    // ── CtmWeights::new ────────────────────────────────────────

    #[test]
    fn weights_new_creates_correct_start_state_sizes() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.start_activated.len(), cfg.d_model);
        assert_eq!(w.start_trace.len(), cfg.d_model * cfg.memory_length);
    }

    #[test]
    fn weights_new_kv_proj_dimensions() {
        let raw_input = 10;
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), raw_input);
        assert_eq!(w.kv_proj.in_dim, raw_input);
        assert_eq!(w.kv_proj.out_dim, cfg.d_input);
        assert_eq!(w.kv_ln_gamma.len(), cfg.d_input);
        assert_eq!(w.kv_ln_beta.len(), cfg.d_input);
    }

    #[test]
    fn weights_new_mha_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.mha_in_proj.in_dim, cfg.d_input);
        assert_eq!(w.mha_in_proj.out_dim, 3 * cfg.d_input);
        assert_eq!(w.mha_out_proj.in_dim, cfg.d_input);
        assert_eq!(w.mha_out_proj.out_dim, cfg.d_input);
    }

    #[test]
    fn weights_new_output_proj_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.output_proj.in_dim, cfg.n_synch_out);
        assert_eq!(w.output_proj.out_dim, cfg.out_dims);
    }

    #[test]
    fn weights_new_sync_indices_correct_length() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.sync_out_left.len(), cfg.n_synch_out);
        assert_eq!(w.sync_out_right.len(), cfg.n_synch_out);
        assert_eq!(w.sync_action_left.len(), cfg.n_synch_action);
        assert_eq!(w.sync_action_right.len(), cfg.n_synch_action);
    }

    #[test]
    fn weights_new_sync_indices_in_range() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        for &idx in &w.sync_out_left {
            assert!(idx < cfg.d_model, "sync_out_left index {idx} >= d_model {}", cfg.d_model);
        }
        for &idx in &w.sync_action_left {
            assert!(idx < cfg.d_model);
        }
    }

    #[test]
    fn weights_new_decay_params() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        assert_eq!(w.decay_params_out.len(), cfg.n_synch_out);
        assert_eq!(w.decay_params_action.len(), cfg.n_synch_action);
        assert!(w.decay_params_out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn weights_no_exit_gate_when_none_strategy() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        assert!(w.exit_gate.is_none());
    }

    #[test]
    fn weights_has_exit_gate_when_adaptive() {
        let mut cfg = small_config();
        cfg.exit_strategy = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 };
        let w = CtmWeights::new(cfg.clone(), 10);
        assert!(w.exit_gate.is_some());
        let gate = w.exit_gate.as_ref().unwrap();
        assert_eq!(gate.in_dim, cfg.n_synch_out);
        assert_eq!(gate.out_dim, 1);
    }

    // ── Deep NLM stages ────────────────────────────────────────

    #[test]
    fn weights_shallow_nlm_no_stage2() {
        let cfg = small_config(); // deep_nlms = false
        let w = CtmWeights::new(cfg, 10);
        assert!(w.nlm_stage2.is_none());
    }

    #[test]
    fn weights_deep_nlm_has_stage2() {
        let cfg = small_config_deep();
        let w = CtmWeights::new(cfg, 10);
        assert!(w.nlm_stage2.is_some());
    }

    // ── n_params ───────────────────────────────────────────────

    #[test]
    fn n_params_positive_and_consistent() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let n = w.n_params();
        assert!(n > 0, "n_params should be positive");

        // Create the same config again — deterministic, so should match
        let cfg2 = small_config();
        let w2 = CtmWeights::new(cfg2, 10);
        assert_eq!(w2.n_params(), n, "n_params should be deterministic");
    }

    #[test]
    fn n_params_deep_greater_than_shallow() {
        let w_shallow = CtmWeights::new(small_config(), 10);
        let w_deep = CtmWeights::new(small_config_deep(), 10);
        assert!(
            w_deep.n_params() > w_shallow.n_params(),
            "deep NLM should have more params: {} vs {}",
            w_deep.n_params(), w_shallow.n_params()
        );
    }

    #[test]
    fn n_params_includes_gate_when_present() {
        let mut cfg_no_gate = small_config();
        cfg_no_gate.exit_strategy = ExitStrategy::None;
        let mut cfg_gate = small_config();
        cfg_gate.exit_strategy = ExitStrategy::AdaptiveGate { beta: 0.1, threshold: 0.5 };

        let w_no = CtmWeights::new(cfg_no_gate, 10);
        let w_yes = CtmWeights::new(cfg_gate, 10);

        assert!(
            w_yes.n_params() > w_no.n_params(),
            "gate should add params: {} vs {}",
            w_yes.n_params(), w_no.n_params()
        );
    }

    // ── CtmState::new ──────────────────────────────────────────

    #[test]
    fn state_new_dimensions() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg.clone(), 10);
        let s = CtmState::new(&w);

        assert_eq!(s.activated.len(), cfg.d_model);
        assert_eq!(s.trace.len(), cfg.d_model * cfg.memory_length);
        assert_eq!(s.alpha_out.len(), cfg.n_synch_out);
        assert_eq!(s.beta_out.len(), cfg.n_synch_out);
        assert!(s.alpha_action.is_none());
        assert!(s.beta_action.is_none());
        assert!(s.episodic.is_none());
    }

    #[test]
    fn state_new_copies_start_state() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::new(&w);

        assert_eq!(s.activated, w.start_activated);
        assert_eq!(s.trace, w.start_trace);
    }

    #[test]
    fn state_new_sync_accumulators_zeroed() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::new(&w);

        assert!(s.alpha_out.iter().all(|&v| v == 0.0));
        assert!(s.beta_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn state_with_episodic_creates_memory() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);
        let s = CtmState::with_episodic(&w, 8, 4, 2);

        assert!(s.episodic.is_some());
    }

    // ── Save/load round-trip ───────────────────────────────────

    #[test]
    fn weights_save_load_roundtrip() {
        let cfg = small_config();
        let w = CtmWeights::new(cfg, 10);

        let dir = std::env::temp_dir().join("modgrad_ctm_test_weights");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_weights.json");
        let path_str = path.to_str().unwrap();

        w.save(path_str).unwrap();
        let w2 = CtmWeights::load(path_str).unwrap();

        assert_eq!(w2.n_params(), w.n_params());
        assert_eq!(w2.start_activated, w.start_activated);
        assert_eq!(w2.start_trace, w.start_trace);
        assert_eq!(w2.config.d_model, w.config.d_model);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Self-pairing ───────────────────────────────────────────

    #[test]
    fn self_pairing_matches_left_right() {
        let mut cfg = small_config();
        cfg.n_random_pairing_self = 4;
        let w = CtmWeights::new(cfg, 10);

        for i in 0..4 {
            assert_eq!(
                w.sync_out_left[i], w.sync_out_right[i],
                "self-pairing mismatch at index {i}"
            );
        }
    }
}
