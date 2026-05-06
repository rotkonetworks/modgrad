//! Brain checkpoint save/load.
//!
//! `penumbra_train` runs imitation training in the typed cascade
//! (`RegionalWeightsTyped<Cpu>`). To use those trained weights in
//! `penumbra_live_arena`, we need to persist them to disk and load
//! them back. Path C's typed tensors aren't directly serializable
//! because their buffers may live on a device — so the format goes
//! through the untyped `RegionalWeights` (Serialize-deriving, all
//! `Vec<f32>` buffers), which `modgrad-persist` already handles.
//!
//! Round-trip:
//!   1. **Save**: typed → host vecs → overwrite a fresh `RegionalWeights`
//!      built from the same config → `persist::save`.
//!   2. **Load**: `persist::load` → untyped → `from_untyped` → typed.
//!
//! Step 1 needs `copy_typed_into_untyped`, which walks every weight
//! tensor in the typed cascade and assigns the host-extracted values
//! to the corresponding field of an untyped template. The template
//! is always built from `RegionalWeights::new(typed.config.clone())`
//! so the random-init buffers exist with correct shapes — they're
//! immediately overwritten.

use anyhow::Result;
use std::path::Path;

use modgrad_ctm::graph::{RegionalWeights, RegionalWeightsTyped};
use modgrad_device::backend::tensor::Device;

/// Persist a typed brain to `path`. Format follows `modgrad-persist`'s
/// extension-based dispatch (`.json` → JSON, anything else → wincode).
pub fn save<D: Device>(typed: &RegionalWeightsTyped<D>, path: impl AsRef<Path>) -> Result<()> {
    let untyped = typed_to_untyped(typed)?;
    modgrad_persist::persist::save(&untyped, path)
        .map_err(|e| anyhow::anyhow!("persist::save: {e}"))?;
    Ok(())
}

/// Load a typed brain from `path`. Lifts the persisted untyped
/// representation back into `RegionalWeightsTyped<D>` via
/// `from_untyped`, which uploads to the target device.
pub fn load<D: Device>(path: impl AsRef<Path>) -> Result<RegionalWeightsTyped<D>> {
    let untyped: RegionalWeights = modgrad_persist::persist::load(path)
        .map_err(|e| anyhow::anyhow!("persist::load: {e}"))?;
    RegionalWeightsTyped::<D>::from_untyped(&untyped)
        .map_err(|e| anyhow::anyhow!("from_untyped: {e}"))
}

/// Perturb every weight tensor in `typed` with `N(0, std)` Gaussian
/// noise. Used by the PBT loop to mutate copied top-of-leaderboard
/// weights so the bottom-replaced agents explore nearby weight
/// space rather than becoming identical clones.
///
/// Uses a deterministic xorshift RNG so a given (seed, std) pair
/// always produces the same perturbation — important for repeatable
/// arena runs. Walks the same set of fields as `typed_to_untyped`
/// (every linear, super-linear, layer-norm gamma/beta, start state,
/// decay param, synapse block).
pub fn perturb_weights_in_place<D: Device>(
    typed: &mut RegionalWeightsTyped<D>,
    seed: u64, std: f32,
) -> Result<()> {
    if std <= 0.0 { return Ok(()); }
    let mut rng = XorRng::new(seed);

    perturb_tensor::<D>(&mut typed.embeddings, &mut rng, std)?;
    perturb_tensor::<D>(&mut typed.global_decay, &mut rng, std)?;
    perturb_linear::<D>(&mut typed.obs_proj, &mut rng, std)?;
    perturb_linear::<D>(&mut typed.output_proj, &mut rng, std)?;
    if let Some(g) = typed.outer_exit_gate.as_mut() {
        perturb_linear::<D>(g, &mut rng, std)?;
    }
    for c in typed.connection_synapses.iter_mut() {
        perturb_linear::<D>(c, &mut rng, std)?;
    }
    for r in typed.regions.iter_mut() {
        perturb_ctm_typed::<D>(r, &mut rng, std)?;
    }
    Ok(())
}

fn perturb_ctm_typed<D: Device>(
    t: &mut modgrad_ctm::weights::CtmWeightsTyped<D>,
    rng: &mut XorRng, std: f32,
) -> Result<()> {
    perturb_linear::<D>(&mut t.kv_proj, rng, std)?;
    perturb_linear::<D>(&mut t.q_proj, rng, std)?;
    perturb_linear::<D>(&mut t.mha_in_proj, rng, std)?;
    perturb_linear::<D>(&mut t.mha_out_proj, rng, std)?;
    perturb_linear::<D>(&mut t.output_proj, rng, std)?;
    if let Some(g) = t.exit_gate.as_mut() { perturb_linear::<D>(g, rng, std)?; }
    perturb_super_linear::<D>(&mut t.nlm_stage1, rng, std)?;
    if let Some(s2) = t.nlm_stage2.as_mut() { perturb_super_linear::<D>(s2, rng, std)?; }
    perturb_tensor::<D>(&mut t.start_activated, rng, std)?;
    perturb_tensor::<D>(&mut t.start_trace, rng, std)?;
    perturb_tensor::<D>(&mut t.kv_ln_gamma, rng, std)?;
    perturb_tensor::<D>(&mut t.kv_ln_beta, rng, std)?;
    perturb_tensor::<D>(&mut t.decay_params_out, rng, std)?;
    perturb_tensor::<D>(&mut t.decay_params_action, rng, std)?;
    perturb_synapse_block::<D>(&mut t.synapse.first_projection, rng, std)?;
    for d in t.synapse.down_blocks.iter_mut() { perturb_synapse_block::<D>(d, rng, std)?; }
    for u in t.synapse.up_blocks.iter_mut() { perturb_synapse_block::<D>(u, rng, std)?; }
    for g in t.synapse.skip_ln_gamma.iter_mut() { perturb_tensor::<D>(g, rng, std)?; }
    for b in t.synapse.skip_ln_beta.iter_mut() { perturb_tensor::<D>(b, rng, std)?; }
    Ok(())
}

fn perturb_synapse_block<D: Device>(
    b: &mut modgrad_ctm::synapse::SynapseBlockTyped<D>,
    rng: &mut XorRng, std: f32,
) -> Result<()> {
    perturb_linear::<D>(&mut b.linear, rng, std)?;
    perturb_tensor::<D>(&mut b.ln_gamma, rng, std)?;
    perturb_tensor::<D>(&mut b.ln_beta, rng, std)?;
    Ok(())
}

fn perturb_linear<D: Device>(
    l: &mut modgrad_device::backend::tensor::Linear<D>,
    rng: &mut XorRng, std: f32,
) -> Result<()> {
    perturb_tensor::<D>(&mut l.weight, rng, std)?;
    perturb_tensor::<D>(&mut l.bias, rng, std)?;
    Ok(())
}

fn perturb_super_linear<D: Device>(
    s: &mut modgrad_device::backend::tensor::SuperLinear<D>,
    rng: &mut XorRng, std: f32,
) -> Result<()> {
    perturb_tensor::<D>(&mut s.weights, rng, std)?;
    perturb_tensor::<D>(&mut s.biases, rng, std)?;
    Ok(())
}

fn perturb_tensor<D: Device>(
    t: &mut modgrad_device::backend::tensor::Tensor<D>,
    rng: &mut XorRng, std: f32,
) -> Result<()> {
    let mut h: Vec<f32> = t.to_vec()?;
    for v in h.iter_mut() {
        *v += std * rng.next_gauss();
    }
    *t = modgrad_device::backend::tensor::Tensor::<D>::from_slice(&h)?;
    Ok(())
}

/// Lightweight deterministic PRNG — Box-Muller Gaussian samples.
/// Lives here to keep checkpoint::perturb self-contained.
pub struct XorRng { state: u64 }
impl XorRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xdead_beef_cafe_babe } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Build an untyped `RegionalWeights` whose buffers carry the
/// trained values from the typed cascade. Uses the typed config as
/// the template and overwrites every weight buffer via `.to_vec()`
/// host extraction.
pub fn typed_to_untyped<D: Device>(
    typed: &RegionalWeightsTyped<D>,
) -> Result<RegionalWeights> {
    let mut out = RegionalWeights::new(typed.config.clone());

    // Embeddings.
    out.embeddings = typed.embeddings.to_vec()
        .map_err(|e| anyhow::anyhow!("embeddings.to_vec: {e}"))?;

    // Connection synapses.
    let n_conn = typed.connection_synapses.len().min(out.connection_synapses.len());
    for i in 0..n_conn {
        out.connection_synapses[i].weight = typed.connection_synapses[i].weight.to_vec()
            .map_err(|e| anyhow::anyhow!("conn[{i}].weight: {e}"))?;
        out.connection_synapses[i].bias = typed.connection_synapses[i].bias.to_vec()
            .map_err(|e| anyhow::anyhow!("conn[{i}].bias: {e}"))?;
    }

    // obs_proj + output_proj.
    out.obs_proj.weight = typed.obs_proj.weight.to_vec()?;
    out.obs_proj.bias = typed.obs_proj.bias.to_vec()?;
    out.output_proj.weight = typed.output_proj.weight.to_vec()?;
    out.output_proj.bias = typed.output_proj.bias.to_vec()?;

    // global_decay.
    out.global_decay = typed.global_decay.to_vec()?;

    // outer_exit_gate (optional).
    if let (Some(t), Some(u)) = (
        typed.outer_exit_gate.as_ref(), out.outer_exit_gate.as_mut(),
    ) {
        u.weight = t.weight.to_vec()?;
        u.bias = t.bias.to_vec()?;
    }

    // Per-region CTMs. Each region has its own bag of weights; we
    // overwrite them field by field. Mirrors CtmWeightsTyped<D>::from_untyped
    // in reverse.
    let n_regions = typed.regions.len().min(out.regions.len());
    for r in 0..n_regions {
        copy_ctm_typed_into_untyped(&typed.regions[r], &mut out.regions[r])?;
    }

    Ok(out)
}

fn copy_ctm_typed_into_untyped<D: Device>(
    t: &modgrad_ctm::weights::CtmWeightsTyped<D>,
    u: &mut modgrad_ctm::weights::CtmWeights,
) -> Result<()> {
    // Linears.
    u.kv_proj.weight = t.kv_proj.weight.to_vec()?;
    u.kv_proj.bias = t.kv_proj.bias.to_vec()?;
    u.q_proj.weight = t.q_proj.weight.to_vec()?;
    u.q_proj.bias = t.q_proj.bias.to_vec()?;
    u.mha_in_proj.weight = t.mha_in_proj.weight.to_vec()?;
    u.mha_in_proj.bias = t.mha_in_proj.bias.to_vec()?;
    u.mha_out_proj.weight = t.mha_out_proj.weight.to_vec()?;
    u.mha_out_proj.bias = t.mha_out_proj.bias.to_vec()?;
    u.output_proj.weight = t.output_proj.weight.to_vec()?;
    u.output_proj.bias = t.output_proj.bias.to_vec()?;
    if let (Some(tg), Some(ug)) = (t.exit_gate.as_ref(), u.exit_gate.as_mut()) {
        ug.weight = tg.weight.to_vec()?;
        ug.bias = tg.bias.to_vec()?;
    }

    // SuperLinear NLM stages.
    u.nlm_stage1.weights = t.nlm_stage1.weights.to_vec()?;
    u.nlm_stage1.biases = t.nlm_stage1.biases.to_vec()?;
    if let (Some(ts), Some(us)) = (t.nlm_stage2.as_ref(), u.nlm_stage2.as_mut()) {
        us.weights = ts.weights.to_vec()?;
        us.biases = ts.biases.to_vec()?;
    }

    // Bare buffers.
    u.start_activated = t.start_activated.to_vec()?;
    u.start_trace = t.start_trace.to_vec()?;
    u.kv_ln_gamma = t.kv_ln_gamma.to_vec()?;
    u.kv_ln_beta = t.kv_ln_beta.to_vec()?;
    u.decay_params_out = t.decay_params_out.to_vec()?;
    u.decay_params_action = t.decay_params_action.to_vec()?;

    // SynapseUNet — walk per-block.
    let n_down = t.synapse.down_blocks.len().min(u.synapse.down_blocks.len());
    let n_up = t.synapse.up_blocks.len().min(u.synapse.up_blocks.len());
    copy_synapse_block(&t.synapse.first_projection, &mut u.synapse.first_projection)?;
    for i in 0..n_down {
        copy_synapse_block(&t.synapse.down_blocks[i], &mut u.synapse.down_blocks[i])?;
    }
    for i in 0..n_up {
        copy_synapse_block(&t.synapse.up_blocks[i], &mut u.synapse.up_blocks[i])?;
    }
    let n_skip = t.synapse.skip_ln_gamma.len()
        .min(u.synapse.skip_ln_gamma.len())
        .min(t.synapse.skip_ln_beta.len())
        .min(u.synapse.skip_ln_beta.len());
    for i in 0..n_skip {
        u.synapse.skip_ln_gamma[i] = t.synapse.skip_ln_gamma[i].to_vec()?;
        u.synapse.skip_ln_beta[i] = t.synapse.skip_ln_beta[i].to_vec()?;
    }

    Ok(())
}

fn copy_synapse_block<D: Device>(
    t: &modgrad_ctm::synapse::SynapseBlockTyped<D>,
    u: &mut modgrad_ctm::synapse::SynapseBlock,
) -> Result<()> {
    u.linear.weight = t.linear.weight.to_vec()?;
    u.linear.bias = t.linear.bias.to_vec()?;
    u.ln_gamma = t.ln_gamma.to_vec()?;
    u.ln_beta = t.ln_beta.to_vec()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use modgrad_device::backend::tensor::Cpu;
    use modgrad_ctm::config::{CtmConfig, ExitStrategy};
    use modgrad_ctm::graph::{AuxLossConfig, Connection, RegionalConfig};

    fn small_cfg() -> RegionalConfig {
        let mk = |d: usize| CtmConfig {
            iterations: 1, d_model: d, d_input: 8, heads: 1,
            n_synch_out: 4, n_synch_action: 4, synapse_depth: 1,
            memory_length: 4, deep_nlms: false, memory_hidden_dims: 2,
            out_dims: 6, n_random_pairing_self: 0, min_width: 4,
            ..Default::default()
        };
        RegionalConfig {
            regions: vec![mk(8), mk(8)],
            region_names: vec!["a".into(), "b".into()],
            connections: vec![
                Connection { from: vec![0], to: 1, receives_observation: true, observation_scale: 0 },
            ],
            outer_ticks: 1,
            exit_strategy: ExitStrategy::None,
            n_global_sync: 4,
            out_dims: 6,
            raw_obs_dim: 8,
            obs_scale_dims: vec![8],
            aux_losses: AuxLossConfig::default(),
            router: None,
            cereb_mode: Default::default(),
        }
    }

    #[test]
    fn typed_to_untyped_round_trip_preserves_buffers() {
        let cfg = small_cfg();
        let untyped = RegionalWeights::new(cfg.clone());
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&untyped).unwrap();

        let extracted = typed_to_untyped(&typed).unwrap();

        // Embeddings round-trip exactly.
        assert_eq!(extracted.embeddings, untyped.embeddings);
        // obs_proj weight matches.
        assert_eq!(extracted.obs_proj.weight, untyped.obs_proj.weight);
        assert_eq!(extracted.output_proj.weight, untyped.output_proj.weight);
        // Per-region kv_proj matches.
        for r in 0..untyped.regions.len() {
            assert_eq!(extracted.regions[r].kv_proj.weight, untyped.regions[r].kv_proj.weight);
            assert_eq!(extracted.regions[r].nlm_stage1.weights, untyped.regions[r].nlm_stage1.weights);
        }
    }

    #[test]
    fn save_load_preserves_brain() {
        let cfg = small_cfg();
        let untyped = RegionalWeights::new(cfg.clone());
        let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&untyped).unwrap();

        let tmp = std::env::temp_dir().join("penumbra_arena_brain_test.bin");
        save(&typed, &tmp).unwrap();
        let loaded: RegionalWeightsTyped<Cpu> = load(&tmp).unwrap();

        // Embeddings should round-trip exactly.
        let orig = typed.embeddings.to_vec().unwrap();
        let back = loaded.embeddings.to_vec().unwrap();
        assert_eq!(orig.len(), back.len());
        for (a, b) in orig.iter().zip(&back) {
            assert!((a - b).abs() < 1e-9);
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
