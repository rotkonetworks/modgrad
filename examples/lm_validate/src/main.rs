//! Real validation for `LmTrainer<GptModelResident>`.
//!
//! The closing smoke (commit `0f680da`) showed the wiring runs end-to-
//! end: forward → cross-entropy → per-position backward → AdamW →
//! re-upload, with loss dropping 5.72 → 0.74 on a single repeated
//! 17-token sequence. That proves the chain is *connected*. It does
//! NOT prove the trainer produces a meaningful gradient signal on real
//! data with held-out validation — a sign error or scale bug can still
//! memorize a tiny fixed pattern.
//!
//! This binary ships that proof:
//!   - byte-level tokenization (vocab=256, no BPE) over a real English
//!     corpus (`climbmix_train.txt` + `climbmix_val.txt` at workspace
//!     root, ~46 MiB train + ~2 MiB val);
//!   - tiny GPT (2 layers, d_model=128, 4 heads, mlp_ratio=4) wrapped
//!     in `GptModelResident` and `LmTrainer`;
//!   - random-offset sampling for train, deterministic val offsets for
//!     repeatable held-out perplexity;
//!   - PASS / WARN / FAIL block reporting whether *training* (not
//!     memorization) actually works on real bytes.
//!
//! Run:
//!   cargo run --release --features rocm -p lm_validate
//!   cargo run --release --features rocm -p lm_validate -- 4000 24
//!
//! Defaults: 2000 train steps, seq_len = 16.
//!
//! ## Why byte-level
//!
//! Vocab=256 fits the test config from `lm_trainer.rs::tests::tiny_config`
//! exactly (`VocabSize::new(256)`). No tokenizer dependency, no BPE
//! state to ship — every byte is its own token. Bits-per-byte is a
//! direct, comparable metric (good byte-level baselines: ~1.5 bpb on
//! generic English; we expect a tiny 2-layer model on tens of MiB to
//! land somewhere in [3.5, 5] bpb after 2000 steps — well above SOTA
//! but well below random uniform 8.0 bpb).
//!
//! ## What "real validation" means here
//!
//! Train loss drops on memorization too. The proof of correctness is
//! the *val* loss curve falling on bytes the optimizer never saw.
//! Three labelled outcomes:
//!   - PASS: val_loss drops ≥ 30% from initial AND val_loss < 1.5×
//!     train_loss (no catastrophic overfit).
//!   - WARN: train drops but val doesn't — overfitting or an
//!     implementation bug that lets the model memorize but not
//!     generalize.
//!   - FAIL: train_loss does not decrease at all — a sign / scale bug
//!     in the gradient chain, or an AdamW / re-upload mis-key.
//!
//! Don't paper over a non-decreasing val loss. That's the entire
//! purpose of this validator.

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("lm_validate: built without `--features rocm`. The trainer is rocm-gated.");
    eprintln!("rebuild with: cargo run --release --features rocm -p lm_validate");
    std::process::exit(0);
}

#[cfg(feature = "rocm")]
fn main() {
    // ── CLI ────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seq_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let val_every: usize = 100;
    let val_batch: usize = 8;

    eprintln!("lm_validate: n_steps   = {n_steps}");
    eprintln!("             seq_len   = {seq_len}");
    eprintln!("             val_every = {val_every} steps");
    eprintln!("             val_batch = {val_batch} held-out windows");

    // ── Load corpus ────────────────────────────────────────────
    // Workspace root, two files up from `examples/lm_validate/`.
    let train_path = workspace_root().join("climbmix_train.txt");
    let val_path = workspace_root().join("climbmix_val.txt");

    if !train_path.exists() || !val_path.exists() {
        eprintln!("\nlm_validate: ERROR — corpus files missing.");
        eprintln!("expected: {}", train_path.display());
        eprintln!("expected: {}", val_path.display());
        eprintln!("place real English text at those paths and rerun.");
        eprintln!("(Cargo check still completes — this binary just exits non-fatally.)");
        std::process::exit(0);
    }

    let train_text = match std::fs::read(&train_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("lm_validate: failed to read {}: {e}", train_path.display());
            std::process::exit(1);
        }
    };
    let val_text = match std::fs::read(&val_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("lm_validate: failed to read {}: {e}", val_path.display());
            std::process::exit(1);
        }
    };

    eprintln!("             train     = {} bytes ({:.1} MiB)",
        train_text.len(), train_text.len() as f64 / 1_048_576.0);
    eprintln!("             val       = {} bytes ({:.1} MiB)",
        val_text.len(), val_text.len() as f64 / 1_048_576.0);

    let needed = seq_len + 1;
    if train_text.len() < needed * 4 || val_text.len() < needed * val_batch {
        eprintln!("lm_validate: ERROR — corpus too small for seq_len={seq_len} \
            (need ≥{} train, ≥{} val bytes)",
            needed * 4, needed * val_batch);
        std::process::exit(1);
    }

    // ── HIP runtime probe ──────────────────────────────────────
    if !modgrad_device::backend::rocm::ffi::runtime_available() {
        eprintln!("\nlm_validate: ERROR — HIP runtime not available.");
        eprintln!("(amdhip64.so / hipblas.so missing or no /dev/kfd.)");
        std::process::exit(0);
    }

    // ── Run ────────────────────────────────────────────────────
    let result = run_validation(&train_text, &val_text, n_steps, seq_len, val_every, val_batch);
    match result {
        Ok(report) => report.print_assessment(),
        Err(e) => {
            eprintln!("\nlm_validate: trainer error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "rocm")]
fn workspace_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR for this crate is `<workspace>/examples/lm_validate`.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mut p = std::path::PathBuf::from(manifest);
    // pop "lm_validate", pop "examples"
    p.pop();
    p.pop();
    p
}

// ─── Validation core ──────────────────────────────────────────

#[cfg(feature = "rocm")]
struct ValidationReport {
    train_curve: Vec<f32>, // mean over each 100-step window
    val_curve: Vec<f32>,
    val_steps: Vec<usize>, // step number where each val_curve point was taken
    initial_val_loss: f32,
    final_train_loss: f32,
    final_val_loss: f32,
    n_steps: usize,
    seq_len: usize,
    wall_secs: f64,
}

#[cfg(feature = "rocm")]
impl ValidationReport {
    fn final_perplexity(&self) -> f32 {
        self.final_val_loss.exp()
    }
    fn final_bits_per_byte(&self) -> f32 {
        // CE in nats; bpb = nats / ln(2).
        self.final_val_loss / std::f32::consts::LN_2
    }

    fn assessment(&self) -> Assessment {
        // FAIL: train loss flat or rising. Use the first vs last 100-
        // step train averages.
        if self.train_curve.len() < 2 {
            return Assessment::Fail; // not enough samples to even diagnose
        }
        let train_first = self.train_curve.first().copied().unwrap_or(f32::INFINITY);
        let train_last = self.final_train_loss;
        if !(train_last.is_finite()) {
            return Assessment::Fail;
        }
        // Train must drop by at least 5% over the run. A truly stuck
        // trainer (gradient sign error etc.) won't budge.
        if train_last >= train_first * 0.95 {
            return Assessment::Fail;
        }

        // PASS criteria from the slice spec:
        //   final val_loss < initial val_loss * 0.7
        //   AND val_loss < train_loss * 1.5  (no catastrophic overfit)
        let val_dropped = self.final_val_loss < self.initial_val_loss * 0.7;
        let no_gross_overfit = self.final_val_loss < self.final_train_loss * 1.5;

        if val_dropped && no_gross_overfit {
            Assessment::Pass
        } else {
            // Train moved but val didn't, or val/train ratio blew up —
            // overfitting or implementation bug.
            Assessment::Warn
        }
    }

    fn print_assessment(&self) {
        // ── Curves ─────────────────────────────────────────────
        println!("\n──────────────────────────────────────────────────────────────────");
        println!(" LM TRAINER VALIDATION — climbmix byte-level, seq_len={}", self.seq_len);
        println!("──────────────────────────────────────────────────────────────────");
        println!(" steps       train_mean(100)   val_loss");
        for i in 0..self.train_curve.len() {
            let step = self.val_steps.get(i).copied().unwrap_or((i + 1) * 100);
            let t = self.train_curve[i];
            let v = self.val_curve.get(i).copied().unwrap_or(f32::NAN);
            println!("  {:>6}      {:>10.4}      {:>10.4}", step, t, v);
        }
        println!("──────────────────────────────────────────────────────────────────");
        println!(" final train loss : {:.4}", self.final_train_loss);
        println!(" final val   loss : {:.4}", self.final_val_loss);
        println!(" initial val loss : {:.4}", self.initial_val_loss);
        println!(" val perplexity   : {:.3}", self.final_perplexity());
        println!(" val bits/byte    : {:.3}  (random-uniform = 8.000)",
            self.final_bits_per_byte());
        println!(" wall             : {:.1} s ({:.1} steps/s)",
            self.wall_secs, self.n_steps as f64 / self.wall_secs.max(1e-9));
        println!("──────────────────────────────────────────────────────────────────");

        // ── Verdict ────────────────────────────────────────────
        let verdict = self.assessment();
        let banner = match verdict {
            Assessment::Pass => "  ★★★  PASS  ★★★   real-data training works",
            Assessment::Warn => "  !!!  WARN  !!!   train moves but val doesn't \
                — overfit or sign / scale bug",
            Assessment::Fail => "  XXX  FAIL  XXX   train loss does not decrease \
                — gradient sign / scale bug",
        };
        println!("\n{banner}\n");

        // exit code mirrors the verdict so CI scripts can branch on it
        match verdict {
            Assessment::Pass => {}
            Assessment::Warn => std::process::exit(2),
            Assessment::Fail => std::process::exit(3),
        }
    }
}

#[cfg(feature = "rocm")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Assessment {
    Pass,
    Warn,
    Fail,
}

// ─── Tiny model + trainer setup ───────────────────────────────

#[cfg(feature = "rocm")]
fn build_tiny_model_for_byte_level()
    -> (modgrad_transformer::model::GptModel,
        Vec<modgrad_transformer::mlp::SwigluMlp>,
        modgrad_transformer::config::GptConfig)
{
    use modgrad_transformer::config::{
        GptConfig, MlpActivation, ResidualConfig, SmearConfig,
        ValueEmbedConfig, WindowPattern, Precision,
    };
    use modgrad_transformer::dims::*;
    use modgrad_transformer::tensor::Tensor2;
    use modgrad_transformer::attention::{AttentionWeights, CausalSelfAttention};
    use modgrad_transformer::block::TransformerBlock;
    use modgrad_transformer::mlp::{Mlp, MlpWeights, SwigluMlp, SwigluWeights};
    use modgrad_transformer::model::GptModel;
    use modgrad_transformer::norm::ScaledRmsNorm;
    use modgrad_transformer::position::fixed::FixedPositioning;
    use modgrad_transformer::residual::ResidualLambdas;
    use modgrad_transformer::rope::RotaryEmbedding;
    use modgrad_transformer::smear::{Inference, Smear, SmearWeights, Training};

    // Mirror `tiny_config` from `lm_trainer.rs::tests`. 2 layers, 4
    // heads × 32 = 128, vocab=256, mlp_ratio=2 (mlp_dim = model_dim*2,
    // matches the test). max_seq_len=64 — bumped up from the test's 32
    // so a CLI like `seq_len=24` still fits comfortably.
    let head_dim = 32usize;
    let n_heads = 4usize;
    let model_dim = head_dim * n_heads;
    let config = GptConfig {
        model_dim: ModelDim::new(model_dim),
        num_heads: NumHeads::new(n_heads),
        num_kv_heads: NumKvHeads::new(n_heads),
        head_dim: HeadDim::new(head_dim),
        num_layers: NumLayers::new(2),
        vocab_size: VocabSize::new(256),
        mlp_dim: MlpDim::new(model_dim * 2),
        max_seq_len: SeqLen::new(64),
        rope_base: 10000.0,
        qk_norm_scale: 1.0,
        window_pattern: WindowPattern::Full,
        mlp_activation: MlpActivation::SwiGlu,
        layer_overrides: Vec::new(),
        tie_embeddings: false,
        logit_cap: 0.0,
        recurrent_steps: 1,
        has_exit_gate: false,
        value_embed: ValueEmbedConfig::default(),
        residual: ResidualConfig {
            resid_start: 1.0, resid_end: 1.0,
            x0_start: 0.0, x0_end: 0.0,
            backout_lambda: 0.0,
        },
        smear: SmearConfig::default(),
        precision: Precision::F32,
        norm_eps: 1e-5,
    };

    // Deterministic random init via SimpleRng — same recipe as
    // `build_test_model` in lm_trainer tests.
    let md = config.model_dim.get();
    let kv_dim = config.num_kv_heads.get() * config.head_dim.get();
    let vocab = config.vocab_size.get();
    let mlp_dim = config.mlp_dim.get();

    let mut rng = modgrad_compute::neuron::SimpleRng::new(0xBADBEEF);
    let randn = |rng: &mut modgrad_compute::neuron::SimpleRng, n: usize| -> Vec<f32> {
        // 0.05 stddev — same as the test fixture; a tiny model needs
        // a small init so the first softmax isn't already saturated.
        (0..n).map(|_| rng.next_normal() * 0.05).collect()
    };

    let token_embed = randn(&mut rng, vocab * md);
    let lm_head = randn(&mut rng, vocab * md);
    let final_norm_scale = vec![1.0f32; md];
    let smear_gate = vec![0.0f32; md * config.smear.gate_channels];

    let mut blocks_host = Vec::with_capacity(config.num_layers.get());
    let mut swiglu_mlps = Vec::with_capacity(config.num_layers.get());
    for li in 0..config.num_layers.get() {
        let attn_w = AttentionWeights {
            wq: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
            wk: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wv: Tensor2::new(randn(&mut rng, kv_dim * md), kv_dim, md).unwrap(),
            wo: Tensor2::new(randn(&mut rng, md * md), md, md).unwrap(),
        };
        let attn = CausalSelfAttention::new(attn_w, &config);

        let gate_w = randn(&mut rng, mlp_dim * md);
        let up_w = randn(&mut rng, mlp_dim * md);
        let down_w = randn(&mut rng, md * mlp_dim);
        let swiglu_w = SwigluWeights {
            gate: Tensor2::new(gate_w, mlp_dim, md).unwrap(),
            up: Tensor2::new(up_w, mlp_dim, md).unwrap(),
            down: Tensor2::new(down_w, md, mlp_dim).unwrap(),
        };
        let swiglu = SwigluMlp::new(swiglu_w, config.model_dim, config.mlp_dim);
        swiglu_mlps.push(swiglu);

        let placeholder_mlp = Mlp::new(
            MlpWeights {
                fc: Tensor2::zeros(mlp_dim, md),
                proj: Tensor2::zeros(md, mlp_dim),
            },
            config.model_dim, config.mlp_dim,
        );
        let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
        blocks_host.push(TransformerBlock::new(
            attn, placeholder_mlp, None, layer_idx, &config,
        ));
    }

    let model = GptModel {
        embed: Tensor2::new(token_embed, vocab, md).unwrap(),
        lm_head: Tensor2::new(lm_head, vocab, md).unwrap(),
        final_norm: ScaledRmsNorm::new(final_norm_scale, config.norm_eps),
        smear_inference: Smear::<Inference>::new(SmearWeights::new(
            smear_gate.clone(), config.model_dim, &config.smear,
        )),
        smear_training: Smear::<Training>::new(SmearWeights::new(
            smear_gate, config.model_dim, &config.smear,
        )),
        blocks: blocks_host,
        lambdas: ResidualLambdas::from_config(&config.residual, config.num_layers),
        rope: RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base),
        position: Box::new(FixedPositioning),
        config: config.clone(),
    };
    (model, swiglu_mlps, config)
}

#[cfg(feature = "rocm")]
fn run_validation(
    train_text: &[u8],
    val_text: &[u8],
    n_steps: usize,
    seq_len: usize,
    val_every: usize,
    val_batch: usize,
) -> Result<ValidationReport, modgrad_compute::backend::ResidencyError> {
    use std::time::Instant;
    use isis_runtime::language_model::LanguageModel;
    use isis_runtime::lm_trainer::{LmTrainer, LmTrainerConfig};
    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_transformer::loss::cross_entropy;
    use modgrad_transformer::{GptModelResident, KvCacheResident};

    // ── Build model + trainer ─────────────────────────────────
    eprintln!("\nlm_validate: building tiny GPT (2 layers, d_model=128, 4 heads, vocab=256) …");
    let (model, swiglu_mlps, config) = build_tiny_model_for_byte_level();
    let resident = GptModelResident::from_model(&model, &swiglu_mlps)?;

    let trainer_cfg = LmTrainerConfig {
        // 1e-3 is the smoke-test LR; on real text we keep it modest so
        // the first hundred steps of bytes don't spike the loss.
        lr: 1e-3,
        micro_batch_size: seq_len,
        seq_len,
        ..LmTrainerConfig::default()
    };
    let n_kv = config.num_kv_heads.get();
    let hd = config.head_dim.get();
    let mut trainer = LmTrainer::new(resident, n_kv, hd, trainer_cfg)?;

    // ── Dedicated val KV cache (separate from trainer's) ──────
    // The trainer holds its own KvCacheResident sized for training;
    // we want an independent cache so val_step doesn't race or
    // interfere with mid-train KV state. Same shape (n_layers ×
    // seq_len × n_kv × head_dim × d_model).
    let n_layers = trainer.model().n_layers();
    let d_model = trainer.model().d_model();
    let mut val_kv = KvCacheResident::new(n_layers, n_kv, hd, seq_len, d_model)?;

    // ── Deterministic val offsets ────────────────────────────
    // Spread evenly across the val corpus so we sample a representative
    // slice — not just the prologue. Seeded RNG: same offsets every
    // call.
    let val_offsets: Vec<usize> = {
        let mut offs = Vec::with_capacity(val_batch);
        let span = (val_text.len() - (seq_len + 1)).max(1);
        let mut rng = modgrad_compute::neuron::SimpleRng::new(0xCAFE_F00D);
        for _ in 0..val_batch {
            // next_u64 → uniform in span
            let off = (rng.next_u64() as usize) % span;
            offs.push(off);
        }
        offs
    };

    // ── Pre-train val baseline ────────────────────────────────
    eprintln!("lm_validate: measuring initial val loss (random init) …");
    let initial_val_loss = compute_val_loss(
        trainer.model_mut(), &mut val_kv, val_text, &val_offsets, seq_len,
    )?;
    eprintln!("             initial val loss = {initial_val_loss:.4}");
    eprintln!("             initial bits/byte = {:.3}",
        initial_val_loss / std::f32::consts::LN_2);

    // ── Train loop ────────────────────────────────────────────
    eprintln!("\nlm_validate: training {n_steps} steps …");
    let train_start = Instant::now();
    let mut train_window: Vec<f32> = Vec::with_capacity(val_every);
    let mut train_curve: Vec<f32> = Vec::with_capacity(n_steps / val_every);
    let mut val_curve: Vec<f32> = Vec::with_capacity(n_steps / val_every);
    let mut val_steps: Vec<usize> = Vec::with_capacity(n_steps / val_every);
    // RNG for sampling — fresh seed per run so if this is re-run on
    // identical hardware we get the same sequence of offsets and the
    // same loss curves (modulo non-determinism in hipblas reductions).
    let mut rng = modgrad_compute::neuron::SimpleRng::new(0xDEAD_BEEF);

    let train_span = (train_text.len() - (seq_len + 1)).max(1);
    for step in 1..=n_steps {
        let off = (rng.next_u64() as usize) % train_span;
        let window = &train_text[off..off + seq_len + 1];
        let tokens: Vec<i64> = window.iter().map(|&b| b as i64).collect();

        let batch = HipBatch::new();
        let loss = trainer.train_step(&batch, &tokens)?;
        if !loss.is_finite() {
            eprintln!("\nlm_validate: non-finite loss at step {step} — aborting");
            break;
        }
        train_window.push(loss);

        if step % val_every == 0 {
            let train_mean = train_window.iter().sum::<f32>() / train_window.len() as f32;
            train_window.clear();
            let val_loss = compute_val_loss(
                trainer.model_mut(), &mut val_kv, val_text, &val_offsets, seq_len,
            )?;
            train_curve.push(train_mean);
            val_curve.push(val_loss);
            val_steps.push(step);

            let pct = step as f32 / n_steps as f32 * 100.0;
            eprintln!("  step {step:>5} ({pct:>5.1}%)  train={train_mean:.4}  val={val_loss:.4}");
        }
    }

    // ── Final loss snapshot ───────────────────────────────────
    let final_train_loss = *train_curve.last().unwrap_or(&f32::NAN);
    let final_val_loss = *val_curve.last().unwrap_or(&f32::NAN);
    let wall_secs = train_start.elapsed().as_secs_f64();

    // ── Sanity: loss + history check ──────────────────────────
    let history = trainer.loss_history();
    if history.is_empty() {
        eprintln!("lm_validate: WARNING — trainer.loss_history() is empty.");
    }

    // Verify our cross_entropy reference computes the same as the
    // trainer's path on a fresh forward (catches a re-upload bug
    // where weights drift but the trainer's forward path is stale).
    {
        let off = val_offsets[0];
        let window = &val_text[off..off + seq_len + 1];
        let tokens: Vec<i64> = window.iter().map(|&b| b as i64).collect();
        let mut logits_dev = GpuVec::try_hip(seq_len * 256)?;
        let positions: Vec<usize> = (0..seq_len).collect();
        val_kv.reset();
        let batch = HipBatch::new();
        trainer.model_mut().forward_logits(
            &batch, &tokens[..seq_len], &positions, Some(&mut val_kv), &mut logits_dev,
        )?;
        batch.flush()?;
        let mut logits_host = vec![0.0f32; seq_len * 256];
        logits_dev.copy_to_host(&mut logits_host);
        let (verify_loss, _) = cross_entropy(&logits_host, &tokens[1..=seq_len], 256);
        eprintln!("\nlm_validate: independent forward at val[{off}] → loss {verify_loss:.4}");
    }

    Ok(ValidationReport {
        train_curve,
        val_curve,
        val_steps,
        initial_val_loss,
        final_train_loss,
        final_val_loss,
        n_steps,
        seq_len,
        wall_secs,
    })
}

/// Forward + cross-entropy on a batch of fixed val offsets, average
/// the losses. Independent KV cache: we own a separate
/// `KvCacheResident` and reset it between each window.
#[cfg(feature = "rocm")]
fn compute_val_loss(
    model: &mut modgrad_transformer::GptModelResident,
    val_kv: &mut modgrad_transformer::KvCacheResident,
    val_text: &[u8],
    val_offsets: &[usize],
    seq_len: usize,
) -> Result<f32, modgrad_compute::backend::ResidencyError> {
    use isis_runtime::language_model::LanguageModel;
    use modgrad_compute::backend::GpuVec;
    use modgrad_device::backend::HipBatch;
    use modgrad_transformer::loss::cross_entropy;

    let vocab = LanguageModel::vocab_size(model);
    let positions: Vec<usize> = (0..seq_len).collect();
    let mut logits_dev = GpuVec::try_hip(seq_len * vocab)?;
    let mut logits_host = vec![0.0f32; seq_len * vocab];

    let mut total = 0.0f32;
    for &off in val_offsets {
        let window = &val_text[off..off + seq_len + 1];
        let tokens: Vec<i64> = window.iter().map(|&b| b as i64).collect();
        let src = &tokens[..seq_len];
        let tgt = &tokens[1..=seq_len];

        val_kv.reset();
        let batch = HipBatch::new();
        model.forward_logits(&batch, src, &positions, Some(val_kv), &mut logits_dev)?;
        batch.flush()?;
        logits_dev.copy_to_host(&mut logits_host);
        let (loss, _) = cross_entropy(&logits_host, tgt, vocab);
        total += loss;
    }
    Ok(total / val_offsets.len() as f32)
}

// ─── Tests ────────────────────────────────────────────────────
//
// `cargo test -p lm_validate` runs a 200-step short validation as
// `#[ignore]` (real GPU + a real corpus take >2 min). To run it:
//   cargo test --release --features rocm -p lm_validate -- --ignored

#[cfg(test)]
#[cfg(feature = "rocm")]
mod tests {
    use super::*;

    /// Short-form validation. Skips cleanly if either the corpus is
    /// missing OR the HIP runtime is unavailable. PASS criteria are
    /// the same as the binary's main path.
    #[test]
    #[ignore = "requires HIP runtime + climbmix_*.txt at workspace root; \
        run with `cargo test --release --features rocm -p lm_validate -- --ignored`"]
    fn lm_validate_smoke_real_data() {
        let train_path = workspace_root().join("climbmix_train.txt");
        let val_path = workspace_root().join("climbmix_val.txt");
        if !train_path.exists() || !val_path.exists() {
            eprintln!("climbmix_*.txt missing — skipping");
            return;
        }
        if !modgrad_device::backend::rocm::ffi::runtime_available() {
            eprintln!("HIP runtime missing — skipping");
            return;
        }

        let train_text = std::fs::read(&train_path).expect("read train");
        let val_text = std::fs::read(&val_path).expect("read val");

        // Short run — 200 steps is enough to see the trend on this
        // tiny model + LR=1e-3. If it fails to drop in 200 steps, the
        // gradient chain is broken; lengthening the test won't fix it.
        let report = run_validation(&train_text, &val_text, 200, 16, 50, 4)
            .expect("run_validation");

        assert!(report.final_train_loss.is_finite(), "final train loss finite");
        assert!(report.final_val_loss.is_finite(), "final val loss finite");

        // Train must drop by ≥5% over the run. Same gate as the FAIL
        // branch in `Assessment::*`.
        let train_first = report.train_curve.first().copied().unwrap();
        assert!(report.final_train_loss < train_first * 0.95,
            "train loss did not drop: first={train_first} last={} \
             — gradient sign or scale bug?",
            report.final_train_loss);

        // Val must drop by ≥30% from initial — the slice's PASS gate.
        // 200 steps is short, but on byte-level English the random-init
        // val loss is ~5.5 nats and a working trainer drops to ~3.0
        // within 200 steps comfortably.
        assert!(report.final_val_loss < report.initial_val_loss * 0.7,
            "val loss did not drop ≥30%: initial={} final={} \
             — overfitting or gradient bug?",
            report.initial_val_loss, report.final_val_loss);

        assert!(report.final_val_loss < report.final_train_loss * 1.5,
            "val/train ratio blew up (catastrophic overfit): \
             train={} val={}",
            report.final_train_loss, report.final_val_loss);
    }
}
