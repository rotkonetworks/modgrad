//! Real-Qwen integration: measure baseline Qwen NLL vs Qwen+brain NLL
//! on a token sequence using `BrainLogitModulator`. This is the
//! redshiftzero comparison primitive on real weights — the synthetic-
//! data version was validated in `crates/modgrad-ctm/src/logit_modulator.rs`.
//!
//! Pipeline:
//!   1. Load Qwen2.5-0.5B (safetensors) → `GptModelResident`
//!   2. Load tokenizer; encode a sample sequence
//!   3. Forward Qwen over the sequence → per-position logits
//!   4. Compute baseline NLL on the next-token targets
//!   5. Forward brain (`RegionalBrain`/`NeuralComputer`) token-by-token
//!      → per-step brain output vector
//!   6. Apply `BrainLogitModulator` with three settings:
//!         a) alpha=0 (sanity: must equal baseline bit-exactly)
//!         b) alpha=1 random projection (must differ — content-causal)
//!         c) (deferred to next slice) alpha=1 trained projection
//!
//! Run on a host with Qwen weights:
//!   cargo run --release --features rocm -p brain_qwen_nll -- \
//!       --text "Once upon a time"
//!
//! Without HIP / weights this binary cleanly exits with a helpful
//! message — same pattern as `qwen_chat`.

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("brain_qwen_nll: requires --features rocm; rebuild with:");
    eprintln!("  cargo run --release --features rocm -p brain_qwen_nll -- --text \"...\"");
}

#[cfg(feature = "rocm")]
fn main() {
    use std::time::Instant;

    use clap::Parser;
    use modgrad_device::backend::HipBatch;
    use modgrad_device::backend::rocm::ffi::runtime_available;
    use modgrad_compute::backend::GpuVec;
    use modgrad_io::qwen2::{
        load_qwen2_5_0_5b, QWEN2_5_0_5B_NUM_KV_HEADS, QWEN2_5_0_5B_HEAD_DIM,
    };
    use modgrad_io::tokenizer::HfTokenizer;
    use modgrad_transformer::kv_cache_resident::KvCacheResident;

    use modgrad_ctm::graph::{NeuralComputer, RegionalConfig, RegionalWeights};
    use modgrad_ctm::logit_modulator::{
        BrainLogitModulator, LowRankBrainLogitModulator,
        LogitModulator, nll_per_token, cross_entropy_grad,
    };

    /// For each test position, compute baseline (Qwen-alone) NLL and
    /// trained (Qwen+brain) NLL on the actual target token. Print
    /// the positions sorted by improvement (Δ = baseline − trained,
    /// positive = brain helped). Also reports the per-position token
    /// strings via the tokenizer.
    fn print_per_position_breakdown<M: LogitModulator>(
        modulator: &M,
        test_qwen: &[&[f32]],
        test_brain: &[&[f32]],
        test_tgts: &[usize],
        token_ids: &[i64],
        n_train: usize,
        tokenizer: &HfTokenizer,
    ) {
        let vocab = modulator.vocab();
        let n = test_qwen.len();
        let mut deltas: Vec<(usize, f32, f32, f32, String, String)> = Vec::with_capacity(n);
        let mut scratch = vec![0.0f32; vocab];

        let log_softmax_at = |logits: &[f32], target: usize| -> f32 {
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let lse: f32 = logits.iter()
                .map(|&l| (l - max_l).exp()).sum::<f32>().ln() + max_l;
            lse - logits[target]
        };

        for t in 0..n {
            let qwen_t = test_qwen[t];
            let target = test_tgts[t];
            let baseline_nll = log_softmax_at(qwen_t, target);
            modulator.modulate_into(qwen_t, test_brain[t], &mut scratch);
            let trained_nll = log_softmax_at(&scratch, target);
            // Recover the input token at this prediction position (the
            // token whose successor we're predicting) for readability.
            let abs_pos = n_train + t;
            let input_tok = tokenizer.decode(&[token_ids[abs_pos] as u32])
                .unwrap_or_else(|_| "??".into());
            let target_tok = tokenizer.decode(&[token_ids[abs_pos + 1] as u32])
                .unwrap_or_else(|_| "??".into());
            deltas.push((t, baseline_nll, trained_nll, baseline_nll - trained_nll,
                         input_tok, target_tok));
        }

        // Sort by improvement (descending).
        deltas.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!();
        eprintln!("---- per-position breakdown (test set) ----");
        eprintln!("  pos | baseline NLL | trained NLL |    Δ     | context → target");
        eprintln!("  ----+--------------+-------------+----------+---------------------------");
        for (t, base, trained, delta, in_tok, tgt_tok) in &deltas {
            let in_tok_clean: String = in_tok.chars()
                .map(|c| if c.is_control() { '·' } else { c }).collect();
            let tgt_tok_clean: String = tgt_tok.chars()
                .map(|c| if c.is_control() { '·' } else { c }).collect();
            eprintln!("  {:>3} |   {:>8.4}   |  {:>8.4}   | {:>+7.4} | {:>15} → {}",
                t, base, trained, delta,
                format!("{:?}", in_tok_clean), format!("{:?}", tgt_tok_clean));
        }
        let n_helped = deltas.iter().filter(|d| d.3 > 0.0).count();
        let n_hurt = deltas.iter().filter(|d| d.3 < 0.0).count();
        eprintln!("  ────────");
        eprintln!("  {n_helped} positions helped, {n_hurt} hurt (out of {n})");
    }

    #[derive(Parser, Debug)]
    #[command(name = "brain_qwen_nll")]
    struct Args {
        #[arg(long, default_value = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors")]
        model: String,

        #[arg(long, default_value = "/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json")]
        tokenizer: String,

        /// Text whose per-token NLL we measure. Tokenized once, then
        /// fed to both Qwen and the brain. Should be at least a few
        /// tokens long for the NLL average to be meaningful. Ignored
        /// when --text-file is set.
        #[arg(long, default_value =
            "The quick brown fox jumps over the lazy dog. \
             The early bird catches the worm. \
             A stitch in time saves nine. \
             Every cloud has a silver lining. \
             Actions speak louder than words.")]
        text: String,

        /// Path to a text file. When set, replaces --text. Use this
        /// to run on a longer corpus where modulator capacity isn't
        /// the bottleneck (with 37-token --text default, even rank=2
        /// has 18K params/example — overfits past epoch ~14).
        #[arg(long)]
        text_file: Option<String>,

        #[arg(long, default_value_t = 256)]
        max_seq: usize,

        /// Fraction of tokens reserved for the held-out test split.
        /// 0.3 means train on first 70% of prediction positions, hold
        /// out the last 30% for evaluation. Set to 0.0 to skip the
        /// training phase entirely (just baseline + sanity).
        #[arg(long, default_value_t = 0.3)]
        test_fraction: f32,

        /// Number of full-sequence gradient updates over the train
        /// split. Plain SGD on the modulator only; brain + Qwen
        /// frozen.
        #[arg(long, default_value_t = 200)]
        epochs: usize,

        /// Learning rate for the modulator's SGD steps.
        #[arg(long, default_value_t = 0.05)]
        lr: f32,

        /// LoRA rank for the modulator. The dense modulator at
        /// brain_dim=512, vocab=152K is 78M params — guaranteed to
        /// overfit a small text. Default rank=8 drops to ~1.4M params
        /// (~57× reduction). Set to 0 to use the dense modulator.
        #[arg(long, default_value_t = 8)]
        rank: usize,

        /// Per-position NLL breakdown. After training, prints the test
        /// positions sorted by NLL improvement (Qwen-alone NLL minus
        /// Qwen+brain NLL). Lets us see WHICH positions benefit from
        /// brain attachment instead of just averaging across all of
        /// them — separates generic content-causal modulation
        /// (uniform improvement) from specific semantic recall
        /// (clustered improvement at recall positions).
        #[arg(long)]
        per_position: bool,

        /// Recall-shaped loss: only count training gradient at
        /// positions where the target token has appeared earlier in
        /// the sequence. Forces the optimizer to find corrections
        /// that benefit recall-specific predictions, not generic
        /// content-causal patterns. Without this, the loss averages
        /// over all positions and the modulator drifts toward
        /// fitting common patterns (sentence-start names,
        /// verb-prep transitions) rather than recall.
        #[arg(long)]
        recall_only_loss: bool,

        /// What signal feeds into the modulator. Used to ablate
        /// whether brain dynamics are doing useful work, or whether
        /// the modulator's improvement could come from any
        /// content-causal feature extractor:
        ///   "real"   — RegionalBrain via NeuralComputer.step (default)
        ///   "random" — fresh random Gaussian per token (no
        ///              cross-token state — pure noise)
        ///   "embed"  — deterministic per-token-id vector (sin-based
        ///              hash — content-causal but no temporal
        ///              dependency or memory)
        ///   "zero"   — vec![0.0; brain_dim] (modulator learns a
        ///              constant bias only)
        #[arg(long, default_value = "real")]
        brain_mode: String,
    }

    let args = Args::parse();

    if !runtime_available() {
        eprintln!("brain_qwen_nll: HIP runtime unavailable — exiting cleanly.");
        return;
    }

    // ── 1. Load tokenizer + model ──
    eprintln!("brain_qwen_nll: loading tokenizer from {}", args.tokenizer);
    let tokenizer = match HfTokenizer::from_file(&args.tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain_qwen_nll: tokenizer load failed: {e}");
            return;
        }
    };

    eprintln!("brain_qwen_nll: loading model from {}", args.model);
    let load_start = Instant::now();
    let mut model = match load_qwen2_5_0_5b(&args.model, args.max_seq) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("brain_qwen_nll: model load failed: {e}");
            eprintln!("  hint: download Qwen2.5-0.5B safetensors first.");
            return;
        }
    };
    eprintln!("brain_qwen_nll: model loaded in {:.2}s",
        load_start.elapsed().as_secs_f32());

    let vocab = model.vocab_size();
    let model_dim = model.model_dim();
    eprintln!("brain_qwen_nll: vocab={vocab}, model_dim={model_dim}");

    // ── 2. Encode the text. The first token is "context"; logits at
    // position t predict token t+1, so we need ≥ 2 tokens. ──
    let text: String = match args.text_file.as_ref() {
        Some(p) => match std::fs::read_to_string(p) {
            Ok(s) => {
                eprintln!("brain_qwen_nll: loaded {} bytes from {p}", s.len());
                s
            }
            Err(e) => {
                eprintln!("brain_qwen_nll: read --text-file {p} failed: {e}");
                return;
            }
        },
        None => args.text.clone(),
    };
    let token_ids: Vec<i64> = match tokenizer.encode(&text) {
        Ok(ids) => ids.into_iter().map(|id| id as i64).collect(),
        Err(e) => {
            eprintln!("brain_qwen_nll: encode failed: {e}");
            return;
        }
    };
    let n = token_ids.len();
    if n < 2 {
        eprintln!("brain_qwen_nll: need ≥2 tokens, got {n} — extend --text");
        return;
    }
    eprintln!("brain_qwen_nll: text {} tokens, max_seq={}",
        n, args.max_seq);

    let positions: Vec<usize> = (0..n).collect();

    // ── 3. Run Qwen forward; harvest logits[t] for all t. ──
    let mut kv_cache = KvCacheResident::new(
        model.num_layers(),
        QWEN2_5_0_5B_NUM_KV_HEADS,
        QWEN2_5_0_5B_HEAD_DIM,
        args.max_seq,
        model.model_dim(),
    ).expect("alloc kv cache");

    let batch = HipBatch::new();
    let mut qwen_logits_dev = GpuVec::try_hip(n * vocab).expect("alloc logits");

    let qwen_t = Instant::now();
    model.forward(&batch, &token_ids, &positions, &mut kv_cache, &mut qwen_logits_dev)
        .expect("qwen forward");
    batch.flush().expect("flush");
    let qwen_ms = qwen_t.elapsed().as_millis();
    eprintln!("brain_qwen_nll: qwen forward {n} tokens in {qwen_ms} ms");

    // Copy logits to host. Predictions: logits[t] predicts token[t+1],
    // so we use logits[0..n-1] against targets token_ids[1..n].
    let mut qwen_logits_host = vec![0.0f32; n * vocab];
    qwen_logits_dev.copy_to_host(&mut qwen_logits_host);

    // Per-position logit slices (logits at position t).
    let qwen_per_pos: Vec<&[f32]> = (0..n - 1)
        .map(|t| &qwen_logits_host[t * vocab..(t + 1) * vocab])
        .collect();
    let targets: Vec<usize> = token_ids[1..].iter().map(|&id| id as usize).collect();

    // ── 4. Baseline Qwen-only NLL. Modulator at alpha=0 → bypass.
    // brain_dim matches the brain output we'll later compute (defined
    // below as BRAIN_OUT_DIM). For the baseline path with alpha=0 the
    // brain output is unused, but the modulator instance still needs
    // shape parameters set correctly. ──
    const BRAIN_DIM_FOR_MODULATOR: usize = 512;
    let m_baseline = BrainLogitModulator::new(BRAIN_DIM_FOR_MODULATOR, vocab);
    assert_eq!(m_baseline.alpha, 0.0);
    let baseline_nll = nll_per_token(&m_baseline, &qwen_per_pos, None, &targets);
    eprintln!("brain_qwen_nll: BASELINE Qwen-only NLL = {baseline_nll:.4}");

    // ── 5. Compute brain outputs per token. The --brain-mode flag
    // selects between the real brain and ablation baselines that test
    // whether brain dynamics are doing useful work. ──
    const BRAIN_OUT_DIM: usize = 512;
    eprintln!("brain_qwen_nll: brain mode = {}, out_dim={BRAIN_OUT_DIM}",
              args.brain_mode);
    let brain_t = Instant::now();
    let brain_outputs: Vec<Vec<f32>> = match args.brain_mode.as_str() {
        "real" => {
            let mut cfg = RegionalConfig::eight_region(16, BRAIN_OUT_DIM, 2);
            cfg.router = None;
            let w = RegionalWeights::new(cfg);
            let mut nc = NeuralComputer::new(w);
            token_ids.iter().map(|&id| nc.step((id as usize) % 256)).collect()
        }
        "random" => {
            // Fresh random Gaussian per token — no cross-token state.
            // PCG-style PRNG keyed off token position so the run is
            // deterministic and reproducible.
            (0..n).map(|t| {
                let mut s = 0xdeadbeef_u64
                    .wrapping_add(t as u64)
                    .wrapping_mul(6364136223846793005);
                (0..BRAIN_OUT_DIM).map(|_| {
                    s = s.wrapping_mul(6364136223846793005)
                          .wrapping_add(1442695040888963407);
                    let u1 = ((s >> 40) as f32 / (1u64 << 24) as f32).max(1e-10);
                    s = s.wrapping_mul(6364136223846793005)
                          .wrapping_add(1442695040888963407);
                    let u2 = (s >> 40) as f32 / (1u64 << 24) as f32;
                    (-2.0 * u1.ln()).sqrt()
                        * (2.0 * std::f32::consts::PI * u2).cos()
                }).collect()
            }).collect()
        }
        "embed" => {
            // Deterministic per-token-id sinusoidal vector. Content-
            // causal (same token → same output) but no temporal
            // dependency, no memory.
            token_ids.iter().map(|&id| {
                let id = id as usize;
                (0..BRAIN_OUT_DIM).map(|i| {
                    ((id as f32 * 0.13 + i as f32 * 0.21).sin())
                }).collect()
            }).collect()
        }
        "zero" => {
            (0..n).map(|_| vec![0.0f32; BRAIN_OUT_DIM]).collect()
        }
        other => {
            eprintln!("brain_qwen_nll: unknown --brain-mode {other:?} \
                       (expected: real / random / embed / zero)");
            return;
        }
    };
    let brain_ms = brain_t.elapsed().as_millis();
    eprintln!("brain_qwen_nll: brain forward {n} ticks in {brain_ms} ms");
    debug_assert_eq!(brain_outputs[0].len(), BRAIN_OUT_DIM);

    // Per-position brain output for the prediction targets (positions
    // 0..n-1 are the predicting positions, same as qwen_per_pos).
    let brain_per_pos: Vec<&[f32]> = brain_outputs[..n - 1].iter()
        .map(|v| v.as_slice()).collect();

    // ── 6a. Sanity: alpha=0 with brain attached → equals baseline. ──
    let alpha_zero_nll = nll_per_token(
        &m_baseline, &qwen_per_pos, Some(&brain_per_pos), &targets);
    let drift = (alpha_zero_nll - baseline_nll).abs();
    if drift > 1e-5 {
        eprintln!("brain_qwen_nll: WARN alpha=0 NLL drift = {drift:.6e} \
                   (expected 0; numerical inconsistency in fast-path)");
    }
    eprintln!("brain_qwen_nll: alpha=0      NLL = {alpha_zero_nll:.4}  \
               (Δ vs baseline = {drift:.2e}, must be 0)");

    // ── 6b. alpha=1 random projection: content-causal but uninformed. ──
    let mut m_random = BrainLogitModulator::new(BRAIN_DIM_FOR_MODULATOR, vocab);
    m_random.alpha = 1.0;
    // Tame the random init: the projection is large so raw alpha=1
    // would dominate Qwen's logit magnitudes by orders of magnitude.
    // Scale weights to roughly match Qwen logit scale.
    for w in m_random.proj.weight.iter_mut() { *w *= 0.01; }
    let alpha_one_nll = nll_per_token(
        &m_random, &qwen_per_pos, Some(&brain_per_pos), &targets);
    eprintln!("brain_qwen_nll: alpha=1 RAND NLL = {alpha_one_nll:.4}  \
               (Δ vs baseline = {:+.4})", alpha_one_nll - baseline_nll);

    // ── 7. Train/test split + training loop. ──
    if args.test_fraction <= 0.0 || args.epochs == 0 {
        eprintln!();
        eprintln!("---- redshiftzero summary (no-train mode) ----");
        eprintln!("  baseline (Qwen alone)          : {baseline_nll:.4}");
        eprintln!("  alpha=0   (sanity)             : {alpha_zero_nll:.4}");
        eprintln!("  alpha=1   (random brain proj)  : {alpha_one_nll:.4}");
        return;
    }

    let n_pred = n - 1;
    let n_test = ((n_pred as f32) * args.test_fraction).round() as usize;
    let n_train = n_pred.saturating_sub(n_test);
    if n_train < 2 || n_test < 1 {
        eprintln!("brain_qwen_nll: not enough tokens for {:.0}% test split \
                   (n_pred={n_pred}, n_train={n_train}, n_test={n_test}); \
                   extend --text or lower --test-fraction",
                  args.test_fraction * 100.0);
        return;
    }
    eprintln!();
    eprintln!("brain_qwen_nll: train/test split: {n_train} train + {n_test} test \
               positions ({}% / {}%)",
              (100.0 * n_train as f32 / n_pred as f32).round() as i32,
              (100.0 * n_test  as f32 / n_pred as f32).round() as i32);

    let train_qwen: Vec<&[f32]>  = qwen_per_pos[..n_train].to_vec();
    let train_brain: Vec<&[f32]> = brain_per_pos[..n_train].to_vec();
    let train_tgts: Vec<usize>   = targets[..n_train].to_vec();
    let test_qwen: Vec<&[f32]>   = qwen_per_pos[n_train..].to_vec();
    let test_brain: Vec<&[f32]>  = brain_per_pos[n_train..].to_vec();
    let test_tgts: Vec<usize>    = targets[n_train..].to_vec();

    // Recall mask: train_recall_mask[t] is true iff target token
    // train_tgts[t] has appeared at some earlier position in the
    // sequence. Used by --recall-only-loss to focus gradient on
    // recall-specific positions only.
    let train_recall_mask: Vec<bool> = (0..n_train).map(|t| {
        let target = train_tgts[t];
        // Was this target token seen at any position 0..=t in the
        // input sequence? (token_ids[..=t] are the tokens fed to the
        // model up to and including position t.)
        token_ids[..=t].iter().any(|&id| id as usize == target)
    }).collect();
    let n_recall = train_recall_mask.iter().filter(|&&b| b).count();
    if args.recall_only_loss {
        eprintln!("brain_qwen_nll: --recall-only-loss enabled — \
                   {n_recall}/{n_train} train positions are recall positions");
        if n_recall == 0 {
            eprintln!("brain_qwen_nll: ZERO recall positions in train split — \
                       no gradient signal. Disabling --recall-only-loss.");
        }
    }
    let active_recall_mask = if args.recall_only_loss && n_recall > 0 {
        Some(&train_recall_mask)
    } else {
        None
    };

    let test_baseline_nll = nll_per_token(&m_baseline, &test_qwen, None, &test_tgts);
    let test_random_nll   = nll_per_token(&m_random,   &test_qwen, Some(&test_brain), &test_tgts);
    eprintln!("brain_qwen_nll: TEST baseline (Qwen alone)        = {test_baseline_nll:.4}");
    eprintln!("brain_qwen_nll: TEST alpha=1 random init          = {test_random_nll:.4}");

    // Train modulator (low-rank or dense based on --rank flag).
    eprintln!("brain_qwen_nll: training modulator (rank={}) for {} epochs at lr={} …",
              args.rank, args.epochs, args.lr);
    let train_t = Instant::now();
    let (test_trained_nll, train_trained_nll, n_params) = if args.rank == 0 {
        // Dense path.
        let mut m_trained = BrainLogitModulator::new(BRAIN_DIM_FOR_MODULATOR, vocab);
        m_trained.alpha = 1.0;
        for w in m_trained.proj.weight.iter_mut() { *w *= 0.01; }
        let np = m_trained.proj.weight.len() + m_trained.proj.bias.len();
        eprintln!("  dense modulator: {} params", np);

        let mut d_w = vec![0.0f32; vocab * BRAIN_DIM_FOR_MODULATOR];
        let mut d_b = vec![0.0f32; vocab];
        let mut modulated = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let scale = match active_recall_mask {
            Some(_) => 1.0 / n_recall.max(1) as f32,
            None    => 1.0 / n_train as f32,
        };
        for epoch in 0..args.epochs {
            for t in 0..n_train {
                if let Some(mask) = active_recall_mask {
                    if !mask[t] { continue; }
                }
                m_trained.modulate_into(train_qwen[t], train_brain[t], &mut modulated);
                cross_entropy_grad(&modulated, train_tgts[t], &mut d_modulated);
                for v in 0..vocab { d_modulated[v] *= scale; }
                m_trained.backward(train_brain[t], &d_modulated, &mut d_w, &mut d_b);
            }
            m_trained.sgd_step(&mut d_w, &mut d_b, args.lr);
            if epoch == 0 || (epoch + 1) % (args.epochs.max(10) / 10) == 0
                || epoch == args.epochs - 1 {
                let train_nll = nll_per_token(
                    &m_trained, &train_qwen, Some(&train_brain), &train_tgts);
                let test_nll = nll_per_token(
                    &m_trained, &test_qwen, Some(&test_brain), &test_tgts);
                eprintln!("  epoch {:>3}/{}  train NLL = {train_nll:.4}  test NLL = {test_nll:.4}",
                          epoch + 1, args.epochs);
            }
        }
        let test  = nll_per_token(&m_trained, &test_qwen, Some(&test_brain), &test_tgts);
        let train = nll_per_token(&m_trained, &train_qwen, Some(&train_brain), &train_tgts);
        if args.per_position {
            print_per_position_breakdown(
                &m_trained, &test_qwen, &test_brain, &test_tgts,
                &token_ids, n_train, &tokenizer);
        }
        (test, train, np)
    } else {
        // Low-rank path.
        let mut m_trained = LowRankBrainLogitModulator::new(
            BRAIN_DIM_FOR_MODULATOR, args.rank, vocab);
        m_trained.alpha = 1.0;
        // Scale the up-projection so alpha=1 starts in the same logit
        // magnitude regime as Qwen's logits (down-projection is small,
        // up-projection multiplies by it before adding to logits).
        for w in m_trained.up.weight.iter_mut() { *w *= 0.01; }
        let np = m_trained.n_params();
        eprintln!("  low-rank modulator (rank={}): {} params", args.rank, np);

        let mut d_dw = vec![0.0f32; args.rank * BRAIN_DIM_FOR_MODULATOR];
        let mut d_db = vec![0.0f32; args.rank];
        let mut d_uw = vec![0.0f32; vocab * args.rank];
        let mut d_ub = vec![0.0f32; vocab];
        let mut modulated = vec![0.0f32; vocab];
        let mut d_modulated = vec![0.0f32; vocab];
        let mut z_scratch = vec![0.0f32; args.rank];
        let scale = match active_recall_mask {
            Some(_) => 1.0 / n_recall.max(1) as f32,
            None    => 1.0 / n_train as f32,
        };
        for epoch in 0..args.epochs {
            for t in 0..n_train {
                if let Some(mask) = active_recall_mask {
                    if !mask[t] { continue; }
                }
                m_trained.modulate_into_with_scratch(
                    train_qwen[t], train_brain[t], &mut modulated, &mut z_scratch);
                cross_entropy_grad(&modulated, train_tgts[t], &mut d_modulated);
                for v in 0..vocab { d_modulated[v] *= scale; }
                m_trained.backward(train_brain[t], &d_modulated,
                    &mut d_dw, &mut d_db, &mut d_uw, &mut d_ub);
            }
            m_trained.sgd_step(&mut d_dw, &mut d_db, &mut d_uw, &mut d_ub, args.lr);
            if epoch == 0 || (epoch + 1) % (args.epochs.max(10) / 10) == 0
                || epoch == args.epochs - 1 {
                let train_nll = nll_per_token(
                    &m_trained, &train_qwen, Some(&train_brain), &train_tgts);
                let test_nll = nll_per_token(
                    &m_trained, &test_qwen, Some(&test_brain), &test_tgts);
                eprintln!("  epoch {:>3}/{}  train NLL = {train_nll:.4}  test NLL = {test_nll:.4}",
                          epoch + 1, args.epochs);
            }
        }
        let test  = nll_per_token(&m_trained, &test_qwen, Some(&test_brain), &test_tgts);
        let train = nll_per_token(&m_trained, &train_qwen, Some(&train_brain), &train_tgts);
        if args.per_position {
            print_per_position_breakdown(
                &m_trained, &test_qwen, &test_brain, &test_tgts,
                &token_ids, n_train, &tokenizer);
        }
        (test, train, np)
    };
    let train_ms = train_t.elapsed().as_millis();
    eprintln!("brain_qwen_nll: training done in {train_ms} ms");
    eprintln!("brain_qwen_nll: modulator params = {n_params}, train positions = {n_train}, \
               params-per-example = {}",
              n_params / n_train.max(1));

    eprintln!();
    eprintln!("---- redshiftzero summary (with training) ----");
    eprintln!("  baseline (Qwen alone, full)         : {baseline_nll:.4}");
    eprintln!("  alpha=0   sanity (must == baseline) : {alpha_zero_nll:.4}");
    eprintln!();
    eprintln!("  TEST set ({n_test} held-out positions):");
    eprintln!("    Qwen alone                        : {test_baseline_nll:.4}");
    eprintln!("    + brain (random init)             : {test_random_nll:.4}  (Δ {:+.4})",
              test_random_nll - test_baseline_nll);
    eprintln!("    + brain (TRAINED, {} epochs)       : {test_trained_nll:.4}  (Δ {:+.4})",
              args.epochs, test_trained_nll - test_baseline_nll);
    eprintln!();
    eprintln!("  TRAIN set ({n_train} positions, for overfit context):");
    eprintln!("    Qwen alone                        : {:.4}",
              nll_per_token(&m_baseline, &train_qwen, None, &train_tgts));
    eprintln!("    + brain (TRAINED)                 : {train_trained_nll:.4}");
    eprintln!();
    let test_improvement = test_baseline_nll - test_trained_nll;
    if test_improvement > 0.0 {
        eprintln!("  → trained brain BEATS Qwen alone on held-out by {test_improvement:.4} nats/token");
    } else {
        eprintln!("  → trained brain UNDERPERFORMS Qwen alone on held-out by {:.4} nats/token", -test_improvement);
        eprintln!("    (modulator overfit train; lower --lr or --epochs, or use larger corpus)");
    }
}
