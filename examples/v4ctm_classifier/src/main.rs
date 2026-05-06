//! Train V4Ctm as a standalone classifier on imagenet10.
//!
//! Pipeline: image → VisualCortex.spatial_tokens → V4 tokens →
//! CtmWeights (configured for n_classes output) → predictions[tick] →
//! StepwiseCE loss → backward → SGD apply.
//!
//! After training, save weights to `/tmp/v4ctm_imagenet10.bin`. Then
//! `attention_viz` can load them and re-run the saccade-trace test:
//! does training the V4Ctm produce Sakana-style attention movement
//! between ticks?
//!
//! This matches Sakana's CTM-as-classifier setup: the attention is
//! shaped by classification loss, not by some downstream consumer.
//!
//! CLI / env:
//!   `MODGRAD_VARIANT=cifar_ln`     cortex variant for V4 token extraction
//!   `MODGRAD_EPOCHS=10`            training epochs
//!   `MODGRAD_LR=0.01`              learning rate (SGD with grad-clip)
//!   `MODGRAD_BATCH_SIZE=8`         gradient accumulation batch
//!   `MODGRAD_TICKS=4`              CTM iterations
//!   `MODGRAD_OUT=/tmp/v4ctm_imagenet10.bin`  save path

use modgrad_codec::cifar::{load_feat, CifarImage};
use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::weights::CtmWeights;
use modgrad_ctm::train::{accumulate_gradients, Ctm, CtmGradients};
use modgrad_traits::{Brain, LossFn, StepwiseCE, TokenInput};
use modgrad_device::backend::{ops, AdamWArgs};

/// AdamW state for V4Ctm's big tensors (NLM s1/s2, KV/Q/MHA projections,
/// output_proj). Mirrors `RegionInnerAdamW` from modgrad-ctm but built
/// here because that type is `pub(crate)`.
///
/// Synapse U-Net + small init/scalar tensors (start_activated, kv_ln,
/// decay params, exit gate) stay on plain SGD via
/// `CtmGradients::apply_minor` — same split as the SDK uses internally.
struct V4CtmAdamW {
    nlm_s1_w_m: Vec<f32>, nlm_s1_w_v: Vec<f32>,
    nlm_s1_b_m: Vec<f32>, nlm_s1_b_v: Vec<f32>,
    nlm_s2_w_m: Option<Vec<f32>>, nlm_s2_w_v: Option<Vec<f32>>,
    nlm_s2_b_m: Option<Vec<f32>>, nlm_s2_b_v: Option<Vec<f32>>,
    kv_w_m: Vec<f32>, kv_w_v: Vec<f32>,
    kv_b_m: Vec<f32>, kv_b_v: Vec<f32>,
    q_w_m: Vec<f32>, q_w_v: Vec<f32>,
    q_b_m: Vec<f32>, q_b_v: Vec<f32>,
    mhain_w_m: Vec<f32>, mhain_w_v: Vec<f32>,
    mhain_b_m: Vec<f32>, mhain_b_v: Vec<f32>,
    mhaout_w_m: Vec<f32>, mhaout_w_v: Vec<f32>,
    mhaout_b_m: Vec<f32>, mhaout_b_v: Vec<f32>,
    out_w_m: Vec<f32>, out_w_v: Vec<f32>,
    out_b_m: Vec<f32>, out_b_v: Vec<f32>,
    t: u32,
    lr: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl V4CtmAdamW {
    fn new(w: &CtmWeights, lr: f32) -> Self {
        let nlm_s1_w_n = w.nlm_stage1.weights.len();
        let nlm_s1_b_n = w.nlm_stage1.biases.len();
        let (s2_w_n, s2_b_n) = w.nlm_stage2.as_ref().map(|s| (s.weights.len(), s.biases.len())).unzip();
        Self {
            nlm_s1_w_m: vec![0.0; nlm_s1_w_n], nlm_s1_w_v: vec![0.0; nlm_s1_w_n],
            nlm_s1_b_m: vec![0.0; nlm_s1_b_n], nlm_s1_b_v: vec![0.0; nlm_s1_b_n],
            nlm_s2_w_m: s2_w_n.map(|n| vec![0.0; n]),
            nlm_s2_w_v: s2_w_n.map(|n| vec![0.0; n]),
            nlm_s2_b_m: s2_b_n.map(|n| vec![0.0; n]),
            nlm_s2_b_v: s2_b_n.map(|n| vec![0.0; n]),
            kv_w_m: vec![0.0; w.kv_proj.weight.len()],
            kv_w_v: vec![0.0; w.kv_proj.weight.len()],
            kv_b_m: vec![0.0; w.kv_proj.bias.len()],
            kv_b_v: vec![0.0; w.kv_proj.bias.len()],
            q_w_m: vec![0.0; w.q_proj.weight.len()],
            q_w_v: vec![0.0; w.q_proj.weight.len()],
            q_b_m: vec![0.0; w.q_proj.bias.len()],
            q_b_v: vec![0.0; w.q_proj.bias.len()],
            mhain_w_m: vec![0.0; w.mha_in_proj.weight.len()],
            mhain_w_v: vec![0.0; w.mha_in_proj.weight.len()],
            mhain_b_m: vec![0.0; w.mha_in_proj.bias.len()],
            mhain_b_v: vec![0.0; w.mha_in_proj.bias.len()],
            mhaout_w_m: vec![0.0; w.mha_out_proj.weight.len()],
            mhaout_w_v: vec![0.0; w.mha_out_proj.weight.len()],
            mhaout_b_m: vec![0.0; w.mha_out_proj.bias.len()],
            mhaout_b_v: vec![0.0; w.mha_out_proj.bias.len()],
            out_w_m: vec![0.0; w.output_proj.weight.len()],
            out_w_v: vec![0.0; w.output_proj.weight.len()],
            out_b_m: vec![0.0; w.output_proj.bias.len()],
            out_b_v: vec![0.0; w.output_proj.bias.len()],
            t: 0, lr, weight_decay: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8,
        }
    }

    fn step(&mut self, w: &mut CtmWeights, g: &mut CtmGradients) {
        self.t += 1;
        let bc1_inv = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let bc2_inv = 1.0 / (1.0 - self.beta2.powi(self.t as i32));
        let mut adam = |wts: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32]| {
            ops::adamw(AdamWArgs {
                w: wts, g: grad, m, v,
                lr: self.lr, beta1: self.beta1, beta2: self.beta2, eps: self.eps,
                weight_decay: self.weight_decay, bc1_inv, bc2_inv,
            }).expect("adamw dispatch");
        };
        adam(&mut w.nlm_stage1.weights, &g.nlm_s1_w, &mut self.nlm_s1_w_m, &mut self.nlm_s1_w_v);
        adam(&mut w.nlm_stage1.biases,  &g.nlm_s1_b, &mut self.nlm_s1_b_m, &mut self.nlm_s1_b_v);
        if let (Some(s2), Some(g_w), Some(g_b),
                Some(m_w), Some(v_w), Some(m_b), Some(v_b)) = (
            w.nlm_stage2.as_mut(),
            g.nlm_s2_w.as_ref(), g.nlm_s2_b.as_ref(),
            self.nlm_s2_w_m.as_mut(), self.nlm_s2_w_v.as_mut(),
            self.nlm_s2_b_m.as_mut(), self.nlm_s2_b_v.as_mut(),
        ) {
            adam(&mut s2.weights, g_w, m_w, v_w);
            adam(&mut s2.biases,  g_b, m_b, v_b);
        }
        adam(&mut w.kv_proj.weight,    &g.kv_proj_w, &mut self.kv_w_m, &mut self.kv_w_v);
        adam(&mut w.kv_proj.bias,      &g.kv_proj_b, &mut self.kv_b_m, &mut self.kv_b_v);
        adam(&mut w.q_proj.weight,     &g.q_proj_w,  &mut self.q_w_m,  &mut self.q_w_v);
        adam(&mut w.q_proj.bias,       &g.q_proj_b,  &mut self.q_b_m,  &mut self.q_b_v);
        adam(&mut w.mha_in_proj.weight, &g.mha_in_w, &mut self.mhain_w_m, &mut self.mhain_w_v);
        adam(&mut w.mha_in_proj.bias,   &g.mha_in_b, &mut self.mhain_b_m, &mut self.mhain_b_v);
        adam(&mut w.mha_out_proj.weight,&g.mha_out_w,&mut self.mhaout_w_m,&mut self.mhaout_w_v);
        adam(&mut w.mha_out_proj.bias,  &g.mha_out_b,&mut self.mhaout_b_m,&mut self.mhaout_b_v);
        adam(&mut w.output_proj.weight, &g.out_proj_w, &mut self.out_w_m, &mut self.out_w_v);
        adam(&mut w.output_proj.bias,   &g.out_proj_b, &mut self.out_b_m, &mut self.out_b_v);
        // Small/scalar tensors + synapse stay on plain SGD.
        g.apply_minor(w, self.lr);
    }
}

/// Paths overridable via MODGRAD_TRAIN_PATH / MODGRAD_EVAL_PATH so we
/// can swap imagenet10 (1k samples) for CIFAR-10 native (50k samples)
/// without recompiling.
fn train_path() -> String {
    std::env::var("MODGRAD_TRAIN_PATH")
        .unwrap_or_else(|_| "/tmp/retina_imagenet10_train.feat".into())
}
fn eval_path() -> String {
    std::env::var("MODGRAD_EVAL_PATH")
        .unwrap_or_else(|_| "/tmp/retina_imagenet10_eval.feat".into())
}
const N_CLASSES: usize = 10;

fn build_cortex() -> VisualCortex {
    let v = std::env::var("MODGRAD_VARIANT").unwrap_or_else(|_| "cifar_ln".into());
    match v.as_str() {
        "cifar_ln"     => VisualCortex::cifar_ln(),
        "cifar"        => VisualCortex::cifar(),
        "dog_only_ln"  => VisualCortex::cifar_retina_only_ln(32, 32),
        "random"       => VisualCortex::random(32, 32),
        other => panic!("unknown variant '{other}'"),
    }
}

fn build_classifier(token_dim: usize, ticks: usize) -> CtmWeights {
    let d_model: usize = std::env::var("MODGRAD_D_MODEL").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(128);
    let heads: usize = (d_model / 16).max(1).min(8);
    let cfg = CtmConfig {
        iterations: ticks,
        d_model,
        d_input: d_model,
        heads,
        n_synch_out: d_model,
        n_synch_action: d_model,
        synapse_depth: 2,
        memory_length: 8,
        deep_nlms: true,
        memory_hidden_dims: 4,
        out_dims: N_CLASSES,                 // ← 10-way classifier head
        n_random_pairing_self: 0,
        min_width: (d_model / 4).max(4),
        exit_strategy: ExitStrategy::None,
        collect_trajectories: false,
    };
    CtmWeights::new(cfg, token_dim)
}

fn extract_tokens(cortex: &VisualCortex, img: &CifarImage) -> (Vec<f32>, usize, usize) {
    let scales = cortex.spatial_tokens_multiscale(&img.pixels);
    let (v4_tok, n_tokens, channels) = (scales[2].0.clone(), scales[2].1, scales[2].2);
    (v4_tok, n_tokens, channels)
}

fn eval_acc(w: &CtmWeights, cortex: &VisualCortex, eval: &[CifarImage], n_max: usize) -> f32 {
    use modgrad_traits::Brain;
    let mut correct = 0usize;
    let mut total = 0usize;
    for img in eval.iter().take(n_max) {
        let (tokens, n_tokens, token_dim) = extract_tokens(cortex, img);
        let state = modgrad_ctm::train::Ctm::init_state(w);
        let input = modgrad_traits::TokenInput { tokens, n_tokens, token_dim };
        let (output, _state) = modgrad_ctm::train::Ctm::forward(w, state, &input);
        if let Some(last) = output.predictions.last() {
            let pred = (0..N_CLASSES)
                .max_by(|&a, &b| last[a].partial_cmp(&last[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            if pred == img.label { correct += 1; }
        }
        total += 1;
    }
    correct as f32 / total.max(1) as f32
}

fn scale_gradients(g: &mut CtmGradients, s: f32) {
    let scale = |v: &mut Vec<f32>| { for x in v.iter_mut() { *x *= s; } };
    scale(&mut g.nlm_s1_w); scale(&mut g.nlm_s1_b);
    if let Some(v) = g.nlm_s2_w.as_mut() { scale(v); }
    if let Some(v) = g.nlm_s2_b.as_mut() { scale(v); }
    scale(&mut g.kv_proj_w); scale(&mut g.kv_proj_b);
    scale(&mut g.kv_ln_d_gamma); scale(&mut g.kv_ln_d_beta);
    scale(&mut g.q_proj_w); scale(&mut g.q_proj_b);
    scale(&mut g.mha_in_w); scale(&mut g.mha_in_b);
    scale(&mut g.mha_out_w); scale(&mut g.mha_out_b);
    scale(&mut g.d_decay_out); scale(&mut g.d_decay_action);
    scale(&mut g.out_proj_w); scale(&mut g.out_proj_b);
    scale(&mut g.d_start_activated); scale(&mut g.d_start_trace);
    if let Some(v) = g.exit_gate_w.as_mut() { scale(v); }
    if let Some(v) = g.exit_gate_b.as_mut() { scale(v); }
    // Synapse U-Net grads — leave unscaled; apply_minor's path will see
    // them at 1× and apply its own SGD lr. (Matches RegionInnerAdamW
    // which doesn't touch synapse either.)
}

fn shuffle(idx: &mut [usize], seed: &mut u64) {
    for i in (1..idx.len()).rev() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((*seed >> 33) as usize) % (i + 1);
        idx.swap(i, r);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs: usize = std::env::var("MODGRAD_EPOCHS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(10);
    let lr: f32 = std::env::var("MODGRAD_LR").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let bs: usize = std::env::var("MODGRAD_BATCH_SIZE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(8);
    let ticks: usize = std::env::var("MODGRAD_TICKS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let out_path: String = std::env::var("MODGRAD_OUT")
        .unwrap_or_else(|_| "/tmp/v4ctm_imagenet10.bin".into());

    let cortex = build_cortex();
    eprintln!("variant: {}", std::env::var("MODGRAD_VARIANT").unwrap_or("cifar_ln".into()));

    let tp = train_path();
    let ep = eval_path();
    eprintln!("train_path: {tp}");
    eprintln!("eval_path:  {ep}");
    let mut train = load_feat(&tp)?;
    let mut eval = load_feat(&ep)?;
    // MODGRAD_MAX_TRAIN / MODGRAD_MAX_EVAL: cap dataset for quick smoke
    // runs on big datasets like CIFAR-10 native (50k/10k).
    if let Ok(n) = std::env::var("MODGRAD_MAX_TRAIN") {
        let n: usize = n.parse().unwrap_or(usize::MAX);
        train.truncate(n);
    }
    if let Ok(n) = std::env::var("MODGRAD_MAX_EVAL") {
        let n: usize = n.parse().unwrap_or(usize::MAX);
        eval.truncate(n);
    }
    eprintln!("data: {} train, {} eval", train.len(), eval.len());

    // Probe one image to discover token shape.
    let (_, n_tokens, token_dim) = extract_tokens(&cortex, &train[0]);
    eprintln!("V4 grid: {} tokens × {} dim", n_tokens, token_dim);

    let mut w = build_classifier(token_dim, ticks);
    eprintln!("classifier: d_model={} heads=8 ticks={ticks} out_dims={N_CLASSES} params={}",
        128, w.n_params());

    let optimizer_kind = std::env::var("MODGRAD_OPTIMIZER").unwrap_or_else(|_| "adamw".into());
    let mut adamw = if optimizer_kind == "adamw" {
        Some(V4CtmAdamW::new(&w, lr))
    } else {
        None
    };
    eprintln!("optimizer: {}", optimizer_kind);

    let loss_fn = StepwiseCE { n_classes: N_CLASSES, lookahead: 1 };
    let mut idx: Vec<usize> = (0..train.len()).collect();
    let mut rng: u64 = 0xCAFE_BABE;

    eprintln!("\nepochs={epochs} lr={lr} bs={bs}");
    let acc0 = eval_acc(&w, &cortex, &eval, 200);
    eprintln!("ep 0/{epochs}  pre-train eval_acc(200)={:.1}%", acc0 * 100.0);

    for ep in 0..epochs {
        shuffle(&mut idx, &mut rng);
        let t_ep = std::time::Instant::now();
        let mut ep_loss = 0.0f32;
        let mut ep_correct = 0usize;
        let mut n_seen = 0usize;
        for chunk in idx.chunks(bs) {
            let mut batch_grads = CtmGradients::zeros(&w);
            for &i in chunk {
                let img = &train[i];
                let (tokens, n_tok, td) = extract_tokens(&cortex, img);
                let state = Ctm::init_state(&w);
                let input = TokenInput { tokens, n_tokens: n_tok, token_dim: td };
                let (output, _state, cache) = Ctm::forward_cached(&w, state, &input);
                let target = [img.label];
                let (loss, d_preds) = loss_fn.compute(
                    &output.predictions, &output.certainties, &target as &[usize],
                );
                let step_grads = Ctm::backward(&w, cache, &d_preds);
                accumulate_gradients(&mut batch_grads, &step_grads);
                ep_loss += loss;
                if let Some(last) = output.predictions.last() {
                    let pred = (0..N_CLASSES).max_by(|&a, &b|
                        last[a].partial_cmp(&last[b]).unwrap_or(std::cmp::Ordering::Equal)
                    ).unwrap();
                    if pred == img.label { ep_correct += 1; }
                }
                n_seen += 1;
            }
            // Average gradients over batch.
            let inv_bs = 1.0 / chunk.len() as f32;
            scale_gradients(&mut batch_grads, inv_bs);
            if let Some(ref mut adam) = adamw {
                adam.step(&mut w, &mut batch_grads);
            } else {
                batch_grads.apply(&mut w, lr, /*clip=*/5.0);
            }
        }
        let train_acc = ep_correct as f32 / n_seen.max(1) as f32;
        let test_acc = eval_acc(&w, &cortex, &eval, 200);
        eprintln!(
            "ep {:>2}/{epochs}  loss={:.3}  train={:.1}%  eval(200)={:.1}%  {:.1}s",
            ep + 1, ep_loss / n_seen.max(1) as f32, train_acc * 100.0,
            test_acc * 100.0, t_ep.elapsed().as_secs_f32(),
        );
    }

    w.save(&out_path)?;
    eprintln!("\nsaved: {}", out_path);

    // ─── Linear-probe baseline on the same V4 features ───────────
    //
    // The CTM is 154k params; a 10×128 linear probe is 1280 params —
    // 100× simpler. If V4Ctm < linear probe at the same data scale,
    // CTM ticks aren't earning their weights and the architecture is
    // overfitting more than it's discovering structure. This is the
    // honest "is the CTM doing anything" diagnostic.
    eprintln!("\n─── linear-probe baseline (mean-pooled V4) ───");
    let mut train_feat = vec![0.0f32; train.len() * token_dim];
    let mut train_lab = vec![0usize; train.len()];
    let mut eval_feat = vec![0.0f32; eval.len() * token_dim];
    let mut eval_lab = vec![0usize; eval.len()];
    let pool_into = |feats: &mut [f32], labs: &mut [usize], imgs: &[CifarImage]| {
        for (i, img) in imgs.iter().enumerate() {
            let (toks, n_tok, ch) = extract_tokens(&cortex, img);
            let f = &mut feats[i * ch..(i + 1) * ch];
            for t in 0..n_tok {
                for c in 0..ch { f[c] += toks[t * ch + c]; }
            }
            let inv = 1.0 / n_tok as f32;
            for c in 0..ch { f[c] *= inv; }
            labs[i] = img.label;
        }
    };
    pool_into(&mut train_feat, &mut train_lab, &train);
    pool_into(&mut eval_feat, &mut eval_lab, &eval);
    let lp_acc = train_linear_probe(&train_feat, &train_lab, &eval_feat, &eval_lab, token_dim);
    let v4ctm_acc = eval_acc(&w, &cortex, &eval, 1000);
    eprintln!(
        "\n═══ benchmark report ═══\n  linear probe (mean-pool V4 → softmax): {:.1}%\n  V4Ctm classifier (trained):           {:.1}%\n  Δ (V4Ctm − LP): {:+.1}pp\n",
        lp_acc * 100.0, v4ctm_acc * 100.0, (v4ctm_acc - lp_acc) * 100.0,
    );
    if v4ctm_acc >= lp_acc {
        eprintln!("  → V4Ctm beats its linear baseline. CTM ticks add value.");
    } else {
        eprintln!("  → V4Ctm UNDER-performs its linear baseline.");
        eprintln!("    CTM is overfitting on {} samples; needs more data,", train.len());
        eprintln!("    smaller model, or stronger regularization.");
    }
    Ok(())
}

/// Plain linear probe: 30 epochs of SGD on softmax-CE over mean-pooled
/// V4 features. Mirrors `cifar10_probe`'s `train_linear_probe` so the
/// comparison is apples-to-apples.
fn train_linear_probe(
    train_feat: &[f32], train_lab: &[usize],
    eval_feat: &[f32], eval_lab: &[usize],
    feat_dim: usize,
) -> f32 {
    const LP_EPOCHS: usize = 30;
    const LP_LR: f32 = 0.05;
    let n_train = train_lab.len();
    let mut w = vec![0.0f32; N_CLASSES * feat_dim];
    let mut b = vec![0.0f32; N_CLASSES];
    let mut idx: Vec<usize> = (0..n_train).collect();
    let mut rng: u64 = 0xC0FFEE;
    for _ep in 0..LP_EPOCHS {
        shuffle(&mut idx, &mut rng);
        for &i in &idx {
            let f = &train_feat[i * feat_dim..(i + 1) * feat_dim];
            let y = train_lab[i];
            let mut logits = vec![0.0f32; N_CLASSES];
            for k in 0..N_CLASSES {
                let mut s = b[k];
                for j in 0..feat_dim { s += w[k * feat_dim + j] * f[j]; }
                logits[k] = s;
            }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for k in 0..N_CLASSES { sum += (logits[k] - mx).exp(); }
            for k in 0..N_CLASSES {
                let p = (logits[k] - mx).exp() / sum;
                let g = p - if k == y { 1.0 } else { 0.0 };
                b[k] -= LP_LR * g;
                for j in 0..feat_dim { w[k * feat_dim + j] -= LP_LR * g * f[j]; }
            }
        }
    }
    let n_eval = eval_lab.len();
    let mut correct = 0;
    for i in 0..n_eval {
        let f = &eval_feat[i * feat_dim..(i + 1) * feat_dim];
        let mut best = 0; let mut bv = f32::NEG_INFINITY;
        for k in 0..N_CLASSES {
            let mut s = b[k];
            for j in 0..feat_dim { s += w[k * feat_dim + j] * f[j]; }
            if s > bv { bv = s; best = k; }
        }
        if best == eval_lab[i] { correct += 1; }
    }
    correct as f32 / n_eval.max(1) as f32
}
