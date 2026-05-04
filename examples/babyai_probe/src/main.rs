//! BabyAI behavioral-cloning probe — RGB pipeline through VisualCortex.
//!
//! Reads `export_rgb_demos.py`'s BAYR binary (egocentric 56×56 RGB
//! frames + Qwen mission tokens + expert action), pushes each frame
//! through a `VisualCortex` (retina + V1 + V2 + V4), feeds V4 tokens
//! into an 8-region `RegionalBrain`, and trains the brain to predict
//! the bot's action.
//!
//! Goal: prove the visual hierarchy actually improves performance.
//! Compare four retina variants on the same demos:
//!
//!   A  no retina      raw 56×56×3 pixels reshaped into tokens (no priors)
//!   B  random retina  VisualCortex::random(56, 56) (architecture, no priors)
//!   C  Gabor priors   VisualCortex::new(56, 56) — DoG + Gabor V1 + random V2/V4
//!   D  pretrained     load STL-10-pretrained .bin from `pretrain_retina`
//!
//! Win condition: D > C > B > A on eval accuracy. That sequence
//! demonstrates each layer of inductive prior + training contributes.
//!
//! Brain training uses `RegionalAdamW` with mini-batch grad
//! accumulation — the SDK pattern from `examples/mazes`.
//!
//! This iteration: variant C end-to-end. A/B/D are 30-line additions
//! once the C pipeline is verified.

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::config::ExitStrategy;
use modgrad_ctm::graph::{
    RegionalAdamW, RegionalBrain, RegionalConfig, RegionalGradients, RegionalWeights,
};
use modgrad_traits::{Brain, Encoder, LossFn, StepwiseCE, TokenInput};

const N_ACTIONS: usize = 7;
// Ticks of CTM "thinking" per sample. With d_model=1024 cortical, each
// tick is heavy; 4 keeps wall-clock manageable while still giving
// enough recurrent depth.
const TICKS: usize = 4;
const IMG_H: usize = 56;
const IMG_W: usize = 56;
const IMG_PIXELS: usize = 3 * IMG_H * IMG_W; // 9408

#[allow(dead_code)]
struct Step {
    mission_id: u32,
    /// CHW f32 in [0, 1].
    rgb: Vec<f32>,
    action: u8,
}

struct Dataset {
    #[allow(dead_code)]
    missions: Vec<Vec<u32>>,
    img_h: usize,
    img_w: usize,
    steps: Vec<Step>,
}

fn load(path: impl AsRef<Path>) -> IoResult<Dataset> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    assert!(buf.len() >= 32, "file too short for header");
    assert_eq!(&buf[..4], b"BAYR", "bad magic — expected BAYR");
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    assert_eq!(version, 1);
    let n_steps    = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let n_missions = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    let img_h      = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
    let img_w      = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
    let mt_off     = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;
    let steps_off  = u32::from_le_bytes(buf[28..32].try_into().unwrap()) as usize;
    let img_bytes  = 3 * img_h * img_w;
    let step_record = 4 + img_bytes + 1;

    let mut missions = Vec::with_capacity(n_missions);
    let mut o = mt_off;
    for _ in 0..n_missions {
        let n_tok = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize; o += 4;
        let mut ids = Vec::with_capacity(n_tok);
        for _ in 0..n_tok {
            ids.push(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap())); o += 4;
        }
        missions.push(ids);
    }
    assert_eq!(o, steps_off, "mission table didn't end at steps_off");

    let mut steps = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let base = steps_off + i * step_record;
        let mid = u32::from_le_bytes(buf[base..base + 4].try_into().unwrap());
        let raw_bytes = &buf[base + 4..base + 4 + img_bytes];
        // u8 → f32 in [0, 1].
        let rgb: Vec<f32> = raw_bytes.iter().map(|&b| b as f32 / 255.0).collect();
        let action = buf[base + 4 + img_bytes];
        steps.push(Step { mission_id: mid, rgb, action });
    }
    Ok(Dataset { missions, img_h, img_w, steps })
}

fn shuffle_idx(idx: &mut [usize], seed: &mut u64) {
    for i in (1..idx.len()).rev() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((*seed >> 33) as usize) % (i + 1);
        idx.swap(i, r);
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0; let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; best = i; } }
    best
}

fn tier0_always_forward(steps: &[Step]) -> f32 {
    let total = steps.len() as f32;
    let correct = steps.iter().filter(|s| s.action == 2).count() as f32;
    correct / total
}

// ─── 8-region brain on top of VisualCortex output ─────────────────

fn build_brain(obs_dim: usize) -> RegionalWeights {
    // Phase 5: select brain size via env var so the speedup table
    // can be filled across multiple scales. Defaults to billion for
    // continuity with the Phase 1-4 measurements. Router stays off
    // (forward_cached_resident falls back to host when router is set).
    let size = std::env::var("MODGRAD_BRAIN_SIZE").unwrap_or_else(|_| "billion".into());
    let mut cfg = match size.as_str() {
        "small"   => RegionalConfig::eight_region_small(obs_dim, N_ACTIONS, TICKS),
        "medium"  => RegionalConfig::eight_region_medium(obs_dim, N_ACTIONS, TICKS),
        "large"   => RegionalConfig::eight_region_large(obs_dim, N_ACTIONS, TICKS),
        "billion" | _ => RegionalConfig::eight_region_billion(obs_dim, N_ACTIONS, TICKS),
    };
    cfg.exit_strategy = ExitStrategy::None;
    cfg.router = None;
    RegionalWeights::new(cfg)
}

fn add_grads(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) { *d += s; }
}

fn accumulate(batch: &mut RegionalGradients, sample: &RegionalGradients) {
    for r in 0..batch.region_grads.len() {
        add_grads(&mut batch.region_grads[r].nlm_s1_w,  &sample.region_grads[r].nlm_s1_w);
        add_grads(&mut batch.region_grads[r].nlm_s1_b,  &sample.region_grads[r].nlm_s1_b);
        add_grads(&mut batch.region_grads[r].kv_proj_w, &sample.region_grads[r].kv_proj_w);
        add_grads(&mut batch.region_grads[r].kv_proj_b, &sample.region_grads[r].kv_proj_b);
        add_grads(&mut batch.region_grads[r].q_proj_w,  &sample.region_grads[r].q_proj_w);
        add_grads(&mut batch.region_grads[r].q_proj_b,  &sample.region_grads[r].q_proj_b);
        add_grads(&mut batch.region_grads[r].mha_in_w,  &sample.region_grads[r].mha_in_w);
        add_grads(&mut batch.region_grads[r].mha_in_b,  &sample.region_grads[r].mha_in_b);
        add_grads(&mut batch.region_grads[r].mha_out_w, &sample.region_grads[r].mha_out_w);
        add_grads(&mut batch.region_grads[r].mha_out_b, &sample.region_grads[r].mha_out_b);
        add_grads(&mut batch.region_grads[r].out_proj_w, &sample.region_grads[r].out_proj_w);
        add_grads(&mut batch.region_grads[r].out_proj_b, &sample.region_grads[r].out_proj_b);
    }
    add_grads(&mut batch.output_proj_dw, &sample.output_proj_dw);
    add_grads(&mut batch.output_proj_db, &sample.output_proj_db);
    for ci in 0..batch.connection_dw.len() {
        add_grads(&mut batch.connection_dw[ci], &sample.connection_dw[ci]);
        add_grads(&mut batch.connection_db[ci], &sample.connection_db[ci]);
    }
}


fn main() {
    // CLI: <demos_path> [cortex_path]
    //   demos_path  default /tmp/babyai_rgb_demos.bin
    //   cortex_path optional — when set, loads a pretrained VisualCortex
    //               via VisualCortex::load instead of fresh new(). This is
    //               how variant D (STL-10 pretrained) gets compared.
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/babyai_rgb_demos.bin".into());
    let cortex_path = std::env::args().nth(2);
    eprintln!("loading {path}");
    let ds = load(&path).expect("load demos");
    eprintln!(
        "  {} steps  {} missions  img={}×{}",
        ds.steps.len(), ds.missions.len(), ds.img_h, ds.img_w,
    );
    assert_eq!((ds.img_h, ds.img_w), (IMG_H, IMG_W),
               "expected {IMG_H}×{IMG_W}; got {}×{}", ds.img_h, ds.img_w);
    assert_eq!(ds.steps[0].rgb.len(), IMG_PIXELS);

    let mut hist = [0usize; N_ACTIONS];
    for s in &ds.steps { hist[s.action as usize] += 1; }
    let names = ["L", "R", "F", "pickup", "drop", "toggle", "done"];
    eprintln!("  action histogram:");
    for k in 0..N_ACTIONS {
        let pct = 100.0 * hist[k] as f32 / ds.steps.len() as f32;
        eprintln!("    {:<7} {:>5}  ({pct:>4.1}%)", names[k], hist[k]);
    }

    let t0 = tier0_always_forward(&ds.steps);
    eprintln!("\n[Tier 0] always-Forward: {:.1}%", t0 * 100.0);

    let mut order: Vec<usize> = (0..ds.steps.len()).collect();
    let mut rng: u64 = 0x5EED_5EED_5EED_5EED;
    shuffle_idx(&mut order, &mut rng);
    // Override via env var for smoke runs. Default 2000 keeps Phase 0
    // measurements meaningful; smaller values (e.g. MAX_SAMPLES=64) let
    // us validate the dispatch_profile output without burning CPU.
    let max_samples: usize = std::env::var("MODGRAD_MAX_SAMPLES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    if order.len() > max_samples { order.truncate(max_samples); }
    let split = (order.len() as f32 * 0.8) as usize;
    let train: Vec<Step> = order[..split].iter().map(|&i| Step {
        mission_id: ds.steps[i].mission_id,
        rgb: ds.steps[i].rgb.clone(),
        action: ds.steps[i].action,
    }).collect();
    let eval: Vec<Step> = order[split..].iter().map(|&i| Step {
        mission_id: ds.steps[i].mission_id,
        rgb: ds.steps[i].rgb.clone(),
        action: ds.steps[i].action,
    }).collect();

    let (variant_name, cortex) = match &cortex_path {
        Some(p) => {
            eprintln!("\n[Variant D] PRETRAINED  (loaded from {p})");
            let mut c = VisualCortex::load(p).expect("load pretrained cortex");
            // Pretrain saves at the training resolution; reset spatial
            // dims so the same weights work at our 56×56 input.
            c.input_h = IMG_H;
            c.input_w = IMG_W;
            ("Variant D — pretrained", c)
        }
        None => {
            eprintln!("\n[Variant C] Gabor priors  (VisualCortex::new({IMG_H}, {IMG_W}))");
            ("Variant C — Gabor priors", VisualCortex::new(IMG_H, IMG_W))
        }
    };
    let _ = variant_name;
    eprintln!("  train={}  eval={}", train.len(), eval.len());

    // Phase 5 profile mode: run BOTH host and resident forwards
    // back-to-back, report side-by-side speedup. Brain size selected
    // via MODGRAD_BRAIN_SIZE (small / medium / large / billion).
    // Activated by MODGRAD_PROFILE_ONLY=N.
    if std::env::var_os("MODGRAD_PROFILE_ONLY").is_some() {
        let n = std::env::var("MODGRAD_PROFILE_ONLY")
            .ok().and_then(|s| s.parse::<usize>().ok()).filter(|&n| n > 0)
            .unwrap_or(4);
        let n = n.min(train.len()).max(1);
        let size = std::env::var("MODGRAD_BRAIN_SIZE").unwrap_or_else(|_| "billion".into());
        eprintln!("\n[phase5-profile] brain size = {size}, {n} forward passes per path");
        let w = build_brain(cortex.token_dim());

        // ── Host path. ──
        let t_host = std::time::Instant::now();
        for i in 0..n {
            let input = cortex.encode(&train[i].rgb);
            let state = RegionalBrain::init_state(&w);
            let _ = RegionalBrain::forward(&w, state, &input);
        }
        let host_total = t_host.elapsed();
        let host_per = 1000.0 * host_total.as_secs_f32() / n as f32;

        // ── Resident path (only with --features rocm). ──
        #[cfg(feature = "rocm")]
        let resident_per = {
            use modgrad_ctm::resident::RegionalResidentCache;
            use modgrad_device::backend::HipBatch;
            let cache = RegionalResidentCache::from_weights(&w).expect("cache build");
            let batch = HipBatch::new();
            let t = std::time::Instant::now();
            for i in 0..n {
                let input = cortex.encode(&train[i].rgb);
                let fresh = cache.fresh(&w).expect("cache fresh");
                let _ = modgrad_ctm::RegionalBrain::forward_cached_resident(
                    &w, fresh, &batch, &input,
                ).expect("forward_cached_resident");
            }
            batch.flush().expect("flush");
            let resident_total = t.elapsed();
            1000.0 * resident_total.as_secs_f32() / n as f32
        };
        #[cfg(not(feature = "rocm"))]
        let resident_per = f32::NAN;

        eprintln!();
        eprintln!("[phase5-profile] forward-only timings, {n} samples each:");
        eprintln!("  host      : {:>8.1} ms/forward", host_per);
        eprintln!("  resident  : {:>8.1} ms/forward", resident_per);
        if resident_per.is_finite() && resident_per > 0.0 {
            eprintln!("  speedup   : {:>8.2}× (host / resident)", host_per / resident_per);
        }
        modgrad_ctm::dispatch_profile::dump();
        return;
    }

    // Sanity: cortex output magnitude + cross-sample diversity.
    {
        let probe0 = cortex.encode(&train[0].rgb);
        let probe1 = cortex.encode(&train[1].rgb);
        let probe7 = cortex.encode(&train[7].rgb);
        let n = probe0.tokens.len() as f32;
        let mean = probe0.tokens.iter().sum::<f32>() / n;
        let var = probe0.tokens.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let mn = probe0.tokens.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = probe0.tokens.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // Cross-sample similarity: cosine between sample 0 and sample {1,7} feature vectors.
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() };
        let norm = |a: &[f32]| -> f32 { dot(a, a).sqrt() };
        let cos01 = dot(&probe0.tokens, &probe1.tokens) / (norm(&probe0.tokens) * norm(&probe1.tokens) + 1e-8);
        let cos07 = dot(&probe0.tokens, &probe7.tokens) / (norm(&probe0.tokens) * norm(&probe7.tokens) + 1e-8);
        eprintln!(
            "  cortex output: n_tokens={} token_dim={}  mean={:.3} std={:.3} min={:.3} max={:.3}",
            probe0.n_tokens, probe0.token_dim, mean, var.sqrt(), mn, mx,
        );
        eprintln!(
            "  cross-sample cosine: train[0] vs train[1]={cos01:.3}  vs train[7]={cos07:.3}  (low = diverse, near 1.0 = collapsed)",
        );
    }
    // Dump cortex features for sklearn-side analysis (cheap, always do it).
    dump_cortex_features(&cortex, &train, &eval, "/tmp/babyai_cortex_feats.bin");

    // bs=16 + lr=1e-4 because (a) larger model → smaller LR, (b) batch=16
    // keeps activation memory comfortable on 8 GB VRAM with d_model=1024.
    let w = study_brain(&cortex, &train, &eval, 3, 16, 1e-4);

    if std::env::var_os("MODGRAD_SKIP_DIAGNOSE").is_none() {
        eprintln!("\n══════ DIAGNOSTIC STUDY: what is the model actually doing? ══════");
        diagnose(&w, &cortex, &train, &eval);
    }

    // Phase 0a: dispatch profile. Activated by env var
    // MODGRAD_PROFILE_DISPATCH=1 — off by default. When on, dumps a
    // per-dispatch-type table after the run.
    eprintln!();
    modgrad_ctm::dispatch_profile::dump();
    let _ = w;
}

/// Write cortex(rgb) features as a flat binary for sklearn comparison:
///
///   header: "FCDS" + n_train (u32) + n_eval (u32) + feat_dim (u32)
///   then  n_train × (feat_dim f32 + label u32)
///   then  n_eval  × (feat_dim f32 + label u32)
fn dump_cortex_features(
    cortex: &VisualCortex,
    train: &[Step], eval: &[Step],
    path: &str,
) {
    use std::io::Write;
    let probe = cortex.encode(&train[0].rgb);
    let feat_dim = probe.tokens.len();
    let mut f = std::fs::File::create(path).expect("create feats file");
    f.write_all(b"FCDS").unwrap();
    f.write_all(&(train.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(eval.len()  as u32).to_le_bytes()).unwrap();
    f.write_all(&(feat_dim     as u32).to_le_bytes()).unwrap();
    let dump = |steps: &[Step], f: &mut std::fs::File| {
        for s in steps {
            let inp = cortex.encode(&s.rgb);
            for v in &inp.tokens {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.write_all(&(s.action as u32).to_le_bytes()).unwrap();
        }
    };
    dump(train, &mut f); dump(eval, &mut f);
    eprintln!("  dumped cortex features → {path}  feat_dim={feat_dim}  ({} + {} samples)", train.len(), eval.len());
}

/// Train and return the trained weights.
///
/// Two paths share the optimizer + loss + accumulator:
///   - With `--features rocm` and `MODGRAD_USE_RESIDENT=1`, dispatches
///     through `forward_cached_resident` + `backward_cached_resident`.
///     Weights are mirrored on-device via `RegionalResidentCache`,
///     re-synced after each `opt.step`. HIP queue is flushed per
///     batch to stay under the ~256-call rocBLAS async limit
///     ([HIP queue overflow] memory).
///   - Otherwise the host path: `forward_cached` + `backward`.
fn study_brain(
    cortex: &VisualCortex,
    train: &[Step], eval: &[Step],
    epochs: usize, batch_size: usize, lr: f32,
) -> RegionalWeights {
    let obs_dim = cortex.token_dim();
    let mut w = build_brain(obs_dim);
    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let loss_fn = StepwiseCE { n_classes: N_ACTIONS, lookahead: 1 };
    let mut idx: Vec<usize> = (0..train.len()).collect();
    let mut rng: u64 = 0xBABE_FACE_BABE_FACE;

    let use_resident = std::env::var_os("MODGRAD_USE_RESIDENT").is_some();
    eprintln!(
        "  brain: 8-region  ticks={TICKS}  obs_dim={obs_dim}  bs={batch_size}  lr={lr:.0e}  path={}",
        if use_resident { "resident" } else { "host" },
    );

    #[cfg(feature = "rocm")]
    if use_resident {
        use modgrad_ctm::resident::{RegionalResidentCache, RegionalGradientsResident};
        use modgrad_device::backend::HipBatch;
        use std::time::{Duration, Instant};
        let mut cache = RegionalResidentCache::from_weights(&w).expect("build cache");
        // Phase 3c: device-resident NLM dW accumulator. Allocated
        // once, zeroed per batch, downloaded into batch_grads via
        // add_to_host before each opt.step. Lifts the 36.7% NLM
        // backward bucket from CPU to GPU.
        let resident_grads = RegionalGradientsResident::from_weights(&w)
            .expect("build resident grads");
        let batch = HipBatch::new();
        // Phase 2 instrumentation — per-section accumulators for the
        // training loop. Find which bucket dominates the 3 s/sample
        // wall-clock (forward profile alone is 260 ms).
        let mut t_encode = Duration::ZERO;
        let mut t_fwd    = Duration::ZERO;
        let mut t_loss   = Duration::ZERO;
        let mut t_bwd    = Duration::ZERO;
        let mut t_accum  = Duration::ZERO;
        let mut t_flush  = Duration::ZERO;
        let mut t_opt    = Duration::ZERO;
        let mut t_sync   = Duration::ZERO;
        let mut t_eval   = Duration::ZERO;
        let mut n_train_samples = 0usize;

        for ep in 0..epochs {
            shuffle_idx(&mut idx, &mut rng);
            let t_ep = std::time::Instant::now();
            let mut total_loss = 0.0f32;
            let mut n_seen = 0usize;
            for chunk in idx.chunks(batch_size) {
                let mut batch_grads = RegionalGradients::zeros(&w);
                let mut batch_loss = 0.0f32;
                // Zero the device-resident NLM dW accumulator before
                // each batch — host batch_grads is already zero, the
                // device buffers need their own reset.
                resident_grads.zero().expect("zero resident grads");
                for &i in chunk {
                    let step = &train[i];

                    let t = Instant::now();
                    let input = cortex.encode(&step.rgb);
                    t_encode += t.elapsed();

                    let t = Instant::now();
                    let fresh = cache.fresh(&w).expect("fresh cache");
                    let (output, _state, fwd_cache) =
                        modgrad_ctm::RegionalBrain::forward_cached_resident(
                            &w, fresh, &batch, &input,
                        ).expect("forward_cached_resident");
                    t_fwd += t.elapsed();

                    let t = Instant::now();
                    let target = [step.action as usize];
                    let (loss, d_preds) = loss_fn.compute(
                        &output.predictions, &output.certainties, &target,
                    );
                    batch_loss += loss;
                    t_loss += t.elapsed();

                    let t = Instant::now();
                    // Phase 3c: route NLM dW through the resident
                    // kernel. The returned sample_grads has zero
                    // nlm_s1_w / nlm_s2_w; resident_grads accumulates
                    // them on device.
                    let sample_grads = modgrad_ctm::RegionalBrain::backward_cached_resident_with_grads_resident(
                        &w, fwd_cache, &d_preds, &resident_grads,
                    );
                    t_bwd += t.elapsed();

                    let t = Instant::now();
                    accumulate(&mut batch_grads, &sample_grads);
                    t_accum += t.elapsed();

                    n_train_samples += 1;
                }
                let t = Instant::now();
                batch.flush().expect("hip flush per batch");
                t_flush += t.elapsed();
                total_loss += batch_loss;
                n_seen += chunk.len();
                // Consolidate device-resident NLM dW into the host
                // batch_grads before opt.step consumes them.
                resident_grads.add_to_host(&mut batch_grads)
                    .expect("add_to_host resident grads");
                let t = Instant::now();
                opt.step(&mut w, &mut batch_grads);
                t_opt += t.elapsed();
                let t = Instant::now();
                cache.sync_from_weights(&w).expect("re-sync cache after opt.step");
                t_sync += t.elapsed();
            }
            let eval_sub: Vec<Step> = eval.iter().take(256).map(|s| Step {
                mission_id: s.mission_id, rgb: s.rgb.clone(), action: s.action,
            }).collect();
            let t = Instant::now();
            let acc = eval_accuracy_silent(&w, cortex, &eval_sub);
            t_eval += t.elapsed();
            eprintln!(
                "  epoch {:>2}/{}  loss={:.3}  eval_acc(256)={:.1}%  {:.1}s",
                ep + 1, epochs, total_loss / n_seen as f32, acc * 100.0,
                t_ep.elapsed().as_secs_f32(),
            );
        }

        // Phase 2 timing dump.
        let bucket = |label: &str, d: Duration, total_ms: f64| {
            let ms = d.as_secs_f64() * 1000.0;
            let pct = if total_ms > 0.0 { 100.0 * ms / total_ms } else { 0.0 };
            let avg = if n_train_samples > 0 { ms / n_train_samples as f64 } else { 0.0 };
            eprintln!("    {:<12}  {:>9.1} ms total  {:>6.2} ms/sample  {:>5.1}%",
                      label, ms, avg, pct);
        };
        let total = t_encode + t_fwd + t_loss + t_bwd + t_accum + t_flush + t_opt + t_sync + t_eval;
        let total_ms = total.as_secs_f64() * 1000.0;
        eprintln!("\n[phase2-profile] training-loop sections (n_train_samples={n_train_samples}):");
        bucket("encode",   t_encode, total_ms);
        bucket("forward",  t_fwd, total_ms);
        bucket("loss",     t_loss, total_ms);
        bucket("backward", t_bwd, total_ms);
        bucket("accum",    t_accum, total_ms);
        bucket("flush",    t_flush, total_ms);
        bucket("opt.step", t_opt, total_ms);
        bucket("cache_sync", t_sync, total_ms);
        bucket("eval",     t_eval, total_ms);
        eprintln!("    {:<12}  {:>9.1} ms total", "TOTAL", total_ms);

        return w;
    }

    // Host path
    for ep in 0..epochs {
        shuffle_idx(&mut idx, &mut rng);
        let t_ep = std::time::Instant::now();
        let mut total_loss = 0.0f32;
        let mut n_seen = 0usize;
        for chunk in idx.chunks(batch_size) {
            let mut batch_grads = RegionalGradients::zeros(&w);
            let mut batch_loss = 0.0f32;
            for &i in chunk {
                let step = &train[i];
                let input = cortex.encode(&step.rgb);
                let state = RegionalBrain::init_state(&w);
                let (output, _state, cache) = RegionalBrain::forward_cached(&w, state, &input);
                let target = [step.action as usize];
                let (loss, d_preds) = loss_fn.compute(
                    &output.predictions, &output.certainties, &target,
                );
                batch_loss += loss;
                let sample_grads = RegionalBrain::backward(&w, cache, &d_preds);
                accumulate(&mut batch_grads, &sample_grads);
            }
            total_loss += batch_loss;
            n_seen += chunk.len();
            opt.step(&mut w, &mut batch_grads);
        }
        let eval_sub: Vec<Step> = eval.iter().take(256).map(|s| Step {
            mission_id: s.mission_id, rgb: s.rgb.clone(), action: s.action,
        }).collect();
        let acc = eval_accuracy_silent(&w, cortex, &eval_sub);
        eprintln!(
            "  epoch {:>2}/{}  loss={:.3}  eval_acc(256)={:.1}%  {:.1}s",
            ep + 1, epochs, total_loss / n_seen as f32, acc * 100.0,
            t_ep.elapsed().as_secs_f32(),
        );
    }
    w
}

fn eval_accuracy_silent(w: &RegionalWeights, cortex: &VisualCortex, steps: &[Step]) -> f32 {
    let mut correct = 0usize;
    for step in steps {
        let input = cortex.encode(&step.rgb);
        let state = RegionalBrain::init_state(w);
        let (output, _state) = RegionalBrain::forward(w, state, &input);
        if let Some(last) = output.predictions.last() {
            if argmax(last) == step.action as usize { correct += 1; }
        }
    }
    correct as f32 / steps.len() as f32
}

/// Five focused diagnostics on the trained brain.
fn diagnose(w: &RegionalWeights, cortex: &VisualCortex, train: &[Step], eval: &[Step]) {
    let names = ["L", "R", "F", "pickup", "drop", "toggle", "done"];

    // ── 1. Per-tick predictions on 5 train samples ──────────────
    eprintln!("\n[1] per-tick logits/argmax on 5 train samples — does the model 'think' or freeze?");
    for i in 0..5usize.min(train.len()) {
        let s = &train[i];
        let input = cortex.encode(&s.rgb);
        let state = RegionalBrain::init_state(w);
        let (output, _) = RegionalBrain::forward(w, state, &input);
        eprint!("  sample {} (true={}): per-tick argmax = [", i, names[s.action as usize]);
        for (t, p) in output.predictions.iter().enumerate() {
            let a = argmax(p);
            eprint!("{}{}", names[a], if t + 1 < output.predictions.len() { ", " } else { "" });
        }
        let last = output.predictions.last().unwrap();
        let mn = last.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let spread = mx - mn;
        eprintln!("]  last logits range=[{mn:.2}, {mx:.2}] spread={spread:.2}");
        if i == 0 {
            // Detailed last-tick logits for first sample.
            eprint!("    last-tick logits: ");
            for k in 0..N_ACTIONS {
                eprint!("{}={:+.2} ", names[k], last[k]);
            }
            eprintln!();
        }
    }

    // ── 2. Does input matter? Compare predictions on real vs zero vs random input ──
    eprintln!("\n[2] does the input matter? (compare predictions on real / zero / random / shuffled)");
    let s0 = &train[0];
    let real_input = cortex.encode(&s0.rgb);
    let token_dim = real_input.token_dim;
    let n_tokens  = real_input.n_tokens;
    let zero_input = TokenInput {
        tokens: vec![0.0; n_tokens * token_dim], n_tokens, token_dim,
    };
    let mut rand_rng: u64 = 0xDEAD_BEEF;
    let mut rand_tokens = vec![0.0f32; n_tokens * token_dim];
    for v in rand_tokens.iter_mut() {
        rand_rng = rand_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (((rand_rng >> 11) as u32) as f32 / (1u32 << 21) as f32 - 1.0) * 0.2;
    }
    let rand_input = TokenInput { tokens: rand_tokens, n_tokens, token_dim };

    let pred_for = |inp: &TokenInput| -> Vec<f32> {
        let st = RegionalBrain::init_state(w);
        let (out, _) = RegionalBrain::forward(w, st, inp);
        out.predictions.last().unwrap().clone()
    };
    let p_real = pred_for(&real_input);
    let p_zero = pred_for(&zero_input);
    let p_rand = pred_for(&rand_input);

    let l2 = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    };
    eprintln!("  real → argmax={}  logits[done]={:+.2} logits[F]={:+.2}",
              names[argmax(&p_real)], p_real[6], p_real[2]);
    eprintln!("  zero → argmax={}  logits[done]={:+.2} logits[F]={:+.2}",
              names[argmax(&p_zero)], p_zero[6], p_zero[2]);
    eprintln!("  rand → argmax={}  logits[done]={:+.2} logits[F]={:+.2}",
              names[argmax(&p_rand)], p_rand[6], p_rand[2]);
    eprintln!("  L2(real, zero) = {:.3}    L2(real, rand) = {:.3}", l2(&p_real, &p_zero), l2(&p_real, &p_rand));
    eprintln!("  L2(zero, rand) = {:.3}", l2(&p_zero, &p_rand));

    // ── 3. Output projection weight magnitude per class ──
    eprintln!("\n[3] output projection bias per class — is one class systematically favored?");
    // output_proj is a Linear projecting [n_global_sync] → [out_dims].
    // Read its weight rows (one per output class).
    let op = &w.output_proj;
    let in_dim = op.in_dim;
    let out_dim = op.out_dim;
    eprintln!("  output_proj: {in_dim} → {out_dim}");
    for k in 0..out_dim.min(N_ACTIONS) {
        let row = &op.weight[k * in_dim..(k + 1) * in_dim];
        let row_mean = row.iter().sum::<f32>() / row.len() as f32;
        let row_l1 = row.iter().map(|v| v.abs()).sum::<f32>() / row.len() as f32;
        let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let row_min = row.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("    class {} ({:>7}): bias={:+.3}  weight mean={:+.3} |w|_avg={:.3} range=[{:+.2}, {:+.2}]",
                  k, names[k], op.bias[k], row_mean, row_l1, row_min, row_max);
    }

    // ── 4. Cortex output statistics across samples ──
    eprintln!("\n[4] cortex output stats across 50 samples (variance across samples = useful signal)");
    let mut feats: Vec<Vec<f32>> = Vec::new();
    for i in 0..50.min(train.len()) {
        let inp = cortex.encode(&train[i].rgb);
        feats.push(inp.tokens);
    }
    let dim = feats[0].len();
    // Per-feature std across samples.
    let mut means = vec![0.0f32; dim];
    for f in &feats { for j in 0..dim { means[j] += f[j]; } }
    for j in 0..dim { means[j] /= feats.len() as f32; }
    let mut vars = vec![0.0f32; dim];
    for f in &feats { for j in 0..dim { vars[j] += (f[j] - means[j]).powi(2); } }
    for j in 0..dim { vars[j] /= feats.len() as f32; }
    let std_mean = vars.iter().map(|v| v.sqrt()).sum::<f32>() / dim as f32;
    let std_max  = vars.iter().map(|v| v.sqrt()).fold(f32::NEG_INFINITY, f32::max);
    let std_min  = vars.iter().map(|v| v.sqrt()).fold(f32::INFINITY, f32::min);
    eprintln!("  per-feature std across 50 samples: mean={std_mean:.4} min={std_min:.4} max={std_max:.4}");
    eprintln!("  → if mean std is near zero, the cortex output is nearly identical across samples");

    // ── 5. Eval prediction histogram + accuracy by class ──
    eprintln!("\n[5] per-class eval performance (where does the model get things right/wrong?)");
    let mut by_class_total = [0usize; N_ACTIONS];
    let mut by_class_correct = [0usize; N_ACTIONS];
    let mut pred_hist = [0usize; N_ACTIONS];
    let n_eval = eval.len().min(400);
    for s in eval.iter().take(n_eval) {
        let inp = cortex.encode(&s.rgb);
        let st = RegionalBrain::init_state(w);
        let (out, _) = RegionalBrain::forward(w, st, &inp);
        let p = out.predictions.last().unwrap();
        let pred = argmax(p);
        pred_hist[pred] += 1;
        by_class_total[s.action as usize] += 1;
        if pred == s.action as usize { by_class_correct[s.action as usize] += 1; }
    }
    eprintln!("  pred_hist (over {n_eval} eval): {pred_hist:?}");
    for k in 0..N_ACTIONS {
        if by_class_total[k] > 0 {
            let acc = 100.0 * by_class_correct[k] as f32 / by_class_total[k] as f32;
            eprintln!("    true={:<7} n={:>3}  correct={:>3}  acc={:5.1}%",
                      names[k], by_class_total[k], by_class_correct[k], acc);
        }
    }
}
