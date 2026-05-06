//! Homeostatic market-making brain.
//!
//! A regional CTM that picks a quoting policy (spread × skew) for ZECBTC,
//! trained online via REINFORCE against realized paper PnL. The brain has
//! an interoceptive feedback loop: its own performance drives `pain` and
//! `satisfaction` scalars that re-enter as observation features on the next
//! tick. Losing => pain rises => obs shifts => behavior adapts.
//!
//! The market is synthetic for now (random walk with mean-reverting depth/flow
//! imbalance). The action set is 9 arms = {tight, mid, wide} × {long, neutral, short}.
//!
//! Usage:
//!   zec_mm_homeostatic                       # 5000 steps, default config
//!   zec_mm_homeostatic --steps 50000         # longer run
//!   zec_mm_homeostatic --no-pain             # ablate interoception (zeros it)
//!   zec_mm_homeostatic --seed 42

use modgrad_ctm::graph::*;

const N_ARMS: usize = 9;
const SPREADS_BPS: [f32; 3] = [5.0, 15.0, 30.0];
const SKEWS: [f32; 3] = [-1.0, 0.0, 1.0];

const OBS_DIM: usize = 8;
// [mid_z, dmid_z, depth_imb, flow_imb, inv_norm, pnl_norm, pain, satisfaction]

const QUOTE_SIZE: f32 = 1.0;
const INV_CAP: f32 = 5.0;
// Sized to typical paper-PnL fluctuation range, not the underlying notional.
// With INV_CAP * mid ≈ 5 * 0.005 = 0.025 BTC of position, per-tick PnL swings
// are O(1e-4) and full-rally PnL O(1e-2). Setting CAPITAL = 1e-2 makes
// drawdown / capital land in [0, ~1] so pain is actually informative.
const CAPITAL: f32 = 0.01;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut steps: usize = 5_000;
    let mut seed: u64 = 0;
    let mut lr: f32 = 1e-3;
    let mut embed_dim: usize = 32;
    let mut ticks: usize = 4;
    let mut log_every: usize = 100;
    let mut no_pain = false;
    let mut save_path = "zec_mm.bin".to_string();
    let mut load = false;
    let mut replay_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { steps = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--embed" => { embed_dim = args[i + 1].parse().unwrap(); i += 2; }
            "--ticks" => { ticks = args[i + 1].parse().unwrap(); i += 2; }
            "--log-every" => { log_every = args[i + 1].parse().unwrap(); i += 2; }
            "--no-pain" => { no_pain = true; i += 1; }
            "--save" => { save_path = args[i + 1].clone(); i += 2; }
            "--load" => { load = true; i += 1; }
            "--replay" => { replay_path = Some(args[i + 1].clone()); i += 2; }
            "--help" | "-h" => { print_help(); return; }
            _ => { i += 1; }
        }
    }

    eprintln!(
        "zec_mm_homeostatic — steps={steps} seed={seed} lr={lr} \
        embed={embed_dim} ticks={ticks} pain={}",
        if no_pain { "off" } else { "on" }
    );

    // Build brain
    let mut cfg = RegionalConfig::four_region(OBS_DIM, N_ARMS, ticks);
    cfg.raw_obs_dim = OBS_DIM;
    cfg.obs_scale_dims = vec![OBS_DIM];
    cfg.out_dims = N_ARMS;
    let _ = embed_dim; // four_region embeds inputs at obs_dim; embed_dim ignored for now

    let mut w = if load && std::path::Path::new(&save_path).exists() {
        eprintln!("loading {save_path}");
        RegionalWeights::load(&save_path).expect("load failed")
    } else {
        RegionalWeights::new(cfg)
    };
    let mut opt = RegionalAdamW::new(&w).with_lr(lr).with_clip(5.0);
    let mut grads = RegionalGradients::zeros(&w);
    w.print_summary();

    // Market & book
    let mut market: Box<dyn TickStream> = match &replay_path {
        Some(p) => {
            let m = ReplayMarket::load(p).expect("replay load failed");
            eprintln!("replay: {} ticks from {p}", m.len());
            Box::new(m)
        }
        None => Box::new(SyntheticMarket::new(seed)),
    };
    let mut book = PaperBook::new();
    let mut intero = Interoception::new();

    let mut prev_mid: f32 = 0.0;
    let mut mid_ema: f32 = 0.0;
    let mut mid_var: f32 = 1e-6;

    let mut pnl_log: Vec<f32> = Vec::with_capacity(steps);
    let mut arm_picks = [0usize; N_ARMS];

    for step in 0..steps {
        let tick = market.next();

        // Online normalization for mid
        let alpha = 0.001f32;
        if step == 0 {
            mid_ema = tick.mid;
            mid_var = (tick.mid * 1e-3).powi(2).max(1e-12);
        } else {
            let d = tick.mid - mid_ema;
            mid_ema += alpha * d;
            mid_var = (1.0 - alpha) * mid_var + alpha * d * d;
        }
        let mid_std = mid_var.sqrt().max(1e-9);
        let mid_z = (tick.mid - mid_ema) / mid_std;
        let dmid_z = if step == 0 { 0.0 } else { (tick.mid - prev_mid) / mid_std };

        let inv_norm = book.inv / INV_CAP;
        let pnl_norm = (book.equity(tick.mid) / CAPITAL).clamp(-1.0, 1.0);
        let (pain, satisfaction) = if no_pain { (0.0, 0.0) } else { intero.read() };

        let obs: [f32; OBS_DIM] = [
            mid_z.clamp(-3.0, 3.0),
            dmid_z.clamp(-3.0, 3.0),
            tick.depth_imb,
            tick.flow_imb,
            inv_norm,
            pnl_norm,
            pain,
            satisfaction,
        ];

        // Forward+train: REINFORCE-style closure. We do not yet know reward,
        // so first do a forward to pick action, then run the book, then do
        // the train step with the realized reward as target.
        // Forward via the same generic-step mechanism: pass reward=0 to get
        // logits; we re-run with the real reward after stepping the book.
        // (Cheap because regional_train_step_generic re-does the forward;
        // workspace optimization is a future TODO.)
        let logits = forward_logits(&w, &obs);
        let action = sample_action(&logits, step as u64 + seed);
        arm_picks[action] += 1;

        let (sp_bps, skew) = arm_params(action);
        let pnl_before = book.equity(tick.mid);
        book.quote_and_step(tick.mid, sp_bps, skew, &tick);
        let pnl_after = book.equity(tick.mid);
        let reward = pnl_after - pnl_before; // realized PnL change this tick

        intero.update(reward, book.equity(tick.mid));

        // Now do the gradient step with the actual reward
        grads.zero();
        let r = reward;
        let chosen = action;
        let (loss, _d_obs) = regional_train_step_generic(
            &w,
            &mut grads,
            &obs,
            move |preds: &[Vec<f32>], _certs: &[[f32; 2]]| {
                reinforce_grad(preds, chosen, r)
            },
        );
        opt.step(&mut w, &mut grads);

        pnl_log.push(reward);
        prev_mid = tick.mid;

        if step % log_every == 0 || step == steps - 1 {
            let recent = pnl_log.len().min(log_every);
            let recent_pnl: f32 = pnl_log[pnl_log.len() - recent..].iter().sum();
            let total: f32 = pnl_log.iter().sum();
            eprintln!(
                "step {step:6}  mid={:.6}  inv={:+.2}  recent_pnl={:+.6} total={:+.6} \
                action=arm{action}(sp{:.0},sk{:+.0}) pain={:.2} sat={:.2} loss={:.4}",
                tick.mid, book.inv, recent_pnl, total, sp_bps, skew, pain, satisfaction, loss
            );
        }
    }

    // Final report
    eprintln!("\n=== Final ===");
    eprintln!("total_pnl: {:+.6} BTC ({:.2}% of capital)",
        book.equity(prev_mid),
        100.0 * book.equity(prev_mid) / CAPITAL);
    eprintln!("inventory: {:+.2} ZEC", book.inv);
    eprintln!("fills:     {}", book.fills);
    eprintln!("peak:      {:+.6}", intero.peak);
    eprintln!("max_dd:    {:+.6}", intero.max_dd);
    eprintln!("\nArm pick distribution:");
    for a in 0..N_ARMS {
        let (sp, sk) = arm_params(a);
        let pct = 100.0 * arm_picks[a] as f32 / steps as f32;
        eprintln!("  arm {a} sp={sp:>4.0} sk={sk:+.0}: {:>6} ({:>5.1}%)", arm_picks[a], pct);
    }

    if !load {
        w.save(&save_path).expect("save failed");
        eprintln!("saved {save_path}");
    }
}

// ── Brain helpers ──────────────────────────────────────────────────

/// Forward pass that returns the last-tick logits over actions.
/// Implementation note: we use regional_forward (no gradient) by routing
/// through the train step with a reward of zero — the train step's
/// closure receives the predictions, which we copy out before discarding
/// the (zero) gradient.
fn forward_logits(w: &RegionalWeights, obs: &[f32]) -> Vec<f32> {
    let mut tmp = RegionalGradients::zeros(w);
    let mut captured: Vec<f32> = Vec::new();
    let cap_ref = std::cell::RefCell::new(&mut captured);
    let _ = regional_train_step_generic(
        w,
        &mut tmp,
        obs,
        |preds: &[Vec<f32>], _certs: &[[f32; 2]]| {
            if let Some(last) = preds.last() {
                let mut b = cap_ref.borrow_mut();
                **b = last.clone();
            }
            // zero loss, zero gradients => no weight update needed
            (0.0, vec![vec![0.0; preds[0].len()]; preds.len()])
        },
    );
    captured
}

/// REINFORCE gradient: -reward * d(log softmax[chosen])/d(logits)
/// Returns (loss, d_predictions per tick) with grad only on the last tick.
fn reinforce_grad(
    preds: &[Vec<f32>],
    chosen: usize,
    reward: f32,
) -> (f32, Vec<Vec<f32>>) {
    let k = preds.len();
    if k == 0 { return (0.0, Vec::new()); }
    let last = &preds[k - 1];
    let n = last.len();

    // softmax with stable subtraction
    let max_l = last.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = last.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let softmax: Vec<f32> = exp_s.iter().map(|&e| e / sum).collect();

    let log_p = softmax[chosen].max(1e-10).ln();
    let loss = -reward * log_p;

    // grad on logits = -reward * (1[i==chosen] - softmax[i])
    let grad: Vec<f32> = (0..n).map(|i| {
        let ind = if i == chosen { 1.0 } else { 0.0 };
        -reward * (ind - softmax[i])
    }).collect();

    let mut d_preds = vec![vec![0.0f32; n]; k];
    d_preds[k - 1] = grad;
    (loss, d_preds)
}

/// Sample an action from softmax logits. Deterministic per (logits, seed).
fn sample_action(logits: &[f32], seed: u64) -> usize {
    if logits.is_empty() { return 0; }
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_s: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp_s.iter().sum();
    let mut u = lcg(seed) as f32 / u64::MAX as f32; // [0,1)
    for (i, &e) in exp_s.iter().enumerate() {
        let p = e / sum;
        if u < p { return i; }
        u -= p;
    }
    logits.len() - 1
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn arm_params(arm: usize) -> (f32, f32) {
    let sp_idx = arm / 3;
    let sk_idx = arm % 3;
    (SPREADS_BPS[sp_idx], SKEWS[sk_idx])
}

// ── Tick streams ───────────────────────────────────────────────────

struct Tick {
    mid: f32,
    depth_imb: f32, // [-1, 1]
    flow_imb: f32,  // [-1, 1]
}

trait TickStream {
    fn next(&mut self) -> Tick;
}

struct SyntheticMarket {
    state: u64,
    mid: f32,
    depth_imb: f32,
    flow_imb: f32,
}

impl SyntheticMarket {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1), mid: 0.005, depth_imb: 0.0, flow_imb: 0.0 }
    }

    fn rand_unit(&mut self) -> f32 {
        self.state = lcg(self.state);
        (self.state as f64 / u64::MAX as f64) as f32
    }

    fn rand_normal(&mut self) -> f32 {
        let u1 = self.rand_unit().max(1e-9);
        let u2 = self.rand_unit();
        ((-2.0 * u1.ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

impl TickStream for SyntheticMarket {
    fn next(&mut self) -> Tick {
        self.mid *= (1.0 + 0.0001 * self.rand_normal()).max(0.5);
        self.depth_imb = (0.95 * self.depth_imb + 0.1 * self.rand_normal()).clamp(-1.0, 1.0);
        self.flow_imb = (0.85 * self.flow_imb + 0.3 * self.rand_normal()).clamp(-1.0, 1.0);
        Tick { mid: self.mid, depth_imb: self.depth_imb, flow_imb: self.flow_imb }
    }
}

/// Reads a flat f32 file of (mid, depth_imb, flow_imb) triples produced by
/// scripts/export_ticks.py. Loops on overrun.
struct ReplayMarket {
    data: Vec<f32>, // flat, len % 3 == 0
    cur: usize,
}

impl ReplayMarket {
    fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() % 12 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file size {} is not a multiple of 12 bytes (3×f32)", bytes.len()),
            ));
        }
        let n = bytes.len() / 4;
        let mut data = vec![0f32; n];
        for i in 0..n {
            let off = i * 4;
            data[i] = f32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]);
        }
        Ok(Self { data, cur: 0 })
    }

    fn len(&self) -> usize { self.data.len() / 3 }
}

impl TickStream for ReplayMarket {
    fn next(&mut self) -> Tick {
        let n = self.len();
        if n == 0 {
            return Tick { mid: 0.005, depth_imb: 0.0, flow_imb: 0.0 };
        }
        let i = self.cur % n;
        self.cur = self.cur.wrapping_add(1);
        let off = i * 3;
        Tick {
            mid: self.data[off],
            depth_imb: self.data[off + 1],
            flow_imb: self.data[off + 2],
        }
    }
}

// ── Paper book ─────────────────────────────────────────────────────

struct PaperBook {
    inv: f32,
    cash: f32,
    fills: usize,
}

impl PaperBook {
    fn new() -> Self { Self { inv: 0.0, cash: 0.0, fills: 0 } }

    fn equity(&self, mid: f32) -> f32 {
        self.cash + self.inv * mid
    }

    /// Place quotes at (mid - h - sk*h/2, mid + h - sk*h/2) where h = sp_bps * mid / 1e4.
    /// Use the tick's flow_imb as a proxy for taker direction & quote-cross probability.
    /// Positive flow_imb => taker buying => our ask may be lifted.
    /// Negative flow_imb => taker selling => our bid may be hit.
    fn quote_and_step(&mut self, mid: f32, sp_bps: f32, skew: f32, tick: &Tick) {
        let h = mid * sp_bps / 1e4;
        let sk = mid * sp_bps / 2.0 / 1e4 * skew;
        let bid = mid - h - sk;
        let ask = mid + h - sk;

        // Probability that our quote is filled this tick, scaled by |flow_imb|
        // and inversely by spread (tighter quote => higher fill probability).
        let intensity = (tick.flow_imb.abs()).clamp(0.0, 1.0);
        let tightness = (15.0 / sp_bps).clamp(0.1, 3.0);
        let p_fill = (intensity * tightness * 0.3).clamp(0.0, 0.95);

        // Hash the quote prices to a deterministic random for repeatability
        let h_state = lcg(((bid * 1e9) as u64).wrapping_add((ask * 1e9) as u64));
        let r = h_state as f64 / u64::MAX as f64;

        if tick.flow_imb < 0.0 && self.inv < INV_CAP {
            // takers selling => our bid hit
            if r < p_fill as f64 {
                self.inv += QUOTE_SIZE;
                self.cash -= bid * QUOTE_SIZE;
                self.fills += 1;
            }
        } else if tick.flow_imb > 0.0 && self.inv > -INV_CAP {
            // takers buying => our ask lifted
            if r < p_fill as f64 {
                self.inv -= QUOTE_SIZE;
                self.cash += ask * QUOTE_SIZE;
                self.fills += 1;
            }
        }
    }
}

// ── Interoception ──────────────────────────────────────────────────

struct Interoception {
    pain: f32,
    satisfaction: f32,
    peak: f32,
    max_dd: f32,
    pnl_ema: f32,
}

impl Interoception {
    fn new() -> Self {
        Self { pain: 0.0, satisfaction: 0.0, peak: 0.0, max_dd: 0.0, pnl_ema: 0.0 }
    }

    fn update(&mut self, reward: f32, equity: f32) {
        if equity > self.peak { self.peak = equity; }
        let dd = (self.peak - equity).max(0.0);
        if dd > self.max_dd { self.max_dd = dd; }

        // pain: rises with current drawdown relative to capital, decays over time
        let dd_norm = (dd / CAPITAL).clamp(0.0, 1.0);
        self.pain = 0.9 * self.pain + 0.1 * dd_norm;

        // satisfaction: EMA of step PnL, clipped to [0, 1] of capital scale
        self.pnl_ema = 0.99 * self.pnl_ema + 0.01 * reward;
        let s = (self.pnl_ema * 1000.0 / CAPITAL).clamp(0.0, 1.0);
        self.satisfaction = 0.95 * self.satisfaction + 0.05 * s;
    }

    fn read(&self) -> (f32, f32) { (self.pain, self.satisfaction) }
}

fn print_help() {
    eprintln!("zec_mm_homeostatic — homeostatic MM brain on synthetic ZECBTC");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  zec_mm_homeostatic [--steps N] [--seed N] [--lr F]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --steps N        training steps (default 5000)");
    eprintln!("  --seed N         RNG seed for synthetic market (default 0)");
    eprintln!("  --lr F           learning rate (default 1e-3)");
    eprintln!("  --embed N        embedding dim (default 32; ignored for now)");
    eprintln!("  --ticks N        inner CTM ticks (default 4)");
    eprintln!("  --log-every N    log every N steps (default 100)");
    eprintln!("  --no-pain        zero out pain/satisfaction (ablation)");
    eprintln!("  --save PATH      checkpoint path (default zec_mm.bin)");
    eprintln!("  --load           load checkpoint if it exists");
    eprintln!("  --replay PATH    load f32-triple binary tick stream (overrides synthetic market)");
}
