//! Multi-agent arena — N CTM market makers compete on a shared
//! synthetic Penumbra replay. Pipeline: pretrain each agent on its
//! own seed → eval all on a shared replay → rank → PBT-mutate the
//! bottom quartile → repeat. Optional Sakana-style cooperation
//! (`COOP_ALPHA`) and diversity (`DIVERSITY_BETA`) terms.

use anyhow::Result;

use modgrad_ctm::graph::{
    RegionalAdamWTyped, RegionalGradientsTyped,
    RegionalWeights, RegionalWeightsTyped,
};
use modgrad_device::backend::tensor::Cpu;

use penumbra_arena::agent::RegionState;
use penumbra_arena::arena::Leaderboard;
use penumbra_arena::chain::PositionParams;
use penumbra_arena::default_mm_cfg;
use penumbra_arena::features::{BlockSnapshot, FeatureEncoder};
use penumbra_arena::motor::{BarbellConfig, BarbellDecoder};
use penumbra_arena::replay::{SyntheticReplay, SyntheticReplayConfig, oracle_motor};
use penumbra_arena::reward::{RewardWeights, simulate_fills, step_reward_from_fills};

struct ArenaAgent {
    account_index: u32,
    weights: RegionalWeightsTyped<Cpu>,
    optimizer: RegionalAdamWTyped<Cpu>,
    encoder: FeatureEncoder,
    motor: BarbellDecoder,
    cumulative_reward: f64,
    total_fills: u64,
    arb_flips: u64,
    /// Stdev of per-seed rewards (0 when `EVAL_SEEDS=1`).
    reward_stdev: f64,
    /// Pearson(book_skew, per-block flip count). >0 means brain
    /// selectively flips when book is skewed; ≈0 means it doesn't
    /// use the signal; NaN when flip variance is zero.
    skew_flip_corr: f64,
    state: RegionState,
    /// Held only to seed `RegionState::reset_in_place` between phases.
    untyped_ref: RegionalWeights,
}

impl ArenaAgent {
    fn new(account_index: u32, untyped: RegionalWeights, lr: f32) -> Result<Self> {
        let weights = RegionalWeightsTyped::<Cpu>::from_untyped(&untyped)
            .map_err(|e| anyhow::anyhow!("from_untyped: {e}"))?;
        let optimizer = RegionalAdamWTyped::<Cpu>::new(&weights)
            .map_err(|e| anyhow::anyhow!("RegionalAdamWTyped::new: {e}"))?
            .with_lr(lr).with_clip(1.0);
        let state = RegionState::from_weights(&untyped);
        Ok(Self {
            account_index,
            weights, optimizer,
            encoder: FeatureEncoder::new(),
            motor: BarbellDecoder::new(BarbellConfig::default()),
            cumulative_reward: 0.0,
            total_fills: 0,
            arb_flips: 0,
            reward_stdev: 0.0,
            skew_flip_corr: f64::NAN,
            state,
            untyped_ref: untyped,
        })
    }

    fn reset_region_state(&mut self) {
        self.state.reset_in_place(&self.untyped_ref);
    }

    /// One imitation training step against `oracle_motor`. Region state
    /// is persistent across calls within an epoch so the CTM trains in
    /// the same regime it'll be evaluated in. Caller `pretrain_one`
    /// resets state at epoch boundaries.
    fn imitation_step(
        &mut self, replay: &mut SyntheticReplay, batch_grads: &mut RegionalGradientsTyped<Cpu>,
    ) -> Result<f64> {
        let bf = replay.step();
        let obs = self.encoder.encode(&bf);
        let (out, cache) = self.weights.regional_forward_typed_with_cache(
            &obs,
            &mut self.state.region_activated, &mut self.state.region_trace,
            &mut self.state.global_alpha,     &mut self.state.global_beta,
        ).map_err(|e| anyhow::anyhow!("forward: {e}"))?;

        let target = oracle_motor(&bf);
        let last_tick = out.predictions.len() - 1;
        let pred = &out.predictions[last_tick];
        let mse: f64 = pred.iter().zip(&target)
            .map(|(p, t)| ((p - t) as f64).powi(2)).sum::<f64>() / pred.len() as f64;

        let mut d_preds: Vec<Vec<f32>> = (0..out.predictions.len())
            .map(|_| vec![0.0f32; pred.len()]).collect();
        for i in 0..pred.len() {
            d_preds[last_tick][i] = 2.0 * (pred[i] - target[i]) / pred.len() as f32;
        }
        self.weights.regional_backward_typed(&cache, &d_preds, batch_grads)
            .map_err(|e| anyhow::anyhow!("backward: {e}"))?;

        self.encoder.push(BlockSnapshot {
            mid: bf.quote.mid,
            spread_bps: bf.quote.spread_bps(),
            own_fills_in_block: 0.0,
        });
        Ok(mse)
    }

    /// Decode positions for a given block (no chain submit). Region
    /// state mutates in place across calls — caller decides when to
    /// reset (typically once before each eval phase).
    fn decide(
        &mut self, bf: &penumbra_arena::chain::BlockFeatures,
    ) -> Result<Vec<PositionParams>> {
        let obs = self.encoder.encode(bf);
        let out = self.weights.regional_forward_typed(
            &obs,
            &mut self.state.region_activated, &mut self.state.region_trace,
            &mut self.state.global_alpha,     &mut self.state.global_beta,
        ).map_err(|e| anyhow::anyhow!("forward: {e}"))?;

        let last_tick = out.predictions.len() - 1;
        let positions = self.motor.decode(&out.predictions[last_tick], bf.quote.mid, bf.balance.base);
        Ok(positions)
    }
}

fn pretrain_one(
    agent: &mut ArenaAgent, seed: u64, pretrain_steps: usize, epochs: usize, bs: usize,
    replay_cfg: &SyntheticReplayConfig,
) -> Result<f64> {
    let mut batch_grads = RegionalGradientsTyped::<Cpu>::zeros(&agent.weights)
        .map_err(|e| anyhow::anyhow!("grads zero: {e}"))?;
    let mut last_epoch_mse = 0.0f64;
    // Optimizer moments persist across epochs; region state + encoder don't.
    for epoch in 0..epochs.max(1) {
        agent.reset_region_state();
        agent.encoder = FeatureEncoder::new();
        let mut replay = SyntheticReplay::new(
            replay_cfg.clone(),
            seed.wrapping_add(epoch as u64),
        );
        let mut epoch_mse = 0.0f64;
        for step_i in 0..pretrain_steps {
            let mse = agent.imitation_step(&mut replay, &mut batch_grads)?;
            epoch_mse += mse;
            if (step_i + 1) % bs == 0 {
                agent.optimizer.step(&mut agent.weights, &mut batch_grads)
                    .map_err(|e| anyhow::anyhow!("opt.step: {e}"))?;
                batch_grads.zero().map_err(|e| anyhow::anyhow!("grads zero: {e}"))?;
            }
        }
        last_epoch_mse = epoch_mse / pretrain_steps.max(1) as f64;
    }
    Ok(last_epoch_mse)
}

fn evaluate_all(
    agents: &mut [ArenaAgent], eval_blocks: usize, eval_seed: u64,
    replay_cfg: &SyntheticReplayConfig,
) -> Result<EvalSamples> {
    let mut eval_replay = SyntheticReplay::new(replay_cfg.clone(), eval_seed);
    for agent in agents.iter_mut() {
        agent.encoder = FeatureEncoder::new();
        agent.cumulative_reward = 0.0;
        agent.total_fills = 0;
        agent.arb_flips = 0;
        agent.reset_region_state();
    }

    let reward_weights = RewardWeights::default();
    let mut blocks = Vec::with_capacity(eval_blocks + 2);
    for _ in 0..(eval_blocks + 2) { blocks.push(eval_replay.step()); }

    let mut samples = EvalSamples {
        mid: Vec::with_capacity(eval_blocks),
        skew: Vec::with_capacity(eval_blocks),
        flips_per_agent: vec![Vec::with_capacity(eval_blocks); agents.len()],
        cum_reward_per_agent: vec![Vec::with_capacity(eval_blocks); agents.len()],
    };
    for (block_i, bf) in blocks.iter().enumerate().take(eval_blocks) {
        let placed_mid = bf.quote.mid;
        let fill_mid = blocks[block_i + 1].quote.mid;
        let post_fill_mid = blocks[block_i + 2].quote.mid;
        let bid = bf.book.bid_depth_within(50.0);
        let ask = bf.book.ask_depth_within(50.0);
        let skew = if bid + ask > 1e-9 { (bid - ask) / (bid + ask) } else { 0.0 };
        samples.mid.push(placed_mid);
        samples.skew.push(skew);
        for (i, agent) in agents.iter_mut().enumerate() {
            let positions = agent.decide(bf)?;
            let n_pos = positions.len();
            let flipped = positions.iter()
                .filter(|p| p.is_arb_flip(placed_mid)).count() as u64;
            agent.arb_flips += flipped;
            samples.flips_per_agent[i].push(flipped as f64);
            let fills = simulate_fills(&positions, placed_mid, fill_mid, post_fill_mid);
            let n_filled = fills.iter().filter(|f| f.filled).count() as u64;
            agent.total_fills += n_filled;
            let r = step_reward_from_fills(&fills, n_pos, &reward_weights);
            agent.cumulative_reward += r;
            samples.cum_reward_per_agent[i].push(agent.cumulative_reward);
            agent.encoder.push(BlockSnapshot {
                mid: bf.quote.mid,
                spread_bps: bf.quote.spread_bps(),
                own_fills_in_block: n_filled as f64,
            });
        }
    }
    Ok(samples)
}

/// Per-block traces from one eval pass.
struct EvalSamples {
    mid: Vec<f64>,
    /// Book skew at offset 50 bps, in [−1, 1].
    skew: Vec<f64>,
    /// `flips_per_agent[i][block]`.
    flips_per_agent: Vec<Vec<f64>>,
    /// `cum_reward_per_agent[i][block]` — running total since eval start.
    cum_reward_per_agent: Vec<Vec<f64>>,
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 { return f64::NAN; }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0; let mut dx2 = 0.0; let mut dy2 = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let dx = x - mx; let dy = y - my;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-12 { f64::NAN } else { num / denom }
}

/// Run `evaluate_all` `n_seeds` times, capture per-seed reward per
/// agent, and write the mean back into `cumulative_reward` and the
/// across-seed stdev into `reward_stdev`. arb_flips and total_fills
/// accumulate across seeds (so the totals are still informative).
fn evaluate_all_multi_seed(
    agents: &mut [ArenaAgent], eval_blocks: usize,
    base_seed: u64, n_seeds: usize, replay_cfg: &SyntheticReplayConfig,
) -> Result<Option<EvalSamples>> {
    let n_seeds = n_seeds.max(1);
    let n_agents = agents.len();
    let mut per_seed_rewards: Vec<Vec<f64>> = vec![Vec::with_capacity(n_seeds); n_agents];
    let mut total_fills = vec![0u64; n_agents];
    let mut total_flips = vec![0u64; n_agents];
    let mut all_skew: Vec<f64> = Vec::with_capacity(n_seeds * eval_blocks);
    let mut all_flips_per_agent: Vec<Vec<f64>> = vec![
        Vec::with_capacity(n_seeds * eval_blocks); n_agents];
    let mut first_seed_samples: Option<EvalSamples> = None;
    for s in 0..n_seeds {
        let seed = base_seed.wrapping_add(s as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let samples = evaluate_all(agents, eval_blocks, seed, replay_cfg)?;
        for (i, a) in agents.iter().enumerate() {
            per_seed_rewards[i].push(a.cumulative_reward);
            total_fills[i] += a.total_fills;
            total_flips[i] += a.arb_flips;
        }
        all_skew.extend_from_slice(&samples.skew);
        for (i, fl) in samples.flips_per_agent.iter().enumerate() {
            all_flips_per_agent[i].extend_from_slice(fl);
        }
        if first_seed_samples.is_none() { first_seed_samples = Some(samples); }
    }
    for (i, a) in agents.iter_mut().enumerate() {
        let xs = &per_seed_rewards[i];
        let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
        let var: f64 = if xs.len() > 1 {
            xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
        } else { 0.0 };
        a.cumulative_reward = mean;
        a.reward_stdev = var.sqrt();
        a.total_fills = total_fills[i];
        a.arb_flips   = total_flips[i];
        a.skew_flip_corr = pearson(&all_skew, &all_flips_per_agent[i]);
    }
    Ok(first_seed_samples)
}

fn print_leaderboard(agents: &[ArenaAgent], gen_label: &str) {
    let mut lb = Leaderboard::new(agents.len() as u32);
    for (i, agent) in agents.iter().enumerate() {
        lb.entries[i].cumulative_reward = agent.cumulative_reward;
    }
    eprintln!("\n[leaderboard {gen_label}]  reward = fee − adverse·|drift|·base − gas");
    for (rank, e) in lb.ranked().iter().enumerate() {
        let agent = &agents[e.agent_id as usize];
        let corr = if agent.skew_flip_corr.is_finite() {
            format!("{:+.3}", agent.skew_flip_corr)
        } else {
            " n/a ".into()
        };
        eprintln!("  #{} account={:>2}  reward={:+.4} ±{:.4}  fills={}  arb_flips={}  skew↔flip={}",
            rank + 1, agent.account_index,
            e.cumulative_reward, agent.reward_stdev,
            agent.total_fills, agent.arb_flips, corr);
    }
    let max_r = agents.iter().map(|a| a.cumulative_reward).fold(f64::NEG_INFINITY, f64::max);
    let min_r = agents.iter().map(|a| a.cumulative_reward).fold(f64::INFINITY, f64::min);
    let total_flips: u64 = agents.iter().map(|a| a.arb_flips).sum();
    eprintln!("  dispersion: max−min = {:+.6}   population arb_flips = {}", max_r - min_r, total_flips);
}

/// Sakana-inspired selection score for PBT. Combines:
///   * the agent's own cumulative reward
///   * `α · min(arena_rewards)` — soft-min term that protects the
///     weakest peer; high-α arenas behave more like a coalition
///   * `β · |own_flip_rate − mean_flip_rate|` — entropy bonus
///     rewarding strategy divergence from the population mean,
///     so a niche strategy isn't out-selected by a dominant clone
/// All defaults to 0 → behaviour identical to plain reward ranking.
fn selection_scores(agents: &[ArenaAgent], coop_alpha: f64, diversity_beta: f64) -> Vec<f64> {
    let n = agents.len();
    if n == 0 { return Vec::new(); }
    let min_r = agents.iter().map(|a| a.cumulative_reward)
        .fold(f64::INFINITY, f64::min);
    let total_decisions: f64 = agents.iter()
        .map(|a| (a.total_fills + a.arb_flips).max(1) as f64).sum::<f64>() / n as f64;
    let mean_flip_rate: f64 = agents.iter()
        .map(|a| (a.arb_flips as f64) / total_decisions.max(1.0)).sum::<f64>() / n as f64;
    agents.iter().map(|a| {
        let own_flip_rate = (a.arb_flips as f64) / total_decisions.max(1.0);
        let entropy = (own_flip_rate - mean_flip_rate).abs();
        a.cumulative_reward + coop_alpha * min_r + diversity_beta * entropy
    }).collect()
}

/// Replace bottom-quartile agents with copies of top-quartile weights
/// + Gaussian mutation. Returns the number of agents replaced.
fn pbt_replace_bottom(
    agents: &mut [ArenaAgent], mutation_std: f32, gen_idx: usize,
    coop_alpha: f64, diversity_beta: f64,
) -> Result<usize> {
    let n = agents.len();
    if n < 2 { return Ok(0); }
    let n_replace = (n / 4).max(1);
    // Rank by Sakana-style adjusted score (collapses to plain reward
    // when α=β=0 — current default).
    let scores = selection_scores(agents, coop_alpha, diversity_beta);
    let mut ranks: Vec<usize> = (0..n).collect();
    ranks.sort_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Top n_replace candidates copy into bottom n_replace slots.
    let mut replaced = 0;
    for k in 0..n_replace {
        let top = ranks[k];
        let bot = ranks[n - 1 - k];
        if top == bot { continue; }
        // Clone the donor weights via the untyped representation —
        // RegionalWeightsTyped doesn't expose Clone directly because
        // device buffers may live on GPU; round-tripping through
        // host vecs is the cleanest in-process duplicate.
        let donor_untyped = penumbra_arena::checkpoint::typed_to_untyped::<Cpu>(&agents[top].weights)?;
        let mut mutant = modgrad_ctm::graph::RegionalWeightsTyped::<Cpu>::from_untyped(&donor_untyped)
            .map_err(|e| anyhow::anyhow!("from_untyped for mutant: {e}"))?;
        // Per-pair seed: gen_idx scrambled with the bot index.
        let seed = (gen_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (bot as u64).wrapping_mul(0xBF58476D1CE4E5B9);
        penumbra_arena::checkpoint::perturb_weights_in_place::<Cpu>(
            &mut mutant, seed, mutation_std,
        )?;
        agents[bot].weights = mutant;
        // Re-init optimizer state (fresh AdamW moments for the new weights).
        agents[bot].optimizer = modgrad_ctm::graph::RegionalAdamWTyped::<Cpu>::new(&agents[bot].weights)
            .map_err(|e| anyhow::anyhow!("RegionalAdamWTyped::new for bot {bot}: {e}"))?
            .with_lr(agents[bot].optimizer.lr).with_clip(1.0);
        replaced += 1;
    }
    Ok(replaced)
}

fn main() -> Result<()> {
    let n_agents: usize = std::env::var("AGENTS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let pretrain_steps: usize = std::env::var("PRETRAIN").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let pretrain_epochs: usize = std::env::var("PRETRAIN_EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let eval_blocks: usize = std::env::var("EVAL").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let bs: usize = std::env::var("BS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let lr: f32 = std::env::var("LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-2);
    let n_generations: usize = std::env::var("GENERATIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_eval_seeds: usize = std::env::var("EVAL_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mutation_std: f32 = std::env::var("MUTATION_STD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let coop_alpha: f64 = std::env::var("COOP_ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let diversity_beta: f64 = std::env::var("DIVERSITY_BETA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let skewed = std::env::var("SKEWED").is_ok();
    let replay_cfg = if skewed {
        SyntheticReplayConfig::um_usdc_skewed()
    } else {
        SyntheticReplayConfig::um_usdc_default()
    };

    eprintln!("== penumbra arena (PBT) ==");
    eprintln!("  agents={n_agents}  generations={n_generations}");
    eprintln!("  pretrain={pretrain_steps}×{pretrain_epochs}ep  eval={eval_blocks}×{n_eval_seeds}seeds  bs={bs}  lr={lr}");
    eprintln!("  mutation_std={mutation_std}  replay={}",
        if skewed { "SKEWED (depth-asymmetry → drift)" } else { "default (symmetric, random walk)" });
    if coop_alpha != 0.0 || diversity_beta != 0.0 {
        eprintln!("  selection: coop_α={coop_alpha}  diversity_β={diversity_beta}  (Sakana-style soft-min + entropy)");
    }

    let untyped = RegionalWeights::new(default_mm_cfg());
    eprintln!("  brain params = {}", untyped.n_params());

    let mut agents: Vec<ArenaAgent> = (1..=n_agents)
        .map(|i| ArenaAgent::new(i as u32, untyped.clone(), lr))
        .collect::<Result<_>>()?;

    let mut last_samples: Option<EvalSamples> = None;
    for gen_idx in 0..n_generations {
        eprintln!("\n══════ generation {} ══════", gen_idx + 1);

        // Pretrain phase — each agent on its own replay seed.
        eprintln!("[pretrain] each agent runs {pretrain_epochs} epoch(s) × {pretrain_steps} imitation steps");
        for (i, agent) in agents.iter_mut().enumerate() {
            let seed = 0xA000 + (gen_idx * 100 + i) as u64;
            let mse = pretrain_one(agent, seed, pretrain_steps, pretrain_epochs, bs, &replay_cfg)?;
            eprintln!("  agent {i}  final_epoch_mse={:.4}", mse);
        }

        // Eval phase — shared replay.
        eprintln!("[eval] {eval_blocks} blocks × {n_eval_seeds} seed(s)");
        last_samples = evaluate_all_multi_seed(
            &mut agents, eval_blocks,
            0xEEE0 + gen_idx as u64, n_eval_seeds, &replay_cfg,
        )?;
        print_leaderboard(&agents, &format!("gen {}", gen_idx + 1));

        // PBT mutation between generations (skip after the last).
        if gen_idx + 1 < n_generations {
            let n_replaced = pbt_replace_bottom(
                &mut agents, mutation_std, gen_idx, coop_alpha, diversity_beta,
            )?;
            eprintln!("\n[pbt] replaced {n_replaced} bottom agents with mutated copies of top");
        }
    }

    if let (Some(chart_path), Some(samples)) = (std::env::var("CHART").ok(), last_samples.as_ref()) {
        let trace = penumbra_arena::viz::ArenaTrace {
            mid: samples.mid.clone(),
            skew: samples.skew.clone(),
            agent_labels: agents.iter()
                .map(|a| format!("acct {}", a.account_index)).collect(),
            cum_reward: samples.cum_reward_per_agent.clone(),
        };
        penumbra_arena::viz::write_svg(&chart_path, &trace)?;
        eprintln!("\n[chart] wrote {chart_path}");
    }

    // Save the highest-reward agent's brain so the live arena can pick
    // it up. Default path matches what penumbra_train and
    // penumbra_live_arena agree on (/tmp/penumbra_arena_brain.bin); the
    // user can override with CHECKPOINT_OUT.
    let out_path = std::env::var("CHECKPOINT_OUT")
        .unwrap_or_else(|_| "/tmp/penumbra_arena_brain.bin".to_string());
    let final_scores = selection_scores(&agents, coop_alpha, diversity_beta);
    let winner = (0..agents.len())
        .max_by(|&a, &b| {
            final_scores[a].partial_cmp(&final_scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(idx) = winner {
        eprintln!("\n[checkpoint] saving winner (agent {} account {}, reward={:+.4} ±{:.4}) to {}",
            idx, agents[idx].account_index,
            agents[idx].cumulative_reward, agents[idx].reward_stdev, out_path);
        penumbra_arena::checkpoint::save(&agents[idx].weights, &out_path)?;
    }

    eprintln!("\nDone.");
    Ok(())
}
