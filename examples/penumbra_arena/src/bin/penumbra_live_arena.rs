//! Live Penumbra arena — N `MmAgent<Cpu, HybridChain>` instances
//! trading against the real chain. Per block: read state → encode →
//! forward → decode barbell → diff against open positions → submit
//! close+open via pcli.
//!
//! **Defaults to dry-run.** Set `LIVE=1` to submit txs (requires
//! master account 0 funded + `arena_setup` having distributed to
//! accounts 1..=N). Loads a pretrained checkpoint from
//! `/tmp/penumbra_arena_brain.bin` if present (override with
//! `CHECKPOINT=...`).

use anyhow::{Context, Result};

#[cfg(feature = "embedded")]
use penumbra_arena::embedded::EmbeddedConfig;
use penumbra_arena::agent::MmAgent;
use penumbra_arena::chain::{Pair, PcliChain, PenumbraChain};
use penumbra_arena::default_mm_cfg;
use penumbra_arena::hybrid::HybridChain;
use penumbra_arena::motor::BarbellConfig;

use modgrad_ctm::graph::{RegionalWeights, RegionalWeightsTyped};
use modgrad_device::backend::tensor::Cpu;

fn make_hybrid(account: u32, dry_run: bool) -> Result<HybridChain> {
    #[cfg(feature = "embedded")]
    {
        let pcli_home = std::path::PathBuf::from("/home/alice/.local/share/pcli");
        let cfg = EmbeddedConfig {
            home: format!("/tmp/penumbra_arena_agent_{account}").into(),
            grpc_url: String::new(),       // pulled from pcli config.toml
            account_index: account,
            custody_path: pcli_home,
        };
        return HybridChain::embedded_plus_pcli(cfg, dry_run);
    }
    #[cfg(not(feature = "embedded"))]
    {
        // Each agent uses its own pcli home directory at
        // `/tmp/penumbra_arena_agent_<account>/pcli/`. That dir holds
        // a copy of the user's wallet (config.toml + custody key +
        // sqlite cache); spawned via `cp -r ~/.local/share/pcli`. The
        // private sqlite means no lock contention between agents, so
        // the per-block per-agent pcli calls can fan out in parallel.
        let home = std::path::PathBuf::from(format!("/tmp/penumbra_arena_agent_{account}/pcli"));
        Ok(HybridChain::pcli_only_with_home(account, dry_run, home))
    }
}

/// Provision a private pcli home for an agent by copying the user's
/// master wallet (`~/.local/share/pcli`) into
/// `/tmp/penumbra_arena_agent_<account>/pcli`. Idempotent — skips the
/// copy when the destination already has a `config.toml`.
fn ensure_agent_home(account: u32) -> Result<()> {
    let home = std::path::PathBuf::from(format!("/tmp/penumbra_arena_agent_{account}/pcli"));
    if home.join("config.toml").exists() {
        return Ok(());
    }
    let parent = home.parent().expect("agent home has parent");
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create_dir_all {parent:?}"))?;
    let src = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".local/share/pcli"))
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/alice/.local/share/pcli"));
    let status = std::process::Command::new("cp")
        .arg("-r").arg(&src).arg(&home)
        .status()
        .with_context(|| format!("cp -r {src:?} {home:?}"))?;
    if !status.success() {
        anyhow::bail!("cp -r failed for agent {account} home setup");
    }
    Ok(())
}

fn main() -> Result<()> {
    let n_agents: u32 = std::env::var("AGENTS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let blocks: u64 = std::env::var("BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let block_secs: u64 = std::env::var("BLOCK_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let live = std::env::var("LIVE").is_ok();
    let dry_run = !live;
    let max_size_frac: f64 = std::env::var("MAX_SIZE_FRAC")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(BarbellConfig::default().max_size_frac);
    let max_offset_bps: f64 = std::env::var("MAX_OFFSET_BPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(BarbellConfig::default().max_offset_bps);
    let inner_only = std::env::var("INNER_ONLY").is_ok();
    let stop_loss_pct: f64 = std::env::var("STOP_LOSS_PCT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100.0);   // disabled = ≥100%
    /// `LADDER_RUNGS=N` switches each agent's decoder into a fixed
    /// 2·N-rung ladder (brain motor ignored). 5 → 10 positions/agent.
    let ladder_rungs: Option<usize> = std::env::var("LADDER_RUNGS")
        .ok().and_then(|s| s.parse().ok()).filter(|n| *n > 0);
    let ladder_inner_bps: f64 = std::env::var("LADDER_INNER_BPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5.0);

    eprintln!("=== penumbra live arena ===");
    eprintln!("  agents     = {n_agents}  (accounts 1..={n_agents})");
    eprintln!("  blocks     = {blocks}");
    eprintln!("  block_secs = {block_secs}");
    eprintln!("  dry_run    = {dry_run}");
    eprintln!("  max_size_frac  = {max_size_frac}");
    eprintln!("  max_offset_bps = {max_offset_bps}");
    eprintln!("  inner_only     = {inner_only}");
    eprintln!("  stop_loss_pct  = {stop_loss_pct}");
    if let Some(n) = ladder_rungs {
        eprintln!("  ladder         = 2×{n} positions/agent at log-spaced offsets [{ladder_inner_bps}, {max_offset_bps}] bps (brain bypassed)");
    }
    if live {
        eprintln!();
        eprintln!("  ⚠ LIVE MODE: txs will be submitted via pcli.");
        eprintln!("  ⚠ Ensure master account 0 has funded sub-accounts 1..={n_agents}");
        eprintln!("    (run `AGENTS={n_agents} cargo run -p penumbra_arena --bin arena_setup` once first).");
    }
    eprintln!();

    // Build N agents. Each gets its own brain — pretrained from
    // CHECKPOINT if set, else a fresh random init (which churns).
    let checkpoint = std::env::var("CHECKPOINT")
        .unwrap_or_else(|_| "/tmp/penumbra_arena_brain.bin".to_string());
    let use_checkpoint = std::path::Path::new(&checkpoint).exists();

    let untyped = RegionalWeights::new(default_mm_cfg());
    eprintln!("brain params per agent = {}", untyped.n_params());
    if use_checkpoint {
        eprintln!("loading pretrained brain from {checkpoint}");
    } else {
        eprintln!("no checkpoint at {checkpoint} — using random init");
        eprintln!("  (run `cargo run -p penumbra_arena --bin penumbra_train` first)");
    }

    /// Per-agent state we track for PnL reporting. Captured at start
    /// (initial portfolio) and refreshed each block.
    #[derive(Clone, Copy, Default)]
    struct AgentBook {
        starting_pv_usdc: f64,
        last_pv_usdc: f64,
        last_base: f64,
        last_quote: f64,
        last_mid: f64,
        cumulative_decisions: u64,
        cumulative_arb_flips: u64,
    }

    // Provision a private pcli home per agent (copy of master wallet).
    // Idempotent — second run reuses existing dirs.
    eprintln!();
    for i in 1..=n_agents {
        ensure_agent_home(i)
            .with_context(|| format!("ensure pcli home for agent {i}"))?;
        eprintln!("  agent {i:>2}  pcli home = /tmp/penumbra_arena_agent_{i}/pcli");
    }

    let mut agents: Vec<(u32, MmAgent<Cpu, HybridChain>)> = Vec::with_capacity(n_agents as usize);
    let mut books: Vec<AgentBook> = vec![AgentBook::default(); n_agents as usize];
    for i in 1..=n_agents {
        let typed = if use_checkpoint {
            penumbra_arena::checkpoint::load::<Cpu>(&checkpoint)
                .with_context(|| format!("load checkpoint for agent {i}"))?
        } else {
            RegionalWeightsTyped::<Cpu>::from_untyped(&untyped)
                .map_err(|e| anyhow::anyhow!("from_untyped agent {i}: {e}"))?
        };
        let chain = make_hybrid(i, dry_run)
            .with_context(|| format!("HybridChain for account {i}"))?;
        let motor_cfg = BarbellConfig {
            max_size_frac,
            max_offset_bps,
            skip_outer: inner_only,
            ladder_rungs_per_side: ladder_rungs,
            ladder_inner_bps,
            ..BarbellConfig::default()
        };
        let agent = MmAgent::<Cpu, HybridChain>::new(
            &untyped, typed, chain, Pair::um_usdc(), motor_cfg,
        ).with_context(|| format!("MmAgent for account {i}"))?;
        agents.push((i, agent));
        eprintln!("  agent {i:>2}  account={i}  ready");
    }

    // Initial book snapshot — establishes the starting portfolio
    // value baseline so per-block PnL is meaningful.
    eprintln!();
    for (idx, (account, agent)) in agents.iter().enumerate() {
        let bf = match agent.chain.block_features(&Pair::um_usdc()) {
            Ok(b) => b,
            Err(e) => { eprintln!("  agent {account}: initial book read failed: {e}"); continue; }
        };
        let pv = bf.balance.base * bf.quote.mid + bf.balance.quote;
        books[idx] = AgentBook {
            starting_pv_usdc: pv,
            last_pv_usdc: pv,
            last_base: bf.balance.base,
            last_quote: bf.balance.quote,
            last_mid: bf.quote.mid,
            cumulative_decisions: 0,
            cumulative_arb_flips: 0,
        };
        eprintln!("  agent {account:>2}  start  base={:.4} UM  quote={:.4} USDC  pv={:.4} USDC",
            bf.balance.base, bf.balance.quote, pv);
    }

    eprintln!();
    for block_i in 0..blocks {
        eprintln!("─── block {} ─────────────────", block_i + 1);
        // Step all agents in parallel — separate pcli homes mean they
        // don't compete for the same wallet sqlite lock. Each thread
        // owns its agent's `&mut MmAgent`, runs step() (full open+close
        // cycle), then refreshes book features, then exits.
        struct StepResult {
            account: u32, idx: usize,
            positions: usize, flips: u64,
            bf: Option<penumbra_arena::chain::BlockFeatures>,
            err: Option<String>,
        }
        let pair = Pair::um_usdc();
        let results: Vec<StepResult> = std::thread::scope(|s| {
            let handles: Vec<_> = agents.iter_mut().enumerate().map(|(idx, (account, agent))| {
                let acct = *account;
                let pair = pair.clone();
                s.spawn(move || -> StepResult {
                    match agent.step() {
                        Ok(positions) => {
                            let flips = agent.last_step_arb_flips;
                            let bf = agent.chain.block_features(&pair).ok();
                            StepResult { account: acct, idx, positions: positions.len(), flips, bf, err: None }
                        }
                        Err(e) => StepResult {
                            account: acct, idx, positions: 0, flips: 0, bf: None,
                            err: Some(e.to_string()),
                        },
                    }
                })
            }).collect();
            handles.into_iter().map(|h| h.join().expect("step panic")).collect()
        });
        for r in &results {
            if let Some(err) = &r.err {
                eprintln!("  agent {:>2}  step error: {err}", r.account);
                continue;
            }
            books[r.idx].cumulative_decisions += r.positions as u64;
            books[r.idx].cumulative_arb_flips += r.flips;
            if let Some(bf) = &r.bf {
                let pv = bf.balance.base * bf.quote.mid + bf.balance.quote;
                let pnl = pv - books[r.idx].starting_pv_usdc;
                let dpnl = pv - books[r.idx].last_pv_usdc;
                books[r.idx].last_pv_usdc = pv;
                books[r.idx].last_base = bf.balance.base;
                books[r.idx].last_quote = bf.balance.quote;
                books[r.idx].last_mid = bf.quote.mid;
                eprintln!(
                    "  agent {:>2}  decided {:>2} (arb_flip={}) pv={:.4} ({:+.6}) Δ={:+.6} USDC",
                    r.account, r.positions, r.flips, pv, pnl, dpnl,
                );
            } else {
                eprintln!("  agent {:>2}  decided {} positions (arb_flip={})  (book read failed)",
                    r.account, r.positions, r.flips);
            }
        }
        // Stop-loss: aggregate arena PnL drop > stop_loss_pct ⇒ halt.
        // Skipped when stop_loss_pct ≥ 100 (the disabled default).
        if stop_loss_pct < 100.0 {
            let starting_total: f64 = books.iter().map(|b| b.starting_pv_usdc).sum();
            let current_total:  f64 = books.iter().map(|b| b.last_pv_usdc).sum();
            if starting_total > 1e-9 {
                let drop_pct = 100.0 * (starting_total - current_total) / starting_total;
                if drop_pct >= stop_loss_pct {
                    eprintln!(
                        "  ⚠ STOP-LOSS HIT: aggregate PV dropped {:.2}% (≥ {:.2}%) — halting at block {}",
                        drop_pct, stop_loss_pct, block_i + 1,
                    );
                    break;
                }
            }
        }
        if block_i + 1 < blocks {
            std::thread::sleep(std::time::Duration::from_secs(block_secs));
        }
    }

    // Final sweep: close all positions on each agent.
    if live {
        eprintln!("\n=== final cleanup ===");
        for (account, _) in &agents {
            let pcli = PcliChain::new(*account, dry_run);
            match pcli.close_all_positions() {
                Ok(()) => eprintln!("  agent {account}: close-all submitted"),
                Err(e) => eprintln!("  agent {account}: close-all failed: {e}"),
            }
        }
    }

    // Final leaderboard: per-agent PnL summary.
    eprintln!("\n=== leaderboard ===");
    eprintln!("  {:>5}  {:>10}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}",
        "acct", "UM", "USDC", "PV (USDC)", "PnL", "decisions", "arb_flips");
    let mut ranked: Vec<(usize, &AgentBook)> = books.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        let pnl_a = a.1.last_pv_usdc - a.1.starting_pv_usdc;
        let pnl_b = b.1.last_pv_usdc - b.1.starting_pv_usdc;
        pnl_b.partial_cmp(&pnl_a).unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, b) in ranked {
        let account = agents[i].0;
        let pnl = b.last_pv_usdc - b.starting_pv_usdc;
        eprintln!("  {:>5}  {:>10.4}  {:>10.4}  {:>12.4}  {:>+10.6}  {:>10}  {:>10}",
            account, b.last_base, b.last_quote, b.last_pv_usdc, pnl, b.cumulative_decisions, b.cumulative_arb_flips);
    }

    eprintln!("\nDone.");
    Ok(())
}
