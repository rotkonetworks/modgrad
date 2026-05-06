//! Provision N Penumbra agent accounts from a master pcli wallet.
//!
//! Usage:
//!   AGENTS=4 BASE_PER_AGENT=10000 QUOTE_PER_AGENT=50 \
//!     cargo run -p penumbra_arena --bin arena_setup
//!
//! The user's pcli wallet must have at least
//!   AGENTS × (BASE_PER_AGENT UM + QUOTE_PER_AGENT USDC)
//! sitting on account 0 before running this. The setup binary:
//!   1. Looks up `pcli view address --account i` for each i in 1..=N
//!   2. Submits two `tx send --source 0 --to <addr_i>` transactions
//!      per account: one for UM, one for USDC
//!   3. Verifies the resulting per-account balance via `view balance`
//!
//! Pass `DRY_RUN=1` to print the commands without executing.
//!
//! After this finishes, the arena binary can run with each agent
//! pointing at its own `PcliChain { account: i }`.

use anyhow::{Context, Result};

use penumbra_arena::chain::{PcliChain, PenumbraChain};

fn main() -> Result<()> {
    let n_agents: u32 = std::env::var("AGENTS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let base_per_agent: f64 = std::env::var("BASE_PER_AGENT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10_000.0);
    let quote_per_agent: f64 = std::env::var("QUOTE_PER_AGENT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let dry_run = std::env::var("DRY_RUN").is_ok();

    eprintln!("== arena setup ==");
    eprintln!("  agents={n_agents}");
    eprintln!("  fund per agent: {base_per_agent} UM + {quote_per_agent} USDC");
    eprintln!("  dry_run={dry_run}");

    // Master account = 0.
    let master = PcliChain::new(0, dry_run);

    // Verify master has enough.
    let master_bal = master.balance().context("master balance")?;
    eprintln!("\n[master account 0] base={:.2} UM  quote={:.2} USDC",
        master_bal.base, master_bal.quote);
    let needed_base = base_per_agent * n_agents as f64;
    let needed_quote = quote_per_agent * n_agents as f64;
    if !dry_run && (master_bal.base < needed_base || master_bal.quote < needed_quote) {
        eprintln!("\n  WARNING: master balance below total need");
        eprintln!("    needed: {needed_base} UM + {needed_quote} USDC");
        eprintln!("    have:   {} UM + {} USDC", master_bal.base, master_bal.quote);
        eprintln!("    proceeding anyway (some sends will fail)");
    }

    for agent_idx in 1..=n_agents {
        eprintln!("\n[fund agent {agent_idx}]");
        let to_addr = master.address_for_account(agent_idx)
            .with_context(|| format!("address for account {agent_idx}"))?;
        eprintln!("  → {to_addr}");

        // pcli value strings: "<amount>penumbra" for UM,
        // "<integer>transfer/channel-2/uusdc" for USDC (uusdc, 6 decimals).
        let um_value = format!("{:.6}penumbra", base_per_agent);
        let usdc_uunits = (quote_per_agent * 1_000_000.0) as u64;
        let usdc_value = format!("{}transfer/channel-2/uusdc", usdc_uunits);

        master.transfer_to_account(agent_idx, &um_value)
            .with_context(|| format!("send UM to account {agent_idx}"))?;
        master.transfer_to_account(agent_idx, &usdc_value)
            .with_context(|| format!("send USDC to account {agent_idx}"))?;
        eprintln!("  + sent {um_value}");
        eprintln!("  + sent {usdc_value}");
    }

    if !dry_run {
        eprintln!("\n[verify balances]");
        for i in 0..=n_agents {
            let chain = PcliChain::new(i, false);
            match chain.balance() {
                Ok(b) => eprintln!("  account {:>2}  base={:.2} UM  quote={:.2} USDC",
                    i, b.base, b.quote),
                Err(e) => eprintln!("  account {:>2}  balance lookup failed: {e}", i),
            }
        }
    }

    eprintln!("\nDone. Each agent now has its own funded Penumbra account.");
    eprintln!("Arena binary can run via: AGENTS={n_agents} cargo run -p penumbra_arena --bin penumbra_arena");
    Ok(())
}
