//! `HybridChain` — pragmatic v0 architecture for the arena agents.
//!
//! Combines the strengths of the two existing chain backends:
//!
//! * **Reads via [`EmbeddedChain`]** (when `--features embedded`):
//!   in-process Storage + tonic gRPC. ~ms per query, no subprocess
//!   overhead. Drives the brain's per-block BlockFeatures pipeline.
//!
//! * **Writes via [`PcliChain`]**: one-shot `pcli` subprocess per
//!   tx. NOT a long-running daemon — pcli loads, signs, submits,
//!   exits. Inherits the user's existing wallet config and tx
//!   plumbing without re-implementing TransactionPlan ourselves.
//!
//! Both backends bind to the same Penumbra account index so the
//! reads and writes are consistent. When in-process tx submission
//! lands inside `EmbeddedChain` (penumbra-sdk-transaction wiring),
//! `HybridChain` becomes a thin wrapper that delegates everything
//! to `EmbeddedChain` and `PcliChain` is retired from the agent path.

use anyhow::Result;

use crate::chain::{
    Balance, BlockFeatures, Fill, OrderBook, Pair, PcliChain, PenumbraChain, PositionId,
    PositionParams, Quote,
};

#[cfg(feature = "embedded")]
use crate::embedded::{EmbeddedChain, EmbeddedConfig};

/// Read-side handle: trait object so callers can swap MockChain
/// in for tests. With `--features embedded`, the typical reader is
/// an `EmbeddedChain`.
pub struct HybridChain {
    /// Penumbra account index this agent owns.
    pub account: u32,
    /// Read-side handle. Typically EmbeddedChain in production,
    /// PcliChain in --no-default-features builds, MockChain in tests.
    /// Must be `Send` so the live arena can step agents in parallel
    /// (each in its own thread).
    pub reader: Box<dyn PenumbraChain + Send>,
    /// Write-side: pcli subprocess. Always PcliChain in v0.
    pub writer: PcliChain,
}

impl HybridChain {
    /// Build a HybridChain with EmbeddedChain (when feature enabled)
    /// for reads and PcliChain for writes. Both bound to `account`.
    #[cfg(feature = "embedded")]
    pub fn embedded_plus_pcli(cfg: EmbeddedConfig, dry_run: bool) -> Result<Self> {
        let account = cfg.account_index;
        let reader: Box<dyn PenumbraChain + Send> = Box::new(EmbeddedChain::new(cfg)?);
        let writer = PcliChain::new(account, dry_run);
        Ok(Self { account, reader, writer })
    }

    /// Build a HybridChain with PcliChain for both reads and writes.
    /// Slower reads (per-query subprocess) but works without the
    /// `embedded` feature compiled in.
    pub fn pcli_only(account: u32, dry_run: bool) -> Self {
        let reader: Box<dyn PenumbraChain + Send> = Box::new(PcliChain::new(account, false));
        let writer = PcliChain::new(account, dry_run);
        Self { account, reader, writer }
    }

    /// Same as `pcli_only` but with a private pcli home — both reader
    /// and writer use `--home <path>` so the chain has its own sqlite
    /// view (no lock contention with peer agents). Caller must have
    /// initialised the home dir (e.g. via `cp -r ~/.local/share/pcli`).
    pub fn pcli_only_with_home(account: u32, dry_run: bool, home: std::path::PathBuf) -> Self {
        let reader: Box<dyn PenumbraChain + Send> = Box::new(
            PcliChain::new(account, false).with_home(home.clone()),
        );
        let writer = PcliChain::new(account, dry_run).with_home(home);
        Self { account, reader, writer }
    }

    /// Build with a caller-supplied reader (e.g. MockChain for tests).
    /// Writer is still PcliChain — pass dry_run=true if the test
    /// shouldn't actually issue pcli commands.
    pub fn with_reader(account: u32, reader: Box<dyn PenumbraChain + Send>, dry_run: bool) -> Self {
        Self { account, reader, writer: PcliChain::new(account, dry_run) }
    }
}

impl PenumbraChain for HybridChain {
    fn account(&self) -> u32 { self.account }

    // ─── Reads → reader (fast path) ─────────────────────────────
    fn fair_price(&self, pair: &Pair) -> Result<Quote> { self.reader.fair_price(pair) }
    fn balance(&self) -> Result<Balance> { self.reader.balance() }
    fn depth(&self, pair: &Pair, offsets_bps: &[f64]) -> Result<OrderBook> {
        self.reader.depth(pair, offsets_bps)
    }
    fn block_features(&self, pair: &Pair) -> Result<BlockFeatures> {
        self.reader.block_features(pair)
    }
    fn pending_fills(&self) -> Result<Vec<Fill>> { self.reader.pending_fills() }
    fn address_for_account(&self, account: u32) -> Result<String> {
        self.reader.address_for_account(account)
    }

    // ─── Writes → writer (PcliChain subprocess) ─────────────────
    fn open_position(&self, p: &PositionParams) -> Result<PositionId> {
        self.writer.open_position(p)
    }
    fn close_position(&self, id: &PositionId) -> Result<()> {
        self.writer.close_position(id)
    }
    fn close_all_positions(&self) -> Result<()> {
        self.writer.close_all_positions()
    }
    fn withdraw_all_positions(&self) -> Result<()> {
        self.writer.withdraw_all_positions()
    }
    fn transfer_to_account(&self, to_account: u32, value: &str) -> Result<()> {
        self.writer.transfer_to_account(to_account, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HybridChain wires reads to PcliChain when no feature is on.
    #[test]
    fn pcli_only_constructs() {
        let h = HybridChain::pcli_only(1, true);
        assert_eq!(h.account(), 1);
    }
}
