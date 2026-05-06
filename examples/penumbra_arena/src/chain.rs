//! `PenumbraChain` — abstraction over the on-chain interface.
//!
//! Three impl tiers, in increasing production-readiness:
//!
//! 1. **`PcliChain`** (v0, here): subprocess wrapper around `pcli`.
//!    Convenient for prototyping and dry-runs because it inherits
//!    pcli's wallet config; slow (full sync per query). Suitable
//!    for setup-time orchestration like `arena_setup`.
//!
//! 2. **`EmbeddedChain`** (v1, stubbed): in-process view+custody
//!    using `penumbra-sdk-*` crates directly. Each agent owns its
//!    own embedded view daemon — no subprocess, no separate process
//!    to manage. State stays in-process; tx submission goes straight
//!    to a fullnode gRPC. This is the production target for the
//!    arena bots: N agents = N `EmbeddedChain` instances, each bound
//!    to a different `AddressIndex` of the same wallet.
//!
//! 3. **`MockChain`** (in `bin/penumbra_mm.rs`): fake market for
//!    offline smoke tests, no chain at all.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};

/// Trading pair identifier, base/quote.
#[derive(Debug, Clone)]
pub struct Pair {
    /// Base asset (the one the agent holds inventory of).
    pub base: String,
    /// Quote asset (the one prices are denominated in).
    pub quote: String,
}

impl Pair {
    pub fn um_usdc() -> Self {
        Pair {
            base: "penumbra".into(),
            quote: "transfer/channel-2/uusdc".into(),
        }
    }
}

/// Bid/ask quote estimate from the chain.
#[derive(Debug, Clone, Copy)]
pub struct Quote {
    /// Price to sell 1 base → quote, in quote-per-base whole units.
    pub sell_price: f64,
    /// Price to buy 1 base ← quote, in quote-per-base whole units.
    pub buy_price: f64,
    /// Midpoint, derived: `(sell + buy) / 2`.
    pub mid: f64,
}

impl Quote {
    pub fn spread_bps(&self) -> f64 {
        if self.mid > 0.0 {
            10_000.0 * (self.buy_price - self.sell_price) / self.mid
        } else {
            0.0
        }
    }
}

/// Wallet balance for one agent.
#[derive(Debug, Clone, Default)]
pub struct Balance {
    /// Base-asset balance (whole units, e.g. UM).
    pub base: f64,
    /// Quote-asset balance (whole units, e.g. USDC).
    pub quote: f64,
}

/// One side of the order-book depth at a given price offset from mid.
/// Penumbra's "order book" is a sum of LP positions across the curve;
/// each `DepthLevel` is the aggregate base-asset reserves available
/// to fill within `offset_bps` of mid on that side.
#[derive(Debug, Clone, Copy, Default)]
pub struct DepthLevel {
    /// Distance from mid in basis points. 0 = at-mid, positive only.
    pub offset_bps: f64,
    /// Total base-asset reserves available on this side at this offset.
    pub total_base: f64,
    /// Tightest competitor fee tier seen at this offset (lowest = most
    /// aggressive). 0 if no competitor present.
    pub min_competitor_fee_bps: u32,
}

/// Order-book snapshot for a pair. Fixed levels at preset offsets so
/// the feature encoder has a stable layout. v0: 4 levels per side at
/// 0/50/100/200 bps offsets from mid.
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    /// Bid side, levels sorted by ascending offset_bps (closest to mid first).
    pub bids: Vec<DepthLevel>,
    /// Ask side, same convention.
    pub asks: Vec<DepthLevel>,
}

impl OrderBook {
    /// Total visible depth on bid side within `max_offset_bps` of mid.
    pub fn bid_depth_within(&self, max_offset_bps: f64) -> f64 {
        self.bids.iter()
            .filter(|d| d.offset_bps <= max_offset_bps)
            .map(|d| d.total_base).sum()
    }
    /// Total visible depth on ask side within `max_offset_bps` of mid.
    pub fn ask_depth_within(&self, max_offset_bps: f64) -> f64 {
        self.asks.iter()
            .filter(|d| d.offset_bps <= max_offset_bps)
            .map(|d| d.total_base).sum()
    }
}

/// Recent block-level features summarising the market state.
#[derive(Debug, Clone, Default)]
pub struct BlockFeatures {
    pub block_height: u64,
    pub quote: Quote,
    pub balance: Balance,
    /// Number of agent-owned positions currently open.
    pub open_positions: u32,
    /// Live order-book snapshot for the trading pair.
    pub book: OrderBook,
}

impl Default for Quote {
    fn default() -> Self { Self { sell_price: 0.0, buy_price: 0.0, mid: 0.0 } }
}

/// Parameters for opening a single LP position.
#[derive(Debug, Clone)]
pub struct PositionParams {
    /// "buy" or "sell" — same orientation pcli uses.
    pub side: String,
    /// Amount of base-asset to commit (whole units).
    pub amount: f64,
    /// Limit price (quote-per-base, whole units).
    pub price: f64,
    /// Fee tier in basis points.
    pub fee_bps: u32,
    /// Auto-close after first fill — typical for MM.
    pub auto_close: bool,
}

impl PositionParams {
    /// True when this position is placed *across* mid for its declared
    /// side — buy at price > mid, or sell at price < mid. With
    /// signed-offset (tanh) decoding this is the brain's arb-seeker
    /// mode: aggressive crossing orders that the chain auction takes
    /// as taker liquidity when mispricing exists.
    pub fn is_arb_flip(&self, mid: f64) -> bool {
        (self.side == "buy"  && self.price >  mid)
            || (self.side == "sell" && self.price <  mid)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionId(pub String);

/// Recent fill against an agent-owned position. Populated on the
/// agent side by diffing successive `pending_fills` snapshots.
#[derive(Debug, Clone)]
pub struct Fill {
    pub position_id: PositionId,
    pub block_height: u64,
    pub side: String,
    pub filled_base: f64,
    pub filled_quote: f64,
}

/// Chain abstraction. PcliChain (v0) implements this via subprocess;
/// a future DirectChain (v1) will implement it via native rust crates.
///
/// **Account-aware.** Each `PenumbraChain` instance is bound to a
/// specific Penumbra account index (set at construction). All
/// transactions debit that account; balance queries return that
/// account's funds; positions are owned by that account. To run N
/// agents on independent accounts in the same wallet, construct N
/// `PcliChain { account: i }` — Penumbra HD wallets give every
/// integer account index its own funds + position pool natively.
pub trait PenumbraChain {
    fn account(&self) -> u32;
    fn fair_price(&self, pair: &Pair) -> Result<Quote>;
    fn balance(&self) -> Result<Balance>;
    /// Order-book depth snapshot for `pair`. Probes the curve at
    /// each offset in `offsets_bps` (per side) by simulating trades
    /// of increasing size. The result is the aggregate base-asset
    /// volume available within each offset band.
    fn depth(&self, pair: &Pair, offsets_bps: &[f64]) -> Result<OrderBook>;
    fn block_features(&self, pair: &Pair) -> Result<BlockFeatures>;
    fn open_position(&self, p: &PositionParams) -> Result<PositionId>;
    fn close_position(&self, id: &PositionId) -> Result<()>;
    fn close_all_positions(&self) -> Result<()>;
    fn withdraw_all_positions(&self) -> Result<()>;
    fn pending_fills(&self) -> Result<Vec<Fill>>;
    /// Internal transfer to another account in the same wallet.
    /// Used by `arena_setup` to fund agent accounts 1..N from a
    /// master account 0. `value` is in pcli value-string form, e.g.
    /// `"50transfer/channel-2/uusdc"` or `"10000penumbra"`.
    fn transfer_to_account(&self, to_account: u32, value: &str) -> Result<()>;
    /// Fetch the receiving address for a target account index.
    /// Required by `transfer_to_account` callers that don't know
    /// the address up front. Equivalent to `pcli view address --account N`.
    fn address_for_account(&self, account: u32) -> Result<String>;
}

/// Standard depth probe offsets. Tight enough to capture inner-rung
/// activity (50 bps), wide enough to characterise the curve shape.
pub const STD_DEPTH_OFFSETS_BPS: &[f64] = &[10.0, 50.0, 100.0, 200.0];

/// `pcli`-subprocess implementation. Same shape as
/// `bot/penumbra_mm_simple.py` but in Rust — JSON-where-pcli-supports-it,
/// stdout regex parsing where it doesn't.
pub struct PcliChain {
    /// Path to the pcli binary. Defaults to "pcli" on $PATH.
    pub pcli: PathBuf,
    /// Penumbra account index this agent controls.
    pub account: u32,
    /// Dry-run mode — prints commands instead of executing.
    pub dry_run: bool,
    /// Per-call timeout in seconds.
    pub timeout_secs: u64,
    /// Optional `--home <path>` for pcli. When set, this PcliChain
    /// instance has its own private wallet sqlite (separate sync state,
    /// no lock contention with other instances). Use when you want
    /// concurrent agents that don't queue behind one another.
    pub home: Option<PathBuf>,
}

impl PcliChain {
    pub fn new(account: u32, dry_run: bool) -> Self {
        Self {
            pcli: PathBuf::from("pcli"),
            account,
            dry_run,
            timeout_secs: 30,
            home: None,
        }
    }

    pub fn with_home(mut self, home: PathBuf) -> Self {
        self.home = Some(home);
        self
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        let mut cmd = Command::new(&self.pcli);
        if let Some(home) = &self.home {
            cmd.arg("--home").arg(home);
        }
        cmd.args(args);
        let out = cmd.output()
            .with_context(|| format!("spawn pcli {:?}", args))?;
        if !out.status.success() {
            return Err(anyhow!("pcli {:?} failed: {}",
                args, String::from_utf8_lossy(&out.stderr)));
        }
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    }
}

impl PenumbraChain for PcliChain {
    fn account(&self) -> u32 { self.account }

    fn fair_price(&self, pair: &Pair) -> Result<Quote> {
        // Native penumbra denom: prepend `u` for micro-units. IBC
        // denoms ("transfer/channel-N/uusdc") already include the
        // micro-prefix and use the full asset string as-is.
        let micro = |denom: &str| -> String {
            if denom.contains('/') { denom.to_string() } else { format!("u{denom}") }
        };
        // Sell 1 base → quote
        let sell_out = self.run(&[
            "query", "dex", "simulate",
            &format!("1000000{}", micro(&pair.base)),
            "--into", &pair.quote,
        ])?;
        let sell_price = parse_simulate_out(&sell_out, &pair.quote, 1.0, 1_000_000.0)
            .ok_or_else(|| anyhow!("could not parse sell simulate"))?;

        // Buy 0.1 quote → base (small slippage probe in the other direction)
        let buy_out = self.run(&[
            "query", "dex", "simulate",
            &format!("100000{}", micro(&pair.quote)),
            "--into", &pair.base,
        ])?;
        let buy_amount = parse_simulate_amount(&buy_out, &pair.base)
            .ok_or_else(|| anyhow!("could not parse buy simulate"))?;
        let buy_price = if buy_amount > 0.0 { 0.1 / buy_amount } else { 0.0 };

        let mid = (sell_price + buy_price) / 2.0;
        Ok(Quote { sell_price, buy_price, mid })
    }

    fn balance(&self) -> Result<Balance> {
        // `pcli view balance` lists every account; we filter by self.account.
        // v0 stub returns zeros if the parse fails — real callers can
        // upgrade once the table-format parser lands.
        let _out = self.run(&["view", "balance"])?;
        let bal = parse_balance_for_account(&_out, self.account);
        Ok(bal.unwrap_or_default())
    }

    fn depth(&self, pair: &Pair, offsets_bps: &[f64]) -> Result<OrderBook> {
        // For each offset, simulate a probe trade sized to push price
        // by ~that offset and read back the volume traded. v0 stub:
        // a single small probe per offset at a fixed size, no actual
        // curve reconstruction. Lands more accurately once we wire
        // `pcli query dex positions --trading-pair <pair>` and parse
        // the reserve curve directly. Until then this returns
        // monotonically-increasing depth proportional to offset.
        let quote = self.fair_price(pair)?;
        let mut book = OrderBook::default();
        for &off_bps in offsets_bps {
            let frac = off_bps / 10_000.0;
            // Placeholder shape: depth grows ~linearly with offset,
            // size depends on quote. Replaced by real position-curve
            // parse once the pcli command is wired.
            let volume = quote.mid * 1_000.0 * frac.max(1e-3);
            book.bids.push(DepthLevel {
                offset_bps: off_bps, total_base: volume, min_competitor_fee_bps: 0,
            });
            book.asks.push(DepthLevel {
                offset_bps: off_bps, total_base: volume, min_competitor_fee_bps: 0,
            });
        }
        Ok(book)
    }

    fn block_features(&self, pair: &Pair) -> Result<BlockFeatures> {
        let quote = self.fair_price(pair)?;
        let balance = self.balance()?;
        let book = self.depth(pair, STD_DEPTH_OFFSETS_BPS)?;
        Ok(BlockFeatures {
            block_height: 0,            // TODO: query latest height
            quote,
            balance,
            open_positions: 0,          // TODO: query position count
            book,
        })
    }

    fn open_position(&self, p: &PositionParams) -> Result<PositionId> {
        // pcli format: "<amount>penumbra@<price>transfer/channel-2/uusdc/<fee>bps"
        // price expressed as quote-base-units per 1 whole base.
        let price_quote_units = (p.price * 1_000_000.0) as u64;
        let order = format!(
            "{:.6}penumbra@{}transfer/channel-2/uusdc/{}bps",
            p.amount, price_quote_units, p.fee_bps,
        );
        let source = self.account.to_string();
        let mut args: Vec<&str> = vec![
            "tx", "position", "order", &p.side, &order,
            "--source", &source,
        ];
        if p.auto_close { args.push("--auto-close"); }

        if self.dry_run {
            eprintln!("  [DRY RUN account={}] pcli {}", self.account, args.join(" "));
            return Ok(PositionId(format!("dry-run-{}-{}", self.account, p.side)));
        }
        let out = self.run(&args)?;
        // pcli prints a multi-line tx log; the position id is the
        // first `plpid1...` token, usually after the "Position id:"
        // prefix on its own line. Pull just that out.
        let id = out.split_whitespace()
            .find(|tok| tok.starts_with("plpid1"))
            .ok_or_else(|| anyhow!("could not parse position id from pcli output: {out}"))?;
        Ok(PositionId(id.to_string()))
    }

    fn close_position(&self, id: &PositionId) -> Result<()> {
        let source = self.account.to_string();
        let args = ["tx", "position", "close", &id.0, "--source", &source];
        if self.dry_run {
            eprintln!("  [DRY RUN account={}] pcli {}", self.account, args.join(" "));
            return Ok(());
        }
        let _ = self.run(&args)?;
        Ok(())
    }

    fn close_all_positions(&self) -> Result<()> {
        let source = self.account.to_string();
        let args = ["tx", "position", "close-all", "--source", &source];
        if self.dry_run {
            eprintln!("  [DRY RUN account={}] pcli {}", self.account, args.join(" "));
            return Ok(());
        }
        let _ = self.run(&args)?;
        Ok(())
    }

    fn withdraw_all_positions(&self) -> Result<()> {
        let source = self.account.to_string();
        let args = ["tx", "position", "withdraw-all", "--source", &source];
        if self.dry_run {
            eprintln!("  [DRY RUN account={}] pcli {}", self.account, args.join(" "));
            return Ok(());
        }
        let _ = self.run(&args)?;
        Ok(())
    }

    fn pending_fills(&self) -> Result<Vec<Fill>> {
        // TODO: query pcli for owned positions filtered by --source,
        // diff against last known reserves to detect fills. v0 returns empty.
        Ok(Vec::new())
    }

    fn transfer_to_account(&self, to_account: u32, value: &str) -> Result<()> {
        let to_addr = self.address_for_account(to_account)?;
        let source = self.account.to_string();
        let args = ["tx", "send", "--to", &to_addr, value, "--source", &source];
        if self.dry_run {
            eprintln!("  [DRY RUN account={} → {}] pcli {}",
                self.account, to_account, args.join(" "));
            return Ok(());
        }
        let _ = self.run(&args)?;
        Ok(())
    }

    fn address_for_account(&self, account: u32) -> Result<String> {
        // pcli view address takes a positional `<ADDRESS_OR_INDEX>` arg.
        let acct = account.to_string();
        let out = self.run(&["view", "address", &acct])?;
        // pcli prints the bech32 address as the only line of output.
        let addr = out.lines()
            .find(|l| l.starts_with("penumbra1"))
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| out.trim().to_string());
        if addr.is_empty() {
            return Err(anyhow!("could not parse address for account {account}"));
        }
        Ok(addr)
    }
}

/// Parse `pcli view balance` output to extract the balance row for
/// a specific account index. The CLI prints a table-ish format with
/// columns ["Account", "Amount Asset"]; we look for the matching
/// account index and split each value entry by asset suffix.
fn parse_balance_for_account(stdout: &str, account: u32) -> Option<Balance> {
    let target = format!("# {}", account);
    let mut bal = Balance::default();
    let mut found = false;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(&target) { continue; }
        found = true;
        // Heuristic: parse any tokens that look like "<amount><asset>".
        for tok in trimmed.split_whitespace() {
            if let Some((num_str, asset)) = split_amount_asset(tok) {
                if let Ok(n) = num_str.parse::<f64>() {
                    if asset.starts_with("penumbra") && !asset.starts_with("upenumbra") {
                        // Whole-unit penumbra amount.
                        bal.base += n;
                    } else if asset.starts_with("upenumbra") {
                        bal.base += n / 1_000_000.0;
                    } else if asset.contains("uusdc") {
                        bal.quote += n / 1_000_000.0;
                    } else if asset == "USDC" || asset.contains("/uusdc") {
                        bal.quote += n;
                    }
                }
            }
        }
    }
    if found { Some(bal) } else { None }
}

/// Split a pcli value token like "1234.5penumbra" or "100transfer/channel-2/uusdc"
/// into (amount, asset). Returns None if the prefix isn't a number.
fn split_amount_asset(tok: &str) -> Option<(&str, &str)> {
    let split = tok.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
    if split == 0 { return None; }
    Some((&tok[..split], &tok[split..]))
}

/// Parse the "1penumbra => NNNNtransfer/channel-2/uusdc" line and
/// return whole-units-per-base price after dividing by `quote_unit_div`.
fn parse_simulate_out(stdout: &str, quote: &str, base_amt: f64, quote_unit_div: f64) -> Option<f64> {
    let quote_marker = quote.split('/').next_back().unwrap_or(quote);
    for line in stdout.lines() {
        if !line.contains("=>") || !line.contains(quote_marker) { continue; }
        let mut parts = line.split("=>");
        let _lhs = parts.next()?;
        let rhs = parts.next()?.trim();
        // RHS like "12345transfer/channel-2/uusdc"
        let num: String = rhs.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        let n: f64 = num.parse().ok()?;
        return Some((n / quote_unit_div) / base_amt);
    }
    None
}

/// Parse the "NNNuusdc => NNNNpenumbra" reverse direction and return
/// the amount of base received (whole units).
fn parse_simulate_amount(stdout: &str, base: &str) -> Option<f64> {
    for line in stdout.lines() {
        if !line.contains("=>") || !line.contains(base) { continue; }
        let mut parts = line.split("=>");
        let _lhs = parts.next()?;
        let rhs = parts.next()?.trim();
        // RHS like "12345penumbra" or "12345mpenumbra"
        let num: String = rhs.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        let n: f64 = num.parse().ok()?;
        // milli prefix → divide by 1000.
        if rhs.contains("mpenumbra") { return Some(n / 1000.0); }
        return Some(n);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sell_simulate() {
        let out = "1penumbra => 13456transfer/channel-2/uusdc";
        let p = parse_simulate_out(out, "transfer/channel-2/uusdc", 1.0, 1_000_000.0);
        assert!(p.is_some());
        // 13456 uusdc / 1e6 / 1 base = 0.013456 USDC/UM
        let v = p.unwrap();
        assert!((v - 0.013456).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn parses_buy_simulate() {
        let out = "100000transfer/channel-2/uusdc => 7500penumbra";
        let amt = parse_simulate_amount(out, "penumbra");
        assert_eq!(amt, Some(7500.0));
    }

    #[test]
    fn quote_spread_bps_works() {
        let q = Quote { sell_price: 0.0130, buy_price: 0.0132, mid: 0.0131 };
        let bps = q.spread_bps();
        assert!((bps - 152.67175).abs() < 1e-3, "got {bps}");
    }

    #[test]
    fn pair_um_usdc_constants() {
        let p = Pair::um_usdc();
        assert_eq!(p.base, "penumbra");
        assert!(p.quote.ends_with("uusdc"));
    }

    #[test]
    fn parse_balance_handles_penumbra_and_uusdc() {
        // Realistic-ish pcli view balance excerpt — values mixed with
        // account markers. Account 0 has 10000 penumbra + 50 USDC.
        let out = "\
            # 0  10000penumbra\n\
            # 0  50000000transfer/channel-2/uusdc\n\
            # 1  500penumbra\n\
        ";
        let b0 = parse_balance_for_account(out, 0).unwrap();
        assert!((b0.base - 10000.0).abs() < 1e-6);
        assert!((b0.quote - 50.0).abs() < 1e-6, "quote {} != 50", b0.quote);
        let b1 = parse_balance_for_account(out, 1).unwrap();
        assert!((b1.base - 500.0).abs() < 1e-6);
        assert_eq!(b1.quote, 0.0);
        assert!(parse_balance_for_account(out, 5).is_none());
    }

    #[test]
    fn split_amount_asset_handles_typical_tokens() {
        assert_eq!(split_amount_asset("10000penumbra"), Some(("10000", "penumbra")));
        assert_eq!(split_amount_asset("50000000transfer/channel-2/uusdc"),
                   Some(("50000000", "transfer/channel-2/uusdc")));
        assert_eq!(split_amount_asset("plain"), None);
    }
}
