//! `EmbeddedChain` — in-process view + custody for one Penumbra account.
//!
//! Replaces the `pcli` subprocess with direct linkage to the
//! `penumbra-sdk-*` crates. Each `EmbeddedChain` owns:
//!
//! * a `Storage` (the scanned chain state — same shape pclientd
//!   maintains on disk, but in-process)
//! * a `SpendKey` derived for one `AddressIndex` of the wallet
//!   (per-agent custody — full spend authority)
//! * a `tonic` gRPC client to a fullnode (for chain queries that
//!   miss the local Storage cache + tx submission)
//! * a `ViewProtocolService` impl backed by Storage (queries balance,
//!   note set, position state)
//!
//! Per-block flow:
//!   1. `block_features()`  →  Storage scan + ObliviousQueryService
//!   2. agent emits 12-dim MOTOR  →  decoded to `Vec<PositionParams>`
//!   3. `submit_block_actions()`  →  build ONE `TransactionPlan`
//!      bundling close + open actions (multi-action tx, see #179),
//!      sign with the agent's SpendKey, submit via `BroadcastTxSync`,
//!      block until inclusion, update local Storage from the witness
//!
//! Strict serialisation per agent: build → submit → confirm → next.
//! Cross-agent isolation: each `EmbeddedChain` has a different
//! `AddressIndex` so their note pools never overlap (#180 deferred —
//! that's only a concern if/when one agent submits multiple txs per
//! block).
//!
//! ## Why feature-gate
//!
//! The penumbra-sdk crate graph is heavy (ark-*, halo2, tonic,
//! tower, ...) and pinned to versions that may conflict with
//! modgrad's existing deps. We isolate the integration behind
//! `--features embedded` so:
//!
//! * default `cargo build` for `penumbra_arena` keeps working with
//!   the current minimal dep set
//! * ark-* / halo2 only appear in builds that opt-in
//! * the `EmbeddedChain` type below is a struct stub when the
//!   feature is off, so the trait wiring + caller code compiles
//!   regardless
//!
//! ## Required path deps (added under `--features embedded`)
//!
//! Mirroring `~/rotko/penumbra/crates/bin/pclientd/Cargo.toml`:
//!
//! * `penumbra-sdk-app`
//! * `penumbra-sdk-asset`
//! * `penumbra-sdk-custody`
//! * `penumbra-sdk-keys`
//! * `penumbra-sdk-proto`        (with `rpc` feature)
//! * `penumbra-sdk-tct`
//! * `penumbra-sdk-transaction`
//! * `penumbra-sdk-view`
//!
//! Plus support: `tonic`, `tokio`, `futures`, `prost`, `directories`.

use std::path::PathBuf;
use anyhow::{Result, anyhow};

use crate::chain::{
    Balance, BlockFeatures, DepthLevel, Fill, OrderBook, Pair, PenumbraChain,
    PositionId, PositionParams, Quote, STD_DEPTH_OFFSETS_BPS,
};

/// Configuration for a single agent's embedded view+custody node.
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    /// Storage path — like `pclientd --home`. Each agent gets its own.
    pub home: PathBuf,
    /// Fullnode gRPC URL for chain queries + tx submission.
    pub grpc_url: String,
    /// Account index this agent controls. Agent's `SpendKey` is
    /// derived for `AddressIndex::new(account_index)`.
    pub account_index: u32,
    /// Path to the master wallet's seed phrase / custody json.
    /// Same wallet, different account_index per agent.
    pub custody_path: PathBuf,
}

/// In-process Penumbra view + custody. Owns its own scanned state.
///
/// **v0 scaffold only.** All `PenumbraChain` methods return
/// `Err(NotImplemented)` until the penumbra-sdk integration lands.
/// The struct is feature-gated such that the heavy SDK deps only
/// appear in `--features embedded` builds.
pub struct EmbeddedChain {
    pub cfg: EmbeddedConfig,
    /// Internal handles wired in once the penumbra-sdk integration
    /// lands. Storage, gRPC client, custody, view-service, etc.
    /// Empty in v0.
    #[cfg(feature = "embedded")]
    inner: embedded_impl::Inner,
}

impl EmbeddedChain {
    /// Construct an embedded chain. v0 just stores the config.
    /// Full impl: open Storage at `cfg.home`, load custody key,
    /// connect gRPC, kick off the background sync task.
    pub fn new(cfg: EmbeddedConfig) -> Result<Self> {
        #[cfg(feature = "embedded")]
        {
            let inner = embedded_impl::Inner::open(&cfg)?;
            return Ok(Self { cfg, inner });
        }
        #[cfg(not(feature = "embedded"))]
        Ok(Self { cfg })
    }
}

#[cfg(not(feature = "embedded"))]
fn not_implemented(method: &'static str) -> anyhow::Error {
    anyhow!(
        "EmbeddedChain::{method} requires `--features embedded`. \
         Build penumbra_arena with that flag to enable in-process \
         view+custody via penumbra-sdk-* crates."
    )
}

impl PenumbraChain for EmbeddedChain {
    fn account(&self) -> u32 { self.cfg.account_index }

    fn fair_price(&self, _pair: &Pair) -> Result<Quote> {
        #[cfg(feature = "embedded")]
        return self.inner.fair_price(_pair);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("fair_price"))
    }

    fn balance(&self) -> Result<Balance> {
        #[cfg(feature = "embedded")]
        return self.inner.balance(self.cfg.account_index);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("balance"))
    }

    fn depth(&self, _pair: &Pair, _offsets_bps: &[f64]) -> Result<OrderBook> {
        #[cfg(feature = "embedded")]
        return self.inner.depth(_pair, _offsets_bps);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("depth"))
    }

    fn block_features(&self, pair: &Pair) -> Result<BlockFeatures> {
        let quote = self.fair_price(pair)?;
        let balance = self.balance()?;
        let book = self.depth(pair, STD_DEPTH_OFFSETS_BPS)
            .unwrap_or_else(|_| OrderBook::default());
        Ok(BlockFeatures {
            block_height: 0,
            quote,
            balance,
            open_positions: 0,
            book,
        })
    }

    fn open_position(&self, _p: &PositionParams) -> Result<PositionId> {
        #[cfg(feature = "embedded")]
        return self.inner.open_position(self.cfg.account_index, _p);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("open_position"))
    }

    fn close_position(&self, _id: &PositionId) -> Result<()> {
        #[cfg(feature = "embedded")]
        return self.inner.close_position(self.cfg.account_index, _id);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("close_position"))
    }

    fn close_all_positions(&self) -> Result<()> {
        #[cfg(feature = "embedded")]
        return self.inner.close_all_positions(self.cfg.account_index);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("close_all_positions"))
    }

    fn withdraw_all_positions(&self) -> Result<()> {
        #[cfg(feature = "embedded")]
        return self.inner.withdraw_all_positions(self.cfg.account_index);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("withdraw_all_positions"))
    }

    fn pending_fills(&self) -> Result<Vec<Fill>> {
        #[cfg(feature = "embedded")]
        return self.inner.pending_fills(self.cfg.account_index);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("pending_fills"))
    }

    fn transfer_to_account(&self, _to: u32, _value: &str) -> Result<()> {
        #[cfg(feature = "embedded")]
        return self.inner.transfer_to_account(self.cfg.account_index, _to, _value);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("transfer_to_account"))
    }

    fn address_for_account(&self, _account: u32) -> Result<String> {
        #[cfg(feature = "embedded")]
        return self.inner.address_for_account(_account);
        #[cfg(not(feature = "embedded"))]
        Err(not_implemented("address_for_account"))
    }
}

// Helpers used by both the stub and any DepthLevel-emitting impl.
#[allow(dead_code)]
pub(crate) fn empty_book(offsets_bps: &[f64]) -> OrderBook {
    let mut book = OrderBook::default();
    for &off in offsets_bps {
        book.bids.push(DepthLevel { offset_bps: off, total_base: 0.0, min_competitor_fee_bps: 0 });
        book.asks.push(DepthLevel { offset_bps: off, total_base: 0.0, min_competitor_fee_bps: 0 });
    }
    book
}

/// Real implementation lives here, gated behind `--features embedded`.
/// Each function maps onto a concrete penumbra-sdk call.
#[cfg(feature = "embedded")]
mod embedded_impl {
    use super::*;
    use std::str::FromStr;
    use penumbra_sdk_keys::{
        keys::{AddressIndex, SpendKey},
        Address,
    };
    use penumbra_sdk_asset::{
        asset::{self, Denom, Metadata},
        STAKING_TOKEN_ASSET_ID,
    };
    use penumbra_sdk_view::{Storage, ViewServer};
    use tokio::runtime::Runtime;
    use url::Url;
    use std::sync::OnceLock;

    /// Flat representation of a `SimulateTradeResponse` — just the
    /// total output amount in base units. Anything we can't parse
    /// → None; caller treats as "couldn't get a quote".
    struct SwapOut {
        output_uunits: u128,
    }

    fn swap_out_from_response(
        resp: penumbra_sdk_proto::core::component::dex::v1::SimulateTradeResponse,
    ) -> Option<SwapOut> {
        let exec = resp.output?;
        let out = exec.output?;
        let amount = out.amount?;
        Some(SwapOut { output_uunits: amount_to_u128_inner(amount) })
    }

    /// Convert a `proto::num::v1::Amount { lo: u64, hi: u64 }` into u128.
    fn amount_to_u128(
        a: penumbra_sdk_proto::core::num::v1::Amount,
    ) -> Option<u128> {
        Some(amount_to_u128_inner(a))
    }

    fn amount_to_u128_inner(a: penumbra_sdk_proto::core::num::v1::Amount) -> u128 {
        ((a.hi as u128) << 64) | a.lo as u128
    }

    /// Channel-2 USDC asset id, computed once on demand. The id is a
    /// deterministic hash of the IBC denom string `transfer/channel-2/uusdc`,
    /// so we don't need a chain query to know what it is.
    fn usdc_asset_id() -> &'static asset::Id {
        static ID: OnceLock<asset::Id> = OnceLock::new();
        ID.get_or_init(|| {
            let denom = Denom { denom: "transfer/channel-2/uusdc".to_string() };
            Metadata::default_for(&denom)
                .expect("USDC denom is well-formed")
                .id()
        })
    }

    /// Holds the spend key (always loaded — needed for cheap
    /// address derivation) plus a lazily-initialised heavy state
    /// (Storage + tokio runtime + ViewServer sync task) that's only
    /// built on first call to a method that actually needs the chain.
    pub struct Inner {
        /// Custody material — full spend authority. Loaded from the
        /// `[custody] spend_key = "..."` field of pcli's config.toml.
        spend_key: SpendKey,
        /// Cached config so heavy_state can be built on first need.
        cfg: EmbeddedConfig,
        /// Lazy heavy state. `None` until a chain-reading method runs.
        /// Locked behind a Mutex so multiple methods can race-init
        /// safely. After the first successful build, every future
        /// access reuses the same Storage / runtime / view_server.
        heavy: std::sync::Mutex<Option<HeavyState>>,
    }

    /// State that requires a network connection to construct.
    /// Lazy-built on the first balance/depth/fair_price call.
    struct HeavyState {
        storage: Storage,
        runtime: Runtime,
        _view_server: ViewServer,
        /// gRPC channel to the fullnode. Cheap to clone (shared
        /// connection pool); used for ad-hoc queries like
        /// SimulationService::SimulateTrade.
        channel: tonic::transport::Channel,
    }

    /// Subset of pcli's config.toml we care about for address derivation.
    #[derive(serde::Deserialize)]
    struct PcliConfig {
        custody: PcliCustody,
        #[serde(default)]
        grpc_url: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct PcliCustody {
        backend: String,
        spend_key: String,
    }

    impl Inner {
        pub fn open(cfg: &EmbeddedConfig) -> Result<Self> {
            let cfg_path = cfg.custody_path.join("config.toml");
            let cfg_text = std::fs::read_to_string(&cfg_path)
                .map_err(|e| anyhow!("read pcli config {:?}: {e}", cfg_path))?;
            let parsed: PcliConfig = toml::from_str(&cfg_text)
                .map_err(|e| anyhow!("parse pcli config: {e}"))?;
            if parsed.custody.backend != "SoftKms" {
                return Err(anyhow!(
                    "EmbeddedChain only supports SoftKms custody backends \
                     (got `{}`); for hardware/threshold custody, the \
                     CustodyClient trait wiring lands later",
                    parsed.custody.backend,
                ));
            }
            let spend_key = SpendKey::from_str(&parsed.custody.spend_key)
                .map_err(|e| anyhow!("parse spend_key: {e}"))?;

            // Apply pcli config grpc_url as a fallback if EmbeddedConfig
            // didn't provide one. The actual gRPC connection happens
            // lazily inside `heavy()` — config-only construction must
            // never make a network call.
            let mut cfg = cfg.clone();
            if cfg.grpc_url.is_empty() {
                cfg.grpc_url = parsed.grpc_url.unwrap_or_default();
            }

            Ok(Self {
                spend_key,
                cfg,
                heavy: std::sync::Mutex::new(None),
            })
        }

        /// Build (or reuse) the heavy chain state. First call opens
        /// Storage + spawns the sync worker; subsequent calls take
        /// the existing handles. Errors propagate to callers — the
        /// agent decides whether to retry or skip the block.
        fn heavy<R>(&self, f: impl FnOnce(&HeavyState) -> Result<R>) -> Result<R> {
            let mut guard = self.heavy.lock().expect("heavy mutex poisoned");
            if guard.is_none() {
                let node = Url::parse(&self.cfg.grpc_url)
                    .map_err(|e| anyhow!("parse grpc_url `{}`: {e}", self.cfg.grpc_url))?;

                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow!("build tokio runtime: {e}"))?;

                let fvk = self.spend_key.full_viewing_key().clone();
                let storage_path = camino::Utf8PathBuf::from_path_buf(self.cfg.home.clone())
                    .map_err(|p| anyhow!("home path {:?} not utf8", p))?;
                let node_clone = node.clone();
                let storage = runtime.block_on(async move {
                    Storage::load_or_initialize(Some(storage_path), &fvk, node_clone).await
                }).map_err(|e| anyhow!("Storage::load_or_initialize: {e}"))?;

                let storage_for_server = storage.clone();
                let node_for_server = node.clone();
                let view_server = runtime.block_on(async move {
                    ViewServer::new(storage_for_server, node_for_server).await
                }).map_err(|e| anyhow!("ViewServer::new: {e}"))?;

                // Build a separate gRPC channel for ad-hoc queries
                // (SimulationService, DexQueryService). ViewServer
                // already opened one internally; we want a clone-able
                // handle for our own use.
                let node_for_channel = node.clone();
                let channel = runtime.block_on(async move {
                    ViewServer::get_pd_channel(node_for_channel).await
                }).map_err(|e| anyhow!("get_pd_channel: {e}"))?;

                *guard = Some(HeavyState {
                    storage, runtime, _view_server: view_server, channel,
                });
            }
            f(guard.as_ref().expect("heavy state just initialised"))
        }

        /// Simulate a sell (1 UM → USDC) and a buy (0.1 USDC → UM)
        /// via Penumbra's SimulationService gRPC. Returns midpoint
        /// + half-spread observable from the current LP curve. Uses
        /// the heavy-state channel; first call triggers Storage open.
        pub fn fair_price(&self, _pair: &Pair) -> Result<Quote> {
            use penumbra_sdk_proto::core::component::dex::v1::{
                simulation_service_client::SimulationServiceClient,
                SimulateTradeRequest,
            };
            use penumbra_sdk_proto::core::asset::v1 as asset_pb;
            use penumbra_sdk_asset::Value;

            self.heavy(|h| {
                let channel = h.channel.clone();
                let um_id = *STAKING_TOKEN_ASSET_ID;
                let usdc_id = *usdc_asset_id();

                let response: (Option<SwapOut>, Option<SwapOut>) = h.runtime.block_on(async move {
                    let mut client = SimulationServiceClient::new(channel);

                    // Sell probe: 1 UM → USDC
                    let sell_input = Value {
                        amount: 1_000_000u64.into(),
                        asset_id: um_id,
                    };
                    let sell_req = SimulateTradeRequest {
                        input: Some(sell_input.into()),
                        output: Some(asset_pb::AssetId::from(usdc_id)),
                        routing: None,
                    };
                    let sell = client.simulate_trade(sell_req).await
                        .map(|r| swap_out_from_response(r.into_inner()))
                        .ok().flatten();

                    // Buy probe: 0.1 USDC → UM
                    let buy_input = Value {
                        amount: 100_000u64.into(),
                        asset_id: usdc_id,
                    };
                    let buy_req = SimulateTradeRequest {
                        input: Some(buy_input.into()),
                        output: Some(asset_pb::AssetId::from(um_id)),
                        routing: None,
                    };
                    let buy = client.simulate_trade(buy_req).await
                        .map(|r| swap_out_from_response(r.into_inner()))
                        .ok().flatten();

                    Ok::<_, anyhow::Error>((sell, buy))
                })?;

                // sell: 1 UM in (1e6 upenumbra) → some amount of uusdc out
                //   sell_price = uusdc_out / 1e6 / 1 UM (whole units)
                let sell_price = response.0
                    .map(|s| (s.output_uunits as f64) / 1_000_000.0)
                    .unwrap_or(0.0);
                // buy: 0.1 USDC in (1e5 uusdc) → some amount of upenumbra out
                //   buy_price = 0.1 USDC / (upenumbra_out / 1e6) UM
                let buy_price = response.1
                    .map(|b| if b.output_uunits == 0 { 0.0 }
                             else { 0.1 / ((b.output_uunits as f64) / 1_000_000.0) })
                    .unwrap_or(0.0);
                let mid = if sell_price > 0.0 && buy_price > 0.0 {
                    (sell_price + buy_price) / 2.0
                } else {
                    sell_price.max(buy_price)
                };
                Ok(Quote { sell_price, buy_price, mid })
            })
        }

        /// Read Storage's current note set, filtered to this agent's
        /// AddressIndex, and aggregate UM + USDC totals. Sums entries
        /// whose `id` matches `STAKING_TOKEN_ASSET_ID` into `base`,
        /// entries matching the channel-2 USDC asset id into `quote`.
        /// Other assets are ignored — this MM is a UM/USDC bot.
        ///
        /// Returns whatever the sync worker has scanned so far —
        /// empty until the initial chain sync catches up. Returned
        /// units are whole tokens (upenumbra → UM via /1e6, uusdc →
        /// USDC via /1e6).
        pub fn balance(&self, account: u32) -> Result<Balance> {
            self.heavy(|h| {
                let storage = h.storage.clone();
                let entries = h.runtime.block_on(async move {
                    storage.balances(Some(AddressIndex::new(account)), None).await
                }).map_err(|e| anyhow!("storage.balances: {e}"))?;

                let um_id = *STAKING_TOKEN_ASSET_ID;
                let usdc_id = *usdc_asset_id();
                let mut base = 0u128;
                let mut quote = 0u128;
                for entry in entries {
                    if entry.id == um_id { base += entry.amount; }
                    else if entry.id == usdc_id { quote += entry.amount; }
                }
                Ok(Balance {
                    // upenumbra has 6 decimals
                    base:  (base  as f64) / 1_000_000.0,
                    // uusdc also has 6 decimals
                    quote: (quote as f64) / 1_000_000.0,
                })
            })
        }

        /// Probe the LP curve via DexQueryService. Streams positions
        /// sorted by effective price closest to the directed pair's
        /// numéraire, aggregates reserves into the standard offset
        /// buckets (≤10, ≤50, ≤100, ≤200 bps from mid).
        ///
        /// v0 simplification: every position contributes to BOTH
        /// bid and ask sides (CFMM positions trade both directions
        /// within their range). Refinement to per-side accounting
        /// is a follow-up once the BarbellDecoder consumes side info.
        pub fn depth(&self, _pair: &Pair, offsets_bps: &[f64]) -> Result<OrderBook> {
            use penumbra_sdk_proto::core::component::dex::v1::{
                query_service_client::QueryServiceClient,
                LiquidityPositionsByPriceRequest, DirectedTradingPair,
            };
            use penumbra_sdk_proto::core::asset::v1 as asset_pb;
            use futures::StreamExt;

            self.heavy(|h| {
                let channel = h.channel.clone();
                let um_id = *STAKING_TOKEN_ASSET_ID;
                let usdc_id = *usdc_asset_id();

                // Fetch the current mid first — needed to compute offsets.
                // Reuse fair_price logic by recomputing here (lightweight).
                let pair_for_quote = Pair::um_usdc();
                let quote = self.fair_price(&pair_for_quote)?;
                let mid = quote.mid;
                if mid <= 0.0 {
                    // No mid available → return empty book rather than error.
                    return Ok(empty_book(offsets_bps));
                }

                let positions = h.runtime.block_on(async move {
                    let mut client = QueryServiceClient::new(channel);
                    let req = LiquidityPositionsByPriceRequest {
                        trading_pair: Some(DirectedTradingPair {
                            start: Some(asset_pb::AssetId::from(um_id)),
                            end:   Some(asset_pb::AssetId::from(usdc_id)),
                        }),
                        limit: 50,
                    };
                    // Wrap the whole stream collection in a 10-second
                    // timeout. Server-streaming gRPC can keep the
                    // stream open on quiet pairs; we'd rather take an
                    // empty book than block the brain forever.
                    let stream_fut = async {
                        let mut stream = client.liquidity_positions_by_price(req).await
                            .map_err(|e| anyhow!("liquidity_positions_by_price: {e}"))?
                            .into_inner();
                        let mut out: Vec<(f64, u128)> = Vec::new();
                        while let Some(item) = stream.next().await {
                            let resp = match item { Ok(r) => r, Err(_) => continue };
                            let pos = match resp.data { Some(p) => p, None => continue };
                            let phi = match pos.phi { Some(p) => p, None => continue };
                            let bare = match phi.component { Some(b) => b, None => continue };
                            let p_amt = match bare.p.and_then(amount_to_u128) { Some(v) => v, None => continue };
                            let q_amt = match bare.q.and_then(amount_to_u128) { Some(v) => v, None => continue };
                            if q_amt == 0 { continue; }
                            let price = (p_amt as f64) / (q_amt as f64);
                            let reserves = pos.reserves.unwrap_or_default();
                            let r1 = reserves.r1.and_then(amount_to_u128).unwrap_or(0);
                            let r2 = reserves.r2.and_then(amount_to_u128).unwrap_or(0);
                            let liq_base = (r1 as f64) / 1_000_000.0
                                + if price > 0.0 { (r2 as f64) / 1_000_000.0 / price } else { 0.0 };
                            out.push((price, liq_base as u128));
                        }
                        Ok::<_, anyhow::Error>(out)
                    };
                    match tokio::time::timeout(std::time::Duration::from_secs(10), stream_fut).await {
                        Ok(Ok(v)) => Ok(v),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Ok(Vec::new()),  // timeout — return empty book
                    }
                })?;

                // Bucket by offset_bps. Each level accumulates total
                // base-asset liquidity for positions inside that band.
                // Both sides get the same population in v0.
                let mut book = OrderBook::default();
                for &off in offsets_bps {
                    let band = mid * (off / 10_000.0);
                    let total: u128 = positions.iter()
                        .filter(|(p, _)| (p - mid).abs() <= band)
                        .map(|(_, l)| *l).sum();
                    let level = DepthLevel {
                        offset_bps: off,
                        total_base: total as f64,
                        min_competitor_fee_bps: 0,
                    };
                    book.bids.push(level);
                    book.asks.push(level);
                }
                Ok(book)
            })
        }

        // ─── Write-side methods (tx submission) ─────────────────
        //
        // **v0 architectural delegation.** EmbeddedChain is read-only.
        // Tx submission requires the full (Planner → TransactionPlan
        // → AuthorizeRequest → Transaction → BroadcastTxSync) flow,
        // which depends on a `ViewClient`-implementing handle that
        // ViewServer doesn't expose directly. The in-process
        // integration needs an internal tonic-localhost bridge or a
        // tower::Service adapter — multi-day scope unrelated to the
        // brain's read-side decision-making.
        //
        // Until that lands, the architecture is:
        //   * `EmbeddedChain` — fast in-process reads (balance, fair_price,
        //     depth, address derivation) used by the agent every block
        //     for brain inputs.
        //   * `PcliChain` — subprocess-pcli writes (open_position etc.)
        //     used at submission time. This is NOT pclientd-the-daemon
        //     (one-shot CLI invocation per tx, no long-running process).
        //
        // The arena binary holds both: `EmbeddedChain` for the
        // BlockFeatures pipeline, `PcliChain` for the diff-and-submit
        // pipeline. Future slice promotes tx submission into
        // `EmbeddedChain` via penumbra-sdk-transaction directly.
        //
        // These methods all return `not_implemented` — callers should
        // route writes through `PcliChain` for now.

        pub fn open_position(&self, _account: u32, _p: &PositionParams) -> Result<PositionId> {
            Err(anyhow!("EmbeddedChain is read-only in v0 — route open_position through PcliChain"))
        }

        pub fn close_position(&self, _account: u32, _id: &PositionId) -> Result<()> {
            Err(anyhow!("EmbeddedChain is read-only in v0 — route close_position through PcliChain"))
        }

        pub fn close_all_positions(&self, _account: u32) -> Result<()> {
            Err(anyhow!("EmbeddedChain is read-only in v0 — route close_all_positions through PcliChain"))
        }

        pub fn withdraw_all_positions(&self, _account: u32) -> Result<()> {
            Err(anyhow!("EmbeddedChain is read-only in v0 — route withdraw_all_positions through PcliChain"))
        }

        pub fn pending_fills(&self, _account: u32) -> Result<Vec<Fill>> {
            // Reads but needs cross-block diffing (cache last-seen
            // reserves per position, compare). v0 returns empty —
            // implementation lands when the agent loop needs it.
            Ok(Vec::new())
        }

        pub fn transfer_to_account(&self, _from: u32, _to: u32, _value: &str) -> Result<()> {
            Err(anyhow!("EmbeddedChain is read-only in v0 — route transfer_to_account through PcliChain"))
        }

        /// Derive the bech32 payment address for `account`. Works
        /// fully in-process — no chain query, no subprocess. This is
        /// the first method to graduate from stub to real impl.
        pub fn address_for_account(&self, account: u32) -> Result<String> {
            let fvk = self.spend_key.full_viewing_key();
            let (addr, _detection_key): (Address, _) =
                fvk.payment_address(AddressIndex::new(account));
            Ok(addr.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_unfeatured_error_without_feature() {
        let cfg = EmbeddedConfig {
            home: "/tmp/penumbra_arena_agent_1".into(),
            grpc_url: "http://localhost:8080".into(),
            account_index: 1,
            custody_path: "/home/alice/.local/share/pcli".into(),
        };
        let chain = EmbeddedChain::new(cfg).expect("config-only construction");
        assert_eq!(chain.account(), 1);

        // Without --features embedded, every PenumbraChain method
        // returns the not_implemented error rather than panicking.
        // (When the feature is on, these should hit the real impl
        // which currently also Errs with TODO messages.)
        let pair = Pair::um_usdc();
        assert!(chain.fair_price(&pair).is_err());
        assert!(chain.balance().is_err());
    }

    /// Exercises the real address derivation path against the user's
    /// pcli wallet at the default location. Skipped if the wallet
    /// isn't present so CI on a fresh box stays green.
    #[cfg(feature = "embedded")]
    #[test]
    fn embedded_address_derivation_matches_pcli_for_account_0() {
        let pcli_home = std::path::PathBuf::from("/home/alice/.local/share/pcli");
        if !pcli_home.join("config.toml").exists() {
            eprintln!("pcli config absent at {pcli_home:?}, skipping");
            return;
        }
        let cfg = EmbeddedConfig {
            home: "/tmp/penumbra_arena_test_home".into(),
            grpc_url: "https://penumbra.rotko.net".into(),
            account_index: 0,
            custody_path: pcli_home.clone(),
        };
        let chain = EmbeddedChain::new(cfg).expect("EmbeddedChain::new");
        let addr_0 = chain.address_for_account(0).expect("address account 0");
        let addr_1 = chain.address_for_account(1).expect("address account 1");

        assert!(addr_0.starts_with("penumbra1"), "addr 0 not bech32: {addr_0}");
        assert!(addr_1.starts_with("penumbra1"), "addr 1 not bech32: {addr_1}");
        assert_ne!(addr_0, addr_1, "different account indices must yield different addresses");

        // Cross-check against `pcli view address <N>` if available.
        // Both should agree for the same wallet, modulo deterministic FVK derivation.
        if let Ok(out0) = std::process::Command::new("pcli").args(["view", "address", "0"]).output() {
            if out0.status.success() {
                let pcli_addr_0 = String::from_utf8_lossy(&out0.stdout)
                    .lines().find(|l| l.starts_with("penumbra1"))
                    .unwrap_or("").trim().to_string();
                assert_eq!(addr_0, pcli_addr_0,
                    "embedded vs pcli account 0 disagree:\n  embedded: {addr_0}\n  pcli:     {pcli_addr_0}");
            }
        }
    }

    /// Exercises the real balance path against the user's pcli wallet.
    /// Skipped if the wallet is absent. Requires network access — the
    /// first call opens Storage and waits for initial sync. Set
    /// `EMBEDDED_BALANCE_PROBE=1` to run; default skips so CI stays fast.
    #[cfg(feature = "embedded")]
    #[test]
    fn embedded_balance_smoke_against_pcli_wallet() {
        if std::env::var_os("EMBEDDED_BALANCE_PROBE").is_none() {
            eprintln!("skipping balance probe (set EMBEDDED_BALANCE_PROBE=1 to run)");
            return;
        }
        let pcli_home = std::path::PathBuf::from("/home/alice/.local/share/pcli");
        if !pcli_home.join("config.toml").exists() {
            eprintln!("pcli config absent at {pcli_home:?}, skipping");
            return;
        }
        let cfg = EmbeddedConfig {
            home: "/tmp/penumbra_arena_balance_probe".into(),
            grpc_url: String::new(),       // pulled from pcli config
            account_index: 0,
            custody_path: pcli_home,
        };
        let chain = EmbeddedChain::new(cfg).expect("EmbeddedChain::new");
        let bal = chain.balance().expect("balance");
        eprintln!("account 0  base={:.4} UM  quote={:.4} USDC", bal.base, bal.quote);
        assert!(bal.base >= 0.0);
        assert!(bal.quote >= 0.0);
    }

    /// Real fair_price probe. Hits SimulationService gRPC; rate-
    /// limited so the test only runs with EMBEDDED_BALANCE_PROBE=1.
    #[cfg(feature = "embedded")]
    #[test]
    fn embedded_fair_price_smoke_against_pcli_wallet() {
        if std::env::var_os("EMBEDDED_BALANCE_PROBE").is_none() {
            eprintln!("skipping fair_price probe (set EMBEDDED_BALANCE_PROBE=1 to run)");
            return;
        }
        let pcli_home = std::path::PathBuf::from("/home/alice/.local/share/pcli");
        if !pcli_home.join("config.toml").exists() { return; }
        let cfg = EmbeddedConfig {
            home: "/tmp/penumbra_arena_fair_price_probe".into(),
            grpc_url: String::new(),
            account_index: 0,
            custody_path: pcli_home,
        };
        let chain = EmbeddedChain::new(cfg).expect("EmbeddedChain::new");
        let pair = Pair::um_usdc();
        let q = chain.fair_price(&pair).expect("fair_price");
        eprintln!("UM/USDC  sell={:.6}  buy={:.6}  mid={:.6}",
            q.sell_price, q.buy_price, q.mid);
        assert!(q.sell_price >= 0.0);
        assert!(q.buy_price >= 0.0);
    }

    /// Real depth probe via DexQueryService gRPC. Hits the live
    /// chain, gated by EMBEDDED_BALANCE_PROBE=1.
    #[cfg(feature = "embedded")]
    #[test]
    fn embedded_depth_smoke_against_pcli_wallet() {
        if std::env::var_os("EMBEDDED_BALANCE_PROBE").is_none() {
            eprintln!("skipping depth probe (set EMBEDDED_BALANCE_PROBE=1 to run)");
            return;
        }
        let pcli_home = std::path::PathBuf::from("/home/alice/.local/share/pcli");
        if !pcli_home.join("config.toml").exists() { return; }
        let cfg = EmbeddedConfig {
            home: "/tmp/penumbra_arena_depth_probe".into(),
            grpc_url: String::new(),
            account_index: 0,
            custody_path: pcli_home,
        };
        let chain = EmbeddedChain::new(cfg).expect("EmbeddedChain::new");
        let pair = Pair::um_usdc();
        let book = chain.depth(&pair, &[10.0, 50.0, 100.0, 200.0]).expect("depth");
        eprintln!("UM/USDC depth (UM-base units within band):");
        for (b, a) in book.bids.iter().zip(book.asks.iter()) {
            eprintln!("  ±{:>5.0} bps  bid={:>14.2}  ask={:>14.2}",
                b.offset_bps, b.total_base, a.total_base);
        }
        assert_eq!(book.bids.len(), 4);
    }

    #[test]
    fn empty_book_helper_has_offsets_at_each_level() {
        let book = empty_book(STD_DEPTH_OFFSETS_BPS);
        assert_eq!(book.bids.len(), STD_DEPTH_OFFSETS_BPS.len());
        for (i, &off) in STD_DEPTH_OFFSETS_BPS.iter().enumerate() {
            assert!((book.bids[i].offset_bps - off).abs() < 1e-9);
        }
    }
}
