//! Single-agent Penumbra MM dry-run.
//!
//! Builds an `MmAgent` with a small typed brain (raw_obs_dim=16) and
//! a `MockChain` so the wiring runs without any real Penumbra testnet
//! connection. Useful as a smoke test for the crate; swap the chain
//! for `PcliChain { dry_run: true }` to print actual pcli commands
//! against your local pcli config.

use anyhow::Result;
use std::cell::RefCell;

use modgrad_ctm::config::{CtmConfig, ExitStrategy};
use modgrad_ctm::graph::{
    AuxLossConfig, Connection, RegionalConfig, RegionalWeights, RegionalWeightsTyped,
};
use modgrad_device::backend::tensor::Cpu;

use penumbra_arena::agent::MmAgent;
use penumbra_arena::chain::{
    Balance, BlockFeatures, DepthLevel, Fill, OrderBook, Pair, PenumbraChain,
    PositionId, PositionParams, Quote, STD_DEPTH_OFFSETS_BPS,
};
use penumbra_arena::features::RAW_OBS_DIM;
use penumbra_arena::motor::BarbellConfig;

/// Mock chain that fakes a tiny UM/USDC market for dry-run testing.
/// Mid drifts on a sine wave; balances are static.
///
/// Defaults match the arena seed allocation per agent:
///   - 10,000 UM (base)
///   - 50 USDC  (quote)
///   - 0.005 USDC/UM mid (current real-world UM/USDC price)
/// Total per-agent capital = $100 at neutral 50/50 inventory.
struct MockChain {
    block: RefCell<u64>,
    base_mid: f64,
    base_balance: f64,
    quote_balance: f64,
    state: RefCell<MockState>,
}

#[derive(Default)]
struct MockState {
    open_ids: Vec<String>,
    next_id: u64,
}

impl MockChain {
    fn new() -> Self {
        Self {
            block: RefCell::new(0),
            base_mid: 0.005,
            base_balance: 10_000.0,
            quote_balance: 50.0,
            state: RefCell::new(MockState::default()),
        }
    }

    fn current_quote(&self) -> Quote {
        let b = *self.block.borrow();
        let drift = ((b as f64) * 0.31).sin() * 1e-4;
        let mid = self.base_mid + drift;
        let half_spread = mid * 0.005;  // 100 bps half-spread
        Quote {
            sell_price: mid - half_spread,
            buy_price:  mid + half_spread,
            mid,
        }
    }
}

impl PenumbraChain for MockChain {
    fn fair_price(&self, _pair: &Pair) -> Result<Quote> { Ok(self.current_quote()) }
    fn balance(&self) -> Result<Balance> {
        Ok(Balance { base: self.base_balance, quote: self.quote_balance })
    }
    fn depth(&self, _pair: &Pair, offsets_bps: &[f64]) -> Result<OrderBook> {
        // Synthetic depth: thin near mid, fattens with offset. Real
        // chain (PcliChain) reads positions out of `pcli query dex`.
        let mid = self.current_quote().mid;
        let mut book = OrderBook::default();
        for &off in offsets_bps {
            let frac = off / 10_000.0;
            let v = mid * 200_000.0 * frac;
            book.bids.push(DepthLevel { offset_bps: off, total_base: v, min_competitor_fee_bps: 30 });
            book.asks.push(DepthLevel { offset_bps: off, total_base: v, min_competitor_fee_bps: 30 });
        }
        Ok(book)
    }
    fn block_features(&self, pair: &Pair) -> Result<BlockFeatures> {
        // Increment the block counter, then drop the borrow before
        // calling `current_quote` which takes its own immutable borrow.
        let block_height = {
            let mut b = self.block.borrow_mut();
            *b += 1;
            *b
        };
        let quote = self.current_quote();
        let open_positions = self.state.borrow().open_ids.len() as u32;
        let book = self.depth(pair, STD_DEPTH_OFFSETS_BPS)?;
        Ok(BlockFeatures {
            block_height,
            quote,
            balance: Balance { base: self.base_balance, quote: self.quote_balance },
            open_positions,
            book,
        })
    }
    fn open_position(&self, p: &PositionParams) -> Result<PositionId> {
        let mut st = self.state.borrow_mut();
        st.next_id += 1;
        let id = format!("mock-{}-{}", p.side, st.next_id);
        st.open_ids.push(id.clone());
        eprintln!("  + open  {:>4} @ {:.6}  size={:.4} fee={}bps",
            p.side, p.price, p.amount, p.fee_bps);
        Ok(PositionId(id))
    }
    fn close_position(&self, id: &PositionId) -> Result<()> {
        self.state.borrow_mut().open_ids.retain(|x| x != &id.0);
        eprintln!("  - close {}", id.0);
        Ok(())
    }
    fn close_all_positions(&self) -> Result<()> {
        let n = self.state.borrow().open_ids.len();
        self.state.borrow_mut().open_ids.clear();
        if n > 0 { eprintln!("  - close-all (n={n})"); }
        Ok(())
    }
    fn withdraw_all_positions(&self) -> Result<()> { Ok(()) }
    fn pending_fills(&self) -> Result<Vec<Fill>> { Ok(Vec::new()) }
    fn account(&self) -> u32 { 0 }
    fn transfer_to_account(&self, _to: u32, _value: &str) -> Result<()> { Ok(()) }
    fn address_for_account(&self, account: u32) -> Result<String> {
        Ok(format!("penumbra-mock-account-{account}"))
    }
}

fn build_small_brain() -> RegionalWeights {
    let mut cfg = small_eight_region_mm_cfg();
    cfg.exit_strategy = ExitStrategy::None;
    cfg.router = None;
    RegionalWeights::new(cfg)
}

/// Tiny 8-region preset matching the encoder shape: raw_obs_dim=16,
/// out_dims=12 (barbell motor). Lifted from `eight_region_small`
/// pattern with shape tweaks.
fn small_eight_region_mm_cfg() -> RegionalConfig {
    let obs_dim = RAW_OBS_DIM;
    let out_dims = penumbra_arena::motor::MOTOR_DIM;
    let ticks = 1;

    // Eight tiny regions. Cortical four are slightly bigger than
    // subcortical to match the eight_region intuition.
    let d_model_cortical = 32;
    let d_model_sub      = 16;

    let mk = |d_model: usize| CtmConfig {
        iterations: 1,
        d_model,
        d_input: 16,
        heads: 2,
        n_synch_out: 8,
        n_synch_action: 8,
        synapse_depth: 1,
        memory_length: 8,
        deep_nlms: false,
        memory_hidden_dims: 4,
        out_dims,
        n_random_pairing_self: 0,
        min_width: 8,
        ..Default::default()
    };

    let regions = vec![
        mk(d_model_cortical),  // 0 INPUT
        mk(d_model_cortical),  // 1 ATTENTION
        mk(d_model_cortical),  // 2 OUTPUT
        mk(d_model_cortical),  // 3 MOTOR
        mk(d_model_sub),       // 4 CEREBELLUM
        mk(d_model_sub),       // 5 BASAL_GANGLIA
        mk(d_model_sub),       // 6 INSULA
        mk(d_model_sub),       // 7 HIPPOCAMPUS
    ];
    let region_names: Vec<String> = [
        "input","attention","output","motor",
        "cerebellum","basal_ganglia","insula","hippocampus",
    ].into_iter().map(String::from).collect();

    let connections = vec![
        Connection { from: vec![3], to: 0, receives_observation: true,  observation_scale: 0 },
        Connection { from: vec![0, 7], to: 1, receives_observation: false, observation_scale: 0 },
        Connection { from: vec![1], to: 2, receives_observation: false, observation_scale: 0 },
        Connection { from: vec![2], to: 3, receives_observation: false, observation_scale: 0 },
        Connection { from: vec![3], to: 4, receives_observation: true,  observation_scale: 0 },
        Connection { from: vec![2], to: 5, receives_observation: false, observation_scale: 0 },
        Connection { from: vec![7], to: 6, receives_observation: false, observation_scale: 0 },
        Connection { from: vec![0,1,2,3], to: 7, receives_observation: false, observation_scale: 0 },
    ];

    let total_neurons: usize = regions.iter().map(|r| r.d_model).sum();
    let n_global_sync = total_neurons.min(64);

    RegionalConfig {
        regions,
        region_names,
        connections,
        outer_ticks: ticks,
        exit_strategy: ExitStrategy::None,
        n_global_sync,
        out_dims,
        raw_obs_dim: obs_dim,
        obs_scale_dims: vec![obs_dim],
        aux_losses: AuxLossConfig::default(),
        router: None,
        cereb_mode: Default::default(),
    }
}

fn main() -> Result<()> {
    eprintln!("== penumbra_arena dry-run ==");
    let untyped = build_small_brain();
    eprintln!("brain params = {}", untyped.n_params());
    let typed = RegionalWeightsTyped::<Cpu>::from_untyped(&untyped)
        .map_err(|e| anyhow::anyhow!("from_untyped: {e}"))?;

    let chain = MockChain::new();
    let pair = Pair::um_usdc();
    let motor_cfg = BarbellConfig::default();
    let mut agent = MmAgent::<Cpu, MockChain>::new(&untyped, typed, chain, pair, motor_cfg)?;

    let n_steps: usize = std::env::var("N_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    for step_i in 0..n_steps {
        eprintln!("\n— block {step_i} —");
        let positions = agent.step()?;
        eprintln!("  brain placed {} positions", positions.len());
    }
    eprintln!("\n== done ==");
    Ok(())
}
