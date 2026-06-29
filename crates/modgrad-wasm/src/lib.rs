//! Browser bindings for the modgrad SDK.
//!
//! This is the REAL SDK compiled to wasm — not a reimplementation. Each export
//! is a thin wrapper over `modgrad_ctm` (and, later, `modgrad_codec`) holding a
//! `thread_local` model and forwarding the demo's calls into the library. The
//! `_inner` functions carry the logic with plain `Result<_, String>` so native
//! tests can exercise them without a JS runtime; the `#[wasm_bindgen]` exports
//! just adapt the error type.

use std::cell::RefCell;

use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::graph::{RegionalState, RegionalTick, RegionalWeights, regional_forward_trace};
use modgrad_ctm::vin::VinReadout;
use wasm_bindgen::prelude::*;

thread_local! {
    /// The trained VIN planner, loaded from the SDK's serde JSON export.
    static LEARNED_VIN: RefCell<Option<VinReadout>> = const { RefCell::new(None) };
    /// Pristine move-head weights, snapshotted before the first consolidation
    /// so `learned_vin_reset` can undo every sleep/dream pass.
    static VIN_PRISTINE: RefCell<Option<(Vec<f32>, Vec<f32>)>> = const { RefCell::new(None) };
    /// The 8-region brain and its visual cortex (retina), loaded together from
    /// the SDK's `{cortex, regional}` serde export.
    static BRAIN: RefCell<Option<RegionalWeights>> = const { RefCell::new(None) };
    static CORTEX: RefCell<Option<VisualCortex>> = const { RefCell::new(None) };
}

fn jserr(e: String) -> JsValue {
    JsValue::from_str(&e)
}

// ── VIN planner ──────────────────────────────────────────────────────────────

fn load_learned_vin_inner(json: &str) -> Result<(), String> {
    let vin: VinReadout =
        serde_json::from_str(json).map_err(|e| format!("load_learned_vin: {e}"))?;
    LEARNED_VIN.with(|v| *v.borrow_mut() = Some(vin));
    VIN_PRISTINE.with(|s| *s.borrow_mut() = None); // fresh model → drop old snapshot
    Ok(())
}

/// Run the learned VIN forward and return `[move_logits (n_dirs) | value-field
/// compass (grid_h*grid_w)]`. The compass cell value is the L1 magnitude of that
/// cell's value vector — the planner's own proximity-to-goal estimate.
fn learned_vin_forward_compass_inner(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
) -> Result<Vec<f32>, String> {
    LEARNED_VIN.with(|v| {
        let vref = v.borrow();
        let vin = vref
            .as_ref()
            .ok_or_else(|| "learned_vin_forward_compass: no VIN loaded".to_string())?;
        let out = vin.forward(tokens, grid_h, grid_w, Some((agent_r, agent_c)));
        let n = grid_h * grid_w;
        let vd = out.value_dim;
        let mut result = Vec::with_capacity(out.move_logits.len() + n);
        result.extend_from_slice(&out.move_logits);
        for cell in 0..n {
            let base = cell * vd;
            let mut s = 0.0f32;
            for d in 0..vd {
                s += out.value_grid[base + d].abs();
            }
            result.push(s);
        }
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn load_learned_vin(json: &str) -> Result<(), JsValue> {
    load_learned_vin_inner(json).map_err(jserr)
}

#[wasm_bindgen]
pub fn learned_vin_forward_compass(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
) -> Result<Vec<f32>, JsValue> {
    learned_vin_forward_compass_inner(tokens, grid_h, grid_w, agent_r, agent_c).map_err(jserr)
}

/// One sleep/consolidation step on the move head toward `target_move`; returns
/// the cross-entropy loss BEFORE the step (`lr = 0` → measure-only). Snapshots
/// the pristine move head once so `learned_vin_reset` can undo every pass.
fn learned_vin_train_inner(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
    target_move: i32,
    lr: f32,
) -> Result<f32, String> {
    if target_move < 0 {
        return Ok(0.0);
    }
    LEARNED_VIN.with(|v| {
        let mut vref = v.borrow_mut();
        let vin = vref
            .as_mut()
            .ok_or_else(|| "learned_vin_train: no VIN loaded".to_string())?;
        VIN_PRISTINE.with(|s| {
            if s.borrow().is_none() {
                *s.borrow_mut() =
                    Some((vin.move_head.weight.clone(), vin.move_head.bias.clone()));
            }
        });
        Ok(vin.consolidate_move(
            tokens,
            grid_h,
            grid_w,
            (agent_r, agent_c),
            target_move as usize,
            lr,
        ))
    })
}

/// Restore the move head to its as-loaded weights, undoing every consolidation.
fn learned_vin_reset_inner() {
    VIN_PRISTINE.with(|s| {
        if let Some((w, b)) = s.borrow().as_ref() {
            LEARNED_VIN.with(|v| {
                if let Some(vin) = v.borrow_mut().as_mut() {
                    vin.move_head.weight = w.clone();
                    vin.move_head.bias = b.clone();
                }
            });
        }
    });
}

#[wasm_bindgen]
pub fn learned_vin_train(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
    target_move: i32,
    lr: f32,
) -> Result<f32, JsValue> {
    learned_vin_train_inner(tokens, grid_h, grid_w, agent_r, agent_c, target_move, lr)
        .map_err(jserr)
}

#[wasm_bindgen]
pub fn learned_vin_reset() {
    learned_vin_reset_inner();
}

#[cfg(test)]
mod tests {
    use super::*;

    // The demo ships the SDK's own serde export; the real VinReadout must
    // deserialize it and produce a sane forward (4 move logits + a value field).
    const VIN_JSON: &str = include_str!(
        "../../../../modgrad.com/public/models/vin_solver_weights.json"
    );

    #[test]
    fn loads_real_vin_export_and_runs_forward() {
        load_learned_vin_inner(VIN_JSON).expect("real VIN json should deserialize");

        // raw_dim is 3 for this model; build a tiny 3x3 grid of zero tokens with
        // the agent at the centre. We only assert shape + finiteness here — the
        // bit-exact parity check happens against the golden vectors in-browser.
        let (h, w, raw_dim) = (3usize, 3usize, 3usize);
        let tokens = vec![0.0f32; h * w * raw_dim];
        let out = learned_vin_forward_compass_inner(&tokens, h, w, 1, 1)
            .expect("forward should run");

        assert_eq!(out.len(), 4 + h * w, "‹4 move logits | h*w compass›");
        assert!(out.iter().all(|x| x.is_finite()), "all outputs finite");
    }

    #[test]
    fn train_then_reset_round_trips_the_move_head() {
        load_learned_vin_inner(VIN_JSON).unwrap();
        let snapshot = || {
            LEARNED_VIN.with(|v| v.borrow().as_ref().unwrap().move_head.weight.clone())
        };
        let before = snapshot();

        let (h, w, raw_dim) = (3usize, 3usize, 3usize);
        let tokens = vec![0.05f32; h * w * raw_dim];

        // measure-only must not mutate
        let loss0 = learned_vin_train_inner(&tokens, h, w, 1, 1, 1, 0.0).unwrap();
        assert!(loss0.is_finite());
        assert_eq!(snapshot(), before, "lr=0 is measure-only");

        // a real step changes the move head, then reset restores it exactly
        let _ = learned_vin_train_inner(&tokens, h, w, 1, 1, 1, 0.3).unwrap();
        assert_ne!(snapshot(), before, "a consolidation step moves the weights");
        learned_vin_reset_inner();
        assert_eq!(snapshot(), before, "reset restores pristine move head");
    }

    // Loads the REAL 8-region brain export and runs the full pixel→retina→brain
    // forward, asserting the per-tick BrainOut shape the demo viz consumes.
    #[test]
    fn loads_real_brain_and_runs_pixel_forward() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../modgrad.com/public/models/brain_solver_weights.json"
        );
        let json = match std::fs::read_to_string(path) {
            Ok(j) => j,
            Err(_) => {
                eprintln!("skip: brain_solver_weights.json not found at {path}");
                return;
            }
        };
        load_brain_weights_inner(&json).expect("real brain json should deserialize");

        let (ih, iw) = CORTEX.with(|c| {
            let r = c.borrow();
            let v = r.as_ref().expect("brain export carries a visual cortex");
            (v.input_h, v.input_w)
        });
        let pixels = vec![1.0f32; 3 * ih * iw]; // all-white maze
        let out = run_brain_pixels_core(&pixels).expect("brain forward should run");

        assert!(out.ticks_used >= 1, "at least one outer tick");
        assert_eq!(out.ticks.len(), out.ticks_used, "one trace entry per tick");
        let t0 = &out.ticks[0];
        assert!(!t0.region_activations.is_empty(), "per-region activations present");
        assert!(!t0.global_sync.is_empty(), "global sync present");
        assert!(
            t0.prediction.iter().all(|x| x.is_finite()),
            "predictions finite"
        );
    }
}

// ── 8-region brain + visual cortex ───────────────────────────────────────────

/// The per-forward introspection the demo's 3D viz animates: one entry per
/// outer tick. Field names match the SDK's `RegionalTick`, so serde→JS yields
/// the exact object the worker reads.
#[derive(serde::Serialize)]
struct BrainOut {
    ticks: Vec<RegionalTick>,
    ticks_used: usize,
}

/// The brain export is `{cortex, regional}`: an optional visual cortex (retina
/// pathway) plus the regional brain. Loaded straight into the real SDK types.
#[derive(serde::Deserialize)]
struct BrainWeights {
    #[serde(default)]
    cortex: Option<VisualCortex>,
    regional: RegionalWeights,
}

fn load_brain_weights_inner(json: &str) -> Result<(), String> {
    let bw: BrainWeights =
        serde_json::from_str(json).map_err(|e| format!("load_brain_weights: {e}"))?;
    CORTEX.with(|c| *c.borrow_mut() = bw.cortex);
    BRAIN.with(|b| *b.borrow_mut() = Some(bw.regional));
    Ok(())
}

/// Pixels `[3 × SIZE × SIZE]` (CHW, ∈[0,1]) → retina → V4 tokens → 8-region
/// brain forward, returning the full per-tick trace as `BrainOut`. Pure Rust so
/// native tests can exercise it; the wasm export wraps it with serde→JS.
fn run_brain_pixels_core(pixels: &[f32]) -> Result<BrainOut, String> {
    CORTEX.with(|c| {
        let cortex_ref = c.borrow();
        let cortex = cortex_ref.as_ref().ok_or_else(|| {
            "run_brain_pixels: no visual cortex loaded (this brain has no retina)".to_string()
        })?;
        let (obs, _n, _d) = cortex.spatial_tokens(pixels);
        BRAIN.with(|b| {
            let brain_ref = b.borrow();
            let brain = brain_ref
                .as_ref()
                .ok_or_else(|| "run_brain_pixels: weights not loaded".to_string())?;
            let mut state = RegionalState::new(brain);
            let (out, ticks) = regional_forward_trace(brain, &mut state, &obs);
            Ok(BrainOut { ticks, ticks_used: out.ticks_used })
        })
    })
}

/// One feature-map stage for the retina overlay (CHW `data`).
#[derive(serde::Serialize)]
struct RetinaMap {
    name: String,
    channels: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

// token layout (HWC, `[n_tokens × channels]`) → CHW `[channels × h × w]`.
fn tokens_to_chw(tok: &[f32], n: usize, ch: usize, h: usize, w: usize) -> Vec<f32> {
    let mut chw = vec![0.0f32; ch * n];
    for ti in 0..n {
        let y = ti / w;
        let x = ti % w;
        for c in 0..ch {
            chw[c * n + y * w + x] = tok[ti * ch + c];
        }
    }
    chw
}

fn retina_maps_value(pixels: &[f32]) -> Result<JsValue, String> {
    CORTEX.with(|c| {
        let cortex_ref = c.borrow();
        let cortex = cortex_ref
            .as_ref()
            .ok_or_else(|| "retina_maps: no visual cortex loaded".to_string())?;
        // [V1, V2, V4, ganglion(=retina)] as (tokens, n_tokens, channels).
        let [v1, v2, v4, retina] = cortex.spatial_tokens_multiscale(pixels);
        let stage = |name: &str, (tok, n, ch): &(Vec<f32>, usize, usize)| {
            // spatial side: spatial_tokens_multiscale keeps each stage square-ish;
            // recover h,w from n by assuming a square map (the demo maze is square).
            let side = (*n as f64).sqrt() as usize;
            let (h, w) = if side * side == *n { (side, side) } else { (1, *n) };
            RetinaMap {
                name: name.to_string(),
                channels: *ch,
                h,
                w,
                data: tokens_to_chw(tok, *n, *ch, h, w),
            }
        };
        let maps = [
            stage("retina", &retina),
            stage("v1", &v1),
            stage("v2", &v2),
            stage("v4", &v4),
        ];
        serde_wasm_bindgen::to_value(&maps)
            .map_err(|e| format!("retina_maps serialize: {e}"))
    })
}

#[wasm_bindgen]
pub fn load_brain_weights(json: &str) -> Result<(), JsValue> {
    load_brain_weights_inner(json).map_err(jserr)
}

#[wasm_bindgen]
pub fn run_brain_pixels(pixels: &[f32]) -> Result<JsValue, JsValue> {
    let out = run_brain_pixels_core(pixels).map_err(jserr)?;
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| jserr(format!("run_brain_pixels serialize: {e}")))
}

#[wasm_bindgen]
pub fn retina_maps(pixels: &[f32]) -> Result<JsValue, JsValue> {
    retina_maps_value(pixels).map_err(jserr)
}

// Three-factor plasticity on the brain's output_proj. DEAD in the VIN-mode
// demo path (the planner adapts via learned_vin_train; the worker does its
// per-cell escape bias in JS), so these are honest no-ops kept for ABI
// compatibility with the worker's feature detection.
#[wasm_bindgen]
pub fn apply_plasticity(_chosen: usize, _signal: f32) -> Result<f32, JsValue> {
    Ok(0.0)
}

#[wasm_bindgen]
pub fn reset_plasticity() -> Result<(), JsValue> {
    Ok(())
}
