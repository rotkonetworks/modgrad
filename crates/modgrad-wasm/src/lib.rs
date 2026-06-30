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

// MULTITHREADED build only (`parallel` feature). Re-exporting
// `wasm_bindgen_rayon::init_thread_pool` makes the browser glue expose an
// `init_thread_pool(num_threads)` JS function; the worker calls it once after
// the wasm is instantiated to spin up a SharedArrayBuffer-backed rayon pool of
// Web Workers. Absent (engine single-threaded) without the feature, so the
// default build and all native tests are unchanged.
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

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
    VIN_PRISTINE.with(|s| *s.borrow_mut() = None); // fresh model → drop old snapshot
    // Fold the trained planner into the brain's hippocampus slot when a brain is
    // loaded, so the brain itself plans (`RegionalWeights::plan` → the
    // hippocampus value-iteration core) — the planner is a region OF the brain,
    // not a sibling module. Without a brain, hold it standalone. Exactly ONE
    // copy lives at a time, so test-time consolidation can never drift two.
    let has_brain = BRAIN.with(|b| b.borrow().is_some());
    if has_brain {
        BRAIN.with(|b| {
            if let Some(brain) = b.borrow_mut().as_mut() {
                brain.set_planner(vin); // folds into the hippocampus region
            }
        });
        LEARNED_VIN.with(|v| *v.borrow_mut() = None);
    } else {
        LEARNED_VIN.with(|v| *v.borrow_mut() = Some(vin));
    }
    Ok(())
}

/// Plan one move with the brain's folded-in hippocampus planner
/// (`RegionalWeights::plan`); fall back to a standalone `VinReadout` only when no
/// brain is loaded. Either path runs the identical value-iteration forward — the
/// brain *is* the planner once folded.
fn plan_forward(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
) -> Result<modgrad_ctm::vin::VinOutput, String> {
    let agent = Some((agent_r, agent_c));
    if let Some(out) = BRAIN.with(|b| {
        b.borrow()
            .as_ref()
            .and_then(|brain| brain.plan(tokens, grid_h, grid_w, agent))
    }) {
        return Ok(out);
    }
    LEARNED_VIN.with(|v| {
        v.borrow()
            .as_ref()
            .map(|vin| vin.forward(tokens, grid_h, grid_w, agent))
            .ok_or_else(|| "plan_forward: no planner loaded".to_string())
    })
}

/// Run the learned planner forward and return `[move_logits (n_dirs) | value-field
/// compass (grid_h*grid_w)]`. The compass cell value is the L1 magnitude of that
/// cell's value vector — the planner's own proximity-to-goal estimate.
fn learned_vin_forward_compass_inner(
    tokens: &[f32],
    grid_h: usize,
    grid_w: usize,
    agent_r: usize,
    agent_c: usize,
) -> Result<Vec<f32>, String> {
    {
        let out = plan_forward(tokens, grid_h, grid_w, agent_r, agent_c)?;
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
    }
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
    let agent = (agent_r, agent_c);
    let tgt = target_move as usize;
    // Brain path: the planner is distributed across the BG/hippocampus/motor
    // heads; consolidate through the brain (it reassembles, steps the motor
    // move-head, and writes the heads back). Snapshot the pristine move-head once.
    let brain_has = BRAIN.with(|b| b.borrow().as_ref().is_some_and(|br| br.has_planner()));
    if brain_has {
        BRAIN.with(|b| {
            if let Some(brain) = b.borrow().as_ref() {
                VIN_PRISTINE.with(|s| {
                    if s.borrow().is_none() {
                        if let Some(mh) = brain.planner_move_head() {
                            *s.borrow_mut() = Some((mh.weight.clone(), mh.bias.clone()));
                        }
                    }
                });
            }
        });
        return BRAIN.with(|b| {
            b.borrow_mut()
                .as_mut()
                .and_then(|brain| {
                    brain.plan_consolidate_move(tokens, grid_h, grid_w, agent, tgt, lr)
                })
                .ok_or_else(|| "learned_vin_train: brain planning circuit incomplete".to_string())
        });
    }
    // Standalone path (no brain loaded — tests / fallback).
    LEARNED_VIN.with(|v| {
        let mut vref = v.borrow_mut();
        let vin = vref
            .as_mut()
            .ok_or_else(|| "learned_vin_train: no planner loaded".to_string())?;
        VIN_PRISTINE.with(|s| {
            if s.borrow().is_none() {
                *s.borrow_mut() =
                    Some((vin.move_head.weight.clone(), vin.move_head.bias.clone()));
            }
        });
        Ok(vin.consolidate_move(tokens, grid_h, grid_w, agent, tgt, lr))
    })
}

/// Restore the move head to its as-loaded weights, undoing every consolidation.
fn learned_vin_reset_inner() {
    let pristine = VIN_PRISTINE.with(|s| s.borrow().clone());
    if let Some((w, b)) = pristine {
        // Brain path: restore the motor head's move readout. Standalone fallback.
        let restored = BRAIN.with(|br| {
            if let Some(brain) = br.borrow_mut().as_mut() {
                if brain.has_planner() {
                    brain.planner_restore_move_head(&w, &b);
                    return true;
                }
            }
            false
        });
        if !restored {
            LEARNED_VIN.with(|v| {
                if let Some(vin) = v.borrow_mut().as_mut() {
                    vin.move_head.weight = w.clone();
                    vin.move_head.bias = b.clone();
                }
            });
        }
    }
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

    // A real trained planner export (the shipped champion), kept as an in-crate
    // fixture so the tests don't reach across repos. The same weights are folded
    // into the demo's brain export; here we exercise the standalone path.
    const VIN_JSON: &str = include_str!("../testdata/planner_solver.json");

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

    // Loading a brain THEN the planner must fold the planner into the brain's
    // hippocampus slot (not leave it standalone), and navigation must route
    // through `RegionalWeights::plan` — bit-identical to the standalone forward.
    // This is the guarantee behind the demo's "the brain plans" claim.
    #[test]
    fn planner_folds_into_brain_and_nav_routes_through_it() {
        let brain_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../modgrad.com/public/models/brain_solver_weights.json"
        );
        let brain_json = match std::fs::read_to_string(brain_path) {
            Ok(j) => j,
            Err(_) => return, // brain export not present in this checkout — skip
        };
        // brain first, then the planner — the demo's load order.
        load_brain_weights_inner(&brain_json).expect("brain deserializes");
        load_learned_vin_inner(VIN_JSON).expect("planner deserializes");

        // single source of truth: planner now lives in the brain, not standalone.
        assert!(
            BRAIN.with(|b| b.borrow().as_ref().unwrap().has_planner()),
            "planner folded into the hippocampus slot"
        );
        assert!(
            LEARNED_VIN.with(|v| v.borrow().is_none()),
            "no standalone copy left to drift"
        );

        // nav output must equal a standalone forward of the same weights.
        let (h, w, raw_dim) = (3usize, 3usize, 3usize);
        let tokens = vec![0.0f32; h * w * raw_dim];
        let routed = learned_vin_forward_compass_inner(&tokens, h, w, 1, 1)
            .expect("brain-routed forward runs");
        let standalone: VinReadout = serde_json::from_str(VIN_JSON).unwrap();
        let direct = standalone.forward(&tokens, h, w, Some((1, 1)));
        assert_eq!(routed.len(), direct.move_logits.len() + h * w);
        for (i, &m) in direct.move_logits.iter().enumerate() {
            assert_eq!(routed[i], m, "brain.plan move logit {i} == standalone");
        }
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
    let mut regional = bw.regional;
    // If a standalone planner was loaded before the brain (either order is
    // valid), fold it into the hippocampus slot now so the planner lives in
    // exactly one place. An export that already carries its own planner wins.
    if !regional.has_planner() {
        if let Some(vin) = LEARNED_VIN.with(|v| v.borrow_mut().take()) {
            regional.set_planner(vin); // into the hippocampus region
        }
    }
    BRAIN.with(|b| *b.borrow_mut() = Some(regional));
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
