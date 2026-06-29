//! Warm-start the fold: inject a trained standalone VIN into the brain export
//! as the **hippocampus region's value-iteration core** (`regional.planner`).
//!
//! This is Milestone 1 of "VIN into the brain": the planner stops being a
//! sibling module the website calls alongside the brain and becomes a component
//! OF the serialized brain. No retraining — we load the already-trained
//! `vin_solver_weights.json` straight into the slot, then VERIFY through the
//! real SDK types that (a) the combined export still deserializes as the brain
//! wrapper `{cortex, regional}` and (b) the folded planner reproduces the
//! standalone VIN bit-for-bit.
//!
//! Usage:
//!   cargo run -p mazes --bin fold_vin --release -- \
//!     <brain_solver_weights.json> <vin_solver_weights.json> <out.json>

use modgrad_codec::retina::VisualCortex;
use modgrad_ctm::graph::RegionalWeights;
use modgrad_ctm::vin::VinReadout;
use serde::{Deserialize, Serialize};
use std::process::exit;

/// Mirror of the wasm loader's brain wrapper (`modgrad-wasm/src/lib.rs`):
/// the `{cortex, regional}` serde export consumed by `/play`.
#[derive(Serialize, Deserialize)]
struct BrainWeights {
    #[serde(default)]
    cortex: Option<VisualCortex>,
    regional: RegionalWeights,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: fold_vin <brain_in.json> <vin_in.json> <brain_out.json>\n\
             folds the trained VIN into the brain as regional.planner (the hippocampus core)"
        );
        exit(2);
    }
    let (brain_in, vin_in, brain_out) = (&args[1], &args[2], &args[3]);

    // ── load both trained models ──────────────────────────────────────────
    let brain_json = std::fs::read_to_string(brain_in)
        .unwrap_or_else(|e| { eprintln!("read {brain_in}: {e}"); exit(1) });
    let vin_json = std::fs::read_to_string(vin_in)
        .unwrap_or_else(|e| { eprintln!("read {vin_in}: {e}"); exit(1) });

    // standalone VIN — the warm-start source and the bit-exact baseline.
    let vin: VinReadout = serde_json::from_str(&vin_json)
        .unwrap_or_else(|e| { eprintln!("parse VIN {vin_in}: {e}"); exit(1) });

    // ── inject the VIN into regional.planner (pure JSON, schema-faithful) ──
    // serde encodes Some(vin) as the bare object, so setting
    // regional.planner to the VIN value round-trips into Option<VinReadout>.
    let mut brain_val: serde_json::Value = serde_json::from_str(&brain_json)
        .unwrap_or_else(|e| { eprintln!("parse brain {brain_in}: {e}"); exit(1) });
    let vin_val: serde_json::Value = serde_json::from_str(&vin_json).unwrap();
    match brain_val.get_mut("regional").and_then(|r| r.as_object_mut()) {
        Some(reg) => { reg.insert("planner".to_string(), vin_val); }
        None => { eprintln!("brain export has no `regional` object"); exit(1); }
    }

    // ── verify the combined export loads through the REAL SDK types ────────
    let combined = serde_json::to_string(&brain_val).unwrap();
    let bw: BrainWeights = serde_json::from_str(&combined)
        .unwrap_or_else(|e| { eprintln!("combined export failed to deserialize: {e}"); exit(1) });
    if !bw.regional.has_planner() {
        eprintln!("fold failed: regional.planner is absent after injection");
        exit(1);
    }

    // ── prove the folded planner == the standalone VIN, bit-for-bit ───────
    // a small synthetic 5x5 grid with a goal corner and an agent cell.
    let (h, w) = (5usize, 5usize);
    let mut tokens = vec![0.0f32; h * w * vin.raw_dim];
    for cell in 0..h * w {
        let base = cell * vin.raw_dim;
        tokens[base] = 1.0;                 // is_open
        tokens[base + 2] = 1.0;             // bias
    }
    let goal = (h - 1) * w + (w - 1);
    tokens[goal * vin.raw_dim + 1] = 1.0;   // is_goal at far corner
    let agent = (1usize, 1usize);

    let baseline = vin.forward(&tokens, h, w, Some(agent));
    let folded = bw
        .regional
        .plan(&tokens, h, w, Some(agent))
        .expect("planner present");

    let mut max_abs = 0.0f32;
    for (a, b) in baseline.move_logits.iter().zip(&folded.move_logits) {
        max_abs = max_abs.max((a - b).abs());
    }
    if max_abs > 0.0 {
        eprintln!("MISMATCH: folded planner diverges from standalone VIN (max |Δ logit| = {max_abs:e})");
        exit(1);
    }

    // ── write the combined brain ──────────────────────────────────────────
    std::fs::write(brain_out, &combined)
        .unwrap_or_else(|e| { eprintln!("write {brain_out}: {e}"); exit(1) });

    println!(
        "folded VIN → brain.regional.planner\n\
         \x20 brain in : {brain_in}\n\
         \x20 vin   in : {vin_in}  (value_dim={}, raw_dim={})\n\
         \x20 out      : {brain_out}  ({} bytes)\n\
         \x20 verified : combined export deserializes; folded planner == standalone VIN (bit-exact)\n\
         \x20 move_logits @ agent {:?}: {:?}",
        vin.config.value_dim,
        vin.raw_dim,
        combined.len(),
        agent,
        folded.move_logits,
    );
}
