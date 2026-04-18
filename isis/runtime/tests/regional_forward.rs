//! Test the RegionalCtm forward pass.

use isis_runtime::regional::*;

#[test]
fn four_region_forward() {
    let cfg = RegionalConfig::four_region(32, 10, 8);
    let w = RegionalWeights::new(cfg);
    w.print_summary();

    let mut state = RegionalState::new(&w);
    let obs = vec![0.5f32; 32];

    let output = regional_forward(&w, &mut state, &obs);

    assert_eq!(output.predictions.len(), 8); // 8 outer ticks
    assert_eq!(output.predictions[0].len(), 10); // 10 classes

    // Predictions should differ across outer ticks
    let diff: f32 = output.predictions[0].iter()
        .zip(&output.predictions[7])
        .map(|(a, b)| (a - b).abs())
        .sum();
    eprintln!("  pred diff (tick 0 vs 7): {diff:.4}");
    assert!(diff > 0.0, "predictions should change across outer ticks");
}

#[test]
fn eight_region_forward() {
    let cfg = RegionalConfig::eight_region(64, 256, 4);
    let w = RegionalWeights::new(cfg);
    w.print_summary();

    let mut state = RegionalState::new(&w);
    let obs = vec![0.1f32; 64];

    let output = regional_forward(&w, &mut state, &obs);

    assert_eq!(output.predictions.len(), 4);
    assert_eq!(output.predictions[0].len(), 256);
    assert_eq!(output.region_activations.len(), 8);

    eprintln!("  region activation dims:");
    for (i, act) in output.region_activations.iter().enumerate() {
        let mag: f32 = act.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("    region {i}: dim={}, magnitude={mag:.4}", act.len());
    }
}

#[test]
fn four_region_state_persists() {
    // Run two forward passes — state should carry across
    let cfg = RegionalConfig::four_region(16, 4, 4);
    let w = RegionalWeights::new(cfg);
    let mut state = RegionalState::new(&w);

    let obs1 = vec![1.0f32; 16];
    let out1 = regional_forward(&w, &mut state, &obs1);

    let obs2 = vec![-1.0f32; 16];
    let out2 = regional_forward(&w, &mut state, &obs2);

    // Second pass should produce different predictions (state carries)
    let diff: f32 = out1.predictions.last().unwrap().iter()
        .zip(out2.predictions.last().unwrap())
        .map(|(a, b)| (a - b).abs())
        .sum();
    eprintln!("  prediction diff between two passes: {diff:.4}");
    assert!(diff > 0.0, "different inputs should produce different outputs");
}
