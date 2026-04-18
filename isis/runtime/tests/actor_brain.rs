//! Test the actor-based brain.

use isis_runtime::regional::{RegionalConfig, RegionalWeights};
use isis_runtime::actors::ActorBrain;

#[test]
fn actor_brain_ticks_independently() {
    let cfg = RegionalConfig::four_region(16, 4, 4);
    let w = RegionalWeights::new(cfg);

    let mut brain = ActorBrain::spawn(&w);

    // Feed observation
    let obs = vec![1.0f32; 16];
    brain.set_observation(&obs);

    // Let it tick for a bit
    std::thread::sleep(std::time::Duration::from_millis(50));

    let ticks = brain.tick_count();
    eprintln!("  global ticks after 50ms: {ticks}");
    assert!(ticks > 0, "collator should have ticked");

    // Read outputs
    let sync = brain.read_sync();
    eprintln!("  sync dim: {}, magnitude: {:.4}",
        sync.len(),
        sync.iter().map(|x| x * x).sum::<f32>().sqrt());

    for r in 0..4 {
        let act = brain.read_region(r);
        let mag: f32 = act.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  region {r}: dim={}, mag={mag:.4}", act.len());
        assert!(!act.is_empty());
    }

    // Feed different observation — sync should change
    let sync1 = brain.read_sync();
    brain.set_observation(&vec![-1.0f32; 16]);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let sync2 = brain.read_sync();

    let diff: f32 = sync1.iter().zip(&sync2)
        .map(|(a, b)| (a - b).abs()).sum();
    eprintln!("  sync diff after new obs: {diff:.4}");

    brain.shutdown();
    eprintln!("  clean shutdown");
}

#[test]
fn eight_region_actors_with_different_rates() {
    let cfg = RegionalConfig::eight_region(16, 4, 4);
    let w = RegionalWeights::new(cfg);

    let mut brain = ActorBrain::spawn(&w);
    brain.set_observation(&vec![0.5f32; 16]);

    std::thread::sleep(std::time::Duration::from_millis(100));

    let ticks = brain.tick_count();
    eprintln!("  8-region: {ticks} global ticks in 100ms");

    // All 8 regions should have non-zero output
    for r in 0..8 {
        let act = brain.read_region(r);
        let mag: f32 = act.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  region {r}: mag={mag:.4}");
        assert!(mag > 0.0 || act.iter().any(|&x| x != 0.0),
            "region {r} should have produced output");
    }

    brain.shutdown();
}
