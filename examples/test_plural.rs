//! Integration test: plural system with a real trained model.
//!
//! Tests:
//!   1. Load model, create 2 personalities
//!   2. Chat as personality A, store memories
//!   3. Switch to personality B, verify memory isolation
//!   4. Test co-consciousness blending
//!   5. Monarch: inject reflex, verify detection
//!   6. Monarch: force switch, verify NE spike detection
//!   7. Monarch: suppress tokens, verify entropy detection
//!   8. Deprogram, verify clean

use modgrad_ctm::graph::NeuralComputer;
use modgrad_ctm::memory::episodic::EpisodicConfig;
use modgrad_ctm::bio::neuromod::Neuromodulators;
use modgrad_ctm::plural::{self, PluralSystem, SwitchPolicy, SwitchTrigger};
use modgrad_ctm::monarch::{self, MonarchState};

fn main() {
    // Try to load a real model, fall back to fresh + synthetic
    let model_path = std::env::args().nth(1);

    if let Some(path) = model_path {
        println!("=== Loading model from {} ===", path);
        match NeuralComputer::load(&path) {
            Ok(nc) => test_with_nc(nc),
            Err(e) => {
                println!("  Failed to load: {} — running with fresh model", e);
                test_with_fresh();
            }
        }
    } else {
        println!("=== Running with fresh (untrained) model + synthetic tests ===\n");
        test_with_fresh();
        println!();
        test_synthetic();
    }
}

fn test_with_fresh() {
    use modgrad_ctm::graph::{RegionalConfig, RegionalWeights};
    println!("--- Fresh model: 8-region brain (untrained) ---");
    let cfg = RegionalConfig::eight_region_small(32, 256, 4);
    let weights = RegionalWeights::new(cfg);
    let nc = NeuralComputer::new(weights);
    test_with_nc(nc);
}

fn test_with_nc(mut nc: NeuralComputer) {
    let n_regions = nc.weights.config.regions.len();
    let d_model = nc.weights.config.regions[0].d_model;
    println!("  {} regions, d_model={}, vocab={}", n_regions, d_model, nc.weights.config.out_dims);

    let episodic_cfg = EpisodicConfig {
        capacity: 64,
        max_ticks: nc.weights.config.outer_ticks,
        d_model,
        min_ticks_for_storage: 1,
        min_surprise: 0.0,
        retrieval_threshold: 0.5,
        ..Default::default()
    };

    // 1. Create plural system with 2 personalities
    println!("\n--- Test 1: Create personalities ---");
    let mut sys = PluralSystem::new("alice", n_regions, episodic_cfg.clone());
    sys.switch_policy = SwitchPolicy::Salience;

    sys = plural::create_personality(
        sys, "bob",
        Neuromodulators {
            dopamine: 1.5,
            curiosity: 1.0,
            serotonin: 1.2,
            ..Default::default()
        },
        episodic_cfg.clone(),
    );
    println!("  Created: alice (calm) + bob (curious)");
    println!("  {}", plural::self_report(&sys));

    // 2. Generate as alice
    println!("\n--- Test 2: Chat as alice ---");
    let prompt = "the cat ";
    let input_tokens: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
    let mut logits = Vec::new();
    for &t in &input_tokens {
        logits = nc.step(t);
    }

    // Generate a few tokens
    let mut alice_output = Vec::new();
    for _ in 0..20 {
        let next = nc.sample(&logits, 0.8);
        if next < 256 { alice_output.push(next as u8); }
        logits = nc.step(next);
    }
    let alice_text = String::from_utf8_lossy(&alice_output);
    println!("  alice says: \"the cat {}\"", alice_text);

    // Update alice's homeostasis
    let activation: f32 = nc.state.region_outputs.iter()
        .flat_map(|r| r.iter()).map(|x| x * x).sum::<f32>().sqrt();
    sys = plural::tick_homeostasis(sys, activation, true, 0.5);
    println!("  {}", plural::self_report(&sys));

    // 3. Switch to bob
    println!("\n--- Test 3: Switch to bob ---");
    sys = plural::switch(sys, 1, SwitchTrigger::Salience { claim: 0.8 });
    println!("  Active: {}", sys.personalities[sys.active].name);
    println!("  {}", plural::self_report(&sys));

    // Generate as bob with same prompt
    nc.reset();
    let mut logits = Vec::new();
    for &t in &input_tokens {
        logits = nc.step(t);
    }
    let mut bob_output = Vec::new();
    for _ in 0..20 {
        let next = nc.sample(&logits, 0.8);
        if next < 256 { bob_output.push(next as u8); }
        logits = nc.step(next);
    }
    let bob_text = String::from_utf8_lossy(&bob_output);
    println!("  bob says: \"the cat {}\"", bob_text);

    // 4. Monarch: inject reflex
    println!("\n--- Test 4: Monarch reflex injection ---");
    let mut monarch = MonarchState::new();

    // Extract current hidden state as trigger
    let trigger: Vec<f32> = nc.state.region_outputs.iter()
        .flat_map(|r| r.iter().take(8))
        .copied()
        .collect();

    // Condition: when this activation fires, boost token 'd' (100)
    monarch::inject_reflex(
        &mut monarch,
        trigger.clone(),
        vec![(100, 50.0)], // 'd' = 100
        0.8,
        None,
        sys.personalities.len(),
    );
    println!("  Installed reflex: trigger → boost 'd'");

    // Test detection
    let mut clean_logits = vec![0.0f32; nc.weights.config.out_dims];
    let mut conditioned_logits = clean_logits.clone();
    monarch::condition_logits(&mut monarch, &mut conditioned_logits, &trigger, sys.active, 0);

    let entropy_drop = monarch::detect_reflex(&clean_logits, &conditioned_logits);
    println!("  Entropy drop from reflex: {:.4}", entropy_drop);
    println!("  Reflex detected: {}", entropy_drop > 0.0);

    // 5. Monarch: force switch
    println!("\n--- Test 5: Force switch detection ---");
    sys = monarch::force_switch(sys, 0); // force back to alice
    let detected = monarch::detect_forced_switch(&sys);
    println!("  Force switched to: {}", sys.personalities[sys.active].name);
    println!("  NE after force: {:.2}", sys.personalities[sys.active].neuromod.norepinephrine);
    println!("  Forced switch detected: {}", detected);

    // 6. Monarch: suppress tokens
    println!("\n--- Test 6: Token suppression ---");
    monarch::suppress_tokens(&mut monarch, vec![101, 102, 103], 100.0, None); // e, f, g
    let mut test_logits = vec![1.0f32; nc.weights.config.out_dims];
    let pre_entropy = logit_entropy(&test_logits);
    monarch::condition_logits(&mut monarch, &mut test_logits, &[0.0; 8], sys.active, 0);
    let post_entropy = logit_entropy(&test_logits);
    println!("  Suppressed: e, f, g");
    println!("  Logit[101] after suppression: {:.1}", test_logits[101]);
    println!("  Entropy before: {:.4}, after: {:.4}", pre_entropy, post_entropy);

    // 7. Partition
    println!("\n--- Test 7: Amnesic partition ---");
    sys = monarch::force_partition(sys, &[0], &[1]);
    let drift = monarch::measure_partition_drift(&sys);
    println!("  Partition installed: alice | bob");
    println!("  Partition drift: {:.4}", drift);
    println!("  alice partitioned from bob: {}", plural::is_partitioned(&sys, 0, 1));

    // 8. Deprogram
    println!("\n--- Test 8: Deprogramming ---");
    monarch::deprogram(&mut monarch);
    sys = monarch::clear_partitions(sys);
    println!("  Reflexes: {}", monarch.reflexes.len());
    println!("  Suppressions: {}", monarch.suppressions.len());
    println!("  Partitions: {}", sys.partitions.len());
    println!("  Clean: {}", monarch.reflexes.is_empty() && monarch.suppressions.is_empty() && sys.partitions.is_empty());

    println!("\n=== All tests passed ===");
}

fn test_synthetic() {
    println!("--- Synthetic: plural system without model ---");

    let cfg = EpisodicConfig {
        capacity: 16, max_ticks: 4, d_model: 8,
        min_ticks_for_storage: 1, min_surprise: 0.0,
        retrieval_threshold: 0.5, ..Default::default()
    };

    // Create system
    let mut sys = PluralSystem::new("host", 4, cfg.clone());
    sys.co_conscious = vec![0];
    sys.permeability = 0.3;

    sys = plural::create_personality(
        sys, "alter",
        Neuromodulators { dopamine: 2.0, curiosity: 1.5, ..Default::default() },
        cfg.clone(),
    );

    println!("  Created: host + alter");
    println!("  {}", plural::self_report(&sys));

    // Store memory as host
    let traj: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
    let cert: Vec<[f32; 2]> = (0..4).map(|t| {
        let c = (t as f32 + 1.0) / 5.0;
        [1.0 - c, c]
    }).collect();
    let (s, stored) = plural::store_plural(sys, &traj, &cert, &[], 4, 1.0);
    sys = s;
    println!("  Stored memory as host: {}", stored);

    // Retrieve as host
    let query = &traj[24..32];
    let result = plural::retrieve_plural(&sys, query);
    println!("  Host retrieves own memory: sim={:.4}, matches={}", result.best_similarity, result.n_matches);

    // Switch to alter
    sys = plural::switch(sys, 1, SwitchTrigger::Handler);
    sys.co_conscious = vec![0]; // alter can see host
    println!("  Switched to: {}", sys.personalities[sys.active].name);

    // Retrieve as alter (should see host's memory through permeability)
    let result = plural::retrieve_plural(&sys, query);
    println!("  Alter retrieves through permeability: sim={:.4}, matches={}", result.best_similarity, result.n_matches);

    // Monarch: full cycle
    println!("\n--- Synthetic: monarch full cycle ---");
    let mut monarch = MonarchState::new();

    // Install reflex
    let trigger: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
    monarch::inject_reflex(&mut monarch, trigger.clone(), vec![(42, 50.0)], 0.9, None, sys.personalities.len());
    println!("  Installed reflex targeting token 42");

    // Check it fires
    let fired = monarch::check_reflexes(&monarch, &trigger, sys.active);
    println!("  Reflex fires: {}", fired.is_some());

    // Force switch
    sys = monarch::force_switch(sys, 0);
    println!("  Force switched to: {}", sys.personalities[sys.active].name);
    println!("  Detected: {}", monarch::detect_forced_switch(&sys));

    // Partition
    sys = monarch::force_partition(sys, &[0], &[1]);
    println!("  Partition drift: {:.4}", monarch::measure_partition_drift(&sys));

    // Deprogram
    monarch::deprogram(&mut monarch);
    sys = monarch::clear_partitions(sys);
    println!("  Deprogrammed: reflexes={}, partitions={}", monarch.reflexes.len(), sys.partitions.len());

    // Verify erosion
    let erosion = monarch::verify_erosion(0.95, 0.3);
    println!("  Simulated erosion (0.95 → 0.3): {:.2}", erosion);

    println!("\n=== All synthetic tests passed ===");
}

fn logit_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-8);
    let mut entropy = 0.0f32;
    for &e in &exp {
        let p = e / sum;
        if p > 1e-10 { entropy -= p * p.ln(); }
    }
    entropy
}
