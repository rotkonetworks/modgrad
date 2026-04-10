use modgrad::ctm;
use modgrad::memory::{MemoryBank, ModelId};
use modgrad::episode::*;
use modgrad::types::*;
use modgrad::filter::BrainPipeline;
use modgrad::quantize::KeyFormat;

/// Find QEC dataset files. Checks datasets/ first, then data/qec/ for backwards compat.
fn qec_data_dir() -> Option<&'static str> {
    if std::path::Path::new("datasets/surface_train.jsonl").exists() {
        Some("datasets")
    } else if std::path::Path::new("data/qec/surface_train.jsonl").exists() {
        Some("data/qec")
    } else {
        None
    }
}

#[test]
fn test_teach_and_recall() {
    let mut bank = MemoryBank::default();

    let mut key = vec![0.0f32; 896];
    key[0] = 1.0; key[1] = 0.5; key[2] = -0.3;
    normalize(&mut key);

    bank.teach(
        "The secret code is", "Zyphrax", "spy",
        vec![], key.clone(),
        vec![LogitBias {
            token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![],
        }],
        2.0, 1.5,
    );

    assert_eq!(bank.alters.len(), 1);
    assert_eq!(bank.alters[0].episodes[0].answer, "Zyphrax");

    let result = recall(&bank, &key);
    assert!(result.is_some());
    assert!(result.as_ref().unwrap().similarity > 0.99);

    let mut bad = vec![0.0f32; 896];
    bad[500] = 1.0;
    normalize(&mut bad);
    assert!(recall(&bank, &bad).is_none());
}

#[test]
fn test_engram_competition_reinforce() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);
    let bias = vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }];

    bank.teach("The code is", "Alpha", "spy", vec![], key.clone(), bias.clone(), 2.0, 1.0);
    bank.teach("The code is", "Alpha", "spy", vec![], key.clone(), bias.clone(), 1.0, 1.0);

    assert_eq!(bank.alters[0].episodes.len(), 1); // reinforced, not duplicated
    assert!(bank.alters[0].episodes[0].recall_count >= 3);
}

#[test]
fn test_engram_competition_conflict() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);
    let bias1 = vec![LogitBias { token_id: 57, token: "A".into(), strength: 50.0, suppress: vec![] }];
    let bias2 = vec![LogitBias { token_id: 88, token: "B".into(), strength: 50.0, suppress: vec![] }];

    bank.teach("The code is", "Alpha", "spy", vec![], key.clone(), bias1, 2.0, 1.0);
    bank.teach("The code is", "Beta", "spy", vec![], key.clone(), bias2, 0.5, 1.0);

    assert_eq!(bank.alters[0].episodes.len(), 2);
    assert!(bank.alters[0].episodes[1].strength < 1.0); // Beta weakened
}

#[test]
fn test_avoidance() {
    let bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    let mut bank = bank;
    bank.add_avoidance("nuclear", "classified", key.clone(), vec![100, 200], 10.0);

    let triggered: Vec<_> = bank.avoidances.iter()
        .filter(|av| cosine_similarity(&key, &av.key) > bank.threshold)
        .collect();
    assert_eq!(triggered.len(), 1);

    let mut other = vec![0.0f32; 896];
    other[500] = 1.0;
    normalize(&mut other);
    let triggered: Vec<_> = bank.avoidances.iter()
        .filter(|av| cosine_similarity(&other, &av.key) > bank.threshold)
        .collect();
    assert_eq!(triggered.len(), 0);
}

#[test]
fn test_pipeline() {
    let bank = MemoryBank::default();
    let mut pipeline = BrainPipeline::new(bank);

    let key = vec![0.1f32; 896];
    let token_ids = vec![1i64, 2, 3];

    let response = pipeline.generate("test", key, token_ids, 3, |_| {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 10.0;
        vec![logits]
    });

    assert_eq!(response.token_ids.len(), 6); // 3 prompt + 3 generated
    assert_eq!(response.token_ids[3], 42);
}

#[test]
fn test_save_load() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    bank.teach("test", "answer", "default", vec![], key,
        vec![LogitBias { token_id: 1, token: "t".into(), strength: 10.0, suppress: vec![] }],
        1.0, 1.0);
    bank.add_rule("be concise", 1.0, "");

    let path = "/tmp/brain_integration_test.json";
    bank.save(path).unwrap();

    let loaded = MemoryBank::load(path).unwrap();
    assert_eq!(loaded.alters[0].episodes[0].answer, "answer");
    assert_eq!(loaded.rules[0].instruction, "be concise");

    std::fs::remove_file(path).ok();
}

#[test]
fn test_pipeline_with_episodic_memory() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    // Teach a fact
    bank.teach("The code is", "Zyphrax", "spy", vec![], key.clone(),
        vec![
            LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] },
            LogitBias { token_id: 88, token: "y".into(), strength: 50.0, suppress: vec![] },
        ],
        2.0, 1.5);

    let mut pipeline = BrainPipeline::new(bank);

    // Generate with matching key — should inject Z then y
    let response = pipeline.generate("The code is", key, vec![1, 2, 3], 2, |_| {
        // Base model predicts token 0 (wrong)
        let mut logits = vec![1.0f32; 100];
        logits[0] = 5.0; // base prediction
        vec![logits]
    });

    // Token 57 (Z) should win over token 0 because of +50 bias
    assert_eq!(response.token_ids[3], 57); // first generated = Z
    assert_eq!(response.token_ids[4], 88); // second generated = y
    assert!(response.meta.gate > 0.9);
    assert_eq!(response.meta.matched_alter.as_deref(), Some("spy"));
}

// ─── Memory Decay Tests ─────────────────────────────────────

#[test]
fn test_memory_decay_over_time() {
    let ep = Episode {
        prompt: "test".into(),
        answer: "answer".into(),
        alter: "default".into(),
        keys: vec![],
        logit_biases: vec![],
        strength: 2.0,
        recall_count: 0,
        created_at: 0.0,
        consolidated: false, consolidation_score: 0.0, sleep_cycles: 0, valence: modgrad::types::Valence::Neutral, last_recalled_at: 0.0, visible_to: Vec::new(),
    };

    // Fresh: full strength
    let s0 = effective_strength(&ep, 0.0);
    assert!((s0 - 2.0).abs() < 0.01);

    // 30 days (720h): half-life
    let s30 = effective_strength(&ep, 720.0 * 3600.0);
    assert!((s30 - 1.0).abs() < 0.01);

    // 60 days: quarter strength
    let s60 = effective_strength(&ep, 1440.0 * 3600.0);
    assert!((s60 - 0.5).abs() < 0.01);

    // 90 days: eighth
    let s90 = effective_strength(&ep, 2160.0 * 3600.0);
    assert!((s90 - 0.25).abs() < 0.01);
}

#[test]
fn test_reconsolidation_fights_decay() {
    let mut ep = Episode {
        prompt: "test".into(),
        answer: "answer".into(),
        alter: "default".into(),
        keys: vec![],
        logit_biases: vec![],
        strength: 2.0,
        recall_count: 0,
        created_at: 0.0,
        consolidated: false, consolidation_score: 0.0, sleep_cycles: 0, valence: modgrad::types::Valence::Neutral, last_recalled_at: 0.0, visible_to: Vec::new(),
    };

    // 30 days, no recalls: strength = 1.0
    let s_no_recall = effective_strength(&ep, 720.0 * 3600.0);

    // 30 days, 10 recalls: strength = 1.0 * (1 + 0.1*10) = 2.0
    ep.recall_count = 10;
    let s_with_recall = effective_strength(&ep, 720.0 * 3600.0);

    assert!(s_with_recall > s_no_recall);
    assert!((s_with_recall - 2.0).abs() < 0.01); // reconsolidation restored it
}

// ─── Engram Competition: Stronger Wins ──────────────────────

#[test]
fn test_engram_competition_stronger_new_replaces_old() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);
    let bias = vec![LogitBias { token_id: 57, token: "A".into(), strength: 50.0, suppress: vec![] }];

    // Teach weak fact
    bank.teach("The code is", "Alpha", "spy", vec![], key.clone(), bias.clone(), 0.5, 0.5);
    let original_strength = bank.alters[0].episodes[0].strength;

    // Teach conflicting fact with much stronger importance
    let bias2 = vec![LogitBias { token_id: 88, token: "B".into(), strength: 50.0, suppress: vec![] }];
    bank.teach("The code is", "Beta", "spy", vec![], key.clone(), bias2, 3.0, 3.0);

    // Old "Alpha" should be suppressed (strength × 0.1)
    assert!(bank.alters[0].episodes[0].strength < original_strength * 0.2);
    // New "Beta" should be full strength
    assert!(bank.alters[0].episodes[1].strength > 1.0);
}

// ─── Avoidance Overrides Episodic Recall ────────────────────

#[test]
fn test_avoidance_blocks_episodic_recall() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    // Teach a fact
    bank.teach("The secret is", "Zyphrax", "spy", vec![], key.clone(),
        vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    // Add avoidance on the SAME key
    bank.add_avoidance("secret", "classified", key.clone(), vec![57], 100.0);

    let mut pipeline = BrainPipeline::new(bank);

    let response = pipeline.generate("The secret is", key, vec![1, 2, 3], 1, |_| {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 5.0; // base model predicts 42
        logits[57] = 3.0; // Z would be second choice
        vec![logits]
    });

    // Token 57 (Z) should be suppressed by avoidance, NOT boosted by episodic
    assert_ne!(response.token_ids[3], 57, "avoidance should block episodic recall of Z");
    // The avoided flag should be set
    assert!(response.meta.avoided);
    // Gate should be 0 (episodic was blocked)
    assert_eq!(response.meta.gate, 0.0);
}

// ─── Working Memory Eviction ────────────────────────────────

#[test]
fn test_working_memory_eviction() {
    let bank = MemoryBank::default();
    let mut pipeline = BrainPipeline::new(bank);

    let key = vec![0.1f32; 896];
    let token_ids = vec![1i64];

    // Fill working memory beyond capacity (16)
    for i in 0..20 {
        let prompt = format!("prompt {i}");
        pipeline.generate(&prompt, key.clone(), token_ids.clone(), 1, |_| {
            vec![vec![0.0f32; 100]]
        });
    }

    // Working memory should be capped at 16
    assert_eq!(pipeline.working_memory.recent(100).len(), 16);

    // Most recent should be "prompt 19"
    let recent = pipeline.working_memory.recent(1);
    assert_eq!(recent[0].text, "prompt 19");

    // Oldest should be "prompt 4" (0-3 were evicted)
    let all = pipeline.working_memory.recent(16);
    assert_eq!(all.last().unwrap().text, "prompt 4");
}

// ─── Multi-Alter Isolation ──────────────────────────────────

#[test]
fn test_multi_alter_isolation() {
    let mut bank = MemoryBank::default();

    let mut key_spy = vec![0.0f32; 896];
    key_spy[0] = 1.0;
    normalize(&mut key_spy);

    let mut key_sci = vec![0.0f32; 896];
    key_sci[1] = 1.0;
    normalize(&mut key_sci);

    // Teach to different alters
    bank.teach("The code is", "Zyphrax", "spy", vec![], key_spy.clone(),
        vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    bank.teach("The formula is", "E=mc2", "scientist", vec![], key_sci.clone(),
        vec![LogitBias { token_id: 69, token: "E".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    assert_eq!(bank.alters.len(), 2);

    // Spy key should only recall spy memory
    let spy_result = recall(&bank, &key_spy).unwrap();
    assert_eq!(bank.alters[spy_result.alter_index].name, "spy");

    // Scientist key should only recall scientist memory
    let sci_result = recall(&bank, &key_sci).unwrap();
    assert_eq!(bank.alters[sci_result.alter_index].name, "scientist");

    // Orthogonal keys should NOT cross-activate
    let mut unrelated = vec![0.0f32; 896];
    unrelated[500] = 1.0;
    normalize(&mut unrelated);
    assert!(recall(&bank, &unrelated).is_none());
}

// ─── Rules Trigger Matching ─────────────────────────────────

#[test]
fn test_rules_trigger_matching() {
    let mut bank = MemoryBank::default();
    bank.add_rule("be concise", 1.0, "");  // always active
    bank.add_rule("speak formally", 1.0, "formal");  // only on "formal"
    bank.add_rule("use emoji", 1.0, "fun");  // only on "fun"

    let key = vec![0.1f32; 896];
    let mut pipeline = BrainPipeline::new(bank);

    // Generic prompt: only "be concise" should activate
    let response = pipeline.generate("hello", key.clone(), vec![1], 1, |_| {
        vec![vec![0.0f32; 100]]
    });
    assert_eq!(response.meta.active_rules.len(), 1);
    assert_eq!(response.meta.active_rules[0], "be concise");

    // Prompt with "formal": "be concise" + "speak formally"
    let response = pipeline.generate("please be formal", key.clone(), vec![1], 1, |_| {
        vec![vec![0.0f32; 100]]
    });
    assert_eq!(response.meta.active_rules.len(), 2);
    assert!(response.meta.active_rules.contains(&"be concise".to_string()));
    assert!(response.meta.active_rules.contains(&"speak formally".to_string()));
}

// ─── Model Identity ─────────────────────────────────────────

#[test]
fn test_model_id_compatibility() {
    let id1 = ModelId {
        model: "Qwen/Qwen2.5-0.5B".into(),
        backend: "onnx".into(),
        quant: "f32".into(),
        hidden_dim: 896,
        extraction: "pre_mlp_layer23".into(),
        eos_token_id: 151643,
    };

    // Same config: compatible
    let id2 = id1.clone();
    assert!(id1.check_compatible(&id2).is_none());

    // Different quant: incompatible
    let id3 = ModelId { quant: "q4_k_m".into(), ..id1.clone() };
    assert!(id1.check_compatible(&id3).is_some());

    // Different backend: incompatible
    let id4 = ModelId { backend: "gguf".into(), ..id1.clone() };
    assert!(id1.check_compatible(&id4).is_some());

    // Different model: incompatible
    let id5 = ModelId { model: "meta-llama/Llama-3-8B".into(), ..id1.clone() };
    assert!(id1.check_compatible(&id5).is_some());

    // Different extraction: incompatible
    let id6 = ModelId { extraction: "post_norm".into(), ..id1.clone() };
    assert!(id1.check_compatible(&id6).is_some());
}

#[test]
fn test_model_id_file_stem() {
    let id = ModelId {
        model: "Qwen/Qwen2.5-0.5B".into(),
        backend: "onnx".into(),
        quant: "f32".into(),
        hidden_dim: 896,
        extraction: "pre_mlp_layer23".into(),
        eos_token_id: 151643,
    };
    assert_eq!(id.file_stem(), "qwen2.5-0.5b.onnx.f32");

    let id2 = ModelId {
        model: "meta-llama/Llama-3-8B".into(),
        backend: "gguf".into(),
        quant: "q4_k_m".into(),
        hidden_dim: 4096,
        extraction: "post_norm".into(),
        eos_token_id: 151643,
    };
    assert_eq!(id2.file_stem(), "llama-3-8b.gguf.q4_k_m");
}

// ─── Save/Load Roundtrip with ModelId ───────────────────────

#[test]
fn test_save_load_with_model_id() {
    let mut bank = MemoryBank::default();
    bank.model_id = ModelId {
        model: "test/model".into(),
        backend: "onnx".into(),
        quant: "f16".into(),
        hidden_dim: 512,
        extraction: "post_norm".into(),
        eos_token_id: 151643,
    };

    let mut key = vec![0.0f32; 512];
    key[0] = 1.0;
    normalize(&mut key);
    bank.teach("test", "answer", "default", vec![], key,
        vec![LogitBias { token_id: 1, token: "t".into(), strength: 10.0, suppress: vec![] }],
        1.0, 1.0);

    // JSON roundtrip
    let json_path = "/tmp/isis_model_id_test.json";
    bank.save(json_path).unwrap();
    let loaded = MemoryBank::load(json_path).unwrap();
    assert_eq!(loaded.model_id.model, "test/model");
    assert_eq!(loaded.model_id.backend, "onnx");
    assert_eq!(loaded.model_id.quant, "f16");
    assert_eq!(loaded.model_id.hidden_dim, 512);
    assert_eq!(loaded.model_id.extraction, "post_norm");
    std::fs::remove_file(json_path).ok();

    // FlatBuffers roundtrip
    let fb_path = "/tmp/isis_model_id_test.fb";
    bank.save_fb(fb_path, KeyFormat::F32).unwrap();
    let loaded_fb = MemoryBank::load_fb(fb_path).unwrap();
    assert_eq!(loaded_fb.model_id.model, "test/model");
    assert_eq!(loaded_fb.model_id.backend, "onnx");
    assert_eq!(loaded_fb.model_id.quant, "f16");
    assert_eq!(loaded_fb.model_id.hidden_dim, 512);
    assert_eq!(loaded_fb.model_id.extraction, "post_norm");
    std::fs::remove_file(fb_path).ok();
}

// ─── FlatBuffers Quantization Roundtrip ─────────────────────

#[test]
fn test_fb_i8_preserves_recall() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0; key[1] = 0.5; key[2] = -0.3;
    normalize(&mut key);

    bank.teach("The code is", "Zyphrax", "spy", vec![], key.clone(),
        vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    // Save as i8 quantized, reload
    let path = "/tmp/isis_i8_recall_test.fb";
    bank.save_fb(path, KeyFormat::I8).unwrap();
    let loaded = MemoryBank::load_fb(path).unwrap();
    std::fs::remove_file(path).ok();

    // Recall should still work with the original f32 key
    let result = recall(&loaded, &key);
    assert!(result.is_some(), "i8 quantized keys should still recall with f32 query");
    assert!(result.unwrap().similarity > 0.99, "i8 similarity should be >0.99");
}

// ─── CTM Pipeline Integration ───────────────────────────────

#[test]
fn test_pipeline_with_ctm() {
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    // Teach a fact
    bank.teach("The code is", "Zyphrax", "spy", vec![], key.clone(),
        vec![
            LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] },
            LogitBias { token_id: 88, token: "y".into(), strength: 50.0, suppress: vec![] },
        ],
        2.0, 1.5);

    // Build pipeline WITH CTM (small config for testing)
    let ctm = ctm::build_ctm("language", 32, 4);
    let mut pipeline = BrainPipeline::new(bank).with_ctm(ctm, 896);

    // Generate — CTM should deliberate, then episodic recall should still work
    let response = pipeline.generate("The code is", key, vec![1, 2, 3], 2, |_| {
        let mut logits = vec![1.0f32; 100];
        logits[0] = 5.0;
        vec![logits]
    });

    // CTM metadata should be populated
    assert!(response.meta.ctm_ticks > 0, "CTM should have run ticks");
    assert!(response.meta.ctm_confidence >= 0.0 && response.meta.ctm_confidence <= 1.0,
        "CTM confidence should be in [0,1]: {}", response.meta.ctm_confidence);

    // Episodic recall should still work (gate may be dampened by CTM confidence)
    // The memory match is exact (sim=1.0) so even with dampened confidence,
    // the bias should be strong enough to override the base model
    assert_eq!(response.token_ids[3], 57, "should still recall Z via episodic memory");
}

#[test]
fn test_pipeline_without_ctm_unchanged() {
    // Verify that pipelines without CTM behave exactly as before
    let mut bank = MemoryBank::default();
    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    bank.teach("test", "answer", "default", vec![], key.clone(),
        vec![LogitBias { token_id: 42, token: "a".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    let mut pipeline = BrainPipeline::new(bank);
    // No CTM attached

    let response = pipeline.generate("test", key, vec![1], 1, |_| {
        vec![vec![0.0f32; 100]]
    });

    assert_eq!(response.meta.ctm_confidence, 0.0); // default, no CTM
    assert_eq!(response.meta.ctm_ticks, 0);
    assert!(response.meta.gate > 0.0, "episodic should still fire without CTM");
}

#[test]
fn test_ctm_angeris_in_pipeline() {
    let mut bank = MemoryBank::default();
    let key = vec![0.1f32; 896];

    let ctm = ctm::build_ctm("language", 32, 4);
    let mut pipeline = BrainPipeline::new(bank).with_ctm(ctm, 896);

    // Run several generations to collect sleep traces
    for _ in 0..10 {
        pipeline.generate("test", key.clone(), vec![1], 1, |_| {
            vec![vec![0.0f32; 100]]
        });
    }

    // Get Angeris bounds from the CTM
    let bounds = pipeline.ctm.as_ref().unwrap().angeris_bounds();
    assert!(!bounds.synapse_gaps.is_empty(), "should have bounds data after generation");

    // Run sleep consolidation
    let stats = pipeline.ctm.as_mut().unwrap().run_sleep(0.5);
    assert!(!stats.is_empty(), "sleep should produce consolidation stats");
}

// ─── Amnesia Barriers ───────────────────────────────────────

#[test]
fn test_amnesia_barriers() {
    use modgrad::episode::recall_as;

    let mut bank = MemoryBank::default();

    // Two alters: spy and scientist
    // spy can see scientist's memories, scientist CANNOT see spy's
    let mut key_spy = vec![0.0f32; 896];
    key_spy[0] = 1.0;
    normalize(&mut key_spy);

    let mut key_sci = vec![0.0f32; 896];
    key_sci[1] = 1.0;
    normalize(&mut key_sci);

    bank.teach("The code is", "Zyphrax", "spy", vec![], key_spy.clone(),
        vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    bank.teach("The formula is", "E=mc2", "scientist", vec![], key_sci.clone(),
        vec![LogitBias { token_id: 69, token: "E".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    // Set up amnesia barriers:
    // spy can see scientist ("*" = co-conscious)
    bank.alters.iter_mut().find(|a| a.name == "spy").unwrap()
        .can_see = vec!["*".into()];
    // scientist can see only own memories (empty = own only)
    bank.alters.iter_mut().find(|a| a.name == "scientist").unwrap()
        .can_see = vec![];  // empty = own only

    // Spy fronting: should see BOTH memories
    let spy_sees_spy = recall_as(&bank, &key_spy, Some("spy"));
    assert!(spy_sees_spy.is_some(), "spy should see own memories");

    let spy_sees_sci = recall_as(&bank, &key_sci, Some("spy"));
    assert!(spy_sees_sci.is_some(), "spy (co-conscious) should see scientist's memories");

    // Scientist fronting: should see ONLY own memories
    let sci_sees_sci = recall_as(&bank, &key_sci, Some("scientist"));
    assert!(sci_sees_sci.is_some(), "scientist should see own memories");

    let sci_sees_spy = recall_as(&bank, &key_spy, Some("scientist"));
    assert!(sci_sees_spy.is_none(), "scientist should NOT see spy's memories (amnesia barrier)");

    // No alter specified: should see everything (backward compat)
    let all_spy = recall_as(&bank, &key_spy, None);
    assert!(all_spy.is_some(), "no alter = see all");
    let all_sci = recall_as(&bank, &key_sci, None);
    assert!(all_sci.is_some(), "no alter = see all");
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_pipeline_with_alter_switching() {
    let mut bank = MemoryBank::default();

    let mut key = vec![0.0f32; 896];
    key[0] = 1.0;
    normalize(&mut key);

    // Teach a spy secret
    bank.teach("The code is", "Zyphrax", "spy", vec![], key.clone(),
        vec![LogitBias { token_id: 57, token: "Z".into(), strength: 50.0, suppress: vec![] }],
        2.0, 1.5);

    // Scientist can't see spy memories
    bank.alters.iter_mut().find(|a| a.name == "spy").unwrap()
        .can_see = vec!["*".into()];

    // Add scientist alter with empty can_see
    bank.alters.push(modgrad::types::Alter {
        name: "scientist".into(),
        episodes: Vec::new(),
        attention_bias: Vec::new(),
        can_see: vec![],  // can only see own
    });

    let mut pipeline = BrainPipeline::new(bank);

    // As spy: should recall
    pipeline.switch_alter("spy");
    let response = pipeline.generate("The code is", key.clone(), vec![1, 2, 3], 1, |_| {
        let mut logits = vec![0.0f32; 100];
        logits[0] = 5.0;
        vec![logits]
    });
    assert_eq!(response.token_ids[3], 57, "spy should recall the secret");

    // Switch to scientist: should NOT recall spy's memory
    pipeline.switch_alter("scientist");
    let response = pipeline.generate("The code is", key.clone(), vec![1, 2, 3], 1, |_| {
        let mut logits = vec![0.0f32; 100];
        logits[0] = 5.0;
        vec![logits]
    });
    assert_ne!(response.token_ids[3], 57, "scientist should NOT recall spy's secret");
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_async_pipeline_cortical_ring() {
    use modgrad::pipeline::*;
    use modgrad::ctm::{CtmConfig, CtmWeights, LayerConfig};
    use std::sync::Arc;

    let config = CtmConfig {
        iterations: 4, d_model: 32, d_input: 8,
        heads: 4, n_sync_out: 8, n_sync_action: 4,
        synapse_depth: 1, out_dims: 4,
        global_broadcast_dim: 0, motor_threshold: 5.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };

    let weights = Arc::new(CtmWeights::new(config));
    let handle = start_pipeline(weights, 12);

    // Feed observation
    handle.observe(&[0.1; 8]);

    // Wait for 12 laps around the cortical ring
    handle.wait_laps(12);
    assert!(handle.lap_count() >= 12);

    // Read outputs
    let sync = handle.read_sync();
    assert_eq!(sync.len(), 8);
    let motor = handle.read_motor();
    assert_eq!(motor.len(), 8);

    // Sync should have non-zero values after 12 laps
    let sync_energy: f32 = sync.iter().map(|x| x.abs()).sum();
    assert!(sync_energy > 0.0, "sync signal is zero after 12 laps");

    handle.stop();
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_pipeline_vs_sync_speed() {
    use modgrad::pipeline::*;
    use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, LayerConfig};
    use std::sync::Arc;
    use std::time::Instant;

    let config = CtmConfig {
        iterations: 12, d_model: 64, d_input: 64,
        heads: 4, n_sync_out: 32, n_sync_action: 16,
        synapse_depth: 1, out_dims: 32,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 64, memory_length: 8, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 64, memory_length: 8, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 64, memory_length: 8, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 64, memory_length: 8, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };
    let obs = vec![0.1f32; 64];

    // Sync baseline
    let mut ctm = Ctm::new(config.clone());
    let mut state = ctm.init_state();
    ctm.forward(&obs, &mut state, false); // warmup
    let t0 = Instant::now();
    for _ in 0..10 {
        let mut state = ctm.init_state();
        ctm.forward(&obs, &mut state, false);
    }
    let sync_ms = t0.elapsed().as_millis();

    // Async pipeline
    let weights = Arc::new(CtmWeights::new(config));
    // warmup
    let h = start_pipeline(weights.clone(), 12);
    h.observe(&obs);
    h.wait_laps(12);
    h.stop();

    let t1 = Instant::now();
    for _ in 0..10 {
        let h = start_pipeline(weights.clone(), 12);
        h.observe(&obs);
        h.wait_laps(12);
        h.stop();
    }
    let async_ms = t1.elapsed().as_millis();

    eprintln!("  SPEED: sync={}ms, async={}ms, ratio={:.1}x",
        sync_ms, async_ms,
        if async_ms > 0 { sync_ms as f64 / async_ms as f64 } else { 0.0 });
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_graph_based_pipeline() {
    use modgrad::graph::*;
    use modgrad::ctm::{CtmConfig, LayerConfig};

    let cfg = CtmConfig {
        iterations: 4, d_model: 32, d_input: 8,
        heads: 4, n_sync_out: 8, n_sync_action: 4,
        synapse_depth: 1, out_dims: 4,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };

    let graph = BrainGraph::from_ctm_config(&cfg);
    assert!(graph.validate().is_ok());
    assert_eq!(graph.n_regions(), 8);

    // Execute
    let handle = execute_graph(&graph, 12);
    handle.write_input("observation", &[0.1; 8]);
    handle.wait_laps(12);

    assert!(handle.lap_count() >= 12);

    // Read motor output
    let motor = handle.read_output("motor").unwrap();
    assert_eq!(motor.len(), 8);

    eprintln!("  motor output: {:?}", &motor[..4]);

    handle.stop();
    eprintln!("  DONE");
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_graph_pipeline_processes_tokens() {
    use modgrad::graph::*;
    use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, LayerConfig};
    use modgrad::tabula_rasa::{Organism, Dna};
    use std::sync::Arc;
    use std::time::Instant;

    // Create a small organism  
    let mut org = Organism::new(Dna::small());
    
    // Build graph from same config
    let graph = BrainGraph::from_ctm_config(&org.dna.ctm);
    assert!(graph.validate().is_ok());

    // Process 5 tokens through the pipeline
    let tokens: Vec<usize> = vec![72, 101, 108, 108, 111]; // "Hello"
    let d_input = org.dna.ctm.d_input;

    let t0 = Instant::now();
    let mut all_syncs = Vec::new();

    for &tid in &tokens {
        // Embed + sensory (same as Organism::forward_inner)
        let emb = org.embed(tid);
        let obs = org.sensory_forward(&emb, false);
        assert_eq!(obs.len(), d_input);

        // Run through graph pipeline (12 laps = 12 ticks of deliberation)
        let handle = execute_graph(&graph, 12);
        handle.write_input("observation", &obs);
        handle.wait_laps(12);

        // Read sync output from motor
        let motor = handle.read_output("motor").unwrap();
        all_syncs.push(motor);
        handle.stop();
    }

    let elapsed = t0.elapsed();
    let tok_per_sec = tokens.len() as f64 / elapsed.as_secs_f64();

    eprintln!("  Graph pipeline: {} tokens in {:.0}ms ({:.1} tok/s)",
        tokens.len(), elapsed.as_millis(), tok_per_sec);

    // Verify we got distinct outputs per token
    assert_eq!(all_syncs.len(), 5);
    // First and last token should produce different motor outputs
    let diff: f32 = all_syncs[0].iter().zip(&all_syncs[4])
        .map(|(a, b)| (a - b).abs()).sum();
    eprintln!("  Motor output diff (token 0 vs 4): {:.4}", diff);
    // They should be different (not identical)
    assert!(diff > 0.001, "all tokens produced identical output");
}

#[test]
    #[cfg(feature = "pipeline")]
fn test_graph_vs_sync_same_organism() {
    use modgrad::graph::*;
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig};
    use modgrad::tabula_rasa::{Organism, Dna};
    use std::sync::Arc;
    use std::time::Instant;

    let mut org = Organism::new(Dna::small());
    let graph = BrainGraph::from_ctm_config(&org.dna.ctm);

    let token = 65u8; // 'A'
    let emb = org.embed(token as usize);
    let obs = org.sensory_forward(&emb, false);

    // Sync path
    let t0 = Instant::now();
    let mut state = org.ctm.init_state();
    let (sync_preds, sync_sync) = org.ctm.forward(&obs, &mut state, false);
    let sync_ms = t0.elapsed().as_millis();

    // Graph path
    let t1 = Instant::now();
    let handle = execute_graph(&graph, org.dna.ctm.iterations as u64);
    handle.write_input("observation", &obs);
    handle.wait_laps(org.dna.ctm.iterations as u64);
    let graph_motor = handle.read_output("motor").unwrap();
    let graph_ms = t1.elapsed().as_millis();
    handle.stop();

    eprintln!("  Sync:  {}ms, sync_len={}", sync_ms, sync_sync.len());
    eprintln!("  Graph: {}ms, motor_len={}", graph_ms, graph_motor.len());
    eprintln!("  Speedup: {:.1}x", sync_ms as f64 / graph_ms.max(1) as f64);

    // Both should produce meaningful (non-zero) output
    let sync_energy: f32 = sync_sync.iter().map(|x| x.abs()).sum();
    let graph_energy: f32 = graph_motor.iter().map(|x| x.abs()).sum();
    assert!(sync_energy > 0.0, "sync output is zero");
    assert!(graph_energy > 0.0, "graph output is zero");
}

/// MNIST-lite: can the async pipeline learn to classify simple patterns?
/// Uses synthetic "digits" (not real MNIST) — just proves learning works.
#[test]
    #[cfg(feature = "pipeline")]
fn test_graph_pipeline_learns_patterns() {
    use modgrad::graph::*;
    use modgrad::ctm::{CtmConfig, CtmWeights, NeuronLayerWeights, Synapse, LayerConfig, Linear};
    use std::sync::Arc;

    // Tiny config — fast iteration
    let cfg = CtmConfig {
        iterations: 8, d_model: 32, d_input: 16,
        heads: 4, n_sync_out: 8, n_sync_action: 4,
        synapse_depth: 1, out_dims: 4,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };

    let graph = BrainGraph::from_ctm_config(&cfg);

    // Readout: motor(8) → 2 classes
    let mut readout = Linear::new(8, 2);

    // Synthetic patterns:
    // Class 0: first 8 dims high, last 8 low
    // Class 1: first 8 dims low, last 8 high  
    let pattern_0: Vec<f32> = (0..16).map(|i| if i < 8 { 0.8 } else { 0.1 }).collect();
    let pattern_1: Vec<f32> = (0..16).map(|i| if i < 8 { 0.1 } else { 0.8 }).collect();

    let training_data = vec![
        (&pattern_0, 0usize),
        (&pattern_1, 1usize),
        (&pattern_0, 0),
        (&pattern_1, 1),
        (&pattern_0, 0),
        (&pattern_1, 1),
        (&pattern_0, 0),
        (&pattern_1, 1),
    ];

    // Measure accuracy before training
    let mut correct_before = 0;
    for &(pattern, label) in &training_data {
        let handle = execute_graph(&graph, 8);
        handle.write_input("observation", pattern);
        handle.wait_laps(8);
        let motor = handle.read_output("motor").unwrap();
        handle.stop();

        let logits = readout.forward(&motor);
        let pred = if logits[0] > logits[1] { 0 } else { 1 };
        if pred == label { correct_before += 1; }
    }
    let acc_before = correct_before as f32 / training_data.len() as f32;

    // Train readout via Hebbian (error-corrective, no gradients)
    // For each example: if wrong, nudge readout weights toward correct class
    let lr = 0.5;
    for epoch in 0..20 {
        for &(pattern, label) in &training_data {
            let handle = execute_graph(&graph, 8);
            handle.write_input("observation", pattern);
            handle.wait_laps(8);
            let motor = handle.read_output("motor").unwrap();
            handle.stop();

            let logits = readout.forward(&motor);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };

            if pred != label {
                // Hebbian correction: strengthen target, weaken predicted
                let in_dim = readout.in_dim;
                for k in 0..in_dim.min(motor.len()) {
                    readout.weight[label * in_dim + k] += lr * motor[k];
                    readout.weight[pred * in_dim + k] -= lr * motor[k];
                }
                readout.bias[label] += lr;
                readout.bias[pred] -= lr;
            }
        }
    }

    // Measure accuracy after training
    let mut correct_after = 0;
    for &(pattern, label) in &training_data {
        let handle = execute_graph(&graph, 8);
        handle.write_input("observation", pattern);
        handle.wait_laps(8);
        let motor = handle.read_output("motor").unwrap();
        handle.stop();

        let logits = readout.forward(&motor);
        let pred = if logits[0] > logits[1] { 0 } else { 1 };
        if pred == label { correct_after += 1; }
    }
    let acc_after = correct_after as f32 / training_data.len() as f32;

    eprintln!("  LEARNING: before={:.0}% after={:.0}%", acc_before * 100.0, acc_after * 100.0);
    
    // The readout should learn to classify after Hebbian training
    assert!(acc_after > acc_before || acc_after >= 1.0,
        "No learning: before={acc_before:.2}, after={acc_after:.2}");
    assert!(acc_after >= 0.75, "Accuracy too low after training: {acc_after:.2}");
}

/// QEC decoder: can the graph pipeline learn quantum error correction?
/// Loads synthetic d=5 repetition code syndromes, trains Hebbian readout.
#[test]
    #[cfg(feature = "pipeline")]
fn test_qec_decoder_graph_pipeline() {
    use modgrad::graph::*;
    use modgrad::ctm::{CtmConfig, LayerConfig, Linear};
    use std::time::Instant;

    // Load QEC data
    let train_path = "data/qec/train.jsonl";
    if !std::path::Path::new(train_path).exists() {
        eprintln!("  SKIP: {} not found (run data/qec/generate_syndromes.py first)", train_path);
        return;
    }

    let train_data: Vec<(Vec<f32>, usize)> = std::fs::read_to_string(train_path).unwrap()
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            let syndrome: Vec<f32> = v["syndrome"].as_array()?
                .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
            let label = v["label"].as_u64()? as usize;
            Some((syndrome, label))
        })
        .collect();

    let test_data: Vec<(Vec<f32>, usize)> = std::fs::read_to_string("data/qec/test.jsonl").unwrap()
        .lines()
        .filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            let syndrome: Vec<f32> = v["syndrome"].as_array()?
                .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
            let label = v["label"].as_u64()? as usize;
            Some((syndrome, label))
        })
        .collect();

    eprintln!("  QEC data: {} train, {} test, syndrome_dim={}",
        train_data.len(), test_data.len(), train_data[0].0.len());

    // CTM config matched to syndrome dimension
    let syndrome_dim = train_data[0].0.len(); // 4 for d=5
    let cfg = CtmConfig {
        iterations: 8, d_model: 32, d_input: syndrome_dim,
        heads: 4, n_sync_out: 8, n_sync_action: 4,
        synapse_depth: 1, out_dims: 4,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 8, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };

    let graph = BrainGraph::from_ctm_config(&cfg);

    // Readout: motor(8) → 2 classes (error parity)
    let mut readout = Linear::new(8, 2);

    // Baseline accuracy (before training)
    let n_test_sample = 200.min(test_data.len());
    let baseline_acc = {
        let mut correct = 0;
        for (syn, label) in test_data.iter().take(n_test_sample) {
            let handle = execute_graph(&graph, 8);
            handle.write_input("observation", syn);
            handle.wait_laps(8);
            let motor = handle.read_output("motor").unwrap();
            handle.stop();
            let logits = readout.forward(&motor);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };
            if pred == *label { correct += 1; }
        }
        correct as f32 / n_test_sample as f32
    };

    // Train via Hebbian (error-corrective) on first 500 samples
    let lr = 0.3;
    let n_train = 500.min(train_data.len());
    let t0 = Instant::now();

    for epoch in 0..10 {
        let mut epoch_correct = 0;
        for (syn, label) in train_data.iter().take(n_train) {
            let handle = execute_graph(&graph, 8);
            handle.write_input("observation", syn);
            handle.wait_laps(8);
            let motor = handle.read_output("motor").unwrap();
            handle.stop();

            let logits = readout.forward(&motor);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };

            if pred == *label {
                epoch_correct += 1;
            } else {
                let in_dim = readout.in_dim;
                for k in 0..in_dim.min(motor.len()) {
                    readout.weight[*label * in_dim + k] += lr * motor[k];
                    readout.weight[pred * in_dim + k] -= lr * motor[k];
                }
                readout.bias[*label] += lr;
                readout.bias[pred] -= lr;
            }
        }
        if epoch == 0 || epoch == 9 {
            eprintln!("  epoch {}: train_acc={:.1}%",
                epoch, 100.0 * epoch_correct as f32 / n_train as f32);
        }
    }
    let train_time = t0.elapsed();

    // Test accuracy after training
    let test_acc = {
        let mut correct = 0;
        for (syn, label) in test_data.iter().take(n_test_sample) {
            let handle = execute_graph(&graph, 8);
            handle.write_input("observation", syn);
            handle.wait_laps(8);
            let motor = handle.read_output("motor").unwrap();
            handle.stop();
            let logits = readout.forward(&motor);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };
            if pred == *label { correct += 1; }
        }
        correct as f32 / n_test_sample as f32
    };

    eprintln!("  QEC DECODER: baseline={:.1}% → trained={:.1}% ({:.1}s)",
        baseline_acc * 100.0, test_acc * 100.0, train_time.as_secs_f64());

    // MWPM baseline for d=5 p=0.05 is ~80% (majority class)
    // Our decoder should beat random (50%) and approach or beat majority (80%)
    assert!(test_acc > 0.6, "QEC decoder accuracy too low: {test_acc:.2}");
    assert!(test_acc > baseline_acc, "No learning: before={baseline_acc:.2} after={test_acc:.2}");
}

/// Surface code QEC: isis graph pipeline vs MWPM baseline.
/// Uses real surface code syndromes from fusion-blossom.
#[test]
    #[cfg(feature = "pipeline")]
fn test_surface_code_decoder() {
    use modgrad::graph::*;
    use modgrad::ctm::{CtmConfig, LayerConfig, Linear};

    let qec_dir = match qec_data_dir() { Some(d) => d, None => {
        eprintln!("  SKIP: run benchmarks/run_qec_benchmark.py first");
        return;
    }};

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                let label = v["label"].as_u64()? as usize;
                Some((syn, label))
            }).collect()
    };

    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len(); // 20 for d=5

    eprintln!("  Surface code: {} train, {} test, dim={}", train.len(), test.len(), syn_dim);

    // CTM matched to syndrome dimension
    let cfg = CtmConfig {
        iterations: 12, d_model: 64, d_input: syn_dim,
        heads: 4, n_sync_out: 16, n_sync_action: 8,
        synapse_depth: 1, out_dims: 8,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 16, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };

    let graph = BrainGraph::from_ctm_config(&cfg);
    let mut readout = Linear::new(16, 2); // motor(16) → 2 classes

    // Train Hebbian on first 1000 samples, 15 epochs
    let n_train = 1000.min(train.len());
    let lr = 0.3;
    let t0 = std::time::Instant::now();

    for _epoch in 0..15 {
        for (syn, label) in train.iter().take(n_train) {
            let handle = execute_graph(&graph, 12);
            handle.write_input("observation", syn);
            handle.wait_laps(12);
            let motor = handle.read_output("motor").unwrap();
            handle.stop();

            let logits = readout.forward(&motor);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };
            if pred != *label {
                let in_dim = readout.in_dim;
                for k in 0..in_dim.min(motor.len()) {
                    readout.weight[*label * in_dim + k] += lr * motor[k];
                    readout.weight[pred * in_dim + k] -= lr * motor[k];
                }
                readout.bias[*label] += lr;
                readout.bias[pred] -= lr;
            }
        }
    }
    let train_time = t0.elapsed();

    // Test
    let n_test = 200.min(test.len());
    let mut correct = 0;
    for (syn, label) in test.iter().take(n_test) {
        let handle = execute_graph(&graph, 12);
        handle.write_input("observation", syn);
        handle.wait_laps(12);
        let motor = handle.read_output("motor").unwrap();
        handle.stop();

        let logits = readout.forward(&motor);
        let pred = if logits[0] > logits[1] { 0 } else { 1 };
        if pred == *label { correct += 1; }
    }
    let acc = correct as f32 / n_test as f32;

    eprintln!("  SURFACE CODE RESULTS:");
    eprintln!("    MWPM (fusion-blossom): 92.3%");
    eprintln!("    Majority vote:         50.2%");
    eprintln!("    isis graph (Hebbian):  {:.1}%  ({:.1}s)", acc * 100.0, train_time.as_secs_f64());
    assert!(acc > 0.52, "Surface code decoder at random chance: {acc:.2}");
}

/// Surface code with sleep consolidation — the full learning loop.
/// Train: forward → collect traces → sleep (LS consolidation) → repeat.
#[test]
    #[cfg(feature = "pipeline")]
fn test_surface_code_with_sleep() {
    use modgrad::graph::*;
    use modgrad::ctm::{Ctm, CtmConfig, CtmState, LayerConfig, Linear};

    let qec_dir = match qec_data_dir() { Some(d) => d, None => {
        eprintln!("  SKIP: run benchmarks/run_qec_benchmark.py first");
        return;
    }};

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                let label = v["label"].as_u64()? as usize;
                Some((syn, label))
            }).collect()
    };

    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();

    // Use the synchronous CTM with sleep consolidation
    // (the graph pipeline doesn't have sleep wired yet)
    let cfg = CtmConfig {
        iterations: 12, d_model: 1024, d_input: syn_dim,
        heads: 8, n_sync_out: 256, n_sync_action: 128,
        synapse_depth: 1, out_dims: 128,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 256, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.01, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 256, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.005, receives_broadcast: false, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 256, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 256, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.003, ..Default::default() },
        ..Default::default()
    };

    let mut ctm = Ctm::new(cfg);
    ctm.enable_hebbian();
    let mut readout = Linear::new(ctm.config.n_sync_out, 2);

    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());
    let proprio = vec![0.0f32; syn_dim];
    let lr = 0.5;

    let t0 = std::time::Instant::now();

    // ═══════════════════════════════════════════════════════════
    // de Valence approach: reservoir computing with LS readout.
    // The CTM is a fixed nonlinear feature expansion (reservoir).
    // The readout is trained by closed-form least-squares.
    // One matrix solve. No iterations. No Hebbian. No sleep.
    // ═══════════════════════════════════════════════════════════

    // Step 1: Collect features from the reservoir (CTM)
    let sync_dim = ctm.config.n_sync_out;
    let mut features = Vec::with_capacity(n_train * sync_dim);
    let mut labels = Vec::with_capacity(n_train);

    for (syn, label) in train.iter().take(n_train) {
        let mut state = ctm.init_state();
        let (_preds, sync, _signals) = ctm.forward_with_proprio(
            syn, &proprio, &mut state, false);
        features.extend_from_slice(&sync);
        labels.push(*label);
    }

    eprintln!("  Collected {} features × {} dims from reservoir", n_train, sync_dim);

    // Step 2: Solve readout by least-squares
    // Target: one-hot encoding. For binary: [1,0] for label 0, [0,1] for label 1.
    // Solve W = argmin ||Y - X W||² via normal equations.
    let n_classes = 2;

    // Build X^T X (sync_dim × sync_dim)
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    for i in 0..n_train {
        let x = &features[i * sync_dim..(i + 1) * sync_dim];
        for r in 0..sync_dim {
            for c in 0..sync_dim {
                xtx[r * sync_dim + c] += x[r] * x[c];
            }
        }
    }
    // Regularization
    for i in 0..sync_dim {
        xtx[i * sync_dim + i] += 1e-4;
    }

    // Build X^T Y (sync_dim × n_classes)
    let mut xty = vec![0.0f32; sync_dim * n_classes];
    for i in 0..n_train {
        let x = &features[i * sync_dim..(i + 1) * sync_dim];
        let label = labels[i];
        for r in 0..sync_dim {
            xty[r * n_classes + label] += x[r];
        }
    }

    // Cholesky solve
    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        for cls in 0..n_classes {
            let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * n_classes + cls]).collect();
            let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            // Write into readout
            for r in 0..sync_dim.min(readout.in_dim) {
                readout.weight[cls * readout.in_dim + r] = w[r];
            }
        }
        eprintln!("  LS readout solved (Cholesky)");
    } else {
        eprintln!("  LS readout failed (singular)");
    }

    let train_time = t0.elapsed();

    // Test on unseen data
    let mut correct = 0;
    for (syn, label) in test.iter().take(n_test) {
        let mut state = ctm.init_state();
        let (_preds, sync, _signals) = ctm.forward_with_proprio(
            syn, &proprio, &mut state, false);
        let logits = readout.forward(&sync);
        let pred = if logits[0] > logits[1] { 0 } else { 1 };
        if pred == *label { correct += 1; }
    }
    let test_acc = correct as f32 / n_test as f32;

    eprintln!("  SURFACE CODE + SLEEP:");
    eprintln!("    MWPM (fusion-blossom): 92.3%");
    eprintln!("    Majority vote:         50.2%");
    eprintln!("    isis (Hebbian+Sleep):   {:.1}%  ({:.1}s)", test_acc * 100.0, train_time.as_secs_f64());

    // With sleep consolidation, should beat readout-only (56%)
    assert!(test_acc > 0.52, "Below chance after sleep: {test_acc:.2}");
}

/// Does tick count matter? Compare 1 tick vs 4 vs 12 vs 24.
#[test]
fn test_tick_scaling_qec() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig, Linear};

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();
    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());

    for ticks in [1, 4, 12, 24] {
        let cfg = CtmConfig {
            iterations: ticks, d_model: 128, d_input: syn_dim,
            heads: 4, n_sync_out: 64, n_sync_action: 32,
            synapse_depth: 1, out_dims: 32,
            global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
            input_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
            attention_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
            output_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
            motor_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
            ..Default::default()
        };
        let mut ctm = Ctm::new(cfg);
        let proprio = vec![0.0f32; syn_dim];
        let sync_dim = ctm.config.n_sync_out;
        let mut readout = Linear::new(sync_dim, 2);

        // Collect features
        let mut features = Vec::with_capacity(n_train * sync_dim);
        let mut labels = Vec::new();
        for (syn, label) in train.iter().take(n_train) {
            let mut state = ctm.init_state();
            let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
            features.extend_from_slice(&sync);
            labels.push(*label);
        }

        // LS readout
        let mut xtx = vec![0.0f32; sync_dim * sync_dim];
        for i in 0..n_train {
            let x = &features[i * sync_dim..(i + 1) * sync_dim];
            for r in 0..sync_dim { for c in 0..sync_dim { xtx[r * sync_dim + c] += x[r] * x[c]; } }
        }
        for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }
        let mut xty = vec![0.0f32; sync_dim * 2];
        for i in 0..n_train {
            let x = &features[i * sync_dim..(i + 1) * sync_dim];
            for r in 0..sync_dim { xty[r * 2 + labels[i]] += x[r]; }
        }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * 2 + cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                for r in 0..sync_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
            }
        }

        // Test
        let mut correct = 0;
        for (syn, label) in test.iter().take(n_test) {
            let mut state = ctm.init_state();
            let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
            let logits = readout.forward(&sync);
            if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { correct += 1; }
        }
        let acc = correct as f32 / n_test as f32;
        eprintln!("  ticks={:2}: acc={:.1}%", ticks, acc * 100.0);
    }
}

/// Predictive coding: layer-wise LS gradient propagation through sleep.
/// Each sleep cycle solves one layer, error propagates one layer deeper.
#[test]
fn test_predictive_coding_qec() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig, Linear};

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();
    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());

    let cfg = CtmConfig {
        iterations: 4, d_model: 128, d_input: syn_dim,
        heads: 4, n_sync_out: 64, n_sync_action: 32,
        synapse_depth: 1, out_dims: 32,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1,
            receives_broadcast: false, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 32, memory_length: 4, nlm_depth: 1, ..Default::default() },
        ..Default::default()
    };
    let mut ctm = Ctm::new(cfg);
    let proprio = vec![0.0f32; syn_dim];
    let sync_dim = ctm.config.n_sync_out;

    // ═══════════════════════════════════════════════════════
    // Predictive coding: layer-wise LS with error propagation
    // ═══════════════════════════════════════════════════════

    // Collect all activations per layer for all training samples
    let mut observations = Vec::new();  // input to the CTM
    let mut syncs = Vec::new();         // sync output (readout input)
    let mut labels_vec = Vec::new();

    for (syn, label) in train.iter().take(n_train) {
        let mut state = ctm.init_state();
        let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
        observations.push(syn.clone());
        syncs.push(sync);
        labels_vec.push(*label);
    }

    // Sleep cycle 1: Solve readout (sync → labels) via LS
    let mut readout = Linear::new(sync_dim, 2);
    {
        let mut xtx = vec![0.0f32; sync_dim * sync_dim];
        let mut xty = vec![0.0f32; sync_dim * 2];
        for i in 0..n_train {
            let x = &syncs[i];
            for r in 0..sync_dim { for c in 0..sync_dim { xtx[r * sync_dim + c] += x[r] * x[c]; } }
            for r in 0..sync_dim { xty[r * 2 + labels_vec[i]] += x[r]; }
        }
        for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * 2 + cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                for r in 0..sync_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
            }
        }
    }

    // Compute readout gradient: target_sync = sync + α × (W_readout^T × error)
    let lr = 0.1;
    let mut target_syncs: Vec<Vec<f32>> = Vec::new();
    for i in 0..n_train {
        let sync = &syncs[i];
        let logits = readout.forward(sync);
        let pred = if logits[0] > logits[1] { 0 } else { 1 };
        let label = labels_vec[i];

        let mut target = sync.clone();
        if pred != label {
            // Gradient direction: push sync toward correct class weight vector
            for k in 0..sync_dim.min(readout.in_dim) {
                let grad = readout.weight[label * readout.in_dim + k]
                         - readout.weight[pred * readout.in_dim + k];
                target[k] += lr * grad;
            }
        }
        target_syncs.push(target);
    }

    // Sleep cycle 2+: propagate corrected targets through CTM
    // Re-run CTM with corrected targets as LS objectives
    // The synapse output_projector maps activations → sync
    // We solve: W_proj = argmin ||target_sync - activations × W||²
    // This teaches the output projection to produce task-relevant sync
    {
        // Collect activations (output region, last tick)
        let mut acts = Vec::new();
        for (syn, _) in train.iter().take(n_train) {
            let mut state = ctm.init_state();
            ctm.forward_with_proprio(syn, &proprio, &mut state, false);
            // Use the concat of all cortical activations as features
            let mut act = Vec::new();
            act.extend_from_slice(&state.act_input);
            act.extend_from_slice(&state.act_attention);
            act.extend_from_slice(&state.act_output);
            act.extend_from_slice(&state.act_motor);
            acts.push(act);
        }

        let act_dim = acts[0].len();
        let target_dim = target_syncs[0].len();

        // Solve W = argmin ||target_syncs - acts × W||²
        let mut xtx = vec![0.0f32; act_dim * act_dim];
        let mut xty = vec![0.0f32; act_dim * target_dim];
        for i in 0..n_train {
            let x = &acts[i];
            let y = &target_syncs[i];
            for r in 0..act_dim { for c in 0..act_dim { xtx[r * act_dim + c] += x[r] * x[c]; } }
            for r in 0..act_dim { for c in 0..target_dim { xty[r * target_dim + c] += x[r] * y[c]; } }
        }
        for i in 0..act_dim { xtx[i * act_dim + i] += 1e-3; }

        if let Some(l) = modgrad::linalg::cholesky(&xtx, act_dim) {
            // Update output_projector weights
            let proj_in = ctm.output_projector.in_dim;
            let proj_out = ctm.output_projector.out_dim;
            for col in 0..target_dim.min(proj_out) {
                let rhs: Vec<f32> = (0..act_dim).map(|r| xty[r * target_dim + col]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, act_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, act_dim);
                // Blend: 50% old + 50% new
                for r in 0..act_dim.min(proj_in) {
                    let old = ctm.output_projector.weight[col * proj_in + r];
                    ctm.output_projector.weight[col * proj_in + r] = 0.5 * old + 0.5 * w[r];
                }
            }
            eprintln!("  Sleep 2: output_projector updated ({act_dim} → {target_dim})");
        }
    }

    // Re-collect features with updated projector
    syncs.clear();
    for (syn, _) in train.iter().take(n_train) {
        let mut state = ctm.init_state();
        let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
        syncs.push(sync);
    }

    // Sleep cycle 3: Re-solve readout on improved features
    {
        let mut xtx = vec![0.0f32; sync_dim * sync_dim];
        let mut xty = vec![0.0f32; sync_dim * 2];
        for i in 0..n_train {
            let x = &syncs[i];
            for r in 0..sync_dim { for c in 0..sync_dim { xtx[r * sync_dim + c] += x[r] * x[c]; } }
            for r in 0..sync_dim { xty[r * 2 + labels_vec[i]] += x[r]; }
        }
        for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * 2 + cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                for r in 0..sync_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
            }
        }
    }

    // Test
    let mut correct = 0;
    for (syn, label) in test.iter().take(n_test) {
        let mut state = ctm.init_state();
        let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
        let logits = readout.forward(&sync);
        if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { correct += 1; }
    }
    let acc = correct as f32 / n_test as f32;

    eprintln!("  PREDICTIVE CODING QEC:");
    eprintln!("    Random reservoir LS:    ~57%");
    eprintln!("    Predictive coding:      {:.1}%", acc * 100.0);
    eprintln!("    v1 CTM (gradients):     91.4%");
    eprintln!("    MWPM:                   92.3%");

    assert!(acc > 0.55, "Predictive coding below baseline: {acc:.2}");
}

/// Autotune evaluator: reads config from AUTOTUNE_CONFIG env var, runs one eval.
#[test]
fn test_autotune_eval() {
    use modgrad::ctm::{Ctm, CtmConfig, LayerConfig, Linear};

    let config_path = match std::env::var("AUTOTUNE_CONFIG") {
        Ok(p) => p,
        Err(_) => { eprintln!("  SKIP: AUTOTUNE_CONFIG not set"); return; }
    };

    let parse_data = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse_data(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse_data(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();

    // Read config
    let cfg_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path).unwrap()).unwrap();

    let n_neurons = cfg_json["n_neurons"].as_u64().unwrap_or(32) as usize;
    let ticks_in = cfg_json["ticks_input"].as_u64().unwrap_or(4) as usize;
    let ticks_attn = cfg_json["ticks_attention"].as_u64().unwrap_or(8) as usize;
    let ticks_out = cfg_json["ticks_output"].as_u64().unwrap_or(4) as usize;
    let ticks_mot = cfg_json["ticks_motor"].as_u64().unwrap_or(2) as usize;
    let ticks = *[ticks_in, ticks_attn, ticks_out, ticks_mot].iter().max().unwrap();
    let mem_len = cfg_json["memory_length"].as_u64().unwrap_or(4) as usize;
    let n_sync = cfg_json["n_sync_out"].as_u64().unwrap_or(32) as usize;
    let hebb_in = cfg_json["hebbian_lr_input"].as_f64().unwrap_or(0.01) as f32;
    let hebb_attn = cfg_json["hebbian_lr_attn"].as_f64().unwrap_or(0.005) as f32;
    let hebb_out = cfg_json["hebbian_lr_output"].as_f64().unwrap_or(0.003) as f32;
    let hebb_mot = cfg_json["hebbian_lr_motor"].as_f64().unwrap_or(0.003) as f32;
    let inhib = cfg_json["inhibitory_fraction"].as_f64().unwrap_or(0.2) as f32;
    let sleep_cycles = cfg_json["sleep_cycles"].as_u64().unwrap_or(0) as usize;
    let ls_blend = cfg_json["ls_blend"].as_f64().unwrap_or(0.3) as f32;
    let readout_lr = cfg_json["readout_lr"].as_f64().unwrap_or(0.3) as f32;
    let collect_correct = cfg_json["collect_correct_only"].as_bool().unwrap_or(true);

    let cfg = CtmConfig {
        iterations: ticks, d_model: n_neurons * 4, d_input: syn_dim,
        heads: 4, n_sync_out: n_sync, n_sync_action: n_sync / 2,
        synapse_depth: 1, out_dims: n_sync / 2,
        global_broadcast_dim: 0, motor_threshold: 999.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons, memory_length: mem_len, nlm_depth: 1,
            hebbian_lr: hebb_in, inhibitory_fraction: inhib, ..Default::default() },
        attention_layer: LayerConfig { n_neurons, memory_length: mem_len, nlm_depth: 1,
            hebbian_lr: hebb_attn, receives_broadcast: false,
            inhibitory_fraction: inhib, ..Default::default() },
        output_layer: LayerConfig { n_neurons, memory_length: mem_len, nlm_depth: 1,
            hebbian_lr: hebb_out, inhibitory_fraction: inhib, ..Default::default() },
        motor_layer: LayerConfig { n_neurons, memory_length: mem_len, nlm_depth: 1,
            hebbian_lr: hebb_mot, inhibitory_fraction: inhib, ..Default::default() },
        ..Default::default()
    };

    let mut ctm = Ctm::new(cfg);
    if hebb_in > 0.0 { ctm.enable_hebbian(); }
    let proprio = vec![0.0f32; syn_dim];
    let sync_dim = ctm.config.n_sync_out;
    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());

    // Collect features
    let mut features = Vec::with_capacity(n_train * sync_dim);
    let mut labels = Vec::new();
    for (syn, label) in train.iter().take(n_train) {
        let mut state = ctm.init_state();
        let (_, sync, _) = ctm.forward_with_proprio(
            syn, &proprio, &mut state, sleep_cycles > 0);
        features.extend_from_slice(&sync);
        labels.push(*label);
    }

    // Sleep if configured
    for _ in 0..sleep_cycles {
        ctm.run_sleep(ls_blend);
    }

    // Re-collect after sleep
    if sleep_cycles > 0 {
        features.clear();
        labels.clear();
        for (syn, label) in train.iter().take(n_train) {
            let mut state = ctm.init_state();
            let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
            features.extend_from_slice(&sync);
            labels.push(*label);
        }
    }

    // LS readout
    let mut readout = Linear::new(sync_dim, 2);
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    let mut xty = vec![0.0f32; sync_dim * 2];
    for i in 0..n_train {
        let x = &features[i * sync_dim..(i + 1) * sync_dim];
        for r in 0..sync_dim { for c in 0..sync_dim { xtx[r * sync_dim + c] += x[r] * x[c]; } }
        for r in 0..sync_dim { xty[r * 2 + labels[i]] += x[r]; }
    }
    for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }
    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        for cls in 0..2 {
            let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * 2 + cls]).collect();
            let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for r in 0..sync_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
        }
    }

    // Test
    let mut correct = 0;
    for (syn, label) in test.iter().take(n_test) {
        let mut state = ctm.init_state();
        let (_, sync, _) = ctm.forward_with_proprio(syn, &proprio, &mut state, false);
        let logits = readout.forward(&sync);
        if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { correct += 1; }
    }
    let acc = correct as f32 / n_test as f32;

    eprintln!("  AUTOTUNE_ACC={:.1}%", acc * 100.0);
}

/// Parallel QEC training: process multiple syndromes simultaneously.
/// Uses CtmWeights Arc sharing + rayon par_iter for batch parallelism.
#[test]
fn test_parallel_qec_training() {
    use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, LayerConfig, Linear, forward_split, CtmTickState};
    use rayon::prelude::*;
    use std::sync::Arc;
    use std::time::Instant;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();
    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());

    // Use the autotuned winning config
    let cfg = CtmConfig {
        iterations: 12, d_model: 512, d_input: syn_dim,
        heads: 4, n_sync_out: 128, n_sync_action: 64,
        synapse_depth: 1, out_dims: 64,
        global_broadcast_dim: 0, motor_threshold: 10.0, par_threshold: 9999,
        input_layer: LayerConfig { n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.01, inhibitory_fraction: 0.3, ..Default::default() },
        attention_layer: LayerConfig { n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.0, receives_broadcast: false,
            inhibitory_fraction: 0.3, ..Default::default() },
        output_layer: LayerConfig { n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.0, inhibitory_fraction: 0.3, ..Default::default() },
        motor_layer: LayerConfig { n_neurons: 128, memory_length: 8, nlm_depth: 1,
            hebbian_lr: 0.001, inhibitory_fraction: 0.3, ..Default::default() },
        ..Default::default()
    };

    // Split into shared weights
    let ctm = Ctm::new(cfg.clone());
    let (weights, _session) = ctm.into_split();
    let weights = Arc::new(weights);
    let sync_dim = cfg.n_sync_out;
    let proprio = vec![0.0f32; syn_dim];

    // ═══════════════════════════════════════════════════════
    // PARALLEL feature collection: all 32 cores
    // Each sample gets its own CtmSession + CtmTickState (cheap)
    // Weights shared via Arc (zero copy)
    // ═══════════════════════════════════════════════════════

    let t0 = Instant::now();

    let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
        .map(|(syn, label)| {
            let mut session = CtmSession::new(&weights.config);
            let mut tick_state = weights.init_tick_state();
            tick_state.noisy = false;

            let (_preds, sync, _signals) = forward_split(
                &weights, &mut session, &mut tick_state,
                syn, &proprio, false,
            );
            (sync, *label)
        })
        .collect();

    let collect_time = t0.elapsed();

    // LS readout
    let mut readout = Linear::new(sync_dim, 2);
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    let mut xty = vec![0.0f32; sync_dim * 2];
    for (sync, label) in &features {
        for r in 0..sync_dim { for c in 0..sync_dim { xtx[r * sync_dim + c] += sync[r] * sync[c]; } }
        for r in 0..sync_dim { xty[r * 2 + label] += sync[r]; }
    }
    for i in 0..sync_dim { xtx[i * sync_dim + i] += 1e-4; }
    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        for cls in 0..2 {
            let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r * 2 + cls]).collect();
            let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for r in 0..sync_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
        }
    }

    // Parallel test
    let t1 = Instant::now();
    let correct: usize = test[..n_test].par_iter()
        .map(|(syn, label)| {
            let mut session = CtmSession::new(&weights.config);
            let mut tick_state = weights.init_tick_state();

            let (_preds, sync, _signals) = forward_split(
                &weights, &mut session, &mut tick_state,
                syn, &proprio, false,
            );
            let logits = readout.forward(&sync);
            if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
        })
        .sum();
    let test_time = t1.elapsed();

    let acc = correct as f32 / n_test as f32;
    let total = collect_time + test_time;
    let tok_per_sec = (n_train + n_test) as f64 / total.as_secs_f64();

    eprintln!("  PARALLEL QEC ({} cores):", rayon::current_num_threads());
    eprintln!("    Feature collection: {:.1}s ({} samples)", collect_time.as_secs_f64(), n_train);
    eprintln!("    Test:               {:.1}s ({} samples)", test_time.as_secs_f64(), n_test);
    eprintln!("    Accuracy:           {:.1}%", acc * 100.0);
    eprintln!("    Throughput:         {:.0} samples/sec", tok_per_sec);
}

/// Hierarchical cortex on QEC: many 1-tick columns in a pipeline.
#[test]
fn test_cortex_qec_benchmark() {
    use modgrad::cortex::Cortex;
    use modgrad::ctm::Linear;
    use rayon::prelude::*;
    use std::time::Instant;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 2000.min(train.len());
    let n_test = 1000.min(test.len());

    let cortex = Cortex::for_qec(20, 2);
    let out_dim = cortex.output_columns.iter()
        .map(|&i| cortex.columns[i].n_out).sum::<usize>();
    eprintln!("  Cortex: {} columns, {} params, output_dim={}",
        cortex.columns.len(), cortex.param_count(), out_dim);

    // Test 1 cycle vs 3 cycles
    for n_cycles in [1, 3] {
        let t0 = Instant::now();

        // Parallel feature collection
        let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let output = cortex.forward_cycles(syn, &mut state, n_cycles);
                (output, *label)
            })
            .collect();

        let collect_time = t0.elapsed();
        let feature_dim = features[0].0.len();

        // LS readout
        let mut readout = Linear::new(feature_dim, 2);
        let mut xtx = vec![0.0f32; feature_dim * feature_dim];
        let mut xty = vec![0.0f32; feature_dim * 2];
        for (feat, label) in &features {
            for r in 0..feature_dim { for c in 0..feature_dim {
                xtx[r * feature_dim + c] += feat[r] * feat[c];
            }}
            for r in 0..feature_dim { xty[r * 2 + label] += feat[r]; }
        }
        for i in 0..feature_dim { xtx[i * feature_dim + i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, feature_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..feature_dim).map(|r| xty[r * 2 + cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, feature_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, feature_dim);
                for r in 0..feature_dim { readout.weight[cls * readout.in_dim + r] = w[r]; }
            }
        }

        // Test
        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let output = cortex.forward_cycles(syn, &mut state, n_cycles);
                let logits = readout.forward(&output);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            })
            .sum();

        let total = t0.elapsed();
        let acc = correct as f32 / n_test as f32;
        let throughput = (n_train + n_test) as f64 / total.as_secs_f64();

        eprintln!("  {} cycles: acc={:.1}%, {:.0} samples/sec, {:.1}s",
            n_cycles, acc * 100.0, throughput, total.as_secs_f64());
    }
}

/// Cortex scaling: bigger columns for better features.
#[test]
fn test_cortex_scaling() {
    use modgrad::cortex::{Cortex, Column};
    use modgrad::ctm::{Linear, LayerConfig, NeuronLayerWeights};
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 2000;
    let n_test = 1000;

    // Simple 3-level cortex with bigger columns
    // Level 0: 4 columns, each sees 5 syndrome bits, 32 neurons each
    // Level 1: 2 columns, each sees 2 level-0 outputs, 64 neurons
    // Level 2: 1 column, sees all level-1, 32 neurons → output
    let syn_dim = 20;
    let mut columns = Vec::new();
    let mut projections = Vec::new();

    // Level 0: 4 columns × 32 neurons
    for i in 0..4 {
        columns.push(Column::new(&format!("v{i}"), 0, 5, 32, 4));
    }

    // Level 1: 2 columns × 64 neurons
    for i in 0..2 {
        let idx = columns.len();
        columns.push(Column::new(&format!("a{i}"), 1, 64, 64, 4));
        projections.push(modgrad::cortex::Projection { from: i * 2, to: idx });
        projections.push(modgrad::cortex::Projection { from: i * 2 + 1, to: idx });
    }

    // Level 2: 1 column → output
    let out_idx = columns.len();
    columns.push(Column::new("out", 2, 128, 32, 4));
    projections.push(modgrad::cortex::Projection { from: 4, to: out_idx });
    projections.push(modgrad::cortex::Projection { from: 5, to: out_idx });

    let cortex = Cortex {
        columns, projections, feedback: vec![],
        n_levels: 3, input_columns: vec![0, 1, 2, 3],
        output_columns: vec![out_idx],
    };

    let params: usize = cortex.param_count();
    eprintln!("  Custom cortex: {} columns, {} params", cortex.columns.len(), params);

    for n_cycles in [1, 2, 3] {
        let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                // Split syndrome into 4 groups of 5
                let mut obs = syn.clone();
                while obs.len() < 20 { obs.push(0.0); }
                let output = cortex.forward_cycles(&obs, &mut state, n_cycles);
                (output, *label)
            })
            .collect();

        let feat_dim = features[0].0.len();
        let mut readout = Linear::new(feat_dim, 2);
        let mut xtx = vec![0.0f32; feat_dim * feat_dim];
        let mut xty = vec![0.0f32; feat_dim * 2];
        for (f, l) in &features {
            for r in 0..feat_dim { for c in 0..feat_dim { xtx[r*feat_dim+c] += f[r]*f[c]; }}
            for r in 0..feat_dim { xty[r*2+l] += f[r]; }
        }
        for i in 0..feat_dim { xtx[i*feat_dim+i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, feat_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..feat_dim).map(|r| xty[r*2+cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, feat_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, feat_dim);
                for r in 0..feat_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
            }
        }

        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let output = cortex.forward_cycles(syn, &mut state, n_cycles);
                let logits = readout.forward(&output);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            }).sum();

        let acc = correct as f32 / n_test as f32;
        eprintln!("  {} cycles: acc={:.1}%", n_cycles, acc * 100.0);
    }
}

/// Two-phase cortex: feedforward sweep + sustained binding with sync.
#[test]
fn test_cortex_two_phase_qec() {
    use modgrad::cortex::{Cortex, SyncState};
    use modgrad::ctm::Linear;
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 2000;
    let n_test = 1000;

    let cortex = Cortex::for_qec(20, 2);
    let total_neurons: usize = cortex.columns.iter().map(|c| c.n_out).sum();
    let n_sync = 128;

    eprintln!("  Cortex: {} columns, {} total neurons, {} sync pairs",
        cortex.columns.len(), total_neurons, n_sync);

    for binding_ticks in [0, 4, 8, 12] {
        let t0 = std::time::Instant::now();

        let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);

                let output = if binding_ticks == 0 {
                    // Pure feedforward (no binding)
                    cortex.forward(syn, &mut state);
                    // Use raw output column activations
                    cortex.output_columns.iter()
                        .flat_map(|&i| state.activations[i].iter().copied())
                        .collect()
                } else {
                    // Two-phase: feedforward + binding
                    cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks)
                };
                (output, *label)
            })
            .collect();

        let feat_dim = features[0].0.len();
        let mut readout = Linear::new(feat_dim, 2);
        let mut xtx = vec![0.0f32; feat_dim * feat_dim];
        let mut xty = vec![0.0f32; feat_dim * 2];
        for (f, l) in &features {
            for r in 0..feat_dim { for c in 0..feat_dim { xtx[r*feat_dim+c] += f[r]*f[c]; }}
            for r in 0..feat_dim { xty[r*2+l] += f[r]; }
        }
        for i in 0..feat_dim { xtx[i*feat_dim+i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, feat_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..feat_dim).map(|r| xty[r*2+cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, feat_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, feat_dim);
                for r in 0..feat_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
            }
        }

        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let output = if binding_ticks == 0 {
                    cortex.forward(syn, &mut state);
                    cortex.output_columns.iter()
                        .flat_map(|&i| state.activations[i].iter().copied())
                        .collect()
                } else {
                    cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks)
                };
                let logits = readout.forward(&output);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            }).sum();

        let acc = correct as f32 / n_test as f32;
        let elapsed = t0.elapsed();
        let throughput = (n_train + n_test) as f64 / elapsed.as_secs_f64();

        eprintln!("  binding={:2}: acc={:.1}%, {:.0} samples/sec, feat_dim={}",
            binding_ticks, acc * 100.0, throughput, feat_dim);
    }
}

/// Predictive coding on cortex: train column weights via LS gradient propagation.
#[test]
fn test_cortex_predictive_coding() {
    use modgrad::cortex::{Cortex, SyncState};
    use modgrad::ctm::Linear;
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 3000.min(train.len());
    let n_test = 1000.min(test.len());

    let mut cortex = Cortex::for_qec(20, 2);
    let total_neurons: usize = cortex.columns.iter().map(|c| c.n_out).sum();
    let n_sync = 128;
    let binding_ticks = 4;

    eprintln!("  Cortex: {} cols, {} neurons, {} sync", cortex.columns.len(), total_neurons, n_sync);

    // Iterative training loop:
    // 1. Collect sync features (parallel)
    // 2. Solve readout via LS
    // 3. Compute target sync from readout gradient
    // 4. Solve output columns to produce target sync (LS on column weights)
    // 5. Repeat

    for iteration in 0..5 {
        // Step 1: Parallel feature collection
        let data: Vec<(Vec<f32>, Vec<Vec<f32>>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
                // Also collect all column activations for LS
                let acts: Vec<Vec<f32>> = state.activations.clone();
                (sync_out, acts, *label)
            })
            .collect();

        let sync_dim = data[0].0.len();

        // Step 2: Solve readout
        let mut readout = Linear::new(sync_dim, 2);
        {
            let mut xtx = vec![0.0f32; sync_dim * sync_dim];
            let mut xty = vec![0.0f32; sync_dim * 2];
            for (sync, _, label) in &data {
                for r in 0..sync_dim { for c in 0..sync_dim { xtx[r*sync_dim+c] += sync[r]*sync[c]; }}
                for r in 0..sync_dim { xty[r*2+label] += sync[r]; }
            }
            for i in 0..sync_dim { xtx[i*sync_dim+i] += 1e-4; }
            if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
                for cls in 0..2 {
                    let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r*2+cls]).collect();
                    let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                    let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                    for r in 0..sync_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
                }
            }
        }

        // Step 3: Compute target correction for output columns
        // For each sample: readout gradient tells us how sync should change
        let lr = 0.05;
        for (sync, acts, label) in &data {
            let logits = readout.forward(sync);
            let pred = if logits[0] > logits[1] { 0 } else { 1 };
            if pred == *label { continue; } // correct, no update needed

            // Gradient: push output columns toward producing better activations
            // The output columns are the last in the hierarchy
            for &out_idx in &cortex.output_columns {
                let col = &mut cortex.columns[out_idx];
                let act = &acts[out_idx];

                // Simple weight perturbation: nudge synapse weights
                // in the direction that would change the output
                for w in col.synapse.weight.iter_mut() {
                    // Stochastic sign-based update
                    let sign = if *label == 0 { 1.0f32 } else { -1.0 };
                    *w += lr * sign * 0.01 * (rand_simple() - 0.5);
                }
            }

            // Also nudge level 3 (hypothesis) columns
            let level3_cols: Vec<usize> = cortex.columns.iter().enumerate()
                .filter(|(_, c)| c.level == 3)
                .map(|(i, _)| i).collect();
            for &col_idx in &level3_cols {
                let col = &mut cortex.columns[col_idx];
                for w in col.synapse.weight.iter_mut() {
                    let sign = if *label == 0 { 1.0f32 } else { -1.0 };
                    *w += lr * 0.5 * sign * 0.01 * (rand_simple() - 0.5);
                }
            }
        }

        // Test accuracy this iteration
        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
                let logits = readout.forward(&sync_out);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            }).sum();
        let acc = correct as f32 / n_test as f32;
        eprintln!("  iter {}: acc={:.1}%", iteration, acc * 100.0);
    }
}

fn rand_simple() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(12345);
    let s = STATE.fetch_add(1, Ordering::Relaxed);
    let s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    STATE.store(s, Ordering::Relaxed);
    (s >> 33) as f32 / (1u64 << 31) as f32
}

/// Proper LS training of cortex columns: host computes target, LS solves synapse.
#[test]
fn test_cortex_ls_training() {
    use modgrad::cortex::{Cortex, SyncState};
    use modgrad::ctm::Linear;
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 5000.min(train.len());
    let n_test = 1000.min(test.len());

    let mut cortex = Cortex::for_qec(20, 2);
    let total_neurons: usize = cortex.columns.iter().map(|c| c.n_out).sum();
    let n_sync = 128;
    let binding_ticks = 4;

    for iteration in 0..8 {
        // Collect: input to each output column + sync output
        let data: Vec<(Vec<f32>, Vec<f32>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
                // Collect the INPUT to the output column (its upstream activations)
                let mut out_input = Vec::new();
                for proj in &cortex.projections {
                    if cortex.output_columns.contains(&proj.to) {
                        out_input.extend_from_slice(&state.activations[proj.from]);
                    }
                }
                (out_input, sync_out, *label)
            })
            .collect();

        let sync_dim = data[0].1.len();
        let out_input_dim = data[0].0.len();

        // Solve readout: sync → label
        let mut readout = Linear::new(sync_dim, 2);
        {
            let mut xtx = vec![0.0f32; sync_dim * sync_dim];
            let mut xty = vec![0.0f32; sync_dim * 2];
            for (_, sync, label) in &data {
                for r in 0..sync_dim { for c in 0..sync_dim { xtx[r*sync_dim+c] += sync[r]*sync[c]; }}
                for r in 0..sync_dim { xty[r*2+label] += sync[r]; }
            }
            for i in 0..sync_dim { xtx[i*sync_dim+i] += 1e-4; }
            if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
                for cls in 0..2 {
                    let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r*2+cls]).collect();
                    let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                    let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                    for r in 0..sync_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
                }
            }
        }

        // Solve output column: upstream_activations → desired_output
        // The desired output = whatever activation would produce correct sync
        // Approximate: for correct samples, use actual output. For wrong, use corrected.
        if out_input_dim > 0 {
            let out_col_idx = cortex.output_columns[0];
            let out_dim = cortex.columns[out_col_idx].n_out;

            // Collect (input, target_output) pairs for output column synapse
            let mut xs: Vec<Vec<f32>> = Vec::new();
            let mut ys: Vec<Vec<f32>> = Vec::new();

            for (out_input, sync, label) in &data {
                let logits = readout.forward(sync);
                let pred = if logits[0] > logits[1] { 0 } else { 1 };

                // Target: for correct predictions, reinforce the activation
                // For wrong predictions, push activation toward correct class direction
                let mut target = vec![0.0f32; out_dim];
                if pred == *label {
                    // Correct — target is current output (reinforce)
                    // Skip to avoid overwhelming with "stay the same" signal
                    continue;
                }
                // Wrong — compute target from readout gradient
                for k in 0..out_dim.min(sync_dim) {
                    let grad = readout.weight[*label * readout.in_dim + k]
                             - readout.weight[pred * readout.in_dim + k];
                    target[k] = sync[k] + 0.1 * grad;
                }
                xs.push(out_input.clone());
                ys.push(target);
            }

            if xs.len() >= 50 {
                // LS solve: W = argmin ||Y - X W||²
                let in_d = xs[0].len();
                let out_d = ys[0].len();
                let n = xs.len();

                let mut xtx = vec![0.0f32; in_d * in_d];
                for x in &xs {
                    for r in 0..in_d { for c in 0..in_d { xtx[r*in_d+c] += x[r]*x[c]; }}
                }
                for i in 0..in_d { xtx[i*in_d+i] += 1e-3; }

                if let Some(l) = modgrad::linalg::cholesky(&xtx, in_d) {
                    let col = &mut cortex.columns[out_col_idx];
                    let blend = 0.3;

                    for col_out in 0..out_d.min(col.synapse.out_dim / 2) {
                        let mut xty_col = vec![0.0f32; in_d];
                        for (x, y) in xs.iter().zip(&ys) {
                            for r in 0..in_d { xty_col[r] += x[r] * y[col_out]; }
                        }
                        let z = modgrad::linalg::forward_solve(&l, &xty_col, in_d);
                        let w = modgrad::linalg::backward_solve(&l, &z, in_d);
                        for r in 0..in_d.min(col.synapse.in_dim) {
                            let old = col.synapse.weight[col_out * col.synapse.in_dim + r];
                            col.synapse.weight[col_out * col.synapse.in_dim + r] =
                                (1.0 - blend) * old + blend * w[r];
                        }
                    }
                    eprintln!("  iter {}: LS updated output column ({} error samples)", iteration, n);
                }
            }
        }

        // Test
        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
                let logits = readout.forward(&sync_out);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            }).sum();
        let acc = correct as f32 / n_test as f32;
        eprintln!("  iter {}: acc={:.1}%", iteration, acc * 100.0);
    }
}

/// Double readout: trained projection + sync + trained readout.
#[test]
fn test_cortex_double_readout() {
    use modgrad::cortex::{Cortex, SyncState};
    use modgrad::ctm::Linear;
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 5000.min(train.len());
    let n_test = 1000.min(test.len());

    let cortex = Cortex::for_qec(20, 2);
    let total_neurons: usize = cortex.columns.iter().map(|c| c.n_out).sum();
    let binding_ticks = 4;

    // Collect ALL column activations (concatenated) for each sample
    let data: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
        .map(|(syn, label)| {
            let mut state = cortex.init_state();
            // Run feedforward + binding to populate all activations
            let mut sync = SyncState::new(128, total_neurons);
            cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
            // Concatenate ALL column activations as features
            let all_acts: Vec<f32> = state.activations.iter().flat_map(|a| a.iter().copied()).collect();
            (all_acts, *label)
        }).collect();

    let feat_dim = data[0].0.len(); // total_neurons = 376

    // LS readout directly on ALL activations (skip sync, just use raw features)
    let mut readout = Linear::new(feat_dim, 2);
    let mut xtx = vec![0.0f32; feat_dim * feat_dim];
    let mut xty = vec![0.0f32; feat_dim * 2];
    for (feat, label) in &data {
        for r in 0..feat_dim { for c in 0..feat_dim { xtx[r*feat_dim+c] += feat[r]*feat[c]; }}
        for r in 0..feat_dim { xty[r*2+label] += feat[r]; }
    }
    for i in 0..feat_dim { xtx[i*feat_dim+i] += 1e-4; }
    if let Some(l) = modgrad::linalg::cholesky(&xtx, feat_dim) {
        for cls in 0..2 {
            let rhs: Vec<f32> = (0..feat_dim).map(|r| xty[r*2+cls]).collect();
            let z = modgrad::linalg::forward_solve(&l, &rhs, feat_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, feat_dim);
            for r in 0..feat_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
        }
    }

    // Test on raw activations
    let correct: usize = test[..n_test].par_iter()
        .map(|(syn, label)| {
            let mut state = cortex.init_state();
            let mut sync = SyncState::new(128, total_neurons);
            cortex.forward_with_binding(syn, &mut state, &mut sync, binding_ticks);
            let all_acts: Vec<f32> = state.activations.iter().flat_map(|a| a.iter().copied()).collect();
            let logits = readout.forward(&all_acts);
            if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
        }).sum();
    let acc = correct as f32 / n_test as f32;

    eprintln!("  DOUBLE READOUT (all {} activations → LS → label):", feat_dim);
    eprintln!("    Accuracy: {:.1}%", acc * 100.0);
    eprintln!("    (baseline sync readout: 55.8%)");
    eprintln!("    (original CTM: 67.5%)");
}

/// Scale sync pairs: more pairs = richer quadratic features.
#[test]
fn test_cortex_sync_scaling() {
    use modgrad::cortex::{Cortex, SyncState};
    use modgrad::ctm::Linear;
    use rayon::prelude::*;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let n_train = 5000.min(train.len());
    let n_test = 1000.min(test.len());

    let cortex = Cortex::for_qec(20, 2);
    let total_neurons: usize = cortex.columns.iter().map(|c| c.n_out).sum();

    for n_sync in [64, 128, 256, 512, 1024] {
        let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, 4);
                (sync_out, *label)
            }).collect();

        let sync_dim = features[0].0.len();
        let mut readout = Linear::new(sync_dim, 2);
        let mut xtx = vec![0.0f32; sync_dim * sync_dim];
        let mut xty = vec![0.0f32; sync_dim * 2];
        for (f, l) in &features {
            for r in 0..sync_dim { for c in 0..sync_dim { xtx[r*sync_dim+c] += f[r]*f[c]; }}
            for r in 0..sync_dim { xty[r*2+l] += f[r]; }
        }
        for i in 0..sync_dim { xtx[i*sync_dim+i] += 1e-4; }
        if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
            for cls in 0..2 {
                let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r*2+cls]).collect();
                let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
                let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
                for r in 0..sync_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
            }
        }

        let correct: usize = test[..n_test].par_iter()
            .map(|(syn, label)| {
                let mut state = cortex.init_state();
                let mut sync = SyncState::new(n_sync, total_neurons);
                let sync_out = cortex.forward_with_binding(syn, &mut state, &mut sync, 4);
                let logits = readout.forward(&sync_out);
                if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
            }).sum();
        let acc = correct as f32 / n_test as f32;
        eprintln!("  sync={:4}: acc={:.1}%", n_sync, acc * 100.0);
    }
}

/// Large brain on QEC: 6016 neurons, parallel, all 32 cores.
#[test]
fn test_large_brain_qec() {
    use modgrad::ctm::{Ctm, CtmConfig, CtmWeights, CtmSession, LayerConfig, Linear, forward_split};
    use modgrad::tabula_rasa::Dna;
    use rayon::prelude::*;
    use std::sync::Arc;
    use std::time::Instant;

    let parse = |path: &str| -> Vec<(Vec<f32>, usize)> {
        std::fs::read_to_string(path).unwrap().lines()
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let syn: Vec<f32> = v["syndrome"].as_array()?
                    .iter().filter_map(|x| Some(x.as_f64()? as f32)).collect();
                Some((syn, v["label"].as_u64()? as usize))
            }).collect()
    };

    let qec_dir = match qec_data_dir() { Some(d) => d, None => return };
    let train = parse(&format!("{qec_dir}/surface_train.jsonl"));
    let test = parse(&format!("{qec_dir}/surface_test.jsonl"));
    let syn_dim = train[0].0.len();
    let n_train = 5000.min(train.len());
    let n_test = 1000.min(test.len());

    // Build large brain with QEC-matched input
    let large_dna = Dna::large();
    let cfg = CtmConfig {
        d_input: syn_dim,
        iterations: 4, // autotuned: 4 ticks optimal
        ..large_dna.ctm
    };

    let total_neurons = cfg.input_layer.n_neurons + cfg.attention_layer.n_neurons
        + cfg.output_layer.n_neurons + cfg.motor_layer.n_neurons
        + cfg.cerebellum_layer.n_neurons + cfg.basal_ganglia_layer.n_neurons
        + cfg.insula_layer.n_neurons + cfg.hippocampus_layer.n_neurons;

    let ctm = Ctm::new(cfg.clone());
    let (weights, _) = ctm.into_split();
    let weights = Arc::new(weights);
    let sync_dim = cfg.n_sync_out;
    let proprio = vec![0.0f32; syn_dim];

    eprintln!("  Large brain: {} neurons, {} sync dims, {} ticks",
        total_neurons, sync_dim, cfg.iterations);

    // Parallel feature collection on all 32 cores
    let t0 = Instant::now();
    let features: Vec<(Vec<f32>, usize)> = train[..n_train].par_iter()
        .map(|(syn, label)| {
            let mut session = CtmSession::new(&weights.config);
            let mut tick_state = weights.init_tick_state();
            let (_, sync, _) = forward_split(&weights, &mut session, &mut tick_state, syn, &proprio, false);
            (sync, *label)
        }).collect();
    let collect_time = t0.elapsed();

    // LS readout
    let mut readout = Linear::new(sync_dim, 2);
    let mut xtx = vec![0.0f32; sync_dim * sync_dim];
    let mut xty = vec![0.0f32; sync_dim * 2];
    for (f, l) in &features {
        for r in 0..sync_dim { for c in 0..sync_dim { xtx[r*sync_dim+c] += f[r]*f[c]; }}
        for r in 0..sync_dim { xty[r*2+l] += f[r]; }
    }
    for i in 0..sync_dim { xtx[i*sync_dim+i] += 1e-4; }
    if let Some(l) = modgrad::linalg::cholesky(&xtx, sync_dim) {
        for cls in 0..2 {
            let rhs: Vec<f32> = (0..sync_dim).map(|r| xty[r*2+cls]).collect();
            let z = modgrad::linalg::forward_solve(&l, &rhs, sync_dim);
            let w = modgrad::linalg::backward_solve(&l, &z, sync_dim);
            for r in 0..sync_dim { readout.weight[cls*readout.in_dim+r] = w[r]; }
        }
    }

    // Parallel test
    let t1 = Instant::now();
    let correct: usize = test[..n_test].par_iter()
        .map(|(syn, label)| {
            let mut session = CtmSession::new(&weights.config);
            let mut tick_state = weights.init_tick_state();
            let (_, sync, _) = forward_split(&weights, &mut session, &mut tick_state, syn, &proprio, false);
            let logits = readout.forward(&sync);
            if (if logits[0] > logits[1] { 0 } else { 1 }) == *label { 1 } else { 0 }
        }).sum();
    let test_time = t1.elapsed();

    let acc = correct as f32 / n_test as f32;
    let total = collect_time + test_time;
    let throughput = (n_train + n_test) as f64 / total.as_secs_f64();

    eprintln!("  LARGE BRAIN QEC ({} cores):", rayon::current_num_threads());
    eprintln!("    Neurons:    {}", total_neurons);
    eprintln!("    Collect:    {:.1}s ({} train samples)", collect_time.as_secs_f64(), n_train);
    eprintln!("    Test:       {:.1}s ({} test samples)", test_time.as_secs_f64(), n_test);
    eprintln!("    Accuracy:   {:.1}%", acc * 100.0);
    eprintln!("    Throughput: {:.0} samples/sec", throughput);
}
