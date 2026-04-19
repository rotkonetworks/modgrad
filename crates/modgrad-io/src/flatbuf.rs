//! FlatBuffers serialization for isis memory banks.
//!
//! Converts between our runtime types (types.rs) and the generated
//! FlatBuffers schema. Supports f32/f16/i8 key quantization.

use flatbuffers::FlatBufferBuilder;

#[path = "generated/isis_generated.rs"]
#[allow(unused_imports, dead_code, clippy::all, warnings)]
mod generated;

use generated::isis::fb as fb;

use crate::memory::MemoryBank as RtBank;
use modgrad_persist::quantize::{self, KeyFormat};
use crate::types::*;

/// Serialize a MemoryBank to FlatBuffers bytes with the given key format.
pub fn serialize(bank: &RtBank, format: KeyFormat) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1024 * 1024);

    // Build ModelId
    let mid_model = fbb.create_string(&bank.model_id.model);
    let mid_backend = fbb.create_string(&bank.model_id.backend);
    let mid_quant = fbb.create_string(&bank.model_id.quant);
    let mid_extraction = fbb.create_string(&bank.model_id.extraction);
    let model_id = fb::ModelId::create(&mut fbb, &fb::ModelIdArgs {
        model: Some(mid_model),
        backend: Some(mid_backend),
        quant: Some(mid_quant),
        hidden_dim: bank.model_id.hidden_dim,
        extraction: Some(mid_extraction),
    });

    // Build alters
    let alters: Vec<_> = bank
        .alters
        .iter()
        .map(|alter| {
            let name = fbb.create_string(&alter.name);
            let episodes: Vec<_> = alter
                .episodes
                .iter()
                .map(|ep| build_episode(&mut fbb, ep, format))
                .collect();
            let episodes = fbb.create_vector(&episodes);
            fb::Alter::create(&mut fbb, &fb::AlterArgs {
                name: Some(name),
                episodes: Some(episodes),
            })
        })
        .collect();
    let alters = fbb.create_vector(&alters);

    // Build rules
    let rules: Vec<_> = bank
        .rules
        .iter()
        .map(|rule| {
            let instruction = fbb.create_string(&rule.instruction);
            let trigger = fbb.create_string(&rule.trigger);
            fb::Rule::create(&mut fbb, &fb::RuleArgs {
                instruction: Some(instruction),
                priority: rule.priority,
                trigger: Some(trigger),
                active: rule.active,
            })
        })
        .collect();
    let rules = fbb.create_vector(&rules);

    // Build avoidances
    let avoidances: Vec<_> = bank
        .avoidances
        .iter()
        .map(|av| {
            let pattern = fbb.create_string(&av.pattern);
            let reason = fbb.create_string(&av.reason);
            let key = build_key_data(&mut fbb, &av.key, format);
            let suppress = fbb.create_vector(&av.suppress_token_ids);
            fb::Avoidance::create(&mut fbb, &fb::AvoidanceArgs {
                pattern: Some(pattern),
                reason: Some(reason),
                key: Some(key),
                suppress_token_ids: Some(suppress),
                strength: av.strength,
                active: av.active,
            })
        })
        .collect();
    let avoidances = fbb.create_vector(&avoidances);

    let _key_dim = bank.model_id.hidden_dim;

    let fb_format = match format {
        KeyFormat::F32 => fb::KeyFormat::F32,
        KeyFormat::F16 => fb::KeyFormat::F16,
        KeyFormat::I8 => fb::KeyFormat::I8,
    };

    let root = fb::MemoryBank::create(&mut fbb, &fb::MemoryBankArgs {
        version: bank.version,
        model_id: Some(model_id),
        threshold: bank.threshold,
        key_format: fb_format,
        alters: Some(alters),
        rules: Some(rules),
        avoidances: Some(avoidances),
    });

    fbb.finish(root, Some("isis"));
    fbb.finished_data().to_vec()
}

/// Deserialize a FlatBuffers buffer back to a MemoryBank.
pub fn deserialize(buf: &[u8]) -> Result<RtBank, String> {
    let fb_bank = flatbuffers::root::<fb::MemoryBank>(buf)
        .map_err(|e| format!("invalid flatbuffer: {e}"))?;

    let model_id = if let Some(mid) = fb_bank.model_id() {
        crate::memory::ModelId {
            model: mid.model().unwrap_or("").into(),
            backend: mid.backend().unwrap_or("").into(),
            quant: mid.quant().unwrap_or("").into(),
            hidden_dim: mid.hidden_dim(),
            extraction: mid.extraction().unwrap_or("").into(),
            eos_token_id: 151643, // FIXME: hardcoded to Qwen2-style EOS;
            // different tokenizers use different IDs. Loading a non-Qwen
            // model via this path gives wrong EOS detection. Add
            // `eos_token_id` to the FlatBuffers schema and read it back.
        }
    } else {
        crate::memory::ModelId::default()
    };

    let mut bank = RtBank {
        version: fb_bank.version(),
        model_id,
        threshold: fb_bank.threshold(),
        alters: Vec::new(),
        rules: Vec::new(),
        avoidances: Vec::new(),
    };

    if let Some(alters) = fb_bank.alters() {
        for fb_alter in alters {
            let mut alter = Alter {
                name: fb_alter.name().unwrap_or("").into(),
                episodes: Vec::new(),
                attention_bias: Vec::new(),
                can_see: Vec::new(),
            };
            if let Some(episodes) = fb_alter.episodes() {
                for fb_ep in episodes {
                    alter.episodes.push(read_episode(&fb_ep));
                }
            }
            bank.alters.push(alter);
        }
    }

    if let Some(rules) = fb_bank.rules() {
        for fb_rule in rules {
            bank.rules.push(Rule {
                instruction: fb_rule.instruction().unwrap_or("").into(),
                priority: fb_rule.priority(),
                trigger: fb_rule.trigger().unwrap_or("").into(),
                active: fb_rule.active(),
            });
        }
    }

    if let Some(avoidances) = fb_bank.avoidances() {
        for fb_av in avoidances {
            let key = fb_av
                .key()
                .map(|k| read_key_data(&k))
                .unwrap_or_default();
            bank.avoidances.push(Avoidance {
                pattern: fb_av.pattern().unwrap_or("").into(),
                reason: fb_av.reason().unwrap_or("").into(),
                key,
                suppress_token_ids: fb_av
                    .suppress_token_ids()
                    .map(|v| v.iter().collect())
                    .unwrap_or_default(),
                strength: fb_av.strength(),
                active: fb_av.active(),
            });
        }
    }

    Ok(bank)
}

// --- Key quantization helpers ---

fn build_key_data<'a, A: flatbuffers::Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    key: &[f32],
    format: KeyFormat,
) -> flatbuffers::WIPOffset<fb::KeyData<'a>> {
    match format {
        KeyFormat::F32 => {
            let data = fbb.create_vector(key);
            fb::KeyData::create(fbb, &fb::KeyDataArgs {
                format: fb::KeyFormat::F32,
                f32_data: Some(data),
                f16_data: None,
                i8_data: None,
                i8_scale: 0.0,
            })
        }
        KeyFormat::F16 => {
            let f16_vals = quantize::f32_to_f16(key);
            let data = fbb.create_vector(&f16_vals);
            fb::KeyData::create(fbb, &fb::KeyDataArgs {
                format: fb::KeyFormat::F16,
                f32_data: None,
                f16_data: Some(data),
                i8_data: None,
                i8_scale: 0.0,
            })
        }
        KeyFormat::I8 => {
            let (i8_vals, scale) = quantize::f32_to_i8(key);
            let data = fbb.create_vector(&i8_vals);
            fb::KeyData::create(fbb, &fb::KeyDataArgs {
                format: fb::KeyFormat::I8,
                f32_data: None,
                f16_data: None,
                i8_data: Some(data),
                i8_scale: scale,
            })
        }
    }
}

fn read_key_data(kd: &fb::KeyData) -> Vec<f32> {
    match kd.format() {
        fb::KeyFormat::F16 => {
            if let Some(data) = kd.f16_data() {
                let vals: Vec<u16> = data.iter().collect();
                quantize::f16_to_f32(&vals)
            } else {
                Vec::new()
            }
        }
        fb::KeyFormat::I8 => {
            if let Some(data) = kd.i8_data() {
                let vals: Vec<i8> = data.iter().collect();
                quantize::i8_to_f32(&vals, kd.i8_scale())
            } else {
                Vec::new()
            }
        }
        _ => {
            // F32 or unknown → use f32_data
            if let Some(data) = kd.f32_data() {
                data.iter().collect()
            } else {
                Vec::new()
            }
        }
    }
}

// --- Episode builders ---

fn build_episode<'a, A: flatbuffers::Allocator + 'a>(
    fbb: &mut FlatBufferBuilder<'a, A>,
    ep: &Episode,
    format: KeyFormat,
) -> flatbuffers::WIPOffset<fb::Episode<'a>> {
    let prompt = fbb.create_string(&ep.prompt);
    let answer = fbb.create_string(&ep.answer);
    let alter = fbb.create_string(&ep.alter);

    let keys: Vec<_> = ep
        .keys
        .iter()
        .map(|ck| {
            let key = build_key_data(fbb, &ck.key, format);
            let token = fbb.create_string(&ck.token);
            fb::ContentKey::create(fbb, &fb::ContentKeyArgs {
                key: Some(key),
                token: Some(token),
                position: ck.position,
            })
        })
        .collect();
    let keys = fbb.create_vector(&keys);

    let biases: Vec<_> = ep
        .logit_biases
        .iter()
        .map(|lb| {
            let token = fbb.create_string(&lb.token);
            let suppress: Vec<_> = lb
                .suppress
                .iter()
                .map(|&(tid, bias)| {
                    fb::SuppressEntry::create(fbb, &fb::SuppressEntryArgs {
                        token_id: tid,
                        bias,
                    })
                })
                .collect();
            let suppress = fbb.create_vector(&suppress);
            fb::LogitBias::create(fbb, &fb::LogitBiasArgs {
                token_id: lb.token_id,
                token: Some(token),
                strength: lb.strength,
                suppress: Some(suppress),
            })
        })
        .collect();
    let biases = fbb.create_vector(&biases);

    fb::Episode::create(fbb, &fb::EpisodeArgs {
        prompt: Some(prompt),
        answer: Some(answer),
        alter: Some(alter),
        keys: Some(keys),
        logit_biases: Some(biases),
        strength: ep.strength,
        recall_count: ep.recall_count,
        created_at: ep.created_at,
        consolidated: ep.consolidated,
    })
}

fn read_episode(fb_ep: &fb::Episode) -> Episode {
    let keys = fb_ep
        .keys()
        .map(|ks| {
            ks.iter()
                .map(|ck| ContentKey {
                    key: ck.key().map(|k| read_key_data(&k)).unwrap_or_default(),
                    token: ck.token().unwrap_or("").into(),
                    position: ck.position(),
                })
                .collect()
        })
        .unwrap_or_default();

    let logit_biases = fb_ep
        .logit_biases()
        .map(|lbs| {
            lbs.iter()
                .map(|lb| LogitBias {
                    token_id: lb.token_id(),
                    token: lb.token().unwrap_or("").into(),
                    strength: lb.strength(),
                    suppress: lb
                        .suppress()
                        .map(|s| s.iter().map(|se| (se.token_id(), se.bias())).collect())
                        .unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default();

    Episode {
        prompt: fb_ep.prompt().unwrap_or("").into(),
        answer: fb_ep.answer().unwrap_or("").into(),
        alter: fb_ep.alter().unwrap_or("").into(),
        keys,
        logit_biases,
        strength: fb_ep.strength(),
        recall_count: fb_ep.recall_count(),
        created_at: fb_ep.created_at(),
        consolidated: fb_ep.consolidated(),
        consolidation_score: 0.0,
        sleep_cycles: 0,
        valence: Valence::Neutral,
        last_recalled_at: 0.0,
        visible_to: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bank() -> RtBank {
        let mut bank = RtBank::default();
        bank.alters.push(Alter {
            name: "default".into(),
            attention_bias: Vec::new(),
            can_see: Vec::new(),
            episodes: vec![Episode {
                prompt: "capital of France".into(),
                answer: "Paris".into(),
                alter: "default".into(),
                keys: vec![ContentKey {
                    key: (0..896).map(|i| (i as f32) / 896.0 - 0.5).collect(),
                    token: "France".into(),
                    position: 3,
                }],
                logit_biases: vec![LogitBias {
                    token_id: 12345,
                    token: "Paris".into(),
                    strength: 5.0,
                    suppress: vec![(999, -5.0)],
                }],
                strength: 1.0,
                recall_count: 0,
                created_at: 1700000000.0,
                consolidated: false, consolidation_score: 0.0, sleep_cycles: 0,
                valence: Valence::Neutral, last_recalled_at: 0.0, visible_to: Vec::new(),
            }],
        });
        bank.rules.push(Rule {
            instruction: "Be concise".into(),
            priority: 1.0,
            trigger: String::new(),
            active: true,
        });
        bank
    }

    #[test]
    fn roundtrip_f32() {
        let bank = make_test_bank();
        let bytes = serialize(&bank, KeyFormat::F32);
        let restored = deserialize(&bytes).unwrap();

        assert_eq!(restored.version, bank.version);
        assert_eq!(restored.model_id.model, bank.model_id.model);
        assert_eq!(restored.alters.len(), 1);
        assert_eq!(restored.alters[0].episodes.len(), 1);
        assert_eq!(restored.alters[0].episodes[0].answer, "Paris");
        assert_eq!(restored.alters[0].episodes[0].keys[0].key.len(), 896);
        assert_eq!(restored.rules.len(), 1);

        // f32 should be lossless
        let orig_key = &bank.alters[0].episodes[0].keys[0].key;
        let rest_key = &restored.alters[0].episodes[0].keys[0].key;
        assert_eq!(orig_key, rest_key);
    }

    #[test]
    fn roundtrip_f16() {
        let bank = make_test_bank();
        let bytes = serialize(&bank, KeyFormat::F16);
        let restored = deserialize(&bytes).unwrap();

        let orig_key = &bank.alters[0].episodes[0].keys[0].key;
        let rest_key = &restored.alters[0].episodes[0].keys[0].key;
        let sim = quantize::cosine_f32(orig_key, rest_key);
        assert!(sim > 0.9999, "f16 roundtrip sim = {sim}");
    }

    #[test]
    fn roundtrip_i8() {
        let bank = make_test_bank();
        let bytes = serialize(&bank, KeyFormat::I8);
        let restored = deserialize(&bytes).unwrap();

        let orig_key = &bank.alters[0].episodes[0].keys[0].key;
        let rest_key = &restored.alters[0].episodes[0].keys[0].key;
        let sim = quantize::cosine_f32(orig_key, rest_key);
        assert!(sim > 0.999, "i8 roundtrip sim = {sim}");
    }

    #[test]
    fn compression_ratio() {
        let bank = make_test_bank();
        let json_bytes = serde_json::to_string(&bank).unwrap().len();
        let f32_bytes = serialize(&bank, KeyFormat::F32).len();
        let f16_bytes = serialize(&bank, KeyFormat::F16).len();
        let i8_bytes = serialize(&bank, KeyFormat::I8).len();

        // FlatBuffers should always be smaller than JSON for float-heavy data
        assert!(f32_bytes < json_bytes, "f32 fb={f32_bytes} >= json={json_bytes}");
        assert!(f16_bytes < f32_bytes, "f16={f16_bytes} >= f32={f32_bytes}");
        assert!(i8_bytes < f16_bytes, "i8={i8_bytes} >= f16={f16_bytes}");

        eprintln!("JSON: {json_bytes} bytes");
        eprintln!("FB f32: {f32_bytes} bytes ({:.1}x smaller)", json_bytes as f64 / f32_bytes as f64);
        eprintln!("FB f16: {f16_bytes} bytes ({:.1}x smaller)", json_bytes as f64 / f16_bytes as f64);
        eprintln!("FB i8:  {i8_bytes} bytes ({:.1}x smaller)", json_bytes as f64 / i8_bytes as f64);
    }
}
