//! nanochat-rs — train a GPT on text using the modgrad SDK.
//!
//! Usage:
//!   cargo run --release -p nanochat-rs -- --data train.txt --steps 200

use modgrad::traits::{Brain, LastTickCE, LossFn, SeqInput};
use modgrad::trainer::{train, TrainConfig, Sample, SampleProvider, StderrLogger};
use modgrad::optim::ConstantLR;
use modgrad::transformer::config::*;
use modgrad::transformer::dims::*;
use modgrad::transformer::weights::{GptWeights, BlockWeights};
use modgrad::transformer::offload::WeightOffloader;
use modgrad::transformer::rope::RotaryEmbedding;
use modgrad::transformer::train::{Transformer, TransformerWeights};
use modgrad::neuron::SimpleRng;

use std::io::Read;

// ─── Data ─────────────────────────────────────────────────────

struct ByteData { data: Vec<u8>, pos: usize, seq_len: usize }

impl SampleProvider<SeqInput, usize> for ByteData {
    fn next_sample(&mut self) -> Option<Sample<SeqInput, usize>> {
        if self.pos + self.seq_len + 1 > self.data.len() { return None; }
        let ids: Vec<u32> = self.data[self.pos..self.pos + self.seq_len]
            .iter().map(|&b| b as u32).collect();
        let target = self.data[self.pos + self.seq_len] as usize;
        self.pos += 1;
        Some(Sample { input: SeqInput { token_ids: ids }, target })
    }
    fn reset(&mut self) { self.pos = 0; }
}

// ─── Main ─────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_path = "train_climbmix.txt".to_string();
    let mut steps = 200usize;
    let mut depth = 4usize;
    let mut lr = 3e-4f32;
    let mut seq_len = 64usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => { data_path = args[i+1].clone(); i += 2; }
            "--steps" => { steps = args[i+1].parse().unwrap(); i += 2; }
            "--depth" => { depth = args[i+1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i+1].parse().unwrap(); i += 2; }
            "--seq-len" => { seq_len = args[i+1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    // Config — nanochat scaling: width = 128 * depth
    let n_embd = (128 * depth).max(256).min(4096);
    let hd = 64;
    let config = GptConfig {
        model_dim: ModelDim::new(n_embd),
        num_heads: NumHeads::new(n_embd / hd),
        num_kv_heads: NumKvHeads::new(n_embd / hd),
        head_dim: HeadDim::new(hd),
        num_layers: NumLayers::new(depth),
        vocab_size: VocabSize::new(256),
        mlp_dim: MlpDim::new(4 * n_embd),
        max_seq_len: SeqLen::new(seq_len),
        ..Default::default()
    };

    // Init weights
    let mut rng = SimpleRng::new(42);
    let md = n_embd;
    let kv = config.num_kv_heads.get() * hd;
    let mlp = 4 * n_embd;
    let s = |fan: usize| (2.0 / fan as f32).sqrt();
    let r = |rng: &mut SimpleRng, n: usize, sc: f32| -> Vec<f32> {
        (0..n).map(|_| rng.next_normal() * sc).collect()
    };

    let blocks: Vec<BlockWeights> = (0..depth).map(|li| {
        let layer_idx = LayerIdx::new(li, config.num_layers).unwrap();
        let (ve_t, ve_g) = if config.has_value_embed(layer_idx) {
            (Some(r(&mut rng, 256 * kv, s(kv))),
             Some(r(&mut rng, config.num_kv_heads.get() * config.value_embed.gate_channels, s(12))))
        } else { (None, None) };
        BlockWeights {
            wq: r(&mut rng, md*md, s(md)), wk: r(&mut rng, kv*md, s(md)),
            wv: r(&mut rng, kv*md, s(md)), wo: r(&mut rng, md*md, s(md)),
            mlp_fc: r(&mut rng, mlp*md, s(md)), mlp_proj: r(&mut rng, md*mlp, s(mlp)),
            ve_table: ve_t, ve_gate: ve_g,
        }
    }).collect();

    let norm_scale = vec![1.0f32; md];
    let offloader = WeightOffloader::from_weights(
        GptWeights {
            token_embed: r(&mut rng, 256 * md, s(md)),
            lm_head: r(&mut rng, 256 * md, s(md)),
            final_norm_scale: norm_scale.clone(),
            smear_gate: r(&mut rng, md * config.smear.gate_channels, s(24)),
            blocks,
        },
        &config, 0,
    );
    let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_base);

    let mut weights = TransformerWeights {
        offloader, config: config.clone(), rope, norm_scale,
    };

    let n_params: usize = weights.offloader.layers.iter().map(|l| l.data.len()).sum::<usize>()
        + weights.offloader.embed.len() + weights.offloader.lm_head.len();
    eprintln!("nanochat-rs: depth={depth} d_model={md} params={:.1}M seq_len={seq_len}",
        n_params as f64 / 1e6);

    // Data
    let mut text = Vec::new();
    std::fs::File::open(&data_path).unwrap().read_to_end(&mut text).unwrap();
    eprintln!("data: {data_path} ({:.1}MB)\n", text.len() as f64 / 1e6);
    let mut data = ByteData { data: text, pos: 0, seq_len };

    // Train — one call
    let report = train::<Transformer, _, _, _, _>(
        &mut weights, &mut data, &LastTickCE,
        &ConstantLR { lr },
        &mut None, &mut StderrLogger,
        &TrainConfig { total_steps: steps, micro_batch: 1, log_every: 20,
                       token_dim: md, ..Default::default() },
    );

    eprintln!("done: loss={:.3} in {:.1}s", report.final_loss, report.elapsed_secs);
}
