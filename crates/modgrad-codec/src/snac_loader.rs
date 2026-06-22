//! Load `hubertsiuzdak/snac_24khz` weights into the modgrad `SnacDecoder24k`.
//!
//! Expects the safetensors file written by `scripts/snac_export.py`
//! (weight_norm already fused). Tensor names follow PyTorch:
//!
//! ```text
//!   decoder.model.0.weight       [768, 1, 7]      depthwise conv
//!   decoder.model.0.bias         [768]
//!   decoder.model.1.weight       [1024, 768, 1]   1×1 projection
//!   decoder.model.1.bias         [1024]
//!   decoder.model.2..5           DecoderBlock     (see read_decoder_block)
//!   decoder.model.6.alpha        [1, 64, 1]       final snake α
//!   decoder.model.7.weight       [1, 64, 7]       final mono conv
//!   decoder.model.7.bias         [1]
//!   quantizer.quantizers.N.codebook.weight     [4096, 8]
//!   quantizer.quantizers.N.out_proj.weight     [768, 8, 1]
//!   quantizer.quantizers.N.out_proj.bias       [768]
//! ```

use std::path::Path;

use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

use crate::snac::{
    Codebook, DecoderBlock, NoiseBlock, ResidualUnit, ResidualVQ,
    SnacConv1d, SnacConvTranspose1d, SnacDecoder24k, Snake1d,
};

/// Per-DecoderBlock rate + (in_channels, out_channels) for the 24 kHz config.
/// Order matches `decoder.model.{2,3,4,5}`.
const BLOCK_SPEC: [(usize, usize, usize); 4] = [
    /* (rate, in_ch, out_ch) */
    (8, 1024, 512),
    (8,  512, 256),
    (4,  256, 128),
    (2,  128,  64),
];

/// Dilations used by SNAC residual units within each block.
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];

/// SNAC's three codebooks at the decoder side. Strides match
/// `config.json::vq_strides = [4, 2, 1]`.
const CODEBOOK_STRIDES: [usize; 3] = [4, 2, 1];

const CODE_DIM: usize = 8;
const N_CODES: usize = 4096;
const DECODER_IN_CH: usize = 768;

impl SnacDecoder24k {
    /// Build a `SnacDecoder24k` from the safetensors file dumped by
    /// `scripts/snac_export.py`.
    pub fn load_from_safetensors(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("open {:?}: {e}", path.as_ref()))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| format!("mmap: {e}"))?
        };
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("safetensors deserialize: {e}"))?;

        // ── Initial conv pair (depthwise + 1×1 projection) ──
        let conv_in_dw = read_conv1d(&st, "decoder.model.0", 768, 768, 7,
                                     /*stride*/1, /*dilation*/1, /*padding*/3, /*groups*/768)?;
        let conv_in_proj = read_conv1d(&st, "decoder.model.1", 768, 1024, 1,
                                       1, 1, 0, 1)?;

        // ── 4 DecoderBlocks ──
        let mut blocks = Vec::with_capacity(4);
        for (idx, &(rate, in_ch, out_ch)) in BLOCK_SPEC.iter().enumerate() {
            let prefix = format!("decoder.model.{}", idx + 2);
            blocks.push(read_decoder_block(&st, &prefix, rate, in_ch, out_ch)?);
        }

        // ── Final Snake + Conv1d ──
        let snake_final = read_snake(&st, "decoder.model.6.alpha", 64)?;
        let conv_out = read_conv1d(&st, "decoder.model.7", 64, 1, 7,
                                   1, 1, 3, 1)?;

        // ── ResidualVQ (3 codebooks) ──
        let codebooks = (0..3)
            .map(|i| read_codebook(&st, i, CODEBOOK_STRIDES[i]))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(SnacDecoder24k {
            vq: ResidualVQ { codebooks },
            conv_in_dw,
            conv_in_proj,
            blocks,
            snake_final,
            conv_out,
        })
    }
}

fn read_decoder_block(
    st: &SafeTensors, prefix: &str, rate: usize, in_ch: usize, out_ch: usize,
) -> Result<DecoderBlock, String> {
    // block.0.alpha          → pre-Snake α (in_ch channels)
    let snake_pre = read_snake(st, &format!("{prefix}.block.0.alpha"), in_ch)?;

    // block.1.weight/bias    → ConvTranspose1d in_ch → out_ch, k=2·rate, stride=rate
    // SNAC uses padding=ceil(stride/2) on every transposed conv — see snac.layers.
    let upsample = read_convt1d(
        st, &format!("{prefix}.block.1"), in_ch, out_ch,
        /*kernel*/   2 * rate,
        /*stride*/   rate,
        /*padding*/  (rate + 1) / 2,    // = ceil(rate / 2)
        /*output_padding*/ rate % 2,
    )?;

    // block.2.linear.weight  → NoiseBlock 1×1 conv [out_ch, out_ch]
    let noise = read_noise_block(st, &format!("{prefix}.block.2.linear.weight"), out_ch)?;

    // block.3-5 = ResidualUnit at dilations 1, 3, 9
    let res_units: Vec<ResidualUnit> = RESIDUAL_DILATIONS
        .iter()
        .enumerate()
        .map(|(i, &dil)| read_residual_unit(st, &format!("{prefix}.block.{}", i + 3),
                                            out_ch, dil))
        .collect::<Result<Vec<_>, _>>()?;
    let mut iter = res_units.into_iter();
    let res1 = iter.next().unwrap();
    let res3 = iter.next().unwrap();
    let res9 = iter.next().unwrap();

    Ok(DecoderBlock { snake_pre, upsample, noise, res1, res3, res9 })
}

fn read_residual_unit(
    st: &SafeTensors, prefix: &str, channels: usize, dilation: usize,
) -> Result<ResidualUnit, String> {
    // ResidualUnit layout:
    //   block.0.alpha            first Snake
    //   block.1.weight/bias      dilated DEPTHWISE conv (k=7, in/groups=1)
    //   block.2.alpha            second Snake
    //   block.3.weight/bias      pointwise conv (k=1, groups=1)
    let snake_a = read_snake(st, &format!("{prefix}.block.0.alpha"), channels)?;
    // Same-length output requires pad = (k-1)·d/2 = 3·d for k=7.
    let conv_a = read_conv1d(st, &format!("{prefix}.block.1"),
                             channels, channels, 7,
                             /*stride*/1, /*dilation*/dilation,
                             /*padding*/3 * dilation, /*groups*/channels)?;
    let snake_b = read_snake(st, &format!("{prefix}.block.2.alpha"), channels)?;
    let conv_b = read_conv1d(st, &format!("{prefix}.block.3"),
                             channels, channels, 1, 1, 1, 0, 1)?;
    Ok(ResidualUnit { snake_a, conv_a, snake_b, conv_b })
}

fn read_codebook(st: &SafeTensors, idx: usize, stride: usize) -> Result<Codebook, String> {
    let prefix = format!("quantizer.quantizers.{idx}");
    let embedding = read_f32_tensor(st, &format!("{prefix}.codebook.weight"))?;
    // PyTorch saves [4096, 8] for embedding — already in (n_codes, code_dim) row-major.
    debug_assert_eq!(embedding.len(), N_CODES * CODE_DIM);

    // out_proj.weight saved as [768, 8, 1] in PyTorch — squeeze the trailing 1.
    let mut op_w = read_f32_tensor(st, &format!("{prefix}.out_proj.weight"))?;
    let op_w_shape = read_shape(st, &format!("{prefix}.out_proj.weight"))?;
    debug_assert_eq!(op_w_shape, vec![DECODER_IN_CH, CODE_DIM, 1]);
    debug_assert_eq!(op_w.len(), DECODER_IN_CH * CODE_DIM);
    op_w.shrink_to_fit();

    let op_b = read_f32_tensor(st, &format!("{prefix}.out_proj.bias"))?;
    debug_assert_eq!(op_b.len(), DECODER_IN_CH);

    Ok(Codebook {
        embedding,
        n_codes: N_CODES,
        code_dim: CODE_DIM,
        out_proj_weight: op_w,
        out_proj_bias: op_b,
        stride,
    })
}

// ─── Low-level tensor helpers ───────────────────────────────────────────────

fn read_shape(st: &SafeTensors, name: &str) -> Result<Vec<usize>, String> {
    st.tensor(name)
        .map_err(|e| format!("missing tensor {name}: {e}"))
        .map(|t| t.shape().to_vec())
}

fn read_f32_tensor(st: &SafeTensors, name: &str) -> Result<Vec<f32>, String> {
    let t = st.tensor(name).map_err(|e| format!("missing {name}: {e}"))?;
    if t.dtype() != Dtype::F32 {
        return Err(format!("{name}: expected F32, got {:?}", t.dtype()));
    }
    let bytes = t.data();
    if bytes.len() % 4 != 0 {
        return Err(format!("{name}: byte len {} not multiple of 4", bytes.len()));
    }
    let mut out = vec![0.0f32; bytes.len() / 4];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(out)
}

fn read_conv1d(
    st: &SafeTensors, prefix: &str,
    in_channels: usize, out_channels: usize, kernel_size: usize,
    stride: usize, dilation: usize, padding: usize, groups: usize,
) -> Result<SnacConv1d, String> {
    let weight = read_f32_tensor(st, &format!("{prefix}.weight"))?;
    let bias = read_f32_tensor(st, &format!("{prefix}.bias"))?;
    let expected_w = out_channels * (in_channels / groups) * kernel_size;
    if weight.len() != expected_w {
        return Err(format!("{prefix}.weight: expected {expected_w}, got {}",
            weight.len()));
    }
    if bias.len() != out_channels {
        return Err(format!("{prefix}.bias: expected {out_channels}, got {}", bias.len()));
    }
    Ok(SnacConv1d {
        weight, bias,
        in_channels, out_channels, kernel_size,
        stride, dilation, padding, groups,
    })
}

fn read_convt1d(
    st: &SafeTensors, prefix: &str,
    in_channels: usize, out_channels: usize, kernel_size: usize,
    stride: usize, padding: usize, output_padding: usize,
) -> Result<SnacConvTranspose1d, String> {
    let weight = read_f32_tensor(st, &format!("{prefix}.weight"))?;
    let bias = read_f32_tensor(st, &format!("{prefix}.bias"))?;
    let expected_w = in_channels * out_channels * kernel_size;
    if weight.len() != expected_w {
        return Err(format!("{prefix}.weight: expected {expected_w}, got {}",
            weight.len()));
    }
    if bias.len() != out_channels {
        return Err(format!("{prefix}.bias: expected {out_channels}, got {}", bias.len()));
    }
    Ok(SnacConvTranspose1d {
        weight, bias,
        in_channels, out_channels, kernel_size,
        stride, padding, output_padding,
    })
}

fn read_snake(st: &SafeTensors, name: &str, channels: usize) -> Result<Snake1d, String> {
    let raw = read_f32_tensor(st, name)?;
    // PyTorch stores α as [1, C, 1]; we want flat [C].
    if raw.len() != channels {
        return Err(format!("{name}: expected {channels} alpha values, got {}", raw.len()));
    }
    Ok(Snake1d { alpha: raw, channels })
}

fn read_noise_block(st: &SafeTensors, name: &str, channels: usize) -> Result<NoiseBlock, String> {
    // PyTorch stores [C, C, 1] — squeeze trailing 1, keep row-major.
    let w = read_f32_tensor(st, name)?;
    if w.len() != channels * channels {
        return Err(format!("{name}: expected {} entries, got {}",
            channels * channels, w.len()));
    }
    Ok(NoiseBlock { gate_weight: w, channels })
}
