//! ONNX backend for isis.

use ort::session::Session;
use ort::value::Tensor;

use crate::backend::Backend;
use crate::episode::normalize;

type BoxErr = Box<dyn std::error::Error>;

/// ONNX inference backend.
pub struct OnnxBackend {
    backbone: Session,
    lm_head: Session,
    hidden_dim: usize,
    vocab_size: usize,
}

impl OnnxBackend {
    /// Load backbone and lm_head ONNX models.
    pub fn load(
        backbone_path: &str,
        lm_head_path: &str,
    ) -> Result<Self, BoxErr> {
        let backbone = Session::builder()?.commit_from_file(backbone_path)?;
        let lm_head = Session::builder()?.commit_from_file(lm_head_path)?;

        // Probe hidden_dim from backbone output shape: [batch, seq, hidden_dim]
        let hidden_dim = backbone.outputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied())
            .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            .unwrap_or_else(|| {
                eprintln!("WARNING: could not probe hidden_dim from ONNX, defaulting to 896");
                896
            });

        // Probe vocab_size from lm_head output shape: [batch, seq, vocab_size]
        let vocab_size = lm_head.outputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied())
            .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            .unwrap_or_else(|| {
                eprintln!("WARNING: could not probe vocab_size from ONNX, defaulting to 151936");
                151936
            });

        Ok(Self { backbone, lm_head, hidden_dim, vocab_size })
    }
}

impl Backend for OnnxBackend {
    fn get_key(&mut self, token_ids: &[i64]) -> Result<Vec<f32>, BoxErr> {
        let (hidden, _) = self.run_backbone(token_ids)?;
        let mut key = hidden.last().unwrap().clone();
        normalize(&mut key);
        Ok(key)
    }

    fn forward(&mut self, token_ids: &[i64]) -> Result<Vec<Vec<f32>>, BoxErr> {
        let (_, full) = self.run_backbone(token_ids)?;
        self.run_lm_head(&full)
    }

    fn run_backbone(&mut self, token_ids: &[i64]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), BoxErr> {
        let seq_len = token_ids.len();
        let shape = vec![1, seq_len];
        let input = Tensor::<i64>::from_array((shape.as_slice(), token_ids.to_vec()))?;

        let outputs = self.backbone.run(ort::inputs![input])?;

        let (_, h_data) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, f_data) = outputs[1].try_extract_tensor::<f32>()?;

        let d = self.hidden_dim;
        let mut hidden = Vec::with_capacity(seq_len);
        let mut full = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let offset = t * d;
            hidden.push(h_data[offset..offset + d].to_vec());
            full.push(f_data[offset..offset + d].to_vec());
        }
        Ok((hidden, full))
    }

    fn run_lm_head(&mut self, full_hidden: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, BoxErr> {
        let seq_len = full_hidden.len();
        let d = self.hidden_dim;
        let mut flat = Vec::with_capacity(seq_len * d);
        for h in full_hidden {
            flat.extend_from_slice(h);
        }
        let shape = vec![1, seq_len, d];
        let input = Tensor::<f32>::from_array((shape.as_slice(), flat))?;

        let outputs = self.lm_head.run(ort::inputs![input])?;
        let (_, l_data) = outputs[0].try_extract_tensor::<f32>()?;

        let v = self.vocab_size;
        let mut logits = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let offset = t * v;
            logits.push(l_data[offset..offset + v].to_vec());
        }
        Ok(logits)
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// Keep Inference as a type alias for backward compatibility.
pub type Inference = OnnxBackend;
