//! modgrad-gemma — Gemma-4 GGUF inference on top of the `modgrad-device` SDK.
//!
//! This is an APPLICATION crate, not part of the general SDK: it hardcodes
//! Gemma-4's architecture (1.0 attention scale, partial-rotary freq factors,
//! V=K on global layers, sandwich norms, final logit softcap, the harmony chat
//! format). The reusable primitives it stands on — the ROCm/HIP backend, the
//! K-quant matvec kernels, the GGUF loader — live in `modgrad-device`.

pub mod rocm_gemma;
// `LanguageModel` impl that serves Gemma over the model-agnostic modgrad-serve.
pub mod serve_model;
