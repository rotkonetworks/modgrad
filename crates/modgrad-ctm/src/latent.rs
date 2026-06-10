//! Reference `LatentThinker` implementor on the Path-C typed substrate.
//!
//! `LinearLatent` is the shallowest legitimate latent: one Linear applied
//! to every patch (`patch_dim → patch_dim`), batched across the patch
//! sequence. Its only job is to **lock the `LatentThinker` contract**
//! end-to-end (forward + backward, finite-difference gradchecked) on the
//! same `Tensor<D>` substrate where `CtmLatent` (A2) will live. The
//! production latent — the recurrent CTM, or the existing resident
//! transformer — swaps in behind the identical trait.

use modgrad_device::backend::tensor::{Device, Linear, Tensor};
use modgrad_device::backend::BackendError;
use modgrad_traits::{LatentThinker, SurpriseModel};

use crate::forward::{ctm_backward_typed, ctm_forward_typed_with_cache, CtmCacheTyped};
use crate::synapse::UNetGradsTyped;
use crate::weights::{CtmGradientsTyped, CtmWeights, CtmWeightsTyped};

/// One Linear over each patch, batched across the patch sequence. Proves
/// the seam; not the production latent.
pub struct LinearLatent<D: Device> {
    linear: Linear<D>,
    /// Accumulated weight/bias gradients (one `think_backward` per call,
    /// caller zeros between minibatches).
    pub d_w: Tensor<D>,
    pub d_b: Tensor<D>,
    patch_dim: usize,
}

impl<D: Device> LinearLatent<D> {
    pub fn from_host(weight: &[f32], bias: &[f32], patch_dim: usize) -> Result<Self, BackendError> {
        Ok(Self {
            linear: Linear::<D>::from_host(weight, bias, patch_dim, patch_dim)?,
            d_w: Tensor::<D>::zeros(patch_dim * patch_dim)?,
            d_b: Tensor::<D>::zeros(patch_dim)?,
            patch_dim,
        })
    }
}

impl<D: Device> LatentThinker for LinearLatent<D> {
    type Cache = Tensor<D>; // the input patches, retained for backward
    type Error = BackendError;

    fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    fn think_forward(
        &mut self,
        patches: &[f32],
        n_patches: usize,
    ) -> Result<(Vec<f32>, Tensor<D>), BackendError> {
        let x = Tensor::<D>::from_slice(patches)?;
        let mut y = Tensor::<D>::zeros(n_patches * self.patch_dim)?;
        self.linear.forward_batched(&x, &mut y, n_patches)?;
        Ok((y.to_vec()?, x))
    }

    fn think_backward(
        &mut self,
        d_thought: &[f32],
        cache: &Tensor<D>,
        n_patches: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let dy = Tensor::<D>::from_slice(d_thought)?;
        let mut dx = Tensor::<D>::zeros(n_patches * self.patch_dim)?;
        self.linear.backward_batched(&dy, cache, &mut self.d_w, &mut self.d_b, &mut dx, n_patches)?;
        Ok(dx.to_vec()?)
    }
}

/// The CTM as the BLT latent (A2 — the core move). Refines each patch by
/// *thinking* about it: the patch is the CTM's observation, the CTM runs
/// its adaptive-tick loop, and the final-tick prediction is the refined
/// ("thought") patch. Depth comes from thinking TIME, not stacked layers.
///
/// State (neuron trace/activated) resets per patch in this cut — within-
/// event working memory only. CROSS-patch/cross-event memory is the
/// hippocampus's job (episodic recall, A8), not raw neuron-state carry —
/// the brain-correct split. The CTM must be configured with observation
/// dim == out_dims == `patch_dim` (patch in, same-shape thought out).
pub struct CtmLatent<D: Device> {
    typed: CtmWeightsTyped<D>,
    start_activated: Vec<f32>,
    start_trace: Vec<f32>,
    pub grads: CtmGradientsTyped<D>,
    pub unet_grads: UNetGradsTyped<D>,
    patch_dim: usize,
}

impl<D: Device> CtmLatent<D> {
    /// `w` must be a CTM whose observation dim (raw_input_dim) and
    /// `config.out_dims` both equal `patch_dim`.
    pub fn from_weights(w: &CtmWeights, patch_dim: usize) -> Result<Self, BackendError> {
        let typed = CtmWeightsTyped::<D>::from_untyped(w)?;
        let grads = CtmGradientsTyped::<D>::zeros(&typed)?;
        let unet_grads = UNetGradsTyped::<D>::zeros(&typed.synapse)?;
        Ok(Self {
            typed,
            start_activated: w.start_activated.clone(),
            start_trace: w.start_trace.clone(),
            grads,
            unet_grads,
            patch_dim,
        })
    }
}

impl<D: Device> LatentThinker for CtmLatent<D> {
    /// One CTM forward cache per patch.
    type Cache = Vec<CtmCacheTyped<D>>;
    type Error = BackendError;

    fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    fn think_forward(
        &mut self,
        patches: &[f32],
        n_patches: usize,
    ) -> Result<(Vec<f32>, Self::Cache), BackendError> {
        let pd = self.patch_dim;
        let mut thought = vec![0.0f32; n_patches * pd];
        let mut caches = Vec::with_capacity(n_patches);
        for p in 0..n_patches {
            let mut act = self.start_activated.clone();
            let mut tr = self.start_trace.clone();
            let (out, cache) = ctm_forward_typed_with_cache::<D>(
                &self.typed, &mut act, &mut tr, &patches[p * pd..(p + 1) * pd],
            )?;
            let last = out.predictions.last().expect("at least one tick");
            thought[p * pd..(p + 1) * pd].copy_from_slice(last);
            caches.push(cache);
        }
        Ok((thought, caches))
    }

    fn think_backward(
        &mut self,
        d_thought: &[f32],
        cache: &Self::Cache,
        n_patches: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let pd = self.patch_dim;
        let k = self.typed.config.iterations;
        let mut d_patches = vec![0.0f32; n_patches * pd];
        for p in 0..n_patches {
            // Loss touches only the final-tick prediction (the thought).
            let mut d_preds: Vec<Vec<f32>> = vec![vec![0.0f32; pd]; k];
            d_preds[k - 1].copy_from_slice(&d_thought[p * pd..(p + 1) * pd]);
            let mut d_obs = Tensor::<D>::zeros(pd)?;
            ctm_backward_typed::<D>(
                &self.typed, &cache[p], &d_preds,
                &mut self.grads, &mut self.unet_grads, &mut d_obs,
            )?;
            d_patches[p * pd..(p + 1) * pd].copy_from_slice(&d_obs.to_vec()?);
        }
        Ok(d_patches)
    }
}

/// Reference `SurpriseModel` (A4): predicts the next patch from the current
/// patch via one Linear (`patch_dim → patch_dim`). Locks the SurpriseModel
/// contract; the production predictor (a CTM / small forward model that also
/// serves as the cerebellum, A7) swaps in behind the same trait.
pub struct LinearSurprise<D: Device> {
    linear: Linear<D>,
    pub d_w: Tensor<D>,
    pub d_b: Tensor<D>,
    patch_dim: usize,
}

impl<D: Device> LinearSurprise<D> {
    pub fn from_host(weight: &[f32], bias: &[f32], patch_dim: usize) -> Result<Self, BackendError> {
        Ok(Self {
            linear: Linear::<D>::from_host(weight, bias, patch_dim, patch_dim)?,
            d_w: Tensor::<D>::zeros(patch_dim * patch_dim)?,
            d_b: Tensor::<D>::zeros(patch_dim)?,
            patch_dim,
        })
    }
}

impl<D: Device> SurpriseModel for LinearSurprise<D> {
    type Cache = Tensor<D>; // the ctx input, for backward
    type Error = BackendError;

    fn patch_dim(&self) -> usize {
        self.patch_dim
    }

    fn predict_next(&mut self, ctx: &[f32]) -> Result<(Vec<f32>, Tensor<D>), BackendError> {
        let x = Tensor::<D>::from_slice(ctx)?;
        let y = self.linear.forward(&x)?;
        Ok((y.to_vec()?, x))
    }

    fn predict_backward(&mut self, d_pred: &[f32], cache: &Tensor<D>) -> Result<Vec<f32>, BackendError> {
        let dy = Tensor::<D>::from_slice(d_pred)?;
        let mut dx = Tensor::<D>::zeros(self.patch_dim)?;
        self.linear.backward(&dy, cache, &mut self.d_w, &mut self.d_b, &mut dx)?;
        Ok(dx.to_vec()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CtmConfig, ExitStrategy};
    use modgrad_device::backend::tensor::Cpu;

    /// Lock the LatentThinker contract: think_forward/think_backward
    /// finite-difference-gradcheck (loss = Σ thought, d_thought = ones ⇒
    /// dL/dw = analytic d_w).
    #[test]
    fn linear_latent_contract_fd_gradcheck() {
        let pd = 4usize;
        let n_patches = 5usize;
        let w: Vec<f32> = (0..pd * pd).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let b: Vec<f32> = (0..pd).map(|i| (i as f32 - 2.0) * 0.05).collect();
        let patches: Vec<f32> = (0..n_patches * pd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2).collect();

        // Analytic grads at base point.
        let mut lat = LinearLatent::<Cpu>::from_host(&w, &b, pd).unwrap();
        let (_thought, cache) = lat.think_forward(&patches, n_patches).unwrap();
        let d_thought = vec![1.0f32; n_patches * pd];
        let d_patches = lat.think_backward(&d_thought, &cache, n_patches).unwrap();
        assert_eq!(d_patches.len(), n_patches * pd);
        let dw = lat.d_w.to_vec().unwrap();

        let loss_of = |wp: &[f32]| -> f32 {
            let mut l = LinearLatent::<Cpu>::from_host(wp, &b, pd).unwrap();
            l.think_forward(&patches, n_patches).unwrap().0.iter().sum()
        };
        const EPS: f32 = 1e-3;
        for idx in [0usize, pd * pd / 2, pd * pd - 1] {
            let mut wp = w.clone();
            wp[idx] += EPS;
            let lp = loss_of(&wp);
            let mut wm = w.clone();
            wm[idx] -= EPS;
            let lm = loss_of(&wm);
            let fd = (lp - lm) / (2.0 * EPS);
            let a = dw[idx];
            let abs_diff = (fd - a).abs();
            let rel = abs_diff / fd.abs().max(a.abs()).max(1e-4);
            // Near-zero grads sit at the FD floor (~|L|·ulp/2EPS ~1e-4).
            assert!(rel < 1e-3 || abs_diff < 3e-4,
                "d_w[{idx}]: analytic={a} fd={fd} rel={rel} abs={abs_diff}");
        }
    }

    /// A2 core move: the CTM-as-latent contract. think_forward streams
    /// patches through the CTM's tick loop; think_backward must match FD
    /// (loss = Σ thought, d_thought = ones) on a CTM weight (output_proj).
    #[test]
    fn ctm_latent_contract_fd_gradcheck() {
        let pd = 4usize;
        let cfg = CtmConfig {
            iterations: 2, d_model: 4, d_input: 8, heads: 2,
            n_synch_out: 4, n_synch_action: 4, synapse_depth: 2,
            memory_length: 4, deep_nlms: false, memory_hidden_dims: 0,
            out_dims: pd, n_random_pairing_self: 0, min_width: 2,
            exit_strategy: ExitStrategy::None, collect_trajectories: false,
        };
        let w = CtmWeights::new(cfg.clone(), pd); // raw_input_dim = pd
        let n_patches = 3usize;
        let patches: Vec<f32> = (0..n_patches * pd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2).collect();

        // Analytic.
        let mut lat = CtmLatent::<Cpu>::from_weights(&w, pd).unwrap();
        let (_th, cache) = lat.think_forward(&patches, n_patches).unwrap();
        let d_thought = vec![1.0f32; n_patches * pd];
        let d_patches = lat.think_backward(&d_thought, &cache, n_patches).unwrap();
        assert_eq!(d_patches.len(), n_patches * pd);
        let dw = lat.grads.out_proj_w.to_vec().unwrap();

        let loss_of = |wp: &CtmWeights| -> f32 {
            let mut l = CtmLatent::<Cpu>::from_weights(wp, pd).unwrap();
            l.think_forward(&patches, n_patches).unwrap().0.iter().sum()
        };
        const EPS: f32 = 1e-3;
        let n = dw.len();
        for idx in [0usize, n / 2, n - 1] {
            let mut wp = w.clone();
            let orig = wp.output_proj.weight[idx];
            wp.output_proj.weight[idx] = orig + EPS;
            let lp = loss_of(&wp);
            wp.output_proj.weight[idx] = orig - EPS;
            let lm = loss_of(&wp);
            let fd = (lp - lm) / (2.0 * EPS);
            let a = dw[idx];
            let abs_diff = (fd - a).abs();
            let rel = abs_diff / fd.abs().max(a.abs()).max(1e-4);
            assert!(rel < 2e-2 || abs_diff < 3e-4,
                "out_proj_w[{idx}]: analytic={a} fd={fd} rel={rel} abs={abs_diff}");
        }
    }

    /// A8 — the property the whole architecture rests on: episodic recall
    /// is CONTENT-addressable and does not die when a window slides. Store
    /// 100 episodes (far past any fixed context window), then recall an
    /// EARLY one (#5) by content — despite 95 episodes stored after it. A
    /// sliding context window would have dropped #5; the hippocampus
    /// retrieves it by similarity. This is continuous episodic memory
    /// without the context-window death.
    #[test]
    fn episodic_recall_is_content_addressable_not_recency() {
        use modgrad_memory::episodic::{retrieve, store, EpisodicConfig, EpisodicMemory};
        let d = 4usize;
        let cfg = EpisodicConfig {
            capacity: 256,
            max_ticks: 2,
            d_model: d,
            min_ticks_for_storage: 0,
            min_surprise: 0.0,
            retrieval_threshold: 0.0,
            ..Default::default()
        };
        let mut mem = EpisodicMemory::new(cfg);

        let n_ep = 100usize;
        let mut patterns: Vec<Vec<f32>> = Vec::with_capacity(n_ep);
        for i in 0..n_ep {
            let v: Vec<f32> = (0..d).map(|j| (((i * 7 + j * 13) % 17) as f32) - 8.0).collect();
            patterns.push(v.clone());
            let traj: Vec<f32> = v.iter().chain(v.iter()).copied().collect(); // [2 × d]
            let cert = [[0.5f32, 0.5]; 2];
            let (m, _) = store(mem, &traj, &cert, &[], 2, 1.0);
            mem = m;
        }
        assert!(mem.len() > 0);

        // Recall the OLD episode #5 by content.
        let r = retrieve(&mem, &patterns[5]);
        let recalled = &r.blended_final_state;
        assert_eq!(recalled.len(), d);
        let dist = |p: &[f32]| -> f32 {
            recalled.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum()
        };
        // The recall is dominated by the content-matched episode #5, not by
        // recency (#50, #99 were stored much later).
        assert!(dist(&patterns[5]) < dist(&patterns[50]),
            "recall not content-addressed to #5 vs #50");
        assert!(dist(&patterns[5]) < dist(&patterns[99]),
            "recall not content-addressed to #5 vs #99 (recency would win a window)");
    }

    /// A4: the SurpriseModel contract. predict_next/predict_backward
    /// FD-gradcheck under the ½‖predicted−actual‖² surprise loss.
    #[test]
    fn linear_surprise_contract_fd_gradcheck() {
        let pd = 4usize;
        let w: Vec<f32> = (0..pd * pd).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let b: Vec<f32> = (0..pd).map(|i| (i as f32 - 2.0) * 0.05).collect();
        let ctx: Vec<f32> = (0..pd).map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.3).collect();
        let actual: Vec<f32> = (0..pd).map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.2).collect();

        // Analytic: loss = surprise(predict(ctx), actual); d_pred = pred - actual.
        let mut sm = LinearSurprise::<Cpu>::from_host(&w, &b, pd).unwrap();
        let (pred, cache) = sm.predict_next(&ctx).unwrap();
        let d_pred = sm.surprise_grad(&pred, &actual);
        let _d_ctx = sm.predict_backward(&d_pred, &cache).unwrap();
        let dw = sm.d_w.to_vec().unwrap();

        let loss_of = |wp: &[f32]| -> f32 {
            let mut s = LinearSurprise::<Cpu>::from_host(wp, &b, pd).unwrap();
            let (p, _) = s.predict_next(&ctx).unwrap();
            s.surprise(&p, &actual)
        };
        const EPS: f32 = 1e-3;
        for idx in [0usize, pd * pd / 2, pd * pd - 1] {
            let mut wp = w.clone();
            wp[idx] += EPS;
            let lp = loss_of(&wp);
            let mut wm = w.clone();
            wm[idx] -= EPS;
            let lm = loss_of(&wm);
            let fd = (lp - lm) / (2.0 * EPS);
            let a = dw[idx];
            let abs_diff = (fd - a).abs();
            let rel = abs_diff / fd.abs().max(a.abs()).max(1e-4);
            assert!(rel < 1e-3 || abs_diff < 3e-4,
                "d_w[{idx}]: analytic={a} fd={fd} rel={rel} abs={abs_diff}");
        }
    }
}
