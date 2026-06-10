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
use modgrad_traits::LatentThinker;

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
