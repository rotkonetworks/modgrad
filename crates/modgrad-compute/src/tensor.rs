//! Tensor: the compute primitive for forward and backward passes.
//!
//! A Tensor is a 2D f32 array with an optional gradient buffer.
//! All brain computation flows through tensors:
//!   observation → [synapse matmul] → [SiLU] → [NLM] → ... → loss
//!   loss → [backward chain] → gradients → weight updates
//!
//! Design: Henry de Valence style.
//!   - Zero-copy where possible (slices into existing buffers)
//!   - GPU-friendly (contiguous row-major, aligned)
//!   - No autograd graph — backward is explicit (we know the graph statically)

/// 2D tensor: [rows × cols], row-major, f32.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn from_slice(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data: data.to_vec(), rows, cols }
    }

    /// View a row as a slice.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn len(&self) -> usize { self.data.len() }
}

// ─── Forward ops ────────────────────────────────────────────

/// Y = X @ W^T + bias.  X:[batch × in], W:[out × in], bias:[out] → Y:[batch × out]
pub fn linear_forward(x: &Tensor, w: &Tensor, bias: &[f32]) -> Tensor {
    let batch = x.rows;
    let in_dim = x.cols;
    let out_dim = w.rows;
    assert_eq!(w.cols, in_dim);
    assert_eq!(bias.len(), out_dim);

    let mut y = Tensor::zeros(batch, out_dim);
    for b in 0..batch {
        for o in 0..out_dim {
            let mut sum = bias[o];
            let w_row = &w.data[o * in_dim..(o + 1) * in_dim];
            let x_row = &x.data[b * in_dim..(b + 1) * in_dim];
            for k in 0..in_dim {
                sum += w_row[k] * x_row[k];
            }
            y.data[b * out_dim + o] = sum;
        }
    }
    y
}

/// SiLU: y = x * sigmoid(x)
pub fn silu_forward(x: &Tensor) -> Tensor {
    let mut y = Tensor::zeros(x.rows, x.cols);
    for (yi, &xi) in y.data.iter_mut().zip(x.data.iter()) {
        let sig = 1.0 / (1.0 + (-xi).exp());
        *yi = xi * sig;
    }
    y
}

/// Cat along cols: [batch × a] + [batch × b] → [batch × (a+b)]
pub fn cat_cols(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.rows, b.rows);
    let batch = a.rows;
    let cols = a.cols + b.cols;
    let mut out = Tensor::zeros(batch, cols);
    for r in 0..batch {
        out.data[r * cols..r * cols + a.cols].copy_from_slice(a.row(r));
        out.data[r * cols + a.cols..r * cols + cols].copy_from_slice(b.row(r));
    }
    out
}

/// Residual add: y = a + b (elementwise)
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.data.len(), b.data.len());
    let mut y = Tensor::zeros(a.rows, a.cols);
    for (yi, (&ai, &bi)) in y.data.iter_mut().zip(a.data.iter().zip(b.data.iter())) {
        *yi = ai + bi;
    }
    y
}

/// Cross-entropy loss for binary/multi-class classification.
/// logits: [batch × n_classes], targets: [batch] (class indices)
/// Returns (scalar loss, d_logits gradient)
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> (f32, Tensor) {
    let batch = logits.rows;
    let n_cls = logits.cols;
    let mut total_loss = 0.0f32;
    let mut d_logits = Tensor::zeros(batch, n_cls);

    for b in 0..batch {
        let row = logits.row(b);
        let max_l: f32 = row.iter().fold(f32::MIN, |a, &b| a.max(b));
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let loss = (sum.ln() + max_l) - row[targets[b]];
        total_loss += loss;

        // Gradient: softmax - one_hot
        let d_row = d_logits.row_mut(b);
        for c in 0..n_cls {
            d_row[c] = exps[c] / sum;
        }
        d_row[targets[b]] -= 1.0;
    }

    (total_loss / batch as f32, d_logits)
}

// ─── Backward ops ───────────────────────────────────────────

/// Backward for Y = X @ W^T + bias.
/// Returns (dX, dW, dbias).
pub fn linear_backward(dy: &Tensor, x: &Tensor, w: &Tensor) -> (Tensor, Tensor, Vec<f32>) {
    let batch = dy.rows;
    let out_dim = dy.cols;
    let in_dim = x.cols;

    // dX = dY @ W  (not transposed — W is [out × in], dY is [batch × out])
    let mut dx = Tensor::zeros(batch, in_dim);
    for b in 0..batch {
        for i in 0..in_dim {
            let mut sum = 0.0;
            for o in 0..out_dim {
                sum += dy.data[b * out_dim + o] * w.data[o * in_dim + i];
            }
            dx.data[b * in_dim + i] = sum;
        }
    }

    // dW = dY^T @ X  → [out × in]
    let mut dw = Tensor::zeros(out_dim, in_dim);
    for o in 0..out_dim {
        for i in 0..in_dim {
            let mut sum = 0.0;
            for b in 0..batch {
                sum += dy.data[b * out_dim + o] * x.data[b * in_dim + i];
            }
            dw.data[o * in_dim + i] = sum / batch as f32;
        }
    }

    // dbias = mean(dY, axis=0)
    let mut dbias = vec![0.0f32; out_dim];
    for b in 0..batch {
        for o in 0..out_dim {
            dbias[o] += dy.data[b * out_dim + o];
        }
    }
    for o in 0..out_dim { dbias[o] /= batch as f32; }

    (dx, dw, dbias)
}

/// Backward for SiLU: dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
pub fn silu_backward(dy: &Tensor, x: &Tensor) -> Tensor {
    let mut dx = Tensor::zeros(x.rows, x.cols);
    for (i, (&dyi, &xi)) in dy.data.iter().zip(x.data.iter()).enumerate() {
        let sig = 1.0 / (1.0 + (-xi).exp());
        dx.data[i] = dyi * (sig + xi * sig * (1.0 - sig));
    }
    dx
}

/// Backward for cat_cols: split gradient back into two parts
pub fn cat_cols_backward(dy: &Tensor, a_cols: usize) -> (Tensor, Tensor) {
    let b_cols = dy.cols - a_cols;
    let batch = dy.rows;
    let mut da = Tensor::zeros(batch, a_cols);
    let mut db = Tensor::zeros(batch, b_cols);
    for r in 0..batch {
        da.data[r * a_cols..(r + 1) * a_cols]
            .copy_from_slice(&dy.data[r * dy.cols..r * dy.cols + a_cols]);
        db.data[r * b_cols..(r + 1) * b_cols]
            .copy_from_slice(&dy.data[r * dy.cols + a_cols..(r + 1) * dy.cols]);
    }
    (da, db)
}

// ─── Adam optimizer ─────────────────────────────────────────

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: usize,
    m: Vec<f32>,  // first moment
    v: Vec<f32>,  // second moment
}

impl Adam {
    pub fn new(n_params: usize, lr: f32) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }

    /// Update params in-place given gradients.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len().min(grads.len()) {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── Energy-Based CTM ───────────────────────────────────────

/// Energy-Based CTM: learns to verify input-prediction compatibility.
/// Architecture: synapse + NLM tick loop + energy head.
///
/// Training: start from random prediction, optimize via energy gradient
/// descent through the tick loop, loss on final prediction vs target.
///
/// This is the EBT approach (arXiv:2507.02092): the tick loop IS thinking,
/// energy minimization IS reasoning, and the energy scalar IS verification.
///
/// Also supports standard forward mode (MiniCTM-compatible) for comparison.
pub struct MiniCtm {
    pub d_in: usize,
    pub d_hid: usize,
    pub n_ticks: usize,
    pub n_classes: usize,
    // Synapse: mixes input with hidden state
    pub syn_w: Tensor,      // [d_hid × (d_in + d_hid)]
    pub syn_b: Vec<f32>,    // [d_hid]
    // NLM: recurrent processing within tick
    pub nlm_w: Tensor,      // [d_hid × d_hid]
    pub nlm_b: Vec<f32>,    // [d_hid]
    // Output readout
    pub out_w: Tensor,      // [n_classes × d_hid]
    pub out_b: Vec<f32>,    // [n_classes]
    // Energy head: maps (context_hidden, candidate_prediction) → scalar energy
    // When present, enables EBT mode: predictions refined by energy minimization
    pub energy_w: Tensor,   // [1 × (d_hid + n_classes)]
    pub energy_b: Vec<f32>, // [1]
}

impl MiniCtm {
    pub fn new(d_in: usize, d_hid: usize, n_ticks: usize, n_classes: usize) -> Self {
        use super::neuron::SimpleRng;
        let mut rng = SimpleRng::new(42);
        let init = |rows: usize, cols: usize, rng: &mut SimpleRng| -> (Tensor, Vec<f32>) {
            let scale = (2.0 / cols as f32).sqrt();
            let w = Tensor {
                data: (0..rows * cols).map(|_| rng.next_normal() * scale).collect(),
                rows, cols,
            };
            let b = vec![0.0; rows];
            (w, b)
        };

        let (syn_w, syn_b) = init(d_hid, d_in + d_hid, &mut rng);
        let (nlm_w, nlm_b) = init(d_hid, d_hid, &mut rng);
        let (out_w, out_b) = init(n_classes, d_hid, &mut rng);
        let (energy_w, energy_b) = init(1, d_hid + n_classes, &mut rng);

        Self { d_in, d_hid, n_ticks, n_classes,
               syn_w, syn_b, nlm_w, nlm_b, out_w, out_b,
               energy_w, energy_b }
    }

    /// Run the tick loop: input → hidden state after T ticks.
    pub fn tick_loop(&self, x: &Tensor) -> Tensor {
        let batch = x.rows;
        let mut h = Tensor::zeros(batch, self.d_hid);
        for _ in 0..self.n_ticks {
            let xh = cat_cols(x, &h);
            let syn_out = linear_forward(&xh, &self.syn_w, &self.syn_b);
            let syn_act = silu_forward(&syn_out);
            h = add(&h, &syn_act);
            let nlm_out = linear_forward(&h, &self.nlm_w, &self.nlm_b);
            h = silu_forward(&nlm_out);
        }
        h
    }

    /// Standard forward: returns logits [batch × n_classes]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.tick_loop(x);
        linear_forward(&h, &self.out_w, &self.out_b)
    }

    /// Compute energy: how compatible is this (context, prediction) pair?
    /// Lower energy = higher compatibility = better prediction.
    /// context_h: [batch × d_hid] from tick_loop
    /// prediction: [batch × n_classes] candidate prediction (e.g. softmax distribution)
    pub fn energy(&self, context_h: &Tensor, prediction: &Tensor) -> Vec<f32> {
        let batch = context_h.rows;
        let hp = cat_cols(context_h, prediction);
        let e = linear_forward(&hp, &self.energy_w, &self.energy_b);
        // Return scalar energy per sample
        (0..batch).map(|b| e.data[b]).collect()
    }

    /// EBT forward: start from random prediction, refine by energy minimization.
    /// This IS System 2 Thinking — the tick loop runs, then the prediction is
    /// iteratively refined by descending the energy landscape.
    /// Returns (final_prediction, energies_per_step)
    pub fn forward_ebt(&self, x: &Tensor, n_refine_steps: usize, step_size: f32) -> (Tensor, Vec<f32>) {
        let batch = x.rows;
        let n_cls = self.n_classes;

        // Run tick loop to get context representation
        let h = self.tick_loop(x);

        // Initialize prediction as random (uniform over classes)
        let mut pred = Tensor::zeros(batch, n_cls);
        {
            use super::neuron::SimpleRng;
            let mut rng = SimpleRng::new(999);
            for v in &mut pred.data { *v = rng.next_normal() * 0.1; }
        }

        let mut energies = Vec::with_capacity(n_refine_steps);

        for _step in 0..n_refine_steps {
            // Compute energy for current prediction
            let e_vals = self.energy(&h, &pred);
            let avg_energy: f32 = e_vals.iter().sum::<f32>() / batch as f32;
            energies.push(avg_energy);

            // Compute gradient of energy w.r.t. prediction (finite differences)
            // For each element of pred, nudge it and measure energy change
            let eps = 0.001;
            let mut grad = Tensor::zeros(batch, n_cls);
            for b in 0..batch {
                for c in 0..n_cls {
                    let idx = b * n_cls + c;
                    let orig = pred.data[idx];

                    pred.data[idx] = orig + eps;
                    let e_plus = self.energy(&h, &pred);

                    pred.data[idx] = orig - eps;
                    let e_minus = self.energy(&h, &pred);

                    pred.data[idx] = orig;
                    grad.data[idx] = (e_plus[b] - e_minus[b]) / (2.0 * eps);
                }
            }

            // Gradient descent on energy landscape
            for i in 0..pred.data.len() {
                pred.data[i] -= step_size * grad.data[i];
            }
        }

        (pred, energies)
    }

    /// Forward + backward: returns (loss, gradients for all weights)
    pub fn forward_backward(&self, x: &Tensor, targets: &[usize]) -> (f32, MiniCtmGrads) {
        let batch = x.rows;

        // Forward pass — save intermediates for backward
        let mut h_history: Vec<Tensor> = Vec::with_capacity(self.n_ticks + 1);
        let mut syn_pre_act: Vec<Tensor> = Vec::with_capacity(self.n_ticks);
        let mut nlm_pre_act: Vec<Tensor> = Vec::with_capacity(self.n_ticks);
        let mut xh_history: Vec<Tensor> = Vec::with_capacity(self.n_ticks);
        let mut h_pre_residual: Vec<Tensor> = Vec::with_capacity(self.n_ticks);

        let mut h = Tensor::zeros(batch, self.d_hid);
        h_history.push(h.clone());

        for _ in 0..self.n_ticks {
            let xh = cat_cols(x, &h);
            let syn_out = linear_forward(&xh, &self.syn_w, &self.syn_b);
            let syn_act = silu_forward(&syn_out);
            let h_res = h.clone();
            h = add(&h, &syn_act);  // residual
            let nlm_out = linear_forward(&h, &self.nlm_w, &self.nlm_b);

            xh_history.push(xh);
            syn_pre_act.push(syn_out);
            h_pre_residual.push(h_res);
            nlm_pre_act.push(nlm_out.clone());

            h = silu_forward(&nlm_out);
            h_history.push(h.clone());
        }

        let logits = linear_forward(&h, &self.out_w, &self.out_b);
        let (loss, d_logits) = cross_entropy(&logits, targets);

        // Backward pass
        let mut grads = MiniCtmGrads::zeros(self);

        // d_logits → out layer
        let (mut dh, d_out_w, d_out_b) = linear_backward(&d_logits, &h, &self.out_w);
        grads.out_w = d_out_w;
        grads.out_b = d_out_b;

        // Backprop through ticks (reverse order)
        for t in (0..self.n_ticks).rev() {
            // backward through silu(nlm_out) → h
            dh = silu_backward(&dh, &nlm_pre_act[t]);

            // backward through nlm linear
            let (dh_nlm, _d_nlm_w, _d_nlm_b) = linear_backward(&dh, &h_history[t + 1], &self.nlm_w);
            // Wait — h input to nlm is AFTER residual add, before silu.
            // Actually: nlm_out = linear(h_after_residual), h_final = silu(nlm_out)
            // So dh_nlm goes back to h_after_residual
            let _ = dh_nlm; // we need to recompute properly

            // Let me redo this correctly:
            // h_after_residual = h_prev + syn_act
            // nlm_out = linear(h_after_residual, nlm_w)
            // h_next = silu(nlm_out)

            // dh is gradient of loss w.r.t. h_next
            // dh → silu_backward → d_nlm_out
            // d_nlm_out → linear_backward(d_nlm_out, h_after_residual, nlm_w) → (d_h_after_res, d_nlm_w, d_nlm_b)
            // d_h_after_res → (d_h_prev + d_syn_act)  [residual splits gradient]
            // d_syn_act → silu_backward → d_syn_out
            // d_syn_out → linear_backward(d_syn_out, xh, syn_w) → (d_xh, d_syn_w, d_syn_b)
            // d_xh → split → (d_x [discard], d_h_prev_from_cat)
            // d_h_prev = d_h_prev_from_residual + d_h_prev_from_cat

            // Redo properly:
            let h_after_res = &{
                let mut har = h_pre_residual[t].clone();
                let syn_act = silu_forward(&syn_pre_act[t]);
                for (a, b) in har.data.iter_mut().zip(syn_act.data.iter()) { *a += b; }
                har
            };

            let (d_h_after_res, d_nlm_w_t, d_nlm_b_t) = linear_backward(&silu_backward(&dh, &nlm_pre_act[t]), h_after_res, &self.nlm_w);

            // Accumulate nlm gradients
            for (g, d) in grads.nlm_w.data.iter_mut().zip(d_nlm_w_t.data.iter()) { *g += d; }
            for (g, d) in grads.nlm_b.iter_mut().zip(d_nlm_b_t.iter()) { *g += d; }

            // Residual: d_h_after_res → d_h_prev + d_syn_act
            let d_syn_act = d_h_after_res.clone(); // same gradient flows to both branches
            let d_syn_out = silu_backward(&d_syn_act, &syn_pre_act[t]);
            let (d_xh, d_syn_w_t, d_syn_b_t) = linear_backward(&d_syn_out, &xh_history[t], &self.syn_w);

            // Accumulate syn gradients
            for (g, d) in grads.syn_w.data.iter_mut().zip(d_syn_w_t.data.iter()) { *g += d; }
            for (g, d) in grads.syn_b.iter_mut().zip(d_syn_b_t.iter()) { *g += d; }

            // Split d_xh → d_x (discard) + d_h_prev
            let (_, d_h_from_cat) = cat_cols_backward(&d_xh, self.d_in);

            // d_h for next iteration = d_h from residual + d_h from cat
            dh = Tensor::zeros(batch, self.d_hid);
            for (i, (&a, &b)) in d_h_after_res.data.iter().zip(d_h_from_cat.data.iter()).enumerate() {
                dh.data[i] = a + b;
            }
        }

        // Average gradients over ticks
        let nt = self.n_ticks as f32;
        for g in grads.syn_w.data.iter_mut() { *g /= nt; }
        for g in grads.syn_b.iter_mut() { *g /= nt; }
        for g in grads.nlm_w.data.iter_mut() { *g /= nt; }
        for g in grads.nlm_b.iter_mut() { *g /= nt; }

        (loss, grads)
    }

    pub fn param_count(&self) -> usize {
        self.syn_w.len() + self.syn_b.len() +
        self.nlm_w.len() + self.nlm_b.len() +
        self.out_w.len() + self.out_b.len() +
        self.energy_w.len() + self.energy_b.len()
    }

    pub fn apply_grads(&mut self, grads: &MiniCtmGrads, opt: &mut Adam) {
        let mut all_params: Vec<f32> = Vec::with_capacity(self.param_count());
        all_params.extend_from_slice(&self.syn_w.data);
        all_params.extend_from_slice(&self.syn_b);
        all_params.extend_from_slice(&self.nlm_w.data);
        all_params.extend_from_slice(&self.nlm_b);
        all_params.extend_from_slice(&self.out_w.data);
        all_params.extend_from_slice(&self.out_b);
        all_params.extend_from_slice(&self.energy_w.data);
        all_params.extend_from_slice(&self.energy_b);

        let mut all_grads: Vec<f32> = Vec::with_capacity(self.param_count());
        all_grads.extend_from_slice(&grads.syn_w.data);
        all_grads.extend_from_slice(&grads.syn_b);
        all_grads.extend_from_slice(&grads.nlm_w.data);
        all_grads.extend_from_slice(&grads.nlm_b);
        all_grads.extend_from_slice(&grads.out_w.data);
        all_grads.extend_from_slice(&grads.out_b);
        all_grads.extend_from_slice(&grads.energy_w.data);
        all_grads.extend_from_slice(&grads.energy_b);

        opt.step(&mut all_params, &all_grads);

        let mut off = 0;
        macro_rules! copy_back {
            ($field:expr) => {
                let n = $field.len();
                $field.copy_from_slice(&all_params[off..off + n]);
                off += n;
            };
        }
        copy_back!(self.syn_w.data);
        copy_back!(self.syn_b);
        copy_back!(self.nlm_w.data);
        copy_back!(self.nlm_b);
        copy_back!(self.out_w.data);
        copy_back!(self.out_b);
        copy_back!(self.energy_w.data);
        copy_back!(self.energy_b);
    }
}

pub struct MiniCtmGrads {
    pub syn_w: Tensor,
    pub syn_b: Vec<f32>,
    pub nlm_w: Tensor,
    pub nlm_b: Vec<f32>,
    pub out_w: Tensor,
    pub out_b: Vec<f32>,
    pub energy_w: Tensor,
    pub energy_b: Vec<f32>,
}

impl MiniCtmGrads {
    pub fn zeros(model: &MiniCtm) -> Self {
        Self {
            syn_w: Tensor::zeros(model.syn_w.rows, model.syn_w.cols),
            syn_b: vec![0.0; model.syn_b.len()],
            nlm_w: Tensor::zeros(model.nlm_w.rows, model.nlm_w.cols),
            nlm_b: vec![0.0; model.nlm_b.len()],
            out_w: Tensor::zeros(model.out_w.rows, model.out_w.cols),
            out_b: vec![0.0; model.out_b.len()],
            energy_w: Tensor::zeros(model.energy_w.rows, model.energy_w.cols),
            energy_b: vec![0.0; model.energy_b.len()],
        }
    }
}
