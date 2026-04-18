//! Training loop abstraction — composes Brain + Loss + Optimizer + Scheduler
//! + Checkpoint + DataStream into a single, configurable training loop.
//!
//! Each part is swappable. The Trainer doesn't know about Sakana, multiregion,
//! or any specific architecture — it operates through the Brain trait.

use modgrad_traits::{Brain, LossFn};
use super::optim::Scheduler;
use super::checkpoint::CheckpointManager;

/// Result of one training step.
#[derive(Debug, Clone)]
pub struct StepReport {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub is_best: bool,
}

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub steps_completed: usize,
    pub final_loss: f32,
    pub best_loss: f32,
    pub best_step: usize,
    pub elapsed_secs: f64,
}

/// Configuration for training.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub total_steps: usize,
    pub micro_batch: usize,
    pub accum_steps: usize,
    pub log_every: usize,
    pub save_every: usize,
    pub token_dim: usize,
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            total_steps: 10_000,
            micro_batch: 8,
            accum_steps: 1,
            log_every: 100,
            save_every: 1000,
            token_dim: 128,
            grad_clip: 5.0,
        }
    }
}

/// Hook called after each training step. The training loop doesn't
/// know what the hook does — dream/sleep, logging, checkpointing,
/// curriculum adjustment — all are implementations of this trait.
///
/// The hook receives the current weights (mutable) and step metadata.
/// It can modify weights (e.g., dream phase applies its own gradient).
pub trait StepHook<W> {
    fn after_step(&mut self, weights: &mut W, step: usize, lr: f32);
}

/// No-op hook. Use when no post-step processing is needed.
impl<W> StepHook<W> for () {
    fn after_step(&mut self, _weights: &mut W, _step: usize, _lr: f32) {}
}

/// Callback for logging.
pub trait Logger {
    fn log_step(&mut self, report: &StepReport);
    fn log_message(&mut self, msg: &str);
}

/// Default logger: stderr.
pub struct StderrLogger;

impl Logger for StderrLogger {
    fn log_step(&mut self, r: &StepReport) {
        eprintln!("  step {:6}: loss={:.4} lr={:.6}{}",
            r.step, r.loss, r.lr,
            if r.is_best { " *best*" } else { "" });
    }
    fn log_message(&mut self, msg: &str) { eprintln!("  {}", msg); }
}

// ═══════════════════════════════════════════════════════════════
// UNIFIED SAMPLE PROVIDER
// ═══════════════════════════════════════════════════════════════

/// A training sample: input + target.
/// Generic over Input (what the Brain consumes) and Target (what the Loss needs).
#[derive(Debug, Clone)]
pub struct Sample<I, T> {
    pub input: I,
    pub target: T,
}

/// Provides training samples. Generic over input and target types.
pub trait SampleProvider<I, T> {
    fn next_sample(&mut self) -> Option<Sample<I, T>>;
    fn reset(&mut self);
}

/// Adapter: wraps a DataStream into a SampleProvider<usize> for
/// classification (next-code prediction).
pub struct ClassificationAdapter<D> {
    pub stream: D,
    pub embed_fn: Box<dyn Fn(&[modgrad_data::tokenize::Code]) -> Vec<f32>>,
    pub token_dim: usize,
}

impl<D: modgrad_data::data_stream::DataStream> SampleProvider<modgrad_traits::TokenInput, usize> for ClassificationAdapter<D> {
    fn next_sample(&mut self) -> Option<Sample<modgrad_traits::TokenInput, usize>> {
        let s = self.stream.next_sample()?;
        let tokens = (self.embed_fn)(&s.context);
        let n_tokens = s.context.len();
        let token_dim = self.token_dim;
        let target = s.target.global_index(
            &modgrad_data::tokenize::CodeLayout::text_only()); // TODO: pass layout
        let input = modgrad_traits::TokenInput { tokens, n_tokens, token_dim };
        Some(Sample { input, target })
    }
    fn reset(&mut self) { self.stream.reset(); }
}

// ═══════════════════════════════════════════════════════════════
// THE TRAINING LOOP — generic over Target type
// ═══════════════════════════════════════════════════════════════

/// Composable training loop — pure functions composed into a pipeline.
///
/// Four pure functions composed:
///   Scheduler: step → lr
///   Brain::forward_cached: (weights, state, input) → (output, state, cache)
///   LossFn::compute: (output, target) → (loss, d_output)
///   Brain::backward + apply_gradients: (weights, cache, d_output) → updated weights
///
/// The Brain's Input type flows through: SampleProvider produces B::Input,
/// Brain::forward_cached consumes it. No type erasure.
pub fn train<B, L, T, S, Log, H>(
    weights: &mut B::Weights,
    data: &mut dyn SampleProvider<B::Input, T>,
    loss_fn: &L,
    scheduler: &S,
    checkpointer: &mut Option<CheckpointManager>,
    logger: &mut Log,
    hook: &mut H,
    config: &TrainConfig,
) -> TrainReport
where
    B: Brain,
    L: LossFn<Target = T>,
    T: Clone,
    S: Scheduler,
    Log: Logger,
    H: StepHook<B::Weights>,
{
    let t0 = std::time::Instant::now();
    let mut ema_loss = 5.0f32;
    let mut best_loss = f32::MAX;
    let mut best_step = 0;

    logger.log_message(&format!("training: {} steps, micro_batch={}",
        config.total_steps, config.micro_batch));

    for step in 0..config.total_steps {
        let lr = scheduler.get_lr(step);
        let mut step_loss = 0.0f32;

        for _mb in 0..config.micro_batch {
            let sample = match data.next_sample() {
                Some(s) => s,
                None => {
                    data.reset();
                    match data.next_sample() {
                        Some(s) => s,
                        None => break,
                    }
                }
            };

            // Forward — Brain's Input type flows through from SampleProvider
            let state = B::init_state(weights);
            let (output, _state, cache) = B::forward_cached(
                weights, state, &sample.input);

            // Loss — generic over target type
            let (loss, d_preds) = loss_fn.compute(
                &output.predictions, &output.certainties, &sample.target);

            // Backward
            let mut grads = B::backward(weights, cache, &d_preds);

            // Apply
            B::apply_gradients(weights, &mut grads, lr, config.grad_clip);

            step_loss += loss;
        }

        // Post-step hook (dream, consolidation, etc.)
        hook.after_step(weights, step, lr);

        step_loss /= config.micro_batch as f32;
        ema_loss = 0.99 * ema_loss + 0.01 * step_loss;

        let is_best = ema_loss < best_loss;
        if is_best {
            best_loss = ema_loss;
            best_step = step;
        }

        if step % config.log_every == 0 || step == config.total_steps - 1 {
            logger.log_step(&StepReport { step, loss: ema_loss, lr, is_best });
        }

        if let Some(ckpt) = checkpointer {
            if ckpt.should_save(step) {
                logger.log_message(&format!("checkpoint at step {}", step));
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    logger.log_message(&format!("done: {} steps in {:.1}s ({:.1} steps/s), best_loss={:.4}",
        config.total_steps, elapsed, config.total_steps as f64 / elapsed, best_loss));

    TrainReport {
        steps_completed: config.total_steps,
        final_loss: ema_loss,
        best_loss,
        best_step,
        elapsed_secs: elapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_provider_trait_object() {
        use modgrad_traits::TokenInput;
        // Verify SampleProvider is object-safe
        struct Dummy;
        impl SampleProvider<TokenInput, usize> for Dummy {
            fn next_sample(&mut self) -> Option<Sample<TokenInput, usize>> {
                let input = TokenInput { tokens: vec![0.0; 8], n_tokens: 2, token_dim: 4 };
                Some(Sample { input, target: 0 })
            }
            fn reset(&mut self) {}
        }
        let mut d: Box<dyn SampleProvider<TokenInput, usize>> = Box::new(Dummy);
        assert!(d.next_sample().is_some());
    }
}
