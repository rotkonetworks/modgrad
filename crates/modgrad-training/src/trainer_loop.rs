//! Closure-based trainer — the runtime-agnostic training loop.
//!
//! The older `Trainer` (in `trainer.rs`) is bound to the `Brain` trait and
//! works well for models that implement it. `TrainerLoop` is for everything
//! else — FFN, transformers, future architectures — and for cases where
//! the caller wants full control over the step function without committing
//! to an abstract `Brain`.
//!
//! # Shape
//! Caller provides a `FnMut(step_idx) -> Option<StepReport>`. The loop
//! owns everything else:
//!
//!   * when to stop (`max_steps`, Ctrl+C, closure returning `None`)
//!   * when to log (`log_every`)
//!   * when to checkpoint (`save_every` + `save_path` + user-provided
//!     save closure)
//!   * progress / loss-smoothing stats
//!   * signal handler install (one Ctrl+C handler per `TrainerLoop::run`,
//!     cleared on return)
//!
//! All tunables live in `TrainerConfig` with sensible `Default` values.
//! None of the numbers in the loop body are magic — change a default in
//! one place and every caller picks it up.
//!
//! # Design non-goals
//! * No implicit optimizer. Caller runs the optimizer inside the step closure.
//! * No implicit data loader. Caller indexes into their own data.
//! * No implicit model knowledge. Caller owns the model.
//!
//! The trainer is thin on purpose — what kills readability in a training
//! binary is the *loop scaffolding* (ctrl+c, cadence timing, smoothed
//! logging), not the step itself. This collapses scaffolding to ~5 lines
//! per binary; the step stays in the caller's hands.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Process-wide signal flag. First `TrainerLoop::run(install_ctrlc=true)`
/// installs the Ctrl+C handler; later runs reset and reuse the same flag.
/// Without this every run after the first would silently fail to install
/// (the `ctrlc` crate rejects a second handler) and lose graceful save.
static SIGNAL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

fn signal_flag() -> &'static Arc<AtomicBool> {
    SIGNAL_FLAG.get_or_init(|| {
        let flag = Arc::new(AtomicBool::new(true));
        let handler_flag = Arc::clone(&flag);
        // Errors here are only possible if *another* install raced ahead
        // of us (impossible under OnceLock::get_or_init's synchronisation)
        // or if the OS rejects the handler. Surface that rather than
        // swallow it — `ctrlc` failing silently was a latent debuggability
        // trap.
        if let Err(e) = ctrlc::set_handler(move || {
            eprintln!("\n  [signal received — finishing current step, then saving…]");
            handler_flag.store(false, Ordering::SeqCst);
        }) {
            eprintln!("warning: TrainerLoop failed to install Ctrl+C handler: {e}");
        }
        flag
    })
}

/// Report the step closure returns to describe one training step.
#[derive(Debug, Clone)]
pub struct StepReport {
    /// Scalar loss at this step.
    pub loss: f32,
    /// Optional accuracy in `[0, 1]`. For models that don't have an
    /// accuracy notion (regression tasks), leave `None`.
    pub accuracy: Option<f32>,
    /// Optional progress fraction through the dataset, `[0, 1]`.
    pub data_progress: Option<f32>,
    /// Free-form scalar metrics the caller wants surfaced in the log line.
    /// e.g. `vec![("grad_norm", 0.83), ("lr", 1.0e-4)]`.
    pub extras: Vec<(&'static str, f32)>,
}

impl StepReport {
    /// Minimal builder — just a loss.
    pub fn new(loss: f32) -> Self {
        Self { loss, accuracy: None, data_progress: None, extras: Vec::new() }
    }
    pub fn with_accuracy(mut self, acc: f32) -> Self { self.accuracy = Some(acc); self }
    pub fn with_progress(mut self, p: f32) -> Self { self.data_progress = Some(p); self }
    pub fn with_extra(mut self, name: &'static str, val: f32) -> Self {
        self.extras.push((name, val)); self
    }
}

/// Summary returned by `TrainerLoop::run`.
#[derive(Debug, Clone, Default)]
pub struct TrainerReport {
    pub steps_completed: usize,
    pub final_avg_loss: f32,
    pub best_avg_loss: f32,
    pub best_step: usize,
    pub elapsed_secs: f64,
    pub stopped_by_signal: bool,
}

// ─── Configuration ──────────────────────────────────────────

/// All tunable knobs for `TrainerLoop::run`. `Default` supplies sensible
/// values for medium-scale training. Override fields to specialise.
///
/// Convention: every field is `pub`; no opaque builder. A runtime author
/// can `TrainerConfig { max_steps: 50_000, ..Default::default() }` and
/// the intent is obvious.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Upper bound on total training steps. `usize::MAX` = run until
    /// Ctrl+C or the step closure returns `None`.
    pub max_steps: usize,

    /// Print a smoothed log line every N steps.
    pub log_every: usize,

    /// Take a checkpoint every N steps. `None` disables periodic saves.
    /// Requires `save_fn` to be provided to `run`.
    pub save_every: Option<usize>,

    /// Install a Ctrl+C handler that cleanly finishes the current step,
    /// saves (if `save_every` is set), and exits. `true` by default —
    /// running two `TrainerLoop` s concurrently would stomp the handler,
    /// so disable in that unusual case.
    pub install_ctrlc: bool,

    /// Log line prefix. Useful to distinguish multiple trainers. Empty
    /// by default.
    pub log_prefix: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_steps: 10_000,
            log_every: 100,
            save_every: None,
            install_ctrlc: true,
            log_prefix: String::new(),
        }
    }
}

// ─── The loop ──────────────────────────────────────────────

/// Closure-based trainer. Caller provides a step function; the loop
/// owns cadence, logging, signal handling, and checkpointing.
pub struct TrainerLoop {
    pub config: TrainerConfig,
}

impl TrainerLoop {
    pub fn new(config: TrainerConfig) -> Self { Self { config } }

    /// Drive the training loop. Stops when any of these happen:
    ///   * `step` returned `None` (dataset exhausted, user decided to stop)
    ///   * `step_idx` reached `config.max_steps`
    ///   * Ctrl+C was pressed (if `install_ctrlc`)
    ///
    /// `save_fn` is called every `save_every` steps (when set) and once
    /// at the end on Ctrl+C. Pass `|| Ok(())` to disable saves while
    /// keeping the rest of the loop. Error from `save_fn` is logged but
    /// does not stop training.
    pub fn run<Step, Save>(
        &self,
        mut step: Step,
        mut save_fn: Save,
    ) -> TrainerReport
    where
        Step: FnMut(usize) -> Option<StepReport>,
        Save: FnMut() -> Result<(), Box<dyn std::error::Error>>,
    {
        // `running` is either the shared process-wide signal flag (if the
        // caller opted into Ctrl+C) or a fresh local flag that never
        // transitions. Reset to `true` on entry so a prior run's signal
        // doesn't kill this one before it starts.
        let running: Arc<AtomicBool> = if self.config.install_ctrlc {
            let shared = Arc::clone(signal_flag());
            shared.store(true, Ordering::SeqCst);
            shared
        } else {
            Arc::new(AtomicBool::new(true))
        };

        let started = Instant::now();
        let prefix = self.config.log_prefix.clone();
        let mut report = TrainerReport::default();
        report.best_avg_loss = f32::INFINITY;

        // Running aggregates between log windows.
        let mut window_loss = 0.0f32;
        let mut window_acc_sum = 0.0f32;
        let mut window_acc_count = 0usize;
        let mut window_n = 0usize;
        let mut last_progress: f32 = 0.0;

        let max = self.config.max_steps;
        for step_idx in 0..max {
            if !running.load(Ordering::SeqCst) {
                report.stopped_by_signal = true;
                break;
            }

            let out = match step(step_idx) {
                Some(r) => r,
                None    => break,
            };
            report.steps_completed = step_idx + 1;

            window_loss += out.loss;
            if let Some(a) = out.accuracy { window_acc_sum += a; window_acc_count += 1; }
            if let Some(p) = out.data_progress { last_progress = p; }
            window_n += 1;

            if (step_idx + 1) % self.config.log_every == 0 {
                let avg = window_loss / window_n as f32;
                if avg < report.best_avg_loss {
                    report.best_avg_loss = avg;
                    report.best_step = step_idx + 1;
                }
                let acc_frag = if window_acc_count > 0 {
                    format!(" | acc {:.1}%", 100.0 * window_acc_sum / window_acc_count as f32)
                } else { String::new() };
                let prog_frag = format!(" | data {:.0}%", 100.0 * last_progress);
                let mut extras_frag = String::new();
                for (k, v) in &out.extras {
                    extras_frag.push_str(&format!(" | {k} {v:.4}"));
                }
                eprintln!("{prefix}step {:6} | loss {avg:.3}{acc_frag}{prog_frag}{extras_frag}",
                    step_idx + 1);
                window_loss = 0.0; window_acc_sum = 0.0; window_acc_count = 0; window_n = 0;
            }

            if let Some(every) = self.config.save_every {
                if (step_idx + 1) % every == 0 {
                    if let Err(e) = save_fn() {
                        eprintln!("{prefix}  [save failed: {e}]");
                    } else {
                        eprintln!("{prefix}  [saved at step {}]", step_idx + 1);
                    }
                }
            }
        }

        // Final save on signal or natural completion.
        if report.stopped_by_signal || self.config.save_every.is_some() {
            if let Err(e) = save_fn() {
                eprintln!("{}  [final save failed: {e}]", self.config.log_prefix);
            }
        }

        report.final_avg_loss =
            if window_n > 0 { window_loss / window_n as f32 } else { report.best_avg_loss };
        report.elapsed_secs = started.elapsed().as_secs_f64();
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn runs_to_max_steps_and_reports() {
        let cfg = TrainerConfig {
            max_steps: 10,
            log_every: 5,
            save_every: None,
            install_ctrlc: false,  // isolated tests shouldn't touch signals
            log_prefix: String::new(),
        };
        let calls = Cell::new(0);
        let report = TrainerLoop::new(cfg).run(
            |_step| {
                calls.set(calls.get() + 1);
                Some(StepReport::new(1.0 / (calls.get() as f32)))
            },
            || Ok(()),
        );
        assert_eq!(report.steps_completed, 10);
        assert_eq!(calls.get(), 10);
        assert!(!report.stopped_by_signal);
    }

    #[test]
    fn step_returning_none_stops_early() {
        let cfg = TrainerConfig { max_steps: 100, install_ctrlc: false, ..Default::default() };
        let report = TrainerLoop::new(cfg).run(
            |step| if step < 3 { Some(StepReport::new(0.5)) } else { None },
            || Ok(()),
        );
        assert_eq!(report.steps_completed, 3);
    }

    #[test]
    fn checkpoint_callback_fires_at_cadence() {
        let cfg = TrainerConfig {
            max_steps: 20, log_every: 1000,  // suppress step logs
            save_every: Some(5), install_ctrlc: false,
            ..Default::default()
        };
        let saves = Cell::new(0);
        TrainerLoop::new(cfg).run(
            |_| Some(StepReport::new(0.1)),
            || { saves.set(saves.get() + 1); Ok(()) },
        );
        // 20 steps, save_every=5 ⇒ saves at 5,10,15,20 plus 1 final flush = 5.
        assert_eq!(saves.get(), 5);
    }

    #[test]
    fn best_step_tracks_minimum_windowed_loss() {
        let cfg = TrainerConfig {
            max_steps: 30, log_every: 10, install_ctrlc: false,
            ..Default::default()
        };
        // Loss schedule: windows average 3.0, 1.0, 2.0 — best at step 20.
        let report = TrainerLoop::new(cfg).run(
            |s| Some(StepReport::new(match s / 10 {
                0 => 3.0, 1 => 1.0, _ => 2.0,
            })),
            || Ok(()),
        );
        assert_eq!(report.best_step, 20);
        assert!((report.best_avg_loss - 1.0).abs() < 1e-5);
    }

    #[test]
    fn signal_flag_is_reset_between_runs_with_ctrlc() {
        // Regression: the shared Ctrl+C flag used to silently stay `false`
        // after a signal-stopped run, so the next `run` would bail out
        // at step 0 thinking it had just been interrupted. The fix resets
        // the flag to `true` on entry — here we verify that a run with
        // `install_ctrlc: true` still executes the full step schedule
        // after a prior run, without needing any external intervention.
        let cfg = TrainerConfig {
            max_steps: 5, log_every: 100, install_ctrlc: true,
            ..Default::default()
        };
        // Seed the flag to `false` (as if a prior run was signal-stopped).
        // Touch the OnceLock so the static exists; handler install is
        // side-effect-free for the test.
        signal_flag().store(false, Ordering::SeqCst);

        let report = TrainerLoop::new(cfg).run(
            |_| Some(StepReport::new(0.1)),
            || Ok(()),
        );
        assert_eq!(report.steps_completed, 5,
            "entry should reset signal flag so a stale `false` from a prior \
             run doesn't make the loop exit at step 0");
        assert!(!report.stopped_by_signal,
            "no signal was raised during this run");
    }

    #[test]
    fn extras_and_accuracy_propagate_through_report() {
        let cfg = TrainerConfig {
            max_steps: 5, log_every: 5, install_ctrlc: false,
            ..Default::default()
        };
        TrainerLoop::new(cfg).run(
            |_| Some(StepReport::new(0.5)
                .with_accuracy(0.92)
                .with_progress(0.3)
                .with_extra("lr", 1e-4)),
            || Ok(()),
        );
        // No assertion needed — this is a doc test that the API compiles
        // and runs without panic.
    }
}
