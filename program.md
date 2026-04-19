# program.md — autoresearch for modgrad / isis

An autonomous-researcher contract. Based on `karpathy/autoresearch`, adapted
for modgrad's Rust codebase. The driving agent (Claude Code with
`--dangerously-skip-permissions`) reads this file, edits code, runs one
5-minute experiment, decides keep or revert, and loops indefinitely.

## Setup (first-run checks)

Before starting the experiment loop, agree with the human on:

1. **Run tag**: propose one based on today's date (e.g. `apr19`). The
   branch `autoresearch/<tag>` must not already exist. Create it from
   `main`: `git checkout -b autoresearch/<tag>`.
2. **Validation file**: `val.txt` must exist at repo root. If it
   doesn't, create a stable split *once* off the main training corpus:
   ```
   tail -c 262144 train_climbmix.txt > val.txt
   ```
   Do **not** re-create `val.txt` on later runs — that changes the
   metric denominator and makes runs incomparable.
3. **Baseline build works**: `cargo build --release -p isis` completes
   with no errors.
4. **Baseline train + eval works**:
   ```
   cargo build --release -p isis && \
     ./target/release/isis learn-ffn --budget 300 --val-data val.txt \
       /tmp/autoresearch_ckpt.bin train_climbmix_5m.txt > run.log 2>&1 && \
     grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
5. **`results.tsv` has a header**: if missing, create it (see below).
6. **Confirm and go**: briefly summarise what you see and wait for
   confirmation, then start the loop.

## Experimentation — what you CAN change

You edit these directories. Anything in them is fair game: architecture,
hyperparameters, optimizer internals, training step body.

- `crates/modgrad-ffn/src/**` — the FFN cerebellum model (architecture,
  forward/backward, optimizer state for the FFN).
- `crates/modgrad-ctm/src/**` — CTM regional graph (only relevant if you
  plan experiments with `isis learn`, not `isis learn-ffn`).
- `isis/src/main.rs` — training loop body, model-size selection,
  dream-phase cadence inside the `learn_ffn` function. **Don't** touch
  the `main()` CLI parsing or the `eval_ffn` function — those are part
  of the evaluation harness.

## Experimentation — what you CANNOT change

Hands off. These define the ground truth; mutating them invalidates the
val_bpb comparison.

- `crates/modgrad-training/src/metrics.rs` — `BitsPerByte` and friends.
- `crates/modgrad-training/src/autoresearch.rs` — the summary format the
  agent grep-parses.
- `crates/modgrad-training/src/checkpoint_bundle.rs` — checkpoint format.
- `crates/modgrad-device/src/**` — GPU kernels. Fragile; one bad
  dispatch hangs gfx1102.
- `isis/src/main.rs` — only the `eval_ffn` function and
  `compute_ffn_val_bpb`. Read-only.
- `val.txt` — the validation set. Re-creating this changes the metric.
- `Cargo.toml` dependencies — no adding crates. Work within what's
  already declared.

## The metric

**`val_bpb` (validation bits-per-byte). Lower is better.** It's the only
number that decides keep vs. discard. Computed as
`cross_entropy_in_nats / n_targets / ln(2)` over the full `val.txt`
with non-overlapping context-length windows. Tokenizer-agnostic so
architectural changes are fairly compared.

**VRAM** is a soft constraint. Some growth is acceptable for real
val_bpb gains. Don't blow up VRAM dramatically.

**Simplicity** breaks ties. A 0.001 bpb gain that adds 20 lines of hacky
code is probably not worth keeping. A 0 bpb change that deletes code is
a keep. Err toward simpler.

## Choosing a task

Two tasks share the same autoresearch contract (same `val_bpb:` grep,
same keep/discard decision rule). Pick one per branch — don't interleave
them in the same branch, because the two `val_bpb` numbers mean different
things and aren't directly comparable.

- **Language** (`isis learn-ffn`): FFN cerebellum trained on byte-level
  text. `val_bpb` = cross-entropy per byte on `val.txt`. Standard
  LM-style iteration: architecture, LR schedule, activation, width.
  Results go in `results-language.tsv`.

- **Mazes** (`mazes` binary): CTM trained on route prediction from maze
  pixels. `val_bpb` = `1 - first_step_accuracy` (0 = perfect routing,
  1 = random guess). Iteration focus: CTM ticks, d_model, exit
  strategy, loss mix. Results go in `results-mazes.tsv`.

The "CAN change" / "CANNOT change" sets below depend on which task
you pick; see the per-task subsections under "Running one experiment".

## Running one experiment

### Language task (`learn-ffn`)

```
cargo build --release -p isis && \
  ./target/release/isis learn-ffn --budget 300 --val-data val.txt \
    /tmp/autoresearch_ffn.bin train_climbmix_5m.txt > run.log 2>&1
```

Flags:
- `--budget 300` — 5-minute wall-clock training cap.
- `--val-data val.txt` — triggers end-of-run val_bpb eval + summary.
- `/tmp/autoresearch_ffn.bin` — throwaway path, fresh file per run
  (do not reuse a checkpoint across experiments — biases the start).

Task-specific mutable set (in addition to the global "CAN change" list):
- `crates/modgrad-ffn/src/**`
- `isis/src/main.rs` → `learn_ffn` function body only
- **Do not** touch `eval_ffn` / `compute_ffn_val_bpb` — evaluation harness.

### Maze task (`mazes`)

```
cargo build --release -p mazes && \
  ./target/release/mazes --size 21 --ticks 16 --steps 5000 --batch 8 \
    --autoresearch-summary > run.log 2>&1
```

Flags:
- `--steps 5000` — per-experiment iteration cap (mazes doesn't yet have
  `--budget`; keep steps low enough to finish in ~5 minutes).
- `--autoresearch-summary` — triggers the summary print after the
  200-maze eval.
- `--size` / `--ticks` / `--d-model` — task knobs, fair game to mutate.

Task-specific mutable set:
- `crates/modgrad-ctm/src/**` — the CTM regional graph and core.
- `examples/mazes/src/main.rs` → training body only.
- **Do not** touch the `eval()` function in `examples/mazes/src/main.rs`
  or `examples/mazes/src/maze_gen.rs` — evaluation harness.

### Reading the result (either task)

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

If the grep output is empty the run crashed. `tail -n 80 run.log` for
the compiler or runtime error. Dumb bugs (typo, missing import) — fix
and re-run. Fundamentally broken idea — log as `crash` and move on.

## Logging results

`results-language.tsv` or `results-mazes.tsv` at repo root, depending
on the task this branch is iterating on. Tab-separated (commas break
descriptions). Header row (identical for both files):
```
commit	val_bpb	memory_gb	status	description
```

Keep tasks in separate files because the `val_bpb` numbers aren't
directly comparable across tasks — mazes reports `1 - accuracy`, language
reports actual cross-entropy per byte.

Five columns:
1. `git rev-parse --short HEAD` (7 chars).
2. val_bpb, six decimals, e.g. `1.234567`. Use `0.000000` for crashes.
3. peak memory in GB, `.1f`. Divide `peak_vram_mb` by 1024. Use `0.0`
   for CPU-only or crashes.
4. Status: `keep`, `discard`, or `crash`.
5. Short human-readable description of the experimental change.

Example:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	0.0	keep	baseline
b2c3d4e	1.220100	0.0	keep	widen hidden from 2048 to 3072
c3d4e5f	1.245900	0.0	discard	replace SwiGLU with GeLU
d4e5f6g	0.000000	0.0	crash	quadruple model width (OOM)
```

**Do NOT commit `results-*.tsv`.** Already in `.gitignore`. The file is
your scratch log, not repo history; git commit messages carry the keep
decisions.

## The loop

On each iteration:

1. **Check git state**: current branch/commit.
2. **Plan one change** to a file in the "CAN change" list. State the
   hypothesis in one sentence before editing.
3. **Edit the code**.
4. **Commit**: `git add -A && git commit -m "<short description>"`.
5. **Check you didn't touch read-only files**:
   ```
   ./scripts/autoresearch-check.sh HEAD~1
   ```
   If it reports a violation, `git reset --hard HEAD~1` and try a
   different mutation. The ground-truth evaluation harness is sacred;
   accidentally editing it invalidates every prior `val_bpb` row in
   `results.tsv`.
6. **Delete the prior checkpoint** so the new run starts fresh
   (resuming a prior checkpoint biases the val_bpb). For the language
   task:
   ```
   rm -f /tmp/autoresearch_ffn.bin
   ```
   Mazes trains from scratch every call and has no checkpoint to delete.
7. **Run** (pick the command for this branch's task — see "Running one
   experiment" above for both). Default (language):
   ```
   cargo build --release -p isis && \
     ./target/release/isis learn-ffn --budget 300 --val-data val.txt \
       /tmp/autoresearch_ffn.bin train_climbmix_5m.txt > run.log 2>&1
   ```
8. **Extract results**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
9. **Log to results.tsv**.
10. **Decide**:
    - val_bpb strictly lower than the current branch tip → **keep** (do
      nothing, the commit stands).
    - val_bpb equal or higher → **discard**:
      `git reset --hard HEAD~1`.
    - crash → **discard**:
      `git reset --hard HEAD~1`, log `crash`.

Repeat. Forever.

## Hard constraints

- **Never run `cargo clean`** or delete `target/`. Incremental rebuilds
  depend on the warm compile cache; blowing it away costs ~5 minutes
  per loop iteration and burns the subscription quota.
- **Never skip git hooks** (`--no-verify`, `--no-gpg-sign`). If a hook
  fails, investigate and fix, don't bypass.
- **Never `git push`**. The experiment branch is local-only until the
  human decides what to upstream.
- **10-minute hard timeout**: if an experiment exceeds 10 minutes total,
  kill it and treat it as a crash. Budget is 5 min; 10 min is 2× as a
  safety margin for compile cold starts.

## Autonomy contract

Once the experiment loop has begun, do not pause to ask "should I keep
going?" or "is this a good stopping point?". The human might be asleep
or away and expects the loop to run indefinitely until manually stopped
(Ctrl+C / branch abandoned / human interrupts).

If you run out of ideas, think harder: re-read these files for new
angles, combine near-misses, try more radical architectural changes
(different normalization, different gating, different width/depth
tradeoffs). The loop runs until the human stops it, period.

## Ideas to start from (not prescriptive)

- Width vs depth: same param budget reshuffled.
- Tied vs untied input/output embeddings.
- SwiGLU vs GeGLU vs plain GeLU.
- Learning-rate schedule shape (warmup length, decay curve).
- Pre-norm vs post-norm placement.
- AdamW vs Muon on the linear layers.
- Batch size / gradient-accumulation at fixed token throughput.

Most ideas won't work. Discard cheerfully.
