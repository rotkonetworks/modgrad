# brain_qwen_nll test corpora

Reproducible corpora for the redshiftzero ablation runs documented
in commit `5b12bda` and the recall-corpus follow-up.

## `computing_history.txt` (265 tokens, narrative)

Domain-coherent paragraph about the history of computing. Diverse
vocabulary, low repetition. Used as the canonical "narrative" test
where random-brain ablation must fail (it has no learnable structure
on diverse text).

Reference results (rank=2, lr=0.01, 30 epochs):

  brain-mode=real    Δ -0.0027 ← only mode that beats baseline
  brain-mode=random  Δ +0.0025 (worse — noise has no structure)
  brain-mode=embed   Δ +0.0001 (~tied)
  brain-mode=zero    Δ +0.0001 (~tied)

Conclusion: brain dynamics ARE load-bearing on diverse text.

## `recall.txt` (137 tokens, repeated-fact patterns)

"X is Y" facts in repetitive structure. Same surface forms appear
in multiple sentences. Used to test whether the prior "real-only
beats baseline" result generalises.

Reference results (rank=2, lr=0.01, 30 epochs):

  brain-mode=real    Δ -0.0009
  brain-mode=random  Δ -0.0032 (BIGGER win than real)

Conclusion: on highly-repetitive data, the modulator can fit using
ANY content-causal feature — random per-position noise becomes a
sufficient distinguishing signature for the repetitive targets, and
beats brain dynamics here because noise gives a cleaner per-position
fingerprint than the brain's slowly-evolving state.

This DOESN'T disprove the narrative-corpus result; it sharpens what
"average NLL across positions" measures vs. what "specific semantic
recall" would measure. The two effects are conflated under the
current metric.

## What's not yet measured

A proper held-out recall test (referenced fact never appears in
training; per-position logit reporting on answer token; query
position-specific comparison) would isolate (b) from (a). That's a
separate `recall_probe` binary on top of the same modulator stack —
no new architectural primitives needed, just a more discriminating
evaluation harness.

## Running

    cargo run --release --features rocm -p brain_qwen_nll -- \
        --text-file examples/brain_qwen_nll/corpora/computing_history.txt \
        --rank 2 --lr 0.01 --epochs 30 --max-seq 512 \
        --brain-mode real
