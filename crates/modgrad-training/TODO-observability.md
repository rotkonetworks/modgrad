# Observability Pipeline

Structured metrics from training and inference → debugger analysis.

## Training-time metrics (modgrad-training)
- Per-step: loss, lr, grad_norm, tokens/sec, memory
- Per-layer: activation mean/var, dead neuron ratio, gradient magnitude, weight norm
- Per-tick (CTM): certainty evolution, min-CE tick, thinking efficiency
- Alerts: NaN, loss spike, gradient explosion, dead neuron threshold

## Inference-time metrics (isis runtime)
- Per-request: latency, tokens generated, ticks used, certainty at exit
- Per-model: throughput, p50/p95/p99 latency, GPU utilization
- Memory: KV cache size, active sessions, pressure

## Pipeline
```
training loop / inference server
  → modgrad-training::metrics (structured events)
  → telemetry transport (QUIC / local socket)
  → debugger (Polars for analysis, live dashboard)
```

## Dependencies
- modgrad-training already has metrics.rs (basic loss/accuracy tracking)
- sdk/io has telemetry.rs + telemetry_transport.rs (QUIC, feature-gated)
- debugger/ is a separate workspace crate — add polars here for analysis
- Polars lazy API for streaming large training logs without full memory load

## Prerequisite
- Wire up the Brain trait + generic trainer first — metrics should flow through
  the trait boundary, not be hardcoded per architecture
