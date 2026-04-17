#!/bin/bash
# Overnight training: CTM cortex with frozen Qwen cerebellum
# Run from modgrad root: bash scripts/train_overnight.sh
#
# This trains the cortex to predict English bytes using the frozen
# Qwen2.5-0.5B as its cerebellum (world model).
#
# Expected: ~2.7 tok/sec, context=16, ~8 hours = ~120K tokens, ~7500 steps
# Check progress: tail -f train_cereb.log

set -e

CHECKPOINT="train_cereb.bin"
DATA="isis/runtime/src/small_text.bin"
CEREB="/steam/llm/hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors"
CONTEXT=16

echo "Building..."
cargo build --release 2>&1 | tail -1

echo ""
echo "Training: CTM cortex + frozen Qwen2.5-0.5B cerebellum"
echo "  Checkpoint: $CHECKPOINT"
echo "  Data: $DATA"
echo "  Cerebellum: $CEREB"
echo "  Context: $CONTEXT"
echo "  Ctrl+C to save and stop"
echo ""

# Remove stale checkpoint for clean run
rm -f "$CHECKPOINT" "${CHECKPOINT%.bin}.opt.bin"

exec target/release/isis learn "$CHECKPOINT" "$DATA" \
    --context "$CONTEXT" \
    --frozen-cereb "$CEREB" \
    2>&1 | tee train_cereb.log
