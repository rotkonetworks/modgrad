#!/usr/bin/env python3
"""Export Qwen2.5-0.5B to ONNX with hidden states output.

Produces a single model.onnx that outputs both hidden_states and logits.
Our OnnxCerebellum uses the hidden_states (896-dim) as the frozen
cerebellum output, projected into cortex space.
"""

import torch
import onnx
from onnx import helper, TensorProto
import numpy as np
import os, shutil

MODEL_DIR = "/steam/llm/qwen2.5-0.5b-onnx"

# The optimum export already created model.onnx with logits output.
# We need to find the last hidden state node (before lm_head) and
# add it as an additional output.

print("Loading ONNX model...")
model = onnx.load(os.path.join(MODEL_DIR, "model.onnx"))

# Find the lm_head matmul — it's the last MatMul before the logits output
# The input to this MatMul is the last hidden state
logits_output = model.graph.output[0]
print(f"Logits output: {logits_output.name}")

# Walk backwards from logits to find the hidden state tensor
# The logits are typically: hidden_states -> MatMul(lm_head_weight) -> logits
# Find the node that produces the logits output
logits_node = None
for node in model.graph.node:
    if logits_output.name in node.output:
        logits_node = node
        break

if logits_node is None:
    # The output might be a renamed alias — search by following the graph
    # Look for the last MatMul in the graph (lm_head is typically a MatMul)
    matmuls = [n for n in model.graph.node if n.op_type == "MatMul"]
    if matmuls:
        logits_node = matmuls[-1]
        print(f"Last MatMul: {logits_node.name}, inputs: {list(logits_node.input)}")

if logits_node and logits_node.op_type == "MatMul":
    hidden_state_name = logits_node.input[0]
    print(f"Hidden state tensor: {hidden_state_name}")

    # Add hidden_states as output
    hidden_output = helper.make_tensor_value_info(
        "hidden_states",
        TensorProto.FLOAT,
        ["batch_size", "sequence_length", 896]
    )

    # We need to create an Identity node to expose the hidden state
    identity_node = helper.make_node(
        "Identity",
        inputs=[hidden_state_name],
        outputs=["hidden_states"],
        name="hidden_states_output"
    )
    model.graph.node.append(identity_node)
    model.graph.output.append(hidden_output)

    # Save modified model with external data
    backbone_path = os.path.join(MODEL_DIR, "backbone.onnx")
    onnx.save(model, backbone_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="backbone.onnx_data")

    # External data is saved alongside backbone.onnx

    print(f"Saved backbone with hidden_states output: {backbone_path}")

    # Verify
    m = onnx.load(backbone_path, load_external_data=False)
    print("Outputs:")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {dims}")
else:
    print("Could not find lm_head MatMul — trying alternative approach")
    # List last 10 nodes
    for node in model.graph.node[-10:]:
        print(f"  {node.op_type}: {node.name} -> {list(node.output)}")
