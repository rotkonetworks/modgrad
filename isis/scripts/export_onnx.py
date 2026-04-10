#!/usr/bin/env python3
"""Export Qwen 2.5-0.5B as ONNX for the brain crate.

Exports two models:
1. backbone.onnx: token_ids → hidden_state at layer 23 (for key computation)
2. lm_head.onnx: hidden_state → logits (for generation)

Both together = full inference pipeline.
Separately = teach only needs backbone, not lm_head.
"""

import torch
import torch.nn as nn
import os


def export():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen 2.5-0.5B...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    model.eval()

    D = model.config.hidden_size  # 896
    n_layers = model.config.num_hidden_layers  # 24
    V = model.config.vocab_size
    target_layer = n_layers - 1  # 23

    print(f"  D={D}, layers={n_layers}, vocab={V}")

    os.makedirs("models", exist_ok=True)

    # ─── Export backbone: tokens → hidden at layer 23 ─────────
    class BackboneToLayer23(nn.Module):
        def __init__(self, model, target_layer):
            super().__init__()
            self.embed = model.model.embed_tokens
            self.layers = model.model.layers[:target_layer + 1]
            self.target_layer = target_layer
            self.rotary = model.model.rotary_emb
            self.post_attn_norm = model.model.layers[target_layer].post_attention_layernorm

        def forward(self, input_ids):
            x = self.embed(input_ids)
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            pe = self.rotary(x, pos)

            for i, layer in enumerate(self.layers):
                if i == self.target_layer:
                    residual = x
                    x_normed = layer.input_layernorm(x)
                    attn_out = layer.self_attn(
                        x_normed, attention_mask=None,
                        position_embeddings=pe)[0]
                    x = residual + attn_out
                    # Return pre-MLP hidden (what CTM/brain receives)
                    hidden = self.post_attn_norm(x)
                    # Also complete the layer for full hidden
                    full = x + layer.mlp(hidden)
                    return hidden, full
                else:
                    x = layer(x, position_embeddings=pe)
            return x, x

    # ─── Export lm_head: full_hidden → logits ─────────────────
    class LMHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Remaining layers after target + final norm + lm_head
            self.final_norm = model.model.norm
            self.lm_head = model.lm_head

        def forward(self, full_hidden):
            x = self.final_norm(full_hidden)
            return self.lm_head(x)

    # Export backbone
    print("Exporting backbone (tokens → hidden)...")
    backbone = BackboneToLayer23(model, target_layer)
    dummy_ids = torch.randint(0, V, (1, 32))

    with torch.no_grad():
        hidden, full = backbone(dummy_ids)
        print(f"  hidden: {hidden.shape}, full: {full.shape}")

    torch.onnx.export(
        backbone, dummy_ids,
        "models/backbone.onnx",
        input_names=["input_ids"],
        output_names=["hidden", "full_hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "hidden": {0: "batch", 1: "seq_len"},
            "full_hidden": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    backbone_size = os.path.getsize("models/backbone.onnx")
    print(f"  Saved models/backbone.onnx ({backbone_size / 1e6:.1f} MB)")

    # Export lm_head
    print("Exporting lm_head (hidden → logits)...")
    head = LMHead(model)
    dummy_hidden = torch.randn(1, 32, D)

    torch.onnx.export(
        head, dummy_hidden,
        "models/lm_head.onnx",
        input_names=["full_hidden"],
        output_names=["logits"],
        dynamic_axes={
            "full_hidden": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    head_size = os.path.getsize("models/lm_head.onnx")
    print(f"  Saved models/lm_head.onnx ({head_size / 1e6:.1f} MB)")

    # Save tokenizer
    tokenizer.save_pretrained("models/tokenizer")
    print(f"  Saved models/tokenizer/")

    # Verify
    print("\nVerifying ONNX export...")
    import onnxruntime as ort
    sess_backbone = ort.InferenceSession("models/backbone.onnx")
    sess_head = ort.InferenceSession("models/lm_head.onnx")

    test_text = "The capital of France is"
    ids = tokenizer.encode(test_text, add_special_tokens=False)
    input_ids = torch.tensor([ids]).numpy()

    hidden_out, full_out = sess_backbone.run(None, {"input_ids": input_ids})
    logits_out = sess_head.run(None, {"full_hidden": full_out})

    next_token = logits_out[0][0, -1].argmax()
    predicted = tokenizer.decode([next_token])
    print(f"  '{test_text}' → '{predicted}'")
    print(f"  Hidden shape: {hidden_out.shape}")
    print(f"  Logits shape: {logits_out[0].shape}")

    print("\nDone. Total model size:",
          f"{(backbone_size + head_size) / 1e6:.1f} MB")


if __name__ == "__main__":
    export()
