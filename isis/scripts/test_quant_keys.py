"""Test whether quantized models produce compatible hidden states for isis memory keys.

Core question: if we teach facts using f32 hidden states, can we recall them
using hidden states from a quantized model?

Tests:
1. f32 vs f16 hidden states (same prompt → cosine similarity)
2. f32 vs int8 (bitsandbytes) hidden states
3. Cross-recall: f32 keys vs quantized query keys
4. Causal attention guarantee under quantization (identical prefix → sim≈1.0?)
"""

import torch
import numpy as np
import json
from pathlib import Path


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def get_hidden_states(model, tokenizer, texts, device="cpu"):
    """Run texts through model, return last hidden state at last token position."""
    keys = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # Last hidden state, last token position
        h = out.hidden_states[-1][0, -1].float().cpu().numpy()
        h = h / (np.linalg.norm(h) + 1e-10)
        keys.append(h)
    return keys


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test prompts — same ones we taught to isis
    prompts = [
        "You are",
        "Your identity is",
        "How you think is",
        "Your purpose is",
        "When you are wrong you",
        "Your memory works by",
    ]

    # Also test the causal attention guarantee
    causal_pairs = [
        ("The capital of France is", "The capital of France is"),  # identical → should be 1.0
        ("The capital of France is", "The capital of Germany is"),  # different → should be < 1.0
        ("You are", "You are"),  # identical
    ]

    # --- f32 baseline ---
    print("\n=== f32 (baseline) ===")
    model_f32 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model_f32.eval()
    keys_f32 = get_hidden_states(model_f32, tokenizer, prompts)
    print(f"  Got {len(keys_f32)} keys, dim={len(keys_f32[0])}")

    # --- f16 ---
    print("\n=== f16 ===")
    model_f16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    model_f16.eval()
    keys_f16 = get_hidden_states(model_f16, tokenizer, prompts)

    print("  f32 vs f16 per-prompt similarity:")
    for i, p in enumerate(prompts):
        sim = cosine_sim(keys_f32[i], keys_f16[i])
        print(f"    {p:30s} sim={sim:.6f}")

    # --- bf16 ---
    print("\n=== bf16 ===")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model_bf16.eval()
    keys_bf16 = get_hidden_states(model_bf16, tokenizer, prompts)

    print("  f32 vs bf16 per-prompt similarity:")
    for i, p in enumerate(prompts):
        sim = cosine_sim(keys_f32[i], keys_bf16[i])
        print(f"    {p:30s} sim={sim:.6f}")

    # --- int8 (bitsandbytes) ---
    has_bnb = False
    try:
        import bitsandbytes
        print("\n=== int8 (bitsandbytes) ===")
        model_int8 = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, trust_remote_code=True, device_map="auto"
        )
        model_int8.eval()
        keys_int8 = get_hidden_states(model_int8, tokenizer, prompts, device="cuda")
        has_bnb = True

        print("  f32 vs int8 per-prompt similarity:")
        for i, p in enumerate(prompts):
            sim = cosine_sim(keys_f32[i], keys_int8[i])
            print(f"    {p:30s} sim={sim:.6f}")
    except Exception as e:
        print(f"\n=== int8 skipped: {e} ===")

    # --- int4 (bitsandbytes) ---
    try:
        if has_bnb:
            print("\n=== int4 (bitsandbytes NF4) ===")
            from transformers import BitsAndBytesConfig
            q4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_int4 = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=q4_config,
                trust_remote_code=True, device_map="auto"
            )
            model_int4.eval()
            keys_int4 = get_hidden_states(model_int4, tokenizer, prompts, device="cuda")

            print("  f32 vs int4 per-prompt similarity:")
            for i, p in enumerate(prompts):
                sim = cosine_sim(keys_f32[i], keys_int4[i])
                print(f"    {p:30s} sim={sim:.6f}")
    except Exception as e:
        print(f"\n=== int4 skipped: {e} ===")

    # --- Causal attention guarantee under quantization ---
    print("\n=== Causal Attention Guarantee ===")
    for dtype_name, model in [("f32", model_f32), ("f16", model_f16), ("bf16", model_bf16)]:
        print(f"\n  {dtype_name}:")
        for text_a, text_b in causal_pairs:
            ka = get_hidden_states(model, tokenizer, [text_a])[0]
            kb = get_hidden_states(model, tokenizer, [text_b])[0]
            sim = cosine_sim(ka, kb)
            label = "IDENTICAL" if text_a == text_b else "DIFFERENT"
            print(f"    [{label}] \"{text_a}\" vs \"{text_b}\": sim={sim:.6f}")

    # --- Cross-recall test: can f16 keys recall f32-taught facts? ---
    print("\n=== Cross-Recall Test ===")
    print("  Can f16/bf16 hidden states match f32 memory keys (threshold=0.70)?")
    threshold = 0.70
    for dtype_name, keys_q in [("f16", keys_f16), ("bf16", keys_bf16)]:
        matches = 0
        for i, p in enumerate(prompts):
            sim = cosine_sim(keys_f32[i], keys_q[i])
            if sim >= threshold:
                matches += 1
        print(f"  {dtype_name}: {matches}/{len(prompts)} recalled (threshold={threshold})")

    # --- Load existing isis.json keys and test against live model ---
    isis_path = Path("isis.json")
    if isis_path.exists():
        print("\n=== isis.json Key Validation ===")
        bank = json.loads(isis_path.read_text())
        for alter in bank.get("alters", []):
            for ep in alter.get("episodes", []):
                prompt = ep["prompt"]
                # Find the PROMPT_END key
                stored_key = None
                for ck in ep["keys"]:
                    if ck["position"] == -1:
                        stored_key = np.array(ck["key"])
                        break
                if stored_key is None:
                    continue

                stored_key = stored_key / (np.linalg.norm(stored_key) + 1e-10)

                for dtype_name, model in [("f32", model_f32), ("f16", model_f16), ("bf16", model_bf16)]:
                    live_key = get_hidden_states(model, tokenizer, [prompt])[0]
                    sim = cosine_sim(stored_key, live_key)
                    status = "MATCH" if sim >= threshold else "MISS"
                    print(f"  [{status}] {dtype_name} \"{prompt:30s}\" sim={sim:.6f}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    f16_sims = [cosine_sim(keys_f32[i], keys_f16[i]) for i in range(len(prompts))]
    bf16_sims = [cosine_sim(keys_f32[i], keys_bf16[i]) for i in range(len(prompts))]
    print(f"  f32 vs f16:  min={min(f16_sims):.6f}  max={max(f16_sims):.6f}  mean={np.mean(f16_sims):.6f}")
    print(f"  f32 vs bf16: min={min(bf16_sims):.6f}  max={max(bf16_sims):.6f}  mean={np.mean(bf16_sims):.6f}")
    print()
    if min(f16_sims) > 0.95:
        print("  VERDICT: f16 keys are COMPATIBLE with f32 memory banks")
    elif min(f16_sims) > 0.70:
        print("  VERDICT: f16 keys MIGHT work with f32 memory banks (above threshold)")
    else:
        print("  VERDICT: f16 keys are INCOMPATIBLE — need separate memory banks per dtype")

    if min(bf16_sims) > 0.95:
        print("  VERDICT: bf16 keys are COMPATIBLE with f32 memory banks")
    elif min(bf16_sims) > 0.70:
        print("  VERDICT: bf16 keys MIGHT work with f32 memory banks (above threshold)")
    else:
        print("  VERDICT: bf16 keys are INCOMPATIBLE — need separate memory banks per dtype")


if __name__ == "__main__":
    main()
