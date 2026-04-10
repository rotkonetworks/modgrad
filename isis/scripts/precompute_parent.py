"""Pre-compute parent hidden states for the curriculum.

Runs the Qwen backbone once on all curriculum text, saves hidden states
as a binary file. The child organism then trains fast without the backbone.

Usage: python precompute_parent.py [model_dir] [curriculum_file] [output_file]
"""

import sys
import numpy as np
import onnxruntime as ort

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    curriculum = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else "parent_states.bin"

    # Load backbone
    print(f"Loading backbone from {model_dir}...")
    backbone = ort.InferenceSession(f"{model_dir}/backbone.onnx")
    print(f"  Inputs: {[i.name for i in backbone.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in backbone.get_outputs()]}")

    # Load or generate curriculum
    if curriculum and curriculum != "":
        with open(curriculum, 'rb') as f:
            data = f.read()
        print(f"Loaded {len(data)} bytes from {curriculum}")
    else:
        # Generate staged curriculum inline
        from isis_curriculum import generate_all
        data = generate_all()
        print(f"Generated {len(data)} bytes of staged curriculum")

    # Process in chunks
    chunk_size = 128
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Processing {len(chunks)} chunks of {chunk_size} bytes...")

    all_hiddens = []
    for i, chunk in enumerate(chunks):
        # Convert bytes to token IDs
        ids = np.array([[b for b in chunk]], dtype=np.int64)

        # Run backbone
        hidden, full = backbone.run(None, {"input_ids": ids})
        # hidden shape: [1, seq_len, hidden_dim]
        all_hiddens.append(hidden[0])  # [seq_len, hidden_dim]

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(chunks)}] hidden_dim={hidden.shape[-1]}")

    # Save as binary: header + flat f32 data
    hidden_dim = all_hiddens[0].shape[-1]
    total_tokens = sum(h.shape[0] for h in all_hiddens)

    print(f"\nSaving {total_tokens} token states (dim={hidden_dim}) to {output}...")
    with open(output, 'wb') as f:
        # Header
        f.write(b"PRNT")  # magic
        f.write(np.uint32(1).tobytes())  # version
        f.write(np.uint32(hidden_dim).tobytes())
        f.write(np.uint32(len(chunks)).tobytes())
        f.write(np.uint32(chunk_size).tobytes())
        f.write(np.uint64(total_tokens).tobytes())

        # Per-chunk: [seq_len as u32] + [seq_len bytes (token IDs)] + [seq_len × hidden_dim as f32]
        for i, h in enumerate(all_hiddens):
            chunk = chunks[i]
            f.write(np.uint32(h.shape[0]).tobytes())
            # Store the raw bytes as token IDs
            f.write(bytes(chunk))
            # Pad to seq_len if chunk was shorter
            if len(chunk) < h.shape[0]:
                f.write(b'\x00' * (h.shape[0] - len(chunk)))
            f.write(h.astype(np.float32).tobytes())

    size_mb = total_tokens * hidden_dim * 4 / 1024 / 1024
    print(f"Done: {size_mb:.1f} MB ({total_tokens} tokens × {hidden_dim} dims)")


if __name__ == "__main__":
    main()
