// Matrix-vector multiply: y = W*x + b
// W is [out_dim x in_dim] row-major
// Each workgroup computes one output element
__kernel void matvec(
    __global const float* W,
    __global const float* b,
    __global const float* x,
    __global float* y,
    uint out_dim,
    uint in_dim
) {
    uint row = get_global_id(0);
    if (row >= out_dim) return;

    float sum = b[row];
    __global const float* w_row = W + row * in_dim;

    // Vectorized inner loop
    uint i = 0;
    for (; i + 4 <= in_dim; i += 4) {
        sum += w_row[i]   * x[i];
        sum += w_row[i+1] * x[i+1];
        sum += w_row[i+2] * x[i+2];
        sum += w_row[i+3] * x[i+3];
    }
    for (; i < in_dim; i++) {
        sum += w_row[i] * x[i];
    }

    y[row] = sum;
}

// Fused synapse: matvec → GLU → SiLU → layer_norm
// W is [out_dim*2 x in_dim], produces out_dim outputs
// scratch is [out_dim*2] temp space
__kernel void synapse_fused(
    __global const float* W,
    __global const float* b,
    __global const float* x,
    __global float* output,
    __global float* scratch,
    uint out_dim,
    uint in_dim
) {
    uint row = get_global_id(0);
    uint total_rows = out_dim * 2;
    if (row >= total_rows) return;

    // Step 1: matvec into scratch
    float sum = b[row];
    __global const float* w_row = W + row * in_dim;
    for (uint i = 0; i < in_dim; i++) {
        sum += w_row[i] * x[i];
    }
    scratch[row] = sum;

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Only first out_dim threads continue for GLU + SiLU
    if (row >= out_dim) return;

    // Step 2: GLU — output[i] = scratch[i] * sigmoid(scratch[i + out_dim])
    float val = scratch[row];
    float gate = 1.0f / (1.0f + exp(-scratch[row + out_dim]));
    float glu_out = val * gate;

    // Step 3: SiLU — x * sigmoid(x)
    float silu_out = glu_out / (1.0f + exp(-glu_out));

    output[row] = silu_out;
}

// GLU activation
__kernel void glu(
    __global const float* input,
    __global float* output,
    uint half_dim
) {
    uint i = get_global_id(0);
    if (i >= half_dim) return;
    float gate = 1.0f / (1.0f + exp(-input[i + half_dim]));
    output[i] = input[i] * gate;
}

// SiLU (swish) in-place
__kernel void silu(
    __global float* x,
    uint n
) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + exp(-v));
}

// Elementwise add: y[i] += alpha * x[i]
__kernel void axpy(
    __global float* y,
    __global const float* x,
    float alpha,
    uint n
) {
    uint i = get_global_id(0);
    if (i >= n) return;
    y[i] += alpha * x[i];
}
