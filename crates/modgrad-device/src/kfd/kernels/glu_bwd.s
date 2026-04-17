// GLU backward: given d_out[i], cached input x[i] and x[n+i]:
//   d_val[i] = d_out[i] * sigmoid(x[n+i])
//   d_gate[i] = d_out[i] * x[i] * sigmoid(x[n+i]) * (1 - sigmoid(x[n+i]))
//
// Writes d_input[i] = d_val[i], d_input[n+i] = d_gate[i]
// This handles the standard GLU (not per-neuron — same kernel works for both
// since per-neuron GLU just has different n per neuron, but we flatten it).
//
// kernarg layout (28 bytes):
//   +0x00: d_out pointer (u64) — [N] f32
//   +0x08: cached_input pointer (u64) — [2*N] f32 (val | gate)
//   +0x10: d_input pointer (u64) — [2*N] f32 output
//   +0x18: N (u32) — half-size (output elements)
//
// dispatch: grid = ceil(N/256), block = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl glu_bwd
.p2align 8
.type glu_bwd, @function
glu_bwd:
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00 // d_out ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // cached_input ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // d_input ptr
    s_load_b32   s8,       s[0:1], 0x18 // N
    s_waitcnt    lgkmcnt(0)

    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // global_id = i

    v_cmp_lt_u32 vcc_lo, v0, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ldone

    v_lshlrev_b32 v1, 2, v0             // i * 4

    // Load d_out[i], input[i] (val), input[N+i] (gate_val)
    global_load_b32 v2, v1, s[2:3]      // d_out[i]
    global_load_b32 v3, v1, s[4:5]      // input[i] = val

    // input[N+i]: offset = (N + i) * 4 = N*4 + i*4
    s_lshl_b32  s11, s8, 2              // N * 4
    v_add_nc_u32 v8, s11, v1            // (N+i) * 4
    global_load_b32 v4, v8, s[4:5]      // input[N+i] = gate_val

    s_waitcnt    vmcnt(0)

    // sigmoid(gate_val)
    v_xor_b32   v5, 0x80000000, v4      // -gate_val
    v_mul_f32    v5, 0x3FB8AA3B, v5     // -gate_val * log2(e)
    v_exp_f32    v5, v5                  // exp(-gate_val)
    v_add_f32    v5, 1.0, v5            // 1 + exp(-gate_val)
    v_rcp_f32    v5, v5                  // v5 = sigmoid(gate_val)

    // d_val = d_out * sigmoid
    v_mul_f32    v6, v2, v5             // d_val

    // d_gate = d_out * val * sigmoid * (1 - sigmoid)
    v_sub_f32    v7, 1.0, v5            // 1 - sigmoid
    v_mul_f32    v7, v5, v7             // sigmoid * (1 - sigmoid)
    v_mul_f32    v7, v3, v7             // val * sigmoid * (1 - sigmoid)
    v_mul_f32    v7, v2, v7             // d_out * val * sigmoid * (1 - sigmoid)

    // Store d_input[i] = d_val, d_input[N+i] = d_gate
    global_store_b32 v1, v6, s[6:7]     // d_input[i]
    global_store_b32 v8, v7, s[6:7]     // d_input[N+i]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel glu_bwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 28
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 9
    .amdhsa_next_free_sgpr 22
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            glu_bwd
    .symbol:          glu_bwd.kd
    .kernarg_segment_size: 28
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      9
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
...
.end_amdgpu_metadata
