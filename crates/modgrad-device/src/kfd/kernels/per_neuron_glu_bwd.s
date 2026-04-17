// Per-neuron GLU backward:
//   For each neuron n, output index j (0..half):
//     val = input[n*out_per + j]
//     gate_v = input[n*out_per + half + j]
//     s = sigmoid(gate_v)
//     d_val = d_out[n*half + j] * s
//     d_gate = d_out[n*half + j] * val * s * (1 - s)
//     d_input[n*out_per + j] = d_val
//     d_input[n*out_per + half + j] = d_gate
//
// kernarg layout (32 bytes):
//   +0x00: d_out pointer (u64) — [N*half] f32
//   +0x08: cached_input pointer (u64) — [N*out_per] f32
//   +0x10: d_input pointer (u64) — [N*out_per] f32 output
//   +0x18: N (u32) — number of neurons
//   +0x1c: out_per (u32) — elements per neuron (val+gate)
//
// dispatch: grid = ceil(N*half/256), block = 256
// half = out_per / 2

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl per_neuron_glu_bwd
.p2align 8
.type per_neuron_glu_bwd, @function
per_neuron_glu_bwd:
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00 // d_out ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // cached_input ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // d_input ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // N | out_per
    s_waitcnt    lgkmcnt(0)

    // s8 = N, s9 = out_per
    s_lshr_b32  s10, s9, 1              // half = out_per / 2

    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // global_id

    // total output elements = N * half
    s_mul_i32    s11, s8, s10           // N * half
    v_cmp_lt_u32 vcc_lo, v0, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Ldone

    // Decompose global_id → (neuron, j)
    // neuron = global_id / half, j = global_id % half
    v_cvt_f32_u32 v1, v0
    v_cvt_f32_u32 v2, s10
    v_rcp_f32    v2, v2
    v_mul_f32    v3, v1, v2
    v_cvt_u32_f32 v3, v3               // neuron (approx)
    v_mul_lo_u32 v4, v3, s10
    v_sub_nc_u32 v4, v0, v4            // j = global_id - neuron * half
    // Fix rounding
    v_cmp_ge_u32 vcc_lo, v4, s10
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_subrev_nc_u32 v5, s10, v4
    v_cndmask_b32 v4, v4, v5, vcc_lo   // corrected j

    // d_out[neuron*half + j] — byte offset = global_id * 4
    v_lshlrev_b32 v10, 2, v0
    global_load_b32 v11, v10, s[2:3]    // d_out

    // input[neuron*out_per + j] = val
    v_mul_lo_u32 v12, v3, s9           // neuron * out_per
    v_add_nc_u32 v13, v12, v4          // neuron*out_per + j
    v_lshlrev_b32 v14, 2, v13
    global_load_b32 v15, v14, s[4:5]   // val

    // input[neuron*out_per + half + j] = gate_val
    v_add_nc_u32 v16, v13, s10         // neuron*out_per + half + j
    v_lshlrev_b32 v17, 2, v16
    global_load_b32 v18, v17, s[4:5]   // gate_val

    s_waitcnt    vmcnt(0)

    // sigmoid(gate_val)
    v_xor_b32   v19, 0x80000000, v18   // -gate_val
    v_mul_f32    v19, 0x3FB8AA3B, v19  // * log2(e)
    v_exp_f32    v19, v19               // exp(-gate_val)
    v_add_f32    v19, 1.0, v19         // 1 + exp(-gate_val)
    v_rcp_f32    v19, v19               // sigmoid

    // d_val = d_out * sigmoid
    v_mul_f32    v20, v11, v19

    // d_gate = d_out * val * sigmoid * (1 - sigmoid)
    v_sub_f32    v21, 1.0, v19         // 1 - sigmoid
    v_mul_f32    v21, v19, v21         // sigmoid * (1 - sigmoid)
    v_mul_f32    v21, v15, v21         // val * sigmoid * (1 - sigmoid)
    v_mul_f32    v21, v11, v21         // d_out * val * sigmoid * (1 - sigmoid)

    // Store d_input[neuron*out_per + j] = d_val
    global_store_b32 v14, v20, s[6:7]
    // Store d_input[neuron*out_per + half + j] = d_gate
    global_store_b32 v17, v21, s[6:7]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel per_neuron_glu_bwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 22
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
  - .name:            per_neuron_glu_bwd
    .symbol:          per_neuron_glu_bwd.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      22
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
      - { .size: 4, .offset: 28, .value_kind: by_value }
...
.end_amdgpu_metadata
