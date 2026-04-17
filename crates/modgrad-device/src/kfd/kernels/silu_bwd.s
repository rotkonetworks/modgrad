// SiLU backward: d_input[i] = d_out[i] * (s + x * s * (1 - s))
// where s = sigmoid(x[i]) = 1 / (1 + exp(-x[i]))
//
// kernarg layout (24 bytes):
//   +0x00: d_out pointer (u64) — [N] f32 (upstream gradient)
//   +0x08: pre_silu pointer (u64) — [N] f32 (cached pre-activation x)
//   +0x10: d_input pointer (u64) — [N] f32 (output gradient)
//   +0x18: N (u32)
//
// dispatch: grid = ceil(N/256), block = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl silu_bwd
.p2align 8
.type silu_bwd, @function
silu_bwd:
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00 // d_out ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // pre_silu ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // d_input ptr
    s_load_b32   s8,       s[0:1], 0x18 // N
    s_waitcnt    lgkmcnt(0)

    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // global_id

    v_cmp_lt_u32 vcc_lo, v0, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ldone

    v_lshlrev_b32 v1, 2, v0             // byte offset

    // Load d_out[i] and x[i]
    global_load_b32 v2, v1, s[2:3]      // d_out[i]
    global_load_b32 v3, v1, s[4:5]      // x[i] (pre-activation)
    s_waitcnt    vmcnt(0)

    // sigmoid(x) = 1 / (1 + exp(-x))
    // -x
    v_xor_b32   v4, 0x80000000, v3      // -x (flip sign)
    // exp(-x): use v_exp_f32 which computes 2^x, so exp(y) = 2^(y * log2(e))
    // log2(e) = 1.4426950408... ≈ 0x3FB8AA3B
    v_mul_f32    v4, 0x3FB8AA3B, v4     // -x * log2(e)
    v_exp_f32    v4, v4                  // 2^(-x * log2(e)) = exp(-x)
    // 1 + exp(-x)
    v_add_f32    v4, 1.0, v4            // 1 + exp(-x)
    // sigmoid = 1 / (1 + exp(-x))
    v_rcp_f32    v4, v4                  // v4 = sigmoid(x)

    // d_silu = s + x * s * (1 - s)
    v_sub_f32    v5, 1.0, v4            // 1 - s
    v_mul_f32    v5, v4, v5             // s * (1 - s)
    v_fmac_f32   v4, v3, v5            // s + x * s * (1 - s)

    // d_input = d_out * d_silu
    v_mul_f32    v6, v2, v4

    global_store_b32 v1, v6, s[6:7]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel silu_bwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 28
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 7
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
  - .name:            silu_bwd
    .symbol:          silu_bwd.kd
    .kernarg_segment_size: 28
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      7
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
...
.end_amdgpu_metadata
