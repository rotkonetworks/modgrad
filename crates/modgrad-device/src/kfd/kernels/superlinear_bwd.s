// SuperLinear backward: batched per-neuron gradient computation.
//
// Two kernels in one code object:
//
// 1. superlinear_bwd_dw: d_weight accumulation (outer product per neuron)
//    dW[n*O*K + o*K + k] += d_out[n*O + o] * input[n*K + k]
//    dispatch: ceil(N*O*K / 256) workgroups, 256 threads
//    Each thread handles one dW element.
//
// 2. superlinear_bwd_dx: d_input computation (transposed matvec per neuron)
//    dX[n*K + k] = sum_o d_out[n*O + o] * W[n*O*K + o*K + k]
//    dispatch: ceil(N*K / 256) workgroups, 256 threads
//    Each thread handles one dX element, looping over O.
//
// Shared kernarg layout (48 bytes):
//   +0x00: W pointer (u64) — [N*O*K] f32 (weights, read-only for bwd_dx)
//   +0x08: dW pointer (u64) — [N*O*K] f32 (gradient, accumulated for bwd_dw)
//   +0x10: d_out pointer (u64) — [N*O] f32
//   +0x18: input pointer (u64) — [N*K] f32 (cached forward input)
//   +0x20: dX pointer (u64) — [N*K] f32 (d_input output for bwd_dx)
//   +0x28: N (u32)
//   +0x2c: O (u32)
//   +0x30: K (u32)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

// ═══════════════════════════════════════════════════════════
// Kernel 1: d_weight accumulation
// ═══════════════════════════════════════════════════════════

.text
.globl superlinear_bwd_dw
.p2align 8
.type superlinear_bwd_dw, @function
superlinear_bwd_dw:
    s_mov_b32 s20, s2                    // save wg_id

    s_load_b64   s[2:3],   s[0:1], 0x08 // dW ptr
    s_load_b64   s[4:5],   s[0:1], 0x10 // d_out ptr
    s_load_b64   s[6:7],   s[0:1], 0x18 // input ptr
    s_load_b64   s[8:9],   s[0:1], 0x28 // N | O
    s_load_b32   s10,      s[0:1], 0x30 // K
    s_waitcnt    lgkmcnt(0)

    // s8=N, s9=O, s10=K
    // global_id = wg_id*256 + tid
    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // v0 = global_id

    // total = N*O*K
    s_mul_i32    s11, s8, s9            // N*O
    s_mul_i32    s12, s11, s10          // N*O*K
    v_cmp_lt_u32 vcc_lo, v0, s12
    s_and_saveexec_b32 s13, vcc_lo
    s_cbranch_execz .Ldw_done

    // Decompose global_id into (neuron, o, k):
    //   flat_idx = global_id
    //   neuron = flat_idx / (O*K)
    //   remainder = flat_idx % (O*K)
    //   o = remainder / K
    //   k = remainder % K

    // neuron = global_id / (O*K)
    s_mul_i32    s14, s9, s10           // O*K
    v_cvt_f32_u32 v1, v0               // float(global_id)
    v_cvt_f32_u32 v2, s14              // float(O*K)
    v_rcp_f32    v2, v2
    v_mul_f32    v3, v1, v2
    v_cvt_u32_f32 v3, v3               // neuron (approx)

    // remainder = global_id - neuron * O*K
    v_mul_lo_u32 v4, v3, s14
    v_sub_nc_u32 v4, v0, v4            // remainder

    // Fix rounding
    v_cmp_ge_u32 vcc_lo, v4, s14
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_subrev_nc_u32 v5, s14, v4
    v_cndmask_b32 v4, v4, v5, vcc_lo   // corrected remainder

    // o = remainder / K
    v_cvt_f32_u32 v5, v4
    v_cvt_f32_u32 v6, s10
    v_rcp_f32    v6, v6
    v_mul_f32    v7, v5, v6
    v_cvt_u32_f32 v7, v7               // o (approx)

    // k = remainder - o * K
    v_mul_lo_u32 v8, v7, s10
    v_sub_nc_u32 v8, v4, v8            // k

    // Fix rounding
    v_cmp_ge_u32 vcc_lo, v8, s10
    v_add_co_ci_u32 v7, vcc_lo, v7, 0, vcc_lo
    v_subrev_nc_u32 v9, s10, v8
    v_cndmask_b32 v8, v8, v9, vcc_lo   // corrected k

    // Now v3=neuron, v7=o, v8=k

    // d_out[neuron*O + o]
    v_mul_lo_u32 v10, v3, s9           // neuron*O
    v_add_nc_u32 v10, v10, v7          // neuron*O + o
    v_lshlrev_b32 v10, 2, v10
    global_load_b32 v11, v10, s[4:5]   // d_out[neuron*O + o]

    // input[neuron*K + k]
    v_mul_lo_u32 v12, v3, s10          // neuron*K
    v_add_nc_u32 v12, v12, v8          // neuron*K + k
    v_lshlrev_b32 v12, 2, v12
    global_load_b32 v13, v12, s[6:7]   // input[neuron*K + k]

    // dW[global_id] (current value)
    v_lshlrev_b32 v14, 2, v0
    global_load_b32 v15, v14, s[2:3]

    s_waitcnt    vmcnt(0)

    // dW += d_out * input
    v_fmac_f32   v15, v11, v13
    global_store_b32 v14, v15, s[2:3]

.Ldw_done:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// ═══════════════════════════════════════════════════════════
// Kernel 2: d_input computation (W^T @ d_out per neuron)
// ═══════════════════════════════════════════════════════════

.globl superlinear_bwd_dx
.p2align 8
.type superlinear_bwd_dx, @function
superlinear_bwd_dx:
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00 // W ptr
    s_load_b64   s[4:5],   s[0:1], 0x10 // d_out ptr
    s_load_b64   s[8:9],   s[0:1], 0x20 // dX ptr
    s_load_b64   s[10:11], s[0:1], 0x28 // N | O
    s_load_b32   s12,      s[0:1], 0x30 // K
    s_waitcnt    lgkmcnt(0)

    // s10=N, s11=O, s12=K
    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // global_id

    // total = N*K
    s_mul_i32    s13, s10, s12
    v_cmp_lt_u32 vcc_lo, v0, s13
    s_and_saveexec_b32 s14, vcc_lo
    s_cbranch_execz .Ldx_done

    // Decompose: neuron = global_id / K, k = global_id % K
    v_cvt_f32_u32 v1, v0
    v_cvt_f32_u32 v2, s12
    v_rcp_f32    v2, v2
    v_mul_f32    v3, v1, v2
    v_cvt_u32_f32 v3, v3               // neuron
    v_mul_lo_u32 v4, v3, s12
    v_sub_nc_u32 v4, v0, v4            // k
    // Fix rounding
    v_cmp_ge_u32 vcc_lo, v4, s12
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo
    v_subrev_nc_u32 v5, s12, v4
    v_cndmask_b32 v4, v4, v5, vcc_lo

    // v3=neuron, v4=k
    // dX[global_id] = sum_o d_out[neuron*O + o] * W[neuron*O*K + o*K + k]

    // Base addresses
    s_mul_i32    s15, s11, s12          // O*K
    v_mul_lo_u32 v5, v3, s15           // neuron*O*K (W base for this neuron)
    v_add_nc_u32 v5, v5, v4            // + k → first W element for this (neuron, k)

    v_mul_lo_u32 v6, v3, s11           // neuron*O (d_out base)

    v_mov_b32    v10, 0                 // accumulator = 0
    s_mov_b32    s16, 0                 // o = 0

.Ldx_loop:
    s_cmp_ge_u32 s16, s11              // o >= O?
    s_cbranch_scc1 .Ldx_store

    // W[neuron*O*K + o*K + k] = W at offset (neuron*O*K + k) + o*K
    // v5 = neuron*O*K + k (base), need to add o*K
    s_mul_i32    s17, s16, s12          // o*K
    v_add_nc_u32 v7, s17, v5           // neuron*O*K + o*K + k
    v_lshlrev_b32 v7, 2, v7
    global_load_b32 v8, v7, s[2:3]     // W[neuron*O*K + o*K + k]

    // d_out[neuron*O + o]
    v_add_nc_u32 v9, s16, v6           // neuron*O + o
    v_lshlrev_b32 v9, 2, v9
    global_load_b32 v11, v9, s[4:5]    // d_out[neuron*O + o]

    s_waitcnt    vmcnt(0)
    v_fmac_f32   v10, v8, v11          // acc += W * d_out

    s_add_u32    s16, s16, 1
    s_branch     .Ldx_loop

.Ldx_store:
    // Store dX[global_id]
    v_lshlrev_b32 v12, 2, v0
    global_store_b32 v12, v10, s[8:9]

.Ldx_done:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// ═══════════════════════════════════════════════════════════
// Kernel descriptors
// ═══════════════════════════════════════════════════════════

.rodata

.p2align 6
.amdhsa_kernel superlinear_bwd_dw
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 52
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 22
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel superlinear_bwd_dx
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 52
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 13
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
  - .name:            superlinear_bwd_dw
    .symbol:          superlinear_bwd_dw.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      16
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 40, .value_kind: by_value }
      - { .size: 4, .offset: 44, .value_kind: by_value }
      - { .size: 4, .offset: 48, .value_kind: by_value }
  - .name:            superlinear_bwd_dx
    .symbol:          superlinear_bwd_dx.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      13
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 40, .value_kind: by_value }
      - { .size: 4, .offset: 44, .value_kind: by_value }
      - { .size: 4, .offset: 48, .value_kind: by_value }
...
.end_amdgpu_metadata
