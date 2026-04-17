// L2 norm squared reduction: result = sum(x[i]^2) for i in 0..N
// Used for gradient clipping: compute norm of all gradients.
//
// Two-pass approach:
//   Pass 1: each WG reduces 256 elements → partial_sums[wg_id]
//   Pass 2: reduce partial_sums → final result
// For simplicity, this kernel does pass 1 only.
// The caller does pass 2 on CPU (typically <1000 partial sums).
//
// kernarg layout (24 bytes):
//   +0x00: x pointer (u64) — [N] f32 input
//   +0x08: partial_sums pointer (u64) — [ceil(N/256)] f32 output
//   +0x10: N (u32)
//
// dispatch: grid = ceil(N/256), block = 256
// LDS: 256 × 4 = 1024 bytes

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl reduce_l2_sq
.p2align 8
.type reduce_l2_sq, @function
reduce_l2_sq:
    s_mov_b32 s20, s2                    // wg_id

    s_load_b64   s[2:3],   s[0:1], 0x00 // x ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // partial_sums ptr
    s_load_b32   s6,       s[0:1], 0x10 // N
    s_waitcnt    lgkmcnt(0)

    // global_id
    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0

    // Load x[global_id], compute x^2
    v_mov_b32    v3, 0                   // partial = 0
    v_cmp_lt_u32 vcc_lo, v0, s6
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lreduce

    v_lshlrev_b32 v1, 2, v0
    global_load_b32 v2, v1, s[2:3]
    s_waitcnt    vmcnt(0)
    v_mul_f32    v3, v2, v2             // x^2

.Lreduce:
    s_mov_b32    exec_lo, s10

    // LDS reduction
    v_lshlrev_b32 v10, 2, v0
    // Use local tid for LDS (v0 - wg_id*256)
    v_sub_nc_u32 v11, v0, s21          // local_tid
    v_lshlrev_b32 v10, 2, v11          // LDS offset = local_tid * 4
    ds_write_b32 v10, v3
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s11, 128
.Lr:
    v_cmp_lt_u32 vcc_lo, v11, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lr_skip
    v_mov_b32    v5, s11
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32  v7, v6
    ds_read_b32  v8, v10
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt    lgkmcnt(0)
.Lr_skip:
    s_mov_b32    exec_lo, s12
    s_barrier
    s_lshr_b32  s11, s11, 1
    s_cmp_gt_u32 s11, 0
    s_cbranch_scc1 .Lr

    // Thread 0 of WG stores partial sum
    v_cmp_eq_u32 vcc_lo, v11, 0
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Ldone

    v_mov_b32    v9, 0
    ds_read_b32  v3, v9
    s_waitcnt    lgkmcnt(0)

    // Store partial_sums[wg_id]
    s_lshl_b32  s16, s20, 2
    v_mov_b32   v5, s16
    global_store_b32 v5, v3, s[4:5]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel reduce_l2_sq
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 20
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 12
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
  - .name:            reduce_l2_sq
    .symbol:          reduce_l2_sq.kd
    .kernarg_segment_size: 20
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      12
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 16, .value_kind: by_value }
...
.end_amdgpu_metadata
