// Affine LayerNorm backward:
//   d_gamma[i] += d_out[i] * normalized[i]
//   d_beta[i] += d_out[i]
//   d_norm[i] = d_out[i] * gamma[i]
//   d_input[i] = inv_std * (d_norm[i] - mean(d_norm) - normalized[i] * mean(d_norm * normalized))
//
// Single-workgroup kernel with LDS reduction for mean computations.
// Handles n <= 256 (1 element per thread, single wave safe).
//
// kernarg layout (56 bytes):
//   +0x00: d_out pointer (u64) — [N] f32
//   +0x08: normalized pointer (u64) — [N] f32 (cached from forward)
//   +0x10: gamma pointer (u64) — [N] f32
//   +0x18: d_gamma pointer (u64) — [N] f32 (accumulated)
//   +0x20: d_beta pointer (u64) — [N] f32 (accumulated)
//   +0x28: d_input pointer (u64) — [N] f32 (output)
//   +0x30: inv_std (f32) — cached from forward
//   +0x34: N (u32)
//
// dispatch: grid = [1, 1, 1], block = [256, 1, 1]
// LDS: 256 * 4 = 1024 bytes

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl ln_bwd
.p2align 8
.type ln_bwd, @function
ln_bwd:
    s_load_b64   s[2:3],   s[0:1], 0x00 // d_out ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // normalized ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // gamma ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // d_gamma ptr
    s_load_b64   s[10:11], s[0:1], 0x20 // d_beta ptr
    s_load_b64   s[12:13], s[0:1], 0x28 // d_input ptr
    s_load_b64   s[14:15], s[0:1], 0x30 // inv_std(f32) | N(u32)
    s_waitcnt    lgkmcnt(0)

    // s14 = inv_std (as u32 bits), s15 = N
    // tid = v0

    v_lshlrev_b32 v10, 2, v0            // LDS offset = tid * 4
    v_lshlrev_b32 v1, 2, v0             // byte offset = tid * 4

    // ─── Load d_out[tid], normalized[tid], gamma[tid] ───
    v_mov_b32    v20, 0                  // d_out (default 0 if tid >= N)
    v_mov_b32    v21, 0                  // normalized
    v_mov_b32    v22, 0                  // gamma

    v_cmp_lt_u32 vcc_lo, v0, s15
    s_and_saveexec_b32 s16, vcc_lo
    s_cbranch_execz .Lload_done

    global_load_b32 v20, v1, s[2:3]     // d_out[tid]
    global_load_b32 v21, v1, s[4:5]     // normalized[tid]
    global_load_b32 v22, v1, s[6:7]     // gamma[tid]
    s_waitcnt    vmcnt(0)

    // Accumulate d_gamma[tid] += d_out * normalized
    global_load_b32 v23, v1, s[8:9]     // current d_gamma[tid]
    s_waitcnt    vmcnt(0)
    v_fmac_f32   v23, v20, v21          // d_gamma += d_out * normalized
    global_store_b32 v1, v23, s[8:9]

    // Accumulate d_beta[tid] += d_out
    global_load_b32 v24, v1, s[10:11]   // current d_beta[tid]
    s_waitcnt    vmcnt(0)
    v_add_f32    v24, v24, v20
    global_store_b32 v1, v24, s[10:11]

.Lload_done:
    s_mov_b32    exec_lo, s16

    // d_norm = d_out * gamma
    v_mul_f32    v25, v20, v22          // d_norm[tid]

    // ─── Reduction 1: mean(d_norm) ───
    ds_write_b32 v10, v25
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s17, 128
.Lr1:
    v_cmp_lt_u32 vcc_lo, v0, s17
    s_and_saveexec_b32 s18, vcc_lo
    s_cbranch_execz .Lr1_skip
    v_mov_b32    v5, s17
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32  v7, v6
    ds_read_b32  v8, v10
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt    lgkmcnt(0)
.Lr1_skip:
    s_mov_b32    exec_lo, s18
    s_barrier
    s_lshr_b32  s17, s17, 1
    s_cmp_gt_u32 s17, 0
    s_cbranch_scc1 .Lr1

    // mean_dn = LDS[0] / N
    v_mov_b32    v9, 0
    ds_read_b32  v26, v9                 // sum of d_norm
    s_waitcnt    lgkmcnt(0)
    v_cvt_f32_u32 v27, s15
    v_rcp_f32    v27, v27
    v_mul_f32    v26, v26, v27          // v26 = mean(d_norm)

    // ─── Reduction 2: mean(d_norm * normalized) ───
    v_mul_f32    v28, v25, v21          // d_norm * normalized
    ds_write_b32 v10, v28
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s17, 128
.Lr2:
    v_cmp_lt_u32 vcc_lo, v0, s17
    s_and_saveexec_b32 s18, vcc_lo
    s_cbranch_execz .Lr2_skip
    v_mov_b32    v5, s17
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32  v7, v6
    ds_read_b32  v8, v10
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt    lgkmcnt(0)
.Lr2_skip:
    s_mov_b32    exec_lo, s18
    s_barrier
    s_lshr_b32  s17, s17, 1
    s_cmp_gt_u32 s17, 0
    s_cbranch_scc1 .Lr2

    // mean_dn_xhat = LDS[0] / N
    ds_read_b32  v29, v9
    s_waitcnt    lgkmcnt(0)
    v_mul_f32    v29, v29, v27          // v29 = mean(d_norm * normalized)

    // ─── Compute d_input ───
    // d_input[i] = inv_std * (d_norm[i] - mean_dn - normalized[i] * mean_dn_xhat)
    v_cmp_lt_u32 vcc_lo, v0, s15
    s_and_saveexec_b32 s18, vcc_lo
    s_cbranch_execz .Ldone

    v_sub_f32    v30, v25, v26          // d_norm - mean_dn
    v_mul_f32    v31, v21, v29          // normalized * mean_dn_xhat
    v_sub_f32    v30, v30, v31          // d_norm - mean_dn - normalized * mean_dn_xhat
    v_mov_b32    v31, s14               // inv_std
    v_mul_f32    v30, v31, v30          // inv_std * (...)

    global_store_b32 v1, v30, s[12:13]  // d_input[tid]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel ln_bwd
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 56
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 19
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
  - .name:            ln_bwd
    .symbol:          ln_bwd.kd
    .kernarg_segment_size: 56
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      19
    .vgpr_count:      32
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 40, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 48, .value_kind: by_value }
      - { .size: 4, .offset: 52, .value_kind: by_value }
...
.end_amdgpu_metadata
