// Layer normalization in-place: x[i] = (x[i] - mean) * rsqrt(var + eps)
//
// Single-workgroup kernel using LDS for parallel reduction.
// Handles n <= 1024 (up to 4 elements per thread with 256 threads).
// eps = 1e-5
//
// kernarg layout (16 bytes):
//   +0x00: x pointer (u64) — [n] f32, read and written in-place
//   +0x08: n (u32)
//
// dispatch: grid = [256, 1, 1], block = [256, 1, 1]  (single workgroup!)
// LDS: 256 * 4 = 1024 bytes for reduction scratch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl layer_norm_fwd
.p2align 8
.type layer_norm_fwd, @function
layer_norm_fwd:
    // s[0:1] = kernarg ptr, s2 = wg_id (always 0), v0 = local_id = tid
    s_load_b64  s[2:3], s[0:1], 0x00    // x ptr
    s_load_b32  s4,     s[0:1], 0x08    // n
    s_waitcnt   lgkmcnt(0)

    // tid = v0 (0..255), already our global_id since single WG
    // LDS offset for this thread: tid * 4
    v_lshlrev_b32 v10, 2, v0            // v10 = tid * 4 (LDS byte offset)

    // ─── Phase 1: Load elements, compute partial sum for mean ───
    // Each thread processes elements at indices: tid, tid+256, tid+512, tid+768
    // Cache loaded values in v20-v23, use v24 for partial sum

    v_mov_b32 v24, 0                     // partial sum = 0
    v_mov_b32 v20, 0                     // cached x[tid]
    v_mov_b32 v21, 0                     // cached x[tid+256]
    v_mov_b32 v22, 0                     // cached x[tid+512]
    v_mov_b32 v23, 0                     // cached x[tid+768]

    // Element 0: tid
    v_cmp_lt_u32 vcc_lo, v0, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lload1
    v_lshlrev_b32 v1, 2, v0
    global_load_b32 v20, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v20
.Lload1:
    s_mov_b32 exec_lo, s10

    // Element 1: tid + 256
    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lload2
    v_lshlrev_b32 v1, 2, v2
    global_load_b32 v21, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v21
.Lload2:
    s_mov_b32 exec_lo, s10

    // Element 2: tid + 512
    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lload3
    v_lshlrev_b32 v1, 2, v3
    global_load_b32 v22, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v22
.Lload3:
    s_mov_b32 exec_lo, s10

    // Element 3: tid + 768
    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lload4
    v_lshlrev_b32 v1, 2, v4
    global_load_b32 v23, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v23
.Lload4:
    s_mov_b32 exec_lo, s10

    // ─── Reduce partial sums for mean ───
    // Store partial sum to LDS[tid]
    ds_write_b32 v10, v24
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Tree reduction: stride 128, 64, 32, 16, 8, 4, 2, 1
    s_mov_b32 s11, 128
.Lmean_reduce:
    v_cmp_lt_u32 vcc_lo, v0, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lmean_skip

    // partner LDS offset = (tid + stride) * 4
    v_mov_b32 v5, s11
    v_lshlrev_b32 v5, 2, v5             // stride * 4
    v_add_nc_u32 v6, v10, v5            // (tid + stride) * 4

    ds_read_b32 v7, v6                   // partner value
    ds_read_b32 v8, v10                  // own value
    s_waitcnt lgkmcnt(0)
    v_add_f32 v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt lgkmcnt(0)

.Lmean_skip:
    s_mov_b32 exec_lo, s12
    s_barrier
    s_lshr_b32 s11, s11, 1
    s_cmp_gt_u32 s11, 0
    s_cbranch_scc1 .Lmean_reduce

    // Broadcast mean: LDS[0] / n
    v_mov_b32 v9, 0                      // LDS offset 0
    ds_read_b32 v25, v9                  // v25 = total sum
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_u32 v26, s4               // v26 = float(n)
    v_rcp_f32 v26, v26                   // v26 = 1/n
    v_mul_f32 v25, v25, v26             // v25 = mean

    // ─── Phase 2: Compute (x[i] - mean)^2 partial sums for variance ───
    v_mov_b32 v24, 0                     // partial var sum

    // Element 0
    v_cmp_lt_u32 vcc_lo, v0, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lvar1
    v_sub_f32 v20, v20, v25             // x[i] - mean (also update cached value)
    v_mul_f32 v1, v20, v20              // (x[i] - mean)^2
    v_add_f32 v24, v24, v1
.Lvar1:
    s_mov_b32 exec_lo, s10

    // Element 1
    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lvar2
    v_sub_f32 v21, v21, v25
    v_mul_f32 v1, v21, v21
    v_add_f32 v24, v24, v1
.Lvar2:
    s_mov_b32 exec_lo, s10

    // Element 2
    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lvar3
    v_sub_f32 v22, v22, v25
    v_mul_f32 v1, v22, v22
    v_add_f32 v24, v24, v1
.Lvar3:
    s_mov_b32 exec_lo, s10

    // Element 3
    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lvar4
    v_sub_f32 v23, v23, v25
    v_mul_f32 v1, v23, v23
    v_add_f32 v24, v24, v1
.Lvar4:
    s_mov_b32 exec_lo, s10

    // ─── Reduce variance partial sums ───
    ds_write_b32 v10, v24
    s_waitcnt lgkmcnt(0)
    s_barrier

    s_mov_b32 s11, 128
.Lvar_reduce:
    v_cmp_lt_u32 vcc_lo, v0, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lvar_skip

    v_mov_b32 v5, s11
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5

    ds_read_b32 v7, v6
    ds_read_b32 v8, v10
    s_waitcnt lgkmcnt(0)
    v_add_f32 v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt lgkmcnt(0)

.Lvar_skip:
    s_mov_b32 exec_lo, s12
    s_barrier
    s_lshr_b32 s11, s11, 1
    s_cmp_gt_u32 s11, 0
    s_cbranch_scc1 .Lvar_reduce

    // Compute inv_std = rsqrt(var/n + eps)
    v_mov_b32 v9, 0
    ds_read_b32 v26, v9                  // v26 = total var sum
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_u32 v27, s4
    v_rcp_f32 v27, v27                   // 1/n
    v_mul_f32 v26, v26, v27             // var = sum / n
    v_add_f32 v26, 0x3727c5ac, v26      // var + 1e-5  (eps = 0x3727c5ac)
    v_rsq_f32 v26, v26                   // inv_std = rsqrt(var + eps)

    // ─── Phase 3: Normalize and store ───
    // v20-v23 already hold (x[i] - mean), just multiply by inv_std

    // Element 0
    v_cmp_lt_u32 vcc_lo, v0, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lstore1
    v_mul_f32 v20, v20, v26
    v_lshlrev_b32 v1, 2, v0
    global_store_b32 v1, v20, s[2:3]
.Lstore1:
    s_mov_b32 exec_lo, s10

    // Element 1
    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lstore2
    v_mul_f32 v21, v21, v26
    v_lshlrev_b32 v1, 2, v2
    global_store_b32 v1, v21, s[2:3]
.Lstore2:
    s_mov_b32 exec_lo, s10

    // Element 2
    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lstore3
    v_mul_f32 v22, v22, v26
    v_lshlrev_b32 v1, 2, v3
    global_store_b32 v1, v22, s[2:3]
.Lstore3:
    s_mov_b32 exec_lo, s10

    // Element 3
    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s4
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lstore4
    v_mul_f32 v23, v23, v26
    v_lshlrev_b32 v1, 2, v4
    global_store_b32 v1, v23, s[2:3]
.Lstore4:
    s_mov_b32 exec_lo, s10

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel layer_norm_fwd
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 28
    .amdhsa_next_free_sgpr 13
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
  - .name:            layer_norm_fwd
    .symbol:          layer_norm_fwd.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      13
    .vgpr_count:      28
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8,  .value_kind: by_value }
...
.end_amdgpu_metadata
