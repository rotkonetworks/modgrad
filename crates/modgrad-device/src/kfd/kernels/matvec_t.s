// Transposed matvec: dx = W^T @ d_out (backward input gradient)
// W is [out_dim × in_dim] row-major. We compute:
//   dx[j] = sum_i W[i*in_dim + j] * d_out[i]   for j = 0..in_dim
//
// Each workgroup computes ONE output element dx[j].
// 256 threads cooperate on the dot product via LDS reduction
// (same pattern as matvec_tiled, but strided W access).
//
// kernarg layout (32 bytes):
//   +0x00: W pointer (u64) — [out_dim × in_dim] row-major f32
//   +0x08: d_out pointer (u64) — [out_dim] f32
//   +0x10: dx pointer (u64) — [in_dim] f32 output
//   +0x18: out_dim (u32) — reduction dimension (number of rows)
//   +0x1c: in_dim (u32) — output dimension (number of cols)
//
// dispatch: grid = [in_dim, 1, 1] workgroups, block = [256, 1, 1]
// LDS: 256 × 4 = 1024 bytes for reduction

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl matvec_t_tiled
.p2align 8
.type matvec_t_tiled, @function
matvec_t_tiled:
    // s[0:1] = kernarg ptr, s2 = wg_id (= col j), v0 = tid (0..255)
    s_mov_b32 s20, s2                    // save wg_id = j (output column)

    s_load_b64   s[2:3],   s[0:1], 0x00 // W ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // d_out ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // dx ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // out_dim | in_dim
    s_waitcnt    lgkmcnt(0)

    // s8 = out_dim (reduction dim), s9 = in_dim, s20 = j (column)
    // Bounds check: if j >= in_dim, skip
    s_cmp_ge_u32 s20, s9
    s_cbranch_scc1 .Ldone

    // ─── Phase 1: Partial dot product (strided by 256) ───
    // Each thread sums: W[tid + t*256][j] * d_out[tid + t*256]
    // W element at row i, col j: offset = i * in_dim + j
    v_mov_b32    v3, 0                   // partial_sum = 0
    s_mov_b32    s13, 0                  // tile counter

.Ltile:
    // i = tid + tile * 256
    s_lshl_b32  s14, s13, 8             // tile * 256
    v_add_nc_u32 v4, s14, v0            // v4 = i = tile*256 + tid

    // Bounds: i < out_dim?
    v_cmp_lt_u32 vcc_lo, v4, s8
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Ltile_skip

    // W[i * in_dim + j]: byte offset = (i * in_dim + j) * 4
    v_mul_lo_u32 v5, v4, s9             // i * in_dim
    v_add_nc_u32 v5, s20, v5            // i * in_dim + j (SGPR s20 in src0)
    v_lshlrev_b32 v5, 2, v5             // × 4
    global_load_b32 v6, v5, s[2:3]      // W[i][j]

    // d_out[i]: byte offset = i * 4
    v_lshlrev_b32 v7, 2, v4
    global_load_b32 v8, v7, s[4:5]      // d_out[i]

    s_waitcnt    vmcnt(0)
    v_fmac_f32   v3, v6, v8             // partial_sum += W[i][j] * d_out[i]

.Ltile_skip:
    s_mov_b32    exec_lo, s15

    // Next tile
    s_add_u32    s13, s13, 1
    s_lshl_b32  s14, s13, 8
    s_cmp_lt_u32 s14, s8                // next_base < out_dim?
    s_cbranch_scc1 .Ltile

    // ─── Phase 2: LDS reduction ───
    v_lshlrev_b32 v10, 2, v0            // LDS offset = tid * 4
    ds_write_b32 v10, v3
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s14, 128
.Lreduce:
    v_cmp_lt_u32 vcc_lo, v0, s14
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Lreduce_skip

    v_mov_b32    v5, s14
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32  v7, v6
    ds_read_b32  v8, v10
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt    lgkmcnt(0)

.Lreduce_skip:
    s_mov_b32    exec_lo, s15
    s_barrier
    s_lshr_b32  s14, s14, 1
    s_cmp_gt_u32 s14, 0
    s_cbranch_scc1 .Lreduce

    // ─── Phase 3: Thread 0 stores result ───
    v_cmp_eq_u32 vcc_lo, v0, 0
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Ldone

    v_mov_b32    v9, 0
    ds_read_b32  v3, v9
    s_waitcnt    lgkmcnt(0)

    // Store dx[j] (no bias)
    s_lshl_b32  s16, s20, 2             // j * 4
    v_mov_b32   v5, s16
    global_store_b32 v5, v3, s[6:7]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel matvec_t_tiled
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 11
    .amdhsa_next_free_sgpr 21
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
  - .name:            matvec_t_tiled
    .symbol:          matvec_t_tiled.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      11
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
      - { .size: 4, .offset: 28, .value_kind: by_value }
...
.end_amdgpu_metadata
