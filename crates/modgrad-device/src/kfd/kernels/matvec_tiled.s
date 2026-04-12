// Tiled matvec: y = W*x + b
// Each workgroup computes ONE output row. 256 threads cooperate on
// the dot product via LDS reduction. This saturates GPU memory
// bandwidth even for small output dimensions.
//
// vs naive matvec (1 thread/row): 128-row matvec uses 128 threads.
// Tiled: 128 WGs × 256 threads = 32K threads → full GPU utilization.
//
// kernarg layout (40 bytes, same as matvec):
//   +0x00: W pointer (u64) — [out_dim × in_dim] row-major f32
//   +0x08: b pointer (u64) — [out_dim] f32
//   +0x10: x pointer (u64) — [in_dim] f32
//   +0x18: y pointer (u64) — [out_dim] f32
//   +0x20: out_dim (u32)
//   +0x24: in_dim (u32)
//
// dispatch: grid = [out_dim, 1, 1] workgroups, block = [256, 1, 1]
// LDS: 256 × 4 = 1024 bytes for reduction

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl matvec_tiled
.p2align 8
.type matvec_tiled, @function
matvec_tiled:
    // s[0:1] = kernarg ptr, s2 = wg_id (= row), v0 = tid (0..255)
    s_mov_b32 s20, s2                    // save wg_id = row

    s_load_b64   s[2:3],   s[0:1], 0x00 // W ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // b ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // x ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // y ptr
    s_load_b64   s[10:11], s[0:1], 0x20 // out_dim | in_dim
    s_waitcnt    lgkmcnt(0)

    // s10 = out_dim, s11 = in_dim, s20 = row
    // Bounds check: if row >= out_dim, skip entire WG
    s_cmp_ge_u32 s20, s10
    s_cbranch_scc1 .Ldone

    // W row base byte offset = row × in_dim × 4
    s_mul_i32    s21, s20, s11           // row × in_dim
    s_lshl_b32  s21, s21, 2             // × 4 bytes

    // ─── Phase 1: Partial dot product (strided by 256) ───
    // Each thread accumulates: sum += W[row][tid+t*256] × x[tid+t*256]
    // for t = 0, 1, 2, ... while index < in_dim
    v_mov_b32    v3, 0                   // v3 = partial_sum = 0.0
    s_mov_b32    s13, 0                  // tile counter

.Ltile:
    // i = tid + tile × 256
    s_lshl_b32  s14, s13, 8             // tile × 256
    v_add_nc_u32 v4, v0, s14            // v4 = i

    // Bounds: if smallest i in next tile >= in_dim, this is last tile
    // But first process current tile (some threads may be out of bounds)
    v_cmp_lt_u32 vcc_lo, v4, s11
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Ltile_skip

    // Byte offset into W row: s21 (row base) + i×4
    v_lshlrev_b32 v5, 2, v4             // i × 4
    v_add_nc_u32 v5, v5, s21            // row_base + i×4
    global_load_b32 v6, v5, s[2:3]      // W[row][i]

    // x[i]: byte offset = i×4
    v_lshlrev_b32 v7, 2, v4
    global_load_b32 v8, v7, s[6:7]      // x[i]

    s_waitcnt    vmcnt(0)
    v_fmac_f32   v3, v6, v8             // partial_sum += W×x

.Ltile_skip:
    s_mov_b32    exec_lo, s15            // restore all threads

    // Next tile
    s_add_u32    s13, s13, 1
    s_lshl_b32  s14, s13, 8             // next tile base
    s_cmp_lt_u32 s14, s11               // next_base < in_dim?
    s_cbranch_scc1 .Ltile

    // ─── Phase 2: LDS reduction ───
    // 256 threads write partial sums to LDS, tree-reduce to LDS[0]
    v_lshlrev_b32 v10, 2, v0            // LDS offset = tid × 4
    ds_write_b32 v10, v3
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s14, 128
.Lreduce:
    v_cmp_lt_u32 vcc_lo, v0, s14
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Lreduce_skip

    v_mov_b32    v5, s14
    v_lshlrev_b32 v5, 2, v5             // stride × 4
    v_add_nc_u32 v6, v10, v5            // partner offset
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
    ds_read_b32  v3, v9                  // total dot product
    s_waitcnt    lgkmcnt(0)

    // Add bias
    s_lshl_b32  s16, s20, 2             // row × 4
    v_mov_b32   v5, s16
    global_load_b32 v6, v5, s[4:5]      // b[row]
    s_waitcnt    vmcnt(0)
    v_add_f32    v3, v3, v6

    // Store y[row]
    global_store_b32 v5, v3, s[8:9]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel matvec_tiled
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
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
  - .name:            matvec_tiled
    .symbol:          matvec_tiled.kd
    .kernarg_segment_size: 40
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
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 32, .value_kind: by_value }
      - { .size: 4, .offset: 36, .value_kind: by_value }
...
.end_amdgpu_metadata
