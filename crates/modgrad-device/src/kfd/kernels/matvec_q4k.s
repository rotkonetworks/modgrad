// Fused Q4_K_M tiled matvec: y = Q4(W)*x + bias
//
// v3: Double-buffered prefetch. Loads next block's header while
// computing current block. Uses vmcnt(N) to overlap memory and compute.
// Same A/B ping-pong pattern as MIOpen conv kernels.
//
// Q4_K_M block (144 bytes per 256 elements):
//   +0x00: d (f16), dmin (f16)   — 4 bytes
//   +0x04: scales[12]            — packed 6-bit scales/mins
//   +0x10: qs[128]               — 4-bit quantized values
//
// kernarg layout (40 bytes):
//   +0x00: W_q4 pointer (u64)
//   +0x08: x pointer (u64)
//   +0x10: bias pointer (u64)
//   +0x18: y pointer (u64)
//   +0x20: out_dim (u32)
//   +0x24: blocks_per_row (u32)
//
// dispatch: grid=[out_dim,1,1] WGs, block=[256,1,1]
// LDS: 1024 bytes (reduction only)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl matvec_q4k
.p2align 8
.type matvec_q4k, @function
matvec_q4k:
    s_mov_b32 s20, s2                    // row = wg_id

    s_load_b64   s[2:3],   s[0:1], 0x00 // W_q4 ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // x ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // bias ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // y ptr
    s_load_b64   s[10:11], s[0:1], 0x20 // out_dim | blocks_per_row
    s_waitcnt    lgkmcnt(0)

    s_cmp_ge_u32 s20, s10
    s_cbranch_scc1 .Ldone

    // ─── Precompute per-thread constants (same for all blocks) ───

    // qs_idx and is_high from tid
    v_cmp_ge_u32 vcc_lo, v0, 128
    v_sub_nc_u32 v1, v0, 128
    v_cndmask_b32 v1, v0, v1, vcc_lo    // v1 = qs_idx
    v_cndmask_b32 v2, 0, 1, vcc_lo      // v2 = is_high

    // scale_idx (is): 0-7
    v_lshrrev_b32 v3, 5, v0
    v_lshrrev_b32 v4, 5, v1
    v_cmp_lt_u32 vcc_lo, v0, 128
    v_cndmask_b32 v3, v4, v3, vcc_lo
    v_lshlrev_b32 v3, 1, v3
    v_add_nc_u32 v4, v3, v2              // v4 = is (0..7)

    // Byte position in scales: shift = (is%4)*8
    v_and_b32    v5, 3, v4
    v_lshlrev_b32 v5, 3, v5             // v5 = shift (constant across blocks)

    // LDS offset for reduction
    v_lshlrev_b32 v10, 2, v0            // tid * 4

    // Row Q4 base = row * blocks_per_row * 144
    s_mul_i32    s21, s20, s11
    s_mul_i32    s21, s21, 144           // s21 = row_base

    // Accumulator
    v_mov_b32    v30, 0                  // acc = 0.0

    // ═══════════════════════════════════════════════════════════
    // Double-buffered block loop
    //
    // Pipeline: while computing block N, prefetch block N+1's header.
    // Block header = 16 bytes (d, dmin, scales) via global_load_b128.
    // qs byte + x value loaded per-thread after header arrives.
    //
    // Registers:
    //   v[12:15] = current block header (A buffer)
    //   v[16:19] = prefetch block header (B buffer)
    // ═══════════════════════════════════════════════════════════

    // Prefetch first block header (block 0)
    v_mov_b32    v6, s21                 // block 0 base = row_base
    global_load_b128 v[12:15], v6, s[2:3]

    s_mov_b32    s13, 0                  // blk = 0

.Lblk:
    // ─── Prefetch next block's header while we wait for current ───
    s_add_u32    s17, s13, 1             // next_blk = blk + 1
    s_cmp_lt_u32 s17, s11               // next_blk < blocks_per_row?
    s_cbranch_scc0 .Lno_prefetch

    s_mul_i32    s18, s17, 144
    s_add_u32    s18, s18, s21           // next block base
    v_mov_b32    v6, s18
    global_load_b128 v[16:19], v6, s[2:3] // prefetch into B buffer
    // This load flies in background — we don't wait for it yet

.Lno_prefetch:
    // ─── Wait for current block header (A buffer: v[12:15]) ───
    // vmcnt(1) = wait until at most 1 load outstanding (the prefetch)
    // If no prefetch was issued, vmcnt(0) is fine too — vmcnt(1) still works
    s_waitcnt    vmcnt(1)

    // ─── Extract d, dmin from v12 ───
    v_cvt_f32_f16 v20, v12
    v_lshrrev_b32 v6, 16, v12
    v_cvt_f32_f16 v21, v6

    // ─── Parallel scale extraction from v13, v14, v15 ───
    v_lshrrev_b32 v22, v5, v13
    v_and_b32    v22, 0xFF, v22          // byte from scales[0..3]
    v_lshrrev_b32 v23, v5, v14
    v_and_b32    v23, 0xFF, v23          // byte from scales[4..7]
    v_lshrrev_b32 v24, v5, v15
    v_and_b32    v24, 0xFF, v24          // byte from scales[8..11]

    // Path A (is < 4): sc = byte0 & 63, m = byte4 & 63
    v_and_b32    v25, 63, v22
    v_and_b32    v26, 63, v23

    // Path B (is >= 4): compound extraction
    v_and_b32    v27, 0xF, v24           // scales[is+4] & 0xF
    v_lshrrev_b32 v6, 6, v22            // scales[is-4] >> 6
    v_and_b32    v6, 3, v6
    v_lshlrev_b32 v6, 4, v6
    v_or_b32     v27, v27, v6            // sc_hi

    v_lshrrev_b32 v28, 4, v24
    v_and_b32    v28, 0xF, v28           // scales[is+4] >> 4
    v_lshrrev_b32 v6, 6, v23
    v_and_b32    v6, 3, v6
    v_lshlrev_b32 v6, 4, v6
    v_or_b32     v28, v28, v6            // m_hi

    // Select path
    v_cmp_lt_u32 vcc_lo, v4, 4
    v_cndmask_b32 v25, v27, v25, vcc_lo  // sc
    v_cndmask_b32 v26, v28, v26, vcc_lo  // m

    // d*sc, dmin*m
    v_cvt_f32_u32 v25, v25
    v_cvt_f32_u32 v26, v26
    v_mul_f32    v27, v20, v25           // d_sc
    v_mul_f32    v28, v21, v26           // dmin_m

    // ─── Load qs byte (current block) ───
    s_mul_i32    s14, s13, 144
    s_add_u32    s14, s14, s21           // current block base
    v_add_nc_u32 v6, v1, 16             // 16 + qs_idx
    v_add_nc_u32 v6, v6, s14
    // Aligned dword load + byte extraction
    v_and_b32    v7, 3, v6              // byte_in_word
    v_and_b32    v6, 0xFFFFFFFC, v6     // align to 4
    global_load_b32 v8, v6, s[2:3]

    // ─── Load x[element] (current block) ───
    s_lshl_b32  s16, s13, 8             // blk * 256
    v_add_nc_u32 v9, v0, s16            // tid + blk*256
    v_lshlrev_b32 v9, 2, v9
    global_load_b32 v11, v9, s[4:5]     // x[elem]

    // Wait for qs + x (2 loads), prefetch still in flight
    s_waitcnt    vmcnt(1)                // wait for qs and x, keep prefetch

    // Extract byte from aligned dword
    v_lshlrev_b32 v7, 3, v7             // byte_in_word * 8
    v_lshrrev_b32 v8, v7, v8
    v_and_b32    v8, 0xFF, v8            // qs byte

    // Extract nibble
    v_lshrrev_b32 v9, 4, v8             // high nibble
    v_and_b32    v8, 0xF, v8            // low nibble
    v_cmp_eq_u32 vcc_lo, v2, 1
    v_cndmask_b32 v8, v8, v9, vcc_lo   // nibble

    // ─── Dequant + FMA ───
    v_cvt_f32_u32 v8, v8
    v_mul_f32    v8, v27, v8             // d * sc * nibble
    v_sub_f32    v8, v8, v28             // - dmin * m
    // vmcnt(0) for x value (it was the 2nd load, should be ready)
    s_waitcnt    vmcnt(0)
    v_fmac_f32   v30, v8, v11            // acc += val * x[elem]

    // ─── Swap buffers: B→A for next iteration ───
    // Copy prefetched header from v[16:19] to v[12:15]
    s_add_u32    s13, s13, 1             // blk++
    s_cmp_ge_u32 s13, s11               // blk >= blocks_per_row?
    s_cbranch_scc1 .Lblk_done

    v_mov_b32    v12, v16
    v_mov_b32    v13, v17
    v_mov_b32    v14, v18
    v_mov_b32    v15, v19
    s_branch     .Lblk

.Lblk_done:
    // ─── LDS reduction ───
    ds_write_b32 v10, v30
    s_waitcnt    lgkmcnt(0)
    s_barrier

    s_mov_b32    s14, 128
.Lreduce:
    v_cmp_lt_u32 vcc_lo, v0, s14
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Lreduce_skip
    v_mov_b32    v6, s14
    v_lshlrev_b32 v6, 2, v6
    v_add_nc_u32 v7, v10, v6
    ds_read_b32  v8, v7
    ds_read_b32  v9, v10
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v9, v9, v8
    ds_write_b32 v10, v9
    s_waitcnt    lgkmcnt(0)
.Lreduce_skip:
    s_mov_b32    exec_lo, s15
    s_barrier
    s_lshr_b32  s14, s14, 1
    s_cmp_gt_u32 s14, 0
    s_cbranch_scc1 .Lreduce

    // ─── Thread 0: bias + store ───
    v_cmp_eq_u32 vcc_lo, v0, 0
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32    v6, 0
    ds_read_b32  v30, v6
    s_waitcnt    lgkmcnt(0)
    s_lshl_b32  s16, s20, 2
    v_mov_b32    v6, s16
    global_load_b32 v7, v6, s[6:7]
    s_waitcnt    vmcnt(0)
    v_add_f32    v30, v30, v7
    global_store_b32 v6, v30, s[8:9]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel matvec_q4k
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 31
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
  - .name:            matvec_q4k
    .symbol:          matvec_q4k.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      31
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
