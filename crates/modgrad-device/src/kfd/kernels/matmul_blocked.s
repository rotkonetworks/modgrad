// rdna3 register-blocked LDS matmul: Y = X * W^T + B
//
// 4x4 output tile per thread: each thread computes 16 elements.
// TM=128, TN=32, TK=8, 256 threads/WG.
//   threads_m = 128/4 = 32, threads_n = 32/4 = 8
//   thread_m = lid % 32, thread_n = lid / 32
//
// LDS layout (TRANSPOSED for vectorized reads):
//   W: W[TK][TM] at offset 0       — 8*128*4 = 4096 bytes
//   X: X[TK][TN] at offset 4096    — 8*32*4  = 1024 bytes
//   Total: 5120 bytes
//
// Cooperative load (per thread, 4 W elements + 1 X element):
//   W: 128*8 = 1024 elts / 256 threads = 4 per thread
//     tile_row = lid/2, tile_col = (lid%2)*4
//     global: W[wg_m*128 + tile_row][k_tile + tile_col .. +3]
//     LDS (transposed): W[tile_col+i][tile_row] for i=0..3
//       = 4 scattered ds_store_b32 (stride = TM*4 = 512 bytes)
//   X: 32*8 = 256 elts / 256 threads = 1 per thread
//     x_row = lid/8, x_col = lid%8
//     global: X[wg_n*32 + x_row][k_tile + x_col]
//     LDS (transposed): X[x_col][x_row] at LDS_X + x_col*TN*4 + x_row*4
//       = 1 ds_store_b32
//
// Compute phase (per thread, per tk step):
//   W: ds_load_b128 reads W[tk][thread_m*4 .. thread_m*4+3] (contiguous after transpose)
//   X: ds_load_b128 reads X[tk][thread_n*4 .. thread_n*4+3] (contiguous after transpose)
//   16 FMAs: outer product of 4 W values * 4 X values
//   2 LDS reads per 16 FMAs = 8x better than non-blocked kernel
//
// dispatch: grid.x = ceil(M/128)*ceil(N/32), block.x = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.set TM, 128
.set TN, 32
.set TK, 8
.set BM, 4              // block size per thread in M
.set BN, 4              // block size per thread in N
.set LDS_X, 4096        // W: TK*TM*4 = 8*128*4 = 4096
.set LDS_SZ, 5120       // + X: TK*TN*4 = 8*32*4 = 1024

.text
.globl matmul_blocked
.p2align 8
.type matmul_blocked, @function
matmul_blocked:
    s_mov_b32 s20, s2                       // save wg_id

    s_load_b64   s[2:3],   s[0:1], 0x00    // W
    s_load_b64   s[4:5],   s[0:1], 0x08    // B
    s_load_b64   s[6:7],   s[0:1], 0x10    // X
    s_load_b64   s[8:9],   s[0:1], 0x18    // Y
    s_load_b64   s[10:11], s[0:1], 0x20    // M, K
    s_load_b32   s12,      s[0:1], 0x28    // N
    s_waitcnt    lgkmcnt(0)

    // num_wg_m = ceil(M/128)
    s_add_u32 s13, s10, TM - 1
    s_lshr_b32 s13, s13, 7                 // /128

    // wg_m = wg_id % num_wg_m, wg_n = wg_id / num_wg_m
    s_mov_b32 s14, 0
    s_mov_b32 s15, s20
.Ldiv:
    s_cmp_lt_u32 s15, s13
    s_cbranch_scc1 .Ldiv_done
    s_sub_u32 s15, s15, s13
    s_add_u32 s14, s14, 1
    s_branch .Ldiv
.Ldiv_done:
    // s15 = wg_m, s14 = wg_n

    // thread decomp: 32 threads in M, 8 in N
    v_and_b32 v1, 31, v0                   // thread_m = lid & 31
    v_lshrrev_b32 v2, 5, v0               // thread_n = lid >> 5

    // global output coords (base of 4x4 block)
    s_lshl_b32 s16, s15, 7                 // wg_m * 128
    s_lshl_b32 s17, s14, 5                 // wg_n * 32
    v_lshlrev_b32 v3, 2, v1               // thread_m * 4 = base_m offset
    v_add_nc_u32 v3, s16, v3              // global_m_base = wg_m*128 + thread_m*4
    v_lshlrev_b32 v4, 2, v2               // thread_n * 4 = base_n offset
    v_add_nc_u32 v4, s17, v4              // global_n_base = wg_n*32 + thread_n*4

    // ======== Initialize 16 accumulators with bias ========
    // acc[i][j] for i=0..3, j=0..3 in v16..v31
    // acc[i][j] = B[global_m_base + i]
    // Load 4 bias values
    v_mov_b32 v16, 0
    v_mov_b32 v17, 0
    v_mov_b32 v18, 0
    v_mov_b32 v19, 0
    v_mov_b32 v20, 0
    v_mov_b32 v21, 0
    v_mov_b32 v22, 0
    v_mov_b32 v23, 0
    v_mov_b32 v24, 0
    v_mov_b32 v25, 0
    v_mov_b32 v26, 0
    v_mov_b32 v27, 0
    v_mov_b32 v28, 0
    v_mov_b32 v29, 0
    v_mov_b32 v30, 0
    v_mov_b32 v31, 0

    // load B[global_m_base+0..3]
    v_lshlrev_b32 v5, 2, v3               // global_m_base * 4
    v_cmp_lt_u32 vcc_lo, v3, s10           // bounds check
    s_and_saveexec_b32 s18, vcc_lo
    global_load_b128 v[16:19], v5, s[4:5]  // B[m+0..3] → acc[0..3][0]
    s_mov_b32 exec_lo, s18
    s_waitcnt vmcnt(0)
    // Copy bias to all 4 N columns: acc[i][j] = B[m+i] for j=0..3
    v_mov_b32 v20, v16                     // acc[0][1] = B[m+0]
    v_mov_b32 v24, v16                     // acc[0][2]
    v_mov_b32 v28, v16                     // acc[0][3]
    v_mov_b32 v21, v17                     // acc[1][1] = B[m+1]
    v_mov_b32 v25, v17                     // acc[1][2]
    v_mov_b32 v29, v17                     // acc[1][3]
    v_mov_b32 v22, v18                     // acc[2][1]
    v_mov_b32 v26, v18                     // acc[2][2]
    v_mov_b32 v30, v18                     // acc[2][3]
    v_mov_b32 v23, v19                     // acc[3][1]
    v_mov_b32 v27, v19                     // acc[3][2]
    v_mov_b32 v31, v19                     // acc[3][3]

    // ======== Precompute cooperative load offsets ========

    // W coop: tile_row = lid/2 (0..127), tile_col = (lid%2)*4 (0 or 4)
    v_lshrrev_b32 v5, 1, v0               // tile_row = lid >> 1
    v_and_b32 v6, 1, v0                    // lid & 1
    v_lshlrev_b32 v6, 2, v6               // tile_col = (lid&1)*4

    // W global byte offset for k_tile=0:
    //   ((wg_m*128 + tile_row) * K + tile_col) * 4
    v_add_nc_u32 v7, s16, v5              // wg_m*128 + tile_row
    v_mul_lo_u32 v7, v7, s11              // * K
    v_add_nc_u32 v7, v7, v6              // + tile_col
    v_lshlrev_b32 v7, 2, v7              // * 4 bytes
    // v7 = W global load voffset (running, += TK*4 each iter)

    // W LDS store offsets (transposed): W[tile_col+i][tile_row]
    // base = tile_col * TM * 4 + tile_row * 4
    v_mul_lo_u32 v8, v6, TM              // tile_col * 128
    v_lshlrev_b32 v8, 2, v8              // * 4 bytes
    v_lshlrev_b32 v9, 2, v5              // tile_row * 4
    v_add_nc_u32 v8, v8, v9              // base LDS offset for W store
    // stride between consecutive tile_col values = TM*4 = 512

    // X coop: x_row = lid/8 (0..31), x_col = lid%8 (0..7)
    v_lshrrev_b32 v9, 3, v0               // x_row = lid >> 3
    v_and_b32 v10, 7, v0                   // x_col = lid & 7

    // X global byte offset for k_tile=0:
    //   ((wg_n*32 + x_row) * K + x_col) * 4
    v_add_nc_u32 v11, s17, v9             // wg_n*32 + x_row
    v_mul_lo_u32 v11, v11, s11            // * K
    v_add_nc_u32 v11, v11, v10           // + x_col
    v_lshlrev_b32 v11, 2, v11            // * 4 bytes
    // v11 = X global load voffset (running, += TK*4 each iter)

    // X LDS store offset (transposed): X[x_col][x_row]
    // = LDS_X + x_col * TN * 4 + x_row * 4
    v_mul_lo_u32 v12, v10, TN             // x_col * 32
    v_lshlrev_b32 v12, 2, v12            // * 4
    v_lshlrev_b32 v13, 2, v9             // x_row * 4
    v_add_nc_u32 v12, v12, v13
    v_add_nc_u32 v12, LDS_X, v12         // + LDS_X

    // Compute-phase LDS read bases (transposed layout)
    // W[tk][thread_m*4+0..3]: base = tk * TM * 4 + thread_m * 4 * 4
    //   = tk * 512 + thread_m * 16
    // We use offset for tk, so base = thread_m * 16
    v_lshlrev_b32 v13, 4, v1              // thread_m * 16

    // X[tk][thread_n*4+0..3]: base = LDS_X + tk * TN * 4 + thread_n * 4 * 4
    //   = LDS_X + tk * 128 + thread_n * 16
    v_lshlrev_b32 v14, 4, v2              // thread_n * 16
    v_add_nc_u32 v14, LDS_X, v14         // + LDS_X

    // ======== Tile loop over K (with prefetch) ========
    // Prefetch hides global memory latency by issuing the next tile's
    // load during the current tile's compute phase.
    // v[40:43] = prefetch W, v44 = prefetch X
    // v[32:35] = LDS W read, v[36:39] = LDS X read
    s_mov_b32 s18, 0                       // k_tile = 0

    // Prologue: load first tile into prefetch regs
    global_load_b128 v[40:43], v7, s[2:3]  // W tile 0
    global_load_b32 v44, v11, s[6:7]       // X tile 0

.Ltile_loop:
    // Wait for current tile's global data (first iter: prologue, then: prefetch)
    s_waitcnt vmcnt(0)

    // Store prefetched data to LDS (transposed)
    ds_store_b32 v8, v40                   // W[tile_col+0][tile_row]
    ds_store_b32 v8, v41 offset:512        // W[tile_col+1][tile_row]
    ds_store_b32 v8, v42 offset:1024       // W[tile_col+2][tile_row]
    ds_store_b32 v8, v43 offset:1536       // W[tile_col+3][tile_row]
    ds_store_b32 v12, v44                   // X[x_col][x_row]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Compute from LDS — 8 k-steps × 16 FMAs = 128 FMAs
    // Prefetch issued AFTER first LDS reads to not stall the barrier→compute path

    // tk=0
    ds_load_b128 v[32:35], v13 offset:0
    ds_load_b128 v[36:39], v14 offset:0
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v16, v32, v36
    v_fmac_f32 v17, v33, v36
    v_fmac_f32 v18, v34, v36
    v_fmac_f32 v19, v35, v36
    v_fmac_f32 v20, v32, v37
    v_fmac_f32 v21, v33, v37
    v_fmac_f32 v22, v34, v37
    v_fmac_f32 v23, v35, v37
    v_fmac_f32 v24, v32, v38
    v_fmac_f32 v25, v33, v38
    v_fmac_f32 v26, v34, v38
    v_fmac_f32 v27, v35, v38
    v_fmac_f32 v28, v32, v39
    v_fmac_f32 v29, v33, v39
    v_fmac_f32 v30, v34, v39
    v_fmac_f32 v31, v35, v39

    // Issue prefetch after first tk step — overlaps with tk=1..7 compute
    v_add_nc_u32 v7, v7, TK * 4           // W global += 32 bytes
    v_add_nc_u32 v11, v11, TK * 4         // X global += 32 bytes
    s_add_u32 s18, s18, TK
    global_load_b128 v[40:43], v7, s[2:3]  // prefetch next W
    global_load_b32 v44, v11, s[6:7]       // prefetch next X

    // tk=1..7
    .irp TK_OFF, 512, 1024, 1536, 2048, 2560, 3072, 3584
    ds_load_b128 v[32:35], v13 offset:\TK_OFF
    ds_load_b128 v[36:39], v14 offset:(\TK_OFF / 4)
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v16, v32, v36
    v_fmac_f32 v17, v33, v36
    v_fmac_f32 v18, v34, v36
    v_fmac_f32 v19, v35, v36
    v_fmac_f32 v20, v32, v37
    v_fmac_f32 v21, v33, v37
    v_fmac_f32 v22, v34, v37
    v_fmac_f32 v23, v35, v37
    v_fmac_f32 v24, v32, v38
    v_fmac_f32 v25, v33, v38
    v_fmac_f32 v26, v34, v38
    v_fmac_f32 v27, v35, v38
    v_fmac_f32 v28, v32, v39
    v_fmac_f32 v29, v33, v39
    v_fmac_f32 v30, v34, v39
    v_fmac_f32 v31, v35, v39
    .endr

    s_barrier

    // Loop if prefetched tile is valid
    s_cmp_lt_u32 s18, s11                  // s18 < K?
    s_cbranch_scc1 .Ltile_loop

.Ltile_done:
    // ======== Store 16 output elements ========
    // Y[global_n_base+j][global_m_base+i] for i=0..3, j=0..3
    // Y layout: row-major, Y[n][m], stride = M
    // offset = (global_n_base + j) * M + (global_m_base + i)

    // Compute base offset: global_n_base * M + global_m_base
    v_mul_lo_u32 v5, v4, s10              // global_n_base * M
    v_add_nc_u32 v5, v5, v3              // + global_m_base
    v_lshlrev_b32 v5, 2, v5              // * 4 bytes

    // Store row j=0: acc[0..3][0] = v16..v19
    global_store_b128 v5, v[16:19], s[8:9]

    // Row j=1: offset += M*4
    s_lshl_b32 s19, s10, 2                // M * 4
    v_add_nc_u32 v5, s19, v5
    global_store_b128 v5, v[20:23], s[8:9]

    // Row j=2
    v_add_nc_u32 v5, s19, v5
    global_store_b128 v5, v[24:27], s[8:9]

    // Row j=3
    v_add_nc_u32 v5, s19, v5
    global_store_b128 v5, v[28:31], s[8:9]

    s_waitcnt vmcnt(0)
    s_endpgm

// ======== Kernel descriptor ========
.rodata
.p2align 6
.amdhsa_kernel matmul_blocked
    .amdhsa_group_segment_fixed_size LDS_SZ
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 48
    .amdhsa_next_free_sgpr 21
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            matmul_blocked
    .symbol:          matmul_blocked.kd
    .kernarg_segment_size: 48
    .group_segment_fixed_size: 5120
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      48
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 32, .value_kind: by_value }
      - { .size: 4, .offset: 36, .value_kind: by_value }
      - { .size: 4, .offset: 40, .value_kind: by_value }
...
.end_amdgpu_metadata
