// rdna3 LDS-tiled matmul: Y = X * W^T + B
//
// Each workgroup computes a 32x8 tile of Y.
// 256 threads/WG: thread_m = lid%32, thread_n = lid/32
// Tiles K in chunks of TK=32. Each tile iteration:
//   1. Cooperative load W[32][32] and X[8][32] to LDS
//   2. barrier
//   3. Each thread reads its W row and X row from LDS, does 32 FMAs
//   4. barrier
//
// LDS layout: W at 0 (4096B), X at 4096 (1024B), total 5120B
//
// Cooperative load assignment (W: 1024 elts / 256 threads = 4 each):
//   tile_row = lid/8, tile_col = (lid%8)*4
//   global_load_b128 loads W[tile_row][tile_col..+3]
//   ds_store_b128 at LDS offset lid*16
//
// Cooperative load assignment (X: 256 elts / 256 threads = 1 each):
//   x_row = lid/32, x_col = lid%32
//   global_load_b32 loads X[x_row][x_col]
//   ds_store_b32 at LDS offset 4096 + lid*4
//
// SGPR: s[0:1]=kernarg, s2=TGID_X
// kernargs (48B): W(u64) B(u64) X(u64) Y(u64) M(u32) K(u32) N(u32)
// dispatch: grid.x = ceil(M/32)*ceil(N/8)*256, block.x = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.set TM, 32
.set TN, 8
.set TK, 32
.set LDS_X, 4096       // W uses 0..4095, X uses 4096..5119
.set LDS_SZ, 5120

.text
.globl matmul_dbg
.p2align 8
.type matmul, @function
matmul_dbg:
    s_mov_b32 s20, s2                       // save wg_id

    s_load_b64   s[2:3],   s[0:1], 0x00    // W
    s_load_b64   s[4:5],   s[0:1], 0x08    // B
    s_load_b64   s[6:7],   s[0:1], 0x10    // X
    s_load_b64   s[8:9],   s[0:1], 0x18    // Y
    s_load_b64   s[10:11], s[0:1], 0x20    // M, K
    s_load_b32   s12,      s[0:1], 0x28    // N
    s_waitcnt    lgkmcnt(0)

    // num_wg_m = ceil(M/32)
    s_add_u32 s13, s10, TM - 1
    s_lshr_b32 s13, s13, 5

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

    // thread decomp
    v_and_b32 v1, 31, v0                   // thread_m = lid & 31
    v_lshrrev_b32 v2, 5, v0               // thread_n = lid >> 5

    // global output coords
    s_lshl_b32 s16, s15, 5                // wg_m * 32
    s_lshl_b32 s17, s14, 3                // wg_n * 8
    v_add_nc_u32 v3, s16, v1              // global_m
    v_add_nc_u32 v4, s17, v2              // global_n

    // load bias into accumulator
    v_lshlrev_b32 v5, 2, v3               // global_m * 4
    v_mov_b32 v6, 0
    v_cmp_lt_u32 vcc_lo, v3, s10
    s_and_saveexec_b32 s18, vcc_lo
    global_load_b32 v6, v5, s[4:5]
    s_mov_b32 exec_lo, s18
    // s_waitcnt vmcnt(0) -- no global loads

    // ======== Precompute cooperative load offsets ========

    // W coop: tile_row = lid/8, tile_col = (lid%8)*4
    v_lshrrev_b32 v7, 3, v0               // tile_row = lid >> 3
    v_and_b32 v8, 7, v0                    // lid & 7
    v_lshlrev_b32 v8, 2, v8               // tile_col = (lid & 7) * 4

    // W global byte offset for k_tile=0:
    //   ((wg_m*32 + tile_row) * K + tile_col) * 4
    v_add_nc_u32 v9, s16, v7              // wg_m*32 + tile_row
    v_mul_lo_u32 v9, v9, s11              // * K
    v_add_nc_u32 v9, v9, v8              // + tile_col
    v_lshlrev_b32 v9, 2, v9              // * 4 bytes
    // v9 = W global load voffset (running, += TK*4 each iter)

    // W LDS store offset = lid * 16 (4 floats * 4 bytes)
    v_lshlrev_b32 v10, 4, v0

    // X coop: x_row = lid/32 = v2, x_col = lid%32 = v1
    // X global byte offset for k_tile=0:
    //   ((wg_n*8 + x_row) * K + x_col) * 4
    v_add_nc_u32 v11, s17, v2             // wg_n*8 + lid/32
    v_mul_lo_u32 v11, v11, s11            // * K
    v_add_nc_u32 v11, v11, v1            // + lid%32
    v_lshlrev_b32 v11, 2, v11            // * 4 bytes
    // v11 = X global load voffset (running, += TK*4 each iter)

    // X LDS store offset = LDS_X + lid * 4
    v_lshlrev_b32 v12, 2, v0
    v_add_nc_u32 v12, LDS_X, v12

    // Compute-phase LDS read bases
    v_lshlrev_b32 v13, 7, v1              // W: thread_m * 128
    v_lshlrev_b32 v14, 7, v2
    v_add_nc_u32 v14, LDS_X, v14         // X: LDS_X + thread_n * 128

    // ======== Tile loop over K ========
    s_mov_b32 s18, 0                       // k_tile = 0

.Ltile_loop:
    s_cmp_ge_u32 s18, s11                  // k_tile >= K?
    s_cbranch_scc1 .Ltile_done

    // Phase 1: cooperative global → LDS
    v_mov_b32 v15, 1.0
    v_mov_b32 v16, 1.0
    v_mov_b32 v17, 1.0
    v_mov_b32 v18, 1.0
    v_mov_b32 v19, 1.0
    // s_waitcnt vmcnt(0) -- no global loads

    ds_store_b128 v10, v[15:18]            // W → LDS
    ds_store_b32 v12, v19                   // X → LDS
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Phase 2: compute 32 FMAs from LDS, unrolled 4x per block (8 blocks)

    // block 0: tk=0..3
    ds_load_b128 v[15:18], v13 offset:0
    ds_load_b128 v[20:23], v14 offset:0
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 1: tk=4..7
    ds_load_b128 v[15:18], v13 offset:16
    ds_load_b128 v[20:23], v14 offset:16
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 2: tk=8..11
    ds_load_b128 v[15:18], v13 offset:32
    ds_load_b128 v[20:23], v14 offset:32
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 3: tk=12..15
    ds_load_b128 v[15:18], v13 offset:48
    ds_load_b128 v[20:23], v14 offset:48
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 4: tk=16..19
    ds_load_b128 v[15:18], v13 offset:64
    ds_load_b128 v[20:23], v14 offset:64
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 5: tk=20..23
    ds_load_b128 v[15:18], v13 offset:80
    ds_load_b128 v[20:23], v14 offset:80
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 6: tk=24..27
    ds_load_b128 v[15:18], v13 offset:96
    ds_load_b128 v[20:23], v14 offset:96
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    // block 7: tk=28..31
    ds_load_b128 v[15:18], v13 offset:112
    ds_load_b128 v[20:23], v14 offset:112
    s_waitcnt lgkmcnt(0)
    v_fmac_f32 v6, v15, v20
    v_fmac_f32 v6, v16, v21
    v_fmac_f32 v6, v17, v22
    v_fmac_f32 v6, v18, v23

    s_barrier

    // advance offsets to next K tile
    v_add_nc_u32 v9, v9, TK * 4           // W += 128 bytes
    v_add_nc_u32 v11, v11, TK * 4         // X += 128 bytes
    s_add_u32 s18, s18, TK
    s_branch .Ltile_loop

.Ltile_done:
    // store Y[global_n][global_m] with bounds check
    v_cmp_lt_u32 vcc_lo, v3, s10           // global_m < M
    v_cmp_lt_u32 s19, v4, s12             // global_n < N
    s_and_b32 s19, vcc_lo, s19
    s_and_saveexec_b32 s20, s19
    s_cbranch_execz .Ldone

    v_mul_lo_u32 v15, v4, s10              // global_n * M
    v_add_nc_u32 v15, v15, v3              // + global_m
    v_lshlrev_b32 v15, 2, v15             // * 4 bytes
    global_store_b32 v15, v6, s[8:9]
    // s_waitcnt vmcnt(0) -- no global loads

.Ldone:
    s_endpgm

// ======== Kernel descriptor ========
.rodata
.p2align 6
.amdhsa_kernel matmul_dbg
    .amdhsa_group_segment_fixed_size LDS_SZ
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 24
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
  - .name:            matmul_dbg
    .symbol:          matmul_dbg.kd
    .kernarg_segment_size: 48
    .group_segment_fixed_size: 5120
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      24
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
