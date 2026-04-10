// TM=32 TT2x2 TK=8 wave32, PLR, single-buffer LDS
.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5
.set TM, 32
.set TN, 32
.set TK, 8
.set LDS_X, 1024
.set LDS_SZ, 2048
.text
.globl matmul_small
.p2align 8
.type matmul_small, @function
matmul_small:
    s_mov_b32 s20, s2
    s_load_b64   s[2:3],   s[0:1], 0x00
    s_load_b64   s[4:5],   s[0:1], 0x08
    s_load_b64   s[6:7],   s[0:1], 0x10
    s_load_b64   s[8:9],   s[0:1], 0x18
    s_load_b64   s[10:11], s[0:1], 0x20
    s_load_b32   s12,      s[0:1], 0x28
    s_waitcnt    lgkmcnt(0)
    s_add_u32 s13, s10, TM - 1
    s_lshr_b32 s13, s13, 5
    s_mov_b32 s14, 0
    s_mov_b32 s15, s20
.Ldiv:
    s_cmp_lt_u32 s15, s13
    s_cbranch_scc1 .Ldiv_done
    s_sub_u32 s15, s15, s13
    s_add_u32 s14, s14, 1
    s_branch .Ldiv
.Ldiv_done:
    v_and_b32 v1, 15, v0
    v_lshrrev_b32 v2, 4, v0
    s_lshl_b32 s16, s15, 5
    s_lshl_b32 s17, s14, 5
    v_lshlrev_b32 v3, 1, v1
    v_add_nc_u32 v3, s16, v3
    v_lshlrev_b32 v4, 1, v2
    v_add_nc_u32 v4, s17, v4
    v_mov_b32 v16, 0
    v_mov_b32 v17, 0
    v_mov_b32 v18, 0
    v_mov_b32 v19, 0
    v_lshlrev_b32 v5, 2, v3
    v_cmp_lt_u32 vcc_lo, v3, s10
    s_and_saveexec_b32 s18, vcc_lo
    global_load_b64 v[16:17], v5, s[4:5]
    s_mov_b32 exec_lo, s18
    s_waitcnt vmcnt(0)
    v_mov_b32 v18, v16
    v_mov_b32 v19, v17
    // W coop (col-major)
    v_lshrrev_b32 v5, 5, v0
    v_and_b32 v6, 31, v0
    v_mul_lo_u32 v7, v5, s10
    v_add_nc_u32 v7, v7, s16
    v_add_nc_u32 v7, v7, v6
    v_lshlrev_b32 v7, 2, v7
    v_mul_lo_u32 v8, v5, TM
    v_add_nc_u32 v8, v8, v6
    v_lshlrev_b32 v8, 2, v8
    s_lshl_b32 s22, s10, 5
    // X coop (coalesced)
    v_lshrrev_b32 v5, 3, v0
    v_and_b32 v6, 7, v0
    v_add_nc_u32 v9, s17, v5
    v_mul_lo_u32 v9, v9, s11
    v_add_nc_u32 v9, v9, v6
    v_lshlrev_b32 v9, 2, v9
    v_mul_lo_u32 v10, v6, TN
    v_add_nc_u32 v10, v10, v5
    v_lshlrev_b32 v10, 2, v10
    v_add_nc_u32 v10, LDS_X, v10
    // LDS read bases
    v_lshlrev_b32 v11, 3, v1
    v_lshlrev_b32 v12, 3, v2
    v_add_nc_u32 v12, LDS_X, v12
    // Prologue
    s_mov_b32 s18, 0
    global_load_b32 v20, v7, s[2:3]
    global_load_b32 v21, v9, s[6:7]
    s_waitcnt vmcnt(0)
    ds_store_b32 v8, v20
    ds_store_b32 v10, v21
    s_waitcnt lgkmcnt(0)
    s_barrier
    v_add_nc_u32 v7, v7, s22
    v_add_nc_u32 v9, v9, TK * 4
    s_add_u32 s18, s18, TK
    s_cmp_ge_u32 s18, s11
    s_cbranch_scc1 .Llast_tile
.Ltile_loop:
    // Prefetch tk=0
    ds_load_2addr_b32 v[22:23], v11 offset0:0 offset1:1
    ds_load_2addr_b32 v[24:25], v12 offset0:0 offset1:1
    global_load_b32 v20, v7, s[2:3]
    global_load_b32 v21, v9, s[6:7]
    s_waitcnt lgkmcnt(0)
    // tk=0
    ds_load_2addr_b32 v[26:27], v11 offset0:(1*32) offset1:(1*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(1*32) offset1:(1*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    // tk=1
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(2*32) offset1:(2*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(2*32) offset1:(2*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    // tk=2
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(3*32) offset1:(3*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(3*32) offset1:(3*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    // tk=3
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(4*32) offset1:(4*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(4*32) offset1:(4*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    // tk=4
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(5*32) offset1:(5*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(5*32) offset1:(5*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    // tk=5
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(6*32) offset1:(6*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(6*32) offset1:(6*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    // tk=6
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(7*32) offset1:(7*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(7*32) offset1:(7*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    // tk=7
    s_waitcnt lgkmcnt(0)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    // Store next tile
    s_waitcnt vmcnt(0)
    ds_store_b32 v8, v20
    ds_store_b32 v10, v21
    s_waitcnt lgkmcnt(0)
    s_barrier
    v_add_nc_u32 v7, v7, s22
    v_add_nc_u32 v9, v9, TK * 4
    s_add_u32 s18, s18, TK
    s_cmp_lt_u32 s18, s11
    s_cbranch_scc1 .Ltile_loop
.Llast_tile:
    ds_load_2addr_b32 v[22:23], v11 offset0:0 offset1:1
    ds_load_2addr_b32 v[24:25], v12 offset0:0 offset1:1
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(1*32) offset1:(1*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(1*32) offset1:(1*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(2*32) offset1:(2*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(2*32) offset1:(2*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(3*32) offset1:(3*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(3*32) offset1:(3*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(4*32) offset1:(4*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(4*32) offset1:(4*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(5*32) offset1:(5*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(5*32) offset1:(5*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[22:23], v11 offset0:(6*32) offset1:(6*32+1)
    ds_load_2addr_b32 v[24:25], v12 offset0:(6*32) offset1:(6*32+1)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    ds_load_2addr_b32 v[26:27], v11 offset0:(7*32) offset1:(7*32+1)
    ds_load_2addr_b32 v[28:29], v12 offset0:(7*32) offset1:(7*32+1)
    s_setprio 1
    v_fmac_f32 v16, v22, v24
    v_fmac_f32 v17, v23, v24
    v_fmac_f32 v18, v22, v25
    v_fmac_f32 v19, v23, v25
    s_setprio 0
    s_waitcnt lgkmcnt(0)
    s_setprio 1
    v_fmac_f32 v16, v26, v28
    v_fmac_f32 v17, v27, v28
    v_fmac_f32 v18, v26, v29
    v_fmac_f32 v19, v27, v29
    s_setprio 0
    // Store
    v_mul_lo_u32 v5, v4, s10
    v_add_nc_u32 v5, v5, v3
    v_lshlrev_b32 v5, 2, v5
    global_store_b64 v5, v[16:17], s[8:9]
    s_lshl_b32 s19, s10, 2
    v_add_nc_u32 v5, v5, s19
    global_store_b64 v5, v[18:19], s[8:9]
    s_waitcnt vmcnt(0)
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel matmul_small
    .amdhsa_group_segment_fixed_size LDS_SZ
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 30
    .amdhsa_next_free_sgpr 23
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
  - .name:            matmul_small
    .symbol:          matmul_small.kd
    .kernarg_segment_size: 48
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      23
    .vgpr_count:      30
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
