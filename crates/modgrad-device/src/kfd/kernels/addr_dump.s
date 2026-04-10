// Debug: dump computed addresses to Y buffer
// Y[lid*8 + 0] = v9  (W offset)
// Y[lid*8 + 1] = v11 (X offset)
// Y[lid*8 + 2] = s2  (W ptr lo)
// Y[lid*8 + 3] = s3  (W ptr hi)
// Y[lid*8 + 4] = s6  (X ptr lo)
// Y[lid*8 + 5] = s7  (X ptr hi)
// Y[lid*8 + 6] = s11 (K)
// Y[lid*8 + 7] = exec_lo

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl addr_dump
.p2align 8
.type addr_dump, @function
addr_dump:
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00
    s_load_b64   s[4:5],   s[0:1], 0x08
    s_load_b64   s[6:7],   s[0:1], 0x10
    s_load_b64   s[8:9],   s[0:1], 0x18
    s_load_b64   s[10:11], s[0:1], 0x20
    s_load_b32   s12,      s[0:1], 0x28
    s_waitcnt    lgkmcnt(0)

    s_add_u32 s13, s10, 31
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

    v_and_b32 v1, 31, v0
    v_lshrrev_b32 v2, 5, v0
    s_lshl_b32 s16, s15, 5
    s_lshl_b32 s17, s14, 3
    v_add_nc_u32 v3, s16, v1
    v_add_nc_u32 v4, s17, v2

    // bias load (same as matmul)
    v_lshlrev_b32 v5, 2, v3
    v_mov_b32 v6, 0
    v_cmp_lt_u32 vcc_lo, v3, s10
    s_and_saveexec_b32 s18, vcc_lo
    global_load_b32 v6, v5, s[4:5]
    s_mov_b32 exec_lo, s18
    s_waitcnt vmcnt(0)

    // precompute (same as matmul)
    v_lshrrev_b32 v7, 3, v0
    v_and_b32 v8, 7, v0
    v_lshlrev_b32 v8, 2, v8
    v_add_nc_u32 v9, s16, v7
    v_mul_lo_u32 v9, v9, s11
    v_add_nc_u32 v9, v9, v8
    v_lshlrev_b32 v9, 2, v9

    v_lshlrev_b32 v10, 4, v0

    v_add_nc_u32 v11, s17, v2
    v_mul_lo_u32 v11, v11, s11
    v_add_nc_u32 v11, v11, v1
    v_lshlrev_b32 v11, 2, v11

    // dump 16 values per thread: Y[lid*16..lid*16+15]
    v_lshlrev_b32 v20, 6, v0              // lid * 64 (16 dwords * 4 bytes)

    global_store_b32 v20, v9, s[8:9] offset:0    // [0] W offset
    global_store_b32 v20, v11, s[8:9] offset:4   // [1] X offset
    v_mov_b32 v21, s14
    global_store_b32 v20, v21, s[8:9] offset:8   // [2] s14 (wg_n)
    v_mov_b32 v21, s15
    global_store_b32 v20, v21, s[8:9] offset:12  // [3] s15 (wg_m)
    v_mov_b32 v21, s16
    global_store_b32 v20, v21, s[8:9] offset:16  // [4] s16 (wg_m*32)
    v_mov_b32 v21, s17
    global_store_b32 v20, v21, s[8:9] offset:20  // [5] s17 (wg_n*8)
    v_mov_b32 v21, s13
    global_store_b32 v20, v21, s[8:9] offset:24  // [6] s13 (num_wg_m)
    v_mov_b32 v21, s20
    global_store_b32 v20, v21, s[8:9] offset:28  // [7] s20 (wg_id)
    global_store_b32 v20, v1, s[8:9] offset:32   // [8] v1 (thread_m)
    global_store_b32 v20, v2, s[8:9] offset:36   // [9] v2 (thread_n)
    v_mov_b32 v21, s11
    global_store_b32 v20, v21, s[8:9] offset:40  // [10] s11 (K)
    v_mov_b32 v21, s10
    global_store_b32 v20, v21, s[8:9] offset:44  // [11] s10 (M)
    v_mov_b32 v21, s12
    global_store_b32 v20, v21, s[8:9] offset:48  // [12] s12 (N)
    // intermediate: v11 before lshlrev = (s17+v2)*s11 + v1
    // let's compute it fresh
    v_add_nc_u32 v21, s17, v2
    global_store_b32 v20, v21, s[8:9] offset:52  // [13] s17+v2
    v_mul_lo_u32 v21, v21, s11
    global_store_b32 v20, v21, s[8:9] offset:56  // [14] (s17+v2)*K
    v_add_nc_u32 v21, v21, v1
    global_store_b32 v20, v21, s[8:9] offset:60  // [15] (s17+v2)*K + v1
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel addr_dump
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 26
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
  - .name:            addr_dump
    .symbol:          addr_dump.kd
    .kernarg_segment_size: 48
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      26
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
