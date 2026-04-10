// test cooperative W tile load to LDS
// each thread: load 4 floats from W, store to LDS, barrier, read own row back
// kernargs: W(u64) Y(u64) M(u32) K(u32)
// dispatch: 1 WG of 256 threads, computes for first 32x8 tile

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl coop_test
.p2align 8
.type coop_test, @function
coop_test:
    s_load_b64   s[2:3], s[0:1], 0x00     // W
    s_load_b64   s[4:5], s[0:1], 0x08     // Y
    s_load_b64   s[6:7], s[0:1], 0x10     // M, K (s6=M, s7=K)
    s_waitcnt    lgkmcnt(0)

    // v0 = lid
    v_and_b32 v1, 31, v0                   // thread_m = lid & 31
    v_lshrrev_b32 v2, 5, v0              // thread_n = lid >> 5

    // W coop load: tile_row = lid/8, tile_col = (lid%8)*4
    v_lshrrev_b32 v3, 3, v0               // tile_row
    v_and_b32 v4, 7, v0                    // lid & 7
    v_lshlrev_b32 v4, 2, v4               // tile_col = (lid&7)*4

    // W global offset = (tile_row * K + tile_col) * 4
    v_mul_lo_u32 v5, v3, s7               // tile_row * K
    v_add_nc_u32 v5, v5, v4              // + tile_col
    v_lshlrev_b32 v5, 2, v5              // * 4 bytes

    // global load 4 floats
    global_load_b128 v[6:9], v5, s[2:3]
    s_waitcnt vmcnt(0)

    // LDS store offset = lid * 16
    v_lshlrev_b32 v10, 4, v0
    ds_store_b128 v10, v[6:9]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Now read back: thread (thread_m, thread_n) reads W[thread_m][0] from LDS
    // LDS offset = thread_m * 32 * 4 = thread_m * 128
    v_lshlrev_b32 v11, 7, v1              // thread_m * 128
    ds_load_b32 v12, v11                   // W[thread_m][0]
    s_waitcnt lgkmcnt(0)

    // Store to Y[lid] = v12
    v_lshlrev_b32 v13, 2, v0              // lid * 4
    global_store_b32 v13, v12, s[4:5]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel coop_test
    .amdhsa_group_segment_fixed_size 5120
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 8
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
  - .name:            coop_test
    .symbol:          coop_test.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 5120
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      8
    .vgpr_count:      16
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 16, .value_kind: by_value }
      - { .size: 4, .offset: 20, .value_kind: by_value }
...
.end_amdgpu_metadata
