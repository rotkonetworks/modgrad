// minimal LDS test: each thread stores lid to LDS, reads back neighbor's value
// output[lid] = lid + 1 (wrapped) to verify LDS sharing works
// kernarg: Y(u64)
// dispatch: global=256, local=256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl lds_test
.p2align 8
.type lds_test, @function
lds_test:
    // s[0:1] = kernarg, s2 = wg_id (unused)
    s_load_b64 s[2:3], s[0:1], 0x00    // Y ptr
    s_waitcnt lgkmcnt(0)

    // v0 = lid
    // store lid to LDS[lid*4]
    v_lshlrev_b32 v1, 2, v0             // lid * 4
    v_mov_b32 v2, v0                     // value = lid
    ds_store_b32 v1, v2
    s_waitcnt lgkmcnt(0)
    s_barrier

    // read neighbor: LDS[(lid+1)%256 * 4]
    v_add_nc_u32 v3, v0, 1
    v_and_b32 v3, 255, v3               // (lid+1) % 256
    v_lshlrev_b32 v3, 2, v3
    ds_load_b32 v4, v3
    s_waitcnt lgkmcnt(0)

    // store to Y[lid]
    global_store_b32 v1, v4, s[2:3]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel lds_test
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
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
  - .name:            lds_test
    .symbol:          lds_test.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      4
    .vgpr_count:      8
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
...
.end_amdgpu_metadata
