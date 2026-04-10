// absolute minimal test: store 42.0 to y[workitem_id]
// kernargs: y pointer (u64) at offset 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl test_store
.p2align 8
.type test_store, @function
test_store:
    // s[0:1] = kernarg pointer
    s_load_b64  s[2:3], s[0:1], 0x00    // y ptr
    s_waitcnt   lgkmcnt(0)

    // build full 64-bit address in v[2:3] = s[2:3] + v0*4
    v_lshlrev_b32 v1, 2, v0             // v1 = workitem_id * 4
    v_add_co_u32 v2, vcc_lo, s2, v1     // v2 = y_lo + offset
    v_add_co_ci_u32 v3, vcc_lo, s3, 0, vcc_lo  // v3 = y_hi + carry

    // store constant 42.0
    v_mov_b32   v4, 0x42280000           // v4 = 42.0f
    flat_store_b32 v[2:3], v4
    s_waitcnt   vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel test_store
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 4
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel
