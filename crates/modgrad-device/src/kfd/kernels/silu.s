// SiLU (swish) in-place: x[i] = x[i] * sigmoid(x[i])
//
// kernarg layout (16 bytes):
//   +0x00: x pointer (u64) — [n] f32, read and written in-place
//   +0x08: n (u32)
//
// dispatch: global_size = n, local_size = 256
// each workitem processes one element.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl silu_fwd
.p2align 8
.type silu_fwd, @function
silu_fwd:
    s_mov_b32 s20, s2

    s_load_b64  s[2:3], s[0:1], 0x00    // x ptr
    s_load_b32  s4,     s[0:1], 0x08    // n
    s_waitcnt   lgkmcnt(0)

    // global_id
    s_lshl_b32  s20, s20, 8
    v_add_nc_u32 v0, s20, v0

    // bounds check
    v_cmp_lt_u32 vcc_lo, v0, s4
    s_and_saveexec_b32 s5, vcc_lo
    s_cbranch_execz .Ldone

    // byte offset
    v_lshlrev_b32 v1, 2, v0

    // load x[i]
    global_load_b32 v4, v1, s[2:3]
    s_waitcnt vmcnt(0)

    // sigmoid(x[i]) = 1 / (1 + exp(-x))
    v_mul_f32 v5, 0xbfb8aa3b, v4        // -x * log2(e)
    v_exp_f32 v5, v5                     // exp(-x)
    v_add_f32 v5, 1.0, v5               // 1 + exp(-x)
    v_rcp_f32 v5, v5                     // sigmoid(x)

    // x[i] = x[i] * sigmoid(x[i])
    v_mul_f32 v4, v4, v5

    // store back
    global_store_b32 v1, v4, s[2:3]

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel silu_fwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 6
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
  - .name:            silu_fwd
    .symbol:          silu_fwd.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      6
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8,  .value_kind: by_value }
...
.end_amdgpu_metadata
