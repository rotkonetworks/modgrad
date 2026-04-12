// GLU activation: output[i] = input[i] * sigmoid(input[n + i])
// input has 2*n elements, output has n elements.
//
// kernarg layout (24 bytes):
//   +0x00: input pointer (u64)  — [2*n] f32
//   +0x08: output pointer (u64) — [n] f32
//   +0x10: n (u32)
//
// dispatch: global_size = n, local_size = 256
// each workitem computes one output element.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl glu_fwd
.p2align 8
.type glu_fwd, @function
glu_fwd:
    // s[0:1] = kernarg pointer, s2 = workgroup_id_x, v0 = local_id
    s_mov_b32 s20, s2

    s_load_b64  s[2:3], s[0:1], 0x00    // input ptr
    s_load_b64  s[4:5], s[0:1], 0x08    // output ptr
    s_load_b32  s6,     s[0:1], 0x10    // n
    s_waitcnt   lgkmcnt(0)

    // global_id = wg_id * 256 + local_id
    s_lshl_b32  s20, s20, 8
    v_add_nc_u32 v0, s20, v0

    // bounds check: if global_id >= n, skip
    v_cmp_lt_u32 vcc_lo, v0, s6
    s_and_saveexec_b32 s7, vcc_lo
    s_cbranch_execz .Ldone

    // byte offset = i * 4
    v_lshlrev_b32 v1, 2, v0             // v1 = i * 4

    // load input[i] (linear half)
    global_load_b32 v4, v1, s[2:3]

    // load input[n + i] (gate half): byte offset = n*4 + i*4
    s_lshl_b32 s8, s6, 2                // s8 = n * 4
    v_add_nc_u32 v5, v1, s8             // v5 = (n + i) * 4
    global_load_b32 v6, v5, s[2:3]

    s_waitcnt vmcnt(0)

    // sigmoid(v6) = 1 / (1 + exp(-v6))
    // exp(-x) = 2^(-x * log2(e)),  log2(e) = 0x3fb8aa3b
    v_mul_f32 v6, 0xbfb8aa3b, v6        // v6 = -x * log2(e)
    v_exp_f32 v6, v6                     // v6 = exp(-x)
    v_add_f32 v6, 1.0, v6               // v6 = 1 + exp(-x)
    v_rcp_f32 v6, v6                     // v6 = sigmoid(x)

    // output[i] = input[i] * sigmoid(input[n+i])
    v_mul_f32 v4, v4, v6

    // store
    global_store_b32 v1, v4, s[4:5]

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel glu_fwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
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
  - .name:            glu_fwd
    .symbol:          glu_fwd.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      8
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 16, .value_kind: by_value }
...
.end_amdgpu_metadata
