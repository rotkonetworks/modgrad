// Sync backward: scatter gradients from pairs back to neurons via atomic add.
//
// For each pair i:
//   scale = 1.0 / max(sqrt(beta[i]), 1e-8)
//   val = d_sync[i] * scale
//   d_act[left[i]]  += val * activated[right[i]]   (atomic)
//   d_act[right[i]] += val * activated[left[i]]     (atomic)
//
// The output d_act array MUST be zero-initialized before dispatch.
//
// kernarg layout (48 bytes):
//   +0x00: d_sync pointer (u64)        — [n_pairs] f32, read
//   +0x08: activated pointer (u64)     — [d_model] f32, read
//   +0x10: beta pointer (u64)          — [n_pairs] f32, read
//   +0x18: left pointer (u64)          — [n_pairs] u32, read (neuron indices)
//   +0x20: right pointer (u64)         — [n_pairs] u32, read (neuron indices)
//   +0x28: d_act pointer (u64)         — [d_model] f32, read+write (atomic add)
//   +0x30: n_pairs (u32)
//
// dispatch: global_size = n_pairs, local_size = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl sync_backward_scatter
.p2align 8
.type sync_backward_scatter, @function
sync_backward_scatter:
    s_mov_b32 s20, s2                    // save wg_id

    // Load all kernargs
    s_load_b64  s[2:3],   s[0:1], 0x00  // d_sync ptr
    s_load_b64  s[4:5],   s[0:1], 0x08  // activated ptr
    s_load_b64  s[6:7],   s[0:1], 0x10  // beta ptr
    s_load_b64  s[8:9],   s[0:1], 0x18  // left ptr
    s_load_b64  s[10:11], s[0:1], 0x20  // right ptr
    s_load_b64  s[12:13], s[0:1], 0x28  // d_act ptr
    s_load_b32  s14,      s[0:1], 0x30  // n_pairs
    s_waitcnt   lgkmcnt(0)

    // global_id = wg_id * 256 + local_id
    s_lshl_b32  s20, s20, 8
    v_add_nc_u32 v0, s20, v0

    // bounds check
    v_cmp_lt_u32 vcc_lo, v0, s14
    s_and_saveexec_b32 s15, vcc_lo
    s_cbranch_execz .Ldone

    // byte offset for pair arrays = i * 4
    v_lshlrev_b32 v1, 2, v0

    // Load d_sync[i], beta[i], left[i], right[i]
    global_load_b32 v2, v1, s[2:3]       // d_sync[i]
    global_load_b32 v3, v1, s[6:7]       // beta[i]
    global_load_b32 v4, v1, s[8:9]       // left[i] (u32 index)
    global_load_b32 v5, v1, s[10:11]     // right[i] (u32 index)
    s_waitcnt vmcnt(0)

    // scale = 1.0 / max(sqrt(beta[i]), 1e-8)
    v_sqrt_f32 v3, v3                    // sqrt(beta)
    v_max_f32  v3, 0x322bcc77, v3        // max(sqrt(beta), 1e-8)
    v_rcp_f32  v3, v3                    // 1 / max(sqrt(beta), 1e-8) = scale
    v_mul_f32  v6, v2, v3                // val = d_sync[i] * scale

    // Load activated[left[i]] and activated[right[i]]
    // Byte offsets: left[i]*4, right[i]*4
    v_lshlrev_b32 v7, 2, v4             // left_byte_off = left[i] * 4
    v_lshlrev_b32 v8, 2, v5             // right_byte_off = right[i] * 4

    global_load_b32 v9,  v7, s[4:5]     // activated[left[i]]
    global_load_b32 v10, v8, s[4:5]     // activated[right[i]]
    s_waitcnt vmcnt(0)

    // contrib_left  = val * activated[right[i]]
    // contrib_right = val * activated[left[i]]
    v_mul_f32 v11, v6, v10              // val * activated[right[i]] → for left
    v_mul_f32 v12, v6, v9               // val * activated[left[i]]  → for right

    // Atomic add to d_act[left[i]] and d_act[right[i]]
    // No return value needed — use the 2-operand (no vdst) form.
    global_atomic_add_f32 v7, v11, s[12:13] offset:0
    global_atomic_add_f32 v8, v12, s[12:13] offset:0

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel sync_backward_scatter
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 56
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 13
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
  - .name:            sync_backward_scatter
    .symbol:          sync_backward_scatter.kd
    .kernarg_segment_size: 56
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      21
    .vgpr_count:      13
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,   .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,   .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 40,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 48,  .value_kind: by_value }
...
.end_amdgpu_metadata
