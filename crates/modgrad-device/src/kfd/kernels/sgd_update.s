// SGD weight update: w[i] -= lr * scale * grad[i]
// Fused gradient scaling + weight update in one pass.
// Also zeros grad[i] after applying (ready for next step).
//
// kernarg layout (32 bytes):
//   +0x00: w pointer (u64) — [N] f32, weights (read-write)
//   +0x08: grad pointer (u64) — [N] f32, gradients (read, zeroed after)
//   +0x10: lr_scale (f32) — learning_rate * gradient_scale
//   +0x14: N (u32) — number of elements
//
// dispatch: grid = ceil(N / 256), block = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl sgd_update
.p2align 8
.type sgd_update, @function
sgd_update:
    // s[0:1] = kernarg ptr, s2 = wg_id, v0 = local_id
    s_mov_b32 s20, s2

    s_load_b64   s[2:3],   s[0:1], 0x00 // w ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // grad ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // lr_scale (f32) | N (u32)
    s_waitcnt    lgkmcnt(0)

    // s6 = lr_scale (as u32 bits), s7 = N
    // global_id = wg_id * 256 + local_id
    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0            // v0 = global_id

    // Bounds check
    v_cmp_lt_u32 vcc_lo, v0, s7
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ldone

    // Byte offset = global_id * 4
    v_lshlrev_b32 v1, 2, v0

    // Load w[i] and grad[i]
    global_load_b32 v2, v1, s[2:3]     // v2 = w[i]
    global_load_b32 v3, v1, s[4:5]     // v3 = grad[i]
    s_waitcnt    vmcnt(0)

    // w[i] -= lr_scale * grad[i]
    // v_fmac_f32 does: dst += src0 * src1
    // We want: w -= lr * grad → w = w + (-lr) * grad = w - lr*grad
    // Use v_fma_f32 (VOP3): dst = src0 * src1 + src2
    // w_new = -lr_scale * grad + w = w - lr_scale * grad
    v_mov_b32   v4, s6                  // v4 = lr_scale bits
    v_xor_b32   v4, 0x80000000, v4     // v4 = -lr_scale (flip sign bit)
    v_fmac_f32  v2, v4, v3             // v2 = w + (-lr_scale) * grad = w - lr*grad

    // Store updated weight
    global_store_b32 v1, v2, s[2:3]

    // Zero gradient for next step
    v_mov_b32   v5, 0
    global_store_b32 v1, v5, s[4:5]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel sgd_update
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 6
    .amdhsa_next_free_sgpr 22
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
  - .name:            sgd_update
    .symbol:          sgd_update.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      6
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 16, .value_kind: by_value }
      - { .size: 4, .offset: 20, .value_kind: by_value }
...
.end_amdgpu_metadata
