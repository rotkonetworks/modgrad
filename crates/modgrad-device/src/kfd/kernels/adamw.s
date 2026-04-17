// AdamW optimizer: per-element update with decoupled weight decay.
//
// For each element i:
//   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
//   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
//   w[i] -= lr * (m[i] * bc1_inv / (sqrt(v[i] * bc2_inv) + eps) + wd * w[i])
//   grad[i] = 0
//
// kernarg layout (64 bytes):
//   +0x00: w pointer (u64)     — [N] f32, weights (read-write)
//   +0x08: grad pointer (u64)  — [N] f32, gradients (read, zeroed after)
//   +0x10: m pointer (u64)     — [N] f32, first moment (read-write)
//   +0x18: v pointer (u64)     — [N] f32, second moment (read-write)
//   +0x20: N (u32)             — number of elements
//   +0x24: lr (f32)            — learning rate
//   +0x28: beta1 (f32)
//   +0x2c: beta2 (f32)
//   +0x30: eps (f32)
//   +0x34: weight_decay (f32)
//   +0x38: bc1_inv (f32)       — 1 / (1 - beta1^t)
//   +0x3c: bc2_inv (f32)       — 1 / (1 - beta2^t)
//
// dispatch: grid = ceil(N / 256), block = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl adamw
.p2align 8
.type adamw, @function
adamw:
    // s[0:1] = kernarg ptr, s2 = wg_id, v0 = local_id
    s_mov_b32 s20, s2

    // Load all kernargs
    s_load_b64   s[2:3],   s[0:1], 0x00  // w ptr
    s_load_b64   s[4:5],   s[0:1], 0x08  // grad ptr
    s_load_b64   s[6:7],   s[0:1], 0x10  // m ptr
    s_load_b64   s[8:9],   s[0:1], 0x18  // v ptr
    s_load_b128  s[12:15], s[0:1], 0x20  // N, lr, beta1, beta2
    s_load_b128  s[16:19], s[0:1], 0x30  // eps, wd, bc1_inv, bc2_inv
    s_waitcnt    lgkmcnt(0)

    // s12=N, s13=lr, s14=beta1, s15=beta2, s16=eps, s17=wd, s18=bc1_inv, s19=bc2_inv

    // global_id = wg_id * 256 + local_id
    s_lshl_b32  s21, s20, 8
    v_add_nc_u32 v0, s21, v0              // v0 = global_id

    // Bounds check: global_id < N
    v_cmp_lt_u32 vcc_lo, v0, s12
    s_and_saveexec_b32 s22, vcc_lo
    s_cbranch_execz .Ldone

    // Byte offset = global_id * 4
    v_lshlrev_b32 v1, 2, v0

    // Load w[i], grad[i], m[i], v[i]
    global_load_b32 v2, v1, s[2:3]       // v2 = w
    global_load_b32 v3, v1, s[4:5]       // v3 = grad
    global_load_b32 v4, v1, s[6:7]       // v4 = m
    global_load_b32 v5, v1, s[8:9]       // v5 = v
    s_waitcnt    vmcnt(0)

    // --- First moment update ---
    // m = beta1 * m + (1 - beta1) * grad
    // Rewrite: m = beta1 * m + grad - beta1 * grad
    //        = grad + beta1 * (m - grad)
    v_sub_f32    v6, v4, v3              // v6 = m - grad
    v_mov_b32    v7, s14                 // v7 = beta1
    v_fma_f32    v4, v7, v6, v3          // v4 = beta1*(m-grad) + grad = new m

    // --- Second moment update ---
    // v = beta2 * v + (1 - beta2) * grad^2
    // Rewrite: v = grad^2 + beta2 * (v - grad^2)
    v_mul_f32    v6, v3, v3              // v6 = grad^2
    v_sub_f32    v8, v5, v6              // v8 = v - grad^2
    v_mov_b32    v9, s15                 // v9 = beta2
    v_fma_f32    v5, v9, v8, v6          // v5 = beta2*(v-grad^2) + grad^2 = new v

    // --- Bias-corrected estimates ---
    // m_hat = m * bc1_inv
    v_mov_b32    v7, s18                 // v7 = bc1_inv
    v_mul_f32    v6, v4, v7              // v6 = m_hat

    // v_hat = v * bc2_inv
    v_mov_b32    v8, s19                 // v8 = bc2_inv
    v_mul_f32    v8, v5, v8              // v8 = v_hat

    // --- Weight update ---
    // w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    v_sqrt_f32   v8, v8                  // v8 = sqrt(v_hat)
    v_mov_b32    v9, s16                 // v9 = eps
    v_add_f32    v8, v8, v9              // v8 = sqrt(v_hat) + eps
    v_rcp_f32    v8, v8                  // v8 = 1 / (sqrt(v_hat) + eps)
    v_mul_f32    v6, v6, v8              // v6 = m_hat / (sqrt(v_hat) + eps)

    // Add weight decay term: v6 = m_hat/(sqrt(v_hat)+eps) + wd * w
    v_mov_b32    v9, s17                 // v9 = wd
    v_fma_f32    v6, v9, v2, v6          // v6 = wd*w + adaptive_term

    // w -= lr * v6
    v_mov_b32    v9, s13                 // v9 = lr
    v_xor_b32    v9, 0x80000000, v9     // v9 = -lr
    v_fmac_f32   v2, v9, v6             // v2 = w + (-lr) * update = w - lr*update

    // Store updated w, m, v
    global_store_b32 v1, v2, s[2:3]      // w[i] = updated
    global_store_b32 v1, v4, s[6:7]      // m[i] = updated
    global_store_b32 v1, v5, s[8:9]      // v[i] = updated

    // Zero gradient
    v_mov_b32    v10, 0
    global_store_b32 v1, v10, s[4:5]

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel adamw
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 64
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 11
    .amdhsa_next_free_sgpr 23
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
  - .name:            adamw
    .symbol:          adamw.kd
    .kernarg_segment_size: 64
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      23
    .vgpr_count:      11
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 32, .value_kind: by_value }
      - { .size: 4, .offset: 36, .value_kind: by_value }
      - { .size: 4, .offset: 40, .value_kind: by_value }
      - { .size: 4, .offset: 44, .value_kind: by_value }
      - { .size: 4, .offset: 48, .value_kind: by_value }
      - { .size: 4, .offset: 52, .value_kind: by_value }
      - { .size: 4, .offset: 56, .value_kind: by_value }
      - { .size: 4, .offset: 60, .value_kind: by_value }
...
.end_amdgpu_metadata
