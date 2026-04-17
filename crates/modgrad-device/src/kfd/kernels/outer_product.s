// Outer product accumulate: dW[i*K+j] += d_out[i] * input[j]
// Used for gradient computation: d_weight = d_out ⊗ input (rank-1 update)
//
// kernarg layout (40 bytes):
//   +0x00: dW pointer (u64) — [M × K] f32, accumulated in-place
//   +0x08: d_out pointer (u64) — [M] f32
//   +0x10: input pointer (u64) — [K] f32
//   +0x18: M (u32) — output dimension (rows)
//   +0x1c: K (u32) — input dimension (cols)
//
// dispatch: grid = ceil(M*K / 256), block = 256
// each thread handles one element dW[idx] where idx = global_id

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl outer_product_acc
.p2align 8
.type outer_product_acc, @function
outer_product_acc:
    // s[0:1] = kernarg ptr, s2 = wg_id, v0 = local_id
    s_mov_b32 s20, s2                    // save wg_id

    s_load_b64   s[2:3],   s[0:1], 0x00 // dW ptr
    s_load_b64   s[4:5],   s[0:1], 0x08 // d_out ptr
    s_load_b64   s[6:7],   s[0:1], 0x10 // input ptr
    s_load_b64   s[8:9],   s[0:1], 0x18 // M | K
    s_waitcnt    lgkmcnt(0)

    // s8 = M, s9 = K
    // global_id = wg_id * 256 + local_id
    s_lshl_b32  s21, s20, 8             // wg_id * 256
    v_add_nc_u32 v0, s21, v0            // v0 = global_id

    // total elements = M * K
    s_mul_i32    s10, s8, s9            // s10 = M * K
    v_cmp_lt_u32 vcc_lo, v0, s10
    s_and_saveexec_b32 s11, vcc_lo
    s_cbranch_execz .Ldone

    // row = global_id / K, col = global_id % K
    // Use integer division: row = global_id / K
    v_cvt_f32_u32 v1, v0               // v1 = float(global_id)
    v_cvt_f32_u32 v2, s9               // v2 = float(K)
    v_rcp_f32    v2, v2                 // v2 = 1.0/K (approximate)
    v_mul_f32    v3, v1, v2             // v3 = global_id / K (approx)
    v_cvt_u32_f32 v3, v3               // v3 = row (truncated)

    // col = global_id - row * K
    v_mul_lo_u32 v4, v3, s9            // v4 = row * K
    v_sub_nc_u32 v4, v0, v4            // v4 = col = global_id - row*K

    // Fix rounding: if col >= K, row++, col -= K
    v_cmp_ge_u32 vcc_lo, v4, s9
    v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo  // row += carry
    v_subrev_nc_u32 v5, s9, v4          // v5 = col - K
    v_cndmask_b32 v4, v4, v5, vcc_lo   // col = carry ? col-K : col

    // Load d_out[row]
    v_lshlrev_b32 v5, 2, v3            // row * 4
    global_load_b32 v6, v5, s[4:5]     // v6 = d_out[row]

    // Load input[col]
    v_lshlrev_b32 v7, 2, v4            // col * 4
    global_load_b32 v8, v7, s[6:7]     // v8 = input[col]

    // Load dW[global_id] (current accumulated value)
    v_lshlrev_b32 v9, 2, v0            // global_id * 4
    global_load_b32 v10, v9, s[2:3]    // v10 = dW[global_id]

    s_waitcnt    vmcnt(0)

    // dW[global_id] += d_out[row] * input[col]
    v_fmac_f32   v10, v6, v8           // v10 += d_out[row] * input[col]
    global_store_b32 v9, v10, s[2:3]   // store back

.Ldone:
    s_waitcnt    vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel outer_product_acc
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 11
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
  - .name:            outer_product_acc
    .symbol:          outer_product_acc.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      22
    .vgpr_count:      11
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
      - { .size: 4, .offset: 28, .value_kind: by_value }
...
.end_amdgpu_metadata
