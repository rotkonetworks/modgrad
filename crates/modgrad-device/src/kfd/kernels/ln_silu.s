// Fused Affine LayerNorm + SiLU in-place.
// Single workgroup, LDS reduction. Handles n ≤ 1024.
//
// Computes:
//   mean = sum(x) / n
//   var  = sum((x - mean)^2) / n
//   x[i] = gamma[i] * (x[i] - mean) * rsqrt(var + eps) + beta[i]
//   x[i] = x[i] * sigmoid(x[i])   // SiLU
//
// Fusing LN+SiLU into one kernel eliminates one dispatch + one VRAM
// round-trip vs separate kernels. For a synapse block this turns
// 3 dispatches (matvec + ln + silu) into 2 (matvec_tiled + ln_silu).
//
// kernarg layout (32 bytes):
//   +0x00: x pointer (u64)     — [n] f32, in-place read/write
//   +0x08: gamma pointer (u64) — [n] f32, LN scale
//   +0x10: beta pointer (u64)  — [n] f32, LN bias
//   +0x18: n (u32)
//
// dispatch: grid = [256, 1, 1], block = [256, 1, 1] (single WG!)
// LDS: 256 × 4 = 1024 bytes

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl ln_silu_fwd
.p2align 8
.type ln_silu_fwd, @function
ln_silu_fwd:
    // s[0:1] = kernarg ptr, v0 = tid
    s_load_b64  s[2:3], s[0:1], 0x00    // x ptr
    s_load_b64  s[4:5], s[0:1], 0x08    // gamma ptr
    s_load_b64  s[6:7], s[0:1], 0x10    // beta ptr
    s_load_b32  s8,     s[0:1], 0x18    // n
    s_waitcnt   lgkmcnt(0)

    v_lshlrev_b32 v10, 2, v0            // LDS offset = tid × 4

    // ─── Load up to 4 elements per thread, compute partial sum ───
    v_mov_b32 v24, 0                     // partial sum
    v_mov_b32 v20, 0                     // cached x[tid]
    v_mov_b32 v21, 0                     // cached x[tid+256]
    v_mov_b32 v22, 0                     // cached x[tid+512]
    v_mov_b32 v23, 0                     // cached x[tid+768]

    // Element 0: tid
    v_cmp_lt_u32 vcc_lo, v0, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ll0
    v_lshlrev_b32 v1, 2, v0
    global_load_b32 v20, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v20
.Ll0:
    s_mov_b32 exec_lo, s10

    // Element 1: tid + 256
    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ll1
    v_lshlrev_b32 v1, 2, v2
    global_load_b32 v21, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v21
.Ll1:
    s_mov_b32 exec_lo, s10

    // Element 2: tid + 512
    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ll2
    v_lshlrev_b32 v1, 2, v3
    global_load_b32 v22, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v22
.Ll2:
    s_mov_b32 exec_lo, s10

    // Element 3: tid + 768
    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ll3
    v_lshlrev_b32 v1, 2, v4
    global_load_b32 v23, v1, s[2:3]
    s_waitcnt vmcnt(0)
    v_add_f32 v24, v24, v23
.Ll3:
    s_mov_b32 exec_lo, s10

    // ─── Reduce for mean ───
    ds_write_b32 v10, v24
    s_waitcnt lgkmcnt(0)
    s_barrier

    s_mov_b32 s11, 128
.Lmean_r:
    v_cmp_lt_u32 vcc_lo, v0, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lmean_s
    v_mov_b32 v5, s11
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32 v7, v6
    ds_read_b32 v8, v10
    s_waitcnt lgkmcnt(0)
    v_add_f32 v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt lgkmcnt(0)
.Lmean_s:
    s_mov_b32 exec_lo, s12
    s_barrier
    s_lshr_b32 s11, s11, 1
    s_cmp_gt_u32 s11, 0
    s_cbranch_scc1 .Lmean_r

    // mean = LDS[0] / n
    v_mov_b32 v9, 0
    ds_read_b32 v25, v9
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_u32 v26, s8
    v_rcp_f32 v26, v26
    v_mul_f32 v25, v25, v26              // v25 = mean

    // ─── Subtract mean, compute (x-mean)^2 partial sums ───
    v_mov_b32 v24, 0

    v_cmp_lt_u32 vcc_lo, v0, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lv0
    v_sub_f32 v20, v20, v25
    v_mul_f32 v1, v20, v20
    v_add_f32 v24, v24, v1
.Lv0:
    s_mov_b32 exec_lo, s10

    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lv1
    v_sub_f32 v21, v21, v25
    v_mul_f32 v1, v21, v21
    v_add_f32 v24, v24, v1
.Lv1:
    s_mov_b32 exec_lo, s10

    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lv2
    v_sub_f32 v22, v22, v25
    v_mul_f32 v1, v22, v22
    v_add_f32 v24, v24, v1
.Lv2:
    s_mov_b32 exec_lo, s10

    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Lv3
    v_sub_f32 v23, v23, v25
    v_mul_f32 v1, v23, v23
    v_add_f32 v24, v24, v1
.Lv3:
    s_mov_b32 exec_lo, s10

    // ─── Reduce for variance ───
    ds_write_b32 v10, v24
    s_waitcnt lgkmcnt(0)
    s_barrier

    s_mov_b32 s11, 128
.Lvar_r:
    v_cmp_lt_u32 vcc_lo, v0, s11
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Lvar_s
    v_mov_b32 v5, s11
    v_lshlrev_b32 v5, 2, v5
    v_add_nc_u32 v6, v10, v5
    ds_read_b32 v7, v6
    ds_read_b32 v8, v10
    s_waitcnt lgkmcnt(0)
    v_add_f32 v8, v8, v7
    ds_write_b32 v10, v8
    s_waitcnt lgkmcnt(0)
.Lvar_s:
    s_mov_b32 exec_lo, s12
    s_barrier
    s_lshr_b32 s11, s11, 1
    s_cmp_gt_u32 s11, 0
    s_cbranch_scc1 .Lvar_r

    // inv_std = rsqrt(var/n + eps)
    ds_read_b32 v26, v9
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_u32 v27, s8
    v_rcp_f32 v27, v27
    v_mul_f32 v26, v26, v27              // var
    v_add_f32 v26, 0x3727c5ac, v26       // var + 1e-5
    v_rsq_f32 v26, v26                   // inv_std

    // ─── Normalize (affine) + SiLU, store ───
    // x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i]
    // x[i] = x[i] * sigmoid(x[i])

    // Element 0
    v_cmp_lt_u32 vcc_lo, v0, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ls0
    v_lshlrev_b32 v1, 2, v0
    global_load_b32 v14, v1, s[4:5]      // gamma[i]
    global_load_b32 v15, v1, s[6:7]      // beta[i]
    s_waitcnt vmcnt(0)
    v_mul_f32 v20, v20, v26              // (x-mean)*inv_std
    v_mul_f32 v20, v20, v14              // * gamma
    v_add_f32 v20, v20, v15              // + beta
    // SiLU: x * sigmoid(x)
    v_mul_f32 v14, 0xbfb8aa3b, v20      // -x * log2(e)
    v_exp_f32 v14, v14                   // exp(-x)
    v_add_f32 v14, 1.0, v14             // 1 + exp(-x)
    v_rcp_f32 v14, v14                   // sigmoid
    v_mul_f32 v20, v20, v14             // x * sigmoid(x)
    global_store_b32 v1, v20, s[2:3]
.Ls0:
    s_mov_b32 exec_lo, s10

    // Element 1
    v_add_nc_u32 v2, v0, 256
    v_cmp_lt_u32 vcc_lo, v2, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ls1
    v_lshlrev_b32 v1, 2, v2
    global_load_b32 v14, v1, s[4:5]
    global_load_b32 v15, v1, s[6:7]
    s_waitcnt vmcnt(0)
    v_mul_f32 v21, v21, v26
    v_mul_f32 v21, v21, v14
    v_add_f32 v21, v21, v15
    v_mul_f32 v14, 0xbfb8aa3b, v21
    v_exp_f32 v14, v14
    v_add_f32 v14, 1.0, v14
    v_rcp_f32 v14, v14
    v_mul_f32 v21, v21, v14
    global_store_b32 v1, v21, s[2:3]
.Ls1:
    s_mov_b32 exec_lo, s10

    // Element 2
    v_add_nc_u32 v3, v0, 512
    v_cmp_lt_u32 vcc_lo, v3, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ls2
    v_lshlrev_b32 v1, 2, v3
    global_load_b32 v14, v1, s[4:5]
    global_load_b32 v15, v1, s[6:7]
    s_waitcnt vmcnt(0)
    v_mul_f32 v22, v22, v26
    v_mul_f32 v22, v22, v14
    v_add_f32 v22, v22, v15
    v_mul_f32 v14, 0xbfb8aa3b, v22
    v_exp_f32 v14, v14
    v_add_f32 v14, 1.0, v14
    v_rcp_f32 v14, v14
    v_mul_f32 v22, v22, v14
    global_store_b32 v1, v22, s[2:3]
.Ls2:
    s_mov_b32 exec_lo, s10

    // Element 3
    v_add_nc_u32 v4, v0, 768
    v_cmp_lt_u32 vcc_lo, v4, s8
    s_and_saveexec_b32 s10, vcc_lo
    s_cbranch_execz .Ls3
    v_lshlrev_b32 v1, 2, v4
    global_load_b32 v14, v1, s[4:5]
    global_load_b32 v15, v1, s[6:7]
    s_waitcnt vmcnt(0)
    v_mul_f32 v23, v23, v26
    v_mul_f32 v23, v23, v14
    v_add_f32 v23, v23, v15
    v_mul_f32 v14, 0xbfb8aa3b, v23
    v_exp_f32 v14, v14
    v_add_f32 v14, 1.0, v14
    v_rcp_f32 v14, v14
    v_mul_f32 v23, v23, v14
    global_store_b32 v1, v23, s[2:3]
.Ls3:
    s_mov_b32 exec_lo, s10

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel ln_silu_fwd
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 28
    .amdhsa_next_free_sgpr 13
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
  - .name:            ln_silu_fwd
    .symbol:          ln_silu_fwd.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      13
    .vgpr_count:      28
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 24, .value_kind: by_value }
...
.end_amdgpu_metadata
