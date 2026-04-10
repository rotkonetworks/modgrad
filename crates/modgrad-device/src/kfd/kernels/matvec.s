// rdna3 matvec kernel: y = W*x + b
// kernarg layout (40 bytes, no hidden args):
//   +0x00: W pointer (u64) — [out_dim x in_dim] row-major f32
//   +0x08: b pointer (u64) — [out_dim] f32 bias
//   +0x10: x pointer (u64) — [in_dim] f32 input
//   +0x18: y pointer (u64) — [out_dim] f32 output
//   +0x20: out_dim (u32)
//   +0x24: in_dim (u32)
//
// each workitem computes one output element (row).
// dispatch: global_size = out_dim, local_size = 1 (or up to 256)
// uses flat_load/flat_store with full 64-bit vgpr addresses (proven pattern)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl matvec
.p2align 8
.type matvec, @function
matvec:
    // s[0:1] = kernarg pointer
    // v0 = workitem id

    // load all kernargs
    s_load_b64   s[2:3],   s[0:1], 0x00    // W ptr
    s_load_b64   s[4:5],   s[0:1], 0x08    // b ptr
    s_load_b64   s[6:7],   s[0:1], 0x10    // x ptr
    s_load_b64   s[8:9],   s[0:1], 0x18    // y ptr
    s_load_b64   s[10:11], s[0:1], 0x20    // out_dim(lo) | in_dim(hi)
    s_waitcnt    lgkmcnt(0)

    // s10 = out_dim, s11 = in_dim (loaded as b64 from offset 0x20)
    // bounds check: if v0 >= out_dim, skip
    v_cmp_lt_u32 vcc_lo, v0, s10
    s_and_saveexec_b32 s12, vcc_lo
    s_cbranch_execz .Ldone

    // row = v0
    // row_byte_off = v0 * in_dim * 4 (byte offset into W for this row)
    v_mul_lo_u32 v1, v0, s11           // v1 = row * in_dim (element offset)
    v_lshlrev_b32 v1, 2, v1            // v1 = row * in_dim * 4 (byte offset)

    // --- load bias b[row] using flat addressing ---
    // addr = b_ptr + row*4
    v_lshlrev_b32 v10, 2, v0           // v10 = row * 4
    v_add_co_u32  v12, vcc_lo, s4, v10 // lo = b_lo + row*4
    v_add_co_ci_u32 v13, vcc_lo, s5, 0, vcc_lo  // hi with carry
    flat_load_b32 v3, v[12:13]         // v3 = b[row]
    s_waitcnt vmcnt(0) lgkmcnt(0)

    // v3 = accumulator (initialized to bias)
    // loop over in_dim: sum += W[row*in_dim + i] * x[i]
    s_mov_b32 s13, 0                   // i = 0

.Lloop:
    s_cmp_ge_u32 s13, s11              // i >= in_dim?
    s_cbranch_scc1 .Lloop_done

    // w_byte_off = row_byte_off + i*4
    s_lshl_b32 s14, s13, 2             // s14 = i * 4

    // --- load W[row][i] via flat ---
    // addr = W_ptr + row_byte_off + i*4
    v_add_nc_u32 v4, v1, s14           // v4 = row_byte_off + i*4
    v_add_co_u32 v14, vcc_lo, s2, v4   // lo
    v_add_co_ci_u32 v15, vcc_lo, s3, 0, vcc_lo  // hi
    flat_load_b32 v5, v[14:15]         // v5 = W[row][i]

    // --- load x[i] via flat ---
    // addr = x_ptr + i*4
    v_mov_b32 v6, s14                  // v6 = i*4
    v_add_co_u32 v16, vcc_lo, s6, v6   // lo
    v_add_co_ci_u32 v17, vcc_lo, s7, 0, vcc_lo  // hi
    flat_load_b32 v7, v[16:17]         // v7 = x[i]

    s_waitcnt vmcnt(0) lgkmcnt(0)

    // sum += W[row][i] * x[i]
    v_fmac_f32 v3, v5, v7

    // i++
    s_add_u32 s13, s13, 1
    s_branch .Lloop

.Lloop_done:
    // --- store y[row] via flat ---
    // addr = y_ptr + row*4
    v_add_co_u32 v18, vcc_lo, s8, v10  // lo = y_lo + row*4
    v_add_co_ci_u32 v19, vcc_lo, s9, 0, vcc_lo  // hi
    flat_store_b32 v[18:19], v3
    s_waitcnt vmcnt(0) lgkmcnt(0)

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel matvec
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 20
    .amdhsa_next_free_sgpr 15
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

// AMDGPU metadata for HIP runtime module loading
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            matvec
    .symbol:          matvec.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      15
    .vgpr_count:      20
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 32, .value_kind: by_value }
      - { .size: 4, .offset: 36, .value_kind: by_value }
...
.end_amdgpu_metadata
