// SuperLinear forward: N independent matvecs with per-neuron weights.
// Y[n*O+o] = B[n*O+o] + sum_k W[n*O*K + o*K + k] * X[n*K + k]
//
// Args: W(ptr), B(ptr), X(ptr), Y(ptr), N(u32), O(u32), K(u32)
// Grid: ceil(N*O / 256) workgroups, 256 threads each.
// Each thread computes one output element.
// K must be multiple of 4 and <= 32. Typical: K=4,8,16.
//
// Memory layout:
//   W: [N * O * K] row-major (same as SuperLinear.weights)
//   B: [N * O]
//   X: [N * K] (trace, flat arena)
//   Y: [N * O] (output)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl superlinear_fwd
.p2align 8
.type superlinear_fwd, @function
superlinear_fwd:
    // s[0:1] = kernarg_segment_ptr
    // s2 = workgroup_id_x

    // Load kernel args (7 args = 48 bytes)
    s_load_b64   s[4:5],   s[0:1], 0x00   // W ptr
    s_load_b64   s[6:7],   s[0:1], 0x08   // B ptr
    s_load_b64   s[8:9],   s[0:1], 0x10   // X ptr
    s_load_b64   s[10:11], s[0:1], 0x18   // Y ptr
    s_load_b32   s12,      s[0:1], 0x20   // N (n_neurons)
    s_load_b32   s13,      s[0:1], 0x24   // O (out_per)
    s_load_b32   s14,      s[0:1], 0x28   // K (in_per)
    s_waitcnt    lgkmcnt(0)

    // Global ID = workgroup_id * 256 + local_id
    s_lshl_b32  s2, s2, 8               // s2 = workgroup_id * 256
    v_add_nc_u32 v1, s2, v0             // v1 = global_id

    // Total outputs = N * O
    s_mul_i32    s15, s12, s13           // s15 = N * O
    v_cmp_lt_u32 vcc_lo, v1, s15        // bounds check
    s_and_saveexec_b32 s16, vcc_lo
    s_cbranch_execz .Lexit

    // Compute neuron = global_id / O, out_idx = global_id % O
    // Use FP32 reciprocal for approximate division, then correct.
    v_cvt_f32_u32 v4, s13               // v4 = float(O)
    v_rcp_iflag_f32 v4, v4              // v4 = 1.0/O (approx)
    v_cvt_f32_u32 v5, v1                // v5 = float(gid)
    v_mul_f32    v5, v5, v4             // v5 = gid / O (float approx)
    v_cvt_u32_f32 v2, v5               // v2 = neuron (truncated)

    // Fix overshoot: if neuron * O > gid, decrement
    v_mul_lo_u32 v4, v2, s13
    v_cmp_gt_u32 vcc_lo, v4, v1
    v_cndmask_b32 v5, 0, 1, vcc_lo
    v_sub_nc_u32 v2, v2, v5
    // Fix undershoot: if (neuron+1)*O <= gid, increment
    v_add_nc_u32 v5, v2, 1
    v_mul_lo_u32 v5, v5, s13
    v_cmp_le_u32 vcc_lo, v5, v1
    v_cndmask_b32 v5, 0, 1, vcc_lo
    v_add_nc_u32 v2, v2, v5
    v_mul_lo_u32 v4, v2, s13            // v4 = neuron * O

    // out_idx = gid - neuron * O
    v_sub_nc_u32 v3, v1, v4             // v3 = out_idx

    // W byte offset = (neuron*O*K + out_idx*K) * 4
    //               = ((neuron*O + out_idx) * K) * 4
    v_add_nc_u32 v5, v4, v3             // neuron*O + out_idx
    v_mul_lo_u32 v5, v5, s14            // * K
    v_lshlrev_b32 v5, 2, v5            // * 4 bytes

    // X byte offset = neuron * K * 4
    v_mul_lo_u32 v6, v2, s14
    v_lshlrev_b32 v6, 2, v6

    // B byte offset = (neuron*O + out_idx) * 4
    v_add_nc_u32 v7, v4, v3
    v_lshlrev_b32 v7, 2, v7

    // Load bias
    global_load_b32 v20, v7, s[6:7]

    // Dot product: accumulate in v21
    v_mov_b32    v21, 0                 // acc = 0.0

    // Loop over K in steps of 4 (vectorized loads)
    s_mov_b32    s17, 0
.Ldot4_loop:
    s_add_u32    s18, s17, 4
    s_cmp_gt_u32 s18, s14               // if counter+4 > K, done with vec loop
    s_cbranch_scc1 .Ldot4_done

    // Load 4 floats from W and X
    global_load_b128 v[12:15], v5, s[4:5]    // W[0..3]
    global_load_b128 v[16:19], v6, s[8:9]    // X[0..3]
    s_waitcnt    vmcnt(0)

    v_fmac_f32   v21, v12, v16
    v_fmac_f32   v21, v13, v17
    v_fmac_f32   v21, v14, v18
    v_fmac_f32   v21, v15, v19

    v_add_nc_u32 v5, v5, 16            // advance W ptr by 4 floats
    v_add_nc_u32 v6, v6, 16            // advance X ptr by 4 floats
    s_add_u32    s17, s17, 4
    s_branch     .Ldot4_loop
.Ldot4_done:

    // Handle remaining 1-3 elements (scalar)
.Ldot1_loop:
    s_cmp_ge_u32 s17, s14
    s_cbranch_scc1 .Ldot1_done
    global_load_b32 v12, v5, s[4:5]
    global_load_b32 v13, v6, s[8:9]
    s_waitcnt    vmcnt(0)
    v_fmac_f32   v21, v12, v13
    v_add_nc_u32 v5, v5, 4
    v_add_nc_u32 v6, v6, 4
    s_add_u32    s17, s17, 1
    s_branch     .Ldot1_loop
.Ldot1_done:

    // Y[gid] = acc + bias
    s_waitcnt    lgkmcnt(0)
    v_add_f32    v21, v21, v20

    // Store
    v_lshlrev_b32 v1, 2, v1
    global_store_b32 v1, v21, s[10:11]

.Lexit:
    s_waitcnt    vmcnt(0)
    s_endpgm

// Kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel superlinear_fwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 22
    .amdhsa_next_free_sgpr 19
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
  - .name:            superlinear_fwd
    .symbol:          superlinear_fwd.kd
    .kernarg_segment_size: 48
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      19
    .vgpr_count:      22
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 32, .value_kind: by_value }
      - { .size: 4, .offset: 36, .value_kind: by_value }
      - { .size: 4, .offset: 40, .value_kind: by_value }
...
.end_amdgpu_metadata
