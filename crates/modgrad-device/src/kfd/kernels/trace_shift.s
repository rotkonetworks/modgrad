// Trace shift: for each neuron, shift trace left by 1, append new activation.
// traces[n*M..(n+1)*M] = [traces[n*M+1..n*M+M], new_activations[n]]
//
// kernarg layout (24 bytes):
//   +0x00: traces pointer (u64)           — [n_neurons * memory_length] f32
//   +0x08: new_activations pointer (u64)  — [n_neurons] f32
//   +0x10: n_neurons (u32)
//   +0x14: memory_length (u32)
//
// dispatch: global_size = n_neurons, local_size = 256
// each workitem handles one neuron's trace shift.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl trace_shift_fwd
.p2align 8
.type trace_shift_fwd, @function
trace_shift_fwd:
    s_mov_b32 s20, s2

    s_load_b64  s[2:3], s[0:1], 0x00    // traces ptr
    s_load_b64  s[4:5], s[0:1], 0x08    // new_activations ptr
    s_load_b64  s[6:7], s[0:1], 0x10    // n_neurons(lo) | memory_length(hi)
    s_waitcnt   lgkmcnt(0)

    // s6 = n_neurons, s7 = memory_length

    // global_id = neuron index
    s_lshl_b32  s20, s20, 8
    v_add_nc_u32 v0, s20, v0

    // bounds check
    v_cmp_lt_u32 vcc_lo, v0, s6
    s_and_saveexec_b32 s8, vcc_lo
    s_cbranch_execz .Ldone

    // base byte offset = neuron * memory_length * 4
    v_mul_lo_u32 v1, v0, s7              // v1 = neuron * memory_length (element offset)
    v_lshlrev_b32 v1, 2, v1             // v1 = neuron * memory_length * 4 (byte offset)

    // Shift left: copy trace[neuron*M + j+1] → trace[neuron*M + j]
    // for j = 0 .. memory_length-2
    s_sub_u32 s9, s7, 1                  // s9 = memory_length - 1 (loop count)
    s_mov_b32 s10, 0                     // j = 0

.Lshift_loop:
    s_cmp_ge_u32 s10, s9                 // j >= memory_length - 1?
    s_cbranch_scc1 .Lshift_done

    // src offset = base + (j+1)*4
    s_add_u32 s11, s10, 1
    s_lshl_b32 s11, s11, 2              // (j+1)*4
    v_add_nc_u32 v2, v1, s11            // base + (j+1)*4

    // load trace[neuron*M + j+1]
    global_load_b32 v4, v2, s[2:3]
    s_waitcnt vmcnt(0)

    // dst offset = base + j*4
    s_lshl_b32 s12, s10, 2
    v_add_nc_u32 v3, v1, s12

    // store to trace[neuron*M + j]
    global_store_b32 v3, v4, s[2:3]

    s_add_u32 s10, s10, 1
    s_branch .Lshift_loop

.Lshift_done:
    // Append: trace[neuron*M + M-1] = new_activations[neuron]
    // load new_activations[neuron]
    v_lshlrev_b32 v5, 2, v0             // neuron * 4
    global_load_b32 v6, v5, s[4:5]
    s_waitcnt vmcnt(0)

    // dst offset = base + (M-1)*4
    s_sub_u32 s12, s7, 1
    s_lshl_b32 s12, s12, 2
    v_add_nc_u32 v3, v1, s12
    global_store_b32 v3, v6, s[2:3]

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel trace_shift_fwd
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
  - .name:            trace_shift_fwd
    .symbol:          trace_shift_fwd.kd
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
      - { .size: 4, .offset: 20, .value_kind: by_value }
...
.end_amdgpu_metadata
