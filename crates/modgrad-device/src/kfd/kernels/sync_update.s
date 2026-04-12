// Sync accumulator update (phase-aware Hebbian binding).
//
// For each pair i:
//   temporal_proximity = has_phase ? exp(-(pL[i]-pR[i])^2 / 0.18) : 1.0
//   pairwise = aL[i] * aR[i] * dopamine * temporal_proximity
//   r = exp(-(decay[i] + decay_shift[i]).clamp(0,15))
//   alpha[i] = initialized ? r*alpha[i] + pairwise : pairwise
//   beta[i]  = initialized ? r*beta[i]  + dopamine  : dopamine
//   sync_out[i] = alpha[i] / max(sqrt(beta[i]), 1e-8)
//
// kernarg layout (80 bytes):
//   +0x00: alpha pointer (u64)             — [n_pairs] f32, read+write
//   +0x08: beta pointer (u64)              — [n_pairs] f32, read+write
//   +0x10: activations_left pointer (u64)  — [n_pairs] f32
//   +0x18: activations_right pointer (u64) — [n_pairs] f32
//   +0x20: phases_left pointer (u64)       — [n_pairs] f32 (or null)
//   +0x28: phases_right pointer (u64)      — [n_pairs] f32 (or null)
//   +0x30: decay pointer (u64)             — [n_pairs] f32
//   +0x38: decay_shift pointer (u64)       — [n_pairs] f32
//   +0x40: dopamine (f32)
//   +0x44: n_pairs (u32)
//   +0x48: initialized (u32)               — 0 or 1
//   +0x4C: has_phase (u32)                 — 0 or 1
//   +0x50: sync_out pointer (u64)          — [n_pairs] f32, write
//
// dispatch: global_size = n_pairs, local_size = 256

.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
.amdhsa_code_object_version 5

.text
.globl sync_update_fwd
.p2align 8
.type sync_update_fwd, @function
sync_update_fwd:
    s_mov_b32 s30, s2                    // save wg_id

    // Load all kernargs (many pointers + scalars)
    s_load_b64  s[2:3],   s[0:1], 0x00  // alpha ptr
    s_load_b64  s[4:5],   s[0:1], 0x08  // beta ptr
    s_load_b64  s[6:7],   s[0:1], 0x10  // act_left ptr
    s_load_b64  s[8:9],   s[0:1], 0x18  // act_right ptr
    s_load_b64  s[10:11], s[0:1], 0x20  // phases_left ptr
    s_load_b64  s[12:13], s[0:1], 0x28  // phases_right ptr
    s_load_b64  s[14:15], s[0:1], 0x30  // decay ptr
    s_load_b64  s[16:17], s[0:1], 0x38  // decay_shift ptr
    s_load_b64  s[18:19], s[0:1], 0x40  // dopamine(f32), n_pairs(u32)
    s_load_b64  s[20:21], s[0:1], 0x48  // initialized(u32), has_phase(u32)
    s_load_b64  s[22:23], s[0:1], 0x50  // sync_out ptr
    s_waitcnt   lgkmcnt(0)

    // s18 = dopamine (as u32 bits), s19 = n_pairs, s20_arg = initialized, s21_arg = has_phase
    // Note: s20 was used for wg_id, now overwritten by kernarg load.
    // wg_id was saved to s30.

    // global_id
    s_lshl_b32  s30, s30, 8
    v_add_nc_u32 v0, s30, v0

    // bounds check
    v_cmp_lt_u32 vcc_lo, v0, s19
    s_and_saveexec_b32 s24, vcc_lo
    s_cbranch_execz .Ldone

    // byte offset = i * 4
    v_lshlrev_b32 v1, 2, v0

    // Load act_left[i], act_right[i]
    global_load_b32 v4, v1, s[6:7]       // act_left[i]
    global_load_b32 v5, v1, s[8:9]       // act_right[i]

    // Load decay[i], decay_shift[i]
    global_load_b32 v8, v1, s[14:15]     // decay[i]
    global_load_b32 v9, v1, s[16:17]     // decay_shift[i]

    s_waitcnt vmcnt(0)

    // ─── Temporal proximity ───
    // if has_phase: exp(-(pL-pR)^2 / (2*0.3*0.3)) = exp(-(pL-pR)^2 / 0.18)
    // else: 1.0
    v_mov_b32 v6, 1.0                    // temporal_proximity = 1.0 (default)
    s_cmp_eq_u32 s21, 0                  // has_phase == 0?
    s_cbranch_scc1 .Lno_phase

    // Load phases
    global_load_b32 v10, v1, s[10:11]    // phases_left[i]
    global_load_b32 v11, v1, s[12:13]    // phases_right[i]
    s_waitcnt vmcnt(0)

    v_sub_f32 v10, v10, v11              // phase_diff = pL - pR
    v_mul_f32 v10, v10, v10              // phase_diff^2
    // -1/0.18 = -5.5555... = 0xc0b1c71c
    v_mul_f32 v10, 0xc0b1c71c, v10      // -phase_diff^2 / 0.18
    // exp(x) = 2^(x * log2(e))
    v_mul_f32 v10, 0x3fb8aa3b, v10      // x * log2(e)
    v_exp_f32 v6, v10                    // temporal_proximity

.Lno_phase:
    // pairwise = aL * aR * dopamine * temporal_proximity
    v_mul_f32 v4, v4, v5                 // aL * aR
    v_mov_b32 v7, s18                    // dopamine (reinterpret bits as f32)
    v_mul_f32 v4, v4, v7                 // * dopamine
    v_mul_f32 v4, v4, v6                 // * temporal_proximity → v4 = pairwise

    // decay_val = clamp(decay + decay_shift, 0, 15)
    v_add_f32 v8, v8, v9                 // decay + decay_shift
    v_max_f32 v8, 0, v8                  // max(0, ...)
    // 15.0 = 0x41700000
    v_min_f32 v8, 0x41700000, v8         // min(15, ...) = clamped

    // r = exp(-decay_val) = 2^(-decay_val * log2(e))
    v_mul_f32 v8, 0xbfb8aa3b, v8        // -decay_val * log2(e)
    v_exp_f32 v8, v8                     // r = exp(-decay_val)

    // Load alpha[i], beta[i]
    global_load_b32 v12, v1, s[2:3]      // alpha[i]
    global_load_b32 v13, v1, s[4:5]      // beta[i]
    s_waitcnt vmcnt(0)

    // Update alpha, beta based on initialized flag
    s_cmp_eq_u32 s20, 0                  // initialized == 0?
    s_cbranch_scc1 .Lnot_init

    // initialized: alpha = r*alpha + pairwise, beta = r*beta + dopamine
    v_fmac_f32 v4, v8, v12              // v4 = pairwise + r*alpha  (v4 was pairwise)
    v_fmac_f32 v7, v8, v13              // v7 = dopamine + r*beta   (v7 was dopamine)
    s_branch .Lcompute_out

.Lnot_init:
    // not initialized: alpha = pairwise, beta = dopamine
    // v4 = pairwise (already), v7 = dopamine (already)

.Lcompute_out:
    // Store alpha[i] = v4, beta[i] = v7
    global_store_b32 v1, v4, s[2:3]      // alpha[i]
    global_store_b32 v1, v7, s[4:5]      // beta[i]

    // sync_out[i] = alpha / max(sqrt(beta), 1e-8)
    v_sqrt_f32 v13, v7                   // sqrt(beta)
    // 1e-8 = 0x322bcc77
    v_max_f32 v13, 0x322bcc77, v13       // max(sqrt(beta), 1e-8)
    v_rcp_f32 v13, v13                   // 1 / max(sqrt(beta), 1e-8)
    v_mul_f32 v4, v4, v13               // alpha * (1 / max(sqrt(beta), 1e-8))

    // Store sync_out[i]
    global_store_b32 v1, v4, s[22:23]

.Ldone:
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

// kernel descriptor
.rodata
.p2align 6
.amdhsa_kernel sync_update_fwd
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 88
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 14
    .amdhsa_next_free_sgpr 31
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
  - .name:            sync_update_fwd
    .symbol:          sync_update_fwd.kd
    .kernarg_segment_size: 88
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size:  32
    .sgpr_count:      31
    .vgpr_count:      14
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0,   .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 8,   .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 16,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 24,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 32,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 40,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 48,  .value_kind: global_buffer, .address_space: global }
      - { .size: 8, .offset: 56,  .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 64,  .value_kind: by_value }
      - { .size: 4, .offset: 68,  .value_kind: by_value }
      - { .size: 4, .offset: 72,  .value_kind: by_value }
      - { .size: 4, .offset: 76,  .value_kind: by_value }
      - { .size: 8, .offset: 80,  .value_kind: global_buffer, .address_space: global }
...
.end_amdgpu_metadata
