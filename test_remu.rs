//! Test GPU kernels using remu (RDNA3 emulator) — no hardware required.
//!
//! Verifies kernel correctness by running through the software emulator
//! and comparing against CPU reference implementations.
//!
//! Kernels tested: matvec_tiled, superlinear, glu, silu, layer_norm, trace_shift
//! (all kernels that use global_load/store with saddr — remu compatible)

use remu::run_asm;
use std::os::raw::c_char;

// ─── ELF parser ─────────────────────────────────────────────

/// Minimal ELF parser: extract the VA-indexed image and find the kernel
/// code entry point. Returns (image, code_offset_in_image).
fn extract_kernel(elf: &[u8]) -> Option<(Vec<u8>, usize)> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" { return None; }

    let r16 = |d: &[u8], o: usize| u16::from_le_bytes(d[o..o+2].try_into().unwrap());
    let r32 = |d: &[u8], o: usize| u32::from_le_bytes(d[o..o+4].try_into().unwrap());
    let r64 = |d: &[u8], o: usize| u64::from_le_bytes(d[o..o+8].try_into().unwrap());

    let e_phoff = r64(elf, 32) as usize;
    let e_phentsize = r16(elf, 54) as usize;
    let e_phnum = r16(elf, 56) as usize;

    let mut max_va: usize = 0;
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        let p_type = r32(elf, ph);
        if p_type != 1 { continue; }
        let p_vaddr = r64(elf, ph + 16) as usize;
        let p_memsz = r64(elf, ph + 40) as usize;
        max_va = max_va.max(p_vaddr + p_memsz);
    }
    if max_va == 0 { return None; }

    let mut image = vec![0u8; max_va];
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        let p_type = r32(elf, ph);
        if p_type != 1 { continue; }
        let p_offset = r64(elf, ph + 8) as usize;
        let p_vaddr = r64(elf, ph + 16) as usize;
        let p_filesz = r64(elf, ph + 32) as usize;
        let end = (p_offset + p_filesz).min(elf.len());
        image[p_vaddr..p_vaddr + (end - p_offset)].copy_from_slice(&elf[p_offset..end]);
    }

    let e_shoff = r64(elf, 40) as usize;
    let e_shentsize = r16(elf, 58) as usize;
    let e_shnum = r16(elf, 60) as usize;
    let e_shstrndx = r16(elf, 62) as usize;
    let _shstrtab_off = r64(elf, e_shoff + e_shstrndx * e_shentsize + 24) as usize;

    let mut symtab_off = 0usize;
    let mut symtab_size = 0usize;
    let mut symtab_entsize = 0usize;
    let mut symtab_link = 0usize;

    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        let sh_type = r32(elf, sh + 4);
        if sh_type == 2 {
            symtab_off = r64(elf, sh + 24) as usize;
            symtab_size = r64(elf, sh + 32) as usize;
            symtab_entsize = r64(elf, sh + 56) as usize;
            symtab_link = r32(elf, sh + 40) as usize;
        }
    }
    if symtab_entsize == 0 { return None; }

    let strtab_off = r64(elf, e_shoff + symtab_link * e_shentsize + 24) as usize;
    let n_syms = symtab_size / symtab_entsize;

    for i in 0..n_syms {
        let sym = symtab_off + i * symtab_entsize;
        if sym + symtab_entsize > elf.len() { break; }
        let name_off = r32(elf, sym) as usize;
        let sym_type = elf[sym + 4] & 0xf;
        if sym_type != 1 { continue; }
        let name_start = strtab_off + name_off;
        let name_end = elf[name_start..].iter().position(|&b| b == 0)
            .map(|p| name_start + p).unwrap_or(name_start);
        let name = std::str::from_utf8(&elf[name_start..name_end]).unwrap_or("");
        if !name.ends_with(".kd") { continue; }

        let kd_vaddr = r64(elf, sym + 8) as usize;
        let entry_off = i64::from_le_bytes(image[kd_vaddr + 0x10..kd_vaddr + 0x18].try_into().unwrap());
        let code_va = (kd_vaddr as i64 + entry_off) as usize;
        return Some((image, code_va));
    }
    None
}

/// Extract a specific kernel by name from a multi-kernel .co file.
fn extract_kernel_by_name(elf: &[u8], name: &str) -> Option<(Vec<u8>, usize)> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" { return None; }

    let r16 = |d: &[u8], o: usize| u16::from_le_bytes(d[o..o+2].try_into().unwrap());
    let r32 = |d: &[u8], o: usize| u32::from_le_bytes(d[o..o+4].try_into().unwrap());
    let r64 = |d: &[u8], o: usize| u64::from_le_bytes(d[o..o+8].try_into().unwrap());

    let e_phoff = r64(elf, 32) as usize;
    let e_phentsize = r16(elf, 54) as usize;
    let e_phnum = r16(elf, 56) as usize;

    let mut max_va: usize = 0;
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        let p_type = r32(elf, ph);
        if p_type != 1 { continue; }
        max_va = max_va.max(r64(elf, ph + 16) as usize + r64(elf, ph + 40) as usize);
    }
    if max_va == 0 { return None; }

    let mut image = vec![0u8; max_va];
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        if r32(elf, ph) != 1 { continue; }
        let off = r64(elf, ph + 8) as usize;
        let va = r64(elf, ph + 16) as usize;
        let fsz = r64(elf, ph + 32) as usize;
        let end = (off + fsz).min(elf.len());
        image[va..va + (end - off)].copy_from_slice(&elf[off..end]);
    }

    let e_shoff = r64(elf, 40) as usize;
    let e_shentsize = r16(elf, 58) as usize;
    let e_shnum = r16(elf, 60) as usize;

    let mut symtab_off = 0usize;
    let mut symtab_size = 0usize;
    let mut symtab_entsize = 0usize;
    let mut symtab_link = 0usize;

    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if r32(elf, sh + 4) == 2 {
            symtab_off = r64(elf, sh + 24) as usize;
            symtab_size = r64(elf, sh + 32) as usize;
            symtab_entsize = r64(elf, sh + 56) as usize;
            symtab_link = r32(elf, sh + 40) as usize;
        }
    }
    if symtab_entsize == 0 { return None; }

    let strtab_off = r64(elf, e_shoff + symtab_link * e_shentsize + 24) as usize;
    let target_kd = format!("{}.kd", name);

    for i in 0..symtab_size / symtab_entsize {
        let sym = symtab_off + i * symtab_entsize;
        if sym + symtab_entsize > elf.len() { break; }
        let name_off = r32(elf, sym) as usize;
        if elf[sym + 4] & 0xf != 1 { continue; }
        let ns = strtab_off + name_off;
        let ne = elf[ns..].iter().position(|&b| b == 0).map(|p| ns + p).unwrap_or(ns);
        let sym_name = std::str::from_utf8(&elf[ns..ne]).unwrap_or("");
        if sym_name != target_kd { continue; }

        let kd_va = r64(elf, sym + 8) as usize;
        let entry_off = i64::from_le_bytes(image[kd_va + 0x10..kd_va + 0x18].try_into().unwrap());
        let code_va = (kd_va as i64 + entry_off) as usize;
        return Some((image, code_va));
    }
    None
}

/// Run a kernel through remu (first kernel in .co).
fn run_kernel(co_bytes: &[u8], kargs: &[u8], gx: u32, gy: u32, gz: u32, lx: u32) {
    let (image, code_va) = extract_kernel(co_bytes).expect("failed to parse ELF");
    let code_bytes = &image[code_va..];
    assert!(code_bytes.len() % 4 == 0);

    let ret = run_asm(
        code_bytes.as_ptr() as *const c_char,
        code_bytes.len() as u32,
        gx, gy, gz, lx, 1, 1,
        kargs.as_ptr() as *const u64,
    );
    assert_eq!(ret, 0, "remu returned error code {}", ret);
}

/// Run a specific named kernel from a multi-kernel .co.
fn run_kernel_by_name(co_bytes: &[u8], name: &str, kargs: &[u8], gx: u32, gy: u32, gz: u32, lx: u32) {
    let (image, code_va) = extract_kernel_by_name(co_bytes, name)
        .unwrap_or_else(|| panic!("kernel '{}' not found in .co", name));
    let code_bytes = &image[code_va..];
    assert!(code_bytes.len() % 4 == 0);

    let ret = run_asm(
        code_bytes.as_ptr() as *const c_char,
        code_bytes.len() as u32,
        gx, gy, gz, lx, 1, 1,
        kargs.as_ptr() as *const u64,
    );
    assert_eq!(ret, 0, "remu '{}' returned error code {}", name, ret);
}

/// Pack kernargs as byte buffer.
struct KArgs(Vec<u8>);
impl KArgs {
    fn new() -> Self { Self(Vec::new()) }
    fn ptr<T>(&mut self, p: *const T) { self.0.extend_from_slice(&(p as u64).to_le_bytes()); }
    fn ptr_mut<T>(&mut self, p: *mut T) { self.0.extend_from_slice(&(p as u64).to_le_bytes()); }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn f32(&mut self, v: f32) { self.0.extend_from_slice(&v.to_le_bytes()); }
    fn bytes(&self) -> &[u8] { &self.0 }
}

/// Deterministic pseudo-random f32 init.
fn prand(buf: &mut [f32], seed: f32, scale: f32) {
    for i in 0..buf.len() {
        buf[i] = ((i as f32 * seed + 0.5).sin()) * scale;
    }
}

/// Compare two f32 slices, return (max_err, avg_err).
fn compare(a: &[f32], b: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_err: f32 = 0.0;
    let mut sum_err: f32 = 0.0;
    for i in 0..a.len() {
        let err = (a[i] - b[i]).abs();
        if err > 0.01 {
            eprintln!("  {}[{}]: gpu={:.6} ref={:.6} err={:.6}", label, i, a[i], b[i], err);
        }
        max_err = max_err.max(err);
        sum_err += err;
    }
    (max_err, sum_err / a.len() as f32)
}

// ─── CPU reference implementations ─────────────────────────

fn cpu_matvec(w: &[f32], b: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for col in 0..k {
            sum += w[row * k + col] * x[col];
        }
        y[row] = sum + b[row];
    }
    y
}

/// CPU reference for Y[n×m] = X[n×k] @ W^T[k×m] + B[m].
/// W is row-major [m×k] (out-major), matching the FFN Linear layout.
/// Y[row*m + col] = bias[col] + Σ_k X[row*k + ki] * W[col*k + ki]
fn cpu_matmul(w: &[f32], b: &[f32], x: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    for row in 0..n {
        for col in 0..m {
            let mut sum = b[col];
            let w_row = &w[col * k..(col + 1) * k];
            let x_row = &x[row * k..(row + 1) * k];
            for ki in 0..k {
                sum += w_row[ki] * x_row[ki];
            }
            y[row * m + col] = sum;
        }
    }
    y
}

fn cpu_superlinear(w: &[f32], b: &[f32], x: &[f32], n: usize, o: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * o];
    for neuron in 0..n {
        for out_j in 0..o {
            let global_out = neuron * o + out_j;
            let w_off = global_out * k;
            let x_off = neuron * k;
            let mut sum = 0.0f32;
            for col in 0..k {
                sum += w[w_off + col] * x[x_off + col];
            }
            y[global_out] = sum + b[global_out];
        }
    }
    y
}

fn cpu_glu(input: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let gate = 1.0 / (1.0 + (-input[n + i]).exp()); // sigmoid
        out[i] = input[i] * gate;
    }
    out
}

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn cpu_layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5f32).sqrt();
    x.iter().map(|&v| (v - mean) * inv_std).collect()
}

fn cpu_trace_shift(traces: &mut [f32], new_acts: &[f32], n_neurons: usize, mem_len: usize) {
    for n in 0..n_neurons {
        let base = n * mem_len;
        for j in 0..mem_len - 1 {
            traces[base + j] = traces[base + j + 1];
        }
        traces[base + mem_len - 1] = new_acts[n];
    }
}

// ─── Tests ──────────────────────────────────────────────────

/// Validate `matmul_blocked` kernel in emulation before we ever dispatch it on
/// real hardware. Shape constraints: M % 128 == 0, K % 8 == 0, N % 32 == 0.
/// Tiling params in the kernel are TM=128, TN=32, TK=8.
fn test_matmul_blocked() -> bool {
    println!("\n--- matmul_blocked ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");

    // Small-but-valid shapes first, then our real FFN shapes.
    // (N, K, M) — batch × in_dim × out_dim.
    let dims: &[(usize, usize, usize)] = &[
        (32,    8, 128),    // one WG
        (32,   64, 128),
        (64,   64, 256),
        (128,  64, 128),
        (32, 1024, 128),
        // Real FFN shapes (gate/up for large model):
        (128, 1024, 5120),
    ];

    let mut all_pass = true;
    for &(n, k, m) in dims {
        // Precondition check — the kernel requires these and we should mirror
        // the host-side validation layer.
        assert_eq!(m % 128, 0, "M={} not divisible by 128", m);
        assert_eq!(k % 8,   0, "K={} not divisible by 8",   k);
        assert_eq!(n % 32,  0, "N={} not divisible by 32",  n);

        let mut w = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; m];
        let mut x = vec![0.0f32; n * k];
        let mut y = vec![0.0f32; n * m];
        prand(&mut w, 0.001, 0.1);
        prand(&mut b, 0.01,  0.05);
        prand(&mut x, 0.007, 1.0);

        let y_ref = cpu_matmul(&w, &b, &x, n, k, m);

        let mut ka = KArgs::new();
        ka.ptr(w.as_ptr());
        ka.ptr(b.as_ptr());
        ka.ptr(x.as_ptr());
        ka.ptr_mut(y.as_mut_ptr());
        ka.u32(m as u32);
        ka.u32(k as u32);
        ka.u32(n as u32);

        let nwg_m = ((m + 127) / 128) as u32;
        let nwg_n = ((n +  31) /  32) as u32;
        let total_wg = nwg_m * nwg_n;
        run_kernel_by_name(co, "matmul_blocked", ka.bytes(), total_wg, 1, 1, 256);

        let (max_err, avg_err) = compare(&y, &y_ref, "y");
        // Tolerance: accumulation over K terms, f32 drift — 1e-2 is safe for our
        // weight scales. Tighter would be ideal but flaky in remu.
        let pass = max_err < 1e-2;
        println!("  N={:3} K={:5} M={:5}: max_err={:.2e} avg_err={:.2e} {}",
            n, k, m, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_matvec_tiled() -> bool {
    println!("\n--- matvec_tiled ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/matvec_tiled.co");
    let dims: &[(usize, usize)] = &[
        (4, 4), (16, 16), (64, 64), (128, 128),
        (256, 256), (256, 288), (512, 640),
        (128, 512), (512, 128),
    ];

    let mut all_pass = true;
    for &(m, k) in dims {
        let mut w = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; m];
        let mut x = vec![0.0f32; k];
        let mut y = vec![0.0f32; m];
        prand(&mut w, 0.001, 0.1);
        prand(&mut b, 0.01, 0.05);
        prand(&mut x, 0.007, 1.0);

        let y_ref = cpu_matvec(&w, &b, &x, m, k);

        let mut ka = KArgs::new();
        ka.ptr(w.as_ptr()); ka.ptr(b.as_ptr());
        ka.ptr(x.as_ptr()); ka.ptr_mut(y.as_mut_ptr());
        ka.u32(m as u32); ka.u32(k as u32);
        run_kernel(co, ka.bytes(), m as u32, 1, 1, 256);

        let (max_err, avg_err) = compare(&y, &y_ref, "y");
        let pass = max_err < 0.01;
        println!("  {}x{}: max_err={:.2e} avg_err={:.2e} {}", m, k, max_err, avg_err,
            if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_superlinear() -> bool {
    println!("\n--- superlinear_fwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/superlinear.co");
    let cases: &[(usize, usize, usize)] = &[
        (4, 8, 16),     // 4 neurons, 8 out, 16 in each
        (16, 4, 32),    // 16 neurons, 4 out, 32 in
        (64, 16, 64),   // medium
        (128, 8, 32),   // training-sized
    ];

    let mut all_pass = true;
    for &(n, o, k) in cases {
        let total_out = n * o;
        let total_in = n * k;
        let total_w = total_out * k;

        let mut w = vec![0.0f32; total_w];
        let mut b = vec![0.0f32; total_out];
        let mut x = vec![0.0f32; total_in];
        let mut y = vec![0.0f32; total_out];
        prand(&mut w, 0.0013, 0.1);
        prand(&mut b, 0.017, 0.05);
        prand(&mut x, 0.0031, 1.0);

        let y_ref = cpu_superlinear(&w, &b, &x, n, o, k);

        let mut ka = KArgs::new();
        ka.ptr(w.as_ptr()); ka.ptr(b.as_ptr());
        ka.ptr(x.as_ptr()); ka.ptr_mut(y.as_mut_ptr());
        ka.u32(n as u32); ka.u32(o as u32); ka.u32(k as u32);

        let nwg = ((total_out as u32) + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&y, &y_ref, "y");
        let pass = max_err < 0.01;
        println!("  n={} o={} k={}: max_err={:.2e} avg_err={:.2e} {}",
            n, o, k, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_glu() -> bool {
    println!("\n--- glu_fwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/glu.co");
    let sizes = [4, 16, 64, 128, 256, 512];

    let mut all_pass = true;
    for &n in &sizes {
        let mut input = vec![0.0f32; 2 * n]; // [data | gates]
        let mut output = vec![0.0f32; n];
        prand(&mut input, 0.0023, 2.0);

        let out_ref = cpu_glu(&input, n);

        let mut ka = KArgs::new();
        ka.ptr(input.as_ptr()); ka.ptr_mut(output.as_mut_ptr());
        ka.u32(n as u32);

        let nwg = ((n as u32) + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&output, &out_ref, "glu");
        let pass = max_err < 1e-5;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} {}",
            n, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_silu() -> bool {
    println!("\n--- silu_fwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/silu.co");
    let sizes = [4, 16, 64, 128, 256, 512];

    let mut all_pass = true;
    for &n in &sizes {
        let mut x = vec![0.0f32; n];
        prand(&mut x, 0.0037, 3.0);
        let x_ref = cpu_silu(&x);

        let mut ka = KArgs::new();
        ka.ptr_mut(x.as_mut_ptr()); // in-place
        ka.u32(n as u32);

        let nwg = ((n as u32) + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&x, &x_ref, "silu");
        let pass = max_err < 1e-5;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} {}",
            n, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_layer_norm_simple() -> bool {
    println!("\n--- layer_norm_fwd (diagnostic) ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/layer_norm.co");

    // NOTE: remu has a bug where LDS writes from one wave aren't visible
    // to reads from other waves after s_barrier. This causes layer_norm
    // (which broadcasts mean/var via LDS to all 8 waves) to fail for n>32.
    // The kernel is correct on real hardware — this is a remu limitation.
    // We test n<=32 (single wave) here.
    let n = 32;
    let mut x = vec![0.0f32; n];
    prand(&mut x, 0.0041, 5.0);
    let x_ref = cpu_layer_norm(&x);

    let mut ka = KArgs::new();
    ka.ptr_mut(x.as_mut_ptr());
    ka.u32(n as u32);
    run_kernel(co, ka.bytes(), 1, 1, 1, 256);

    let (max_err, _) = compare(&x, &x_ref, "ln");
    let pass = max_err < 1e-4;
    println!("  n=32 (single wave): max_err={:.2e} {}", max_err, if pass { "PASS" } else { "FAIL" });
    pass
}

fn test_layer_norm() -> bool {
    println!("\n--- layer_norm_fwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/layer_norm.co");
    // Only test n<=32 (single wave) — remu has a cross-wave LDS bug
    // that causes failures at n>32. The kernel is correct on real hardware.
    let sizes = [4, 16, 32];

    let mut all_pass = true;
    for &n in &sizes {
        let mut x = vec![0.0f32; n];
        prand(&mut x, 0.0041, 5.0);
        let x_ref = cpu_layer_norm(&x);

        let mut ka = KArgs::new();
        ka.ptr_mut(x.as_mut_ptr());
        ka.u32(n as u32);
        run_kernel(co, ka.bytes(), 1, 1, 1, 256);

        let (max_err, avg_err) = compare(&x, &x_ref, "ln");
        let pass = max_err < 1e-4;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} {}",
            n, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_trace_shift() -> bool {
    println!("\n--- trace_shift_fwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/trace_shift.co");
    let cases = [(4, 8), (16, 4), (64, 16), (128, 8)];

    let mut all_pass = true;
    for &(n_neurons, mem_len) in &cases {
        let total = n_neurons * mem_len;
        let mut traces = vec![0.0f32; total];
        let mut new_acts = vec![0.0f32; n_neurons];
        prand(&mut traces, 0.0051, 1.0);
        prand(&mut new_acts, 0.0071, 2.0);

        let mut ref_traces = traces.clone();
        cpu_trace_shift(&mut ref_traces, &new_acts, n_neurons, mem_len);

        let mut ka = KArgs::new();
        ka.ptr_mut(traces.as_mut_ptr());
        ka.ptr(new_acts.as_ptr());
        ka.u32(n_neurons as u32);
        ka.u32(mem_len as u32);

        let nwg = ((n_neurons as u32) + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&traces, &ref_traces, "trace");
        let pass = max_err < 1e-6;
        println!("  n={} m={}: max_err={:.2e} avg_err={:.2e} {}",
            n_neurons, mem_len, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_outer_product() -> bool {
    println!("\n--- outer_product_acc ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/outer_product.co");
    let cases: &[(usize, usize)] = &[
        (4, 4), (16, 16), (64, 32), (128, 64), (256, 256),
    ];

    let mut all_pass = true;
    for &(m, k) in cases {
        let mut dw = vec![0.0f32; m * k];
        let mut d_out = vec![0.0f32; m];
        let mut input = vec![0.0f32; k];
        // Pre-fill dW with existing values (accumulate test)
        prand(&mut dw, 0.0011, 0.5);
        prand(&mut d_out, 0.0023, 1.0);
        prand(&mut input, 0.0037, 2.0);

        // CPU reference: dw[i*k+j] += d_out[i] * input[j]
        let mut dw_ref = dw.clone();
        for i in 0..m {
            for j in 0..k {
                dw_ref[i * k + j] += d_out[i] * input[j];
            }
        }

        let mut ka = KArgs::new();
        ka.ptr_mut(dw.as_mut_ptr());
        ka.ptr(d_out.as_ptr());
        ka.ptr(input.as_ptr());
        ka.u32(m as u32);
        ka.u32(k as u32);

        let total = (m * k) as u32;
        let nwg = (total + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&dw, &dw_ref, "dw");
        let pass = max_err < 1e-4;
        println!("  {}x{}: max_err={:.2e} avg_err={:.2e} {}",
            m, k, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_sgd_update() -> bool {
    println!("\n--- sgd_update ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/sgd_update.co");
    let sizes = [4, 16, 64, 256, 512, 1024];

    let mut all_pass = true;
    for &n in &sizes {
        let mut w = vec![0.0f32; n];
        let mut grad = vec![0.0f32; n];
        prand(&mut w, 0.0019, 1.0);
        prand(&mut grad, 0.0029, 0.5);

        let lr_scale: f32 = 0.001;

        // CPU reference: w -= lr_scale * grad; grad = 0
        let mut w_ref = w.clone();
        for i in 0..n {
            w_ref[i] -= lr_scale * grad[i];
        }

        let mut ka = KArgs::new();
        ka.ptr_mut(w.as_mut_ptr());
        ka.ptr_mut(grad.as_mut_ptr());
        ka.f32(lr_scale);
        ka.u32(n as u32);

        let nwg = ((n as u32) + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&w, &w_ref, "w");
        let grad_zeroed = grad.iter().all(|&v| v == 0.0);
        let pass = max_err < 1e-6 && grad_zeroed;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} grad_zeroed={} {}",
            n, max_err, avg_err, grad_zeroed, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_matvec_t() -> bool {
    println!("\n--- matvec_t_tiled ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/matvec_t.co");

    // dx[j] = sum_i W[i*in_dim + j] * d_out[i]
    let dims: &[(usize, usize)] = &[
        (4, 4), (16, 16), (64, 32), (128, 64),
        (256, 256), (256, 512), (512, 128),
    ];

    let mut all_pass = true;
    for &(out_dim, in_dim) in dims {
        let mut w = vec![0.0f32; out_dim * in_dim];
        let mut d_out = vec![0.0f32; out_dim];
        let mut dx = vec![0.0f32; in_dim];
        prand(&mut w, 0.0013, 0.1);
        prand(&mut d_out, 0.0023, 1.0);

        // CPU reference: dx[j] = sum_i W[i*in_dim + j] * d_out[i]
        let mut dx_ref = vec![0.0f32; in_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                dx_ref[j] += w[i * in_dim + j] * d_out[i];
            }
        }

        // kernargs: W(ptr), d_out(ptr), dx(ptr), out_dim, in_dim
        let mut ka = KArgs::new();
        ka.ptr(w.as_ptr());
        ka.ptr(d_out.as_ptr());
        ka.ptr_mut(dx.as_mut_ptr());
        ka.u32(out_dim as u32);
        ka.u32(in_dim as u32);

        // grid = in_dim WGs (one per output element), block = 256
        run_kernel(co, ka.bytes(), in_dim as u32, 1, 1, 256);

        let (max_err, avg_err) = compare(&dx, &dx_ref, "dx");
        let pass = max_err < 0.01;
        println!("  {}x{}: max_err={:.2e} avg_err={:.2e} {}",
            out_dim, in_dim, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_silu_bwd() -> bool {
    println!("\n--- silu_bwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/silu_bwd.co");
    let sizes = [4, 16, 64, 128, 256, 512];

    let mut all_pass = true;
    for &n in &sizes {
        let mut d_out = vec![0.0f32; n];
        let mut pre = vec![0.0f32; n];
        let mut d_input = vec![0.0f32; n];
        prand(&mut d_out, 0.0019, 1.0);
        prand(&mut pre, 0.0029, 3.0);

        // CPU ref: d_input[i] = d_out[i] * (s + x * s * (1-s))
        let d_ref: Vec<f32> = d_out.iter().zip(&pre).map(|(&d, &x)| {
            let s = 1.0 / (1.0 + (-x).exp());
            d * (s + x * s * (1.0 - s))
        }).collect();

        let mut ka = KArgs::new();
        ka.ptr(d_out.as_ptr());
        ka.ptr(pre.as_ptr());
        ka.ptr_mut(d_input.as_mut_ptr());
        ka.u32(n as u32);

        let nwg = (n as u32 + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&d_input, &d_ref, "silu_bwd");
        let pass = max_err < 1e-4;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} {}",
            n, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_glu_bwd() -> bool {
    println!("\n--- glu_bwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/glu_bwd.co");
    let sizes = [4, 16, 64, 128, 256];

    let mut all_pass = true;
    for &n in &sizes {
        let mut d_out = vec![0.0f32; n];
        let mut cached_input = vec![0.0f32; 2 * n]; // [val | gate]
        let mut d_input = vec![0.0f32; 2 * n];
        prand(&mut d_out, 0.0031, 1.0);
        prand(&mut cached_input, 0.0041, 2.0);

        // CPU ref
        let mut d_ref = vec![0.0f32; 2 * n];
        for i in 0..n {
            let val = cached_input[i];
            let gate_v = cached_input[n + i];
            let s = 1.0 / (1.0 + (-gate_v).exp());
            let d = d_out[i];
            d_ref[i] = d * s;                        // d_val
            d_ref[n + i] = d * val * s * (1.0 - s);  // d_gate
        }

        let mut ka = KArgs::new();
        ka.ptr(d_out.as_ptr());
        ka.ptr(cached_input.as_ptr());
        ka.ptr_mut(d_input.as_mut_ptr());
        ka.u32(n as u32);

        let nwg = (n as u32 + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&d_input, &d_ref, "glu_bwd");
        let pass = max_err < 1e-4;
        println!("  n={}: max_err={:.2e} avg_err={:.2e} {}",
            n, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_ln_bwd() -> bool {
    println!("\n--- ln_bwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/ln_bwd.co");
    // Single-wave only (remu cross-wave LDS bug)
    let sizes = [4, 16, 32];

    let mut all_pass = true;
    for &n in &sizes {
        let mut d_out = vec![0.0f32; n];
        let mut normalized = vec![0.0f32; n];
        let mut gamma = vec![0.0f32; n];
        let mut d_gamma = vec![0.0f32; n];
        let mut d_beta = vec![0.0f32; n];
        let mut d_input = vec![0.0f32; n];
        prand(&mut d_out, 0.0051, 1.0);
        prand(&mut normalized, 0.0061, 1.5);
        prand(&mut gamma, 0.0071, 1.0);
        // Pre-fill d_gamma/d_beta (accumulation test)
        prand(&mut d_gamma, 0.0081, 0.1);
        prand(&mut d_beta, 0.0091, 0.1);
        let inv_std: f32 = 2.5;

        // CPU ref
        let mut dg_ref = d_gamma.clone();
        let mut db_ref = d_beta.clone();
        for i in 0..n {
            dg_ref[i] += d_out[i] * normalized[i];
            db_ref[i] += d_out[i];
        }
        let d_norm: Vec<f32> = (0..n).map(|i| d_out[i] * gamma[i]).collect();
        let nf = n as f32;
        let mean_dn: f32 = d_norm.iter().sum::<f32>() / nf;
        let mean_dn_xhat: f32 = d_norm.iter().zip(&normalized)
            .map(|(&d, &x)| d * x).sum::<f32>() / nf;
        let di_ref: Vec<f32> = (0..n).map(|i| {
            inv_std * (d_norm[i] - mean_dn - normalized[i] * mean_dn_xhat)
        }).collect();

        let mut ka = KArgs::new();
        ka.ptr(d_out.as_ptr());
        ka.ptr(normalized.as_ptr());
        ka.ptr(gamma.as_ptr());
        ka.ptr_mut(d_gamma.as_mut_ptr());
        ka.ptr_mut(d_beta.as_mut_ptr());
        ka.ptr_mut(d_input.as_mut_ptr());
        ka.f32(inv_std);
        ka.u32(n as u32);

        run_kernel(co, ka.bytes(), 1, 1, 1, 256);

        let (di_err, _) = compare(&d_input, &di_ref, "d_input");
        let (dg_err, _) = compare(&d_gamma, &dg_ref, "d_gamma");
        let (db_err, _) = compare(&d_beta, &db_ref, "d_beta");
        let pass = di_err < 1e-4 && dg_err < 1e-4 && db_err < 1e-4;
        println!("  n={}: di_err={:.2e} dg_err={:.2e} db_err={:.2e} {}",
            n, di_err, dg_err, db_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_per_neuron_glu_bwd() -> bool {
    println!("\n--- per_neuron_glu_bwd ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/per_neuron_glu_bwd.co");
    let cases: &[(usize, usize)] = &[
        (4, 8),    // 4 neurons, out_per=8 (half=4)
        (16, 16),  // 16 neurons, out_per=16
        (32, 8),   // 32 neurons
        (64, 4),   // 64 neurons, small out_per
    ];

    let mut all_pass = true;
    for &(n_neurons, out_per) in cases {
        let half = out_per / 2;
        let total_out = n_neurons * half;
        let total_in = n_neurons * out_per;

        let mut d_out = vec![0.0f32; total_out];
        let mut cached_input = vec![0.0f32; total_in];
        let mut d_input = vec![0.0f32; total_in];
        prand(&mut d_out, 0.0031, 1.0);
        prand(&mut cached_input, 0.0041, 2.0);

        // CPU reference
        let mut d_ref = vec![0.0f32; total_in];
        for n in 0..n_neurons {
            let base_in = n * out_per;
            let base_out = n * half;
            for j in 0..half {
                let val = cached_input[base_in + j];
                let gate_v = cached_input[base_in + half + j];
                let s = 1.0 / (1.0 + (-gate_v).exp());
                let d = d_out[base_out + j];
                d_ref[base_in + j] = d * s;
                d_ref[base_in + half + j] = d * val * s * (1.0 - s);
            }
        }

        let mut ka = KArgs::new();
        ka.ptr(d_out.as_ptr());
        ka.ptr(cached_input.as_ptr());
        ka.ptr_mut(d_input.as_mut_ptr());
        ka.u32(n_neurons as u32);
        ka.u32(out_per as u32);

        let nwg = (total_out as u32 + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (max_err, avg_err) = compare(&d_input, &d_ref, "pn_glu_bwd");
        let pass = max_err < 1e-4;
        println!("  n={} op={}: max_err={:.2e} avg_err={:.2e} {}",
            n_neurons, out_per, max_err, avg_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_reduce_l2() -> bool {
    println!("\n--- reduce_l2_sq ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/reduce_l2.co");
    let sizes = [4, 16, 64, 256, 512, 1024, 4096];

    let mut all_pass = true;
    for &n in &sizes {
        let mut x = vec![0.0f32; n];
        prand(&mut x, 0.0053, 2.0);

        let ref_l2sq: f32 = x.iter().map(|v| v * v).sum();

        let nwg = (n as u32 + 255) / 256;
        let mut partial_sums = vec![0.0f32; nwg as usize];

        let mut ka = KArgs::new();
        ka.ptr(x.as_ptr());
        ka.ptr_mut(partial_sums.as_mut_ptr());
        ka.u32(n as u32);

        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        // CPU pass 2: sum partial results
        let gpu_l2sq: f32 = partial_sums.iter().sum();

        let err = (gpu_l2sq - ref_l2sq).abs();
        let rel_err = err / ref_l2sq.abs().max(1e-8);
        let pass = rel_err < 1e-4;
        println!("  n={}: gpu={:.4} ref={:.4} rel_err={:.2e} {}",
            n, gpu_l2sq, ref_l2sq, rel_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn test_superlinear_bwd() -> bool {
    println!("\n--- superlinear_bwd_dw + superlinear_bwd_dx ---");
    let co = include_bytes!("crates/modgrad-device/src/kfd/kernels/superlinear_bwd.co");

    // Need two kernels from the same .co — extract_kernel only returns the first.
    // Both kernels share the same image, just at different code offsets.
    // For this test, we'll extract both by looking for both .kd symbols.
    // Simpler: compile as two separate tests using the same .co.

    let cases: &[(usize, usize, usize)] = &[
        (4, 4, 8),      // 4 neurons, 4 out, 8 in
        (8, 8, 16),     // 8 neurons, 8 out, 16 in
        (16, 4, 32),    // larger
        (32, 8, 16),    // typical training size
    ];

    let mut all_pass = true;
    for &(n, o, k) in cases {
        let total_w = n * o * k;
        let total_out = n * o;
        let total_in = n * k;

        let mut w = vec![0.0f32; total_w];
        let mut d_out = vec![0.0f32; total_out];
        let mut input = vec![0.0f32; total_in];
        let mut dw = vec![0.0f32; total_w];
        prand(&mut w, 0.0013, 0.1);
        prand(&mut d_out, 0.0023, 1.0);
        prand(&mut input, 0.0037, 2.0);
        prand(&mut dw, 0.0047, 0.3); // pre-existing accumulated grads

        // CPU reference: d_weight
        let mut dw_ref = dw.clone();
        for neuron in 0..n {
            let t = &input[neuron * k..(neuron + 1) * k];
            for oi in 0..o {
                let d = d_out[neuron * o + oi];
                for ki in 0..k {
                    dw_ref[neuron * o * k + oi * k + ki] += d * t[ki];
                }
            }
        }

        // CPU reference: d_input
        let mut dx_ref = vec![0.0f32; total_in];
        for neuron in 0..n {
            for ki in 0..k {
                let mut sum = 0.0f32;
                for oi in 0..o {
                    sum += d_out[neuron * o + oi] * w[neuron * o * k + oi * k + ki];
                }
                dx_ref[neuron * k + ki] = sum;
            }
        }

        // ── Test d_weight kernel ──
        // kernargs: W(unused), dW, d_out, input, dX(unused), N, O, K
        let dummy = vec![0.0f32; 1];
        let mut ka = KArgs::new();
        ka.ptr(dummy.as_ptr());          // W (unused by bwd_dw)
        ka.ptr_mut(dw.as_mut_ptr());     // dW
        ka.ptr(d_out.as_ptr());          // d_out
        ka.ptr(input.as_ptr());          // input
        ka.ptr(dummy.as_ptr());          // dX (unused by bwd_dw)
        ka.u32(n as u32);
        ka.u32(o as u32);
        ka.u32(k as u32);

        // superlinear_bwd_dw is the FIRST kernel in the .co
        let nwg = (total_w as u32 + 255) / 256;
        run_kernel(co, ka.bytes(), nwg, 1, 1, 256);

        let (dw_err, _dw_avg) = compare(&dw, &dw_ref, "dw");
        let dw_pass = dw_err < 1e-4;

        // ── Test d_input kernel ──
        // Need to extract second kernel entry point
        // For now, test using a fresh extraction that finds "superlinear_bwd_dx.kd"
        let mut dx = vec![0.0f32; total_in];
        let mut ka2 = KArgs::new();
        ka2.ptr(w.as_ptr());              // W
        ka2.ptr(dummy.as_ptr());          // dW (unused by bwd_dx)
        ka2.ptr(d_out.as_ptr());          // d_out
        ka2.ptr(dummy.as_ptr());          // input (unused by bwd_dx)
        ka2.ptr_mut(dx.as_mut_ptr());     // dX
        ka2.u32(n as u32);
        ka2.u32(o as u32);
        ka2.u32(k as u32);

        // Need to run the second kernel — extract_kernel finds first .kd only
        // Use a helper that finds a specific kernel name
        let nwg2 = (total_in as u32 + 255) / 256;
        run_kernel_by_name(co, "superlinear_bwd_dx", ka2.bytes(), nwg2, 1, 1, 256);

        let (dx_err, _dx_avg) = compare(&dx, &dx_ref, "dx");
        let dx_pass = dx_err < 1e-4;

        let pass = dw_pass && dx_pass;
        println!("  n={} o={} k={}: dw_err={:.2e} dx_err={:.2e} {}",
            n, o, k, dw_err, dx_err, if pass { "PASS" } else { "FAIL" });
        if !pass { all_pass = false; }
    }
    all_pass
}

fn main() {
    println!("=== remu kernel test suite ===");

    let mut all_pass = true;
    all_pass &= test_matmul_blocked();
    all_pass &= test_matvec_tiled();
    all_pass &= test_superlinear();
    all_pass &= test_glu();
    all_pass &= test_silu();
    all_pass &= test_layer_norm_simple();
    all_pass &= test_layer_norm();
    all_pass &= test_trace_shift();
    all_pass &= test_outer_product();
    all_pass &= test_sgd_update();
    all_pass &= test_superlinear_bwd();
    all_pass &= test_matvec_t();
    all_pass &= test_silu_bwd();
    all_pass &= test_glu_bwd();
    all_pass &= test_ln_bwd();
    all_pass &= test_per_neuron_glu_bwd();
    all_pass &= test_reduce_l2();

    println!();
    if all_pass {
        println!("ALL KERNEL TESTS PASSED");
    } else {
        println!("SOME TESTS FAILED");
        std::process::exit(1);
    }
}
