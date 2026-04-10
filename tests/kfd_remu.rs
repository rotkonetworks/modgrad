//! GPU kernel validation via remu (RDNA3 ISA emulator).
//!
//! Runs matmul kernels in software before sending to hardware.
//! No GPU required — catches OOB accesses, wrong SGPRs, etc.
//!
//! run: cargo test --test kfd_remu -- --nocapture

use remu::work_group::WorkGroup;

/// Extract .text section from an AMDGPU ELF code object (.co).
/// Returns (instructions, code_offset_in_file).
fn extract_text(co: &[u8]) -> Vec<u32> {
    // Minimal ELF64 parser — just find .text section
    assert!(co.len() > 64, "too small for ELF");
    assert!(&co[0..4] == b"\x7fELF", "not ELF");
    assert!(co[4] == 2, "not ELF64");

    let shoff = u64::from_le_bytes(co[40..48].try_into().unwrap()) as usize;
    let shentsize = u16::from_le_bytes(co[58..60].try_into().unwrap()) as usize;
    let shnum = u16::from_le_bytes(co[60..62].try_into().unwrap()) as usize;
    let shstrndx = u16::from_le_bytes(co[62..64].try_into().unwrap()) as usize;

    // Find string table for section names
    let strtab_off = shoff + shstrndx * shentsize;
    let str_offset = u64::from_le_bytes(co[strtab_off + 24..strtab_off + 32].try_into().unwrap()) as usize;

    for i in 0..shnum {
        let sh = shoff + i * shentsize;
        let name_idx = u32::from_le_bytes(co[sh..sh + 4].try_into().unwrap()) as usize;
        let name_start = str_offset + name_idx;
        let name_end = co[name_start..].iter().position(|&b| b == 0).unwrap() + name_start;
        let name = std::str::from_utf8(&co[name_start..name_end]).unwrap_or("");

        if name == ".text" {
            let offset = u64::from_le_bytes(co[sh + 24..sh + 32].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(co[sh + 32..sh + 40].try_into().unwrap()) as usize;
            let text = &co[offset..offset + size];
            assert!(size % 4 == 0, ".text not 4-byte aligned");
            return text
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
    }
    panic!(".text section not found");
}

/// Pack matmul kernargs: W(u64) B(u64) X(u64) Y(u64) M(u32) K(u32) N(u32)
fn pack_kernargs(w: *const f32, b: *const f32, x: *const f32, y: *mut f32, m: u32, k: u32, n: u32) -> Vec<u64> {
    let mut args = vec![0u64; 6]; // 48 bytes = 6 u64s
    args[0] = w as u64;
    args[1] = b as u64;
    args[2] = x as u64;
    args[3] = y as u64;
    // M and K packed into one u64 (little-endian: M=low, K=high)
    args[4] = (m as u64) | ((k as u64) << 32);
    // N in low 32 bits
    args[5] = n as u64;
    args
}

fn run_matmul(co: &[u8], w: &[f32], b: &[f32], x: &[f32], y: &mut [f32], m: u32, k: u32, n: u32) {
    run_matmul_tiled(co, w, b, x, y, m, k, n, 32, 8);
}

fn run_matmul_blocked(co: &[u8], w: &[f32], b: &[f32], x: &[f32], y: &mut [f32], m: u32, k: u32, n: u32) {
    run_matmul_tiled(co, w, b, x, y, m, k, n, 128, 32);
}

fn run_matmul_tiled(co: &[u8], w: &[f32], b: &[f32], x: &[f32], y: &mut [f32], m: u32, k: u32, n: u32, tm: u32, tn: u32) {
    let kernel = extract_text(co);
    let args = pack_kernargs(w.as_ptr(), b.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m, k, n);

    let num_wg_m = (m + tm - 1) / tm;
    let num_wg_n = (n + tn - 1) / tn;
    let num_wg = num_wg_m * num_wg_n;

    for wg_id in 0..num_wg {
        let mut wg = WorkGroup::new(1, [wg_id, 0, 0], [256, 1, 1], &kernel, args.as_ptr());
        wg.exec_waves().expect(&format!("kernel fault in workgroup {wg_id}"));
    }
}

/// Reference matmul: Y = X @ W^T + B
fn reference_matmul(w: &[f32], b: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * m];
    for ni in 0..n {
        for mi in 0..m {
            let mut sum = b[mi];
            for ki in 0..k {
                sum += x[ni * k + ki] * w[mi * k + ki];
            }
            y[ni * m + mi] = sum;
        }
    }
    y
}

#[test]
fn test_matmul_identity_32x32() {
    let co = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul.co");
    let m = 32u32;
    let k = 32u32;
    let n = 8u32;

    // W = identity, B = 0.5, X = sequential
    let mut w = vec![0.0f32; (m * k) as usize];
    for i in 0..m.min(k) {
        w[(i * k + i) as usize] = 1.0;
    }
    let b = vec![0.5f32; m as usize];
    let x: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.1).collect();
    let mut y = vec![0.0f32; (n * m) as usize];

    run_matmul(co, &w, &b, &x, &mut y, m, k, n);

    let expected = reference_matmul(&w, &b, &x, m as usize, k as usize, n as usize);
    for i in 0..y.len() {
        assert!(
            (y[i] - expected[i]).abs() < 1e-3,
            "mismatch at {i}: got {}, expected {}",
            y[i],
            expected[i]
        );
    }
    println!("PASS: 32x32 batch=8 identity matmul (remu)");
}

#[test]
fn test_matmul_random_64x64() {
    let co = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul.co");
    let m = 64u32;
    let k = 64u32;
    let n = 8u32;

    // Deterministic pseudo-random
    let mut rng = 12345u64;
    let mut randf = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let w: Vec<f32> = (0..m * k).map(|_| randf() * 0.1).collect();
    let b = vec![0.0f32; m as usize];
    let x: Vec<f32> = (0..n * k).map(|_| randf()).collect();
    let mut y = vec![0.0f32; (n * m) as usize];

    run_matmul(co, &w, &b, &x, &mut y, m, k, n);

    let expected = reference_matmul(&w, &b, &x, m as usize, k as usize, n as usize);
    let max_err = y
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // FP32 accumulation over K=64 terms can have ~1e-4 relative error
    // with values in [-1,1] range, absolute error can reach ~0.01 per term
    let max_abs = expected.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_err = max_err / max_abs.max(1e-6);
    println!("64x64 batch=8 max_err={max_err:.6} max_abs={max_abs:.2} rel_err={rel_err:.6}");
    assert!(rel_err < 1e-4, "relative error too large: {rel_err}");
    println!("PASS: 64x64 batch=8 random matmul (remu)");
}

// ============ Register-blocked kernel tests ============

#[test]
fn test_blocked_identity_128x128() {
    let co = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");
    let m = 128u32;
    let k = 8u32; // one tile iteration
    let n = 32u32;

    let mut w = vec![0.0f32; (m * k) as usize];
    for i in 0..k {
        w[(i * k + i) as usize] = 1.0;
    }
    let b = vec![0.5f32; m as usize];
    let x: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
    let mut y = vec![0.0f32; (n * m) as usize];

    run_matmul_blocked(co, &w, &b, &x, &mut y, m, k, n);

    let expected = reference_matmul(&w, &b, &x, m as usize, k as usize, n as usize);
    let max_err = y.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("blocked 128x8 batch=32: max_err={max_err:.6}");
    assert!(max_err < 1e-3, "max error too large: {max_err}");
    println!("PASS: blocked 128x8 batch=32 identity (remu)");
}

#[test]
fn test_blocked_random_256x256() {
    let co = include_bytes!("../crates/modgrad-device/src/kfd/kernels/matmul_blocked.co");
    let m = 256u32;
    let k = 64u32; // multiple tile iterations (64/8=8)
    let n = 32u32;

    let mut rng = 42u64;
    let mut randf = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let w: Vec<f32> = (0..m * k).map(|_| randf() * 0.1).collect();
    let b = vec![0.0f32; m as usize];
    let x: Vec<f32> = (0..n * k).map(|_| randf()).collect();
    let mut y = vec![0.0f32; (n * m) as usize];

    run_matmul_blocked(co, &w, &b, &x, &mut y, m, k, n);

    let expected = reference_matmul(&w, &b, &x, m as usize, k as usize, n as usize);
    let max_err = y.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs = expected.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let rel_err = max_err / max_abs.max(1e-6);
    println!("blocked 256x64 batch=32: max_err={max_err:.6} rel_err={rel_err:.6}");
    assert!(rel_err < 1e-4, "relative error too large: {rel_err}");
    println!("PASS: blocked 256x64 batch=32 random (remu)");
}
