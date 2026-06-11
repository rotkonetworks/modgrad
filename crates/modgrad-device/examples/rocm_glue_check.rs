//! Isolation-validate the new resident-engine kernels (sdpa, q6k, rope, geglu)
//! against CPU references — TINY synthetic data, no model load. Low GPU risk.
//!
//! cargo run --release -p modgrad-device --features rocm --example rocm_glue_check

#[cfg(all(feature = "rocm", modgrad_hipcc_kernels))]
fn main() {
    use modgrad_device::backend::rocm::{HipBuffer, kern};

    let up = |v: &[f32]| { let b = HipBuffer::new(v.len() * 4).unwrap(); b.copy_from_host(v).unwrap(); b };
    let dn = |b: &HipBuffer, n: usize| { let mut o = vec![0f32; n]; b.copy_to_host(&mut o).unwrap(); o };
    let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);

    // ── geglu ──
    {
        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 3.0).collect();
        let upv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos() * 2.0).collect();
        let gd = up(&gate); let ud = up(&upv); let od = HipBuffer::new(n * 4).unwrap();
        unsafe { kern::geglu(gd.f32_ptr(), ud.f32_ptr(), od.f32_at(0), n).unwrap(); }
        kern::hip_sync().unwrap();
        let got = dn(&od, n);
        let cpu: Vec<f32> = (0..n).map(|i| {
            let v = gate[i]; let c = 0.7978845608f32 * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + c.tanh()) * upv[i]
        }).collect();
        println!("geglu     maxabs={:.6}", maxabs(&got, &cpu));
    }

    // ── rope (NEOX) ──
    {
        let nh = 4; let hd = 64; let pos = 5; let base = 10000.0f32;
        let data: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.03).sin()).collect();
        let dd = up(&data);
        unsafe { kern::rope_neox(dd.f32_at(0), pos, nh, hd, base).unwrap(); }
        kern::hip_sync().unwrap();
        let got = dn(&dd, nh * hd);
        let mut cpu = data.clone();
        let half = hd / 2;
        for h in 0..nh { for i in 0..half {
            let freq = base.powf(-2.0 * i as f32 / hd as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
            let off = h * hd; let x0 = data[off + i]; let x1 = data[off + i + half];
            cpu[off + i] = x0 * c - x1 * s; cpu[off + i + half] = x0 * s + x1 * c;
        }}
        println!("rope      maxabs={:.6}", maxabs(&got, &cpu));
    }

    // ── sdpa (decode) ──
    {
        let nh = 4; let nkv = 2; let hd = 64; let seq = 8; let kv_dim = nkv * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.05).sin()).collect();
        let kc: Vec<f32> = (0..seq * kv_dim).map(|i| (i as f32 * 0.013).cos()).collect();
        let vc: Vec<f32> = (0..seq * kv_dim).map(|i| (i as f32 * 0.017).sin()).collect();
        let qd = up(&q); let kd = up(&kc); let vd = up(&vc); let od = HipBuffer::new(nh * hd * 4).unwrap();
        unsafe { kern::sdpa_decode(qd.f32_ptr(), kd.f32_ptr(), vd.f32_ptr(), od.f32_at(0),
            nh, nkv, hd, seq, kv_dim, scale, 0).unwrap(); }
        kern::hip_sync().unwrap();
        let got = dn(&od, nh * hd);
        // CPU reference
        let mut cpu = vec![0f32; nh * hd];
        let grp = nh / nkv;
        for h in 0..nh {
            let kv_h = h / grp;
            let mut sc = vec![0f32; seq];
            for t in 0..seq {
                let mut d = 0f32;
                for e in 0..hd { d += q[h * hd + e] * kc[t * kv_dim + kv_h * hd + e]; }
                sc[t] = d * scale;
            }
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0f32; for s in sc.iter_mut() { *s = (*s - m).exp(); sum += *s; }
            for e in 0..hd {
                let mut a = 0f32;
                for t in 0..seq { a += sc[t] / sum * vc[t * kv_dim + kv_h * hd + e]; }
                cpu[h * hd + e] = a;
            }
        }
        println!("sdpa      maxabs={:.6}", maxabs(&got, &cpu));
    }

    // ── q6k matvec (a few rows of real token_embd to exercise the format) ──
    {
        use std::io::Cursor;
        use modgrad_device::kfd::gguf::{GgufFile, dequantize_row_q6_k, GgmlType};
        let path = "/home/alice/Downloads/Gemma-4-12B-OBLITERATED.Q4_K_S.gguf";
        if let Ok(file) = std::fs::read(path) {
            let g = GgufFile::parse(&mut Cursor::new(&file)).unwrap();
            let info = &g.tensors["token_embd.weight"];
            if matches!(info.dtype, GgmlType::Q6_K) {
                let in_dim = info.dims[0]; let bpr = in_dim / 256; let rb = bpr * 210;
                let out_dim = 96usize; // only a few rows — tiny upload
                let start = g.data_offset + info.offset;
                let wb = &file[start..start + out_dim * rb];
                let w = HipBuffer::new(wb.len()).unwrap(); w.copy_from_host_bytes(wb).unwrap();
                let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.02).sin() * 5.0).collect();
                let xd = up(&x); let yd = HipBuffer::new(out_dim * 4).unwrap();
                unsafe { kern::matvec_q6k(w.u8_ptr(), xd.f32_ptr(), yd.f32_at(0), out_dim, bpr).unwrap(); }
                kern::hip_sync().unwrap();
                let got = dn(&yd, out_dim);
                let mut cpu = vec![0f32; out_dim];
                for r in 0..out_dim {
                    let mut wr = vec![0f32; in_dim];
                    dequantize_row_q6_k(&wb[r * rb..(r + 1) * rb], &mut wr, bpr);
                    let mut a = 0f64; for k in 0..in_dim { a += wr[k] as f64 * x[k] as f64; }
                    cpu[r] = a as f32;
                }
                println!("q6k       maxabs={:.6}", maxabs(&got, &cpu));
            }
        } else { println!("q6k       (model absent, skipped)"); }
    }
    println!("done — all kernels exercised on tiny data.");
}

#[cfg(not(all(feature = "rocm", modgrad_hipcc_kernels)))]
fn main() { eprintln!("build with --features rocm"); }
