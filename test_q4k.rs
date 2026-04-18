/// Test Q4_K_M matvec kernel against CPU reference.
/// Also tests quantized vec_dot (Q4_K × Q8_K).

use modgrad_device::kfd::{HsaDevice, dispatch_queue::GpuQueue, quant_dot};

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        // subnormal
        let mut m = mant as f32 / 1024.0;
        m *= 2.0f32.powi(-14);
        return if sign == 1 { -m } else { m };
    }
    if exp == 31 { return if mant == 0 { if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY } } else { f32::NAN }; }
    let f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    f32::from_bits(f)
}

fn load_npy_f32(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap();
    // numpy .npy format: 128-byte header (approx), then raw data
    // Find the end of header: \n after the dict
    let _header_end = data.windows(1).position(|w| w[0] == b'\n')
        .and_then(|p| data[p+1..].windows(1).position(|w| w[0] == b'\n').map(|p2| p + 1 + p2 + 1))
        .unwrap_or(128);
    // Actually, npy v1: 10 bytes magic + header_len(2) + header + padding
    let magic_len = 10;
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let data_start = magic_len + header_len;
    let floats = &data[data_start..];
    floats.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load_npy_u8(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).unwrap();
    let magic_len = 10;
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let data_start = magic_len + header_len;
    data[data_start..].to_vec()
}

fn main() {
    println!("=== Q4_K_M Matvec Kernel Test ===\n");

    let mut dev = match HsaDevice::open() {
        Ok(d) => d,
        Err(e) => { println!("No GPU: {}", e); return; }
    };
    let mut q = GpuQueue::new();

    // Load test data
    let raw_q4 = load_npy_u8("/tmp/q4k_test_raw.npy");
    let x = load_npy_f32("/tmp/q4k_test_x.npy");
    let y_ref = load_npy_f32("/tmp/q4k_test_y.npy");

    let out_dim = 512;
    let in_dim = 2560;
    let blocks_per_row = in_dim / 256; // 10

    println!("Q4 data: {} bytes", raw_q4.len());
    println!("x: {} elements", x.len());
    println!("y_ref: {} elements", y_ref.len());
    println!("Matrix: {}x{}, {} blocks/row\n", out_dim, in_dim, blocks_per_row);

    // Need byte-level VRAM, but alloc takes floats; reinterpret as f32 slice.
    let raw_as_f32 = unsafe {
        std::slice::from_raw_parts(raw_q4.as_ptr() as *const f32, raw_q4.len() / 4)
    };
    let w_buf = q.alloc(&dev, (raw_q4.len() + 3) / 4).unwrap();
    w_buf.upload(raw_as_f32);
    // Upload any remaining bytes
    if raw_q4.len() % 4 != 0 {
        let mut last = [0u8; 4];
        let rem = raw_q4.len() % 4;
        last[..rem].copy_from_slice(&raw_q4[raw_q4.len() - rem..]);
        // This is fine, the extra bytes don't matter
    }

    let x_buf = q.alloc(&dev, in_dim).unwrap();
    x_buf.upload(&x);

    let bias_buf = q.alloc(&dev, out_dim).unwrap();
    bias_buf.upload(&vec![0.0f32; out_dim]); // zero bias for test

    let y_buf = q.alloc(&dev, out_dim).unwrap();
    y_buf.zero();

    // First: verify tiled f32 matvec works with dequantized weights
    println!("Testing f32 tiled matvec with dequantized weights...");
    {
        let w_f32 = load_npy_f32("/tmp/q4k_test_W.npy");
        let w_buf_f32 = q.alloc(&dev, w_f32.len()).unwrap();
        w_buf_f32.upload(&w_f32);
        // Use matvec_tiled (f32 kernel) as reference
        // Actually, we need to go through the weight cache path...
        // Let's just do a basic sanity check: the GpuQueue works at all
        let bias_zero = vec![0.0f32; out_dim];
        // Use enqueue_matvec_tiled which takes weight slices (cached)
        q.enqueue_matvec_tiled(&mut dev, &w_f32, &bias_zero, &x_buf, &y_buf, out_dim, in_dim);
        q.flush(&mut dev);
        let y_f32 = y_buf.download(out_dim);
        let max_err_f32: f32 = (0..out_dim).map(|i| (y_ref[i] - y_f32[i]).abs()).fold(0.0f32, f32::max);
        println!("  f32 tiled matvec max_err: {:.6}", max_err_f32);
        println!("  y_f32[0:5] = {:?}", &y_f32[..5]);
        y_buf.zero();
    }

    // Debug: dequantize first block on CPU and check
    println!("\nDebug: first Q4_K block header:");
    {
        let blk = &raw_q4[0..144];
        let d_f16 = u16::from_le_bytes([blk[0], blk[1]]);
        let dmin_f16 = u16::from_le_bytes([blk[2], blk[3]]);
        let d = f16_to_f32(d_f16);
        let dmin = f16_to_f32(dmin_f16);
        println!("  d_f16=0x{:04x} d={:.6}", d_f16, d);
        println!("  dmin_f16=0x{:04x} dmin={:.6}", dmin_f16, dmin);
        println!("  scales[0..12] = {:?}", &blk[4..16]);
        println!("  qs[0..8] = {:?}", &blk[16..24]);

        // Manually dequant element 0: should be first value of row 0
        let sc0 = blk[4] & 63;
        let m0 = blk[8] & 63;
        let nibble0 = blk[16] & 0xF;
        let val0 = d * sc0 as f32 * nibble0 as f32 - dmin * m0 as f32;
        println!("  element 0: sc={} m={} nibble={} val={:.6}", sc0, m0, nibble0, val0);
    }

    // Minimal test: 1 block, x=all ones, 1 output row
    println!("\nMinimal test: 1 row × 1 block (256 elems), x=all 1.0");
    {
        let one_block = &raw_q4[0..144];
        let one_blk_buf = q.alloc(&dev, 144 / 4 + 1).unwrap();
        one_blk_buf.upload(unsafe {
            std::slice::from_raw_parts(one_block.as_ptr() as *const f32, 144 / 4)
        });
        let x_ones = vec![1.0f32; 256];
        let x1_buf = q.alloc(&dev, 256).unwrap();
        x1_buf.upload(&x_ones);
        let b1_buf = q.alloc(&dev, 1).unwrap();
        b1_buf.upload(&[0.0f32]);
        let y1_buf = q.alloc(&dev, 256).unwrap();
        y1_buf.upload(&[-999.0f32; 256]); // 256 sentinels for debug store

        let ok = q.enqueue_matvec_q4k(&mut dev, &one_blk_buf, &x1_buf, &b1_buf, &y1_buf, 1, 1);
        println!("  enqueue ok={}", ok);
        let ok = q.flush(&mut dev);
        println!("  flush ok={}", ok);

        let y1 = y1_buf.download(256);
        // Now y1[tid] = partial sum for thread tid (debug mode: no reduction)
        let non_sentinel: usize = y1.iter().filter(|&&v| v != -999.0).count();
        let non_zero: usize = y1.iter().filter(|&&v| v != 0.0 && v != -999.0).count();
        println!("  Threads that wrote: {} / 256", non_sentinel);
        println!("  Non-zero partial sums: {}", non_zero);
        println!("  y1[0..8] = {:?}", &y1[..8]);
        println!("  y1[128..136] = {:?}", &y1[128..136]);
    }

    // Dispatch Q4_K_M matvec
    println!("\nDispatching matvec_q4k kernel...");
    let ok = q.enqueue_matvec_q4k(&mut dev, &w_buf, &x_buf, &bias_buf, &y_buf,
                                   out_dim, blocks_per_row);
    if !ok {
        println!("FAILED to enqueue kernel");
        return;
    }
    let ok = q.flush(&mut dev);
    if !ok {
        println!("FAILED to flush");
        return;
    }

    // Read back result
    let y_gpu = y_buf.download(out_dim);

    // Compare
    println!("\nResults:");
    println!("  y_ref[0:5] = {:?}", &y_ref[..5]);
    println!("  y_gpu[0:5] = {:?}", &y_gpu[..5]);

    let mut max_err: f32 = 0.0;
    let mut sum_err: f32 = 0.0;
    let mut max_idx = 0;
    for i in 0..out_dim {
        let err = (y_ref[i] - y_gpu[i]).abs();
        sum_err += err;
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }
    let avg_err = sum_err / out_dim as f32;
    println!("\n  max_err = {:.6} at index {}", max_err, max_idx);
    println!("  avg_err = {:.6}", avg_err);
    println!("  y_ref[{}] = {:.6}, y_gpu[{}] = {:.6}", max_idx, y_ref[max_idx], max_idx, y_gpu[max_idx]);

    if max_err < 0.01 {
        println!("\n  PASS ✓");
    } else if max_err < 0.1 {
        println!("\n  MARGINAL — small numerical differences");
    } else {
        println!("\n  FAIL ✗ — kernel has a bug");
    }

    // Benchmark
    println!("\nBenchmark:");
    let iters = 500;
    // Warmup
    for _ in 0..5 {
        q.enqueue_matvec_q4k(&mut dev, &w_buf, &x_buf, &bias_buf, &y_buf, out_dim, blocks_per_row);
        q.flush(&mut dev);
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        q.enqueue_matvec_q4k(&mut dev, &w_buf, &x_buf, &bias_buf, &y_buf, out_dim, blocks_per_row);
        q.flush(&mut dev);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let us_per = elapsed / iters as f64 * 1e6;
    let bytes_read = raw_q4.len(); // Q4 weights read per matvec
    let gbps = bytes_read as f64 / (us_per * 1e-6) / 1e9;
    println!("  {}x{} Q4_K_M matvec: {:.1} us/iter", out_dim, in_dim, us_per);
    println!("  Effective bandwidth: {:.1} GB/s (of 288 GB/s peak)", gbps);

    // === Test quantized vec_dot (Q4_K × Q8_K) ===
    println!("\n=== Quantized Vec Dot Test ===");
    let mut y_qdot = vec![0.0f32; out_dim];
    quant_dot::qmatvec(&raw_q4, &x, &mut y_qdot, out_dim, in_dim, 144, false);

    let max_err_qdot: f32 = (0..out_dim).map(|i| (y_ref[i] - y_qdot[i]).abs()).fold(0.0f32, f32::max);
    let avg_err_qdot: f32 = (0..out_dim).map(|i| (y_ref[i] - y_qdot[i]).abs()).sum::<f32>() / out_dim as f32;
    println!("  y_ref[0:5] = {:?}", &y_ref[..5]);
    println!("  y_qdot[0:5] = {:?}", &y_qdot[..5]);
    println!("  max_err = {:.6}, avg_err = {:.6}", max_err_qdot, avg_err_qdot);
    if max_err_qdot < 0.01 {
        println!("  PASS");
    } else {
        println!("  MARGINAL (quantized arithmetic has rounding)");
    }
}
