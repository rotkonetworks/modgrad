//! CIFAR-10 loader — reads the .feat binary format.
//!
//! Format: "FEAT" [4 bytes] + n_samples [u32] + n_features [u32] + n_classes [u32]
//!         then n_samples × (n_features f32 pixels + 1 f32 label).
//! Pixels are already normalized to [0, 1] as f32, CHW order (3×32×32).

use std::io::{self};
use std::path::Path;

/// One CIFAR-10 image.
pub struct CifarImage {
    /// Pixel data: 3072 f32s (3×32×32, CHW, values in [0,1]).
    pub pixels: Vec<f32>,
    /// Class label: 0-9.
    pub label: usize,
}

/// Load CIFAR-10 from .feat binary file.
pub fn load_feat(path: impl AsRef<Path>) -> io::Result<Vec<CifarImage>> {
    let data = std::fs::read(path)?;
    if data.len() < 16 || &data[0..4] != b"FEAT" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a FEAT file"));
    }

    let n_samples = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let n_features = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let _n_classes = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

    let record_floats = n_features + 1; // features + label
    let record_bytes = record_floats * 4;
    let expected = 16 + n_samples * record_bytes;
    if data.len() < expected {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("file too short: {} < {}", data.len(), expected)));
    }

    let mut images = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let offset = 16 + i * record_bytes;
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                data[offset..].as_ptr() as *const f32,
                record_floats,
            )
        };
        let pixels = floats[..n_features].to_vec();
        let label = floats[n_features] as usize;
        images.push(CifarImage { pixels, label });
    }

    Ok(images)
}

/// Extract 4×4 patches from a 32×32×3 image (CHW format).
/// Returns 64 patches, each 48 floats (4×4×3).
pub fn extract_patches_4x4(pixels: &[f32]) -> Vec<f32> {
    debug_assert_eq!(pixels.len(), 3072); // 3×32×32
    let mut patches = Vec::with_capacity(64 * 48);
    // pixels layout: [C, H, W] = [3, 32, 32]
    for py in 0..8 {
        for px in 0..8 {
            // Extract 4×4 patch across all 3 channels
            for c in 0..3 {
                for dy in 0..4 {
                    for dx in 0..4 {
                        let y = py * 4 + dy;
                        let x = px * 4 + dx;
                        patches.push(pixels[c * 1024 + y * 32 + x]);
                    }
                }
            }
        }
    }
    patches
}

/// CIFAR-10 class names.
pub const CLASSES: [&str; 10] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];
