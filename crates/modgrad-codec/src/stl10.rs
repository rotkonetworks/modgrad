//! STL-10 unlabeled split reader.
//!
//! STL-10 (Coates, Ng, Lee 2011) is a small dataset designed for
//! unsupervised / self-taught feature learning. We only use the
//! **unlabeled split** — 100,000 natural images at 96×96×3 — as
//! pretraining fuel for the cortical layers of [`crate::retina::VisualCortex`].
//!
//! # Download
//!
//! ```text
//! wget https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
//! tar xzf stl10_binary.tar.gz
//! # → stl10_binary/unlabeled_X.bin (2.63 GB)
//! ```
//!
//! Only `unlabeled_X.bin` is needed for this loader; the rest of the
//! tarball (`train_X.bin`, labels, etc.) is for the supervised splits
//! we don't use.
//!
//! # Binary format
//!
//! The file is a flat concatenation of images — no header, no
//! per-image framing. Each image is 96·96·3 = 27,648 bytes:
//!
//! ```text
//! [R plane: 96×96 u8 column-major]
//! [G plane: 96×96 u8 column-major]
//! [B plane: 96×96 u8 column-major]
//! ```
//!
//! "Column-major" means the first 96 bytes of a plane are column 0
//! (rows 0–95), then column 1, etc. — the MATLAB/Fortran convention.
//! Our downstream code expects row-major `[channel × row × col]`
//! layout, so [`Stl10UnlabeledReader::get`] transposes on read.
//!
//! # Streaming access
//!
//! The 2.6 GB file does not fit comfortably in RAM on an 8 GB-class
//! machine alongside the rest of training. This loader opens the file
//! once and `seek`s per image, so memory use is O(one image). Access
//! is random (any index, any order) — the caller owns shuffling.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

/// Reader for an STL-10 `unlabeled_X.bin` file.
///
/// Thread-safe — the internal file handle is mutex-guarded so multiple
/// rayon workers can pull images concurrently.
#[derive(Debug)]
pub struct Stl10UnlabeledReader {
    file: Mutex<File>,
    n_images: usize,
}

impl Stl10UnlabeledReader {
    pub const IMAGE_H: usize = 96;
    pub const IMAGE_W: usize = 96;
    pub const IMAGE_CHANNELS: usize = 3;
    pub const IMAGE_BYTES: usize = Self::IMAGE_H * Self::IMAGE_W * Self::IMAGE_CHANNELS;
    /// Size of the unlabeled split as shipped. Used as a sanity check;
    /// a partial download would not match.
    pub const EXPECTED_N_IMAGES: usize = 100_000;

    /// Open the file and verify its size is a whole multiple of one
    /// image's byte count.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size == 0 || size % Self::IMAGE_BYTES != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "STL-10 file size {} is not a multiple of {} (one image); \
                     file may be truncated or wrong format",
                    size, Self::IMAGE_BYTES,
                ),
            ));
        }
        let n_images = size / Self::IMAGE_BYTES;
        Ok(Self { file: Mutex::new(file), n_images })
    }

    pub fn len(&self) -> usize { self.n_images }

    pub fn is_empty(&self) -> bool { self.n_images == 0 }

    /// Read image at `index`, decode to flat `[3 × 96 × 96]` row-major
    /// `f32` in `[0.0, 1.0]`.
    pub fn get(&self, index: usize) -> io::Result<Vec<f32>> {
        if index >= self.n_images {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("index {} >= {} images", index, self.n_images),
            ));
        }
        let mut buf = vec![0u8; Self::IMAGE_BYTES];
        {
            let mut f = self.file.lock()
                .map_err(|_| io::Error::new(io::ErrorKind::Other,
                    "stl10: file mutex poisoned"))?;
            f.seek(SeekFrom::Start((index * Self::IMAGE_BYTES) as u64))?;
            f.read_exact(&mut buf)?;
        }
        Ok(decode_image(&buf))
    }
}

/// Decode one 27,648-byte STL-10 image blob into flat `[3 × 96 × 96]`
/// row-major `f32` in `[0, 1]`. Handles the column-major → row-major
/// transpose per channel.
fn decode_image(raw: &[u8]) -> Vec<f32> {
    let h = Stl10UnlabeledReader::IMAGE_H;
    let w = Stl10UnlabeledReader::IMAGE_W;
    let plane = h * w;
    debug_assert_eq!(raw.len(), 3 * plane);
    let mut out = vec![0.0f32; 3 * plane];
    for c in 0..3 {
        let src_plane = &raw[c * plane..(c + 1) * plane];
        let dst_plane = &mut out[c * plane..(c + 1) * plane];
        // src is column-major: src[col * h + row]
        // dst is row-major: dst[row * w + col]
        for col in 0..w {
            for row in 0..h {
                dst_plane[row * w + col] = src_plane[col * h + row] as f32 / 255.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a fake STL-10 file: N images, each with a known pixel
    /// pattern so we can verify the column-major→row-major transpose.
    fn fake_stl10(n_images: usize, seed: u64) -> Vec<u8> {
        let per = Stl10UnlabeledReader::IMAGE_BYTES;
        let mut out = vec![0u8; n_images * per];
        let mut s = seed;
        for b in out.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 32) as u8;
        }
        out
    }

    #[test]
    fn decode_image_matches_layout() {
        // Craft an image where each byte equals (c*10 + row_or_col)
        // so we can check transpose landed values where we expect.
        let h = Stl10UnlabeledReader::IMAGE_H;
        let w = Stl10UnlabeledReader::IMAGE_W;
        let plane = h * w;
        let mut raw = vec![0u8; 3 * plane];
        for c in 0..3 {
            for col in 0..w {
                for row in 0..h {
                    // Value encodes channel + col + row so we can find it
                    // after transpose.
                    let v = ((c * 31 + col * 7 + row) & 0xff) as u8;
                    raw[c * plane + col * h + row] = v;
                }
            }
        }
        let decoded = decode_image(&raw);
        assert_eq!(decoded.len(), 3 * plane);
        for c in 0..3 {
            for row in 0..h {
                for col in 0..w {
                    let expected = ((c * 31 + col * 7 + row) & 0xff) as f32 / 255.0;
                    let got = decoded[c * plane + row * w + col];
                    assert!((got - expected).abs() < 1e-6,
                        "c={c} row={row} col={col}: got={got} expected={expected}");
                }
            }
        }
    }

    #[test]
    fn reader_open_and_iterate_fake_file() {
        let tmp = std::env::temp_dir().join("modgrad_stl10_fake.bin");
        let n_images = 4;
        let data = fake_stl10(n_images, 42);
        {
            let mut f = File::create(&tmp).expect("create");
            f.write_all(&data).expect("write");
        }

        let reader = Stl10UnlabeledReader::open(&tmp).expect("open");
        assert_eq!(reader.len(), n_images);
        for i in 0..n_images {
            let img = reader.get(i).expect("get");
            assert_eq!(img.len(), 3 * 96 * 96);
            for v in &img {
                assert!(*v >= 0.0 && *v <= 1.0, "pixel {} out of range", v);
            }
        }

        // Out-of-range request is an error, not a panic.
        assert!(reader.get(n_images).is_err());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn reader_rejects_truncated_file() {
        let tmp = std::env::temp_dir().join("modgrad_stl10_truncated.bin");
        // One image minus one byte — not a multiple of IMAGE_BYTES.
        let data = vec![0u8; Stl10UnlabeledReader::IMAGE_BYTES - 1];
        std::fs::write(&tmp, &data).expect("write");
        let err = Stl10UnlabeledReader::open(&tmp).expect_err("should reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&tmp).ok();
    }
}
