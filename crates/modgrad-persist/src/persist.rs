//! Unified weight persistence — wincode binary, JSON fallback.
//!
//! All weight types should use these functions instead of ad-hoc
//! serde_json calls. Binary (wincode) is 10× faster and 3× smaller
//! than JSON, and writes directly into the final heap allocation
//! on deserialize — no intermediate staging buffer.
//!
//! File format detection: `.bin` → wincode, `.json` → JSON.
//! Unknown extension → binary (the default for new code).
//!
//! **Wire-compat note**: wincode produces identical bytes to
//! bincode 1.x's default encoding, so any `.bin` files that
//! existed under the earlier bincode implementation still load
//! correctly via this wincode path.
//!
//! Every persisted type needs both `#[derive(Serialize, Deserialize)]`
//! (for JSON + foreign-type interop) and `#[derive(SchemaRead,
//! SchemaWrite)]` (for wincode). The derives stack.

use serde::{Serialize, de::DeserializeOwned};
use std::io;
use std::path::Path;
use wincode::config::Configuration;
use wincode::len::UseIntLen;

/// Magic header for isis binary files. Prevents deserializing
/// random data or files from other programs as brain weights.
const MAGIC: &[u8; 4] = b"ISIS";
const VERSION: u8 = 1;

/// Wincode config used for all modgrad persistence. Raises the
/// preallocation-size ceiling from wincode's default 4 MiB (intended
/// as an anti-DoS bound for deserializing untrusted input) to 16 GiB
/// — plenty for a 1B-param f32 model + optimizer state, and aligned
/// with the 32 GiB cap applied at the checkpoint-bundle layer. The
/// zero-copy + LEN-encoding bits stay at their bincode-compatible
/// defaults so existing .bin files still round-trip.
pub type ModgradConfig = Configuration<
    true,
    { 16 * 1024 * 1024 * 1024 }, // 16 GiB
    UseIntLen<u64, 0>,
>;

/// Instance used in the hot path. Created as a const so monomorphised
/// copies fold at compile time rather than per call.
const MODGRAD_CONFIG: ModgradConfig = Configuration::new();

/// Save any type to a file.
/// Format selected by extension: `.json` → JSON, anything else → binary.
/// Binary files get a 5-byte header (magic + version) for integrity.
pub fn save<T>(value: &T, path: impl AsRef<Path>) -> io::Result<()>
where
    T: Serialize + wincode::SchemaWrite<ModgradConfig, Src = T>,
{
    let path = path.as_ref();
    if is_json(path) {
        let data = serde_json::to_string(value)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, data)
    } else {
        let payload = wincode::config::serialize(value, MODGRAD_CONFIG)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("wincode serialize: {e:?}")))?;
        let mut data = Vec::with_capacity(5 + payload.len());
        data.extend_from_slice(MAGIC);
        data.push(VERSION);
        data.extend_from_slice(&payload);
        std::fs::write(path, data)
    }
}

/// Load any type from a file.
/// Format selected by extension: `.json` → JSON, anything else → binary.
/// Binary files must have the ISIS magic header. Falls back to JSON for legacy.
pub fn load<T>(path: impl AsRef<Path>) -> io::Result<T>
where
    T: DeserializeOwned
        + for<'de> wincode::SchemaRead<'de, ModgradConfig, Dst = T>,
{
    let path = path.as_ref();
    let data = std::fs::read(path)?;

    if is_json(path) {
        serde_json::from_slice(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    } else if data.len() >= 5 && &data[..4] == MAGIC {
        let version = data[4];
        if version > VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("file version {} > supported {}", version, VERSION)));
        }
        wincode::config::deserialize(&data[5..], MODGRAD_CONFIG)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("wincode deserialize: {e:?}")))
    } else {
        // No magic header — try JSON fallback for legacy files
        serde_json::from_slice(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                format!("not an ISIS binary file and invalid JSON: {}", e)))
    }
}

/// File size estimate for a serialized value (bincode).
pub fn estimated_size<T: Serialize>(value: &T) -> usize {
    bincode::serialized_size(value).unwrap_or(0) as usize
}

fn is_json(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map_or(false, |e| e.eq_ignore_ascii_case("json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use wincode_derive::{SchemaRead, SchemaWrite};

    #[derive(Debug, PartialEq, Serialize, Deserialize, SchemaRead, SchemaWrite)]
    struct TestWeights {
        data: Vec<f32>,
        name: String,
    }

    /// Wire-compat guard: a file written via the new wincode-based
    /// `save` must be readable by a legacy bincode deserializer, and
    /// vice versa. If this test ever fails, someone changed the
    /// binary encoding in a way that breaks existing .bin files.
    #[derive(Debug, PartialEq, Serialize, Deserialize, SchemaRead, SchemaWrite)]
    struct WireCompatProbe {
        data: Vec<f32>,
        vocab: u32,
        d_model: u32,
    }

    #[test]
    fn wincode_bytes_match_bincode_bytes() {
        let w = WireCompatProbe {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vocab: 256,
            d_model: 128,
        };
        let bin_bytes = bincode::serialize(&w).unwrap();
        let win_bytes = wincode::serialize(&w).unwrap();
        assert_eq!(bin_bytes, win_bytes,
            "wincode must produce byte-identical output to bincode default \
             encoding — existing .bin files depend on it");
    }

    #[test]
    fn load_reads_bincode_written_files() {
        // Write a file via bincode directly (simulating what an older
        // checkpoint on disk looks like), then make sure our new
        // wincode-based `load` reads it cleanly.
        let w = WireCompatProbe {
            data: vec![0.1, 0.2, 0.3],
            vocab: 512,
            d_model: 64,
        };
        let path = format!("/tmp/isis_test_load_reads_bincode_{}.bin",
            std::process::id());
        let payload = bincode::serialize(&w).unwrap();
        let mut bytes = Vec::with_capacity(5 + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, bytes).unwrap();

        let loaded: WireCompatProbe = load(&path).unwrap();
        assert_eq!(w, loaded);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bincode_roundtrip() {
        let w = TestWeights { data: vec![1.0, 2.0, 3.0], name: "test".into() };
        let path = "/tmp/isis_test_bincode.bin";
        save(&w, path).unwrap();
        let loaded: TestWeights = load(path).unwrap();
        assert_eq!(w, loaded);
        let size = std::fs::metadata(path).unwrap().len();
        std::fs::remove_file(path).ok();
        // Bincode should be much smaller than JSON
        assert!(size < 100, "bincode should be compact, got {} bytes", size);
    }

    #[test]
    fn json_roundtrip() {
        let w = TestWeights { data: vec![1.0, 2.0], name: "json".into() };
        let path = "/tmp/isis_test_json.json";
        save(&w, path).unwrap();
        let loaded: TestWeights = load(path).unwrap();
        assert_eq!(w, loaded);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn json_fallback_on_bin_extension() {
        // Save as JSON but with .bin extension — load should try bincode, fail, then JSON
        let w = TestWeights { data: vec![4.0], name: "fallback".into() };
        let path = "/tmp/isis_test_fallback.bin";
        // Write JSON manually with .bin extension
        let json = serde_json::to_string(&w).unwrap();
        std::fs::write(path, json).unwrap();
        // load should fall back to JSON
        let loaded: TestWeights = load(path).unwrap();
        assert_eq!(w, loaded);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn size_comparison() {
        let big = TestWeights {
            data: (0..10000).map(|i| i as f32 * 0.001).collect(),
            name: "large".into(),
        };
        let json_size = serde_json::to_string(&big).unwrap().len();
        let bin_size = estimated_size(&big);
        eprintln!("  JSON: {} bytes, bincode: {} bytes, ratio: {:.1}×",
            json_size, bin_size, json_size as f64 / bin_size as f64);
        assert!(bin_size < json_size / 2, "bincode should be at least 2× smaller");
    }
}
