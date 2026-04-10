//! Unified weight persistence — binary-first, JSON fallback.
//!
//! All weight types should use these functions instead of ad-hoc
//! serde_json calls. Binary (bincode) is 10× faster and 3× smaller.
//! JSON is kept as a human-readable fallback and for importing
//! legacy brain files.
//!
//! File format detection: `.bin` → bincode, `.json` → JSON.
//! Unknown extension → bincode (the default for new code).

use serde::{Serialize, de::DeserializeOwned};
use std::io;
use std::path::Path;

/// Magic header for isis binary files. Prevents deserializing
/// random data or files from other programs as brain weights.
const MAGIC: &[u8; 4] = b"ISIS";
const VERSION: u8 = 1;

/// Save any Serialize type to a file.
/// Format selected by extension: `.json` → JSON, anything else → bincode.
/// Binary files get a 5-byte header (magic + version) for integrity.
pub fn save<T: Serialize>(value: &T, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    if is_json(path) {
        let data = serde_json::to_string(value)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, data)
    } else {
        let payload = bincode::serialize(value)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut data = Vec::with_capacity(5 + payload.len());
        data.extend_from_slice(MAGIC);
        data.push(VERSION);
        data.extend_from_slice(&payload);
        std::fs::write(path, data)
    }
}

/// Load any DeserializeOwned type from a file.
/// Format selected by extension: `.json` → JSON, anything else → bincode.
/// Binary files must have the ISIS magic header. Falls back to JSON for legacy.
pub fn load<T: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
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
        bincode::deserialize(&data[5..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct TestWeights {
        data: Vec<f32>,
        name: String,
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
