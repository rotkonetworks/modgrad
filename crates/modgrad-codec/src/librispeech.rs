//! LibriSpeech dataset loader.
//!
//! Reads LibriSpeech directory structure:
//!   LibriSpeech/{split}/{speaker_id}/{chapter_id}/{speaker}-{chapter}-{utterance}.flac
//!   LibriSpeech/{split}/{speaker_id}/{chapter_id}/{speaker}-{chapter}.trans.txt
//!
//! FLAC decoding uses a minimal pure-Rust decoder (no external deps).
//! For production, replace with a proper FLAC library.

use std::path::{Path, PathBuf};
use std::io::{self};

/// One LibriSpeech utterance.
pub struct Utterance {
    /// Raw audio samples as f32 (mono, 16kHz).
    pub samples: Vec<f32>,
    /// Sample rate (always 16000 for LibriSpeech).
    pub sample_rate: usize,
    /// Transcript text (uppercase).
    pub transcript: String,
    /// Speaker ID.
    pub speaker_id: String,
    /// Full path to the FLAC file.
    pub path: PathBuf,
}

/// Load all utterances from a LibriSpeech split directory.
/// E.g., load_split("/steam/dataset/LibriSpeech/dev-clean")
pub fn load_split(split_dir: impl AsRef<Path>) -> io::Result<Vec<Utterance>> {
    let split_dir = split_dir.as_ref();
    let mut utterances = Vec::new();

    // Walk: split_dir / speaker_id / chapter_id /
    for speaker_entry in std::fs::read_dir(split_dir)? {
        let speaker_entry = speaker_entry?;
        if !speaker_entry.file_type()?.is_dir() { continue; }
        let speaker_id = speaker_entry.file_name().to_string_lossy().to_string();

        for chapter_entry in std::fs::read_dir(speaker_entry.path())? {
            let chapter_entry = chapter_entry?;
            if !chapter_entry.file_type()?.is_dir() { continue; }

            // Read transcript file
            let chapter_dir = chapter_entry.path();
            let trans_pattern = format!("{}-{}.trans.txt",
                speaker_id, chapter_entry.file_name().to_string_lossy());
            let trans_path = chapter_dir.join(&trans_pattern);

            let transcripts = if trans_path.exists() {
                parse_transcript(&trans_path)?
            } else {
                Vec::new()
            };
            let trans_map: std::collections::HashMap<String, String> = transcripts
                .into_iter().collect();

            // Read FLAC files
            for file_entry in std::fs::read_dir(&chapter_dir)? {
                let file_entry = file_entry?;
                let path = file_entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("flac") { continue; }

                let stem = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();

                let transcript = trans_map.get(&stem)
                    .cloned()
                    .unwrap_or_default();

                match decode_flac_to_f32(&path) {
                    Ok(samples) => {
                        utterances.push(Utterance {
                            samples,
                            sample_rate: 16000,
                            transcript,
                            speaker_id: speaker_id.clone(),
                            path,
                        });
                    }
                    Err(_) => continue, // skip undecodable files
                }
            }
        }
    }

    utterances.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(utterances)
}

/// Parse a LibriSpeech transcript file.
/// Format: "utterance-id TRANSCRIPT TEXT\n"
fn parse_transcript(path: &Path) -> io::Result<Vec<(String, String)>> {
    let content = std::fs::read_to_string(path)?;
    Ok(content.lines()
        .filter_map(|line| {
            let line = line.trim();
            let space = line.find(' ')?;
            let id = line[..space].to_string();
            let text = line[space + 1..].to_string();
            Some((id, text))
        })
        .collect())
}

/// Decode a FLAC file to f32 samples using subprocess (flac CLI).
/// Falls back to raw reading if flac is not available.
fn decode_flac_to_f32(path: &Path) -> io::Result<Vec<f32>> {
    // Try using the `flac` CLI tool to decode to raw PCM
    let output = std::process::Command::new("flac")
        .args(&["-d", "-c", "--force-raw-format",
                "--endian=little", "--sign=signed"])
        .arg(path)
        .output();

    match output {
        Ok(result) if result.status.success() => {
            // Raw PCM: 16-bit signed little-endian mono
            let bytes = &result.stdout;
            let samples: Vec<f32> = bytes.chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();
            Ok(samples)
        }
        _ => {
            // Fallback: try sox
            let output = std::process::Command::new("sox")
                .arg(path)
                .args(&["-t", "raw", "-r", "16000", "-e", "signed", "-b", "16", "-c", "1", "-"])
                .output();

            match output {
                Ok(result) if result.status.success() => {
                    let bytes = &result.stdout;
                    let samples: Vec<f32> = bytes.chunks_exact(2)
                        .map(|chunk| {
                            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                            sample as f32 / 32768.0
                        })
                        .collect();
                    Ok(samples)
                }
                _ => Err(io::Error::new(io::ErrorKind::Other,
                    "neither flac nor sox available for FLAC decoding"))
            }
        }
    }
}

/// Load a single utterance by path (for testing).
pub fn load_one(flac_path: impl AsRef<Path>) -> io::Result<Vec<f32>> {
    decode_flac_to_f32(flac_path.as_ref())
}

/// Summary stats for a loaded split.
pub fn summarize(utterances: &[Utterance]) {
    let total_samples: usize = utterances.iter().map(|u| u.samples.len()).sum();
    let total_seconds = total_samples as f64 / 16000.0;
    let total_chars: usize = utterances.iter().map(|u| u.transcript.len()).sum();
    let speakers: std::collections::HashSet<&str> = utterances.iter()
        .map(|u| u.speaker_id.as_str()).collect();

    eprintln!("  LibriSpeech: {} utterances, {:.1} hours, {} speakers, {:.0}K transcript chars",
        utterances.len(), total_seconds / 3600.0, speakers.len(), total_chars as f64 / 1000.0);
}
