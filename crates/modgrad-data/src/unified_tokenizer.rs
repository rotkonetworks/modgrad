//! Unified tokenizer — codec-output token IDs onto a target LLM's vocab.
//!
//! Codec tokenizers (byte-stream, VQ-VAE image, audio codec, timestamp ticks,
//! action tokens) each produce IDs in their own *local* range — e.g. an image
//! VQ code in `0..4096`. The brain consumes a *unified* sequence of i64 token
//! IDs that index a single LLM embedding table (Qwen2.5: 151936 slots).
//!
//! `UnifiedTokenizer` is the deterministic remap: codec-local ID + modality
//! → absolute slot in the target vocab. It is **not** a tokenizer in the BPE
//! sense; bytes pass straight through (Qwen's native byte BPE already covers
//! `0..256`). It is purely an offset-accounting layer — but a one-off-by-one
//! here corrupts every downstream embedding lookup silently, so the offsets
//! are named constants and every range is range-checked.
//!
//! ## Target-vocab assumption
//!
//! The default constructor `for_qwen2_5()` targets Qwen2.5's vocab of 151936
//! tokens. Qwen reserves the high range roughly `151643..151936` for special
//! / control tokens (`<|im_start|>`, `<|im_end|>`, etc.). Codec ranges are
//! parked **below** that region so they cannot collide.
//!
//! Retargeting to a different LLM (Llama, Mistral, …) means revisiting these
//! offsets — those models have different vocab sizes and different special-
//! token regions. `UnifiedTokenizer::new(target_vocab)` lets callers pick a
//! different vocab size, but the *offsets* are still tuned for the
//! Qwen-style "specials live at the top" layout.

// ── Byte range ─────────────────────────────────────────────────
// Qwen2.5's BPE has explicit byte tokens at IDs 0..256. We pass bytes
// through unchanged — `encode_byte(b) == b as i64`. No offset adapter
// needed for this modality.
/// First slot covered by the byte modality (inclusive).
pub const BYTE_START: i64 = 0;
/// One past the last byte slot.
pub const BYTE_END: i64 = 256;

// ── Codec-reserved region ──────────────────────────────────────
// All codec-specific (non-byte) modalities are parked starting at this
// offset. 140000 is far enough above the dense end of Qwen's natural
// vocab to leave room without colliding, and far enough below 151643
// to leave Qwen's specials untouched.
/// Base offset for all reserved codec ranges. All slots below this
/// belong to the LLM's native vocab (or pass through, in the case of
/// bytes 0..256). All slots above 151643 belong to the LLM's specials
/// — the codec ranges are sandwiched between.
pub const CODEC_BASE: i64 = 140_000;

// ── Delimiters ─────────────────────────────────────────────────
// 8 slots: 6 used (img/aud/vid open+close), 2 reserved for future
// modalities. Absolute range: 140000..140008.
/// First delimiter slot.
pub const DELIM_START: i64 = CODEC_BASE;
/// Number of delimiter slots reserved (8; only 6 currently named).
pub const DELIM_COUNT: i64 = 8;
/// One past the last delimiter slot. = 140008.
pub const DELIM_END: i64 = DELIM_START + DELIM_COUNT;

// ── Image VQ codes ─────────────────────────────────────────────
// 4096 codes (12-bit codebook). Absolute range: 140008..144104.
/// First image-VQ slot.
pub const IMAGE_VQ_START: i64 = DELIM_END;
/// Image-VQ codebook size.
pub const IMAGE_VQ_COUNT: i64 = 4096;
/// One past the last image-VQ slot. = 144104.
pub const IMAGE_VQ_END: i64 = IMAGE_VQ_START + IMAGE_VQ_COUNT;

// ── Audio VQ codes ─────────────────────────────────────────────
// 4096 codes. Absolute range: 144104..148200.
/// First audio-VQ slot.
pub const AUDIO_VQ_START: i64 = IMAGE_VQ_END;
/// Audio-VQ codebook size.
pub const AUDIO_VQ_COUNT: i64 = 4096;
/// One past the last audio-VQ slot. = 148200.
pub const AUDIO_VQ_END: i64 = AUDIO_VQ_START + AUDIO_VQ_COUNT;

// ── Timestamps ─────────────────────────────────────────────────
// 400 ticks at 0.5s resolution = 200 seconds of relative timing.
// Absolute range: 148200..148600.
/// First timestamp slot.
pub const TIMESTAMP_START: i64 = AUDIO_VQ_END;
/// Number of timestamp ticks.
pub const TIMESTAMP_COUNT: i64 = 400;
/// One past the last timestamp slot. = 148600.
pub const TIMESTAMP_END: i64 = TIMESTAMP_START + TIMESTAMP_COUNT;

// ── Actions ────────────────────────────────────────────────────
// 278 action tokens (mouse, keyboard, coordinates). Absolute range:
// 148600..148878. End = 148878 < 151643 (Qwen specials) — safe.
/// First action slot.
pub const ACTION_START: i64 = TIMESTAMP_END;
/// Number of action tokens.
pub const ACTION_COUNT: i64 = 278;
/// One past the last action slot. = 148878.
pub const ACTION_END: i64 = ACTION_START + ACTION_COUNT;

// ── Qwen specials ──────────────────────────────────────────────
// Qwen2.5 uses 151643..151936 for special / reserved control tokens.
// All codec ranges above must stay strictly below this.
/// First slot reserved by Qwen2.5 for special tokens. Codec ranges
/// must end at or before this value.
pub const QWEN_SPECIAL_START: i64 = 151_643;
/// Qwen2.5's full vocab size.
pub const QWEN_VOCAB: usize = 151_936;

// Compile-time invariant: action range ends below Qwen specials.
const _: () = assert!(ACTION_END <= QWEN_SPECIAL_START);

/// Canonical `Modality` lives in `modgrad-traits`; re-exported here for
/// ergonomic access. Returned by [`UnifiedTokenizer::decode_modality`].
pub use modgrad_traits::cerebellum::Modality;

/// The six well-known multimodal delimiters. Stable order — index in
/// this enum equals offset from [`DELIM_START`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    /// `<img>` — start of image-VQ run.
    ImgOpen,
    /// `</img>` — end of image-VQ run.
    ImgClose,
    /// `<aud>` — start of audio-VQ run.
    AudOpen,
    /// `</aud>` — end of audio-VQ run.
    AudClose,
    /// `<vid>` — start of video run (interleaved image+audio).
    VidOpen,
    /// `</vid>` — end of video run.
    VidClose,
}

impl Delimiter {
    /// Offset from [`DELIM_START`] for this delimiter (0..6).
    #[inline]
    pub fn index(self) -> i64 {
        match self {
            Delimiter::ImgOpen => 0,
            Delimiter::ImgClose => 1,
            Delimiter::AudOpen => 2,
            Delimiter::AudClose => 3,
            Delimiter::VidOpen => 4,
            Delimiter::VidClose => 5,
        }
    }
}

/// Stable, deterministic remap from codec-output token IDs onto a target
/// LLM vocab (default: Qwen2.5, 151936).
///
/// The `encode_*` methods take **codec-local** ids (e.g. an image VQ index
/// in `0..4096`) and return the absolute slot in the target vocab.
/// `decode_modality` is the inverse — given an absolute slot, what
/// modality does it belong to? It's a cheap range match, not a hash.
///
/// ## Invariants
/// - All codec ranges fit inside `[CODEC_BASE, QWEN_SPECIAL_START)`.
/// - Ranges are pairwise disjoint (see [`UnifiedTokenizer::ranges`]).
/// - Bytes pass through: `encode_byte(b) == b as i64`.
///
/// ## Targeting other LLMs
/// `for_qwen2_5()` is the only "tuned" constructor. The offsets
/// (`CODEC_BASE`, etc.) were chosen for Qwen2.5's specific specials-at-top
/// vocab layout. `new(target_vocab)` lets you reuse the same offsets
/// against a different vocab size, but you should re-audit whether the
/// target LLM's special / reserved tokens collide with `[140000, 148878)`
/// before trusting it.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedTokenizer {
    /// Target LLM vocab size. Stored for validation in `decode_modality`.
    pub target_vocab: usize,
}

impl UnifiedTokenizer {
    /// Tokenizer targeting Qwen2.5's 151936-slot vocab.
    pub fn for_qwen2_5() -> Self {
        Self { target_vocab: QWEN_VOCAB }
    }

    /// Tokenizer targeting a custom vocab size. The codec offsets are
    /// unchanged — caller is responsible for checking that the target
    /// LLM's special-token region doesn't collide with
    /// `[CODEC_BASE, ACTION_END)`.
    pub fn new(target_vocab: usize) -> Self {
        assert!(
            (ACTION_END as usize) <= target_vocab,
            "target_vocab {} is too small to hold codec ranges (need >= {})",
            target_vocab, ACTION_END,
        );
        Self { target_vocab }
    }

    /// Encode a raw byte. Bytes pass through Qwen's native byte BPE
    /// unchanged: `encode_byte(b) == b as i64`.
    #[inline]
    pub fn encode_byte(&self, b: u8) -> i64 {
        b as i64
    }

    /// Encode an image-VQ codebook index. Panics if `code >= 4096`.
    #[inline]
    pub fn encode_image_vq(&self, code: u16) -> i64 {
        assert!(
            (code as i64) < IMAGE_VQ_COUNT,
            "image VQ code {} >= codebook size {}", code, IMAGE_VQ_COUNT,
        );
        IMAGE_VQ_START + code as i64
    }

    /// Encode an audio-VQ codebook index. Panics if `code >= 4096`.
    #[inline]
    pub fn encode_audio_vq(&self, code: u16) -> i64 {
        assert!(
            (code as i64) < AUDIO_VQ_COUNT,
            "audio VQ code {} >= codebook size {}", code, AUDIO_VQ_COUNT,
        );
        AUDIO_VQ_START + code as i64
    }

    /// Encode a timestamp tick (0.5s resolution). Panics if
    /// `tick >= 400` (= 200s of relative time).
    #[inline]
    pub fn encode_timestamp_tick(&self, tick: u16) -> i64 {
        assert!(
            (tick as i64) < TIMESTAMP_COUNT,
            "timestamp tick {} >= max {}", tick, TIMESTAMP_COUNT,
        );
        TIMESTAMP_START + tick as i64
    }

    /// Encode an action ID (mouse / keyboard / coordinate). Panics if
    /// `action_id >= 278`.
    #[inline]
    pub fn encode_action(&self, action_id: u16) -> i64 {
        assert!(
            (action_id as i64) < ACTION_COUNT,
            "action id {} >= max {}", action_id, ACTION_COUNT,
        );
        ACTION_START + action_id as i64
    }

    /// Encode a delimiter to its reserved slot.
    #[inline]
    pub fn encode_delimiter(&self, kind: Delimiter) -> i64 {
        DELIM_START + kind.index()
    }

    /// Inverse of `encode_*`: given an absolute target-vocab slot,
    /// classify which modality (if any) it belongs to. Range-match,
    /// constant-time.
    pub fn decode_modality(&self, tok: i64) -> Modality {
        if tok < 0 || (tok as usize) >= self.target_vocab {
            return Modality::Other;
        }
        if (BYTE_START..BYTE_END).contains(&tok) {
            Modality::Byte
        } else if (DELIM_START..DELIM_END).contains(&tok) {
            Modality::Delimiter
        } else if (IMAGE_VQ_START..IMAGE_VQ_END).contains(&tok) {
            Modality::ImageVq
        } else if (AUDIO_VQ_START..AUDIO_VQ_END).contains(&tok) {
            Modality::AudioVq
        } else if (TIMESTAMP_START..TIMESTAMP_END).contains(&tok) {
            Modality::Timestamp
        } else if (ACTION_START..ACTION_END).contains(&tok) {
            Modality::Action
        } else {
            Modality::Other
        }
    }

    /// All codec-reserved ranges as `(name, start, end)` tuples.
    /// Used by tests; also handy for diagnostics.
    pub fn ranges() -> &'static [(&'static str, i64, i64)] {
        &[
            ("byte",      BYTE_START,      BYTE_END),
            ("delimiter", DELIM_START,     DELIM_END),
            ("image_vq",  IMAGE_VQ_START,  IMAGE_VQ_END),
            ("audio_vq",  AUDIO_VQ_START,  AUDIO_VQ_END),
            ("timestamp", TIMESTAMP_START, TIMESTAMP_END),
            ("action",    ACTION_START,    ACTION_END),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_dont_overlap() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let ranges = UnifiedTokenizer::ranges();

        // Pairwise disjointness.
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                let (na, sa, ea) = ranges[i];
                let (nb, sb, eb) = ranges[j];
                assert!(sa < ea, "{} has empty/inverted range {}..{}", na, sa, ea);
                assert!(sb < eb, "{} has empty/inverted range {}..{}", nb, sb, eb);
                let overlap = sa < eb && sb < ea;
                assert!(
                    !overlap,
                    "ranges {} ({}..{}) and {} ({}..{}) overlap",
                    na, sa, ea, nb, sb, eb,
                );
            }
        }

        // Every codec range fits inside the target vocab.
        for &(name, start, end) in ranges {
            assert!(
                start >= 0 && (end as usize) <= t.target_vocab,
                "{} ({}..{}) escapes target_vocab {}",
                name, start, end, t.target_vocab,
            );
        }

        // All non-byte codec ranges live below Qwen's special-token region.
        for &(name, _start, end) in ranges {
            if name == "byte" { continue; }
            assert!(
                end <= QWEN_SPECIAL_START,
                "{} ends at {} which collides with Qwen specials at {}",
                name, end, QWEN_SPECIAL_START,
            );
        }
    }

    #[test]
    fn byte_passthrough() {
        let t = UnifiedTokenizer::for_qwen2_5();
        for b in [0u8, 1, 65, 127, 200, 255] {
            assert_eq!(t.encode_byte(b), b as i64);
            assert_eq!(t.decode_modality(b as i64), Modality::Byte);
        }
    }

    #[test]
    fn round_trip_image_vq() {
        let t = UnifiedTokenizer::for_qwen2_5();
        for code in [0u16, 100, 1000, 4095] {
            let tok = t.encode_image_vq(code);
            assert!((IMAGE_VQ_START..IMAGE_VQ_END).contains(&tok));
            assert_eq!(t.decode_modality(tok), Modality::ImageVq);
        }
    }

    #[test]
    fn round_trip_audio_vq() {
        let t = UnifiedTokenizer::for_qwen2_5();
        for code in [0u16, 1, 2048, 4095] {
            let tok = t.encode_audio_vq(code);
            assert!((AUDIO_VQ_START..AUDIO_VQ_END).contains(&tok));
            assert_eq!(t.decode_modality(tok), Modality::AudioVq);
        }
    }

    #[test]
    fn round_trip_timestamp() {
        let t = UnifiedTokenizer::for_qwen2_5();
        for tick in [0u16, 1, 199, 399] {
            let tok = t.encode_timestamp_tick(tick);
            assert!((TIMESTAMP_START..TIMESTAMP_END).contains(&tok));
            assert_eq!(t.decode_modality(tok), Modality::Timestamp);
        }
    }

    #[test]
    fn round_trip_action() {
        let t = UnifiedTokenizer::for_qwen2_5();
        for action in [0u16, 1, 100, 277] {
            let tok = t.encode_action(action);
            assert!((ACTION_START..ACTION_END).contains(&tok));
            assert_eq!(t.decode_modality(tok), Modality::Action);
        }
    }

    #[test]
    fn round_trip_delimiter() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let kinds = [
            Delimiter::ImgOpen, Delimiter::ImgClose,
            Delimiter::AudOpen, Delimiter::AudClose,
            Delimiter::VidOpen, Delimiter::VidClose,
        ];
        let mut seen = std::collections::HashSet::new();
        for kind in kinds {
            let tok = t.encode_delimiter(kind);
            assert!((DELIM_START..DELIM_END).contains(&tok));
            assert_eq!(t.decode_modality(tok), Modality::Delimiter);
            // Each delimiter maps to a distinct slot.
            assert!(seen.insert(tok), "delimiter {:?} collides at slot {}", kind, tok);
        }
    }

    #[test]
    fn other_modality_for_qwen_normal_and_special_ranges() {
        let t = UnifiedTokenizer::for_qwen2_5();
        // Qwen normal vocab: somewhere in the middle, away from codec.
        assert_eq!(t.decode_modality(50_000), Modality::Other);
        // Qwen specials.
        assert_eq!(t.decode_modality(QWEN_SPECIAL_START), Modality::Other);
        assert_eq!(t.decode_modality(151_935), Modality::Other);
        // Out of range.
        assert_eq!(t.decode_modality(-1), Modality::Other);
        assert_eq!(t.decode_modality(t.target_vocab as i64), Modality::Other);
    }

    #[test]
    #[should_panic]
    fn image_vq_oob_rejected() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let _ = t.encode_image_vq(4096);
    }

    #[test]
    #[should_panic]
    fn audio_vq_oob_rejected() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let _ = t.encode_audio_vq(4096);
    }

    #[test]
    #[should_panic]
    fn action_oob_rejected() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let _ = t.encode_action(278);
    }

    #[test]
    #[should_panic]
    fn timestamp_oob_rejected() {
        let t = UnifiedTokenizer::for_qwen2_5();
        let _ = t.encode_timestamp_tick(400);
    }

    #[test]
    fn offsets_match_documented_layout() {
        // Lock in the exact layout — if anyone changes an offset,
        // this test makes them update the docs alongside.
        assert_eq!(CODEC_BASE,       140_000);
        assert_eq!(DELIM_START,      140_000);
        assert_eq!(DELIM_END,        140_008);
        assert_eq!(IMAGE_VQ_START,   140_008);
        assert_eq!(IMAGE_VQ_END,     144_104);
        assert_eq!(AUDIO_VQ_START,   144_104);
        assert_eq!(AUDIO_VQ_END,     148_200);
        assert_eq!(TIMESTAMP_START,  148_200);
        assert_eq!(TIMESTAMP_END,    148_600);
        assert_eq!(ACTION_START,     148_600);
        assert_eq!(ACTION_END,       148_878);
        assert!(ACTION_END < QWEN_SPECIAL_START);
    }
}
