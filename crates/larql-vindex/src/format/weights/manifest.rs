//! Shared manifest entry shape used by `write_kquant` to emit
//! `attn_weights_q4k_manifest.json`, `interleaved_kquant_manifest.json`,
//! and `down_features_q4k_manifest.json`. Pulled out so the loaders in
//! `index/storage/ffn_store.rs` can deserialise into a typed struct
//! instead of poking `serde_json::Value` with string keys — silently
//! `unwrap_or(0)`'ing missing fields was a real footgun (a renamed
//! field would silently produce zero-byte slices).
//!
//! One entry describes one tensor's slice within its `.bin` file:
//! - `offset` / `length` — byte range within the file
//! - `format` — quant tag, must round-trip via `quant::registry::lookup`
//! - `shape` — `[rows, padded_cols]` after `pad_rows_to_block`
//! - `key` — original tensor name (for human inspection / round-trip)
//!
//! The fields are deliberately laid out so the JSON shape matches what
//! the previous (string-keyed) loaders expected — switching loaders to
//! typed deserialisation is a no-op on existing on-disk manifests.

use serde::{Deserialize, Serialize};

use super::write_kquant::QuantBlockFormat;

/// One manifest entry describing one Q4_K/Q6_K-encoded tensor slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q4kManifestEntry {
    pub key: String,
    pub shape: Vec<usize>,
    pub format: QuantBlockFormat,
    pub offset: u64,
    pub length: u64,
}

impl Q4kManifestEntry {
    /// Padded row stride in elements (second dim of `shape`). Returns
    /// `None` when the manifest entry has fewer than 2 dimensions —
    /// caller decides whether to error or fall back to `hidden_size`.
    pub fn padded_width(&self) -> Option<usize> {
        self.shape.get(1).copied()
    }

    /// Format tag as the on-disk string (`"Q4_K"` / `"Q6_K"` / …).
    /// `quant::registry::lookup` consumes this directly. Lifetime is
    /// borrowed because the `Other(String)` variant carries a runtime
    /// tag that future formats use to round-trip through the manifest
    /// without a typed enum variant.
    pub fn format_tag(&self) -> &str {
        self.format.tag()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON wire shape stays compatible with the previous string-keyed
    /// loader — `offset`/`length`/`format`/`shape`/`key` field names
    /// are load-bearing for already-extracted vindexes on disk.
    #[test]
    fn round_trip_matches_writer_wire_shape() {
        let entry = Q4kManifestEntry {
            key: "model.layers.0.mlp.down_proj.weight".into(),
            shape: vec![4096, 2560],
            format: QuantBlockFormat::Q6K,
            offset: 1024,
            length: 53760,
        };
        let json = serde_json::to_string(&entry).unwrap();
        // Spot-check the field names — a serde rename would silently
        // break older vindexes that ship the legacy spelling.
        assert!(json.contains("\"key\""));
        assert!(json.contains("\"shape\""));
        assert!(json.contains("\"format\""));
        assert!(json.contains("\"offset\""));
        assert!(json.contains("\"length\""));
        let back: Q4kManifestEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key, entry.key);
        assert_eq!(back.shape, entry.shape);
        assert_eq!(back.offset, entry.offset);
        assert_eq!(back.length, entry.length);
        assert_eq!(back.format_tag(), "Q6_K");
    }

    /// Format tag values are the on-disk strings the registry expects.
    /// Adding a new K-quant format must update `format_tag` so
    /// `quant::registry::lookup` doesn't return `None` and trip the
    /// load-time validation.
    #[test]
    fn format_tag_matches_on_disk_strings() {
        let q4 = Q4kManifestEntry {
            key: "x".into(),
            shape: vec![1, 256],
            format: QuantBlockFormat::Q4K,
            offset: 0,
            length: 0,
        };
        let q6 = Q4kManifestEntry {
            key: "x".into(),
            shape: vec![1, 256],
            format: QuantBlockFormat::Q6K,
            offset: 0,
            length: 0,
        };
        assert_eq!(q4.format_tag(), "Q4_K");
        assert_eq!(q6.format_tag(), "Q6_K");
    }

    /// `padded_width` returns the row stride (second shape dim) for
    /// well-formed entries and `None` for malformed ones (e.g. a 1-D
    /// shape that older code might emit). The W2 down loader uses
    /// this and errors loudly when it returns `None`.
    #[test]
    fn padded_width_extracts_second_dim() {
        let two_d = Q4kManifestEntry {
            key: "x".into(),
            shape: vec![10240, 2560],
            format: QuantBlockFormat::Q4K,
            offset: 0,
            length: 0,
        };
        assert_eq!(two_d.padded_width(), Some(2560));

        let one_d = Q4kManifestEntry {
            key: "x".into(),
            shape: vec![2560],
            format: QuantBlockFormat::Q4K,
            offset: 0,
            length: 0,
        };
        assert_eq!(one_d.padded_width(), None);

        let empty = Q4kManifestEntry {
            key: "x".into(),
            shape: vec![],
            format: QuantBlockFormat::Q4K,
            offset: 0,
            length: 0,
        };
        assert_eq!(empty.padded_width(), None);
    }

    /// A malformed manifest (missing `format` field) is rejected at
    /// parse time — no silent fallback to a default tag. This is the
    /// failure mode the typed deserialiser was added to catch.
    #[test]
    fn missing_format_field_fails_parse() {
        let json = r#"[{"key":"x","shape":[10240,2560],"offset":0,"length":1}]"#;
        let parsed: Result<Vec<Q4kManifestEntry>, _> = serde_json::from_str(json);
        assert!(
            parsed.is_err(),
            "missing `format` must error, not silently default"
        );
    }
}
