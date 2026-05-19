//! Q4_K / Q6_K interleaved FFN (`interleaved_kquant.bin`) plus the
//! feature-major down sidecar (`down_features_q4k.bin`).
//!
//! Both files come with a JSON manifest declaring per-slice format
//! tags; `read_q4k_manifest` validates every tag against
//! `quant::registry` so a renamed format fails loudly at load time
//! instead of silently producing zero-byte slices.
//!
//! `down_features_q4k.bin` is the W2-of-perf-round-4 sidecar — feature-
//! major Q4_K down vectors so per-feature decode skips the
//! `kquant_ffn_layer` whole-layer dequant cache. The legacy interleaved
//! path stays available as the fallback when the sidecar is absent.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::{
    resolve_down_features_kquant, resolve_interleaved_kquant, INTERLEAVED_KQUANT_BIN,
    LEGACY_INTERLEAVED_Q4K_BIN,
};
use crate::format::weights::Q4kManifestEntry;
use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::VindexStorage;
use crate::mmap_util::mmap_demand_paged;

#[cfg(unix)]
use super::FFN_DOWN;
use super::{DownFeaturesQ4kEntry, FFN_COMPONENTS_PER_LAYER};

/// Read + typed-deserialise a Q4_K manifest JSON file. Validates each
/// entry's format tag against `quant::registry`. `display_name` is the
/// filename used in error messages so a parse failure reports which
/// manifest broke. Centralised so both `load_interleaved_kquant` and
/// `load_down_features_q4k` go through the same parse + validation
/// path.
fn read_q4k_manifest(
    path: &std::path::Path,
    display_name: &str,
) -> Result<Vec<Q4kManifestEntry>, VindexError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| VindexError::Parse(format!("{display_name}: {e}")))?;
    let entries: Vec<Q4kManifestEntry> = serde_json::from_str(&text)
        .map_err(|e| VindexError::Parse(format!("{display_name}: {e}")))?;
    for e in &entries {
        if crate::quant::registry::lookup(e.format_tag()).is_none() {
            return Err(VindexError::Parse(format!(
                "{display_name}: unknown format tag {:?} — quant::registry has no entry",
                e.format_tag(),
            )));
        }
    }
    Ok(entries)
}

impl VectorIndex {
    /// Load Q4_K/Q6_K interleaved FFN data (Ollama-compatible, matches attn format).
    ///
    /// Also reads the optional `interleaved_kquant_manifest.json` sidecar emitted
    /// by the streaming Q4 writer. When the manifest is present callers get
    /// per-matrix layout (offsets, lengths, formats) via
    /// [`VectorIndex::interleaved_kquant_layer_data`]. When it's absent — older
    /// vindexes from `build_q4k_weights.rs` — callers fall back to the legacy
    /// uniform-stride path.
    pub fn load_interleaved_kquant(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let resolved = resolve_interleaved_kquant(dir);
        let path = resolved.bin;
        if !path.exists() {
            return Err(VindexError::Parse(format!(
                "interleaved k-quant FFN not found (looked for {} and legacy {})",
                INTERLEAVED_KQUANT_BIN, LEGACY_INTERLEAVED_Q4K_BIN
            )));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: the kquant forward walk reads only the activated
        // features' byte ranges per layer, not the entire 13 GB file.
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });

        let manifest_path = resolved
            .manifest
            .expect("interleaved kquant resolver always pairs a manifest");
        let display_name = manifest_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("interleaved kquant manifest");
        let manifest = if manifest_path.exists() {
            // Typed deserialise — `Q4kManifestEntry` matches the writer's
            // shape, so a renamed field on either side fails loudly here
            // instead of silently producing zero-byte slices.
            let raw = read_q4k_manifest(&manifest_path, display_name)?;
            Some(
                raw.into_iter()
                    .map(|e| {
                        (
                            e.offset as usize,
                            e.length as usize,
                            e.format_tag().to_string(),
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };
        Arc::make_mut(&mut self.storage).set_interleaved_kquant(mmap, manifest);
        Ok(())
    }

    pub fn has_interleaved_kquant(&self) -> bool {
        self.storage.has_interleaved_kquant()
    }

    /// Load feature-major k-quant down weights if present (W2). Silent
    /// no-op when the file is absent — older vindexes still work via the
    /// `kquant_ffn_layer` cache fallback. Idempotent.
    pub fn load_down_features_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let resolved = resolve_down_features_kquant(dir);
        let path = resolved.bin;
        if !path.exists() {
            return Ok(());
        }
        let manifest_path = resolved
            .manifest
            .expect("down-features kquant resolver always pairs a manifest");
        if !manifest_path.exists() {
            return Err(VindexError::Parse(format!(
                "{} present but manifest {} missing",
                path.display(),
                manifest_path.display()
            )));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only the activated features' byte ranges per
        // layer get read in. Same access pattern as `interleaved_kquant.bin`.
        let mmap = Arc::new(unsafe { mmap_demand_paged(&file)? });

        let manifest_display = manifest_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("down-features kquant manifest");
        let raw = read_q4k_manifest(&manifest_path, manifest_display)?;
        let entries: Vec<DownFeaturesQ4kEntry> = raw
            .into_iter()
            .map(|e| {
                let padded_width = e.padded_width().ok_or_else(|| {
                    VindexError::Parse(format!(
                        "{manifest_display} entry has no shape[1] (padded_width)"
                    ))
                })?;
                Ok(DownFeaturesQ4kEntry {
                    offset: e.offset as usize,
                    length: e.length as usize,
                    format: e.format_tag().to_string(),
                    padded_width,
                })
            })
            .collect::<Result<Vec<_>, VindexError>>()?;
        Arc::make_mut(&mut self.storage).set_down_features_q4k(mmap, entries);
        Ok(())
    }

    /// Whether feature-major Q4_K-encoded down vectors are loaded.
    pub fn has_down_features_kquant(&self) -> bool {
        self.storage.has_down_features_kquant()
    }

    /// Per-layer slice of `down_features_q4k.bin` plus the format tag
    /// and the padded row width. Returns `None` when the file isn't
    /// loaded or the layer is out of range. The bytes are feature-major
    /// `[intermediate, padded_width]`, Q4_K/Q6_K-encoded — feature
    /// `feat` lives at byte offset
    /// `feat * bytes_per_row(padded_width)` inside the slice.
    /// Per-layer slice of `down_features_q4k.bin` plus the format tag
    /// and the padded row width. Forwarded through
    /// [`VectorIndex::storage`] (step 4 of the `VindexStorage`
    /// migration).
    pub fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(&[u8], &str, usize)> {
        let (view, fmt, padded_width) = self.storage.down_features_q4k_layer_data(layer)?;
        Some((view.as_slice(), fmt, padded_width))
    }

    /// Per-layer Q4_K/Q6_K FFN slices — [gate, up, down] with formats.
    ///
    /// Returns `None` when the FFN manifest wasn't present at load time
    /// (caller should fall back to uniform-stride). Returns `Some` iff
    /// the manifest has `FFN_COMPONENTS_PER_LAYER` entries for `layer`;
    /// downstream kernels dispatch on the format string (`"Q4_K"` or
    /// `"Q6_K"`).
    pub fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
        // Forwarded through `self.storage` (step 4 of the
        // `VindexStorage` migration). Public signature unchanged so
        // existing callers don't move.
        let arr = self.storage.interleaved_kquant_layer_data(layer)?;
        let mut out: [(&[u8], &str); FFN_COMPONENTS_PER_LAYER] =
            [(&[], ""); FFN_COMPONENTS_PER_LAYER];
        for i in 0..FFN_COMPONENTS_PER_LAYER {
            let (view, fmt) = arr[i];
            out[i] = (view.as_slice(), fmt);
        }
        Some(out)
    }

    /// Prefetch next layer's Q4_K/Q6_K FFN data into the page cache via
    /// MADV_WILLNEED. Counterpart of [`Self::prefetch_interleaved_q4_layer`].
    /// Issues one madvise spanning the layer's gate+up+down matrices.
    ///
    /// When the FFN manifest is loaded (the streaming-writer path), the
    /// span is computed from the layer's three manifest entries — handles
    /// mixed Q4_K/Q6_K layouts where down may be Q6_K (210 B/256) while
    /// gate/up are Q4_K (144 B/256). Without a manifest, falls back to
    /// the legacy uniform Q4_K stride (144 B/256 across all three
    /// matrices) — matches the build_q4k_weights writer.
    #[cfg_attr(not(unix), allow(unused_variables))]
    pub fn prefetch_interleaved_kquant_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(bytes) = self.storage.interleaved_kquant_whole_buffer_view() {
            let mmap: &[u8] = bytes.as_ref();
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return;
            }
            // The trait gives us the layer view directly when a
            // manifest is loaded — that's the correct (start, end)
            // span. Without the per-layer view we fall back to the
            // legacy uniform-Q4_K stride.
            let (start, len) = if let Some(arr) = self.storage.interleaved_kquant_layer_data(layer)
            {
                // Span = first component's start to last component's end.
                let first_start = {
                    let (view, _) = arr[0];
                    view.offset
                };
                let last_end = {
                    let (view, _) = arr[FFN_DOWN];
                    view.offset + view.length
                };
                if first_start >= mmap.len() || last_end <= first_start {
                    return;
                }
                (first_start, last_end - first_start)
            } else {
                // Uniform-stride fallback: matches build_q4k_weights's
                // Q4_K-only writer. Q4_K is 144 bytes per 256 elements.
                use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES};
                let blocks_per_matrix = intermediate * self.hidden_size / K_QUANT_BLOCK_ELEMS;
                let bytes_per_matrix = blocks_per_matrix * Q4_K_BLOCK_BYTES;
                let bytes_per_layer = bytes_per_matrix * 3;
                let s = layer * bytes_per_layer;
                let e = (s + bytes_per_layer).min(mmap.len());
                if s >= mmap.len() || e <= s {
                    return;
                }
                (s, e - s)
            };
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, len, libc::MADV_WILLNEED);
            }
        }
    }
}
