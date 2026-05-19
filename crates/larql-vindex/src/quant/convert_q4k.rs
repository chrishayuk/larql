//! `vindex_to_q4k` — quantise an existing f32/f16 vindex into a
//! Q4_K/Q6_K vindex. Library entry for the `larql convert quantize q4k`
//! CLI subcommand.
//!
//! Q4K uses the GGML "Q4_K_M" mix that Ollama ships with: attention
//! Q/K/O and FFN gate/up at Q4_K, attention V and FFN down at Q6_K.
//! `down_q4k = true` switches FFN down to Q4_K uniformly (saves ~30 MB
//! per layer on 31B, ~1.8 GB total; noise on the scatter-sum averages
//! across the intermediate dimension — empirically close).
//!
//! Shape mirrors `vindex_to_fp4`: take an existing vindex directory,
//! write a new Q4K vindex atomically (`<dst>.tmp/` → `<dst>/`),
//! hard-link auxiliary files, return a `Q4kConvertReport` for CLI
//! display.
//!
//! Precondition: the source vindex must have full model weights
//! (`extract_level: inference` or `all`). The Q4K writer reads every
//! FFN tensor from the source — a browse-only vindex doesn't have
//! them. Callers without the full weights should extract with
//! `--level inference` first.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::config::types::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::weights::{
    load_model_weights, write_model_weights_kquant_with_opts, KquantWriteOptions,
};
use crate::index::storage::ffn_store::{FFN_COMPONENTS_PER_LAYER, FFN_DOWN};
use crate::IndexLoadCallbacks;

#[derive(Debug, Clone, Default)]
pub struct Q4kConvertConfig {
    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Default false
    /// preserves the Ollama-compatible Q4_K_M mix (Q4_K gate/up, Q6_K
    /// down). See `write_model_weights_kquant_with_opts` for the
    /// tradeoff.
    pub down_q4k: bool,
    /// Emit `down_features_q4k.bin` (W2 feature-major down) so per-feature
    /// row decode can skip the `kquant_ffn_layer` cache. Disk grows by
    /// roughly one extra down-leg per layer; load-time RSS drops because
    /// the cache stays empty. See `KquantWriteOptions::feature_major_down`.
    pub feature_major_down: bool,
    /// Overwrite `dst` if it already exists.
    pub force: bool,
}

#[derive(Debug, Clone)]
pub struct Q4kConvertReport {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub down_q4k: bool,
    pub src_ffn_bytes: u64,
    pub dst_ffn_bytes: u64,
    pub compression: f64,
    pub aux_linked_count: usize,
    pub aux_linked_bytes: u64,
    pub wall_time: Duration,
    pub walk_backend: String,
}

/// Silent callbacks for the Q4K writer. The converter surfaces
/// progress at the CLI level; we don't need the per-tensor pings
/// here.
struct SilentCallbacks;
impl IndexLoadCallbacks for SilentCallbacks {}
impl crate::IndexBuildCallbacks for SilentCallbacks {}

/// Convert an f32/f16 vindex at `src` into a Q4K vindex at `dst`.
/// Atomic: writes into `<dst>.tmp/`, renames to `<dst>/` on success.
pub fn vindex_to_q4k(
    src: &Path,
    dst: &Path,
    config: &Q4kConvertConfig,
) -> Result<Q4kConvertReport, VindexError> {
    let t_total = Instant::now();

    if dst.exists() {
        if !config.force {
            return Err(VindexError::Parse(format!(
                "output dir {} exists (use force=true to overwrite)",
                dst.display()
            )));
        }
        std::fs::remove_dir_all(dst)
            .map_err(|e| VindexError::Parse(format!("remove existing dst: {e}")))?;
    }

    let dst_tmp = dst.with_file_name(format!(
        "{}.tmp",
        dst.file_name().and_then(|s| s.to_str()).unwrap_or("out")
    ));
    if dst_tmp.exists() {
        std::fs::remove_dir_all(&dst_tmp)
            .map_err(|e| VindexError::Parse(format!("clean staging dir: {e}")))?;
    }
    std::fs::create_dir_all(&dst_tmp)
        .map_err(|e| VindexError::Parse(format!("create staging dir: {e}")))?;

    // Parse source config and verify preconditions.
    let src_config: VindexConfig = serde_json::from_str(
        &std::fs::read_to_string(src.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("read src index.json: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse src index.json: {e}")))?;

    if !src_config.has_model_weights {
        return Err(VindexError::Parse(format!(
            "src vindex {} has no model weights (extract_level = {:?}); \
             Q4K quantisation requires `--level inference` or higher on the source extract",
            src.display(),
            src_config.extract_level,
        )));
    }
    if src_config.quant != crate::QuantFormat::None {
        return Err(VindexError::Parse(format!(
            "src vindex is already quantised ({}); Q4K conversion requires \
             a float-weights source",
            src_config.quant,
        )));
    }

    // Load ModelWeights from the source vindex. This reads
    // attn_weights.bin / up_weights.bin / down_weights.bin /
    // embeddings.bin / norms.bin / lm_head.bin (as applicable) into
    // the same ModelWeights shape `write_model_weights_kquant_with_opts`
    // consumes.
    let mut cb = SilentCallbacks;
    let weights = load_model_weights(src, &mut cb as &mut dyn IndexLoadCallbacks)?;

    // Seed the staging dir with the source's index.json. The Q4K writer
    // reads dir/index.json to update it in-place (sets has_model_weights
    // and quant=q4k), so the file must exist before write is called.
    std::fs::copy(src.join(INDEX_JSON), dst_tmp.join(INDEX_JSON))
        .map_err(|e| VindexError::Parse(format!("seed staging index.json: {e}")))?;

    // Write Q4K files into the staging directory. Produces
    // attn_weights_q4k.bin + manifest, interleaved_kquant.bin + manifest,
    // lm_head_q4.bin, norms.bin, weight_manifest.json. Also rewrites
    // index.json with quant=q4k.
    let opts = KquantWriteOptions {
        down_proj: if config.down_q4k {
            crate::DownProjFormat::Q4K
        } else {
            crate::DownProjFormat::Q6K
        },
        feature_major_down: config.feature_major_down,
    };
    let mut build_cb = SilentCallbacks;
    write_model_weights_kquant_with_opts(
        &weights,
        &dst_tmp,
        &mut build_cb as &mut dyn crate::IndexBuildCallbacks,
        opts,
    )?;

    // Hard-link auxiliary files: gate_vectors (KNN still needs the
    // float matrix), embeddings, down_meta, tokenizer, feature_labels.
    // Excludes the f32 weight files that the Q4K path replaces.
    let handled_by_writer: std::collections::HashSet<&str> = [
        INDEX_JSON,
        // Written by write_model_weights_kquant:
        ATTN_WEIGHTS_KQUANT_BIN,
        ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
        INTERLEAVED_KQUANT_BIN,
        INTERLEAVED_KQUANT_MANIFEST_JSON,
        LM_HEAD_KQUANT_BIN,
        NORMS_BIN,
    ]
    .iter()
    .copied()
    .collect();
    let skip_from_src: std::collections::HashSet<&str> = [
        // The f32 weight files that the Q4K path replaces — don't
        // hard-link these, they'd bloat the output and be unused.
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        UP_FEATURES_BIN,
        DOWN_FEATURES_BIN,
        INTERLEAVED_BIN,
        LM_HEAD_BIN,
        NORMS_BIN,
        WEIGHT_MANIFEST_JSON,
        INDEX_JSON,
    ]
    .iter()
    .copied()
    .collect();

    let mut aux_linked = 0usize;
    let mut aux_bytes = 0u64;
    for entry in
        std::fs::read_dir(src).map_err(|e| VindexError::Parse(format!("read src dir: {e}")))?
    {
        let entry = entry.map_err(|e| VindexError::Parse(format!("{e}")))?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if skip_from_src.contains(fname_str.as_ref())
            || handled_by_writer.contains(fname_str.as_ref())
        {
            continue;
        }
        let meta = entry
            .metadata()
            .map_err(|e| VindexError::Parse(format!("{e}")))?;
        if !meta.is_file() {
            continue;
        }
        let dst_path = dst_tmp.join(&fname);
        link_or_copy(&entry.path(), &dst_path)?;
        aux_linked += 1;
        aux_bytes += meta.len();
    }

    // The Q4K writer rewrote index.json (quant=q4k, has_model_weights=true).
    // Clear stale checksums — the source's checksums no longer apply to the
    // quantised files. `larql verify` can recompute on demand.
    let written_text = std::fs::read_to_string(dst_tmp.join(INDEX_JSON))
        .map_err(|e| VindexError::Parse(format!("re-read index.json: {e}")))?;
    let mut written_cfg: VindexConfig = serde_json::from_str(&written_text)
        .map_err(|e| VindexError::Parse(format!("parse written index.json: {e}")))?;
    written_cfg.checksums = None;
    std::fs::write(
        dst_tmp.join(INDEX_JSON),
        serde_json::to_string_pretty(&written_cfg)
            .map_err(|e| VindexError::Parse(format!("serialise config: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("write index.json: {e}")))?;

    // Atomic promote.
    std::fs::rename(&dst_tmp, dst).map_err(|e| {
        VindexError::Parse(format!(
            "atomic rename {} → {}: {e}",
            dst_tmp.display(),
            dst.display()
        ))
    })?;

    // Size reporting. FFN src = up_weights.bin + down_weights.bin
    // (already dense f32). FFN dst = interleaved_kquant.bin.
    let src_ffn_bytes = size_of(&src.join(UP_WEIGHTS_BIN)).unwrap_or(0)
        + size_of(&src.join(DOWN_WEIGHTS_BIN)).unwrap_or(0)
        + size_of(&src.join(GATE_VECTORS_BIN)).unwrap_or(0);
    let dst_ffn_bytes = size_of(&dst.join(INTERLEAVED_KQUANT_BIN)).unwrap_or(0)
        + size_of(&dst.join(GATE_VECTORS_BIN)).unwrap_or(0);
    let compression = if dst_ffn_bytes == 0 {
        1.0
    } else {
        src_ffn_bytes as f64 / dst_ffn_bytes as f64
    };

    let walk_backend =
        describe_out_backend(dst).unwrap_or_else(|e| format!("<describe failed: {e:?}>"));

    Ok(Q4kConvertReport {
        src: src.to_path_buf(),
        dst: dst.to_path_buf(),
        down_q4k: config.down_q4k,
        src_ffn_bytes,
        dst_ffn_bytes,
        compression,
        aux_linked_count: aux_linked,
        aux_linked_bytes: aux_bytes,
        wall_time: t_total.elapsed(),
        walk_backend,
    })
}

fn size_of(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

fn describe_out_backend(dst: &Path) -> Result<String, VindexError> {
    use crate::{SilentLoadCallbacks, VectorIndex};
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dst, &mut cb)?;
    Ok(index.describe_ffn_backend())
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<(), VindexError> {
    if dst.exists() {
        std::fs::remove_file(dst)
            .map_err(|e| VindexError::Parse(format!("remove existing {}: {e}", dst.display())))?;
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst).map_err(|e| {
                VindexError::Parse(format!(
                    "copy fallback {} → {}: {e}",
                    src.display(),
                    dst.display()
                ))
            })?;
            Ok(())
        }
    }
}

/// Report from [`add_feature_major_down`].
#[derive(Debug, Clone)]
pub struct AddFeatureMajorDownReport {
    pub vindex: PathBuf,
    /// `true` when the file was already present and we left it alone.
    pub skipped: bool,
    pub num_layers: usize,
    /// Bytes written to `down_features_q4k.bin` (0 when skipped).
    pub bytes_written: u64,
    pub wall_time: Duration,
}

/// Retrofit `down_features_q4k.bin` into an existing Q4K vindex
/// without re-quantising the rest of the weights. Reads the down
/// portion of `interleaved_kquant.bin` per layer, transposes to
/// `[intermediate, hidden]`, re-quantises at the same precision the
/// source used, and writes the W2 file + manifest in place.
///
/// Idempotent: if `down_features_q4k.bin` already exists, returns
/// `Ok` with `skipped: true` and doesn't touch the directory.
///
/// Precondition: the vindex must have `interleaved_kquant.bin` +
/// `interleaved_kquant_manifest.json` (i.e. `quant: q4k` in
/// `index.json`). Browse-only / f32-only vindexes don't.
pub fn add_feature_major_down(vindex_dir: &Path) -> Result<AddFeatureMajorDownReport, VindexError> {
    use crate::format::weights::write_kquant::feature_major_down::FeatureMajorDownState;
    use crate::format::weights::Q4kManifestEntry;

    let started = Instant::now();
    let dst = vindex_dir.join(DOWN_FEATURES_KQUANT_BIN);
    let dst_manifest = vindex_dir.join(DOWN_FEATURES_KQUANT_MANIFEST_JSON);

    if dst.exists() && dst_manifest.exists() {
        let config = crate::format::load::load_vindex_config(vindex_dir)?;
        return Ok(AddFeatureMajorDownReport {
            vindex: vindex_dir.to_path_buf(),
            skipped: true,
            num_layers: config.num_layers,
            bytes_written: 0,
            wall_time: started.elapsed(),
        });
    }

    // Source: interleaved_kquant.bin + manifest.
    let interleaved_path = vindex_dir.join(INTERLEAVED_KQUANT_BIN);
    let interleaved_manifest_path = vindex_dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON);
    if !interleaved_path.exists() || !interleaved_manifest_path.exists() {
        return Err(VindexError::Parse(format!(
            "{} expects {} + {} (run extract with --quant q4k first)",
            vindex_dir.display(),
            INTERLEAVED_KQUANT_BIN,
            INTERLEAVED_KQUANT_MANIFEST_JSON,
        )));
    }
    let manifest_text = std::fs::read_to_string(&interleaved_manifest_path)?;
    let entries: Vec<Q4kManifestEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| VindexError::Parse(format!("{INTERLEAVED_KQUANT_MANIFEST_JSON}: {e}")))?;

    let config = crate::format::load::load_vindex_config(vindex_dir)?;
    let num_layers = config.num_layers;
    if entries.len() < num_layers * FFN_COMPONENTS_PER_LAYER {
        return Err(VindexError::Parse(format!(
            "{INTERLEAVED_KQUANT_MANIFEST_JSON} has {} entries, expected {} \
             ({FFN_COMPONENTS_PER_LAYER} per layer for {num_layers} layers)",
            entries.len(),
            num_layers * FFN_COMPONENTS_PER_LAYER,
        )));
    }

    let file = std::fs::File::open(&interleaved_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| VindexError::Parse(format!("mmap {INTERLEAVED_KQUANT_BIN}: {e}")))?;

    let mut state = FeatureMajorDownState::new(&dst, num_layers)?;

    // Down is the third entry per layer ([gate, up, down] in the writer).
    for layer in 0..num_layers {
        let down = &entries[layer * FFN_COMPONENTS_PER_LAYER + FFN_DOWN];
        let format = down.format.clone();
        let info = crate::quant::registry::lookup(down.format_tag()).ok_or_else(|| {
            VindexError::Parse(format!(
                "unknown quant format {:?} in {INTERLEAVED_KQUANT_MANIFEST_JSON} for layer {layer}",
                down.format_tag(),
            ))
        })?;
        let rows = down.shape.first().copied().ok_or_else(|| {
            VindexError::Parse(format!(
                "down shape missing rows in {INTERLEAVED_KQUANT_MANIFEST_JSON} for layer {layer}"
            ))
        })?;
        let cols = down.shape.get(1).copied().ok_or_else(|| {
            VindexError::Parse(format!(
                "down shape missing cols in {INTERLEAVED_KQUANT_MANIFEST_JSON} for layer {layer}"
            ))
        })?;
        // Source disk layout for down is `[hidden=rows, padded_intermediate=cols]`.
        let n_padded = rows * cols;
        let bytes = &mmap[down.offset as usize..(down.offset + down.length) as usize];
        let dequant = (info.dequantize)(bytes, n_padded)
            .map_err(|e| VindexError::Parse(format!("dequant down layer {layer}: {e}")))?;
        // FeatureMajorDownState::append_layer expects the full
        // `[rows × cols]` padded f32 buffer — exactly what the
        // dequantiser produced.
        state.append_layer(down.key.clone(), &dequant, rows, cols, format)?;
    }

    state.finalize(&dst_manifest)?;

    let bytes_written = std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0);
    Ok(AddFeatureMajorDownReport {
        vindex: vindex_dir.to_path_buf(),
        skipped: false,
        num_layers,
        bytes_written,
        wall_time: started.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_q4k_m_mix() {
        let c = Q4kConvertConfig::default();
        assert!(!c.down_q4k, "Q4K-M default: down stays Q6_K");
        assert!(!c.force);
    }

    #[test]
    fn down_q4k_opt_in_toggles_flag() {
        let c = Q4kConvertConfig {
            down_q4k: true,
            ..Default::default()
        };
        assert!(c.down_q4k);
    }
}
