#![allow(dead_code)]

//! Helpers for the `build_vindex` extraction pipeline.
//!
//! Each function is a discrete pipeline stage or utility used by
//! `super::build::build_vindex`:
//!
//! - `chrono_now`              — ISO-8601 timestamp without `chrono`.
//! - `build_whole_word_vocab`  — reduce the vocab to whole-word tokens
//!                               + matching embedding rows.
//! - `compute_gate_top_tokens` — per-feature top whole-word token (the
//!                               "what activates this feature" label).
//! - `compute_offset_direction`— normalised `embed[output] - embed[input]`
//!                               direction; the relation vector for
//!                               clustering.
//! - `ClusterData`             — collected cluster inputs.
//! - `run_clustering_pipeline` — k-means + label + write
//!                               `relation_clusters.json` /
//!                               `feature_clusters.jsonl`.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::extract::stage_labels::STAGE_RELATION_CLUSTERS;
use crate::format::filenames::{FEATURE_CLUSTERS_JSONL, RELATION_CLUSTERS_JSON};

use larql_models::ModelWeights;
use ndarray::Array2;

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

// ── Timestamp ──────────────────────────────────────────────────────────

/// Simple ISO 8601 timestamp without chrono dependency.
pub(crate) fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Rough UTC timestamp — good enough for provenance
    let days = secs / 86400;
    let years_approx = 1970 + days / 365;
    let remainder_days = days % 365;
    let months = remainder_days / 30 + 1;
    let day = remainder_days % 30 + 1;
    let hour = (secs % 86400) / 3600;
    let min = (secs % 3600) / 60;
    let sec = secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years_approx,
        months.min(12),
        day.min(31),
        hour,
        min,
        sec
    )
}

// ── Whole-word vocab ───────────────────────────────────────────────────

/// Build the whole-word vocabulary: tokens that decode as 3+ char alphabetic words.
/// Returns (token_ids, reduced_embedding_matrix).
pub(crate) fn build_whole_word_vocab(
    tokenizer: &tokenizers::Tokenizer,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    vocab_size: usize,
    hidden_size: usize,
) -> (Vec<usize>, Array2<f32>) {
    let mut ww_ids: Vec<usize> = Vec::new();
    for id in 0..vocab_size {
        if let Ok(tok) = tokenizer.decode(&[id as u32], true) {
            let tok = tok.trim();
            if tok.len() >= 3
                && tok
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '\'')
            {
                ww_ids.push(id);
            }
        }
    }

    let ww_count = ww_ids.len();
    let mut ww_embed = Array2::<f32>::zeros((ww_count, hidden_size));
    for (i, &id) in ww_ids.iter().enumerate() {
        ww_embed.row_mut(i).assign(&embed.row(id));
    }

    eprintln!(
        "    Whole-word vocab: {} tokens (of {})",
        ww_count, vocab_size
    );
    (ww_ids, ww_embed)
}

// ── Gate top tokens ────────────────────────────────────────────────────

/// Compute gate top tokens for features at a layer using whole-word embeddings.
/// Returns a Vec<String> of decoded whole-word tokens, one per feature.
pub(super) fn compute_gate_top_tokens(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    layer: usize,
    num_features: usize,
    ww_ids: &[usize],
    ww_embed: &Array2<f32>,
) -> Vec<String> {
    let gate_key = weights.arch.ffn_gate_key(layer);
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return vec![String::new(); num_features],
    };

    let mut tokens = vec![String::new(); num_features];
    let gbatch = 1024;
    for gstart in (0..num_features).step_by(gbatch) {
        let gend = (gstart + gbatch).min(num_features);
        let chunk = w_gate.slice(ndarray::s![gstart..gend, ..]);
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let proj = cpu.matmul_transb(ww_embed.view(), chunk.view());
        for f in 0..(gend - gstart) {
            let col = proj.column(f);
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in col.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            let tok_id = ww_ids[best_idx];
            tokens[gstart + f] = tokenizer
                .decode(&[tok_id as u32], true)
                .unwrap_or_default()
                .trim()
                .to_string();
        }
    }
    tokens
}

// ── Offset direction ───────────────────────────────────────────────────

/// Compute the offset direction for a gate→down feature pair.
/// Returns normalized(output_embed - input_embed) or None if invalid.
pub(super) fn compute_offset_direction(
    gate_token: &str,
    output_token_id: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    hidden_size: usize,
    vocab_size: usize,
) -> Option<Vec<f32>> {
    if gate_token.is_empty() || output_token_id <= 2 || output_token_id >= vocab_size {
        return None;
    }

    // Get gate token embedding (may be multi-subword)
    let enc = tokenizer.encode(gate_token, false).ok()?;
    let ids = enc.get_ids();
    let valid: Vec<usize> = ids
        .iter()
        .filter(|&&id| id > 2)
        .map(|&id| id as usize)
        .filter(|&id| id < vocab_size)
        .collect();
    if valid.is_empty() {
        return None;
    }

    let mut input_avg = vec![0.0f32; hidden_size];
    for &id in &valid {
        for (j, &v) in weights.embed.row(id).iter().enumerate() {
            input_avg[j] += v;
        }
    }
    let n = valid.len() as f32;
    for v in &mut input_avg {
        *v /= n;
    }

    let output_embed = weights.embed.row(output_token_id);
    let offset: Vec<f32> = output_embed
        .iter()
        .zip(input_avg.iter())
        .map(|(o, i)| o - i)
        .collect();
    let norm: f32 = offset.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        Some(offset.iter().map(|v| v / norm).collect())
    } else {
        None
    }
}

// ── Clustering ─────────────────────────────────────────────────────────

/// Collected data for relation clustering.
pub(super) struct ClusterData {
    pub directions: Vec<f32>,
    pub features: Vec<(usize, usize)>,
    pub top_tokens: Vec<String>,
    #[allow(dead_code)]
    pub input_tokens: Vec<String>,
    pub output_tokens: Vec<String>,
}

/// Run the clustering and labeling pipeline on collected cluster data.
/// Writes relation_clusters.json and feature_clusters.jsonl.
pub(super) fn run_clustering_pipeline(
    data: ClusterData,
    hidden_size: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    output_dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    if data.directions.is_empty() {
        return Ok(());
    }

    callbacks.on_stage(STAGE_RELATION_CLUSTERS);

    let n_features = data.features.len();
    let matrix = ndarray::Array2::from_shape_vec((n_features, hidden_size), data.directions)
        .map_err(|e| VindexError::Parse(format!("cluster data shape: {e}")))?;

    let optimal_k = 512.min(n_features);

    let (centres, assignments, _distances) = crate::clustering::kmeans(&matrix, optimal_k, 50);

    // Load reference databases
    let ref_dbs = crate::clustering::load_reference_databases();

    // Tier 1: output-only matching — Wikidata ONLY for L14-27 features.
    // WordNet is for L0-13 (linguistic). Wikidata is for L14-27 (factual).
    // They don't compete — each database matches its own layer range.
    let wikidata_refs: Vec<&crate::clustering::pair_matching::RelationDatabase> =
        ref_dbs.wikidata.iter().collect();
    let output_labels = if !wikidata_refs.is_empty() {
        crate::clustering::pair_matching::label_clusters_from_outputs(
            &assignments,
            &data.output_tokens,
            optimal_k,
            &wikidata_refs,
        )
    } else {
        vec![None; optimal_k]
    };

    let output_labeled = output_labels.iter().filter(|l| l.is_some()).count();
    eprintln!(
        "  Wikidata output matching: {}/{} clusters labeled",
        output_labeled, optimal_k
    );

    // Tier 2+3: embedding projection + pattern detection
    let (embed_labels, top_tokens_per_cluster) =
        crate::clustering::auto_label_clusters_from_embeddings(
            &centres,
            &weights.embed,
            tokenizer,
            &assignments,
            &data.top_tokens,
            optimal_k,
        );

    // Merge: Wikidata output labels > embedding/pattern labels
    let labels: Vec<String> = (0..optimal_k)
        .map(|c| {
            output_labels[c]
                .clone()
                .unwrap_or_else(|| embed_labels[c].clone())
        })
        .collect();

    let mut counts = vec![0usize; optimal_k];
    for &a in &assignments {
        if a < optimal_k {
            counts[a] += 1;
        }
    }

    // Write relation_clusters.json
    let cluster_result = crate::clustering::ClusterResult {
        k: optimal_k,
        centres: centres.rows().into_iter().map(|r| r.to_vec()).collect(),
        labels,
        counts,
        top_tokens: top_tokens_per_cluster,
    };

    let clusters_json = serde_json::to_string_pretty(&cluster_result)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join(RELATION_CLUSTERS_JSON), clusters_json)?;

    // Write per-feature cluster assignments
    let assign_path = output_dir.join(FEATURE_CLUSTERS_JSONL);
    let mut assign_file = BufWriter::new(std::fs::File::create(&assign_path)?);
    for (i, &(layer, feat)) in data.features.iter().enumerate() {
        let record = serde_json::json!({ "l": layer, "f": feat, "c": assignments[i] });
        serde_json::to_writer(&mut assign_file, &record)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        assign_file.write_all(b"\n")?;
    }
    assign_file.flush()?;

    callbacks.on_stage_done(
        &format!(
            "relation_clusters (k={}, {} features)",
            optimal_k, n_features
        ),
        0.0,
    );

    Ok(())
}
