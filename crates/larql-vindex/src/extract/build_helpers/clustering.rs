//! Relation clustering pipeline — k-means + label + write
//! `relation_clusters.json` / `feature_clusters.jsonl`.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_models::ModelWeights;

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::constants::{MAX_RELATION_CLUSTERS, RELATION_KMEANS_ITERS};
use crate::extract::stage_labels::STAGE_RELATION_CLUSTERS;
use crate::format::filenames::{FEATURE_CLUSTERS_JSONL, RELATION_CLUSTERS_JSON};

/// Collected data for relation clustering.
pub(crate) struct ClusterData {
    pub directions: Vec<f32>,
    pub features: Vec<(usize, usize)>,
    pub top_tokens: Vec<String>,
    #[allow(dead_code)]
    pub input_tokens: Vec<String>,
    pub output_tokens: Vec<String>,
}

/// Run the clustering and labeling pipeline on collected cluster data.
/// Writes `relation_clusters.json` and `feature_clusters.jsonl`.
pub(crate) fn run_clustering_pipeline(
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

    let optimal_k = MAX_RELATION_CLUSTERS.min(n_features);

    let (centres, assignments, _distances) =
        crate::clustering::kmeans(&matrix, optimal_k, RELATION_KMEANS_ITERS);

    let ref_dbs = crate::clustering::load_reference_databases();

    // Tier 1: output-only matching — Wikidata for L14-27 features.
    // WordNet is for L0-13 (linguistic). Each database matches its
    // own layer range.
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

    // Merge: Wikidata output labels > embedding/pattern labels.
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::{vocab_tokenizer, weights_with_embed};
    use super::*;

    #[test]
    fn run_clustering_pipeline_skips_empty_directions() {
        let toks = vocab_tokenizer(&[]);
        let embed = ndarray::Array2::<f32>::zeros((1, 4));
        let weights = weights_with_embed(embed, 1);
        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;
        let result = run_clustering_pipeline(
            ClusterData {
                directions: Vec::new(),
                features: Vec::new(),
                top_tokens: Vec::new(),
                input_tokens: Vec::new(),
                output_tokens: Vec::new(),
            },
            4,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_ok());
        assert!(!tmp.path().join(RELATION_CLUSTERS_JSON).exists());
        assert!(!tmp.path().join(FEATURE_CLUSTERS_JSONL).exists());
    }

    #[test]
    fn run_clustering_pipeline_writes_outputs_for_non_empty_data() {
        // Synthetic two-cluster setup: 6 directions in 4-D, 3 along
        // [1,0,0,0] and 3 along [0,1,0,0]. We only assert file
        // shapes — labels depend on side data files.
        let words = ["paris", "france", "berlin", "germany"];
        let toks = vocab_tokenizer(&words);
        let hidden = 4;
        let vocab_size = 5;
        let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
        for i in 0..4 {
            embed[[i + 1, i]] = 1.0;
        }
        let weights = weights_with_embed(embed, vocab_size);

        let mut directions = Vec::with_capacity(6 * hidden);
        for _ in 0..3 {
            directions.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }
        for _ in 0..3 {
            directions.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
        }
        let features = (0..6).map(|f| (0usize, f)).collect();
        let top_tokens = vec![
            "paris".into(),
            "france".into(),
            "berlin".into(),
            "germany".into(),
            "rome".into(),
            "italy".into(),
        ];
        let output_tokens = top_tokens.clone();

        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;
        let result = run_clustering_pipeline(
            ClusterData {
                directions,
                features,
                top_tokens,
                input_tokens: Vec::new(),
                output_tokens,
            },
            hidden,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_ok(), "pipeline returned: {result:?}");

        let json_path = tmp.path().join(RELATION_CLUSTERS_JSON);
        assert!(json_path.exists(), "relation_clusters.json written");
        let json = std::fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("k").is_some(), "k field present");
        assert!(parsed.get("centres").is_some(), "centres field present");
        assert!(parsed.get("labels").is_some(), "labels field present");
        assert!(parsed.get("counts").is_some(), "counts field present");

        let assign_path = tmp.path().join(FEATURE_CLUSTERS_JSONL);
        assert!(assign_path.exists(), "feature_clusters.jsonl written");
        let jsonl = std::fs::read_to_string(&assign_path).unwrap();
        let lines: Vec<_> = jsonl.lines().collect();
        assert_eq!(lines.len(), 6, "one line per feature");
        for line in lines {
            let r: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(r.get("l").is_some());
            assert!(r.get("f").is_some());
            assert!(r.get("c").is_some());
        }
    }

    #[test]
    fn run_clustering_pipeline_errors_on_shape_mismatch() {
        let toks = vocab_tokenizer(&["x"]);
        let embed = ndarray::Array2::<f32>::zeros((2, 4));
        let weights = weights_with_embed(embed, 2);
        let tmp = tempfile::tempdir().unwrap();
        let mut cb = crate::extract::callbacks::SilentBuildCallbacks;

        let result = run_clustering_pipeline(
            ClusterData {
                directions: vec![1.0; 8],
                features: vec![(0, 0), (0, 1), (0, 2)],
                top_tokens: vec!["a".into(), "b".into(), "c".into()],
                input_tokens: Vec::new(),
                output_tokens: vec!["a".into(), "b".into(), "c".into()],
            },
            4,
            &weights,
            &toks,
            tmp.path(),
            &mut cb,
        );
        assert!(result.is_err(), "shape mismatch must error");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("cluster data shape"));
    }
}
