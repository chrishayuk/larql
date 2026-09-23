//! GW-3A-F: frozen gate-derived postings for the FFN execution-support bridge.
//!
//! This command performs no prompt execution. It builds one maximum-width
//! source-token index, then emits exact per-layer prefixes for the preregistered
//! width ladder. Target labels and GW-0B contributions are not consumed here.

use super::*;
use larql_vindex::format::vindex3::knowledge::{FeatureAddress, KnowledgeView};
use larql_vindex::ndarray::Array1;
use std::collections::{BTreeMap, BTreeSet};

const WIDTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

fn sha_file(path: &Path) -> Result<String> {
    Ok(format!("sha256:{}", sha(&std::fs::read(path)?)))
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn group(candidates: &[FeatureAddress]) -> BTreeMap<usize, Vec<usize>> {
    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for address in candidates {
        grouped
            .entry(address.layer)
            .or_default()
            .push(address.feature);
    }
    grouped
}

fn canonical_preregistration_hash(document: &Value) -> Result<String> {
    let mut bound = document.clone();
    bound
        .as_object_mut()
        .ok_or("GW-3A-F preregistration must be an object")?
        .remove("preregistration_sha256");
    Ok(format!("sha256:{}", sha(&serde_json::to_vec(&bound)?)))
}

/// `--gw3af-postings CONTAINER SEALED_MANIFEST PREREGISTRATION OUTPUT.jsonl`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err(
            "Usage: observatory_record --gw3af-postings CONTAINER SEALED_MANIFEST PREREGISTRATION OUTPUT.jsonl".into(),
        );
    }
    let container = Path::new(&args[0]);
    let manifest_path = Path::new(&args[1]);
    let preregistration_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("GW-3A-F postings output already exists".into());
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    if manifest["schema"] != "larql.gw.phase1.manifest.v1" {
        return Err("GW-3A-F requires a sealed phase-one manifest".into());
    }
    let preregistration: Value = serde_json::from_slice(&std::fs::read(preregistration_path)?)?;
    if preregistration["schema"] != "larql.gw3af.preregistration.v1"
        || preregistration["status"] != "frozen_pre_lookup"
    {
        return Err("GW-3A-F requires a frozen preregistration".into());
    }
    let preregistration_sha256 = canonical_preregistration_hash(&preregistration)?;
    if preregistration["preregistration_sha256"].as_str() != Some(preregistration_sha256.as_str())
        || preregistration["authorities"]["sealed_gw0"]["bundle_sha256"]
            != manifest["bundle_sha256"]
        || preregistration["posting_rule"]["widths"] != json!(WIDTHS)
    {
        return Err("GW-3A-F preregistration identity or frozen ladder mismatch".into());
    }
    let manifest_root = manifest_path.parent().unwrap_or(Path::new("."));
    let census_path = manifest_root.join(
        manifest["census"]["path"]
            .as_str()
            .ok_or("sealed census path missing")?,
    );
    if sha_file(&census_path)? != manifest["census"]["sha256"] {
        return Err("sealed census hash mismatch".into());
    }
    let census = read_jsonl(&census_path)?;
    if census.len() as u64 != manifest["census"]["rows"].as_u64().unwrap_or(0) {
        return Err("sealed census row count mismatch".into());
    }

    eprintln!("GW-3A-F: verifying container and binding knowledge roles...");
    let inspection = inspect_container(container, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("no closed target plan")?;
    let container_identity = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(
            &json!({"index": inspection.index, "graph": inspection.graph}),
        )?)
    );
    if census
        .iter()
        .any(|row| row["provenance"]["container_identity"].as_str() != Some(&container_identity))
    {
        return Err("container identity differs from sealed GW-0 provenance".into());
    }
    if preregistration["authorities"]["container_identity"].as_str()
        != Some(container_identity.as_str())
    {
        return Err("container identity differs from GW-3A-F preregistration".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(container)?;
    let view = KnowledgeView::from_plan(&plan, &store, &tokenizer)?;
    let layers = view.loaded_layers();

    let subjects = census
        .iter()
        .map(|row| {
            row["semantic_edge"]["subject"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "semantic subject missing".into())
        })
        .collect::<Result<BTreeSet<String>>>()?;
    let mut encoded = BTreeMap::<String, Vec<u32>>::new();
    let mut source_ids = BTreeSet::new();
    for subject in &subjects {
        let ids = tokenizer
            .encode(subject.as_str(), false)
            .map_err(|error| format!("tokenize subject {subject:?}: {error}"))?
            .get_ids()
            .to_vec();
        if ids.is_empty() || ids.iter().any(|id| *id as usize >= view.vocab_size()) {
            return Err(format!("subject {subject:?} has no valid source-token key").into());
        }
        source_ids.extend(ids.iter().copied());
        encoded.insert(subject.clone(), ids);
    }

    eprintln!(
        "GW-3A-F: building width-{} postings for {} source tokens across {} layers...",
        WIDTHS[WIDTHS.len() - 1],
        source_ids.len(),
        layers.len()
    );
    let index = view.build_posting_index(
        &layers,
        &source_ids.into_iter().collect::<Vec<_>>(),
        WIDTHS[WIDTHS.len() - 1],
        0,
    )?;
    let (embedding, embed_scale) = view.embedding();
    let plan_sha256 = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    let mut rows = Vec::with_capacity(subjects.len() * WIDTHS.len());

    for (subject_index, subject) in subjects.iter().enumerate() {
        let token_ids = &encoded[subject];
        let mut query = Array1::<f32>::zeros(view.hidden_size());
        for token_id in token_ids {
            query += &embedding
                .row(*token_id as usize)
                .mapv(|value| value * embed_scale);
        }
        query.mapv_inplace(|value| value / token_ids.len() as f32);

        let maximum = index.lookup_at_width(token_ids, None, WIDTHS[WIDTHS.len() - 1]);
        let maximum_by_layer = group(&maximum.candidates);
        let exact_by_layer = layers
            .iter()
            .map(|layer| {
                let count = maximum_by_layer.get(layer).map(Vec::len).unwrap_or(0);
                let features = view
                    .gate_knn(*layer, &query, count)
                    .into_iter()
                    .map(|(feature, _)| feature)
                    .collect::<Vec<_>>();
                (*layer, features)
            })
            .collect::<BTreeMap<_, _>>();

        for width in WIDTHS {
            let lookup = index.lookup_at_width(token_ids, None, width);
            let postings = group(&lookup.candidates);
            let layer_rows = layers
                .iter()
                .map(|layer| {
                    let candidates = postings.get(layer).cloned().unwrap_or_default();
                    let exact = exact_by_layer[layer][..candidates.len()].to_vec();
                    let source_token_postings = token_ids
                        .iter()
                        .map(|token_id| {
                            let features = index
                                .source_postings(*token_id)
                                .iter()
                                .filter(|address| address.layer == *layer)
                                .take(width)
                                .map(|address| address.feature)
                                .collect::<Vec<_>>();
                            json!({"token_id":token_id,"features":features})
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "layer": layer,
                        "eligible_features": view.num_features(*layer),
                        "postings": candidates,
                        "source_token_postings": source_token_postings,
                        "exact_walk_matched": exact,
                    })
                })
                .collect::<Vec<_>>();
            rows.push(json!({
                "schema":"larql.gw3af.postings.v1",
                "preregistration_sha256":preregistration_sha256,
                "bundle_sha256":manifest["bundle_sha256"],
                "container_identity":container_identity,
                "plan_sha256":plan_sha256,
                "vocab_size":view.vocab_size(),
                "subject":subject,
                "subject_token_ids":token_ids,
                "width_per_source_token_per_layer":width,
                "layers":layer_rows,
                "costs":{
                    "candidate_features":lookup.candidates.len(),
                    "eligible_features":lookup.eligible_features,
                    "candidate_fraction":lookup.candidate_fraction(),
                    "source_postings_touched":lookup.source_postings_touched,
                    "target_postings_touched":lookup.target_postings_touched,
                    "logical_bytes_touched":lookup.logical_bytes_touched,
                    "logical_index_bytes":index.logical_index_bytes_at_width(width),
                    "gate_bytes_at_lookup":0,
                    "dot_products_at_lookup":0,
                }
            }));
        }
        eprintln!("GW-3A-F subjects: {}/{}", subject_index + 1, subjects.len());
    }

    let bytes = rows
        .iter()
        .flat_map(|row| {
            let mut line = serde_json::to_vec(row).unwrap();
            line.push(b'\n');
            line
        })
        .collect::<Vec<_>>();
    atomic_json(output, &bytes)?;
    eprintln!("GW-3A-F complete: {} subject/width rows", rows.len());
    Ok(())
}
