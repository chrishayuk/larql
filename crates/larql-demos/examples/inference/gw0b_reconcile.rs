//! GW-0B edge-address reconciliation over the sealed GW-0 corpus.
//!
//! Both commands are offline: neither executes a prompt. Attribution consumes
//! the sealed carrier-before/delta artifacts; promotion consumes only exact
//! WALK addresses (plus optional attributed addresses) and container operands.

use super::*;
use larql_vindex::format::vindex3::knowledge::{
    DenseFfnAttribution, DenseFfnLayerView, FeatureAddress, KnowledgeView,
};
use std::collections::{BTreeMap, BTreeSet};

const RELATIVE_L2_TOLERANCE: f64 = 1e-4;
const RELATIVE_LINF_TOLERANCE: f64 = 1e-4;
const CONTRIBUTION_TOP_K: usize = 100;

fn sha_file(path: &Path) -> Result<String> {
    Ok(format!("sha256:{}", sha(&std::fs::read(path)?)))
}

fn jsonl(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

struct Bundle {
    manifest: Value,
    census: Vec<Value>,
    artifact_root: std::path::PathBuf,
}

impl Bundle {
    fn open(path: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if manifest["schema"] != "larql.gw.phase1.manifest.v1" {
            return Err("GW-0B requires a sealed larql.gw.phase1.manifest.v1 manifest".into());
        }
        let root = path.parent().unwrap_or(Path::new("."));
        let census_path = root.join(
            manifest["census"]["path"]
                .as_str()
                .ok_or("bundle census path missing")?,
        );
        if sha_file(&census_path)? != manifest["census"]["sha256"] {
            return Err("sealed census hash mismatch".into());
        }
        let census = jsonl(&census_path)?;
        if census.len() as u64 != manifest["census"]["rows"].as_u64().unwrap_or(0) {
            return Err("sealed census row count mismatch".into());
        }
        let artifact_root = root.join(
            manifest["artifact_root"]
                .as_str()
                .ok_or("bundle artifact root missing")?,
        );
        Ok(Self {
            manifest,
            census,
            artifact_root,
        })
    }

    fn bundle_hash(&self) -> &str {
        self.manifest["bundle_sha256"].as_str().unwrap()
    }

    fn artifact<'a>(row: &'a Value, kind: &str) -> Result<&'a Value> {
        row["artifacts"]
            .as_array()
            .and_then(|items| items.iter().find(|item| item["kind"] == kind))
            .ok_or_else(|| format!("{} has no {kind} artifact", row["edge_id"]).into())
    }

    fn read_artifact(&self, artifact: &Value) -> Result<Vec<u8>> {
        let path = self
            .artifact_root
            .join(artifact["path"].as_str().ok_or("artifact path missing")?);
        let bytes = std::fs::read(&path)?;
        if bytes.len() as u64 != artifact["bytes"].as_u64().unwrap_or(u64::MAX)
            || format!("sha256:{}", sha(&bytes)) != artifact["sha256"]
        {
            return Err(format!("sealed artifact mismatch: {}", path.display()).into());
        }
        Ok(bytes)
    }
}

fn bind(
    container: &Path,
) -> Result<(
    larql_vindex::format::vindex3::opplan::ComponentOpPlan,
    OperandStore,
    String,
)> {
    let inspection = inspect_container(container, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("no closed target plan")?;
    let identity = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(
            &json!({"index": inspection.index, "graph": inspection.graph}),
        )?)
    );
    let store = OperandStore::open(container, &inspection)?;
    Ok((plan, store, identity))
}

fn verify_identity(bundle: &Bundle, container_identity: &str) -> Result<()> {
    if bundle
        .census
        .iter()
        .any(|row| row["provenance"]["container_identity"].as_str() != Some(container_identity))
    {
        return Err("container identity differs from sealed GW-0 provenance".into());
    }
    Ok(())
}

fn f32s(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err("f32 artifact byte length is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

struct SiteInput {
    edge_id: String,
    before: Vec<f32>,
    delta: Vec<f32>,
}

fn attribution_inputs(bundle: &Bundle, hidden: usize) -> Result<BTreeMap<usize, Vec<SiteInput>>> {
    let mut by_layer = BTreeMap::<usize, Vec<SiteInput>>::new();
    for (row_index, row) in bundle.census.iter().enumerate() {
        let edge_id = row["edge_id"].as_str().ok_or("edge id missing")?;
        let layers: BTreeSet<usize> = row["transition_candidate"]["sites"]
            .as_array()
            .ok_or("candidate sites missing")?
            .iter()
            .filter(|site| site["site"] == "ffn")
            .map(|site| site["layer"].as_u64().unwrap() as usize)
            .collect();
        if layers.is_empty() {
            continue;
        }
        let record_artifact = Bundle::artifact(row, "execution_record")?;
        let record: Value = serde_json::from_slice(&bundle.read_artifact(record_artifact)?)?;
        if record["edge_id"] != edge_id || record["prompt_row_index"] != row_index {
            return Err(format!("{edge_id}: execution-record binding mismatch").into());
        }
        let sites = record["sites"].as_array().ok_or("record sites missing")?;
        let before = f32s(&bundle.read_artifact(Bundle::artifact(row, "carrier-before")?)?)?;
        let delta = f32s(&bundle.read_artifact(Bundle::artifact(row, "delta")?)?)?;
        if before.len() != sites.len() * hidden || delta.len() != sites.len() * hidden {
            return Err(format!("{edge_id}: carrier artifact geometry mismatch").into());
        }
        for layer in layers {
            let matching: Vec<_> = sites
                .iter()
                .enumerate()
                .filter(|(_, site)| site["site"] == "ffn" && site["layer"] == layer)
                .map(|(index, _)| index)
                .collect();
            if matching.len() != 1 {
                return Err(format!(
                    "{edge_id}: expected one recorded FFN write at layer {layer}, found {}",
                    matching.len()
                )
                .into());
            }
            let start = matching[0] * hidden;
            by_layer.entry(layer).or_default().push(SiteInput {
                edge_id: edge_id.to_string(),
                before: before[start..start + hidden].to_vec(),
                delta: delta[start..start + hidden].to_vec(),
            });
        }
    }
    Ok(by_layer)
}

fn attribution_json(
    edge_id: &str,
    value: DenseFfnAttribution,
    bundle: &Bundle,
    container_identity: &str,
    plan_sha256: &str,
    backend_identity: &str,
) -> Value {
    json!({
        "schema":"larql.gw0b.ffn-attribution.v1",
        "edge_id":edge_id,
        "bundle_sha256":bundle.bundle_hash(),
        "container_identity":container_identity,
        "plan_sha256":plan_sha256,
        "backend_identity":backend_identity,
        "weight_image":"production-selected-effective-values",
        "status":"reconstructed",
        "site":{"layer":value.layer,"site":"ffn"},
        "features":value.features,
        "ranking":{"kind":"post-write-contribution-l2","top_k":value.contributions.len()},
        "proof":{
            "relative_l2_error":value.proof.relative_l2_error,
            "relative_linf_error":value.proof.relative_linf_error,
            "max_abs_error":value.proof.max_abs_error,
            "relative_l2_tolerance":value.proof.relative_l2_tolerance,
            "relative_linf_tolerance":value.proof.relative_linf_tolerance,
        },
        "contributions":value.contributions.into_iter().enumerate().map(|(rank, item)| json!({
            "layer":item.layer,"feature":item.feature,"rank":rank+1,
            "activation":item.activation,"contribution_l2":item.contribution_l2,
            "observed_projection_fraction":item.observed_projection_fraction,
        })).collect::<Vec<_>>()
    })
}

/// `--gw0b-attribute CONTAINER SEALED_MANIFEST OUTPUT_DIR [START_LAYER [LIMIT]]`
pub(super) fn attribute(args: &[String]) -> Result<()> {
    if !(3..=5).contains(&args.len()) {
        return Err("Usage: observatory_record --gw0b-attribute CONTAINER SEALED_MANIFEST OUTPUT_DIR [START_LAYER [LIMIT]]".into());
    }
    let container = Path::new(&args[0]);
    let bundle = Bundle::open(Path::new(&args[1]))?;
    let output = Path::new(&args[2]);
    let start = args
        .get(3)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(0usize);
    let limit = args
        .get(4)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(usize::MAX);
    std::fs::create_dir_all(output)?;
    let (plan, store, identity) = bind(container)?;
    verify_identity(&bundle, &identity)?;
    let plan_hash = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    let backend_identity = ProductionBackend::new().identity().to_string();
    let hidden = plan
        .embedding
        .as_ref()
        .and_then(|embedding| embedding.table.shape.get(1))
        .copied()
        .ok_or("plan embedding does not declare hidden size")?;
    let inputs = attribution_inputs(&bundle, hidden)?;
    for (&layer, rows) in inputs
        .iter()
        .filter(|(layer, _)| **layer >= start)
        .take(limit)
    {
        let output_path = output.join(format!("layer-{layer:02}.jsonl"));
        if output_path.exists() {
            return Err(format!(
                "attribution output already exists: {}",
                output_path.display()
            )
            .into());
        }
        eprintln!(
            "GW-0B2 layer {layer}: loading operands for {} writes",
            rows.len()
        );
        let backend = ProductionBackend::new();
        let prepared = PreparedOperands::load(
            &plan,
            &store,
            &backend,
            ExecutionSlice::LayerRange {
                start: layer,
                end: layer + 1,
            },
        )?;
        let view = DenseFfnLayerView::from_prepared(&plan, &store, &prepared, layer)?;
        let mut output_rows = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let result = view.attribute(
                &row.before,
                &row.delta,
                CONTRIBUTION_TOP_K,
                RELATIVE_L2_TOLERANCE,
                RELATIVE_LINF_TOLERANCE,
            );
            output_rows.push(match result {
                Ok(value) => attribution_json(
                    &row.edge_id,
                    value,
                    &bundle,
                    &identity,
                    &plan_hash,
                    &backend_identity,
                ),
                Err(error) => json!({
                    "schema":"larql.gw0b.ffn-attribution.v1","edge_id":row.edge_id,
                    "bundle_sha256":bundle.bundle_hash(),"status":"refused",
                    "container_identity":identity,"plan_sha256":plan_hash,
                    "backend_identity":backend_identity,
                    "weight_image":"production-selected-effective-values",
                    "site":{"layer":layer,"site":"ffn"},
                    "refusal":{"class":"reconstruction_failed","message":error.to_string()},
                    "tolerances":{"relative_l2":RELATIVE_L2_TOLERANCE,"relative_linf":RELATIVE_LINF_TOLERANCE}
                }),
            });
            if (index + 1).is_multiple_of(25) {
                eprintln!("GW-0B2 layer {layer}: {}/{}", index + 1, rows.len());
            }
        }
        let bytes = output_rows
            .iter()
            .flat_map(|row| {
                let mut line = serde_json::to_vec(row).unwrap();
                line.push(b'\n');
                line
            })
            .collect::<Vec<_>>();
        atomic_json(&output_path, &bytes)?;
    }
    Ok(())
}

fn attributed_addresses(
    root: Option<&Path>,
    bundle_hash: &str,
) -> Result<BTreeSet<FeatureAddress>> {
    let mut addresses = BTreeSet::new();
    let Some(root) = root else {
        return Ok(addresses);
    };
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
            continue;
        }
        for row in jsonl(&path)? {
            if row["bundle_sha256"] != bundle_hash || row["status"] != "reconstructed" {
                return Err(format!("unbound or refused attribution in {}", path.display()).into());
            }
            for item in row["contributions"]
                .as_array()
                .ok_or("contributions missing")?
            {
                addresses.insert(FeatureAddress {
                    layer: item["layer"].as_u64().unwrap() as usize,
                    feature: item["feature"].as_u64().unwrap() as usize,
                });
            }
        }
    }
    Ok(addresses)
}

/// `--gw0b-promote CONTAINER SEALED_MANIFEST OUTPUT.jsonl [ATTRIBUTION_DIR]`
pub(super) fn promote(args: &[String]) -> Result<()> {
    if !(3..=4).contains(&args.len()) {
        return Err("Usage: observatory_record --gw0b-promote CONTAINER SEALED_MANIFEST OUTPUT.jsonl [ATTRIBUTION_DIR]".into());
    }
    let container = Path::new(&args[0]);
    let bundle = Bundle::open(Path::new(&args[1]))?;
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("feature-promotion output already exists".into());
    }
    let (plan, store, identity) = bind(container)?;
    verify_identity(&bundle, &identity)?;
    let mut addresses = BTreeSet::new();
    for row in &bundle.census {
        for hit in row["exact_walk_result"]["hits"]
            .as_array()
            .ok_or("WALK hits missing")?
        {
            addresses.insert(FeatureAddress {
                layer: hit["layer"].as_u64().unwrap() as usize,
                feature: hit["feature"].as_u64().unwrap() as usize,
            });
        }
    }
    addresses.extend(attributed_addresses(
        args.get(3).map(Path::new),
        bundle.bundle_hash(),
    )?);
    let tokenizer = larql_vindex::load_vindex_tokenizer(container)?;
    let view = KnowledgeView::from_plan(&plan, &store, &tokenizer)?;
    let mut by_layer = BTreeMap::<usize, Vec<usize>>::new();
    for address in addresses {
        by_layer
            .entry(address.layer)
            .or_default()
            .push(address.feature);
    }
    let plan_hash = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    let mut rows = Vec::new();
    for (layer, features) in by_layer {
        eprintln!(
            "GW-0B1 layer {layer}: promoting {} addresses",
            features.len()
        );
        let metas = view.feature_promotions(layer, &features)?;
        for (feature, meta) in features.into_iter().zip(metas) {
            rows.push(json!({
                "schema":"larql.gw0b.feature-promotion.v1",
                "bundle_sha256":bundle.bundle_hash(),"container_identity":identity,
                "plan_sha256":plan_hash,"address":{"layer":layer,"feature":feature},
                "status":if meta.is_some(){"annotated"}else{"no_decodable_tokens"},
                "candidates":meta.map(|value| value.top_k.into_iter().enumerate().map(|(rank, token)| json!({
                    "rank":rank+1,"token_id":token.token_id,"token":token.token,"score":token.logit
                })).collect::<Vec<_>>()).unwrap_or_default()
            }));
        }
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
    eprintln!("GW-0B1 complete: {} unique feature addresses", rows.len());
    Ok(())
}
