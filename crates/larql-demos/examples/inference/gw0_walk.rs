//! Exact dense V3 gate ranking for the frozen GW-0 input.
use super::*;
use larql_vindex::format::vindex3::knowledge::KnowledgeView;
use larql_vindex::ndarray::Array1;
use std::collections::BTreeMap;

/// `--gw0-walk CONTAINER MANIFEST.json INPUT.jsonl OUTPUT.jsonl`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err(
            "Usage: observatory_record --gw0-walk CONTAINER MANIFEST.json INPUT.jsonl OUTPUT.jsonl"
                .into(),
        );
    }
    let root = Path::new(&args[0]);
    let manifest: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let input_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("exact WALK output already exists".into());
    }
    if format!("sha256:{}", sha(&std::fs::read(input_path)?)) != manifest["rows"]["sha256"] {
        return Err("frozen input hash mismatch".into());
    }
    let rows: Vec<Value> = std::fs::read_to_string(input_path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    eprintln!("Verifying container and binding V3 knowledge roles...");
    let inspection = inspect_container(root, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, root, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("no closed target plan")?;
    let plan_hash = sha(&serde_json::to_vec(&plan)?);
    let store = OperandStore::open(root, &inspection)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(root)?;
    let view = KnowledgeView::from_plan(&plan, &store, &tokenizer)?;
    let (embedding, scale) = view.embedding();
    let layers = view.loaded_layers();
    let mut cache = BTreeMap::<String, Vec<Value>>::new();
    let mut output_rows = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let subject = row["semantic_edge"]["subject"]
            .as_str()
            .ok_or("subject absent")?;
        let hits = if let Some(hits) = cache.get(subject) {
            hits.clone()
        } else {
            let encoded = tokenizer
                .encode(subject, false)
                .map_err(|error| format!("tokenize subject: {error}"))?;
            if encoded.get_ids().is_empty() {
                return Err(format!("empty subject: {subject}").into());
            }
            let mut query = Array1::<f32>::zeros(view.hidden_size());
            let mut count = 0usize;
            for token in encoded.get_ids() {
                if (*token as usize) < embedding.nrows() {
                    query += &embedding.row(*token as usize).mapv(|value| value * scale);
                    count += 1;
                }
            }
            if count == 0 {
                return Err(format!("no in-vocabulary subject tokens: {subject}").into());
            }
            query.mapv_inplace(|value| value / count as f32);
            let mut hits = Vec::new();
            for layer in &layers {
                for (rank, (feature, score)) in
                    view.gate_knn(*layer, &query, 20).into_iter().enumerate()
                {
                    hits.push(json!({"layer":layer,"feature":feature,"rank":rank+1,"score":score}));
                }
            }
            cache.insert(subject.to_string(), hits.clone());
            hits
        };
        output_rows.push(json!({"schema":"larql.gw0.exact-walk.v1","edge_id":row["edge_id"],
            "manifest_sha256":manifest["manifest_sha256"],"query":{"kind":"mean_scaled_subject_token_embedding","subject":subject},
            "top_k_per_layer":20,"layers":layers,"plan_sha256":format!("sha256:{plan_hash}"),"hits":hits}));
        if (index + 1).is_multiple_of(10) {
            eprintln!("Exact WALK {}/{}", index + 1, rows.len());
        }
    }
    let bytes: Vec<u8> = output_rows
        .iter()
        .flat_map(|row| {
            let mut line = serde_json::to_vec(row).unwrap();
            line.push(b'\n');
            line
        })
        .collect();
    atomic_json(output, &bytes)?;
    eprintln!(
        "Exact WALK complete: {} rows, {} unique subjects",
        rows.len(),
        cache.len()
    );
    Ok(())
}
