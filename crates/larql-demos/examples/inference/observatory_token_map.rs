//! Offline token geometry from the existing captured carriers; no inference.
use super::*;

fn unit(values: &[f64]) -> Result<Vec<f64>> {
    let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err("degenerate/non-finite token-map direction".into());
    }
    Ok(values.iter().map(|v| v / norm).collect())
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn plane(a: &[f64], b: &[f64]) -> Result<[Vec<f64>; 2]> {
    if a.len() != b.len() {
        return Err("token-map width mismatch".into());
    }
    let x = unit(a)?;
    let amount = dot(&x, b);
    let y = unit(
        &b.iter()
            .zip(&x)
            .map(|(b, x)| b - amount * x)
            .collect::<Vec<_>>(),
    )?;
    Ok([x, y])
}
fn xy(v: &[f64], axes: &[Vec<f64>; 2]) -> [f64; 2] {
    [dot(v, &axes[0]), dot(v, &axes[1])]
}

/// --token-map CONTAINER RECORD LENSES CARRIERS OUTPUT TOKEN_A TOKEN_B [TOKEN...]
pub(super) fn run(args: &[String]) -> Result<()> {
    if !(7..=21).contains(&args.len()) {
        return Err("Usage: observatory_record --token-map CONTAINER RECORD.json LENSES.json CARRIERS.f32 OUTPUT.json TOKEN_A TOKEN_B [TOKEN ...]".into());
    }
    let output = Path::new(&args[4]);
    if output.exists() {
        return Err("token-map output already exists".into());
    }
    let source_bytes = std::fs::read(&args[1])?;
    let source: Value = serde_json::from_slice(&source_bytes)?;
    let lens_bytes = std::fs::read(&args[2])?;
    let lens: Value = serde_json::from_slice(&lens_bytes)?;
    let payload = std::fs::read(&args[3])?;
    if source["schema"] != "larql.observatory.standard.v1"
        || lens["schema"] != "larql.observatory.lenses.v1"
        || lens["source_sha256"] != sha(&source_bytes)
        || lens["run_id"] != source["run_id"]
        || lens["carrier_payload"]["sha256"] != sha(&payload)
        || source["provenance"] != "executor"
        || lens["provenance"] != "executor"
    {
        return Err("source/lens/carrier binding mismatch".into());
    }
    let root = Path::new(&args[0]);
    let inspection = inspect_container(root, true)?;
    if !inspection.is_coherent() {
        return Err("container verification failed".into());
    }
    let container_hash = sha(&serde_json::to_vec(
        &json!({"index": inspection.index, "graph": inspection.graph}),
    )?);
    if source["identity"]["container"] != format!("sha256:{container_hash}") {
        return Err("container differs from source record".into());
    }
    let outcome = plan_component_ops(&inspection, root, "target")?;
    if !outcome.closed() {
        return Err("token-map plan refused".into());
    }
    let plan = outcome.plan.ok_or("no plan")?;
    if !plan.residual_topology.is_single_stream()
        || source["identity"]["plan"] != format!("sha256:{}", sha(&serde_json::to_vec(&plan)?))
    {
        return Err("plan/topology mismatch".into());
    }
    let store = OperandStore::open(root, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    if source["identity"]["lowering"] != ExecutionProvenance::of(&ops).fingerprint() {
        return Err("realization differs from captured run".into());
    }
    let tokenizer_bytes = std::fs::read(root.join("tokenizer.json"))?;
    if source["identity"]["tokenizer"] != format!("sha256:{}", sha(&tokenizer_bytes)) {
        return Err("tokenizer differs from record".into());
    }
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
    let mut ids = Vec::new();
    for label in &args[5..] {
        let encoded = tokenizer.encode(label.as_str(), false)?;
        if encoded.len() != 1 || ids.contains(&encoded.get_ids()[0]) {
            return Err("landmarks must be distinct single tokens".into());
        }
        ids.push(encoded.get_ids()[0]);
    }
    let directions = head_rows(&plan, &store, &ids, ops.hidden())?
        .iter()
        .map(|row| unit(&row.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()))
        .collect::<Result<Vec<_>>>()?;
    let axes = plane(&directions[0], &directions[1])?;
    let axes_hash = sha(&axes
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>());
    let mut hash_input = axes
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    hash_input.extend(container_hash.as_bytes());
    for id in &ids {
        hash_input.extend(id.to_le_bytes());
    }
    let basis_hash = sha(&hash_input);
    let writes: Vec<_> = source["events"]
        .as_array()
        .ok_or("missing events")?
        .iter()
        .filter(|e| e["kind"] == "CarrierWrite")
        .collect();
    let vocab = lens["vocabulary"]["rows"]
        .as_array()
        .ok_or("missing vocabulary rows")?;
    if writes.len() != vocab.len()
        || lens["carrier_payload"]["width"] != ops.hidden()
        || lens["carrier_payload"]["rows"] != writes.len()
        || payload.len() != writes.len() * ops.hidden() * 4
    {
        return Err("carrier payload geometry differs".into());
    }
    let mut rows = Vec::new();
    for (index, e) in writes.iter().enumerate() {
        let s = &e["stats"];
        let role = match s["site"].as_str() {
            Some("Attention") => "attention_write",
            Some("Ffn") => "ffn_write",
            _ => return Err("unsupported site".into()),
        };
        let site = format!("{}:{}:{role}:main:residual", s["position"], s["layer"]);
        if vocab[index]["site"] != site {
            return Err("payload rows do not align with vocabulary rows".into());
        }
        let offset = index * ops.hidden() * 4;
        let mut values: Vec<f32> = payload[offset..offset + ops.hidden() * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if let Some(scale) = s["layer_scale"].as_f64() {
            backend.scale_row(&mut values, scale as f32);
        }
        let readout = ops.normalize_carrier_for_readout(&backend, &values)?;
        let v = unit(&readout.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())?;
        rows.push(json!({"site":site,"values":xy(&v,&axes),"cosines":directions.iter().map(|d|dot(&v,d).clamp(-1.,1.)).collect::<Vec<_>>() }));
    }
    let mut links = std::collections::BTreeMap::new();
    for (i, a) in directions.iter().enumerate() {
        let mut neighbors: Vec<_> = directions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, b)| (j, dot(a, b).clamp(-1., 1.)))
            .collect();
        neighbors.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (j, cosine) in neighbors.into_iter().take(2) {
            let pair = if i < j { (i, j) } else { (j, i) };
            links.insert(pair, cosine);
        }
    }
    let token_map = json!({
        "method":"Unit-normalized prepared final-norm carrier and unit-normalized stored output-head rows, projected into one fixed orthonormal plane. Geometry is directional; it is not probability or causal evidence.",
        "basis": { "id":"token-direction-plane-v1", "hash":format!("sha256:{basis_hash}"), "axes_hash":format!("sha256:{axes_hash}"), "axes":[format!("{} direction",args[5]),format!("{} orthogonal component",args[6])], "source":format!("sha256:{container_hash}"), "width":ops.hidden(), "dimensions":2, "normalization":"unit-l2", "construction":"Gram-Schmidt of the first two ordered landmark directions; no fitting to the trajectory", "axes_values":axes },
        "landmarks":ids.iter().zip(&args[5..]).zip(&directions).map(|((id,label),d)|json!({"token_id":id,"token":label,"values":xy(d,&axes)})).collect::<Vec<_>>(),
        "rows":rows,
        "graph_method":"Union of each selected landmark’s 2 nearest neighbors by full-dimensional unit-row cosine; selected landmarks only; similarity, not executed or causal connectivity",
        "edges":links.into_iter().map(|((a,b),cosine)|json!({"from":ids[a],"to":ids[b],"cosine":cosine})).collect::<Vec<_>>(),
        "producer": {"source_sha256":sha(include_bytes!("observatory_token_map.rs")),"carrier_sha256":sha(&payload),"inference_executions":0}
    });
    atomic_json(output, &extend_lens_json(&lens_bytes, &token_map)?)?;
    println!(
        "Token map: {} states, {} landmarks; no inference; {}",
        writes.len(),
        ids.len(),
        output.display()
    );
    Ok(())
}

/// Preserve all existing numeric lexemes: generic JSON parse/serialize can
/// round a recorded f64. Append one new field to the validated object instead.
fn extend_lens_json(bytes: &[u8], map: &Value) -> Result<Vec<u8>> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() || value.get("token_map").is_some() {
        return Err("expected an analysis object without an existing token map".into());
    }
    let raw = std::str::from_utf8(bytes)?.trim_end();
    let prefix = raw
        .strip_suffix('}')
        .ok_or("analysis object missing closing brace")?;
    let separator = if value.as_object().unwrap().is_empty() {
        ""
    } else {
        ","
    };
    Ok(format!(
        "{prefix}{separator}\n\"token_map\":{}\n}}\n",
        serde_json::to_string_pretty(map)?
    )
    .into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extending_analysis_preserves_original_numeric_bytes() {
        let bytes =
            br#"{"vocabulary":{"probability":0.00004693670247939259,"logit":18.11517906188965}}"#;
        let output = extend_lens_json(bytes, &json!({"rows":[]})).unwrap();
        assert!(output.starts_with(&bytes[..bytes.len() - 1]));
        assert!(serde_json::from_slice::<Value>(&output)
            .unwrap()
            .get("token_map")
            .is_some());
        assert!(extend_lens_json(&output, &json!({})).is_err());
    }

    #[test]
    fn shared_plane_preserves_direction_and_rejects_degeneracy() {
        let p = plane(&[1., 0., 0.], &[1., 1., 0.]).unwrap();
        assert_eq!(xy(&[1., 0., 0.], &p), [1., 0.]);
        assert!(dot(&p[0], &p[1]).abs() < 1e-12);
        assert_eq!(xy(&[0., 0., 1.], &p), [0., 0.]);
        assert!(plane(&[1., 0.], &[2., 0.]).is_err());
        assert!(unit(&[0., 0.]).is_err());
    }
}
