//! Frozen GW-SUP-1 candidate readout over sealed GW-0 carrier artifacts.
use super::*;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};

const PREREG_SCHEMA: &str = "larql.gwsup1.preregistration.v1";
const CANDIDATE_SCHEMA: &str = "larql.gwsup1.candidates.v1";
const OUTPUT_SCHEMA: &str = "larql.gwsup1.readout-manifest.v1";

pub(super) fn file_sha(path: &Path) -> Result<String> {
    Ok(format!("sha256:{}", sha(&std::fs::read(path)?)))
}

pub(super) fn artifact<'a>(row: &'a Value, kind: &str) -> Result<&'a Value> {
    row["artifacts"]
        .as_array()
        .ok_or("sealed row has no artifact list")?
        .iter()
        .find(|item| item["kind"] == kind)
        .ok_or_else(|| format!("{} has no {kind} artifact", row["edge_id"]).into())
}

pub(super) fn checked_bytes(root: &Path, descriptor: &Value) -> Result<Vec<u8>> {
    let path = root.join(descriptor["path"].as_str().ok_or("artifact path missing")?);
    let bytes = std::fs::read(&path)?;
    let actual = format!("sha256:{}", sha(&bytes));
    if descriptor["sha256"].as_str() != Some(&actual) {
        return Err(format!("sealed artifact hash mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

pub(super) fn f32_rows(bytes: &[u8], rows: usize, width: usize) -> Result<Vec<f32>> {
    if bytes.len() != rows * width * 4 {
        return Err(format!(
            "carrier byte count {} does not describe [{rows}, {width}] f32",
            bytes.len()
        )
        .into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("sealed carrier contains a non-finite value".into());
    }
    Ok(values)
}

pub(super) fn diagnostic(values: &[f32]) -> Value {
    let count = values.len() as f64;
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / count;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    json!({
        "mean": mean,
        "std": variance.sqrt(),
        "range": f64::from(max) - f64::from(min),
    })
}

pub(super) fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(super) fn write_json_line(writer: &mut BufWriter<File>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// `--gwsup1-readout CONTAINER PREREG CANDIDATES SEALED_MANIFEST OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("Usage: observatory_record --gwsup1-readout CONTAINER PREREG CANDIDATES SEALED_MANIFEST OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let candidates_path = Path::new(&args[2]);
    let sealed_path = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    std::fs::create_dir_all(output)?;
    let output_manifest = output.join("readout-manifest.json");
    if output_manifest.exists() {
        return Err(format!("GW-SUP-1 output is already sealed at {}", output.display()).into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA || prereg["status"] != "frozen_pre_readout" {
        return Err("GW-SUP-1 preregistration is not frozen".into());
    }
    let prereg_identity = prereg["preregistration_sha256"]
        .as_str()
        .ok_or("preregistration identity missing")?;
    let candidates: Value = serde_json::from_slice(&std::fs::read(candidates_path)?)?;
    if candidates["schema"] != CANDIDATE_SCHEMA
        || prereg["authorities"]["candidates"]["file_sha256"].as_str()
            != Some(&file_sha(candidates_path)?)
        || prereg["authorities"]["candidates"]["candidate_identity_sha256"]
            != candidates["candidate_identity_sha256"]
    {
        return Err("candidate authority does not match the frozen preregistration".into());
    }
    let token_ids: Vec<u32> = candidates["decoder_vocabulary"]
        .as_array()
        .ok_or("candidate vocabulary missing")?
        .iter()
        .map(|item| {
            item["token_id"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("invalid candidate token ID")
        })
        .collect::<std::result::Result<_, _>>()?;
    if token_ids.len() != 126 {
        return Err("frozen candidate vocabulary no longer has 126 tokens".into());
    }

    let sealed: Value = serde_json::from_slice(&std::fs::read(sealed_path)?)?;
    if prereg["authorities"]["sealed_gw0"]["file_sha256"].as_str() != Some(&file_sha(sealed_path)?)
        || prereg["authorities"]["sealed_gw0"]["bundle_sha256"] != sealed["bundle_sha256"]
        || prereg["authorities"]["sealed_gw0"]["census_sha256"] != sealed["census"]["sha256"]
    {
        return Err("sealed GW-0 authority does not match the preregistration".into());
    }
    let sealed_dir = sealed_path
        .parent()
        .ok_or("sealed manifest has no parent")?;
    let census_path = sealed_dir.join(
        sealed["census"]["path"]
            .as_str()
            .ok_or("sealed census path missing")?,
    );
    if file_sha(&census_path)? != sealed["census"]["sha256"] {
        return Err("sealed census hash mismatch".into());
    }
    let artifact_root = sealed_dir.join(
        sealed["artifact_root"]
            .as_str()
            .ok_or("sealed artifact root missing")?,
    );

    eprintln!("Preparing the frozen production image and selected output head...");
    let inspection = inspect_container(container, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("closed target plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    let container_identity = format!(
        "sha256:{}",
        sha(&serde_json::to_vec(
            &json!({"index": inspection.index, "graph": inspection.graph})
        )?)
    );
    if prereg["authorities"]["plan_sha256"].as_str() != Some(&plan_identity)
        || prereg["authorities"]["container_identity"].as_str() != Some(&container_identity)
    {
        return Err(
            "prepared container or operation plan differs from the frozen authority".into(),
        );
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let selected = ops.select_output_head(&token_ids)?;
    if selected.token_ids() != token_ids {
        return Err("selected head changed candidate order".into());
    }

    let logits_path = output.join("candidate-logits.f32");
    let rows_path = output.join("rows.jsonl");
    let mut logits_writer = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&logits_path)?,
    );
    let mut rows_writer = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&rows_path)?,
    );
    let census_reader = BufReader::new(File::open(&census_path)?);
    let mut row_count = 0usize;
    let mut parity: Option<Value> = None;
    for line in census_reader.lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let edge_id = row["edge_id"].as_str().ok_or("sealed edge ID missing")?;
        let record_bytes = checked_bytes(&artifact_root, artifact(&row, "execution_record")?)?;
        let record: Value = serde_json::from_slice(&record_bytes)?;
        if record["edge_id"] != row["edge_id"]
            || record["identity"]["container"].as_str() != Some(&container_identity)
            || record["identity"]["plan"].as_str() != Some(&plan_identity)
        {
            return Err(format!("{edge_id}: sealed execution identity mismatch").into());
        }
        let sites = record["sites"]
            .as_array()
            .ok_or("execution sites missing")?;
        if sites.len() != 68 {
            return Err(format!("{edge_id}: expected 68 sites, got {}", sites.len()).into());
        }
        let carrier_descriptor = artifact(&row, "carrier-after")?;
        if carrier_descriptor["shape"] != json!([68, ops.hidden()]) {
            return Err(format!("{edge_id}: carrier shape changed").into());
        }
        let carriers = f32_rows(
            &checked_bytes(&artifact_root, carrier_descriptor)?,
            68,
            ops.hidden(),
        )?;
        let mut site_rows = Vec::with_capacity(68);
        for (site_index, (site, carrier)) in sites
            .iter()
            .zip(carriers.chunks_exact(ops.hidden()))
            .enumerate()
        {
            let mut state = carrier.to_vec();
            let scale = site["layer_scale"].as_f64().map(|value| value as f32);
            if let Some(scale) = scale {
                backend.scale_row(&mut state, scale);
            }
            let logits = ops.readout_carrier_selected(&backend, &state, &selected)?;
            if logits.len() != token_ids.len() || logits.iter().any(|value| !value.is_finite()) {
                return Err(
                    format!("{edge_id}: invalid candidate readout at site {site_index}").into(),
                );
            }
            if parity.is_none() {
                let full = ops.readout_carrier(&backend, &state)?;
                let mut mismatch = 0usize;
                let mut max_abs = 0.0f64;
                for (&token, &candidate) in token_ids.iter().zip(&logits) {
                    let expected = full[token as usize];
                    mismatch += usize::from(expected.to_bits() != candidate.to_bits());
                    max_abs = max_abs.max(f64::from((expected - candidate).abs()));
                }
                parity = Some(json!({
                    "edge_id": edge_id,
                    "site_index": site_index,
                    "selected_tokens": token_ids.len(),
                    "bit_mismatches": mismatch,
                    "max_abs_difference": max_abs,
                }));
            }
            for value in &logits {
                logits_writer.write_all(&value.to_le_bytes())?;
            }
            site_rows.push(json!({
                "index": site_index,
                "layer": site["layer"],
                "site": site["site"],
                "layer_scale": scale,
                "carrier_l2": l2(&state),
                "candidate_logits": diagnostic(&logits),
            }));
        }
        write_json_line(
            &mut rows_writer,
            &json!({
                "edge_id": edge_id,
                "row_index": row_count,
                "emergence": row["emergence"],
                "semantic_edge": row["semantic_edge"],
                "split": row["split"],
                "sites": site_rows,
            }),
        )?;
        row_count += 1;
        if row_count.is_multiple_of(25) || row_count == 426 {
            eprintln!("GW-SUP-1 readout {row_count}/426");
        }
    }
    if row_count != 426 {
        return Err(format!("sealed census yielded {row_count} rows, expected 426").into());
    }
    logits_writer.flush()?;
    rows_writer.flush()?;
    drop(logits_writer);
    drop(rows_writer);
    let expected_logits_bytes = row_count * 68 * token_ids.len() * 4;
    if std::fs::metadata(&logits_path)?.len() != expected_logits_bytes as u64 {
        return Err("candidate logit artifact has the wrong byte length".into());
    }
    let parity = parity.ok_or("no parity witness was produced")?;
    if parity["bit_mismatches"] != 0 || parity["max_abs_difference"] != 0.0 {
        return Err(format!("selected/full head parity failed: {parity}").into());
    }
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "readout_complete_pre_adjudication",
        "preregistration_sha256": prereg_identity,
        "candidate_identity_sha256": candidates["candidate_identity_sha256"],
        "authorities": {
            "container_identity": container_identity,
            "plan_sha256": plan_identity,
            "sealed_bundle_sha256": sealed["bundle_sha256"],
            "sealed_census_sha256": sealed["census"]["sha256"],
            "prepared_head_representation": selected.representation(),
            "backend": backend.name(),
            "execution_fingerprint": ExecutionProvenance::of(&ops).fingerprint(),
        },
        "shape": [row_count, 68, token_ids.len()],
        "order": "sealed census row; layer ascending; attention then ffn; frozen candidate token order",
        "candidate_token_ids": token_ids,
        "artifacts": {
            "logits": {"path": "candidate-logits.f32", "dtype": "f32-le", "bytes": expected_logits_bytes, "sha256": file_sha(&logits_path)?},
            "rows": {"path": "rows.jsonl", "rows": row_count, "sha256": file_sha(&rows_path)?},
        },
        "selected_full_head_parity": parity,
        "prompts_reexecuted": false,
        "individual_examples_inspected_before_adjudication": false,
    });
    atomic_json(&output_manifest, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
