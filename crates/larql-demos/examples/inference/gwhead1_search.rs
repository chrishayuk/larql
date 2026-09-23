//! Frozen GW-HEAD-1 exhaustive train-only subset replay.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::head_replay::{
    replay_attention_heads, replay_attention_mixture,
};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PREREG_SCHEMA: &str = "larql.gwhead1.preregistration.v1";
const CAPTURE_SCHEMA: &str = "larql.gwhead1.natural-capture.v1";
const OUTPUT_SCHEMA: &str = "larql.gwhead1.train-subset-replay.v1";
const LAYER: usize = 24;
const ROWS: usize = 426;
const TRAIN_ROWS: usize = 255;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const CANDIDATES: usize = 126;
const SUBSETS: usize = 256;

struct TrainRow {
    original_row: usize,
    edge_id: String,
    relation: String,
    family: String,
    before: Vec<f32>,
    natural: Vec<Vec<f32>>,
    natural_norms: Vec<f64>,
}

fn artifact<'a>(manifest: &'a Value, name: &str) -> Result<&'a Value> {
    let matches: Vec<_> = manifest["artifacts"]
        .as_array()
        .ok_or("capture artifacts missing")?
        .iter()
        .filter(|item| item["path"].as_str() == Some(name))
        .collect();
    match matches.as_slice() {
        [item] => Ok(*item),
        _ => Err(format!("expected one capture artifact `{name}`").into()),
    }
}

fn read_f32(path: &Path, descriptor: &Value, values: usize) -> Result<Vec<f32>> {
    if descriptor["sha256"].as_str() != Some(&file_sha(path)?)
        || std::fs::metadata(path)?.len() != (values * 4) as u64
    {
        return Err(format!("invalid capture artifact {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let result: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if result.len() != values || result.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid f32 values in {}", path.display()).into());
    }
    Ok(result)
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().create_new(true).write(true).open(path)?,
    ))
}

fn write_f32(output: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn rel(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b.abs()
    } else {
        ((a - b) / a).abs()
    }
}

/// `--gwhead1-train-search CONTAINER PREREG CAPTURE_MANIFEST OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("Usage: observatory_record --gwhead1-train-search CONTAINER PREREG CAPTURE_MANIFEST OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let capture_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    std::fs::create_dir_all(output)?;
    let output_manifest = output.join("train-subset-replay-manifest.json");
    if output_manifest.exists() {
        return Err(format!(
            "GW-HEAD-1 train replay is already sealed at {}",
            output.display()
        )
        .into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    let capture: Value = serde_json::from_slice(&std::fs::read(capture_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA
        || prereg["status"] != "frozen_pre_execution"
        || capture["schema"] != CAPTURE_SCHEMA
        || capture["status"] != "natural_capture_complete_pre_subset_search"
        || capture["preregistration_sha256"] != prereg["preregistration_sha256"]
        || capture["reconstruction"]["result"] != "pass"
        || capture["parity"]["full_logit_bit_mismatches"] != 0
    {
        return Err("GW-HEAD-1 preregistration or natural capture is not admissible".into());
    }
    let prereg_identity = prereg["preregistration_sha256"]
        .as_str()
        .ok_or("GW-HEAD-1 identity missing")?;
    let capture_root = capture_path
        .parent()
        .ok_or("capture manifest has no parent")?;
    let heads_descriptor = artifact(&capture, "head-values.f32")?;
    let before_descriptor = artifact(&capture, "carrier-before.f32")?;
    let after_descriptor = artifact(&capture, "carrier-after.f32")?;
    let after_logits_descriptor = artifact(&capture, "candidate-after-logits.f32")?;
    let rows_descriptor = artifact(&capture, "natural-rows.jsonl")?;
    let all_heads = read_f32(
        &capture_root.join("head-values.f32"),
        heads_descriptor,
        ROWS * HEADS * HEAD_DIM,
    )?;
    let all_before = read_f32(
        &capture_root.join("carrier-before.f32"),
        before_descriptor,
        ROWS * HIDDEN,
    )?;
    let all_after = read_f32(
        &capture_root.join("carrier-after.f32"),
        after_descriptor,
        ROWS * HIDDEN,
    )?;
    let all_after_logits = read_f32(
        &capture_root.join("candidate-after-logits.f32"),
        after_logits_descriptor,
        ROWS * CANDIDATES,
    )?;
    let rows_path = capture_root.join("natural-rows.jsonl");
    if rows_descriptor["sha256"].as_str() != Some(&file_sha(&rows_path)?) {
        return Err("natural row hash mismatch".into());
    }
    let natural_rows: Vec<Value> = BufReader::new(File::open(&rows_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if natural_rows.len() != ROWS {
        return Err("natural capture row count changed".into());
    }

    eprintln!("Preparing the frozen GW-HEAD-1 replay image...");
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
        || capture["authorities"]["plan_sha256"].as_str() != Some(&plan_identity)
        || capture["authorities"]["container_identity"].as_str() != Some(&container_identity)
    {
        return Err("train replay image differs from frozen authorities".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let token_ids: Vec<u32> = capture["candidate_token_ids"]
        .as_array()
        .ok_or("candidate token IDs missing")?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .map(|v| v as u32)
                .ok_or("bad candidate token")
        })
        .collect::<std::result::Result<_, _>>()?;
    let selected = ops.select_output_head(&token_ids)?;

    let mut train = Vec::with_capacity(TRAIN_ROWS);
    for (original_row, row) in natural_rows.iter().enumerate() {
        if row["split"] != "train" {
            continue;
        }
        let natural: Vec<Vec<f32>> = (0..HEADS)
            .map(|head| {
                let start = (original_row * HEADS + head) * HEAD_DIM;
                all_heads[start..start + HEAD_DIM].to_vec()
            })
            .collect();
        let natural_norms: Vec<f64> = row["head_contribution_norms"]
            .as_array()
            .ok_or("head contribution norms missing")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or("bad head norm")
            })
            .collect::<std::result::Result<_, _>>()?;
        if natural_norms.len() != HEADS || natural_norms.iter().any(|norm| *norm <= 0.0) {
            return Err("natural head contribution norms are invalid".into());
        }
        train.push(TrainRow {
            original_row,
            edge_id: row["edge_id"]
                .as_str()
                .ok_or("edge ID missing")?
                .to_string(),
            relation: row["semantic_edge"]["relation"]
                .as_str()
                .ok_or("relation missing")?
                .to_string(),
            family: row["semantic_edge"]["prompt_semantic_family"]
                .as_str()
                .ok_or("prompt family missing")?
                .to_string(),
            before: all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN].to_vec(),
            natural,
            natural_norms,
        });
    }
    if train.len() != TRAIN_ROWS {
        return Err(format!("natural capture yielded {} train rows", train.len()).into());
    }

    eprintln!("Building leave-one-semantic-edge-out contribution-matched references...");
    let mut references = Vec::with_capacity(TRAIN_ROWS);
    let mut max_reference_norm_relative_error = 0.0f64;
    for (row_index, row) in train.iter().enumerate() {
        let donors: Vec<&TrainRow> = train
            .iter()
            .filter(|candidate| {
                candidate.relation == row.relation
                    && candidate.family == row.family
                    && candidate.edge_id != row.edge_id
            })
            .collect();
        if donors.is_empty() {
            return Err(
                format!("{} has no leave-one-edge-out reference donors", row.edge_id).into(),
            );
        }
        let mut reference = vec![vec![0.0f32; HEAD_DIM]; HEADS];
        for donor in &donors {
            for (mean_head, donor_head) in reference.iter_mut().zip(&donor.natural) {
                for (mean, value) in mean_head.iter_mut().zip(donor_head) {
                    *mean += *value / donors.len() as f32;
                }
            }
        }
        let unscaled =
            replay_attention_heads(&plan, &ops, &backend, LAYER, &row.before, &reference)?;
        for (head, reference_head) in reference.iter_mut().enumerate() {
            let norm = unscaled.contribution_norms[head];
            if !norm.is_finite() || norm <= 0.0 {
                return Err(
                    format!("{} head {head} has a zero reference direction", row.edge_id).into(),
                );
            }
            let scale = row.natural_norms[head] / norm;
            for value in reference_head {
                *value *= scale as f32;
            }
        }
        let scaled = replay_attention_heads(&plan, &ops, &backend, LAYER, &row.before, &reference)?;
        for head in 0..HEADS {
            max_reference_norm_relative_error = max_reference_norm_relative_error.max(rel(
                row.natural_norms[head],
                scaled.contribution_norms[head],
            ));
        }
        if max_reference_norm_relative_error > 1e-5 {
            return Err(format!("{} contribution-norm matching exceeded 1e-5", row.edge_id).into());
        }
        references.push(reference);
        if (row_index + 1).is_multiple_of(50) || row_index + 1 == TRAIN_ROWS {
            eprintln!("GW-HEAD-1 reference bank {}/{}", row_index + 1, TRAIN_ROWS);
        }
    }

    let logits_path = output.join("train-subset-logits.f32");
    let carriers_path = output.join("train-subset-carriers.f32");
    let references_path = output.join("train-reference-heads.f32");
    let rows_out_path = output.join("train-rows.jsonl");
    if [
        &logits_path,
        &carriers_path,
        &references_path,
        &rows_out_path,
    ]
    .iter()
    .any(|path| path.exists())
    {
        return Err("GW-HEAD-1 train replay output is partial or already exists".into());
    }
    let mut logits_out = writer(&logits_path)?;
    let mut carriers_out = writer(&carriers_path)?;
    let mut references_out = writer(&references_path)?;
    let mut rows_out = writer(&rows_out_path)?;
    for (row_index, (row, reference)) in train.iter().zip(&references).enumerate() {
        for head in reference {
            write_f32(&mut references_out, head)?;
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "train_row": row_index,
                "original_row": row.original_row,
                "edge_id": row.edge_id,
                "relation": row.relation,
                "prompt_semantic_family": row.family,
                "natural_contribution_norms": row.natural_norms,
            }),
        )?;
    }
    let mut full_carrier_bit_mismatches = 0usize;
    let mut full_logit_bit_mismatches = 0usize;
    for mask in 0usize..SUBSETS {
        for (row, reference) in train.iter().zip(&references) {
            let mixture: Vec<Vec<f32>> = (0..HEADS)
                .map(|head| {
                    if mask & (1 << head) != 0 {
                        row.natural[head].clone()
                    } else {
                        reference[head].clone()
                    }
                })
                .collect();
            let replay =
                replay_attention_mixture(&plan, &ops, &backend, LAYER, &row.before, &mixture)?;
            let logits =
                ops.readout_carrier_selected(&backend, &replay.carrier_after, &selected)?;
            if mask == SUBSETS - 1 {
                let natural_after =
                    &all_after[row.original_row * HIDDEN..(row.original_row + 1) * HIDDEN];
                let natural_logits = &all_after_logits
                    [row.original_row * CANDIDATES..(row.original_row + 1) * CANDIDATES];
                full_carrier_bit_mismatches += replay
                    .carrier_after
                    .iter()
                    .zip(natural_after)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                full_logit_bit_mismatches += logits
                    .iter()
                    .zip(natural_logits)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
            write_f32(&mut logits_out, &logits)?;
            write_f32(&mut carriers_out, &replay.carrier_after)?;
        }
        if (mask + 1).is_multiple_of(16) || mask + 1 == SUBSETS {
            eprintln!("GW-HEAD-1 train subsets {}/{}", mask + 1, SUBSETS);
        }
    }
    for output in [
        &mut logits_out,
        &mut carriers_out,
        &mut references_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop(logits_out);
    drop(carriers_out);
    drop(references_out);
    drop(rows_out);
    if full_carrier_bit_mismatches != 0 || full_logit_bit_mismatches != 0 {
        return Err("I_full replay did not reproduce the natural capture bit-for-bit".into());
    }

    let artifacts = [
        (
            &logits_path,
            "f32-le",
            json!([SUBSETS, TRAIN_ROWS, CANDIDATES]),
        ),
        (
            &carriers_path,
            "f32-le",
            json!([SUBSETS, TRAIN_ROWS, HIDDEN]),
        ),
        (
            &references_path,
            "f32-le",
            json!([TRAIN_ROWS, HEADS, HEAD_DIM]),
        ),
        (&rows_out_path, "jsonl", json!([TRAIN_ROWS])),
    ]
    .into_iter()
    .map(|(path, dtype, shape)| {
        Ok(json!({
            "path": path.file_name().and_then(|name| name.to_str()).ok_or("bad artifact path")?,
            "dtype": dtype,
            "shape": shape,
            "bytes": std::fs::metadata(path)?.len(),
            "sha256": file_sha(path)?,
        }))
    })
    .collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "complete_train_only",
        "split": "train",
        "held_out_rows": 0,
        "preregistration_sha256": prereg_identity,
        "natural_capture_sha256": file_sha(capture_path)?,
        "authorities": {
            "container_identity": container_identity,
            "plan_sha256": plan_identity,
            "backend": backend.name(),
            "execution_fingerprint": ExecutionProvenance::of(&ops).fingerprint(),
            "prepared_head_representation": selected.representation(),
        },
        "head_ids": (0..HEADS).collect::<Vec<_>>(),
        "subset_order": "integer mask 0..255; bit h set means head h natural/restored",
        "train_row_order": "ascending original frozen-input row after filtering split=train",
        "reference": {
            "grouping": "relation x prompt_semantic_family x head",
            "donor_exclusion": "same semantic edge",
            "scale": "effective-W_O contribution L2 matched to natural row/head",
            "max_relative_norm_error": max_reference_norm_relative_error,
            "threshold": 1e-5,
        },
        "full_replay_parity": {
            "carrier_bit_mismatches": full_carrier_bit_mismatches,
            "candidate_logit_bit_mismatches": full_logit_bit_mismatches,
        },
        "artifacts": artifacts,
        "validation_or_test_executed": false,
    });
    atomic_json(&output_manifest, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
