//! Direct train-only causal replay for frozen GW-READ-1 candidate components.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::head_replay::{
    replay_attention_mixture, replay_softmax_source_head,
};
use larql_vindex::format::vindex3::opplan::exec::{
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    observe::NoopObserver,
};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const PROTOCOL_SCHEMA: &str = "larql.gwread1.protocol.v2";
const CANDIDATE_SCHEMA: &str = "larql.gwread1.candidate-artifact.v1";
const OUTPUT_SCHEMA: &str = "larql.gwread1.train-candidate-replay.v1";
const LAYER: usize = 24;
const HEAD: usize = 1;
const HEAD_DIM: usize = 256;
const HEADS: usize = 8;
const HIDDEN: usize = 2560;
const ROWS: usize = 426;
const TRAIN_ROWS: usize = 255;
const SOURCE_ROWS: usize = 3792;
const SUBJECT_ROWS: usize = 480;
const CANDIDATE_TOKENS: usize = 126;
const Q_CANDIDATES: usize = 45;
const K_CANDIDATES: usize = 4;
const V_CANDIDATES: usize = 47;

fn artifact<'a>(manifest: &'a Value, name: &str) -> Result<&'a Value> {
    let matches: Vec<_> = manifest["artifacts"]
        .as_array()
        .ok_or("artifacts missing")?
        .iter()
        .filter(|item| item["path"].as_str() == Some(name))
        .collect();
    match matches.as_slice() {
        [item] => Ok(*item),
        _ => Err(format!("expected one artifact `{name}`").into()),
    }
}

fn read_f32(root: &Path, manifest: &Value, name: &str, values: usize) -> Result<Vec<f32>> {
    let descriptor = artifact(manifest, name)?;
    let path = root.join(name);
    if descriptor["sha256"].as_str() != Some(&file_sha(&path)?)
        || std::fs::metadata(&path)?.len() != (values * 4) as u64
    {
        return Err(format!("invalid artifact {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    File::open(&path)?.read_to_end(&mut bytes)?;
    let result: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect();
    if result.len() != values || result.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid f32 values in {}", path.display()).into());
    }
    Ok(result)
}

fn read_u8(root: &Path, manifest: &Value, name: &str, values: usize) -> Result<Vec<u8>> {
    let descriptor = artifact(manifest, name)?;
    let path = root.join(name);
    if descriptor["sha256"].as_str() != Some(&file_sha(&path)?)
        || std::fs::metadata(&path)?.len() != values as u64
    {
        return Err(format!("invalid artifact {}", path.display()).into());
    }
    let bytes = std::fs::read(path)?;
    if bytes.len() != values || bytes.iter().any(|&value| value > 1) {
        return Err("invalid availability artifact".into());
    }
    Ok(bytes)
}

fn jsonl(root: &Path, manifest: &Value, name: &str) -> Result<Vec<Value>> {
    let path = root.join(name);
    if artifact(manifest, name)?["sha256"].as_str() != Some(&file_sha(&path)?) {
        return Err(format!("artifact hash mismatch: {}", path.display()).into());
    }
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn resolve(root: &Path, descriptor: &Value) -> Result<std::path::PathBuf> {
    let path = Path::new(
        descriptor["path"]
            .as_str()
            .ok_or("authority path missing")?,
    );
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    })
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

fn vectors(all: &[f32], start: usize, rows: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|index| {
            let base = (start + index) * HEAD_DIM;
            all[base..base + HEAD_DIM].to_vec()
        })
        .collect()
}

fn candidate_vector(all: &[f32], candidate: usize, row: usize, rows: usize) -> &[f32] {
    let start = (candidate * rows + row) * HEAD_DIM;
    &all[start..start + HEAD_DIM]
}

#[allow(clippy::too_many_arguments)]
fn replay_logits(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    selected_head: &larql_vindex::format::vindex3::opplan::exec::prepared::SelectedOutputHead,
    op: &larql_vindex::format::vindex3::opplan::AttentionOp,
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    natural_heads: &[Vec<f32>],
    before: &[f32],
) -> Result<Vec<f32>> {
    let h1 = replay_softmax_source_head(query, keys, values, op.score_scale, op.logit_softcapping)?;
    let mut heads = natural_heads.to_vec();
    heads[HEAD] = h1.values;
    let replay = replay_attention_mixture(plan, ops, backend, LAYER, before, &heads)?;
    Ok(ops.readout_carrier_selected(backend, &replay.carrier_after, selected_head)?)
}

fn read_gwkey_row(file: &mut File, train_row: usize) -> Result<Vec<f32>> {
    let scalar_offset = ((2 * TRAIN_ROWS) + train_row) * CANDIDATE_TOKENS;
    file.seek(SeekFrom::Start((scalar_offset * 4) as u64))?;
    let mut bytes = vec![0u8; CANDIDATE_TOKENS * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn bits_differ(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn scaled(direction: &[f32], target: &[f32]) -> Result<Vec<f32>> {
    let direction_norm = norm(direction);
    let target_norm = norm(target);
    if direction_norm <= 0.0 || target_norm <= 0.0 {
        return Err("GW-READ-1 reference has zero norm".into());
    }
    Ok(direction
        .iter()
        .map(|value| *value * (target_norm / direction_norm) as f32)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn replay_carrier_logits(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    selected_head: &larql_vindex::format::vindex3::opplan::exec::prepared::SelectedOutputHead,
    op: &larql_vindex::format::vindex3::opplan::AttentionOp,
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    natural_heads: &[Vec<f32>],
    before: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let h1 = replay_softmax_source_head(query, keys, values, op.score_scale, op.logit_softcapping)?;
    let mut heads = natural_heads.to_vec();
    heads[HEAD] = h1.values;
    let replay = replay_attention_mixture(plan, ops, backend, LAYER, before, &heads)?;
    let logits = ops.readout_carrier_selected(backend, &replay.carrier_after, selected_head)?;
    Ok((replay.carrier_after, logits))
}

fn replacement_plan(position: usize, values: Vec<f32>) -> Result<InterventionPlan> {
    let provenance = VectorProvenance::Literal {
        sha256: vector_sha256(&values),
    };
    Ok(InterventionPlan::none().with(Intervention::replace(
        Address::new(LAYER, SublayerSite::Attention, [position])?,
        values,
        provenance,
    )?)?)
}

fn selected_candidate(selection: &Value, candidate: &Value, arm: &str) -> Result<usize> {
    let selected = selection["selected"][arm]
        .as_str()
        .ok_or("selected component missing")?;
    candidate["candidates"][arm]
        .as_array()
        .ok_or("candidate IDs missing")?
        .iter()
        .position(|value| value.as_str() == Some(selected))
        .ok_or_else(|| format!("selected {arm} candidate is absent").into())
}

fn output_artifact(path: &Path, shape: Value) -> Result<Value> {
    Ok(json!({
        "path": path.file_name().and_then(|value| value.to_str()).ok_or("bad path")?,
        "dtype": "f32-le",
        "shape": shape,
        "bytes": std::fs::metadata(path)?.len(),
        "sha256": file_sha(path)?,
    }))
}

/// `--gwread1-train-replay CONTAINER PROTOCOL CANDIDATES CAPTURE SOURCE_CAPTURE HEAD_CAPTURE GWKEY_TRAIN OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 8 {
        return Err("Usage: observatory_record --gwread1-train-replay CONTAINER PROTOCOL CANDIDATES CAPTURE SOURCE_CAPTURE HEAD_CAPTURE GWKEY_TRAIN OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let protocol_path = Path::new(&args[1]);
    let candidate_path = Path::new(&args[2]);
    let capture_path = Path::new(&args[3]);
    let source_path = Path::new(&args[4]);
    let head_path = Path::new(&args[5]);
    let gwkey_train_path = Path::new(&args[6]);
    let output = Path::new(&args[7]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("train-candidate-replay-manifest.json");
    if manifest_path.exists() {
        return Err("GW-READ-1 train candidate replay is already sealed".into());
    }

    let protocol: Value = serde_json::from_slice(&std::fs::read(protocol_path)?)?;
    let candidate: Value = serde_json::from_slice(&std::fs::read(candidate_path)?)?;
    let capture: Value = serde_json::from_slice(&std::fs::read(capture_path)?)?;
    let source: Value = serde_json::from_slice(&std::fs::read(source_path)?)?;
    let head: Value = serde_json::from_slice(&std::fs::read(head_path)?)?;
    let gwkey_train: Value = serde_json::from_slice(&std::fs::read(gwkey_train_path)?)?;
    if protocol["schema"] != PROTOCOL_SCHEMA
        || candidate["schema"] != CANDIDATE_SCHEMA
        || candidate["status"] != "train_only_candidates_fitted_pre_effect_execution"
        || capture["schema"] != "larql.gwread1.candidate-input-capture.v1"
        || source["schema"] != "larql.gwkey1.source-capture.v1"
        || head["schema"] != "larql.gwhead1.natural-capture.v1"
        || gwkey_train["schema"] != "larql.gwkey1.train-source-replay.v1"
        || candidate["protocol_sha256"] != protocol["protocol_sha256"]
        || capture["protocol"]["sha256"] != protocol["protocol_sha256"]
        || candidate["capture_file_sha256"] != file_sha(capture_path)?
        || candidate["source_capture_file_sha256"] != file_sha(source_path)?
        || source["gwhead1_natural_capture_sha256"] != file_sha(head_path)?
        || gwkey_train["natural_source_capture_sha256"] != file_sha(source_path)?
    {
        return Err("inadmissible GW-READ-1 train replay authorities".into());
    }
    if candidate["counts"]["Q"] != Q_CANDIDATES
        || candidate["counts"]["K"] != K_CANDIDATES
        || candidate["counts"]["V"] != V_CANDIDATES
    {
        return Err("GW-READ-1 candidate count changed".into());
    }

    let protocol_root = protocol_path.parent().ok_or("protocol has no parent")?;
    let input_path = resolve(protocol_root, &protocol["authorities"]["input_rows"])?;
    if file_sha(&input_path)? != protocol["authorities"]["input_rows"]["sha256"] {
        return Err("GW-READ-1 input rows changed".into());
    }
    let input_rows: Vec<Value> = BufReader::new(File::open(&input_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    let candidate_root = candidate_path
        .parent()
        .ok_or("candidate manifest has no parent")?;
    let capture_root = capture_path
        .parent()
        .ok_or("capture manifest has no parent")?;
    let source_root = source_path
        .parent()
        .ok_or("source manifest has no parent")?;
    let head_root = head_path.parent().ok_or("head manifest has no parent")?;
    let original_rows = jsonl(capture_root, &capture, "original-rows.jsonl")?;
    let source_rows = jsonl(source_root, &source, "source-rows.jsonl")?;
    let head_rows = jsonl(head_root, &head, "natural-rows.jsonl")?;
    if input_rows.len() != ROWS
        || original_rows.len() != ROWS
        || source_rows.len() != ROWS
        || head_rows.len() != ROWS
    {
        return Err("GW-READ-1 row count changed".into());
    }

    let all_q = read_f32(source_root, &source, "natural-q.f32", ROWS * HEAD_DIM)?;
    let all_v = read_f32(source_root, &source, "source-v.f32", SOURCE_ROWS * HEAD_DIM)?;
    let all_heads = read_f32(head_root, &head, "head-values.f32", ROWS * HEADS * HEAD_DIM)?;
    let all_before = read_f32(head_root, &head, "carrier-before.f32", ROWS * HIDDEN)?;
    let q_candidates = read_f32(
        candidate_root,
        &candidate,
        "q-candidates.f32",
        Q_CANDIDATES * ROWS * HEAD_DIM,
    )?;
    let k_candidates = read_f32(
        candidate_root,
        &candidate,
        "k-candidates.f32",
        K_CANDIDATES * SOURCE_ROWS * HEAD_DIM,
    )?;
    let v_candidates = read_f32(
        candidate_root,
        &candidate,
        "v-candidates.f32",
        V_CANDIDATES * SUBJECT_ROWS * HEAD_DIM,
    )?;
    let q_available = read_u8(
        candidate_root,
        &candidate,
        "q-availability.u8",
        Q_CANDIDATES * ROWS,
    )?;
    let k_available = read_u8(
        candidate_root,
        &candidate,
        "k-availability.u8",
        K_CANDIDATES * SOURCE_ROWS,
    )?;
    let v_available = read_u8(
        candidate_root,
        &candidate,
        "v-availability.u8",
        V_CANDIDATES * SUBJECT_ROWS,
    )?;
    let fixed_k = read_f32(
        candidate_root,
        &candidate,
        "fixed-static-k.f32",
        SOURCE_ROWS * HEAD_DIM,
    )?;
    let fixed_v = read_f32(
        candidate_root,
        &candidate,
        "fixed-static-v.f32",
        SOURCE_ROWS * HEAD_DIM,
    )?;
    let fixed_k_available = read_u8(
        candidate_root,
        &candidate,
        "fixed-static-k-availability.u8",
        SOURCE_ROWS,
    )?;
    let fixed_v_available = read_u8(
        candidate_root,
        &candidate,
        "fixed-static-v-availability.u8",
        SOURCE_ROWS,
    )?;

    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-READ-1 train replay plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if source["authorities"]["plan_sha256"] != plan_identity
        || head["authorities"]["plan_sha256"] != plan_identity
        || gwkey_train["authorities"]["plan_sha256"] != plan_identity
    {
        return Err("GW-READ-1 execution image changed".into());
    }
    let op = plan.layers[LAYER]
        .attention
        .softmax()
        .ok_or("L24 is not softmax attention")?;
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let token_ids: Vec<u32> = head["candidate_token_ids"]
        .as_array()
        .ok_or("candidate token IDs missing")?
        .iter()
        .map(|token| token.as_u64().map(|value| value as u32).ok_or("bad token"))
        .collect::<std::result::Result<_, _>>()?;
    if token_ids.len() != CANDIDATE_TOKENS {
        return Err("candidate token surface changed".into());
    }
    let selected_head = ops.select_output_head(&token_ids)?;
    let gwkey_root = gwkey_train_path
        .parent()
        .ok_or("GW-KEY train manifest has no parent")?;
    let train_reference_rows = artifact(&gwkey_train, "train-reference-k.f32")?["shape"][0]
        .as_u64()
        .ok_or("GW-KEY train reference row count missing")? as usize;
    let gwkey_reference_k = read_f32(
        gwkey_root,
        &gwkey_train,
        "train-reference-k.f32",
        train_reference_rows * HEAD_DIM,
    )?;
    let gwkey_reference_v = read_f32(
        gwkey_root,
        &gwkey_train,
        "train-reference-v.f32",
        train_reference_rows * HEAD_DIM,
    )?;
    let gwkey_logits_path = gwkey_root.join("train-joint-logits.f32");
    if artifact(&gwkey_train, "train-joint-logits.f32")?["sha256"] != file_sha(&gwkey_logits_path)?
    {
        return Err("GW-KEY train logits changed".into());
    }
    let mut gwkey_logits = File::open(gwkey_logits_path)?;

    let q_path = output.join("train-q-logits.f32");
    let k_path = output.join("train-k-logits.f32");
    let v_path = output.join("train-v-logits.f32");
    let rows_path = output.join("train-rows.jsonl");
    let mut q_out = writer(&q_path)?;
    let mut k_out = writer(&k_path)?;
    let mut v_out = writer(&v_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut exact_parity_mismatches = 0usize;
    let mut train_row = 0usize;
    let mut train_reference_offset = 0usize;
    let mut q_coverage = vec![0usize; Q_CANDIDATES];
    let mut k_coverage = vec![0usize; K_CANDIDATES];
    let mut v_coverage = vec![0usize; V_CANDIDATES];

    for original_row in 0..ROWS {
        if source_rows[original_row]["split"] != "train" {
            continue;
        }
        if original_rows[original_row]["edge_id"] != source_rows[original_row]["edge_id"]
            || original_rows[original_row]["edge_id"] != head_rows[original_row]["edge_id"]
            || original_rows[original_row]["edge_id"] != input_rows[original_row]["edge_id"]
        {
            return Err("GW-READ-1 row order changed".into());
        }
        let source_offset = source_rows[original_row]["source_offset"]
            .as_u64()
            .ok_or("source offset missing")? as usize;
        let source_count = source_rows[original_row]["source_count"]
            .as_u64()
            .ok_or("source count missing")? as usize;
        let subject_offset = original_rows[original_row]["subject_offset"]
            .as_u64()
            .ok_or("subject offset missing")? as usize;
        let subject_count = original_rows[original_row]["subject_count"]
            .as_u64()
            .ok_or("subject count missing")? as usize;
        let subject_positions: Vec<usize> = original_rows[original_row]["subject_positions"]
            .as_array()
            .ok_or("subject positions missing")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or("bad subject position")
            })
            .collect::<std::result::Result<_, _>>()?;
        if subject_positions.len() != subject_count {
            return Err("subject capture shape changed".into());
        }

        let query = &all_q[original_row * HEAD_DIM..(original_row + 1) * HEAD_DIM];
        let natural_values = vectors(&all_v, source_offset, source_count);
        let exact_keys = vectors(&gwkey_reference_k, train_reference_offset, source_count);
        let mut exact_values = vectors(&gwkey_reference_v, train_reference_offset, source_count);
        let fixed_keys = vectors(&fixed_k, source_offset, source_count);
        let mut fixed_values = vectors(&fixed_v, source_offset, source_count);
        for &position in &subject_positions {
            exact_values[position] = natural_values[position].clone();
            fixed_values[position] = natural_values[position].clone();
        }
        let natural_heads = vectors(&all_heads, original_row * HEADS, HEADS);
        let before = &all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN];
        let exact_logits = replay_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            query,
            &exact_keys,
            &exact_values,
            &natural_heads,
            before,
        )?;
        exact_parity_mismatches += bits_differ(
            &exact_logits,
            &read_gwkey_row(&mut gwkey_logits, train_row)?,
        );
        write_f32(&mut q_out, &exact_logits)?;
        for candidate_id in 0..Q_CANDIDATES {
            if q_available[candidate_id * ROWS + original_row] == 1 {
                q_coverage[candidate_id] += 1;
            }
            let candidate_query = candidate_vector(&q_candidates, candidate_id, original_row, ROWS);
            let logits = replay_logits(
                &plan,
                &ops,
                &backend,
                &selected_head,
                op,
                candidate_query,
                &exact_keys,
                &exact_values,
                &natural_heads,
                before,
            )?;
            write_f32(&mut q_out, &logits)?;
        }

        write_f32(&mut k_out, &exact_logits)?;
        for candidate_id in 0..K_CANDIDATES {
            let available = (0..source_count).all(|position| {
                k_available[candidate_id * SOURCE_ROWS + source_offset + position] == 1
            });
            if available {
                k_coverage[candidate_id] += 1;
            }
            let keys: Vec<Vec<f32>> = (0..source_count)
                .map(|position| {
                    candidate_vector(
                        &k_candidates,
                        candidate_id,
                        source_offset + position,
                        SOURCE_ROWS,
                    )
                    .to_vec()
                })
                .collect();
            let logits = replay_logits(
                &plan,
                &ops,
                &backend,
                &selected_head,
                op,
                query,
                &keys,
                &exact_values,
                &natural_heads,
                before,
            )?;
            write_f32(&mut k_out, &logits)?;
        }

        if !(0..source_count).all(|position| {
            fixed_k_available[source_offset + position] == 1
                && fixed_v_available[source_offset + position] == 1
        }) {
            return Err("fixed READ-1V K/V treatment is incomplete on train".into());
        }
        let v_exact_logits = replay_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            query,
            &fixed_keys,
            &fixed_values,
            &natural_heads,
            before,
        )?;
        write_f32(&mut v_out, &v_exact_logits)?;
        for candidate_id in 0..V_CANDIDATES {
            let available = (0..subject_count).all(|index| {
                v_available[candidate_id * SUBJECT_ROWS + subject_offset + index] == 1
            });
            if available {
                v_coverage[candidate_id] += 1;
            }
            let mut values = vectors(&fixed_v, source_offset, source_count);
            for (index, &position) in subject_positions.iter().enumerate() {
                values[position] = candidate_vector(
                    &v_candidates,
                    candidate_id,
                    subject_offset + index,
                    SUBJECT_ROWS,
                )
                .to_vec();
            }
            let logits = replay_logits(
                &plan,
                &ops,
                &backend,
                &selected_head,
                op,
                query,
                &fixed_keys,
                &values,
                &natural_heads,
                before,
            )?;
            write_f32(&mut v_out, &logits)?;
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "train_row": train_row,
                "original_row": original_row,
                "edge_id": source_rows[original_row]["edge_id"],
                "semantic_edge": input_rows[original_row]["edge_family_id"],
                "relation": input_rows[original_row]["semantic_edge"]["relation"],
                "prompt_semantic_family": input_rows[original_row]["semantic_edge"]["prompt_semantic_family"],
            }),
        )?;
        train_row += 1;
        train_reference_offset += source_count;
        if train_row.is_multiple_of(25) || train_row == TRAIN_ROWS {
            eprintln!("GW-READ-1 train candidate replay {train_row}/{TRAIN_ROWS}");
        }
    }
    if train_row != TRAIN_ROWS
        || train_reference_offset != train_reference_rows
        || exact_parity_mismatches != 0
    {
        return Err(format!(
            "GW-READ-1 train replay refused: rows={train_row}, exact parity mismatches={exact_parity_mismatches}"
        )
        .into());
    }
    for output in [&mut q_out, &mut k_out, &mut v_out, &mut rows_out] {
        output.flush()?;
    }
    drop((q_out, k_out, v_out, rows_out));
    let artifacts = vec![
        output_artifact(
            &q_path,
            json!([TRAIN_ROWS, Q_CANDIDATES + 1, CANDIDATE_TOKENS]),
        )?,
        output_artifact(
            &k_path,
            json!([TRAIN_ROWS, K_CANDIDATES + 1, CANDIDATE_TOKENS]),
        )?,
        output_artifact(
            &v_path,
            json!([TRAIN_ROWS, V_CANDIDATES + 1, CANDIDATE_TOKENS]),
        )?,
        json!({
            "path": rows_path.file_name().and_then(|value| value.to_str()).ok_or("bad path")?,
            "dtype": "jsonl",
            "shape": [TRAIN_ROWS],
            "bytes": std::fs::metadata(&rows_path)?.len(),
            "sha256": file_sha(&rows_path)?,
        }),
    ];
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "complete_train_only_pre_selection",
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_artifact_sha256": file_sha(candidate_path)?,
        "authorities": {
            "plan_sha256": plan_identity,
            "backend": backend.name(),
            "prepared_head_representation": selected_head.representation(),
            "gwkey_train_replay_sha256": file_sha(gwkey_train_path)?,
        },
        "arm_order": {
            "Q": {"first": "exact GW-KEY path", "candidates": candidate["candidates"]["Q"]},
            "K": {"first": "exact GW-KEY path", "candidates": candidate["candidates"]["K"]},
            "V": {"first": "natural subject V under fixed static K/complement V", "candidates": candidate["candidates"]["V"]},
        },
        "coverage_rows": {"Q": q_coverage, "K": k_coverage, "V": v_coverage},
        "parity": {"exact_gwkey_candidate_logit_bit_mismatches": exact_parity_mismatches},
        "validation_or_test_executed": false,
        "artifacts": artifacts,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

/// `--gwread1-heldout CONTAINER PROTOCOL SELECTION CANDIDATES CAPTURE SOURCE_CAPTURE HEAD_CAPTURE GWKEY_HELDOUT OUTPUT_DIR`
pub(super) fn run_heldout(args: &[String]) -> Result<()> {
    if args.len() != 9 {
        return Err("Usage: observatory_record --gwread1-heldout CONTAINER PROTOCOL SELECTION CANDIDATES CAPTURE SOURCE_CAPTURE HEAD_CAPTURE GWKEY_HELDOUT OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let protocol_path = Path::new(&args[1]);
    let selection_path = Path::new(&args[2]);
    let candidate_path = Path::new(&args[3]);
    let capture_path = Path::new(&args[4]);
    let source_path = Path::new(&args[5]);
    let head_path = Path::new(&args[6]);
    let gwkey_path = Path::new(&args[7]);
    let output = Path::new(&args[8]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("heldout-replay-manifest.json");
    if manifest_path.exists() {
        return Err("GW-READ-1 held-out replay is already sealed".into());
    }
    let protocol: Value = serde_json::from_slice(&std::fs::read(protocol_path)?)?;
    let selection: Value = serde_json::from_slice(&std::fs::read(selection_path)?)?;
    let candidate: Value = serde_json::from_slice(&std::fs::read(candidate_path)?)?;
    let capture: Value = serde_json::from_slice(&std::fs::read(capture_path)?)?;
    let source: Value = serde_json::from_slice(&std::fs::read(source_path)?)?;
    let head: Value = serde_json::from_slice(&std::fs::read(head_path)?)?;
    let gwkey: Value = serde_json::from_slice(&std::fs::read(gwkey_path)?)?;
    if protocol["schema"] != PROTOCOL_SCHEMA
        || selection["schema"] != "larql.gwread1.selection.v1"
        || selection["status"] != "frozen_pre_heldout"
        || selection["heldout_rows_seen"] != 0
        || selection["composition_tuned"] != false
        || candidate["schema"] != CANDIDATE_SCHEMA
        || capture["schema"] != "larql.gwread1.candidate-input-capture.v1"
        || source["schema"] != "larql.gwkey1.source-capture.v1"
        || head["schema"] != "larql.gwhead1.natural-capture.v1"
        || gwkey["schema"] != "larql.gwkey1.heldout-arms.v1"
        || selection["protocol_sha256"] != protocol["protocol_sha256"]
        || candidate["protocol_sha256"] != protocol["protocol_sha256"]
        || selection["authorities"]["candidate_artifact"]["sha256"] != file_sha(candidate_path)?
        || candidate["capture_file_sha256"] != file_sha(capture_path)?
        || candidate["source_capture_file_sha256"] != file_sha(source_path)?
        || source["gwhead1_natural_capture_sha256"] != file_sha(head_path)?
    {
        return Err("inadmissible GW-READ-1 held-out authorities".into());
    }
    let q_selected = selected_candidate(&selection, &candidate, "Q")?;
    let k_selected = selected_candidate(&selection, &candidate, "K")?;
    let v_selected = selected_candidate(&selection, &candidate, "V")?;

    let protocol_root = protocol_path.parent().ok_or("protocol has no parent")?;
    let input_path = resolve(protocol_root, &protocol["authorities"]["input_rows"])?;
    let roles_path = resolve(protocol_root, &protocol["authorities"]["source_roles"])?;
    if file_sha(&input_path)? != protocol["authorities"]["input_rows"]["sha256"]
        || file_sha(&roles_path)? != protocol["authorities"]["source_roles"]["sha256"]
    {
        return Err("GW-READ-1 held-out row authority changed".into());
    }
    let read_lines = |path: &Path| -> Result<Vec<Value>> {
        BufReader::new(File::open(path)?)
            .lines()
            .map(|line| Ok(serde_json::from_str(&line?)?))
            .collect()
    };
    let inputs = read_lines(&input_path)?;
    let roles = read_lines(&roles_path)?;
    let candidate_root = candidate_path
        .parent()
        .ok_or("candidate manifest has no parent")?;
    let capture_root = capture_path
        .parent()
        .ok_or("capture manifest has no parent")?;
    let source_root = source_path
        .parent()
        .ok_or("source manifest has no parent")?;
    let head_root = head_path.parent().ok_or("head manifest has no parent")?;
    let gwkey_root = gwkey_path.parent().ok_or("GW-KEY manifest has no parent")?;
    let original_rows = jsonl(capture_root, &capture, "original-rows.jsonl")?;
    let source_rows = jsonl(source_root, &source, "source-rows.jsonl")?;
    let head_rows = jsonl(head_root, &head, "natural-rows.jsonl")?;
    let gwkey_rows = jsonl(gwkey_root, &gwkey, "heldout-rows.jsonl")?;
    if [
        inputs.len(),
        roles.len(),
        original_rows.len(),
        source_rows.len(),
        head_rows.len(),
    ] != [ROWS; 5]
        || gwkey_rows.len() != 171
    {
        return Err("GW-READ-1 held-out row count changed".into());
    }
    let all_q = read_f32(source_root, &source, "natural-q.f32", ROWS * HEAD_DIM)?;
    let all_k = read_f32(source_root, &source, "source-k.f32", SOURCE_ROWS * HEAD_DIM)?;
    let all_v = read_f32(source_root, &source, "source-v.f32", SOURCE_ROWS * HEAD_DIM)?;
    let all_heads = read_f32(head_root, &head, "head-values.f32", ROWS * HEADS * HEAD_DIM)?;
    let all_before = read_f32(head_root, &head, "carrier-before.f32", ROWS * HIDDEN)?;
    let q_candidates = read_f32(
        candidate_root,
        &candidate,
        "q-candidates.f32",
        Q_CANDIDATES * ROWS * HEAD_DIM,
    )?;
    let k_candidates = read_f32(
        candidate_root,
        &candidate,
        "k-candidates.f32",
        K_CANDIDATES * SOURCE_ROWS * HEAD_DIM,
    )?;
    let v_candidates = read_f32(
        candidate_root,
        &candidate,
        "v-candidates.f32",
        V_CANDIDATES * SUBJECT_ROWS * HEAD_DIM,
    )?;
    let q_available = read_u8(
        candidate_root,
        &candidate,
        "q-availability.u8",
        Q_CANDIDATES * ROWS,
    )?;
    let k_available = read_u8(
        candidate_root,
        &candidate,
        "k-availability.u8",
        K_CANDIDATES * SOURCE_ROWS,
    )?;
    let v_available = read_u8(
        candidate_root,
        &candidate,
        "v-availability.u8",
        V_CANDIDATES * SUBJECT_ROWS,
    )?;
    let fixed_k = read_f32(
        candidate_root,
        &candidate,
        "fixed-static-k.f32",
        SOURCE_ROWS * HEAD_DIM,
    )?;
    let fixed_v = read_f32(
        candidate_root,
        &candidate,
        "fixed-static-v.f32",
        SOURCE_ROWS * HEAD_DIM,
    )?;
    let fixed_k_available = read_u8(
        candidate_root,
        &candidate,
        "fixed-static-k-availability.u8",
        SOURCE_ROWS,
    )?;
    let fixed_v_available = read_u8(
        candidate_root,
        &candidate,
        "fixed-static-v-availability.u8",
        SOURCE_ROWS,
    )?;
    let gwkey_proximal = read_f32(
        gwkey_root,
        &gwkey,
        "heldout-proximal-logits.f32",
        171 * 8 * CANDIDATE_TOKENS,
    )?;
    let gwkey_terminal = read_f32(
        gwkey_root,
        &gwkey,
        "heldout-terminal-logits.f32",
        171 * 8 * CANDIDATE_TOKENS,
    )?;

    let role_names = [
        "bos_system",
        "subject_entity",
        "relation_query",
        "instruction_template",
        "answer_cue",
        "punctuation_separator",
    ];
    let roles_for = |index: usize, count: usize| -> Result<Vec<usize>> {
        let mut result = vec![usize::MAX; count];
        for (role, name) in role_names.iter().enumerate() {
            for position in roles[index]["roles"][name]
                .as_array()
                .ok_or("roles missing")?
            {
                let position = position.as_u64().ok_or("bad role position")? as usize;
                if position >= count || result[position] != usize::MAX {
                    return Err("source roles overlap".into());
                }
                result[position] = role;
            }
        }
        if result.contains(&usize::MAX) {
            return Err("source roles are incomplete".into());
        }
        Ok(result)
    };
    type MeanCell = (Vec<f32>, Vec<f32>, usize);
    let mut means: BTreeMap<(String, usize), MeanCell> = BTreeMap::new();
    for index in 0..ROWS {
        if source_rows[index]["split"] != "train" {
            continue;
        }
        let offset = source_rows[index]["source_offset"]
            .as_u64()
            .ok_or("offset missing")? as usize;
        let count = source_rows[index]["source_count"]
            .as_u64()
            .ok_or("count missing")? as usize;
        let source_roles = roles_for(index, count)?;
        let template = roles[index]["template_id"]
            .as_str()
            .ok_or("template missing")?
            .to_owned();
        for (position, role) in source_roles.into_iter().enumerate() {
            let cell = means
                .entry((template.clone(), role))
                .or_insert_with(|| (vec![0.0; HEAD_DIM], vec![0.0; HEAD_DIM], 0));
            cell.2 += 1;
            let start = (offset + position) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                cell.0[dimension] += all_k[start + dimension];
                cell.1[dimension] += all_v[start + dimension];
            }
        }
    }
    for (keys, values, count) in means.values_mut() {
        for dimension in 0..HEAD_DIM {
            keys[dimension] /= *count as f32;
            values[dimension] /= *count as f32;
        }
    }

    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-READ-1 held-out plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if source["authorities"]["plan_sha256"] != plan_identity
        || head["authorities"]["plan_sha256"] != plan_identity
    {
        return Err("GW-READ-1 held-out execution image changed".into());
    }
    let op = plan.layers[LAYER]
        .attention
        .softmax()
        .ok_or("L24 is not softmax")?;
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let token_ids: Vec<u32> = head["candidate_token_ids"]
        .as_array()
        .ok_or("candidate tokens missing")?
        .iter()
        .map(|v| v.as_u64().map(|x| x as u32).ok_or("bad token"))
        .collect::<std::result::Result<_, _>>()?;
    let selected_head = ops.select_output_head(&token_ids)?;

    let q_proximal_path = output.join("heldout-q-proximal.f32");
    let k_proximal_path = output.join("heldout-k-proximal.f32");
    let v_proximal_path = output.join("heldout-v-proximal.f32");
    let e_proximal_path = output.join("heldout-e-proximal.f32");
    let terminal_path = output.join("heldout-terminal.f32");
    let rows_path = output.join("heldout-rows.jsonl");
    let mut q_out = writer(&q_proximal_path)?;
    let mut k_out = writer(&k_proximal_path)?;
    let mut v_out = writer(&v_proximal_path)?;
    let mut e_out = writer(&e_proximal_path)?;
    let mut terminal_out = writer(&terminal_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut heldout_row = 0usize;
    let mut exact_proximal_mismatches = 0usize;
    let mut exact_terminal_mismatches = 0usize;
    let mut intervention_firings = 0usize;
    for original_row in 0..ROWS {
        if source_rows[original_row]["split"] == "train" {
            continue;
        }
        if gwkey_rows[heldout_row]["original_row"] != original_row
            || source_rows[original_row]["edge_id"] != original_rows[original_row]["edge_id"]
        {
            return Err("GW-READ-1 held-out row order changed".into());
        }
        let offset = source_rows[original_row]["source_offset"]
            .as_u64()
            .ok_or("offset missing")? as usize;
        let count = source_rows[original_row]["source_count"]
            .as_u64()
            .ok_or("count missing")? as usize;
        let subject_offset = original_rows[original_row]["subject_offset"]
            .as_u64()
            .ok_or("subject offset missing")? as usize;
        let subject_positions: Vec<usize> = original_rows[original_row]["subject_positions"]
            .as_array()
            .ok_or("subject positions missing")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize).ok_or("bad subject position"))
            .collect::<std::result::Result<_, _>>()?;
        let source_roles = roles_for(original_row, count)?;
        let template = roles[original_row]["template_id"]
            .as_str()
            .ok_or("template missing")?;
        let natural_keys = vectors(&all_k, offset, count);
        let natural_values = vectors(&all_v, offset, count);
        let mut exact_keys = Vec::with_capacity(count);
        let mut exact_values = Vec::with_capacity(count);
        for (position, role) in source_roles.iter().copied().enumerate() {
            let cell = means
                .get(&(template.to_owned(), role))
                .ok_or("missing exact reference cell")?;
            exact_keys.push(scaled(&cell.0, &natural_keys[position])?);
            exact_values.push(scaled(&cell.1, &natural_values[position])?);
        }
        for &position in &subject_positions {
            exact_values[position] = natural_values[position].clone();
        }
        if q_available[q_selected * ROWS + original_row] == 0
            || !(0..count).all(|position| {
                k_available[k_selected * SOURCE_ROWS + offset + position] == 1
                    && fixed_k_available[offset + position] == 1
                    && fixed_v_available[offset + position] == 1
            })
            || !(0..subject_positions.len())
                .all(|index| v_available[v_selected * SUBJECT_ROWS + subject_offset + index] == 1)
        {
            return Err("selected held-out candidate has missing input".into());
        }
        let natural_query = &all_q[original_row * HEAD_DIM..(original_row + 1) * HEAD_DIM];
        let candidate_query = candidate_vector(&q_candidates, q_selected, original_row, ROWS);
        let candidate_keys: Vec<Vec<f32>> = (0..count)
            .map(|position| {
                candidate_vector(&k_candidates, k_selected, offset + position, SOURCE_ROWS).to_vec()
            })
            .collect();
        let fixed_keys = vectors(&fixed_k, offset, count);
        let mut v_exact_values = vectors(&fixed_v, offset, count);
        let mut v_candidate_values = v_exact_values.clone();
        for (index, &position) in subject_positions.iter().enumerate() {
            v_exact_values[position] = natural_values[position].clone();
            v_candidate_values[position] = candidate_vector(
                &v_candidates,
                v_selected,
                subject_offset + index,
                SUBJECT_ROWS,
            )
            .to_vec();
        }
        let natural_heads = vectors(&all_heads, original_row * HEADS, HEADS);
        let before = &all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN];
        let exact = replay_carrier_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            natural_query,
            &exact_keys,
            &exact_values,
            &natural_heads,
            before,
        )?;
        let q_candidate = replay_carrier_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            candidate_query,
            &exact_keys,
            &exact_values,
            &natural_heads,
            before,
        )?;
        let k_candidate = replay_carrier_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            natural_query,
            &candidate_keys,
            &exact_values,
            &natural_heads,
            before,
        )?;
        let v_exact = replay_carrier_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            natural_query,
            &fixed_keys,
            &v_exact_values,
            &natural_heads,
            before,
        )?;
        let v_candidate = replay_carrier_logits(
            &plan,
            &ops,
            &backend,
            &selected_head,
            op,
            natural_query,
            &fixed_keys,
            &v_candidate_values,
            &natural_heads,
            before,
        )?;
        write_f32(&mut q_out, &exact.1)?;
        write_f32(&mut q_out, &q_candidate.1)?;
        write_f32(&mut k_out, &exact.1)?;
        write_f32(&mut k_out, &k_candidate.1)?;
        write_f32(&mut v_out, &v_exact.1)?;
        write_f32(&mut v_out, &v_candidate.1)?;
        let mut e_cells = Vec::with_capacity(8);
        for mask in 0..8usize {
            let query = if mask & 4 != 0 {
                candidate_query
            } else {
                natural_query
            };
            let keys = if mask & 2 != 0 {
                &candidate_keys
            } else {
                &exact_keys
            };
            let values = if mask & 1 != 0 {
                &v_candidate_values
            } else {
                &v_exact_values
            };
            let cell = replay_carrier_logits(
                &plan,
                &ops,
                &backend,
                &selected_head,
                op,
                query,
                keys,
                values,
                &natural_heads,
                before,
            )?;
            write_f32(&mut e_out, &cell.1)?;
            e_cells.push(cell);
        }
        let gwkey_start = (heldout_row * 8 + 6) * CANDIDATE_TOKENS;
        exact_proximal_mismatches += bits_differ(
            &exact.1,
            &gwkey_proximal[gwkey_start..gwkey_start + CANDIDATE_TOKENS],
        );

        let tokens: Vec<u32> = inputs[original_row]["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt tokens missing")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as u32).ok_or("bad prompt token"))
            .collect::<std::result::Result<_, _>>()?;
        let position = tokens.len() - 1;
        let mut prefix = RowKvState::default();
        {
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut prefix)?;
            for &token in &tokens[..position] {
                session.step_observed(token, &mut NoopObserver)?;
            }
        }
        let terminal_carriers = [
            &exact.0,
            &q_candidate.0,
            &exact.0,
            &k_candidate.0,
            &v_exact.0,
            &v_candidate.0,
            &e_cells[0].0,
            &e_cells[7].0,
        ];
        for (arm, carrier) in terminal_carriers.iter().enumerate() {
            let mut kv = prefix.clone();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            let intervention = replacement_plan(position, (*carrier).clone())?;
            let result = session.step_intervened(
                tokens[position],
                &mut NoopObserver,
                &intervention,
                &HeadInterventionPlan::none(),
            )?;
            if result.firings.len() != 1 {
                return Err("held-out intervention did not fire exactly once".into());
            }
            intervention_firings += 1;
            let full = result.logits.ok_or("terminal logits missing")?;
            let selected: Vec<f32> = token_ids
                .iter()
                .map(|&token| full[token as usize])
                .collect();
            if arm == 0 {
                let start = (heldout_row * 8 + 6) * CANDIDATE_TOKENS;
                exact_terminal_mismatches +=
                    bits_differ(&selected, &gwkey_terminal[start..start + CANDIDATE_TOKENS]);
            }
            write_f32(&mut terminal_out, &selected)?;
        }
        write_json_line(
            &mut rows_out,
            &json!({"heldout_row":heldout_row,"original_row":original_row,"edge_id":source_rows[original_row]["edge_id"],"split":source_rows[original_row]["split"],"relation":inputs[original_row]["semantic_edge"]["relation"],"component_terminal_arms":["Q_exact","Q_candidate","K_exact","K_candidate","V_exact","V_candidate","E_exact","E_all_candidate"],"E_factorial_order":"mask 0..7; Q candidate bit 4, K bit 2, V bit 1"}),
        )?;
        heldout_row += 1;
        if heldout_row.is_multiple_of(25) || heldout_row == 171 {
            eprintln!("GW-READ-1 held-out replay {heldout_row}/171");
        }
    }
    if heldout_row != 171 || exact_proximal_mismatches != 0 || exact_terminal_mismatches != 0 {
        return Err(format!("held-out parity refused: rows={heldout_row}, proximal={exact_proximal_mismatches}, terminal={exact_terminal_mismatches}").into());
    }
    for output in [
        &mut q_out,
        &mut k_out,
        &mut v_out,
        &mut e_out,
        &mut terminal_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop((q_out, k_out, v_out, e_out, terminal_out, rows_out));
    let artifacts = vec![
        output_artifact(&q_proximal_path, json!([171, 2, CANDIDATE_TOKENS]))?,
        output_artifact(&k_proximal_path, json!([171, 2, CANDIDATE_TOKENS]))?,
        output_artifact(&v_proximal_path, json!([171, 2, CANDIDATE_TOKENS]))?,
        output_artifact(&e_proximal_path, json!([171, 8, CANDIDATE_TOKENS]))?,
        output_artifact(&terminal_path, json!([171, 8, CANDIDATE_TOKENS]))?,
        json!({"path":"heldout-rows.jsonl","dtype":"jsonl","shape":[171],"bytes":std::fs::metadata(&rows_path)?.len(),"sha256":file_sha(&rows_path)?}),
    ];
    let manifest = json!({
        "schema":"larql.gwread1.heldout-replay.v1","status":"complete_frozen_validation_and_test",
        "protocol_sha256":protocol["protocol_sha256"],"selection_sha256":selection["selection_sha256"],"selection_file_sha256":file_sha(selection_path)?,"candidate_artifact_sha256":file_sha(candidate_path)?,
        "selected":selection["selected"],"splits":{"validation":87,"test":84},
        "parity":{"gwkey_exact_proximal_bit_mismatches":exact_proximal_mismatches,"gwkey_exact_terminal_bit_mismatches":exact_terminal_mismatches},
        "intervention_firings":intervention_firings,
        "natural_context_inputs":{"carrier_before":true,"non_H1_heads":7,"requires_full_pre_L24_execution":true,"charged_dynamic_input":true,"READ_1E_no_hidden_full_execution_gate":false},
        "factorial":{"order":"mask 0..7; Q candidate bit 4, K bit 2, V bit 1","selection_or_tuning_performed":false,"verdict_cell":7},
        "authorities":{"plan_sha256":plan_identity,"backend":backend.name(),"prepared_head_representation":selected_head.representation(),"gwkey_heldout_sha256":file_sha(gwkey_path)?},
        "artifacts":artifacts,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
