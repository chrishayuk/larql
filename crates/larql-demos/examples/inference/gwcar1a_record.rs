//! CAR-1A paired carrier-replacement replay, isolated from the sealed STATE-1 binary.
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use larql_inference::vindex3::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    head_replay::replay_attention_mixture,
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    kv::RowKvState,
    observe::NoopObserver,
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands, SelectedOutputHead},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const ROWS: usize = 666;
const CANDIDATES: usize = 19;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const WIDTH: usize = 142;

#[derive(Default)]
struct TailCarrier(Vec<f32>);
impl StepObserver for TailCarrier {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.0 = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for value in &mut self.0 {
                *value *= scale;
            }
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}
fn file_sha(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = [0_u8; 1024 * 1024];
    loop {
        let size = file.read(&mut block)?;
        if size == 0 {
            break;
        }
        digest.update(&block[..size]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}
fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn usize_at(value: &Value) -> Result<usize> {
    Ok(value.as_u64().ok_or("missing unsigned integer")? as usize)
}
fn ids(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or("missing ID array")?
        .iter()
        .map(usize_at)
        .collect()
}
fn checked(root: &Path, manifest: &Value, name: &str, count: usize) -> Result<Vec<f32>> {
    let entries: Vec<_> = manifest["artifacts"]
        .as_array()
        .ok_or("artifacts absent")?
        .iter()
        .filter(|entry| entry["path"] == name)
        .collect();
    let path = root.join(name);
    if entries.len() != 1 || entries[0]["sha256"] != file_sha(&path)? {
        return Err(format!("CAR-1A artifact hash mismatch: {name}").into());
    }
    let bytes = std::fs::read(path)?;
    if bytes.len() != count * 4 {
        return Err(format!("CAR-1A artifact size mismatch: {name}").into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("nonfinite CAR-1A artifact: {name}").into());
    }
    Ok(values)
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn head_values(data: &[f32], row: usize) -> Vec<Vec<f32>> {
    (0..HEADS)
        .map(|head| {
            let start = (row * HEADS + head) * HEAD_DIM;
            data[start..start + HEAD_DIM].to_vec()
        })
        .collect()
}
fn paired_heads(arm: usize, natural: &[Vec<f32>], donor: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if arm > 1 || natural.len() != HEADS || donor.len() != HEADS {
        return Err("invalid CAR-1A head geometry".into());
    }
    Ok((0..HEADS)
        .map(|head| {
            if head == 1 && arm == 1 {
                natural[head].clone()
            } else {
                donor[head].clone()
            }
        })
        .collect())
}
fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn put(out: &mut impl Write, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("nonfinite CAR-1A replay output".into());
    }
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
fn artifact(root: &Path, name: &str, rows: usize) -> Result<Value> {
    let path = root.join(name);
    Ok(
        json!({"path":name, "shape":[rows,CANDIDATES,2,WIDTH], "dtype":"f32-le",
        "bytes":std::fs::metadata(&path)?.len(), "sha256":file_sha(&path)?}),
    )
}

#[allow(clippy::too_many_arguments)]
fn terminal(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tail_ops: &PreparedOperands,
    backend: &ProductionBackend,
    prefix: &RowKvState,
    natural_carrier: &[f32],
    target: usize,
    replacement: Vec<f32>,
    selected: &SelectedOutputHead,
) -> Result<Vec<f32>> {
    let intervention = InterventionPlan::none().with(Intervention::replace(
        Address::new(24, SublayerSite::Attention, [target])?,
        replacement.clone(),
        VectorProvenance::Literal {
            sha256: vector_sha256(&replacement),
        },
    )?)?;
    let mut kv = prefix.clone();
    let mut session = DecodeSession::over_prepared(plan, tail_ops, backend, &mut kv)?;
    let mut observer = TailCarrier::default();
    let result =
        session.step_from_carrier_intervened(natural_carrier, &mut observer, &intervention)?;
    if result.firings.len() != 1 {
        return Err("CAR-1A attention-write intervention did not fire once".into());
    }
    Ok(ops.readout_carrier_selected(backend, &observer.0, selected)?)
}

fn run(args: &[String]) -> Result<()> {
    if args.len() < 2 || !["validate", "train", "heldout"].contains(&args[0].as_str()) {
        return Err(
            "Usage: gwcar1a_record validate PROTOCOL | train/heldout PROTOCOL CANDIDATES OUTPUT"
                .into(),
        );
    }
    let stage = args[0].as_str();
    if (stage == "validate" && args.len() != 2) || (stage != "validate" && args.len() != 4) {
        return Err("invalid CAR-1A stage arguments".into());
    }
    let protocol_path = Path::new(&args[1]);
    let protocol = read_json(protocol_path)?;
    let check = std::process::Command::new("python3")
        .args(["scripts/gwcar1a_preregister.py", "validate", &args[1]])
        .output()?;
    if !check.status.success() {
        return Err(String::from_utf8_lossy(&check.stderr).to_string().into());
    }
    if stage == "validate" {
        return Ok(());
    }
    if protocol["status"] != "frozen_before_car1a_outcomes"
        || protocol["runner"]["executable_sha256"] != file_sha(&std::env::current_exe()?)?
    {
        return Err("CAR-1A replay requires a frozen protocol and matching executable".into());
    }
    let candidates_path = Path::new(&args[2]);
    let candidates = read_json(candidates_path)?;
    let check = std::process::Command::new("python3")
        .args([
            "scripts/gwcar1a_candidates.py",
            "validate",
            &args[1],
            &args[2],
        ])
        .output()?;
    if !check.status.success() {
        return Err(String::from_utf8_lossy(&check.stderr).to_string().into());
    }
    if candidates["stage"] != stage {
        return Err("CAR-1A candidate stage mismatch".into());
    }
    let authority = |key: &str| -> Result<&Path> {
        Ok(Path::new(
            protocol["authorities"][key]["path"]
                .as_str()
                .ok_or("CAR-1A authority path absent")?,
        ))
    };
    let execution_path = authority("v2_execution")?;
    let capture_path = authority("v2_capture")?;
    let reference_path = authority(if stage == "train" {
        "state1_train_replay"
    } else {
        "state1_heldout_replay"
    })?;
    let execution = read_json(execution_path)?;
    let capture = read_json(capture_path)?;
    let reference = read_json(reference_path)?;
    let rows = execution["rows"]
        .as_array()
        .ok_or("execution rows absent")?;
    if rows.len() != ROWS {
        return Err("CAR-1A requires 666 execution rows".into());
    }
    let container = Path::new(execution["container"].as_str().ok_or("container absent")?);
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("incoherent CAR-1A execution image".into());
    }
    let plan = outcome.plan.ok_or("CAR-1A plan absent")?;
    if execution["plan_sha256"] != sha(&serde_json::to_vec(&plan)?) {
        return Err("CAR-1A execution plan changed".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let tail_ops = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange {
            start: 24,
            end: plan.layers.len(),
        },
    )?;
    if plan.layers.len() != 34 || ops.hidden() != HIDDEN {
        return Err("CAR-1A model geometry changed".into());
    }
    let captured_heads = checked(
        capture_path.parent().ok_or("capture root absent")?,
        &capture,
        "head-values.f32",
        ROWS * HEADS * HEAD_DIM,
    )?;
    let natural_carriers = checked(
        capture_path.parent().ok_or("capture root absent")?,
        &capture,
        "carrier-before.f32",
        ROWS * HIDDEN,
    )?;
    let candidate_ids: Vec<u32> = ids(&execution["candidate_token_ids"])?
        .into_iter()
        .map(|id| id as u32)
        .collect();
    if candidate_ids.len() != WIDTH {
        return Err("CAR-1A candidate-token dimension changed".into());
    }
    let selected = ops.select_output_head(&candidate_ids)?;
    let stage_rows: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            let split = item["row"]["split"].as_str();
            (stage == "train" && split == Some("train"))
                || (stage == "heldout" && matches!(split, Some("validation" | "test")))
        })
        .collect();
    let count = if stage == "train" { 396 } else { 270 };
    let row_indices: Vec<_> = stage_rows.iter().map(|(index, _)| *index).collect();
    if row_indices.len() != count
        || ids(&candidates["row_indices"])? != row_indices
        || ids(&reference["row_indices"])? != row_indices
    {
        return Err("CAR-1A stage row order changed".into());
    }
    let decoded = checked(
        candidates_path.parent().ok_or("candidate root absent")?,
        &candidates,
        "carriers.f32",
        count * CANDIDATES * HIDDEN,
    )?;
    let reference_contexts = ids(&reference["context_ids"])?;
    let exact_context = reference_contexts
        .iter()
        .position(|id| *id == 128)
        .ok_or("STATE-1 context 128 absent")?;
    let donor_context = reference_contexts
        .iter()
        .position(|id| *id == 0)
        .ok_or("STATE-1 context 0 absent")?;
    let reference_proximal = checked(
        reference_path.parent().ok_or("reference root absent")?,
        &reference,
        "proximal.f32",
        count * reference_contexts.len() * 2 * WIDTH,
    )?;
    let reference_terminal = checked(
        reference_path.parent().ok_or("reference root absent")?,
        &reference,
        "terminal.f32",
        count * reference_contexts.len() * 2 * WIDTH,
    )?;

    let output = Path::new(&args[3]);
    std::fs::create_dir(output)?;
    let mut proximal = writer(&output.join("proximal.f32"))?;
    let mut terminal_out = writer(&output.join("terminal.f32"))?;
    let mut firings = 0_usize;
    for (ordinal, (row, item)) in stage_rows.iter().enumerate() {
        let donor = usize_at(&candidates["donor_row_indices"][*row])?;
        if donor >= ROWS || donor == *row {
            return Err("invalid CAR-1A donor".into());
        }
        let natural = &natural_carriers[row * HIDDEN..(row + 1) * HIDDEN];
        let target_heads = head_values(&captured_heads, *row);
        let donor_heads = head_values(&captured_heads, donor);
        let tokens = ids(&item["row"]["prompt"]["token_ids"])?;
        let target = tokens.len().checked_sub(1).ok_or("empty CAR-1A prompt")?;
        let mut prefix = RowKvState::default();
        {
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut prefix)?;
            for &token in &tokens[..target] {
                session.step_observed(token as u32, &mut NoopObserver)?;
            }
        }
        for candidate in 0..CANDIDATES {
            let start = (ordinal * CANDIDATES + candidate) * HIDDEN;
            let carrier = &decoded[start..start + HIDDEN];
            for arm in 0..2 {
                let heads = paired_heads(arm, &target_heads, &donor_heads)?;
                let replay = replay_attention_mixture(&plan, &ops, &backend, 24, carrier, &heads)?;
                let prox =
                    ops.readout_carrier_selected(&backend, &replay.carrier_after, &selected)?;
                let term = terminal(
                    &plan,
                    &ops,
                    &tail_ops,
                    &backend,
                    &prefix,
                    natural,
                    target,
                    replay.carrier_after,
                    &selected,
                )?;
                firings += 1;
                if candidate <= 1 {
                    let context = if candidate == 0 {
                        exact_context
                    } else {
                        donor_context
                    };
                    let offset =
                        (ordinal * reference_contexts.len() + context) * 2 * WIDTH + arm * WIDTH;
                    if !same(&prox, &reference_proximal[offset..offset + WIDTH])
                        || !same(&term, &reference_terminal[offset..offset + WIDTH])
                    {
                        return Err(format!("CAR-1A STATE-1 identity failed at row {row}, candidate {candidate}, arm {arm}").into());
                    }
                }
                put(&mut proximal, &prox)?;
                put(&mut terminal_out, &term)?;
            }
        }
        if (ordinal + 1) % 10 == 0 || ordinal + 1 == count {
            eprintln!("CAR-1A {stage} rows {}/{}", ordinal + 1, count);
        }
    }
    proximal.flush()?;
    terminal_out.flush()?;
    let manifest = json!({
        "schema":"larql.gwcar1a.replay.v1", "stage":stage,
        "protocol_sha256":protocol["protocol_sha256"],
        "candidate_manifest_sha256":file_sha(candidates_path)?,
        "state1_reference_manifest_sha256":file_sha(reference_path)?,
        "execution_file_sha256":file_sha(execution_path)?,
        "executable_sha256":file_sha(&std::env::current_exe()?)?,
        "row_indices":row_indices, "candidate_ids":(0..CANDIDATES).collect::<Vec<_>>(),
        "h1_arms":["donor","target_exact_natural"],
        "intervention_firings":firings, "identity_bit_mismatches":0,
        "artifacts":[artifact(output,"proximal.f32",count)?,
                     artifact(output,"terminal.f32",count)?],
        "latency_admissible":false,
    });
    let mut out = writer(&output.join("replay.json"))?;
    serde_json::to_writer_pretty(&mut out, &manifest)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    run(&std::env::args().skip(1).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_h1_arms_leave_all_seven_donor_heads_fixed() {
        let natural: Vec<_> = (0..HEADS)
            .map(|head| vec![100.0 + head as f32; HEAD_DIM])
            .collect();
        let donor: Vec<_> = (0..HEADS)
            .map(|head| vec![200.0 + head as f32; HEAD_DIM])
            .collect();
        let baseline = paired_heads(0, &natural, &donor).unwrap();
        let treatment = paired_heads(1, &natural, &donor).unwrap();
        assert_eq!(baseline, donor);
        assert_eq!(treatment[1], natural[1]);
        for head in [0, 2, 3, 4, 5, 6, 7] {
            assert_eq!(treatment[head], donor[head]);
        }
    }
}
