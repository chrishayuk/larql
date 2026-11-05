//! CAR-1B paired carrier-replacement replay, isolated from the sealed STATE-1 binary.
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
    intervene_heads::HeadInterventionPlan,
    kv::RowKvState,
    observe::{AttentionHeadRecord, NoopObserver},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands, SelectedOutputHead},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const ROWS: usize = 597;
const CANDIDATES: usize = 20;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const WIDTH: usize = 57;

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
        return Err(format!("CAR-1B artifact hash mismatch: {name}").into());
    }
    let bytes = std::fs::read(path)?;
    if bytes.len() != count * 4 {
        return Err(format!("CAR-1B artifact size mismatch: {name}").into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("nonfinite CAR-1B artifact: {name}").into());
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
        return Err("invalid CAR-1B head geometry".into());
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
        return Err("nonfinite CAR-1B replay output".into());
    }
    for value in values {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
fn artifact(root: &Path, name: &str, shape: Value) -> Result<Value> {
    let path = root.join(name);
    Ok(json!({"path": name, "shape": shape, "dtype": "f32-le",
        "bytes": std::fs::metadata(&path)?.len(), "sha256": file_sha(&path)?}))
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
    let result = session.step_from_carrier_intervened(
        natural_carrier,
        &mut observer,
        &intervention,
        &HeadInterventionPlan::none(),
    )?;
    if result.firings.len() != 1 {
        return Err("CAR-1B attention-write intervention did not fire once".into());
    }
    Ok(ops.readout_carrier_selected(backend, &observer.0, selected)?)
}

#[derive(Default)]
struct Capture {
    target: usize,
    last: Vec<f32>,
    before: Vec<f32>,
    after: Vec<f32>,
    early: std::collections::BTreeMap<usize, Vec<f32>>,
    heads: std::collections::BTreeMap<usize, Vec<f32>>,
    error: bool,
}
impl StepObserver for Capture {
    fn event(&mut self, _: StepEvent) {}
    fn wants_attention_heads(&self) -> bool {
        true
    }
    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        layer == 24 && position == self.target
    }
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer == 24 && record.position == self.target {
            self.error |= self
                .heads
                .insert(record.head, record.values.to_vec())
                .is_some();
        }
    }
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.position != self.target {
            return;
        }
        if record.site == SublayerSite::Attention {
            if [12, 16].contains(&record.layer) {
                self.error |= self.early.insert(record.layer, self.last.clone()).is_some();
            }
            if record.layer == 24 {
                self.error |= !self.before.is_empty();
                self.before = self.last.clone();
                self.after = record.after.to_vec();
            }
        }
        self.last = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for x in &mut self.last {
                *x *= scale;
            }
        }
    }
}

fn python_check(command: &str, paths: &[&Path]) -> Result<()> {
    let mut cmd = std::process::Command::new("python3");
    cmd.arg("scripts/gwcar1b_runtime.py").arg(command);
    for path in paths {
        cmd.arg(path);
    }
    let result = cmd.output()?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).to_string().into());
    }
    Ok(())
}

fn write_manifest(path: &Path, value: &Value) -> Result<()> {
    let mut out = writer(path)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn context_heads(
    context: usize,
    arm: usize,
    natural: &[Vec<f32>],
    donor: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    if ![0, 128, 255].contains(&context) {
        return Err("unknown reference context".into());
    }
    let mut values = if context == 255 {
        natural.to_vec()
    } else {
        donor.to_vec()
    };
    values[1] = if arm == 1 {
        natural[1].clone()
    } else {
        donor[1].clone()
    };
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn capture(
    protocol: &Value,
    output: &Path,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    selected: &SelectedOutputHead,
) -> Result<()> {
    let names = [
        "carrier-before.f32",
        "carrier-after.f32",
        "enter_l12.f32",
        "enter_l16.f32",
        "subject_embedding.f32",
        "head-values.f32",
        "natural-terminal.f32",
    ];
    let mut files = names
        .iter()
        .map(|name| writer(&output.join(name)))
        .collect::<Result<Vec<_>>>()?;
    let rows: Vec<&Value> = protocol["train_rows"]
        .as_array()
        .ok_or("train rows absent")?
        .iter()
        .chain(
            protocol["confirmation_rows"]
                .as_array()
                .ok_or("confirmation rows absent")?
                .iter(),
        )
        .collect();
    if rows.len() != ROWS {
        return Err("capture population changed".into());
    }
    let historical_path = Path::new(
        protocol["historical_capture"]
            .as_str()
            .ok_or("historical capture absent")?,
    );
    let historical = read_json(historical_path)?;
    let old_carriers = checked(
        historical_path.parent().ok_or("capture root absent")?,
        &historical,
        "carrier-before.f32",
        666 * HIDDEN,
    )?;
    for (index, row) in rows.iter().enumerate() {
        let tokens = ids(&row["token_ids"])?;
        let mut observer = Capture {
            target: tokens.len().checked_sub(1).ok_or("empty prompt")?,
            ..Default::default()
        };
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv)?;
        for token in tokens {
            session.step_observed(token as u32, &mut observer)?;
        }
        if observer.error
            || observer.heads.len() != HEADS
            || observer.early.len() != 2
            || observer.before.len() != HIDDEN
            || observer.after.len() != HIDDEN
            || observer.early.values().any(|v| v.len() != HIDDEN)
        {
            return Err("incomplete or repeated capture".into());
        }
        let heads: Vec<_> = (0..HEADS)
            .map(|head| observer.heads[&head].clone())
            .collect();
        let replay = replay_attention_mixture(plan, ops, backend, 24, &observer.before, &heads)?;
        if !same(&replay.carrier_after, &observer.after) {
            return Err("natural mixture is not bit exact".into());
        }
        if index < 396 {
            let old_index = usize_at(&row["historical_row_index"])?;
            if old_index >= 666
                || !same(
                    &observer.before,
                    &old_carriers[old_index * HIDDEN..(old_index + 1) * HIDDEN],
                )
            {
                return Err(format!("historical train carrier identity failed: {index}").into());
            }
        }
        let subject_tokens = ids(&row["subject_token_ids"])?;
        if subject_tokens.is_empty() {
            return Err("empty subject tokenization".into());
        }
        let mut pooled = vec![0_f64; HIDDEN];
        for &token in &subject_tokens {
            let embedding = ops.embed_token(plan, backend, token as u32)?;
            for (value, sum) in embedding.iter().zip(&mut pooled) {
                *sum += f64::from(*value);
            }
        }
        let pooled: Vec<f32> = pooled
            .iter()
            .map(|x| (x / subject_tokens.len() as f64) as f32)
            .collect();
        put(&mut files[0], &observer.before)?;
        put(&mut files[1], &observer.after)?;
        put(&mut files[2], &observer.early[&12])?;
        put(&mut files[3], &observer.early[&16])?;
        put(&mut files[4], &pooled)?;
        for head in heads {
            put(&mut files[5], &head)?;
        }
        put(
            &mut files[6],
            &ops.readout_carrier_selected(backend, &observer.last, selected)?,
        )?;
        if (index + 1) % 10 == 0 || index + 1 == ROWS {
            eprintln!("CAR-1B capture {}/{ROWS}", index + 1);
        }
    }
    for file in &mut files {
        file.flush()?;
    }
    let entries = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            artifact(
                output,
                name,
                if i == 5 {
                    json!([ROWS, HEADS, HEAD_DIM])
                } else if i == 6 {
                    json!([ROWS, WIDTH])
                } else {
                    json!([ROWS, HIDDEN])
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    write_manifest(
        &output.join("capture.json"),
        &json!({
            "schema": "larql.gwcar1b.capture.v1", "protocol_sha256": protocol["protocol_sha256"],
            "executable_sha256": file_sha(&std::env::current_exe()?)?, "rows": ROWS,
            "identity_bit_mismatches": 0, "artifacts": entries,
            "historical_train_identity_rows": 396, "natural_mixture_identity_rows": ROWS,
            "feature_observation": "final prompt position entering L12/L16; standalone scaled embedding mean",
            "efficiency_claim": false,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn replay(
    protocol_path: &Path,
    protocol: &Value,
    capture_path: &Path,
    output: &Path,
    candidate_path: Option<&Path>,
    reference_path: Option<&Path>,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tail: &PreparedOperands,
    backend: &ProductionBackend,
    selected: &SelectedOutputHead,
) -> Result<()> {
    python_check("check-capture", &[protocol_path, capture_path])?;
    let manifest = read_json(capture_path)?;
    let root = capture_path.parent().ok_or("capture directory absent")?;
    let heads = checked(root, &manifest, "head-values.f32", ROWS * HEADS * HEAD_DIM)?;
    let carriers = checked(root, &manifest, "carrier-before.f32", ROWS * HIDDEN)?;
    let after = checked(root, &manifest, "carrier-after.f32", ROWS * HIDDEN)?;
    let natural_terminal = checked(root, &manifest, "natural-terminal.f32", ROWS * WIDTH)?;
    let decoded = if let Some(path) = candidate_path {
        python_check("check-candidates", &[protocol_path, path])?;
        checked(
            path.parent().ok_or("candidate root absent")?,
            &read_json(path)?,
            "carriers.f32",
            201 * CANDIDATES * HIDDEN,
        )?
    } else {
        Vec::new()
    };
    let reference = if let Some(path) = reference_path {
        let doc = read_json(path)?;
        if doc["protocol_sha256"] != protocol["protocol_sha256"]
            || doc["executable_sha256"] != file_sha(&std::env::current_exe()?)?
            || doc["capture_manifest_sha256"] != file_sha(capture_path)?
            || doc["identity_bit_mismatches"] != 0
            || doc["ids"] != json!([0, 128, 255])
        {
            return Err("reference binding changed".into());
        }
        let root = path.parent().ok_or("reference root absent")?;
        Some((
            checked(root, &doc, "proximal.f32", 201 * 3 * 2 * WIDTH)?,
            checked(root, &doc, "terminal.f32", 201 * 3 * 2 * WIDTH)?,
        ))
    } else {
        None
    };
    let rows = protocol["confirmation_rows"]
        .as_array()
        .ok_or("confirmation rows absent")?;
    if rows.len() != 201 {
        return Err("confirmation row count changed".into());
    }
    let count = if candidate_path.is_some() {
        CANDIDATES
    } else {
        3
    };
    let mut proximal = writer(&output.join("proximal.f32"))?;
    let mut terminals = writer(&output.join("terminal.f32"))?;
    for (index, row) in rows.iter().enumerate() {
        let global = 396 + index;
        let donor_index = usize_at(&row["donor_index"])?;
        if donor_index >= 201 || donor_index == index {
            return Err("invalid donor index".into());
        }
        let donor_global = 396 + donor_index;
        let natural = &carriers[global * HIDDEN..(global + 1) * HIDDEN];
        let donor_carrier = &carriers[donor_global * HIDDEN..(donor_global + 1) * HIDDEN];
        let natural_heads = head_values(&heads, global);
        let donor_heads = head_values(&heads, donor_global);
        let tokens = ids(&row["token_ids"])?;
        let target = tokens.len().checked_sub(1).ok_or("empty prompt")?;
        let mut prefix = RowKvState::default();
        {
            let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut prefix)?;
            for &token in &tokens[..target] {
                session.step_observed(token as u32, &mut NoopObserver)?;
            }
        }
        for candidate in 0..count {
            let context = if candidate_path.is_some() {
                128
            } else {
                [0, 128, 255][candidate]
            };
            let carrier = if candidate_path.is_some() {
                let start = (index * CANDIDATES + candidate) * HIDDEN;
                &decoded[start..start + HIDDEN]
            } else if context == 0 {
                donor_carrier
            } else {
                natural
            };
            for arm in 0..2 {
                let values = if candidate_path.is_some() {
                    paired_heads(arm, &natural_heads, &donor_heads)?
                } else {
                    context_heads(context, arm, &natural_heads, &donor_heads)?
                };
                let mixed = replay_attention_mixture(plan, ops, backend, 24, carrier, &values)?;
                let prox = ops.readout_carrier_selected(backend, &mixed.carrier_after, selected)?;
                let term = terminal(
                    plan,
                    ops,
                    tail,
                    backend,
                    &prefix,
                    natural,
                    target,
                    mixed.carrier_after.clone(),
                    selected,
                )?;
                if candidate_path.is_none()
                    && context == 255
                    && arm == 1
                    && (!same(
                        &mixed.carrier_after,
                        &after[global * HIDDEN..(global + 1) * HIDDEN],
                    ) || !same(
                        &term,
                        &natural_terminal[global * WIDTH..(global + 1) * WIDTH],
                    ))
                {
                    return Err(format!(
                        "fresh all-natural continuation identity failed at {index}"
                    )
                    .into());
                }
                if let Some((ref_prox, ref_term)) = &reference {
                    if candidate <= 1 {
                        let reference_index = if candidate == 0 { 1 } else { 0 };
                        let offset = ((index * 3 + reference_index) * 2 + arm) * WIDTH;
                        if !same(&prox, &ref_prox[offset..offset + WIDTH])
                            || !same(&term, &ref_term[offset..offset + WIDTH])
                        {
                            return Err(format!(
                                "fresh oracle identity failed at {index}/{candidate}/{arm}"
                            )
                            .into());
                        }
                    }
                }
                put(&mut proximal, &prox)?;
                put(&mut terminals, &term)?;
            }
        }
        if (index + 1) % 5 == 0 || index + 1 == 201 {
            eprintln!(
                "CAR-1B {} {}/201",
                if candidate_path.is_some() {
                    "replay"
                } else {
                    "reference"
                },
                index + 1
            );
        }
    }
    proximal.flush()?;
    terminals.flush()?;
    let kind = if candidate_path.is_some() {
        "replay"
    } else {
        "reference"
    };
    write_manifest(
        &output.join(format!("{kind}.json")),
        &json!({
            "schema": format!("larql.gwcar1b.{kind}.v1"), "protocol_sha256": protocol["protocol_sha256"],
            "executable_sha256": file_sha(&std::env::current_exe()?)?,
            "capture_manifest_sha256": file_sha(capture_path)?,
            "candidate_manifest_sha256": candidate_path.map(file_sha).transpose()?,
            "reference_manifest_sha256": reference_path.map(file_sha).transpose()?,
            "ids": if candidate_path.is_some() { (0..CANDIDATES).collect::<Vec<_>>() } else { vec![0,128,255] },
            "h1_arms": ["donor", "target_exact_natural"], "intervention_firings": 201 * count * 2,
            "identity_bit_mismatches": 0, "efficiency_claim": false,
            "artifacts": [artifact(output,"proximal.f32",json!([201,count,2,WIDTH]))?,
                          artifact(output,"terminal.f32",json!([201,count,2,WIDTH]))?],
        }),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let stage = args.first().map(String::as_str).unwrap_or("");
    let expected_args = match stage {
        "capture" => 3,
        "reference" => 4,
        "replay" => 6,
        _ => 0,
    };
    if expected_args == 0 || args.len() != expected_args {
        return Err("Usage: gwcar1b_record capture PROTOCOL OUTPUT | reference PROTOCOL CAPTURE OUTPUT | replay PROTOCOL CAPTURE CANDIDATES REFERENCE OUTPUT".into());
    }
    let protocol_path = Path::new(&args[1]);
    python_check("validate", &[protocol_path])?;
    let protocol = read_json(protocol_path)?;
    if protocol["status"] != "frozen_before_capture"
        || protocol["executable_sha256"] != file_sha(&std::env::current_exe()?)?
    {
        return Err("capture/replay requires matching frozen executable".into());
    }
    let container = Path::new(protocol["container"].as_str().ok_or("container absent")?);
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("incoherent model".into());
    }
    let plan = outcome.plan.ok_or("plan absent")?;
    if protocol["plan_sha256"] != sha(&serde_json::to_vec(&plan)?) {
        return Err("plan changed".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    if plan.layers.len() != 34 || ops.hidden() != HIDDEN {
        return Err("model geometry changed".into());
    }
    let tokens = ids(&protocol["candidate_token_ids"])?;
    if tokens.len() != WIDTH {
        return Err("readout vocabulary changed".into());
    }
    let selected =
        ops.select_output_head(&tokens.iter().map(|&id| id as u32).collect::<Vec<_>>())?;
    let output = Path::new(args.last().unwrap());
    std::fs::create_dir(output)?;
    if stage == "capture" {
        return capture(&protocol, output, &plan, &ops, &backend, &selected);
    }
    let tail = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange {
            start: 24,
            end: plan.layers.len(),
        },
    )?;
    replay(
        protocol_path,
        &protocol,
        Path::new(&args[2]),
        output,
        if stage == "replay" {
            Some(Path::new(&args[3]))
        } else {
            None
        },
        if stage == "replay" {
            Some(Path::new(&args[4]))
        } else {
            None
        },
        &plan,
        &ops,
        &tail,
        &backend,
        &selected,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_change_only_registered_heads() {
        let natural: Vec<_> = (0..8).map(|i| vec![i as f32; 256]).collect();
        let donor: Vec<_> = (0..8).map(|i| vec![100. + i as f32; 256]).collect();
        for context in [0, 128, 255] {
            for arm in 0..2 {
                let values = context_heads(context, arm, &natural, &donor).unwrap();
                assert_eq!(
                    values[1],
                    if arm == 1 { &natural[1] } else { &donor[1] }.clone()
                );
                for head in [0, 2, 3, 4, 5, 6, 7] {
                    assert_eq!(
                        values[head],
                        if context == 255 {
                            &natural[head]
                        } else {
                            &donor[head]
                        }
                        .clone()
                    );
                }
                if context == 128 {
                    assert_eq!(values, paired_heads(arm, &natural, &donor).unwrap());
                }
            }
        }
    }
}
