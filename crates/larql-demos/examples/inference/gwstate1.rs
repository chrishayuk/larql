//! GW-STATE-1 paired exact-H1 context replay; no predictor or donor search.
use super::gwsup1_readout::file_sha;
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::{replay_attention_heads, replay_attention_mixture},
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    observe::NoopObserver,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

const ROWS: usize = 666;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const V2_ARMS: usize = 75;

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

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn usize_at(value: &Value) -> Result<usize> {
    Ok(value.as_u64().ok_or("missing unsigned integer")? as usize)
}
fn ids(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .ok_or("missing array")?
        .iter()
        .map(usize_at)
        .collect()
}
fn finite_f32(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != count * 4 {
        return Err(format!("wrong tensor size: {}", path.display()).into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("nonfinite tensor: {}", path.display()).into());
    }
    Ok(values)
}
fn checked(root: &Path, manifest: &Value, name: &str, count: usize) -> Result<Vec<f32>> {
    let entries: Vec<_> = manifest["artifacts"]
        .as_array()
        .ok_or("artifacts absent")?
        .iter()
        .filter(|entry| entry["path"] == name)
        .collect();
    if entries.len() != 1 || entries[0]["sha256"] != file_sha(&root.join(name))? {
        return Err(format!("tensor hash mismatch: {name}").into());
    }
    finite_f32(&root.join(name), count)
}
fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn put(out: &mut impl Write, data: &[f32]) -> Result<()> {
    if data.iter().any(|value| !value.is_finite()) {
        return Err("nonfinite STATE-1 output".into());
    }
    for value in data {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
fn artifact(root: &Path, name: &str, shape: Value) -> Result<Value> {
    Ok(json!({"path":name,"shape":shape,"dtype":"f32-le",
        "bytes":std::fs::metadata(root.join(name))?.len(),"sha256":file_sha(&root.join(name))?}))
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn head_values(data: &[f32], row: usize) -> Vec<Vec<f32>> {
    (0..HEADS)
        .map(|head| {
            let offset = (row * HEADS + head) * HEAD_DIM;
            data[offset..offset + HEAD_DIM].to_vec()
        })
        .collect()
}
fn mixed_heads(
    context: usize,
    h1_arm: usize,
    natural: &[Vec<f32>],
    donor: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    if context >= 256
        || h1_arm >= 2
        || natural.len() != HEADS
        || donor.len() != HEADS
        || natural
            .iter()
            .chain(donor)
            .any(|head| head.len() != HEAD_DIM || head.iter().any(|value| !value.is_finite()))
    {
        return Err("invalid STATE-1 head-mixture geometry".into());
    }
    let mask = context % 128;
    Ok((0..HEADS)
        .map(|head| {
            if head == 1 {
                if h1_arm == 0 {
                    donor[head].clone()
                } else {
                    natural[head].clone()
                }
            } else {
                let bit = if head == 0 { 0 } else { head - 1 };
                if mask & (1 << bit) != 0 {
                    natural[head].clone()
                } else {
                    donor[head].clone()
                }
            }
        })
        .collect())
}
fn selected_full(logits: &[f32], ids: &[u32]) -> Vec<f32> {
    ids.iter().map(|&id| logits[id as usize]).collect()
}

fn donor_rows(rows: &[Value]) -> Result<Vec<usize>> {
    if rows.len() != ROWS {
        return Err("STATE-1 requires 666 execution rows".into());
    }
    let mut by_edge = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let edge = row["row"]["edge_id"].as_str().ok_or("edge ID absent")?;
        if by_edge.insert(edge, index).is_some() {
            return Err("duplicate edge ID".into());
        }
    }
    let mut result = Vec::with_capacity(ROWS);
    for (index, item) in rows.iter().enumerate() {
        let row = &item["row"];
        let controls = row["control_ids"].as_array().ok_or("controls absent")?;
        let controls: Vec<_> = controls
            .iter()
            .filter(|c| c["kind"] == "same_relation_different_subject")
            .collect();
        if controls.len() != 1 {
            return Err("missing unique STATE-1 matched control".into());
        }
        let donor_edge = controls[0]["paired_edge_id"]
            .as_str()
            .ok_or("donor edge absent")?;
        let donor_index = *by_edge.get(donor_edge).ok_or("donor edge not found")?;
        let donor = &rows[donor_index]["row"];
        if index == donor_index
            || row["subject_id"] == donor["subject_id"]
            || row["split"] != donor["split"]
            || row["semantic_edge"]["relation"] != donor["semantic_edge"]["relation"]
            || row["semantic_edge"]["prompt_semantic_family"]
                != donor["semantic_edge"]["prompt_semantic_family"]
        {
            return Err(format!("invalid matched donor at row {index}").into());
        }
        result.push(donor_index);
    }
    if result.iter().copied().collect::<BTreeSet<_>>().len() != ROWS {
        return Err("STATE-1 matched donors are not a derangement".into());
    }
    Ok(result)
}

fn context_ids(stage: &str, selection: Option<&Value>) -> Result<Vec<usize>> {
    if stage == "train" {
        return Ok((0..256).collect());
    }
    let selection = selection.ok_or("heldout requires frozen train selection")?;
    let ids = ids(&selection["context_ids"])?;
    let mut expected = BTreeSet::from([0, 127, 128, 255]);
    let chosen = selection["carrier_branch_selected"]
        .as_array()
        .ok_or("selected branches absent")?;
    if chosen.len() != 2 {
        return Err("exactly two selected carrier branches required".into());
    }
    for (carrier, value) in chosen.iter().enumerate() {
        if !value.is_null() {
            let context = usize_at(value)?;
            if context >= 256 || context / 128 != carrier {
                return Err("selected context crosses carrier branch".into());
            }
            expected.insert(context);
        }
    }
    let expected: Vec<_> = expected.into_iter().collect();
    if ids != expected {
        return Err("heldout context order differs from frozen selection plus sentinels".into());
    }
    Ok(ids)
}

#[allow(clippy::too_many_arguments)]
fn tail_replacement(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tail_ops: &PreparedOperands,
    backend: &ProductionBackend,
    prefix: &RowKvState,
    carrier: &[f32],
    target: usize,
    replacement: Vec<f32>,
    selected: &larql_vindex::format::vindex3::opplan::exec::prepared::SelectedOutputHead,
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
        carrier,
        &mut observer,
        &intervention,
        &HeadInterventionPlan::none(),
    )?;
    if result.firings.len() != 1 {
        return Err("STATE-1 intervention did not fire exactly once".into());
    }
    ops.readout_carrier_selected(backend, &observer.0, selected)
        .map_err(Into::into)
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() < 2 || !["validate", "train", "heldout"].contains(&args[0].as_str()) {
        return Err("Usage: observatory_record --gwstate1 validate PROTOCOL | train PROTOCOL OUTPUT | heldout PROTOCOL OUTPUT SELECTION".into());
    }
    let stage = args[0].as_str();
    if (stage == "validate" && args.len() != 2)
        || (stage == "train" && args.len() != 3)
        || (stage == "heldout" && args.len() != 4)
    {
        return Err("invalid GW-STATE-1 stage arguments".into());
    }
    let protocol_path = Path::new(&args[1]);
    let protocol = read_json(protocol_path)?;
    let check = std::process::Command::new("python3")
        .args(["scripts/gwstate1_preregister.py", "validate", &args[1]])
        .output()?;
    if !check.status.success() {
        return Err(String::from_utf8_lossy(&check.stderr).to_string().into());
    }
    if stage == "validate" {
        return Ok(());
    }
    if protocol["status"] != "frozen_before_state1_outcomes"
        || protocol["runner"]["executable_sha256"] != file_sha(&std::env::current_exe()?)?
    {
        return Err("STATE-1 replay requires a frozen protocol and matching executable".into());
    }
    let selection = if stage == "heldout" {
        let path = Path::new(&args[3]);
        let value = read_json(path)?;
        let check = std::process::Command::new("python3")
            .args(["scripts/gwstate1_select.py", "validate", &args[1], &args[3]])
            .output()?;
        if !check.status.success() {
            return Err(String::from_utf8_lossy(&check.stderr).to_string().into());
        }
        Some(value)
    } else {
        None
    };
    let contexts = context_ids(stage, selection.as_ref())?;
    let authority = |key: &str| -> Result<&Path> {
        Ok(Path::new(
            protocol["authorities"][key]["path"]
                .as_str()
                .ok_or("authority path absent")?,
        ))
    };
    let execution_path = authority("v2_execution")?;
    let capture_path = authority("v2_capture")?;
    let v2_replay_path = authority("v2_replay")?;
    let execution = read_json(execution_path)?;
    let capture = read_json(capture_path)?;
    let v2_replay = read_json(v2_replay_path)?;
    let rows = execution["rows"]
        .as_array()
        .ok_or("execution rows absent")?;
    let donors = donor_rows(rows)?;
    let container = Path::new(execution["container"].as_str().ok_or("container absent")?);
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("incoherent STATE-1 execution image".into());
    }
    let plan = outcome.plan.ok_or("plan absent")?;
    if execution["plan_sha256"] != format!("sha256:{}", sha(&serde_json::to_vec(&plan)?)) {
        return Err("STATE-1 execution plan changed".into());
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
        return Err("STATE-1 model geometry changed".into());
    }
    let all_heads = checked(
        capture_path.parent().ok_or("capture root absent")?,
        &capture,
        "head-values.f32",
        ROWS * HEADS * HEAD_DIM,
    )?;
    let all_carriers = checked(
        capture_path.parent().ok_or("capture root absent")?,
        &capture,
        "carrier-before.f32",
        ROWS * HIDDEN,
    )?;
    let all_after = checked(
        capture_path.parent().ok_or("capture root absent")?,
        &capture,
        "carrier-after.f32",
        ROWS * HIDDEN,
    )?;
    let candidate_ids: Vec<u32> = ids(&execution["candidate_token_ids"])?
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let selected = ops.select_output_head(&candidate_ids)?;
    let width = candidate_ids.len();
    let v2_proximal = checked(
        v2_replay_path.parent().ok_or("V2 replay root absent")?,
        &v2_replay,
        "proximal.f32",
        ROWS * V2_ARMS * width,
    )?;
    let v2_terminal = checked(
        v2_replay_path.parent().ok_or("V2 replay root absent")?,
        &v2_replay,
        "terminal.f32",
        ROWS * V2_ARMS * width,
    )?;
    let split_rows: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            let split = item["row"]["split"].as_str();
            (stage == "train" && split == Some("train"))
                || (stage == "heldout" && matches!(split, Some("validation" | "test")))
        })
        .collect();
    let expected = if stage == "train" { 396 } else { 270 };
    if split_rows.len() != expected {
        return Err("STATE-1 stage row coverage changed".into());
    }
    let output = Path::new(&args[2]);
    std::fs::create_dir_all(output)?;
    let mut proximal = writer(&output.join("proximal.f32"))?;
    let mut terminal = writer(&output.join("terminal.f32"))?;
    let mut natural_norm_sums = [0.0_f64; 7];
    let mut firings = 0usize;
    for (ordinal, (row, item)) in split_rows.iter().enumerate() {
        let donor = donors[*row];
        let carrier = &all_carriers[row * HIDDEN..(row + 1) * HIDDEN];
        let after = &all_after[row * HIDDEN..(row + 1) * HIDDEN];
        let donor_carrier = &all_carriers[donor * HIDDEN..(donor + 1) * HIDDEN];
        let natural_heads = head_values(&all_heads, *row);
        let donor_heads = head_values(&all_heads, donor);
        let natural = replay_attention_heads(&plan, &ops, &backend, 24, carrier, &natural_heads)?;
        if natural.raw_reconstruction_relative_l2 > 1e-5
            || !natural.raw_reconstruction_relative_l2.is_finite()
            || !same(&natural.carrier_after, after)
            || natural.contribution_norms.len() != HEADS
            || natural
                .contribution_norms
                .iter()
                .any(|norm| !norm.is_finite() || *norm <= 0.0)
        {
            return Err(format!("natural STATE-1 composition parity failed at row {row}").into());
        }
        if stage == "train" {
            for (index, &head) in [0, 2, 3, 4, 5, 6, 7].iter().enumerate() {
                natural_norm_sums[index] += natural.contribution_norms[head];
            }
        }
        let tokens = ids(&item["row"]["prompt"]["token_ids"])?;
        let target = tokens.len().checked_sub(1).ok_or("empty prompt")?;
        let mut prefix = RowKvState::default();
        {
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut prefix)?;
            for &token in &tokens[..target] {
                session.step_observed(token as u32, &mut NoopObserver)?;
            }
        }
        let mut natural_kv = prefix.clone();
        let mut natural_session =
            DecodeSession::over_prepared(&plan, &ops, &backend, &mut natural_kv)?;
        let full_natural = natural_session
            .step_observed(tokens[target] as u32, &mut NoopObserver)?
            .logits
            .ok_or("natural full logits absent")?;
        let natural_terminal = selected_full(&full_natural, &candidate_ids);
        let natural_proximal = ops.readout_carrier_selected(&backend, after, &selected)?;
        let v2_offset = row * V2_ARMS * width;
        if !same(
            &natural_proximal,
            &v2_proximal[v2_offset..v2_offset + width],
        ) || !same(
            &natural_terminal,
            &v2_terminal[v2_offset..v2_offset + width],
        ) {
            return Err(format!("GW-V2 natural authority parity failed at row {row}").into());
        }
        let full_exact = {
            let intervention = InterventionPlan::none().with(Intervention::replace(
                Address::new(24, SublayerSite::Attention, [target])?,
                after.to_vec(),
                VectorProvenance::Literal {
                    sha256: vector_sha256(after),
                },
            )?)?;
            let mut kv = prefix.clone();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            let result = session.step_intervened(
                tokens[target] as u32,
                &mut NoopObserver,
                &intervention,
                &HeadInterventionPlan::none(),
            )?;
            if result.firings.len() != 1 {
                return Err("full exact STATE-1 control did not fire once".into());
            }
            firings += 1;
            selected_full(
                &result.logits.ok_or("exact full logits absent")?,
                &candidate_ids,
            )
        };
        let noop = tail_replacement(
            &plan,
            &ops,
            &tail_ops,
            &backend,
            &prefix,
            carrier,
            target,
            after.to_vec(),
            &selected,
        )?;
        let identity = tail_replacement(
            &plan,
            &ops,
            &tail_ops,
            &backend,
            &prefix,
            carrier,
            target,
            after.to_vec(),
            &selected,
        )?;
        firings += 2;
        if !same(&natural_terminal, &noop)
            || !same(&full_exact, &identity)
            || !same(&natural_terminal, &full_exact)
        {
            return Err(
                format!("STATE-1 natural/noop/exact/identity parity failed at row {row}").into(),
            );
        }
        for &context in &contexts {
            let entering = if context >= 128 {
                carrier
            } else {
                donor_carrier
            };
            for h1_arm in 0..2 {
                let heads = mixed_heads(context, h1_arm, &natural_heads, &donor_heads)?;
                let replay = replay_attention_mixture(&plan, &ops, &backend, 24, entering, &heads)?;
                if context == 255 && h1_arm == 1 && !same(&replay.carrier_after, after) {
                    return Err("STATE-1 all-natural identity composition failed".into());
                }
                let prox =
                    ops.readout_carrier_selected(&backend, &replay.carrier_after, &selected)?;
                let term = tail_replacement(
                    &plan,
                    &ops,
                    &tail_ops,
                    &backend,
                    &prefix,
                    carrier,
                    target,
                    replay.carrier_after,
                    &selected,
                )?;
                if context == 255
                    && h1_arm == 1
                    && (!same(&prox, &natural_proximal) || !same(&term, &natural_terminal))
                {
                    return Err("STATE-1 all-natural selected-logit identity failed".into());
                }
                put(&mut proximal, &prox)?;
                put(&mut terminal, &term)?;
                firings += 1;
            }
        }
        if (ordinal + 1) % 5 == 0 || ordinal + 1 == expected {
            eprintln!("GW-STATE-1 {stage} rows {}/{}", ordinal + 1, expected);
        }
    }
    proximal.flush()?;
    terminal.flush()?;
    let artifacts = vec![
        artifact(
            output,
            "proximal.f32",
            json!([expected, contexts.len(), 2, width]),
        )?,
        artifact(
            output,
            "terminal.f32",
            json!([expected, contexts.len(), 2, width]),
        )?,
    ];
    let row_indices: Vec<_> = split_rows.iter().map(|(index, _)| *index).collect();
    let norms: Vec<_> = natural_norm_sums
        .iter()
        .map(|value| value / expected as f64)
        .collect();
    let manifest = json!({
        "schema":"larql.gwstate1.replay.v1", "stage":stage,
        "protocol_sha256":protocol["protocol_sha256"],
        "execution_file_sha256":file_sha(execution_path)?,
        "capture_file_sha256":file_sha(capture_path)?,
        "v2_replay_file_sha256":file_sha(v2_replay_path)?,
        "executable_sha256":file_sha(&std::env::current_exe()?)?,
        "selection_sha256":if stage == "heldout" { selection.as_ref().map(|s| s["selection_sha256"].clone()) } else { None },
        "row_indices":row_indices, "context_ids":contexts,
        "h1_arms":["donor","target_exact_natural"],
        "artifacts":artifacts, "intervention_firings":firings,
        "parity_bit_mismatches":0, "natural_head_contribution_norms_train_mean":
            if stage == "train" { Some(norms) } else { None },
        "latency_admissible":false,
    });
    atomic_json(
        &output.join("replay.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_matched_controls_form_a_derangement() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let execution = read_json(&root.join("bench/gw-v2/gemma3-4b-it-phase1/execution.json"))
            .expect("bound GW-V2 execution file");
        let donors = donor_rows(execution["rows"].as_array().unwrap()).unwrap();
        assert_eq!(donors.len(), ROWS);
        assert!(donors.iter().enumerate().all(|(row, donor)| row != *donor));
    }

    #[test]
    fn heldout_context_order_is_selected_union_sentinels() {
        let selection = json!({"carrier_branch_selected":[1,129],
            "context_ids":[0,1,127,128,129,255]});
        assert_eq!(
            context_ids("heldout", Some(&selection)).unwrap(),
            [0, 1, 127, 128, 129, 255]
        );
        assert_eq!(context_ids("train", None).unwrap().len(), 256);
    }

    #[test]
    fn h1_arm_and_head_mask_are_independent() {
        let natural: Vec<_> = (0..HEADS)
            .map(|head| vec![100.0 + head as f32; HEAD_DIM])
            .collect();
        let donor: Vec<_> = (0..HEADS)
            .map(|head| vec![200.0 + head as f32; HEAD_DIM])
            .collect();
        let edge = mixed_heads(0, 1, &natural, &donor).unwrap();
        assert_eq!(edge[1], natural[1]);
        assert_eq!(edge[0], donor[0]);
        assert_eq!(edge[7], donor[7]);
        let h2_only = mixed_heads(2, 1, &natural, &donor).unwrap();
        assert_eq!(h2_only[2], natural[2]);
        assert_eq!(h2_only[0], donor[0]);
        let exact = mixed_heads(255, 1, &natural, &donor).unwrap();
        assert_eq!(exact, natural);
        let baseline = mixed_heads(255, 0, &natural, &donor).unwrap();
        assert_eq!(baseline[1], donor[1]);
        assert_eq!(baseline[2], natural[2]);
    }
}
