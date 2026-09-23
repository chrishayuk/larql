//! Hash-bound GW-V2 capture and intervention runner. No candidate selection.
use super::gwsup1_readout::file_sha;
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::{replay_attention_mixture, replay_softmax_source_head},
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    observe::{AttentionHeadRecord, NoopObserver},
};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
#[path = "gwv2_prefix.rs"]
mod prefix;

const DEPTHS: [usize; 8] = [0, 4, 8, 12, 16, 20, 23, 24];
const D: usize = 256;
const H: usize = 2560;
const N: usize = 666;
const CELLS: usize = 35;
type ReferenceMean = (Vec<f32>, Vec<f32>, usize);

#[derive(Default)]
struct TailCarrier(Vec<f32>);
impl StepObserver for TailCarrier {
    fn event(&mut self, _: StepEvent) {}
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.0 = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for x in &mut self.0 {
                *x *= scale;
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
fn floats(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != count * 4 {
        return Err(format!("wrong tensor size {}", path.display()).into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite tensor".into());
    }
    Ok(values)
}
fn checked(root: &Path, manifest: &Value, name: &str, count: usize) -> Result<Vec<f32>> {
    let entries: Vec<_> = manifest["artifacts"]
        .as_array()
        .ok_or("artifacts absent")?
        .iter()
        .filter(|e| e["path"] == name)
        .collect();
    if entries.len() != 1 || entries[0]["sha256"] != file_sha(&root.join(name))? {
        return Err("tensor hash mismatch".into());
    }
    floats(&root.join(name), count)
}
fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn put(out: &mut impl Write, data: &[f32]) -> Result<()> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite output".into());
    }
    for v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
fn artifact(root: &Path, name: &str, shape: Value) -> Result<Value> {
    Ok(
        json!({"path":name,"shape":shape,"dtype":"f32-le","bytes":std::fs::metadata(root.join(name))?.len(),"sha256":file_sha(&root.join(name))?}),
    )
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn vectors(data: &[f32], offset: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| data[(offset + i) * D..(offset + i + 1) * D].to_vec())
        .collect()
}
fn scale(direction: &[f32], natural: &[f32]) -> Result<Vec<f32>> {
    let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    let (a, b) = (norm(direction), norm(natural));
    if a <= 0. || b <= 0. || !a.is_finite() || !b.is_finite() {
        return Err("invalid reference norm".into());
    }
    Ok(direction.iter().map(|x| *x * (b / a) as f32).collect())
}

#[derive(Default)]
struct Capture {
    target: usize,
    positions: Vec<usize>,
    source: BTreeMap<usize, Vec<Vec<f32>>>,
    query: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    heads: BTreeMap<usize, Vec<f32>>,
    before: Vec<f32>,
    after: Vec<f32>,
    last: Vec<f32>,
    error: Option<String>,
}
impl StepObserver for Capture {
    fn event(&mut self, _: StepEvent) {}
    fn wants_attention_heads(&self) -> bool {
        true
    }
    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        position == self.target && DEPTHS.contains(&layer)
    }
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if record.position != self.target {
            return;
        }
        if layer == 24
            && self
                .heads
                .insert(record.head, record.values.to_vec())
                .is_some()
        {
            self.error = Some("duplicate head".into());
        }
        if record.head != 1 || !DEPTHS.contains(&layer) {
            return;
        }
        if record.source_start != 0
            || record.source_values.len() != self.target + 1
            || record.query.len() != D
        {
            self.error = Some("source geometry changed".into());
            return;
        }
        let values = self
            .positions
            .iter()
            .map(|&p| record.source_values[p].to_vec())
            .collect();
        if self.source.insert(layer, values).is_some() {
            self.error = Some("duplicate depth capture".into());
        }
        if layer == 24 {
            self.query = record.query.to_vec();
            self.keys = record.source_keys.iter().map(|k| k.to_vec()).collect();
            self.values = record.source_values.iter().map(|v| v.to_vec()).collect();
        }
    }
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.position != self.target {
            return;
        }
        if record.layer == 24 && record.site == SublayerSite::Attention {
            self.before = self.last.clone();
            self.after = record.after.to_vec();
        }
        self.last = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for v in &mut self.last {
                *v *= scale;
            }
        }
    }
}

fn verify_execution(path: &Path) -> Result<Value> {
    let execution = read_json(path)?;
    let result = std::process::Command::new("python3")
        .args([
            "scripts/gwv2_prepare.py",
            "validate",
            path.to_str().ok_or("invalid binding path")?,
        ])
        .output()?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).to_string().into());
    }
    let lineage: Value = serde_json::from_slice(&result.stdout)?;
    if execution["lineage"] != lineage || execution["schema"] != "larql.gwv2.execution.v1" {
        return Err("unbound execution lineage".into());
    }
    let mut body = execution.clone();
    body.as_object_mut().unwrap().remove("execution_sha256");
    if execution["execution_sha256"] != format!("sha256:{}", sha(&serde_json::to_vec(&body)?)) {
        return Err("execution binding hash mismatch".into());
    }
    let container = Path::new(execution["container"].as_str().ok_or("container absent")?);
    for (name, expected) in execution["container_metadata"]
        .as_object()
        .ok_or("metadata absent")?
    {
        if expected != &file_sha(&container.join(name))? {
            return Err("container changed".into());
        }
    }
    if execution["rows"].as_array().ok_or("rows absent")?.len() != N {
        return Err("wrong row count".into());
    }
    Ok(execution)
}

pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        return Err("Usage: observatory_record --gwv2 capture EXECUTION OUTPUT | replay EXECUTION OUTPUT CAPTURE FIT".into());
    }
    let execution_path = Path::new(&args[1]);
    let e = verify_execution(execution_path)?;
    let output = Path::new(&args[2]);
    std::fs::create_dir_all(output)?;
    let container = Path::new(e["container"].as_str().unwrap());
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("incoherent execution image".into());
    }
    let plan = outcome.plan.ok_or("plan absent")?;
    if e["plan_sha256"] != format!("sha256:{}", sha(&serde_json::to_vec(&plan)?)) {
        return Err("execution plan changed".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    if args[0] == "prefix-check" && args.len() == 4 {
        return prefix::check(
            &e,
            output,
            Path::new(&args[3]),
            &plan,
            &store,
            &ops,
            &backend,
        );
    }
    if args[0] == "inspect" && args.len() == 3 {
        let mapped = ops.mapped_residency();
        let census = ops.residency_census();
        let image = json!({"schema":"larql.gwv2.execution-image.v1","lineage":e["lineage"],"plan":plan,"allocation_bytes":ops.allocation_census().bytes,"prepared_resident_bytes":census.total(),"widened_f32_bytes":census.widened_f32(),"compact_bytes":census.compact(),"mapped_bytes":mapped.mapped_bytes,"mapped_resident_bytes":mapped.resident_bytes,"executable_sha256":file_sha(&std::env::current_exe()?)?});
        return atomic_json(
            &output.join("execution-image.json"),
            &serde_json::to_vec_pretty(&image)?,
        );
    }
    if args[0] == "capture" && args.len() == 3 {
        capture(&e, execution_path, output, &plan, &ops, &backend)
    } else if args[0] == "replay" && args.len() == 5 {
        replay(
            &e,
            execution_path,
            output,
            Path::new(&args[3]),
            Path::new(&args[4]),
            &plan,
            &ops,
            &backend,
        )
    } else {
        Err("invalid GW-V2 stage arguments".into())
    }
}

fn capture(
    e: &Value,
    binding: &Path,
    output: &Path,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
) -> Result<()> {
    let names = [
        "source-depth-v.f32",
        "natural-q.f32",
        "source-k.f32",
        "source-v.f32",
        "head-values.f32",
        "carrier-before.f32",
        "carrier-after.f32",
    ];
    let mut files = names
        .iter()
        .map(|name| writer(&output.join(name)))
        .collect::<Result<Vec<_>>>()?;
    let op = plan.layers[24]
        .attention
        .softmax()
        .ok_or("L24 not softmax")?;
    for (index, item) in e["rows"].as_array().unwrap().iter().enumerate() {
        let tokens = ids(&item["row"]["prompt"]["token_ids"])?;
        let positions = ids(&item["roles"]["subject_entity"])?;
        let mut c = Capture {
            target: tokens.len() - 1,
            positions,
            ..Default::default()
        };
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv)?;
        for token in tokens {
            session.step_observed(token as u32, &mut c)?;
        }
        if let Some(error) = c.error {
            return Err(error.into());
        }
        if c.source.len() != 8 || c.heads.len() != 8 || c.before.len() != H || c.after.len() != H {
            return Err("incomplete capture".into());
        }
        let heads = (0..8).map(|h| c.heads[&h].clone()).collect::<Vec<_>>();
        let h1 = replay_softmax_source_head(
            &c.query,
            &c.keys,
            &c.values,
            op.score_scale,
            op.logit_softcapping,
        )?;
        let mix = replay_attention_mixture(plan, ops, backend, 24, &c.before, &heads)?;
        if !same(&h1.values, &heads[1]) || !same(&mix.carrier_after, &c.after) {
            return Err("natural source/head reconstruction is not bit exact".into());
        }
        for ordinal in 0..c.positions.len() {
            for depth in DEPTHS {
                put(&mut files[0], &c.source[&depth][ordinal])?;
            }
        }
        put(&mut files[1], &c.query)?;
        for v in c.keys {
            put(&mut files[2], &v)?;
        }
        for v in c.values {
            put(&mut files[3], &v)?;
        }
        for v in heads {
            put(&mut files[4], &v)?;
        }
        put(&mut files[5], &c.before)?;
        put(&mut files[6], &c.after)?;
        if (index + 1) % 10 == 0 || index + 1 == N {
            eprintln!("GW-V2 capture {}/{N}", index + 1);
        }
    }
    for f in &mut files {
        f.flush()?;
    }
    let shapes = [
        json!([e["subject_rows"], 8, D]),
        json!([N, D]),
        json!([e["source_rows"], D]),
        json!([e["source_rows"], D]),
        json!([N, 8, D]),
        json!([N, H]),
        json!([N, H]),
    ];
    let artifacts = names
        .iter()
        .zip(shapes)
        .map(|(name, shape)| artifact(output, name, shape))
        .collect::<Result<Vec<_>>>()?;
    let manifest = json!({"schema":"larql.gwv2.capture.v1","lineage":e["lineage"],"execution_file_sha256":file_sha(binding)?,"artifacts":artifacts,"outcomes_captured":false,"source_and_carrier_bit_mismatches":0});
    atomic_json(
        &output.join("capture.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn replay(
    e: &Value,
    binding: &Path,
    output: &Path,
    capture_path: &Path,
    fit_path: &Path,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
) -> Result<()> {
    let container = Path::new(e["container"].as_str().ok_or("container missing")?);
    let inspection = inspect_container(container, true)?;
    let tail_store = OperandStore::open(container, &inspection)?;
    let tail_ops = PreparedOperands::load(
        plan,
        &tail_store,
        backend,
        ExecutionSlice::LayerRange {
            start: 24,
            end: plan.layers.len(),
        },
    )?;
    let capture = read_json(capture_path)?;
    let fit = read_json(fit_path)?;
    if capture["lineage"] != e["lineage"]
        || fit["lineage"] != e["lineage"]
        || capture["execution_file_sha256"] != file_sha(binding)?
        || fit["capture_file_sha256"] != file_sha(capture_path)?
        || fit["execution_file_sha256"] != file_sha(binding)?
        || fit["schema"] != "larql.gwv2.fit.v1"
    {
        return Err("replay lineage mismatch".into());
    }
    let cr = capture_path.parent().unwrap();
    let fr = fit_path.parent().unwrap();
    let source_rows = usize_at(&e["source_rows"])?;
    let subject_rows = usize_at(&e["subject_rows"])?;
    let all_q = checked(cr, &capture, "natural-q.f32", N * D)?;
    let all_k = checked(cr, &capture, "source-k.f32", source_rows * D)?;
    let all_v = checked(cr, &capture, "source-v.f32", source_rows * D)?;
    let all_heads = checked(cr, &capture, "head-values.f32", N * 8 * D)?;
    let before = checked(cr, &capture, "carrier-before.f32", N * H)?;
    let after = checked(cr, &capture, "carrier-after.f32", N * H)?;
    let predicted = checked(fr, &fit, "predictions.f32", 2 * CELLS * subject_rows * D)?;
    let reference_path = Path::new(
        e["reference_source_manifest"]
            .as_str()
            .ok_or("reference absent")?,
    );
    let role_path = Path::new(
        e["reference_roles"]
            .as_str()
            .ok_or("reference roles absent")?,
    );
    if e["reference_source_sha256"] != file_sha(reference_path)?
        || e["reference_roles_sha256"] != file_sha(role_path)?
    {
        return Err("reference authority changed".into());
    }
    let reference = read_json(reference_path)?;
    let rr = reference_path.parent().unwrap();
    let count = usize_at(&reference["site"]["source_rows"])?;
    let ref_k = checked(rr, &reference, "source-k.f32", count * D)?;
    let ref_v = checked(rr, &reference, "source-v.f32", count * D)?;
    let read_lines = |p: &Path| -> Result<Vec<Value>> {
        std::fs::read_to_string(p)?
            .lines()
            .map(|l| Ok(serde_json::from_str(l)?))
            .collect()
    };
    let source_rows_path = rr.join("source-rows.jsonl");
    let source_row_descriptors: Vec<_> = reference["artifacts"]
        .as_array()
        .ok_or("reference artifacts absent")?
        .iter()
        .filter(|v| v["path"] == "source-rows.jsonl")
        .collect();
    if source_row_descriptors.len() != 1
        || source_row_descriptors[0]["sha256"] != file_sha(&source_rows_path)?
    {
        return Err("reference source-row hash mismatch".into());
    }
    let source_metadata = read_lines(&source_rows_path)?;
    let role_metadata = read_lines(role_path)?;
    let mut means: BTreeMap<(String, String), ReferenceMean> = BTreeMap::new();
    for (row, roles) in source_metadata.iter().zip(&role_metadata) {
        if row["edge_id"] != roles["edge_id"] {
            return Err("reference row order mismatch".into());
        }
        if row["split"] != "train" {
            continue;
        }
        let offset = usize_at(&row["source_offset"])?;
        for (role, positions) in roles["roles"].as_object().ok_or("roles absent")? {
            for position in ids(positions)? {
                let cell = means
                    .entry((
                        roles["template_id"].as_str().unwrap().to_owned(),
                        role.clone(),
                    ))
                    .or_insert((vec![0.; D], vec![0.; D], 0));
                for d in 0..D {
                    cell.0[d] += ref_k[(offset + position) * D + d];
                    cell.1[d] += ref_v[(offset + position) * D + d];
                }
                cell.2 += 1;
            }
        }
    }
    for (k, v, n) in means.values_mut() {
        for x in k.iter_mut().chain(v.iter_mut()) {
            *x /= *n as f32;
        }
    }
    let candidate_ids: Vec<u32> = ids(&e["candidate_token_ids"])?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let selected = ops.select_output_head(&candidate_ids)?;
    let op = plan.layers[24]
        .attention
        .softmax()
        .ok_or("L24 not softmax")?;
    let mut names = vec![
        "natural".to_owned(),
        "noop".to_owned(),
        "exact".to_owned(),
        "identity".to_owned(),
        "zero".to_owned(),
    ];
    for family in ["primary", "sham"] {
        for cell in 0..CELLS {
            names.push(format!("{family}-{cell}"));
        }
    }
    let arms = names.len();
    let width = candidate_ids.len();
    let mut proximal = writer(&output.join("proximal.f32"))?;
    let mut terminal = writer(&output.join("terminal.f32"))?;
    let mut entering = writer(&output.join("before.f32"))?;
    let mut elapsed = writer(&output.join("elapsed-seconds.f32"))?;
    for (row, item) in e["rows"].as_array().unwrap().iter().enumerate() {
        let row_clock = Instant::now();
        let tokens = ids(&item["row"]["prompt"]["token_ids"])?;
        let positions = ids(&item["roles"]["subject_entity"])?;
        let offset = usize_at(&item["source_offset"])?;
        let subject_offset = usize_at(&item["subject_offset"])?;
        let natural_k = vectors(&all_k, offset, tokens.len());
        let natural_v = vectors(&all_v, offset, tokens.len());
        let heads = vectors(&all_heads, row * 8, 8);
        let query = &all_q[row * D..(row + 1) * D];
        let carrier = &before[row * H..(row + 1) * H];
        let mut keys = vec![Vec::new(); tokens.len()];
        let mut values = keys.clone();
        let template = item["row"]["prompt"]["template_id"].as_str().unwrap();
        for (role, indices) in item["roles"].as_object().unwrap() {
            for position in ids(indices)? {
                let cell = means
                    .get(&(template.to_owned(), role.clone()))
                    .ok_or("reference cell missing")?;
                keys[position] = scale(&cell.0, &natural_k[position])?;
                values[position] = if role == "subject_entity" {
                    natural_v[position].clone()
                } else {
                    scale(&cell.1, &natural_v[position])?
                };
            }
        }
        let mut prefix = RowKvState::default();
        {
            let mut s = DecodeSession::over_prepared(plan, ops, backend, &mut prefix)?;
            for &t in &tokens[..tokens.len() - 1] {
                s.step_observed(t as u32, &mut NoopObserver)?;
            }
        }
        put(
            &mut entering,
            &ops.readout_carrier_selected(backend, carrier, &selected)?,
        )?;
        let mut natural_terminal = Vec::new();
        let mut exact_terminal = Vec::new();
        let mut exact_proximal = Vec::new();
        for arm in 0..arms {
            let mut kv = prefix.clone();
            let target = tokens.len() - 1;
            let replacement = if arm < 2 {
                after[row * H..(row + 1) * H].to_vec()
            } else {
                let mut treatment = values.clone();
                for (ordinal, &position) in positions.iter().enumerate() {
                    if arm == 4 {
                        treatment[position] = vec![0.; D];
                    } else if arm >= 5 {
                        let index = ((arm - 5) * subject_rows + subject_offset + ordinal) * D;
                        treatment[position] = predicted[index..index + D].to_vec();
                    }
                }
                let h1 = replay_softmax_source_head(
                    query,
                    &keys,
                    &treatment,
                    op.score_scale,
                    op.logit_softcapping,
                )?;
                let mut mixed = heads.clone();
                mixed[1] = h1.values;
                replay_attention_mixture(plan, ops, backend, 24, carrier, &mixed)?.carrier_after
            };
            let prox = ops.readout_carrier_selected(backend, &replacement, &selected)?;
            let logits = if arm == 0 {
                let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv)?;
                session
                    .step_observed(tokens[target] as u32, &mut NoopObserver)?
                    .logits
                    .ok_or("logits absent")?
            } else {
                let provenance = VectorProvenance::Literal {
                    sha256: vector_sha256(&replacement),
                };
                let interventions = InterventionPlan::none().with(Intervention::replace(
                    Address::new(24, SublayerSite::Attention, [target])?,
                    replacement,
                    provenance,
                )?)?;
                let result = if arm == 2 {
                    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv)?;
                    session.step_intervened(
                        tokens[target] as u32,
                        &mut NoopObserver,
                        &interventions,
                        &HeadInterventionPlan::none(),
                    )?
                } else {
                    let mut session =
                        DecodeSession::over_prepared(plan, &tail_ops, backend, &mut kv)?;
                    let mut observer = TailCarrier::default();
                    let mut result = session.step_from_carrier_intervened(
                        carrier,
                        &mut observer,
                        &interventions,
                        &HeadInterventionPlan::none(),
                    )?;
                    // Natural/noop and full-exact/identity controls below require
                    // this selected-row exit to match the full vocabulary exit.
                    result.logits =
                        Some(ops.readout_carrier_selected(backend, &observer.0, &selected)?);
                    result
                };
                if result.firings.len() != 1 {
                    return Err("intervention firing count differs from one".into());
                }
                result.logits.ok_or("logits absent")?
            };
            let term: Vec<f32> = if arm == 0 || arm == 2 {
                candidate_ids.iter().map(|&i| logits[i as usize]).collect()
            } else {
                logits
            };
            if arm == 0 {
                natural_terminal = term.clone();
            }
            if arm == 1 && !same(&term, &natural_terminal) {
                return Err("natural/noop terminal parity failed".into());
            }
            if arm == 2 {
                exact_terminal = term.clone();
                exact_proximal = prox.clone();
            }
            if arm == 3 && (!same(&term, &exact_terminal) || !same(&prox, &exact_proximal)) {
                return Err("exact/identity parity failed".into());
            }
            put(&mut proximal, &prox)?;
            put(&mut terminal, &term)?;
        }
        put(&mut elapsed, &[row_clock.elapsed().as_secs_f32()])?;
        if (row + 1) % 5 == 0 || row + 1 == N {
            eprintln!("GW-V2 complete replay rows {}/{N} ({arms} arms)", row + 1);
        }
    }
    for f in [&mut proximal, &mut terminal, &mut entering, &mut elapsed] {
        f.flush()?;
    }
    let artifacts = vec![
        artifact(output, "proximal.f32", json!([N, arms, width]))?,
        artifact(output, "terminal.f32", json!([N, arms, width]))?,
        artifact(output, "before.f32", json!([N, width]))?,
        artifact(output, "elapsed-seconds.f32", json!([N]))?,
    ];
    let manifest = json!({"schema":"larql.gwv2.replay.v1","lineage":e["lineage"],"execution_file_sha256":file_sha(binding)?,"capture_file_sha256":file_sha(capture_path)?,"fit_file_sha256":file_sha(fit_path)?,"executable_sha256":file_sha(&std::env::current_exe()?)?,"arms":names,"artifacts":artifacts,"parity_bit_mismatches":0,"intervention_firings":N*(arms-1),"latency_admissible":false,"latency_note":"Elapsed row times are operational diagnostics, not warmed exclusive baseline/candidate/baseline measurements."});
    atomic_json(
        &output.join("replay.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
