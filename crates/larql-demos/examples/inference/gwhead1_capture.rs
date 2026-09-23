//! Frozen GW-HEAD-1 natural L24 head capture and reconstruction witness.
use super::gwsup1_readout::{diagnostic, file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::replay_attention_heads,
    observe::{AttentionHeadRecord, NoopObserver},
};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};

const PREREG_SCHEMA: &str = "larql.gwhead1.preregistration.v1";
const CANDIDATE_SCHEMA: &str = "larql.gwsup1.candidates.v1";
const INPUT_SCHEMA: &str = "larql.gw0.input-manifest.v1";
const OUTPUT_SCHEMA: &str = "larql.gwhead1.natural-capture.v1";
const LAYER: usize = 24;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;

#[derive(Default)]
struct L24Capture {
    target_position: usize,
    position: usize,
    last_after: Option<Vec<f32>>,
    before: Option<Vec<f32>>,
    heads: Vec<Option<Vec<f32>>>,
    raw: Option<Vec<f32>>,
    delta: Option<Vec<f32>>,
    after: Option<Vec<f32>>,
    error: Option<String>,
}

impl L24Capture {
    fn new(target_position: usize) -> Self {
        Self {
            target_position,
            heads: vec![None; HEADS],
            ..Self::default()
        }
    }

    fn finish(self) -> Result<Captured> {
        if let Some(error) = self.error {
            return Err(error.into());
        }
        let heads = self
            .heads
            .into_iter()
            .enumerate()
            .map(|(head, values)| values.ok_or_else(|| format!("missing L24 head {head}").into()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Captured {
            before: self.before.ok_or("missing L24 entering carrier")?,
            heads,
            raw: self.raw.ok_or("missing L24 raw attention output")?,
            delta: self.delta.ok_or("missing L24 applied attention delta")?,
            after: self.after.ok_or("missing L24 after carrier")?,
        })
    }
}

struct Captured {
    before: Vec<f32>,
    heads: Vec<Vec<f32>>,
    raw: Vec<f32>,
    delta: Vec<f32>,
    after: Vec<f32>,
}

impl StepObserver for L24Capture {
    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
            if position == self.target_position {
                self.last_after = None;
            }
        }
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer != LAYER || record.position != self.target_position {
            return;
        }
        if record.head >= HEADS
            || record.values.len() != HEAD_DIM
            || record.values.iter().any(|value| !value.is_finite())
            || self.heads[record.head].is_some()
        {
            self.error = Some("invalid or duplicate L24 head capture".into());
            return;
        }
        self.heads[record.head] = Some(record.values.to_vec());
    }

    fn attention_output(&mut self, layer: usize, position: usize, values: &[f32]) {
        if layer == LAYER && position == self.target_position {
            if self.raw.is_some() || values.iter().any(|value| !value.is_finite()) {
                self.error = Some("invalid or duplicate L24 raw output".into());
                return;
            }
            self.before = self.last_after.clone();
            self.raw = Some(values.to_vec());
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.position != self.target_position {
            return;
        }
        if record.layer == LAYER && record.site == SublayerSite::Attention {
            if self.delta.is_some() || self.after.is_some() {
                self.error = Some("duplicate L24 carrier write".into());
            }
            self.delta = Some(record.delta.to_vec());
            self.after = Some(record.after.to_vec());
        }
        let mut next = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for value in &mut next {
                *value *= scale;
            }
        }
        self.last_after = Some(next);
    }
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

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = a
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

fn candidate_logits(full: &[f32], token_ids: &[u32]) -> Result<Vec<f32>> {
    token_ids
        .iter()
        .map(|&token| {
            full.get(token as usize)
                .copied()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("candidate token {token} is outside finite logits").into())
        })
        .collect()
}

/// `--gwhead1-capture CONTAINER PREREG CANDIDATES INPUT_MANIFEST OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("Usage: observatory_record --gwhead1-capture CONTAINER PREREG CANDIDATES INPUT_MANIFEST OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let candidates_path = Path::new(&args[2]);
    let input_manifest_path = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("natural-capture-manifest.json");
    if manifest_path.exists() {
        return Err(format!(
            "GW-HEAD-1 capture is already sealed at {}",
            output.display()
        )
        .into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA || prereg["status"] != "frozen_pre_execution" {
        return Err("GW-HEAD-1 preregistration is not frozen".into());
    }
    let prereg_identity = prereg["preregistration_sha256"]
        .as_str()
        .ok_or("GW-HEAD-1 identity missing")?;
    if prereg["authorities"]["input_manifest"]["sha256"].as_str()
        != Some(&file_sha(input_manifest_path)?)
    {
        return Err("input manifest differs from GW-HEAD-1".into());
    }
    let input_manifest: Value = serde_json::from_slice(&std::fs::read(input_manifest_path)?)?;
    if input_manifest["schema"] != INPUT_SCHEMA {
        return Err("not the frozen GW-0 input manifest".into());
    }
    let input_rows_path = input_manifest_path
        .parent()
        .ok_or("input manifest has no parent")?
        .join(
            input_manifest["rows"]["path"]
                .as_str()
                .ok_or("input rows path missing")?,
        );
    if prereg["authorities"]["input_rows"]["sha256"].as_str() != Some(&file_sha(&input_rows_path)?)
    {
        return Err("input rows differ from GW-HEAD-1".into());
    }

    let candidates: Value = serde_json::from_slice(&std::fs::read(candidates_path)?)?;
    if candidates["schema"] != CANDIDATE_SCHEMA
        || prereg["authorities"]["candidates"]["sha256"].as_str()
            != Some(&file_sha(candidates_path)?)
        || prereg["authorities"]["candidates"]["identity_sha256"]
            != candidates["candidate_identity_sha256"]
    {
        return Err("candidate authority does not match GW-HEAD-1".into());
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
        return Err("GW-HEAD-1 candidate vocabulary no longer has 126 tokens".into());
    }

    eprintln!("Preparing the frozen GW-HEAD-1 production image...");
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
        return Err("prepared image differs from the frozen GW-HEAD-1 authority".into());
    }
    let layer = plan.layers.get(LAYER).ok_or("model has no L24")?;
    let op = layer.attention.softmax().ok_or("L24 is not softmax")?;
    if op.num_q_heads != HEADS
        || op.head_dim != HEAD_DIM
        || op.output_gate.is_some()
        || op.o_bias.is_some()
        || layer.post_attention_norm.is_none()
    {
        return Err(
            "L24 no longer has the frozen ungated, bias-free, post-norm head geometry".into(),
        );
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let selected = ops.select_output_head(&token_ids)?;

    let paths = [
        output.join("head-values.f32"),
        output.join("carrier-before.f32"),
        output.join("raw-attention-output.f32"),
        output.join("applied-delta.f32"),
        output.join("carrier-after.f32"),
        output.join("candidate-before-logits.f32"),
        output.join("candidate-after-logits.f32"),
        output.join("candidate-terminal-logits.f32"),
        output.join("natural-rows.jsonl"),
    ];
    if paths.iter().any(|path| path.exists()) {
        return Err("GW-HEAD-1 capture output is partial or already exists".into());
    }
    let mut heads_out = writer(&paths[0])?;
    let mut before_out = writer(&paths[1])?;
    let mut raw_out = writer(&paths[2])?;
    let mut delta_out = writer(&paths[3])?;
    let mut after_out = writer(&paths[4])?;
    let mut before_logits_out = writer(&paths[5])?;
    let mut after_logits_out = writer(&paths[6])?;
    let mut terminal_logits_out = writer(&paths[7])?;
    let mut rows_out = writer(&paths[8])?;

    let mut rows = 0usize;
    let mut prompt_positions = 0usize;
    let mut bit_mismatches = 0usize;
    let mut max_raw_relative_l2 = 0.0f64;
    let mut max_delta_relative_l2 = 0.0f64;
    let mut max_after_relative_l2 = 0.0f64;
    let mut max_head_reconstruction_relative_l2 = 0.0f64;
    for line in BufReader::new(File::open(&input_rows_path)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let edge_id = row["edge_id"].as_str().ok_or("edge ID missing")?;
        let tokens: Vec<u32> = row["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt token IDs missing")?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .map(|v| v as u32)
                    .ok_or("bad prompt token ID")
            })
            .collect::<std::result::Result<_, _>>()?;
        let target_position = tokens.len().checked_sub(1).ok_or("empty prompt")?;
        if row["capture_request"]["position"].as_u64() != Some(target_position as u64) {
            return Err(format!("{edge_id}: capture position changed").into());
        }

        let mut capture = L24Capture::new(target_position);
        let mut observed_logits = Vec::with_capacity(tokens.len());
        {
            let mut kv = RowKvState::default();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            for &token in &tokens {
                observed_logits.push(
                    session
                        .step_observed(token, &mut capture)?
                        .logits
                        .ok_or("missing observed logits")?,
                );
            }
        }
        {
            let mut kv = RowKvState::default();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            for (&token, observed) in tokens.iter().zip(&observed_logits) {
                let control = session
                    .step_observed(token, &mut NoopObserver)?
                    .logits
                    .ok_or("missing control logits")?;
                if observed.len() != control.len() {
                    return Err(format!("{edge_id}: observed/control vocab mismatch").into());
                }
                bit_mismatches += observed
                    .iter()
                    .zip(&control)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                prompt_positions += 1;
            }
        }
        if bit_mismatches != 0 {
            return Err(format!("{edge_id}: head capture changed production logits").into());
        }
        let captured = capture.finish()?;
        let replay = replay_attention_heads(
            &plan,
            &ops,
            &backend,
            LAYER,
            &captured.before,
            &captured.heads,
        )?;
        let raw_error = relative_l2(&captured.raw, &replay.raw_attention_output);
        let delta_error = relative_l2(&captured.delta, &replay.applied_delta);
        let after_error = relative_l2(&captured.after, &replay.carrier_after);
        max_raw_relative_l2 = max_raw_relative_l2.max(raw_error);
        max_delta_relative_l2 = max_delta_relative_l2.max(delta_error);
        max_after_relative_l2 = max_after_relative_l2.max(after_error);
        max_head_reconstruction_relative_l2 =
            max_head_reconstruction_relative_l2.max(replay.raw_reconstruction_relative_l2);
        if [
            raw_error,
            delta_error,
            after_error,
            replay.raw_reconstruction_relative_l2,
        ]
        .iter()
        .any(|error| !error.is_finite() || *error > 1e-5)
        {
            return Err(format!("{edge_id}: L24 replay reconstruction failed").into());
        }

        let before_logits = ops.readout_carrier_selected(&backend, &captured.before, &selected)?;
        let after_logits =
            ops.readout_carrier_selected(&backend, &replay.carrier_after, &selected)?;
        let terminal_logits = candidate_logits(
            observed_logits.last().ok_or("missing terminal logits")?,
            &token_ids,
        )?;
        for head in &captured.heads {
            write_f32(&mut heads_out, head)?;
        }
        write_f32(&mut before_out, &captured.before)?;
        write_f32(&mut raw_out, &captured.raw)?;
        write_f32(&mut delta_out, &captured.delta)?;
        write_f32(&mut after_out, &captured.after)?;
        write_f32(&mut before_logits_out, &before_logits)?;
        write_f32(&mut after_logits_out, &after_logits)?;
        write_f32(&mut terminal_logits_out, &terminal_logits)?;
        write_json_line(
            &mut rows_out,
            &json!({
                "row_index": rows,
                "edge_id": edge_id,
                "split": row["split"],
                "semantic_edge": row["semantic_edge"],
                "control_ids": row["control_ids"],
                "capture_position": target_position,
                "head_contribution_norms": replay.contribution_norms,
                "reconstruction": {
                    "raw_relative_l2": raw_error,
                    "head_sum_relative_l2": replay.raw_reconstruction_relative_l2,
                    "applied_delta_relative_l2": delta_error,
                    "carrier_after_relative_l2": after_error,
                },
                "candidate_before": diagnostic(&before_logits),
                "candidate_after": diagnostic(&after_logits),
                "candidate_terminal": diagnostic(&terminal_logits),
            }),
        )?;
        rows += 1;
        if rows.is_multiple_of(25) || rows == 426 {
            eprintln!("GW-HEAD-1 natural capture {rows}/426");
        }
    }
    if rows != 426 {
        return Err(format!("GW-HEAD-1 input yielded {rows} rows, expected 426").into());
    }
    for output in [
        &mut heads_out,
        &mut before_out,
        &mut raw_out,
        &mut delta_out,
        &mut after_out,
        &mut before_logits_out,
        &mut after_logits_out,
        &mut terminal_logits_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop(heads_out);
    drop(before_out);
    drop(raw_out);
    drop(delta_out);
    drop(after_out);
    drop(before_logits_out);
    drop(after_logits_out);
    drop(terminal_logits_out);
    drop(rows_out);

    let artifacts = paths
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path.file_name().and_then(|name| name.to_str()).ok_or("bad artifact path")?,
                "bytes": std::fs::metadata(path)?.len(),
                "sha256": file_sha(path)?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "natural_capture_complete_pre_subset_search",
        "preregistration_sha256": prereg_identity,
        "candidate_identity_sha256": candidates["candidate_identity_sha256"],
        "authorities": {
            "container_identity": container_identity,
            "plan_sha256": plan_identity,
            "backend": backend.name(),
            "execution_fingerprint": ExecutionProvenance::of(&ops).fingerprint(),
            "prepared_head_representation": selected.representation(),
        },
        "shape": {
            "rows": rows,
            "heads": [rows, HEADS, HEAD_DIM],
            "carriers": [rows, ops.hidden()],
            "candidate_logits": [rows, token_ids.len()],
        },
        "order": "frozen input row; ascending head ID; component dimension",
        "candidate_token_ids": token_ids,
        "artifacts": artifacts,
        "parity": {
            "prompt_positions": prompt_positions,
            "full_logit_bit_mismatches": bit_mismatches,
        },
        "reconstruction": {
            "threshold": 1e-5,
            "max_raw_relative_l2": max_raw_relative_l2,
            "max_head_sum_relative_l2": max_head_reconstruction_relative_l2,
            "max_applied_delta_relative_l2": max_delta_relative_l2,
            "max_carrier_after_relative_l2": max_after_relative_l2,
            "result": "pass",
        },
        "held_out_outcomes_used_for_selection": false,
        "subset_search_performed": false,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
