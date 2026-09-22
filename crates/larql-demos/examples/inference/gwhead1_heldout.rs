//! Frozen GW-HEAD-1 validation/test arms and canonical downstream execution.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::{replay_attention_heads, replay_attention_mixture},
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    observe::NoopObserver,
};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PREREG_SCHEMA: &str = "larql.gwhead1.preregistration.v1";
const CAPTURE_SCHEMA: &str = "larql.gwhead1.natural-capture.v1";
const SELECTION_SCHEMA: &str = "larql.gwhead1.selection.v1";
const OUTPUT_SCHEMA: &str = "larql.gwhead1.heldout-arms.v1";
const LAYER: usize = 24;
const ROWS: usize = 426;
const HELDOUT_ROWS: usize = 171;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const CANDIDATES: usize = 126;
const FULL_MASK: usize = 255;
const ARMS: [&str; 25] = [
    "I_full",
    "I_ref",
    "I_S_global",
    "I_full_minus_S_global",
    "I_S_relation",
    "I_full_minus_S_relation",
    "I_identity",
    "I_S_head_0",
    "I_S_head_1",
    "I_S_head_2",
    "I_S_head_3",
    "I_S_head_4",
    "I_S_head_5",
    "I_S_head_6",
    "I_S_head_7",
    "I_full_minus_head_0",
    "I_full_minus_head_1",
    "I_full_minus_head_2",
    "I_full_minus_head_3",
    "I_full_minus_head_4",
    "I_full_minus_head_5",
    "I_full_minus_head_6",
    "I_full_minus_head_7",
    "I_zero_S_global",
    "I_only_S_global",
];

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

fn read_f32(root: &Path, manifest: &Value, name: &str, values: usize) -> Result<Vec<f32>> {
    let descriptor = artifact(manifest, name)?;
    let path = root.join(name);
    if descriptor["sha256"].as_str() != Some(&file_sha(&path)?)
        || std::fs::metadata(&path)?.len() != (values * 4) as u64
    {
        return Err(format!("invalid capture artifact {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    File::open(&path)?.read_to_end(&mut bytes)?;
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

fn mask(value: &Value) -> Result<usize> {
    let mut result = 0usize;
    for head in value.as_array().ok_or("selected heads missing")? {
        let head = head.as_u64().ok_or("bad selected head")? as usize;
        if head >= HEADS || result & (1 << head) != 0 {
            return Err("invalid selected head set".into());
        }
        result |= 1 << head;
    }
    if result == 0 {
        return Err("selected head set is empty".into());
    }
    Ok(result)
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

fn bits_differ(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
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

/// `--gwhead1-heldout CONTAINER PREREG SELECTION CAPTURE_MANIFEST OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("Usage: observatory_record --gwhead1-heldout CONTAINER PREREG SELECTION CAPTURE_MANIFEST OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let selection_path = Path::new(&args[2]);
    let capture_path = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    std::fs::create_dir_all(output)?;
    let output_manifest = output.join("heldout-arms-manifest.json");
    if output_manifest.exists() {
        return Err(format!(
            "GW-HEAD-1 held-out arms are already sealed at {}",
            output.display()
        )
        .into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    let selection: Value = serde_json::from_slice(&std::fs::read(selection_path)?)?;
    let capture: Value = serde_json::from_slice(&std::fs::read(capture_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA
        || prereg["status"] != "frozen_pre_execution"
        || selection["schema"] != SELECTION_SCHEMA
        || selection["status"] != "frozen_pre_heldout"
        || capture["schema"] != CAPTURE_SCHEMA
        || capture["status"] != "natural_capture_complete_pre_subset_search"
        || selection["preregistration_sha256"] != prereg["preregistration_sha256"]
        || capture["preregistration_sha256"] != prereg["preregistration_sha256"]
    {
        return Err("GW-HEAD-1 held-out authorities are not frozen together".into());
    }
    let prereg_identity = prereg["preregistration_sha256"]
        .as_str()
        .ok_or("preregistration identity missing")?;
    let selection_identity = selection["selection_sha256"]
        .as_str()
        .ok_or("selection identity missing")?;
    let global_mask = mask(&selection["scopes"]["global"]["selected"]["heads"])?;
    let relation_masks = ["capital", "currency", "language", "hypernym"]
        .into_iter()
        .map(|relation| {
            Ok((
                relation.to_string(),
                mask(&selection["scopes"][relation]["selected"]["heads"])?,
            ))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;

    let capture_root = capture_path
        .parent()
        .ok_or("capture manifest has no parent")?;
    let all_heads = read_f32(
        capture_root,
        &capture,
        "head-values.f32",
        ROWS * HEADS * HEAD_DIM,
    )?;
    let all_before = read_f32(capture_root, &capture, "carrier-before.f32", ROWS * HIDDEN)?;
    let all_after = read_f32(capture_root, &capture, "carrier-after.f32", ROWS * HIDDEN)?;
    let all_after_logits = read_f32(
        capture_root,
        &capture,
        "candidate-after-logits.f32",
        ROWS * CANDIDATES,
    )?;
    let all_terminal_logits = read_f32(
        capture_root,
        &capture,
        "candidate-terminal-logits.f32",
        ROWS * CANDIDATES,
    )?;
    let rows_path = capture_root.join("natural-rows.jsonl");
    if artifact(&capture, "natural-rows.jsonl")?["sha256"].as_str() != Some(&file_sha(&rows_path)?)
    {
        return Err("natural rows hash mismatch".into());
    }
    let rows: Vec<Value> = BufReader::new(File::open(&rows_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if rows.len() != ROWS {
        return Err("natural row count changed".into());
    }
    let input_rows_path = {
        let declared = Path::new(
            prereg["authorities"]["input_rows"]["path"]
                .as_str()
                .ok_or("input rows path missing")?,
        );
        if declared.is_absolute() {
            declared.to_path_buf()
        } else {
            prereg_path
                .parent()
                .ok_or("preregistration has no parent")?
                .join(declared)
        }
    };
    if prereg["authorities"]["input_rows"]["sha256"].as_str() != Some(&file_sha(&input_rows_path)?)
    {
        return Err("input rows hash mismatch".into());
    }
    let input_rows: Vec<Value> = BufReader::new(File::open(&input_rows_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if input_rows.len() != ROWS {
        return Err("input row count changed".into());
    }

    eprintln!("Preparing the frozen GW-HEAD-1 held-out image...");
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
        return Err("held-out image differs from frozen authorities".into());
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

    let proximal_path = output.join("heldout-proximal-logits.f32");
    let carrier_path = output.join("heldout-carriers.f32");
    let terminal_path = output.join("heldout-terminal-logits.f32");
    let rows_out_path = output.join("heldout-rows.jsonl");
    if [
        &proximal_path,
        &carrier_path,
        &terminal_path,
        &rows_out_path,
    ]
    .iter()
    .any(|path| path.exists())
    {
        return Err("GW-HEAD-1 held-out output is partial or already exists".into());
    }
    let mut proximal_out = writer(&proximal_path)?;
    let mut carrier_out = writer(&carrier_path)?;
    let mut terminal_out = writer(&terminal_path)?;
    let mut rows_out = writer(&rows_out_path)?;

    let mut heldout_index = 0usize;
    let mut full_carrier_mismatches = 0usize;
    let mut full_proximal_mismatches = 0usize;
    let mut full_terminal_mismatches = 0usize;
    let mut identity_terminal_mismatches = 0usize;
    let mut intervention_firings = 0usize;
    let mut max_reference_norm_relative_error = 0.0f64;
    for (original_row, row) in rows.iter().enumerate() {
        if row["split"] == "train" {
            continue;
        }
        if row["split"] != "validation" && row["split"] != "test" {
            return Err("unknown held-out split".into());
        }
        let edge_id = row["edge_id"].as_str().ok_or("edge ID missing")?;
        let relation = row["semantic_edge"]["relation"]
            .as_str()
            .ok_or("relation missing")?;
        let family = row["semantic_edge"]["prompt_semantic_family"]
            .as_str()
            .ok_or("prompt family missing")?;
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
        let before = &all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN];
        let donor_rows: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate["split"] == "train"
                    && candidate["semantic_edge"]["relation"].as_str() == Some(relation)
                    && candidate["semantic_edge"]["prompt_semantic_family"].as_str() == Some(family)
            })
            .map(|(index, _)| index)
            .collect();
        if donor_rows.is_empty() {
            return Err(format!("{edge_id}: no frozen train reference donors").into());
        }
        let mut reference = vec![vec![0.0f32; HEAD_DIM]; HEADS];
        for donor in &donor_rows {
            for (head, mean) in reference.iter_mut().enumerate() {
                let start = (*donor * HEADS + head) * HEAD_DIM;
                for (value, donor_value) in mean.iter_mut().zip(&all_heads[start..start + HEAD_DIM])
                {
                    *value += *donor_value / donor_rows.len() as f32;
                }
            }
        }
        let unscaled = replay_attention_heads(&plan, &ops, &backend, LAYER, before, &reference)?;
        for (head, reference_head) in reference.iter_mut().enumerate() {
            if unscaled.contribution_norms[head] <= 0.0 || natural_norms[head] <= 0.0 {
                return Err(format!("{edge_id}: zero reference/natural head norm").into());
            }
            let scale = natural_norms[head] / unscaled.contribution_norms[head];
            for value in reference_head {
                *value *= scale as f32;
            }
        }
        let scaled = replay_attention_heads(&plan, &ops, &backend, LAYER, before, &reference)?;
        for (&natural_norm, &scaled_norm) in natural_norms.iter().zip(&scaled.contribution_norms) {
            max_reference_norm_relative_error = max_reference_norm_relative_error
                .max(((natural_norm - scaled_norm) / natural_norm).abs());
        }
        if max_reference_norm_relative_error > 1e-5 {
            return Err(format!("{edge_id}: held-out contribution norm match failed").into());
        }

        let relation_mask = *relation_masks
            .get(relation)
            .ok_or("relation subset missing")?;
        let mut masks = vec![
            Some(FULL_MASK),
            Some(0),
            Some(global_mask),
            Some(FULL_MASK ^ global_mask),
            Some(relation_mask),
            Some(FULL_MASK ^ relation_mask),
            Some(FULL_MASK),
        ];
        masks.extend((0..HEADS).map(|head| Some(1 << head)));
        masks.extend((0..HEADS).map(|head| Some(FULL_MASK ^ (1 << head))));
        masks.extend([None, None]);
        if masks.len() != ARMS.len() {
            return Err("GW-HEAD-1 held-out arm construction changed".into());
        }
        let mut local_carriers = Vec::with_capacity(ARMS.len());
        let mut proximal_logits = Vec::with_capacity(ARMS.len());
        for (arm, arm_mask) in masks.iter().enumerate() {
            let mixture: Vec<Vec<f32>> = (0..HEADS)
                .map(|head| match (arm, arm_mask) {
                    (23, None) => {
                        if global_mask & (1 << head) != 0 {
                            vec![0.0; HEAD_DIM]
                        } else {
                            natural[head].clone()
                        }
                    }
                    (24, None) => {
                        if global_mask & (1 << head) != 0 {
                            natural[head].clone()
                        } else {
                            vec![0.0; HEAD_DIM]
                        }
                    }
                    (_, Some(mask)) => {
                        if mask & (1 << head) != 0 {
                            natural[head].clone()
                        } else {
                            reference[head].clone()
                        }
                    }
                    _ => unreachable!("zero controls are the final two frozen arms"),
                })
                .collect();
            let replay = replay_attention_mixture(&plan, &ops, &backend, LAYER, before, &mixture)?;
            proximal_logits.push(ops.readout_carrier_selected(
                &backend,
                &replay.carrier_after,
                &selected,
            )?);
            local_carriers.push(replay.carrier_after);
        }
        full_carrier_mismatches += bits_differ(
            &local_carriers[0],
            &all_after[original_row * HIDDEN..(original_row + 1) * HIDDEN],
        );
        full_proximal_mismatches += bits_differ(
            &proximal_logits[0],
            &all_after_logits[original_row * CANDIDATES..(original_row + 1) * CANDIDATES],
        );

        if input_rows[original_row]["edge_id"] != row["edge_id"] {
            return Err(format!("{edge_id}: input/natural row order changed").into());
        }
        let tokens: Vec<u32> = input_rows[original_row]["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt token IDs missing")?
            .iter()
            .map(|token| token.as_u64().map(|v| v as u32).ok_or("bad prompt token"))
            .collect::<std::result::Result<_, _>>()?;
        let final_token = *tokens.last().ok_or("empty prompt")?;
        let position = tokens.len() - 1;
        let mut prefix = RowKvState::default();
        {
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut prefix)?;
            for &token in &tokens[..position] {
                session.step_observed(token, &mut NoopObserver)?;
            }
        }
        let mut terminal_logits = Vec::with_capacity(ARMS.len());
        for (arm, carrier) in local_carriers.iter().enumerate() {
            let mut kv = prefix.clone();
            let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
            let full = if arm == 0 {
                session
                    .step_observed(final_token, &mut NoopObserver)?
                    .logits
                    .ok_or("missing I_full terminal logits")?
            } else {
                let intervention = replacement_plan(position, carrier.clone())?;
                let result = session.step_intervened(
                    final_token,
                    &mut NoopObserver,
                    &intervention,
                    &HeadInterventionPlan::none(),
                )?;
                if result.firings.len() != 1
                    || result.firings[0].layer != LAYER
                    || result.firings[0].site != SublayerSite::Attention
                    || result.firings[0].position != position
                {
                    return Err(format!("{edge_id}: intervention did not fire exactly once").into());
                }
                intervention_firings += 1;
                result.logits.ok_or("missing intervened terminal logits")?
            };
            terminal_logits.push(candidate_logits(&full, &token_ids)?);
        }
        full_terminal_mismatches += bits_differ(
            &terminal_logits[0],
            &all_terminal_logits[original_row * CANDIDATES..(original_row + 1) * CANDIDATES],
        );
        identity_terminal_mismatches += bits_differ(&terminal_logits[0], &terminal_logits[6]);
        if full_carrier_mismatches != 0
            || full_proximal_mismatches != 0
            || full_terminal_mismatches != 0
            || identity_terminal_mismatches != 0
        {
            return Err(format!("{edge_id}: full or identity parity failed").into());
        }

        for arm in 0..ARMS.len() {
            write_f32(&mut proximal_out, &proximal_logits[arm])?;
            write_f32(&mut carrier_out, &local_carriers[arm])?;
            write_f32(&mut terminal_out, &terminal_logits[arm])?;
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "heldout_row": heldout_index,
                "original_row": original_row,
                "edge_id": edge_id,
                "split": row["split"],
                "relation": relation,
                "prompt_semantic_family": family,
                "global_mask": global_mask,
                "relation_mask": relation_mask,
                "arms": ARMS,
                "arm_masks": masks,
                "train_reference_donors": donor_rows.len(),
            }),
        )?;
        heldout_index += 1;
        if heldout_index.is_multiple_of(25) || heldout_index == HELDOUT_ROWS {
            eprintln!("GW-HEAD-1 held-out arms {heldout_index}/{HELDOUT_ROWS}");
        }
    }
    if heldout_index != HELDOUT_ROWS {
        return Err(format!("GW-HEAD-1 yielded {heldout_index} held-out rows").into());
    }
    for output in [
        &mut proximal_out,
        &mut carrier_out,
        &mut terminal_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop(proximal_out);
    drop(carrier_out);
    drop(terminal_out);
    drop(rows_out);

    let artifacts = [
        (
            &proximal_path,
            "f32-le",
            json!([HELDOUT_ROWS, ARMS.len(), CANDIDATES]),
        ),
        (
            &carrier_path,
            "f32-le",
            json!([HELDOUT_ROWS, ARMS.len(), HIDDEN]),
        ),
        (
            &terminal_path,
            "f32-le",
            json!([HELDOUT_ROWS, ARMS.len(), CANDIDATES]),
        ),
        (&rows_out_path, "jsonl", json!([HELDOUT_ROWS])),
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
    let unique_masks: BTreeSet<_> = relation_masks.values().copied().collect();
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "heldout_execution_complete_pre_adjudication",
        "preregistration_sha256": prereg_identity,
        "selection_sha256": selection_identity,
        "natural_capture_sha256": file_sha(capture_path)?,
        "authorities": {
            "container_identity": container_identity,
            "plan_sha256": plan_identity,
            "backend": backend.name(),
            "execution_fingerprint": ExecutionProvenance::of(&ops).fingerprint(),
            "prepared_head_representation": selected.representation(),
        },
        "arms": ARMS,
        "global_mask": global_mask,
        "relation_masks": relation_masks,
        "unique_relation_masks": unique_masks,
        "shape": {"rows": HELDOUT_ROWS, "arms": ARMS.len(), "candidates": CANDIDATES, "hidden": HIDDEN},
        "reference": {
            "source": "train rows only",
            "max_relative_norm_error": max_reference_norm_relative_error,
            "threshold": 1e-5,
        },
        "parity": {
            "full_carrier_bit_mismatches": full_carrier_mismatches,
            "full_proximal_logit_bit_mismatches": full_proximal_mismatches,
            "full_terminal_logit_bit_mismatches": full_terminal_mismatches,
            "identity_terminal_logit_bit_mismatches": identity_terminal_mismatches,
            "intervention_firings": intervention_firings,
            "expected_intervention_firings": HELDOUT_ROWS * (ARMS.len() - 1),
        },
        "artifacts": artifacts,
        "held_out_outcomes_used_for_selection": false,
        "adjudication_performed": false,
    });
    atomic_json(&output_manifest, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
