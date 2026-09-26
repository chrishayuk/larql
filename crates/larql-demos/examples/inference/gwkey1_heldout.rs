//! Frozen GW-KEY-1 held-out K/V source-role arms and downstream execution.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::{replay_attention_mixture, replay_softmax_source_head},
    intervene::{vector_sha256, Address, Intervention, InterventionPlan, VectorProvenance},
    intervene_heads::HeadInterventionPlan,
    observe::NoopObserver,
};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PREREG_SCHEMA: &str = "larql.gwkey1.preregistration.v1";
const SOURCE_SCHEMA: &str = "larql.gwkey1.source-capture.v1";
const HEAD_SCHEMA: &str = "larql.gwhead1.natural-capture.v1";
const SELECTION_SCHEMA: &str = "larql.gwkey1.selection.v1";
const OUTPUT_SCHEMA: &str = "larql.gwkey1.heldout-arms.v1";
const LAYER: usize = 24;
const HEAD: usize = 1;
const ROWS: usize = 426;
const HELDOUT_ROWS: usize = 171;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const CANDIDATES: usize = 126;
const ROLES: usize = 6;
const FULL: usize = 63;
const ARMS: [&str; 8] = [
    "I_full",
    "I_identity",
    "V_I_ref",
    "V_I_S",
    "V_I_full_minus_S",
    "joint_I_ref",
    "joint_I_S",
    "joint_I_full_minus_S",
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
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if result.len() != values || result.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid f32 values in {}", path.display()).into());
    }
    Ok(result)
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

fn resolve(root: &Path, value: &Value, key: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(value[key].as_str().ok_or("authority path missing")?);
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
        return Err("GW-KEY-1 held-out reference has a zero norm".into());
    }
    let scale = target_norm / direction_norm;
    Ok(direction
        .iter()
        .map(|value| *value * scale as f32)
        .collect())
}

fn bits_differ(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
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

fn selected_masks(selection: &Value, family: &str) -> Result<(usize, usize)> {
    let selected = &selection["families"][family]["selected"];
    let key = selected["K_mask"]
        .as_u64()
        .ok_or("selected K mask missing")? as usize;
    let value = selected["V_mask"]
        .as_u64()
        .ok_or("selected V mask missing")? as usize;
    if key > FULL || value > FULL {
        return Err("selected GW-KEY-1 source mask is invalid".into());
    }
    Ok((key, value))
}

/// `--gwkey1-heldout CONTAINER PREREG SELECTION SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err("Usage: observatory_record --gwkey1-heldout CONTAINER PREREG SELECTION SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let selection_path = Path::new(&args[2]);
    let source_path = Path::new(&args[3]);
    let head_path = Path::new(&args[4]);
    let output = Path::new(&args[5]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("heldout-arms-manifest.json");
    if manifest_path.exists() {
        return Err("GW-KEY-1 held-out arms are already sealed".into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    let selection: Value = serde_json::from_slice(&std::fs::read(selection_path)?)?;
    let source: Value = serde_json::from_slice(&std::fs::read(source_path)?)?;
    let head: Value = serde_json::from_slice(&std::fs::read(head_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA
        || prereg["status"] != "frozen_pre_execution"
        || selection["schema"] != SELECTION_SCHEMA
        || selection["status"] != "frozen_pre_heldout"
        || source["schema"] != SOURCE_SCHEMA
        || source["status"] != "natural_source_capture_complete_pre_search"
        || head["schema"] != HEAD_SCHEMA
        || head["status"] != "natural_capture_complete_pre_subset_search"
        || selection["preregistration_sha256"] != prereg["preregistration_sha256"]
        || source["preregistration_sha256"] != prereg["preregistration_sha256"]
        || source["gwhead1_natural_capture_sha256"] != file_sha(head_path)?
    {
        return Err("GW-KEY-1 held-out authorities are inadmissible".into());
    }
    let selection_root = selection_path.parent().ok_or("selection has no parent")?;
    for name in ["preregistration", "train_source_metrics"] {
        let authority = &selection["authorities"][name];
        let path = resolve(selection_root, authority, "path")?;
        if file_sha(&path)? != authority["sha256"] {
            return Err(format!("GW-KEY-1 selection authority {name} changed").into());
        }
    }
    let prereg_root = prereg_path
        .parent()
        .ok_or("preregistration has no parent")?;
    let roles_path = resolve(prereg_root, &prereg["roles"]["artifact"], "path")?;
    let input_path = resolve(prereg_root, &prereg["authorities"]["input_rows"], "path")?;
    if file_sha(&roles_path)? != prereg["roles"]["artifact"]["sha256"]
        || file_sha(&input_path)? != prereg["authorities"]["input_rows"]["sha256"]
    {
        return Err("GW-KEY-1 frozen role or input rows changed".into());
    }
    let role_rows: Vec<Value> = BufReader::new(File::open(&roles_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    let input_rows: Vec<Value> = BufReader::new(File::open(&input_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;

    let source_root = source_path.parent().ok_or("source capture has no parent")?;
    let head_root = head_path.parent().ok_or("head capture has no parent")?;
    let source_rows = jsonl(source_root, &source, "source-rows.jsonl")?;
    let head_rows = jsonl(head_root, &head, "natural-rows.jsonl")?;
    if role_rows.len() != ROWS
        || input_rows.len() != ROWS
        || source_rows.len() != ROWS
        || head_rows.len() != ROWS
    {
        return Err("GW-KEY-1 held-out authority row count changed".into());
    }
    let source_count = source["site"]["source_rows"]
        .as_u64()
        .ok_or("source row count missing")? as usize;
    let all_q = read_f32(source_root, &source, "natural-q.f32", ROWS * HEAD_DIM)?;
    let all_k = read_f32(
        source_root,
        &source,
        "source-k.f32",
        source_count * HEAD_DIM,
    )?;
    let all_v = read_f32(
        source_root,
        &source,
        "source-v.f32",
        source_count * HEAD_DIM,
    )?;
    let all_heads = read_f32(head_root, &head, "head-values.f32", ROWS * HEADS * HEAD_DIM)?;
    let all_before = read_f32(head_root, &head, "carrier-before.f32", ROWS * HIDDEN)?;
    let all_after = read_f32(head_root, &head, "carrier-after.f32", ROWS * HIDDEN)?;
    let all_after_logits = read_f32(
        head_root,
        &head,
        "candidate-after-logits.f32",
        ROWS * CANDIDATES,
    )?;
    let all_terminal_logits = read_f32(
        head_root,
        &head,
        "candidate-terminal-logits.f32",
        ROWS * CANDIDATES,
    )?;
    let role_names: Vec<&str> = prereg["roles"]["ordered_vocabulary"]
        .as_array()
        .ok_or("role vocabulary missing")?
        .iter()
        .map(|value| value.as_str().ok_or("bad role name"))
        .collect::<std::result::Result<_, _>>()?;
    if role_names.len() != ROLES {
        return Err("GW-KEY-1 role vocabulary changed".into());
    }

    let vectors = |all: &[f32], start: usize, rows: usize| -> Vec<Vec<f32>> {
        (0..rows)
            .map(|index| {
                let base = (start + index) * HEAD_DIM;
                all[base..base + HEAD_DIM].to_vec()
            })
            .collect()
    };
    let roles_for = |index: usize, count: usize| -> Result<Vec<usize>> {
        let mut result = vec![usize::MAX; count];
        for (role, name) in role_names.iter().enumerate() {
            for position in role_rows[index]["roles"][name]
                .as_array()
                .ok_or("source role positions missing")?
            {
                let position = position.as_u64().ok_or("bad source position")? as usize;
                if position >= count || result[position] != usize::MAX {
                    return Err("GW-KEY-1 held-out role map is not disjoint".into());
                }
                result[position] = role;
            }
        }
        if result.contains(&usize::MAX) {
            return Err("GW-KEY-1 held-out role map is not exhaustive".into());
        }
        Ok(result)
    };

    type ReferenceCell = (Vec<f32>, Vec<f32>, usize);
    let mut means: BTreeMap<(String, usize), ReferenceCell> = BTreeMap::new();
    for index in 0..ROWS {
        if source_rows[index]["edge_id"] != role_rows[index]["edge_id"]
            || source_rows[index]["edge_id"] != head_rows[index]["edge_id"]
            || source_rows[index]["edge_id"] != input_rows[index]["edge_id"]
        {
            return Err("GW-KEY-1 held-out row order changed".into());
        }
        if source_rows[index]["split"] != "train" {
            continue;
        }
        let offset = source_rows[index]["source_offset"]
            .as_u64()
            .ok_or("source offset missing")? as usize;
        let count = source_rows[index]["source_count"]
            .as_u64()
            .ok_or("source count missing")? as usize;
        let source_roles = roles_for(index, count)?;
        let template = role_rows[index]["template_id"]
            .as_str()
            .ok_or("template missing")?
            .to_string();
        for (position, role) in source_roles.into_iter().enumerate() {
            let entry = means
                .entry((template.clone(), role))
                .or_insert_with(|| (vec![0.0; HEAD_DIM], vec![0.0; HEAD_DIM], 0));
            entry.2 += 1;
            let base = (offset + position) * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                entry.0[dimension] += all_k[base + dimension];
                entry.1[dimension] += all_v[base + dimension];
            }
        }
    }
    for (key, (keys, values, count)) in &mut means {
        if *count == 0 {
            return Err(format!("GW-KEY-1 empty train reference cell {key:?}").into());
        }
        for dimension in 0..HEAD_DIM {
            keys[dimension] /= *count as f32;
            values[dimension] /= *count as f32;
        }
    }

    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-KEY-1 held-out container/plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed target plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if source["authorities"]["plan_sha256"] != plan_identity
        || head["authorities"]["plan_sha256"] != plan_identity
    {
        return Err("GW-KEY-1 held-out execution image changed".into());
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
        .ok_or("candidate token IDs missing")?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .map(|value| value as u32)
                .ok_or("bad candidate token")
        })
        .collect::<std::result::Result<_, _>>()?;
    let selected_head = ops.select_output_head(&token_ids)?;

    let v_selected = selected_masks(&selection, "V")?;
    let joint_selected = selected_masks(&selection, "joint")?;
    let arm_masks = [
        (FULL, FULL),
        (FULL, FULL),
        (FULL, 0),
        v_selected,
        (FULL, FULL ^ v_selected.1),
        (0, 0),
        joint_selected,
        (FULL ^ joint_selected.0, FULL ^ joint_selected.1),
    ];

    let proximal_path = output.join("heldout-proximal-logits.f32");
    let carrier_path = output.join("heldout-carriers.f32");
    let terminal_path = output.join("heldout-terminal-logits.f32");
    let rows_path = output.join("heldout-rows.jsonl");
    if [&proximal_path, &carrier_path, &terminal_path, &rows_path]
        .iter()
        .any(|path| path.exists())
    {
        return Err("GW-KEY-1 held-out output is partial or already exists".into());
    }
    let mut proximal_out = writer(&proximal_path)?;
    let mut carrier_out = writer(&carrier_path)?;
    let mut terminal_out = writer(&terminal_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut heldout_index = 0usize;
    let mut intervention_firings = 0usize;
    let mut full_carrier_mismatches = 0usize;
    let mut full_proximal_mismatches = 0usize;
    let mut full_terminal_mismatches = 0usize;
    let mut identity_terminal_mismatches = 0usize;
    let mut max_reference_norm_relative_error = 0.0f64;
    for original_row in 0..ROWS {
        if source_rows[original_row]["split"] == "train" {
            continue;
        }
        let offset = source_rows[original_row]["source_offset"]
            .as_u64()
            .ok_or("source offset missing")? as usize;
        let count = source_rows[original_row]["source_count"]
            .as_u64()
            .ok_or("source count missing")? as usize;
        let source_roles = roles_for(original_row, count)?;
        let template = role_rows[original_row]["template_id"]
            .as_str()
            .ok_or("template missing")?;
        let natural_keys = vectors(&all_k, offset, count);
        let natural_values = vectors(&all_v, offset, count);
        let mut reference_keys = Vec::with_capacity(count);
        let mut reference_values = Vec::with_capacity(count);
        for (position, role) in source_roles.iter().copied().enumerate() {
            let (key_mean, value_mean, _) = means
                .get(&(template.to_string(), role))
                .ok_or("missing held-out train reference cell")?;
            let key = scaled(key_mean, &natural_keys[position])?;
            let value = scaled(value_mean, &natural_values[position])?;
            max_reference_norm_relative_error = max_reference_norm_relative_error.max(
                ((norm(&key) - norm(&natural_keys[position])) / norm(&natural_keys[position]))
                    .abs(),
            );
            max_reference_norm_relative_error = max_reference_norm_relative_error.max(
                ((norm(&value) - norm(&natural_values[position]))
                    / norm(&natural_values[position]))
                .abs(),
            );
            reference_keys.push(key);
            reference_values.push(value);
        }
        if max_reference_norm_relative_error > 1e-5 {
            return Err("GW-KEY-1 held-out source norm match failed".into());
        }
        let query_start = original_row * HEAD_DIM;
        let query = &all_q[query_start..query_start + HEAD_DIM];
        let natural_heads = vectors(&all_heads, original_row * HEADS, HEADS);
        let before = &all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN];
        let mut local_carriers = Vec::with_capacity(ARMS.len());
        let mut proximal_logits = Vec::with_capacity(ARMS.len());
        for &(key_mask, value_mask) in &arm_masks {
            let keys: Vec<Vec<f32>> = natural_keys
                .iter()
                .zip(&reference_keys)
                .zip(&source_roles)
                .map(|((natural, reference), &role)| {
                    if key_mask & (1 << role) != 0 {
                        natural.clone()
                    } else {
                        reference.clone()
                    }
                })
                .collect();
            let routing = replay_softmax_source_head(
                query,
                &keys,
                &natural_values,
                op.score_scale,
                op.logit_softcapping,
            )?;
            let mut h1 = vec![0.0f32; HEAD_DIM];
            for (position, &weight) in routing.weights.iter().enumerate() {
                let role = source_roles[position];
                let value = if value_mask & (1 << role) != 0 {
                    &natural_values[position]
                } else {
                    &reference_values[position]
                };
                for (accumulator, source) in h1.iter_mut().zip(value) {
                    *accumulator += weight * source;
                }
            }
            let mut mixture = natural_heads.clone();
            mixture[HEAD] = h1;
            let replay = replay_attention_mixture(&plan, &ops, &backend, LAYER, before, &mixture)?;
            proximal_logits.push(ops.readout_carrier_selected(
                &backend,
                &replay.carrier_after,
                &selected_head,
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

        let tokens: Vec<u32> = input_rows[original_row]["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt token IDs missing")?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .map(|value| value as u32)
                    .ok_or("bad prompt token")
            })
            .collect::<std::result::Result<_, _>>()?;
        let position = tokens.len() - 1;
        let final_token = tokens[position];
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
                    .ok_or("missing natural terminal logits")?
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
                    return Err("GW-KEY-1 held-out intervention did not fire exactly once".into());
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
        identity_terminal_mismatches += bits_differ(&terminal_logits[0], &terminal_logits[1]);
        if full_carrier_mismatches != 0
            || full_proximal_mismatches != 0
            || full_terminal_mismatches != 0
            || identity_terminal_mismatches != 0
        {
            return Err("GW-KEY-1 held-out full or identity parity failed".into());
        }
        for arm in 0..ARMS.len() {
            write_f32(&mut proximal_out, &proximal_logits[arm])?;
            write_f32(&mut carrier_out, &local_carriers[arm])?;
            write_f32(&mut terminal_out, &terminal_logits[arm])?;
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "heldout_row":heldout_index,"original_row":original_row,"edge_id":head_rows[original_row]["edge_id"],
                "split":head_rows[original_row]["split"],"relation":head_rows[original_row]["semantic_edge"]["relation"],
                "prompt_semantic_family":head_rows[original_row]["semantic_edge"]["prompt_semantic_family"],
                "template_id":template,"arms":ARMS,"arm_masks":arm_masks,
            }),
        )?;
        heldout_index += 1;
        if heldout_index.is_multiple_of(25) || heldout_index == HELDOUT_ROWS {
            eprintln!("GW-KEY-1 held-out arms {heldout_index}/{HELDOUT_ROWS}");
        }
    }
    if heldout_index != HELDOUT_ROWS {
        return Err(format!("GW-KEY-1 yielded {heldout_index} held-out rows").into());
    }
    for output in [
        &mut proximal_out,
        &mut carrier_out,
        &mut terminal_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop((proximal_out, carrier_out, terminal_out, rows_out));
    let artifacts = [
        (&proximal_path,"f32-le",json!([HELDOUT_ROWS,ARMS.len(),CANDIDATES])),
        (&carrier_path,"f32-le",json!([HELDOUT_ROWS,ARMS.len(),HIDDEN])),
        (&terminal_path,"f32-le",json!([HELDOUT_ROWS,ARMS.len(),CANDIDATES])),
        (&rows_path,"jsonl",json!([HELDOUT_ROWS])),
    ].into_iter().map(|(path,dtype,shape)| Ok(json!({
        "path":path.file_name().and_then(|name|name.to_str()).ok_or("bad artifact path")?,
        "dtype":dtype,"shape":shape,"bytes":std::fs::metadata(path)?.len(),"sha256":file_sha(path)?,
    }))).collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema":OUTPUT_SCHEMA,"status":"complete_frozen_heldout","preregistration_sha256":prereg["preregistration_sha256"],
        "selection_sha256":selection["selection_sha256"],"selection_file_sha256":file_sha(selection_path)?,
        "natural_source_capture_sha256":file_sha(source_path)?,"gwhead1_natural_capture_sha256":file_sha(head_path)?,
        "authorities":{"plan_sha256":plan_identity,"backend":backend.name(),"execution_fingerprint":ExecutionProvenance::of(&ops).fingerprint()},
        "arms":ARMS,"arm_masks":arm_masks,
        "reference":{"grouping":"template_id x source_role x K-or-V","donors":"all train rows only","scale":"per-source natural L2 norm","max_relative_norm_error":max_reference_norm_relative_error,"threshold":1e-5},
        "parity":{"full_carrier_bit_mismatches":full_carrier_mismatches,"full_proximal_bit_mismatches":full_proximal_mismatches,
            "full_terminal_bit_mismatches":full_terminal_mismatches,"identity_terminal_bit_mismatches":identity_terminal_mismatches},
        "intervention_firings":intervention_firings,"expected_intervention_firings":HELDOUT_ROWS*(ARMS.len()-1),
        "artifacts":artifacts,
    });
    if intervention_firings != HELDOUT_ROWS * (ARMS.len() - 1) {
        return Err("GW-KEY-1 held-out intervention firing count changed".into());
    }
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
