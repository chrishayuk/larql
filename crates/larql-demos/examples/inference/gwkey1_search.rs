//! Frozen GW-KEY-1 exhaustive train-only K/V source-role replay.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::head_replay::{
    replay_attention_mixture, replay_softmax_source_head,
};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PREREG_SCHEMA: &str = "larql.gwkey1.preregistration.v1";
const SOURCE_SCHEMA: &str = "larql.gwkey1.source-capture.v1";
const HEAD_SCHEMA: &str = "larql.gwhead1.natural-capture.v1";
const OUTPUT_SCHEMA: &str = "larql.gwkey1.train-source-replay.v1";
const LAYER: usize = 24;
const HEAD: usize = 1;
const ROWS: usize = 426;
const TRAIN_ROWS: usize = 255;
const HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const HIDDEN: usize = 2560;
const CANDIDATES: usize = 126;
const ROLES: usize = 6;
const SUBSETS: usize = 64;
const FULL_MASK: usize = SUBSETS - 1;

struct TrainRow {
    original_row: usize,
    edge_id: String,
    semantic_edge: String,
    template_id: String,
    before: Vec<f32>,
    heads: Vec<Vec<f32>>,
    query: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    role_of_source: Vec<usize>,
}

struct References {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    role_k_delta_sq: [f64; ROLES],
    role_v_delta_sq: [f64; ROLES],
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
    if !direction_norm.is_finite()
        || direction_norm <= 0.0
        || !target_norm.is_finite()
        || target_norm <= 0.0
    {
        return Err("GW-KEY-1 reference direction or target norm is degenerate".into());
    }
    let scale = target_norm / direction_norm;
    Ok(direction
        .iter()
        .map(|value| *value * scale as f32)
        .collect())
}

fn squared_delta(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum()
}

fn replacement_norms(rows: &[References], keys: bool) -> Vec<f64> {
    (0..SUBSETS)
        .map(|mask| {
            rows.iter()
                .map(|row| {
                    let per_role = if keys {
                        &row.role_k_delta_sq
                    } else {
                        &row.role_v_delta_sq
                    };
                    (0..ROLES)
                        .filter(|role| mask & (1 << role) == 0)
                        .map(|role| per_role[role])
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / rows.len() as f64
        })
        .collect()
}

/// `--gwkey1-train-search CONTAINER PREREG SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 5 {
        return Err("Usage: observatory_record --gwkey1-train-search CONTAINER PREREG SOURCE_CAPTURE HEAD_CAPTURE OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let source_path = Path::new(&args[2]);
    let head_path = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("train-source-replay-manifest.json");
    if manifest_path.exists() {
        return Err("GW-KEY-1 train source replay is already sealed".into());
    }

    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    let source: Value = serde_json::from_slice(&std::fs::read(source_path)?)?;
    let head: Value = serde_json::from_slice(&std::fs::read(head_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA
        || prereg["status"] != "frozen_pre_execution"
        || source["schema"] != SOURCE_SCHEMA
        || source["status"] != "natural_source_capture_complete_pre_search"
        || head["schema"] != HEAD_SCHEMA
        || head["status"] != "natural_capture_complete_pre_subset_search"
        || source["preregistration_sha256"] != prereg["preregistration_sha256"]
        || source["parity"]["gwhead_natural_h1_bit_mismatches"] != 0
        || source["parity"]["source_replay_bit_mismatches"] != 0
    {
        return Err("GW-KEY-1 train search authorities are inadmissible".into());
    }
    if source["gwhead1_natural_capture_sha256"] != file_sha(head_path)? {
        return Err("GW-KEY-1 source capture binds another GW-HEAD capture".into());
    }
    let prereg_root = prereg_path
        .parent()
        .ok_or("preregistration has no parent")?;
    let roles_path = resolve(prereg_root, &prereg["roles"]["artifact"], "path")?;
    if file_sha(&roles_path)? != prereg["roles"]["artifact"]["sha256"] {
        return Err("GW-KEY-1 frozen role map changed".into());
    }
    let role_rows: Vec<Value> = BufReader::new(File::open(&roles_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if role_rows.len() != ROWS {
        return Err("GW-KEY-1 role row count changed".into());
    }

    let source_root = source_path.parent().ok_or("source capture has no parent")?;
    let head_root = head_path.parent().ok_or("head capture has no parent")?;
    let source_count = source["site"]["source_rows"]
        .as_u64()
        .ok_or("source row count missing")? as usize;
    let source_rows = jsonl(source_root, &source, "source-rows.jsonl")?;
    let head_rows = jsonl(head_root, &head, "natural-rows.jsonl")?;
    if source_rows.len() != ROWS || head_rows.len() != ROWS {
        return Err("GW-KEY-1 natural row count changed".into());
    }
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
    let all_h1 = read_f32(source_root, &source, "natural-h1.f32", ROWS * HEAD_DIM)?;
    let all_heads = read_f32(head_root, &head, "head-values.f32", ROWS * HEADS * HEAD_DIM)?;
    let all_before = read_f32(head_root, &head, "carrier-before.f32", ROWS * HIDDEN)?;
    let all_natural_logits = read_f32(
        head_root,
        &head,
        "candidate-after-logits.f32",
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
    let mut train = Vec::with_capacity(TRAIN_ROWS);
    for original_row in 0..ROWS {
        if source_rows[original_row]["edge_id"] != role_rows[original_row]["edge_id"]
            || source_rows[original_row]["edge_id"] != head_rows[original_row]["edge_id"]
        {
            return Err("GW-KEY-1 source/head/role row order changed".into());
        }
        if source_rows[original_row]["split"] != "train" {
            continue;
        }
        let offset = source_rows[original_row]["source_offset"]
            .as_u64()
            .ok_or("source offset missing")? as usize;
        let count = source_rows[original_row]["source_count"]
            .as_u64()
            .ok_or("source count missing")? as usize;
        let mut role_of_source = vec![usize::MAX; count];
        for (role, name) in role_names.iter().enumerate() {
            for position in role_rows[original_row]["roles"][name]
                .as_array()
                .ok_or("source role positions missing")?
            {
                let position = position.as_u64().ok_or("bad source position")? as usize;
                if position >= count || role_of_source[position] != usize::MAX {
                    return Err("GW-KEY-1 source role map is not disjoint".into());
                }
                role_of_source[position] = role;
            }
        }
        if role_of_source.contains(&usize::MAX) {
            return Err("GW-KEY-1 source role map is not exhaustive".into());
        }
        let vectors = |all: &[f32], start: usize, rows: usize| -> Vec<Vec<f32>> {
            (0..rows)
                .map(|index| {
                    let base = (start + index) * HEAD_DIM;
                    all[base..base + HEAD_DIM].to_vec()
                })
                .collect()
        };
        let heads = vectors(&all_heads, original_row * HEADS, HEADS);
        let h1_start = original_row * HEAD_DIM;
        if heads[HEAD]
            .iter()
            .zip(&all_h1[h1_start..h1_start + HEAD_DIM])
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("GW-KEY-1 H1 artifacts disagree".into());
        }
        train.push(TrainRow {
            original_row,
            edge_id: source_rows[original_row]["edge_id"]
                .as_str()
                .ok_or("edge ID missing")?
                .to_string(),
            semantic_edge: format!(
                "{}\u{1f}{}\u{1f}{}",
                head_rows[original_row]["semantic_edge"]["subject"]
                    .as_str()
                    .ok_or("semantic subject missing")?,
                head_rows[original_row]["semantic_edge"]["relation"]
                    .as_str()
                    .ok_or("semantic relation missing")?,
                head_rows[original_row]["semantic_edge"]["target"]
                    .as_str()
                    .ok_or("semantic target missing")?,
            ),
            template_id: role_rows[original_row]["template_id"]
                .as_str()
                .ok_or("template ID missing")?
                .to_string(),
            before: all_before[original_row * HIDDEN..(original_row + 1) * HIDDEN].to_vec(),
            heads,
            query: all_q[h1_start..h1_start + HEAD_DIM].to_vec(),
            keys: vectors(&all_k, offset, count),
            values: vectors(&all_v, offset, count),
            role_of_source,
        });
    }
    if train.len() != TRAIN_ROWS {
        return Err(format!("GW-KEY-1 found {} train rows", train.len()).into());
    }

    eprintln!("Building frozen leave-one-semantic-edge-out K/V reference banks...");
    let mut references = Vec::with_capacity(TRAIN_ROWS);
    let mut max_norm_relative_error = 0.0f64;
    for (row_index, row) in train.iter().enumerate() {
        let mut key_means = vec![vec![0.0f32; HEAD_DIM]; ROLES];
        let mut value_means = vec![vec![0.0f32; HEAD_DIM]; ROLES];
        let mut counts = [0usize; ROLES];
        for donor in train.iter().filter(|donor| {
            donor.template_id == row.template_id && donor.semantic_edge != row.semantic_edge
        }) {
            for (position, &role) in donor.role_of_source.iter().enumerate() {
                counts[role] += 1;
                for dimension in 0..HEAD_DIM {
                    key_means[role][dimension] += donor.keys[position][dimension];
                    value_means[role][dimension] += donor.values[position][dimension];
                }
            }
        }
        for role in 0..ROLES {
            if counts[role] == 0
                && row.role_of_source.contains(&role)
                && role_names[role] != "punctuation_separator"
            {
                return Err(
                    format!("{} role {} has no donors", row.edge_id, role_names[role]).into(),
                );
            }
            if counts[role] != 0 {
                for dimension in 0..HEAD_DIM {
                    key_means[role][dimension] /= counts[role] as f32;
                    value_means[role][dimension] /= counts[role] as f32;
                }
            }
        }
        let mut keys = Vec::with_capacity(row.keys.len());
        let mut values = Vec::with_capacity(row.values.len());
        let mut role_k_delta_sq = [0.0; ROLES];
        let mut role_v_delta_sq = [0.0; ROLES];
        for (position, &role) in row.role_of_source.iter().enumerate() {
            if counts[role] == 0 {
                return Err(format!(
                    "{} populated role {} has no donors",
                    row.edge_id, role_names[role]
                )
                .into());
            }
            let key = scaled(&key_means[role], &row.keys[position])?;
            let value = scaled(&value_means[role], &row.values[position])?;
            max_norm_relative_error = max_norm_relative_error
                .max(((norm(&key) - norm(&row.keys[position])) / norm(&row.keys[position])).abs());
            max_norm_relative_error = max_norm_relative_error.max(
                ((norm(&value) - norm(&row.values[position])) / norm(&row.values[position])).abs(),
            );
            role_k_delta_sq[role] += squared_delta(&row.keys[position], &key);
            role_v_delta_sq[role] += squared_delta(&row.values[position], &value);
            keys.push(key);
            values.push(value);
        }
        references.push(References {
            keys,
            values,
            role_k_delta_sq,
            role_v_delta_sq,
        });
        if (row_index + 1).is_multiple_of(50) || row_index + 1 == TRAIN_ROWS {
            eprintln!("GW-KEY-1 reference bank {}/{}", row_index + 1, TRAIN_ROWS);
        }
    }
    if max_norm_relative_error > 1e-5 {
        return Err("GW-KEY-1 source reference norm matching exceeded 1e-5".into());
    }

    eprintln!("Preparing the frozen GW-KEY-1 replay image...");
    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-KEY-1 container/plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed target plan absent")?;
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if source["authorities"]["plan_sha256"] != plan_identity
        || head["authorities"]["plan_sha256"] != plan_identity
    {
        return Err("GW-KEY-1 replay image differs from capture authorities".into());
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
        .map(|token| {
            token
                .as_u64()
                .map(|value| value as u32)
                .ok_or("bad candidate token")
        })
        .collect::<std::result::Result<_, _>>()?;
    if token_ids.len() != CANDIDATES {
        return Err("GW-KEY-1 candidate set changed".into());
    }
    let selected = ops.select_output_head(&token_ids)?;

    eprintln!("Executing all 64 K masks once per train row...");
    let mut weights = vec![vec![Vec::<f32>::new(); SUBSETS]; TRAIN_ROWS];
    for ((row, reference), row_weights) in train.iter().zip(&references).zip(&mut weights) {
        for (key_mask, stored_weights) in row_weights.iter_mut().enumerate() {
            let keys: Vec<Vec<f32>> = row
                .keys
                .iter()
                .zip(&reference.keys)
                .zip(&row.role_of_source)
                .map(|((natural, replaced), &role)| {
                    if key_mask & (1 << role) != 0 {
                        natural.clone()
                    } else {
                        replaced.clone()
                    }
                })
                .collect();
            *stored_weights = replay_softmax_source_head(
                &row.query,
                &keys,
                &row.values,
                op.score_scale,
                op.logit_softcapping,
            )?
            .weights;
        }
    }

    let logits_path = output.join("train-joint-logits.f32");
    let reference_k_path = output.join("train-reference-k.f32");
    let reference_v_path = output.join("train-reference-v.f32");
    let rows_path = output.join("train-rows.jsonl");
    if [
        &logits_path,
        &reference_k_path,
        &reference_v_path,
        &rows_path,
    ]
    .iter()
    .any(|path| path.exists())
    {
        return Err("GW-KEY-1 train replay output is partial or already exists".into());
    }
    let mut logits_out = writer(&logits_path)?;
    let mut reference_k_out = writer(&reference_k_path)?;
    let mut reference_v_out = writer(&reference_v_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut train_source_count = 0usize;
    for (train_row, (row, reference)) in train.iter().zip(&references).enumerate() {
        for vector in &reference.keys {
            write_f32(&mut reference_k_out, vector)?;
        }
        for vector in &reference.values {
            write_f32(&mut reference_v_out, vector)?;
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "train_row":train_row,"original_row":row.original_row,"edge_id":row.edge_id,
                "template_id":row.template_id,"reference_source_offset":train_source_count,
                "source_count":row.keys.len(),"role_of_source":row.role_of_source,
            }),
        )?;
        train_source_count += row.keys.len();
    }

    eprintln!("Executing the frozen 4,096-pair joint K/V surface...");
    let mut natural_logit_bit_mismatches = 0usize;
    for (key_mask, _) in weights[0].iter().enumerate() {
        for value_mask in 0..SUBSETS {
            for (row_index, (row, reference)) in train.iter().zip(&references).enumerate() {
                let mut h1 = vec![0.0f32; HEAD_DIM];
                for (position, &weight) in weights[row_index][key_mask].iter().enumerate() {
                    let role = row.role_of_source[position];
                    let value = if value_mask & (1 << role) != 0 {
                        &row.values[position]
                    } else {
                        &reference.values[position]
                    };
                    for (accumulator, source) in h1.iter_mut().zip(value) {
                        *accumulator += weight * source;
                    }
                }
                let mut mixture = row.heads.clone();
                mixture[HEAD] = h1;
                let replay =
                    replay_attention_mixture(&plan, &ops, &backend, LAYER, &row.before, &mixture)?;
                let logits =
                    ops.readout_carrier_selected(&backend, &replay.carrier_after, &selected)?;
                if key_mask == FULL_MASK && value_mask == FULL_MASK {
                    let start = row.original_row * CANDIDATES;
                    natural_logit_bit_mismatches += logits
                        .iter()
                        .zip(&all_natural_logits[start..start + CANDIDATES])
                        .filter(|(left, right)| left.to_bits() != right.to_bits())
                        .count();
                }
                write_f32(&mut logits_out, &logits)?;
            }
            let pair = key_mask * SUBSETS + value_mask + 1;
            if pair.is_multiple_of(64) || pair == SUBSETS * SUBSETS {
                eprintln!("GW-KEY-1 joint pairs {}/{}", pair, SUBSETS * SUBSETS);
            }
        }
    }
    for output in [
        &mut logits_out,
        &mut reference_k_out,
        &mut reference_v_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop((logits_out, reference_k_out, reference_v_out, rows_out));
    if natural_logit_bit_mismatches != 0 {
        return Err(
            "GW-KEY-1 natural K/V replay did not reproduce captured logits bit-for-bit".into(),
        );
    }

    let k_replacement_norms = replacement_norms(&references, true);
    let v_replacement_norms = replacement_norms(&references, false);
    let artifacts = [
        (&logits_path, "f32-le", json!([SUBSETS,SUBSETS,TRAIN_ROWS,CANDIDATES])),
        (&reference_k_path, "f32-le", json!([train_source_count,HEAD_DIM])),
        (&reference_v_path, "f32-le", json!([train_source_count,HEAD_DIM])),
        (&rows_path, "jsonl", json!([TRAIN_ROWS])),
    ].into_iter().map(|(path,dtype,shape)| Ok(json!({
        "path":path.file_name().and_then(|name|name.to_str()).ok_or("bad artifact path")?,
        "dtype":dtype,"shape":shape,"bytes":std::fs::metadata(path)?.len(),"sha256":file_sha(path)?,
    }))).collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema":OUTPUT_SCHEMA,"status":"complete_train_only","split":"train","held_out_rows":0,
        "preregistration_sha256":prereg["preregistration_sha256"],
        "natural_source_capture_sha256":file_sha(source_path)?,
        "gwhead1_natural_capture_sha256":file_sha(head_path)?,
        "authorities":{"plan_sha256":plan_identity,"backend":backend.name(),
            "execution_fingerprint":ExecutionProvenance::of(&ops).fingerprint(),
            "prepared_head_representation":selected.representation()},
        "mask_order":{"K":"integer 0..63; natural role iff bit set","V":"integer 0..63; natural role iff bit set",
            "artifact":"K-major, then V, then ascending original row filtered to train"},
        "reference":{"grouping":"template_id x source_role x K-or-V","donor_exclusion":"same semantic edge",
            "direction":"mean across every donor source position in cell","scale":"per-source natural L2 norm",
            "max_relative_norm_error":max_norm_relative_error,"threshold":1e-5,
            "replacement_norm":"mean per-row L2 norm of concatenated replaced source vectors",
            "K_by_mask":k_replacement_norms,"V_by_mask":v_replacement_norms},
        "full_replay_parity":{"candidate_logit_bit_mismatches":natural_logit_bit_mismatches},
        "artifacts":artifacts,"validation_or_test_executed":false,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
