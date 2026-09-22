//! Outcome-free GW-READ-1 Q/V candidate-input capture.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PROTOCOL_SCHEMA: &str = "larql.gwread1.protocol.v2";
const PROTOCOL_STATUS: &str = "amended_frozen_pre_candidate_capture_fit_and_execution";
const OUTPUT_SCHEMA: &str = "larql.gwread1.candidate-input-capture.v1";
const LAYERS: [usize; 8] = [0, 4, 8, 12, 16, 20, 23, 24];
const HEAD: usize = 1;
const HEAD_DIM: usize = 256;
const ROWS: usize = 426;

type CapturedQv = (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>);

#[derive(Default)]
struct Capture {
    target: usize,
    subject_positions: Vec<usize>,
    queries: BTreeMap<usize, Vec<f32>>,
    subject_values: BTreeMap<usize, Vec<Vec<f32>>>,
    error: Option<String>,
}

impl Capture {
    fn new(target: usize, subject_positions: Vec<usize>) -> Self {
        Self {
            target,
            subject_positions,
            ..Self::default()
        }
    }

    fn finish(self) -> Result<CapturedQv> {
        if let Some(error) = self.error {
            return Err(error.into());
        }
        let queries = LAYERS
            .iter()
            .map(|layer| {
                self.queries
                    .get(layer)
                    .cloned()
                    .ok_or_else(|| format!("GW-READ-1 missing L{layer} query").into())
            })
            .collect::<Result<Vec<_>>>()?;
        let values = LAYERS
            .iter()
            .map(|layer| {
                self.subject_values
                    .get(layer)
                    .cloned()
                    .ok_or_else(|| format!("GW-READ-1 missing L{layer} subject V").into())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((queries, values))
    }
}

impl StepObserver for Capture {
    fn event(&mut self, _: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        position == self.target && LAYERS.contains(&layer)
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if record.head != HEAD || record.position != self.target || !LAYERS.contains(&layer) {
            return;
        }
        if self.queries.contains_key(&layer)
            || record.source_start != 0
            || record.query.len() != HEAD_DIM
            || self
                .subject_positions
                .iter()
                .any(|&position| position >= record.source_values.len())
        {
            self.error = Some(format!("invalid or duplicate L{layer} H1 capture"));
            return;
        }
        self.queries.insert(layer, record.query.to_vec());
        self.subject_values.insert(
            layer,
            self.subject_positions
                .iter()
                .map(|&position| record.source_values[position].clone())
                .collect(),
        );
    }
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

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() != expected * 4 {
        return Err(format!("unexpected f32 artifact size: {}", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn token_ids(value: &Value) -> Result<Vec<u32>> {
    Ok(value
        .as_array()
        .ok_or("token IDs are not an array")?
        .iter()
        .map(|token| token.as_u64().map(|id| id as u32).ok_or("invalid token ID"))
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

fn positions(value: &Value) -> Result<Vec<usize>> {
    Ok(value
        .as_array()
        .ok_or("role positions are not an array")?
        .iter()
        .map(|position| {
            position
                .as_u64()
                .map(|index| index as usize)
                .ok_or("invalid role position")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

fn artifact(path: &Path, dtype: &str, shape: Value) -> Result<Value> {
    Ok(json!({
        "path": path.file_name().and_then(|value| value.to_str()).ok_or("bad artifact path")?,
        "dtype": dtype,
        "shape": shape,
        "bytes": std::fs::metadata(path)?.len(),
        "sha256": file_sha(path)?,
    }))
}

/// `--gwread1-capture CONTAINER PROTOCOL OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return Err(
            "Usage: observatory_record --gwread1-capture CONTAINER PROTOCOL OUTPUT_DIR".into(),
        );
    }
    let container = Path::new(&args[0]);
    let protocol_path = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("candidate-input-capture-manifest.json");
    if manifest_path.exists() {
        return Err("GW-READ-1 candidate-input capture is already sealed".into());
    }

    let protocol: Value = serde_json::from_slice(&std::fs::read(protocol_path)?)?;
    if protocol["schema"] != PROTOCOL_SCHEMA || protocol["status"] != PROTOCOL_STATUS {
        return Err("inadmissible GW-READ-1 protocol".into());
    }
    let protocol_root = protocol_path.parent().ok_or("protocol has no parent")?;
    let input_path = resolve(protocol_root, &protocol["authorities"]["input_rows"])?;
    let roles_path = resolve(protocol_root, &protocol["authorities"]["source_roles"])?;
    let source_manifest_path = resolve(
        protocol_root,
        &protocol["authorities"]["gwkey1_source_capture"],
    )?;
    for (path, descriptor) in [
        (&input_path, &protocol["authorities"]["input_rows"]),
        (&roles_path, &protocol["authorities"]["source_roles"]),
        (
            &source_manifest_path,
            &protocol["authorities"]["gwkey1_source_capture"],
        ),
    ] {
        if file_sha(path)? != descriptor["sha256"] {
            return Err(format!("GW-READ-1 authority changed: {}", path.display()).into());
        }
    }
    let inputs = read_jsonl(&input_path)?;
    let roles = read_jsonl(&roles_path)?;
    if inputs.len() != ROWS || roles.len() != ROWS {
        return Err("GW-READ-1 population changed".into());
    }

    let source_manifest: Value = serde_json::from_slice(&std::fs::read(&source_manifest_path)?)?;
    if source_manifest["schema"] != "larql.gwkey1.source-capture.v1"
        || source_manifest["parity"]["gwhead_natural_h1_bit_mismatches"] != 0
        || source_manifest["parity"]["source_replay_bit_mismatches"] != 0
    {
        return Err("inadmissible GW-KEY-1 source capture".into());
    }
    let source_root = source_manifest_path
        .parent()
        .ok_or("source manifest has no parent")?;
    let authoritative_rows = read_jsonl(&source_root.join("source-rows.jsonl"))?;
    let source_count = source_manifest["site"]["source_rows"]
        .as_u64()
        .ok_or("source row count missing")? as usize;
    let authoritative_q = read_f32(&source_root.join("natural-q.f32"), ROWS * HEAD_DIM)?;
    let authoritative_v = read_f32(&source_root.join("source-v.f32"), source_count * HEAD_DIM)?;

    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-READ-1 container/plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed plan absent")?;
    if plan.layers.len() <= *LAYERS.last().expect("capture layers") {
        return Err("GW-READ-1 capture layer is outside the plan".into());
    }
    for &layer in &LAYERS {
        let attention = plan.layers[layer]
            .attention
            .softmax()
            .ok_or_else(|| format!("L{layer} is not softmax attention"))?;
        if attention.num_q_heads <= HEAD || attention.head_dim != HEAD_DIM {
            return Err(format!("L{layer} H1 geometry changed").into());
        }
    }
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if source_manifest["authorities"]["plan_sha256"] != plan_identity {
        return Err("GW-READ-1 plan differs from the authoritative source capture".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;

    let q_path = output.join("original-q.f32");
    let v_path = output.join("original-subject-v.f32");
    let rows_path = output.join("original-rows.jsonl");
    let entity_v_path = output.join("entity-only-subject-v.f32");
    let entity_rows_path = output.join("entity-only-rows.jsonl");
    let mut q_out = writer(&q_path)?;
    let mut v_out = writer(&v_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut original_subject_rows = 0usize;
    let mut q_l24_bit_mismatches = 0usize;
    let mut v_l24_bit_mismatches = 0usize;

    for (row, ((input, role), authoritative_row)) in inputs
        .iter()
        .zip(&roles)
        .zip(&authoritative_rows)
        .enumerate()
    {
        if input["edge_id"] != role["edge_id"] || input["edge_id"] != authoritative_row["edge_id"] {
            return Err("GW-READ-1 row authorities disagree".into());
        }
        let tokens = token_ids(&input["prompt"]["token_ids"])?;
        let subject_positions = positions(&role["roles"]["subject_entity"])?;
        if tokens.is_empty() || subject_positions.is_empty() {
            return Err("GW-READ-1 prompt or subject role is empty".into());
        }
        let target = tokens.len() - 1;
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        let mut capture = Capture::new(target, subject_positions.clone());
        for &token in &tokens {
            session.step_observed(token, &mut capture)?;
        }
        let (queries, subject_values) = capture.finish()?;
        let l24 = LAYERS.len() - 1;
        q_l24_bit_mismatches += queries[l24]
            .iter()
            .zip(&authoritative_q[row * HEAD_DIM..(row + 1) * HEAD_DIM])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        let source_offset = authoritative_row["source_offset"]
            .as_u64()
            .ok_or("authoritative source offset missing")? as usize;
        for (subject_index, &position) in subject_positions.iter().enumerate() {
            let start = (source_offset + position) * HEAD_DIM;
            v_l24_bit_mismatches += subject_values[l24][subject_index]
                .iter()
                .zip(&authoritative_v[start..start + HEAD_DIM])
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
        }
        for query in &queries {
            write_f32(&mut q_out, query)?;
        }
        for subject_index in 0..subject_positions.len() {
            for layer_values in &subject_values {
                write_f32(&mut v_out, &layer_values[subject_index])?;
            }
        }
        write_json_line(
            &mut rows_out,
            &json!({
                "row": row,
                "edge_id": input["edge_id"],
                "split": input["split"],
                "relation": input["semantic_edge"]["relation"],
                "subject": input["semantic_edge"]["subject"],
                "template_id": input["prompt"]["template_id"],
                "prompt_semantic_family": input["semantic_edge"]["prompt_semantic_family"],
                "subject_positions": subject_positions,
                "subject_token_ids": subject_positions.iter().map(|&position| tokens[position]).collect::<Vec<_>>(),
                "subject_offset": original_subject_rows,
                "subject_count": subject_values[l24].len(),
            }),
        )?;
        original_subject_rows += subject_values[l24].len();
        if (row + 1).is_multiple_of(25) || row + 1 == ROWS {
            eprintln!("GW-READ-1 original capture {}/{}", row + 1, ROWS);
        }
    }
    for output in [&mut q_out, &mut v_out, &mut rows_out] {
        output.flush()?;
    }
    drop((q_out, v_out, rows_out));
    if q_l24_bit_mismatches != 0 || v_l24_bit_mismatches != 0 {
        return Err("GW-READ-1 L24 capture does not match GW-KEY-1 bitwise".into());
    }

    #[derive(Default)]
    struct Entity {
        subject: String,
        token_ids: Vec<u32>,
        relations: BTreeSet<String>,
        prompt_rows: Vec<usize>,
    }
    let mut entities: Vec<Entity> = Vec::new();
    let mut entity_by_tokens: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    let original_rows = read_jsonl(&rows_path)?;
    for row in &original_rows {
        let tokens = token_ids(&row["subject_token_ids"])?;
        let next = entity_by_tokens.len();
        let entity_row = *entity_by_tokens.entry(tokens.clone()).or_insert(next);
        if entity_row == entities.len() {
            entities.push(Entity {
                subject: row["subject"].as_str().ok_or("subject missing")?.to_owned(),
                token_ids: tokens,
                ..Entity::default()
            });
        } else if entities[entity_row].subject
            != row["subject"].as_str().ok_or("subject missing")?
        {
            return Err("one frozen token sequence names multiple subjects".into());
        }
        entities[entity_row].relations.insert(
            row["relation"]
                .as_str()
                .ok_or("relation missing")?
                .to_owned(),
        );
        entities[entity_row]
            .prompt_rows
            .push(row["row"].as_u64().ok_or("row missing")? as usize);
    }

    let mut entity_v_out = writer(&entity_v_path)?;
    let mut entity_rows_out = writer(&entity_rows_path)?;
    let mut entity_subject_rows = 0usize;
    for (entity_row, entity) in entities.iter().enumerate() {
        let mut tokens = Vec::with_capacity(entity.token_ids.len() + 1);
        tokens.push(2);
        tokens.extend(&entity.token_ids);
        let subject_positions: Vec<usize> = (1..tokens.len()).collect();
        let target = tokens.len() - 1;
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        let mut capture = Capture::new(target, subject_positions);
        for &token in &tokens {
            session.step_observed(token, &mut capture)?;
        }
        let (_, subject_values) = capture.finish()?;
        for subject_index in 0..entity.token_ids.len() {
            for layer_values in &subject_values {
                write_f32(&mut entity_v_out, &layer_values[subject_index])?;
            }
        }
        write_json_line(
            &mut entity_rows_out,
            &json!({
                "entity_row": entity_row,
                "subject": entity.subject,
                "subject_token_ids": entity.token_ids,
                "subject_offset": entity_subject_rows,
                "subject_count": entity.token_ids.len(),
                "relations": entity.relations,
                "prompt_rows": entity.prompt_rows,
                "recipe": "BOS token 2 followed by frozen subject_entity token IDs",
            }),
        )?;
        entity_subject_rows += entity.token_ids.len();
        if (entity_row + 1).is_multiple_of(25) || entity_row + 1 == entities.len() {
            eprintln!(
                "GW-READ-1 entity-only capture {}/{}",
                entity_row + 1,
                entities.len()
            );
        }
    }
    entity_v_out.flush()?;
    entity_rows_out.flush()?;
    drop((entity_v_out, entity_rows_out));

    let artifacts = vec![
        artifact(&q_path, "f32-le", json!([ROWS, LAYERS.len(), HEAD_DIM]))?,
        artifact(
            &v_path,
            "f32-le",
            json!([original_subject_rows, LAYERS.len(), HEAD_DIM]),
        )?,
        artifact(&rows_path, "jsonl", json!([ROWS]))?,
        artifact(
            &entity_v_path,
            "f32-le",
            json!([entity_subject_rows, LAYERS.len(), HEAD_DIM]),
        )?,
        artifact(&entity_rows_path, "jsonl", json!([entities.len()]))?,
    ];
    let manifest = json!({
        "schema": OUTPUT_SCHEMA,
        "status": "candidate_inputs_captured_pre_fit_and_effect_execution",
        "protocol": {"sha256": protocol["protocol_sha256"], "file_sha256": file_sha(protocol_path)?},
        "authorities": {
            "plan_sha256": plan_identity,
            "backend": backend.name(),
            "gwkey1_source_capture_sha256": file_sha(&source_manifest_path)?,
        },
        "site": {"layers": LAYERS, "head": HEAD, "head_dim": HEAD_DIM},
        "counts": {
            "prompt_rows": ROWS,
            "original_subject_rows": original_subject_rows,
            "entity_only_rows": entities.len(),
            "entity_only_subject_rows": entity_subject_rows,
        },
        "parity": {
            "authoritative_l24_q_bit_mismatches": q_l24_bit_mismatches,
            "authoritative_l24_subject_v_bit_mismatches": v_l24_bit_mismatches,
        },
        "outcomes_captured": false,
        "candidate_fitting_performed": false,
        "effect_execution_performed": false,
        "artifacts": artifacts,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
