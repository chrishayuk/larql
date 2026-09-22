//! GW-KEY-1 natural-Q and conditioned source K/V capture for frozen L24H1.
use super::gwsup1_readout::{file_sha, write_json_line};
use super::*;
use larql_vindex::format::vindex3::opplan::exec::{
    head_replay::replay_softmax_source_head, observe::AttentionHeadRecord,
};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

const PREREG_SCHEMA: &str = "larql.gwkey1.preregistration.v1";
const OUTPUT_SCHEMA: &str = "larql.gwkey1.source-capture.v1";
const LAYER: usize = 24;
const HEAD: usize = 1;
const HEAD_DIM: usize = 256;
const ROWS: usize = 426;

#[derive(Default)]
struct Capture {
    target: usize,
    query: Option<Vec<f32>>,
    keys: Option<Vec<Vec<f32>>>,
    values: Option<Vec<Vec<f32>>>,
    weights: Option<Vec<f32>>,
    output: Option<Vec<f32>>,
    error: Option<String>,
}

impl Capture {
    fn new(target: usize) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }
}

impl StepObserver for Capture {
    fn event(&mut self, _: StepEvent) {}
    fn wants_attention_heads(&self) -> bool {
        true
    }
    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        layer == LAYER && position == self.target
    }
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer != LAYER || record.head != HEAD || record.position != self.target {
            return;
        }
        if self.query.is_some()
            || record.source_start != 0
            || record.query.len() != HEAD_DIM
            || record.source_keys.len() != self.target + 1
            || record.source_values.len() != self.target + 1
            || record.weights.len() != self.target + 1
        {
            self.error = Some("invalid or duplicate frozen H1 source capture".into());
            return;
        }
        self.query = Some(record.query.to_vec());
        self.keys = Some(record.source_keys.to_vec());
        self.values = Some(record.source_values.to_vec());
        self.weights = Some(record.weights.to_vec());
        self.output = Some(record.values.to_vec());
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

fn resolve(root: &Path, value: &Value, key: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(value[key].as_str().ok_or("authority path missing")?);
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    })
}

/// `--gwkey1-capture CONTAINER PREREG HEAD_CAPTURE_MANIFEST OUTPUT_DIR`
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("Usage: observatory_record --gwkey1-capture CONTAINER PREREG HEAD_CAPTURE_MANIFEST OUTPUT_DIR".into());
    }
    let container = Path::new(&args[0]);
    let prereg_path = Path::new(&args[1]);
    let head_capture_path = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    std::fs::create_dir_all(output)?;
    let manifest_path = output.join("source-capture-manifest.json");
    if manifest_path.exists() {
        return Err("GW-KEY-1 source capture is already sealed".into());
    }
    let prereg: Value = serde_json::from_slice(&std::fs::read(prereg_path)?)?;
    let head_capture: Value = serde_json::from_slice(&std::fs::read(head_capture_path)?)?;
    if prereg["schema"] != PREREG_SCHEMA
        || prereg["status"] != "frozen_pre_execution"
        || head_capture["schema"] != "larql.gwhead1.natural-capture.v1"
        || head_capture["status"] != "natural_capture_complete_pre_subset_search"
    {
        return Err("GW-KEY-1 authorities are inadmissible".into());
    }
    let root = prereg_path
        .parent()
        .ok_or("preregistration has no parent")?;
    let input_path = resolve(root, &prereg["authorities"]["input_rows"], "path")?;
    let roles_path = resolve(root, &prereg["roles"]["artifact"], "path")?;
    let head_adjudication_path =
        resolve(root, &prereg["authorities"]["gwhead1_adjudication"], "path")?;
    if file_sha(&input_path)? != prereg["authorities"]["input_rows"]["sha256"]
        || file_sha(&roles_path)? != prereg["roles"]["artifact"]["sha256"]
    {
        return Err("GW-KEY-1 frozen input or roles changed".into());
    }
    if file_sha(&head_adjudication_path)? != prereg["authorities"]["gwhead1_adjudication"]["sha256"]
    {
        return Err("GW-HEAD-1 adjudication authority changed".into());
    }
    let head_adjudication: Value =
        serde_json::from_slice(&std::fs::read(&head_adjudication_path)?)?;
    if head_adjudication["adjudication_sha256"]
        != prereg["authorities"]["gwhead1_adjudication"]["identity"]
        || head_adjudication["authorities"]["natural_capture_manifest_sha256"]
            != file_sha(head_capture_path)?
    {
        return Err("GW-KEY-1 natural capture is not the adjudicated GW-HEAD authority".into());
    }
    let inputs: Vec<Value> = BufReader::new(File::open(&input_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    let roles: Vec<Value> = BufReader::new(File::open(&roles_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if inputs.len() != ROWS || roles.len() != ROWS {
        return Err("GW-KEY-1 frozen row count changed".into());
    }

    let inspection = inspect_container(container, true)?;
    let outcome = plan_component_ops(&inspection, container, "target")?;
    if !inspection.is_coherent() || !outcome.closed() {
        return Err("GW-KEY-1 container/plan is not executable".into());
    }
    let plan = outcome.plan.ok_or("closed plan absent")?;
    let op = plan.layers[LAYER]
        .attention
        .softmax()
        .ok_or("L24 is not softmax attention")?;
    if op.num_q_heads != 8 || op.head_dim != HEAD_DIM || op.sinks.is_some() {
        return Err("GW-KEY-1 frozen H1 geometry changed or carries sinks".into());
    }
    let plan_identity = format!("sha256:{}", sha(&serde_json::to_vec(&plan)?));
    if head_capture["authorities"]["plan_sha256"] != plan_identity {
        return Err("GW-KEY-1 does not bind the frozen GW-HEAD execution image".into());
    }
    let store = OperandStore::open(container, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    let head_root = head_capture_path
        .parent()
        .ok_or("head capture manifest has no parent")?;
    let head_descriptor = head_capture["artifacts"]
        .as_array()
        .ok_or("head capture artifacts missing")?
        .iter()
        .find(|item| item["path"] == "head-values.f32")
        .ok_or("head-values artifact missing")?;
    let natural_heads_path = head_root.join("head-values.f32");
    if head_descriptor["sha256"] != file_sha(&natural_heads_path)? {
        return Err("GW-HEAD natural head artifact changed".into());
    }
    let mut natural_head_bytes = Vec::new();
    File::open(&natural_heads_path)?.read_to_end(&mut natural_head_bytes)?;
    let natural_heads: Vec<f32> = natural_head_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if natural_heads.len() != ROWS * 8 * HEAD_DIM {
        return Err("GW-HEAD natural head artifact shape changed".into());
    }

    let query_path = output.join("natural-q.f32");
    let keys_path = output.join("source-k.f32");
    let values_path = output.join("source-v.f32");
    let weights_path = output.join("natural-weights.f32");
    let heads_path = output.join("natural-h1.f32");
    let rows_path = output.join("source-rows.jsonl");
    let mut query_out = writer(&query_path)?;
    let mut keys_out = writer(&keys_path)?;
    let mut values_out = writer(&values_path)?;
    let mut weights_out = writer(&weights_path)?;
    let mut heads_out = writer(&heads_path)?;
    let mut rows_out = writer(&rows_path)?;
    let mut source_offset = 0usize;
    let mut natural_h1_mismatches = 0usize;
    let mut replay_mismatches = 0usize;
    for (index, (input, role)) in inputs.iter().zip(&roles).enumerate() {
        if input["edge_id"] != role["edge_id"] {
            return Err("GW-KEY-1 role/input row order changed".into());
        }
        let tokens: Vec<u32> = input["prompt"]["token_ids"]
            .as_array()
            .ok_or("prompt token IDs missing")?
            .iter()
            .map(|token| token.as_u64().map(|v| v as u32).ok_or("bad token"))
            .collect::<std::result::Result<_, _>>()?;
        let target = tokens.len() - 1;
        let mut kv = RowKvState::default();
        let mut observed = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        let mut capture = Capture::new(target);
        for &token in &tokens {
            observed.step_observed(token, &mut capture)?;
        }
        if let Some(error) = capture.error {
            return Err(error.into());
        }
        let query = capture.query.ok_or("natural Q missing")?;
        let keys = capture.keys.ok_or("source K missing")?;
        let values = capture.values.ok_or("source V missing")?;
        let weights = capture.weights.ok_or("natural weights missing")?;
        let natural = capture.output.ok_or("natural H1 missing")?;
        let natural_start = (index * 8 + HEAD) * HEAD_DIM;
        natural_h1_mismatches += natural
            .iter()
            .zip(&natural_heads[natural_start..natural_start + HEAD_DIM])
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        let replay = replay_softmax_source_head(
            &query,
            &keys,
            &values,
            op.score_scale,
            op.logit_softcapping,
        )?;
        replay_mismatches += replay
            .values
            .iter()
            .zip(&natural)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        replay_mismatches += replay
            .weights
            .iter()
            .zip(&weights)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        write_f32(&mut query_out, &query)?;
        for key in &keys {
            write_f32(&mut keys_out, key)?;
        }
        for value in &values {
            write_f32(&mut values_out, value)?;
        }
        write_f32(&mut weights_out, &weights)?;
        write_f32(&mut heads_out, &natural)?;
        write_json_line(
            &mut rows_out,
            &json!({"row":index,"edge_id":input["edge_id"],"split":input["split"],"source_offset":source_offset,"source_count":keys.len(),"capture_position":target}),
        )?;
        source_offset += keys.len();
        if (index + 1).is_multiple_of(25) || index + 1 == ROWS {
            eprintln!("GW-KEY-1 source capture {}/{}", index + 1, ROWS);
        }
    }
    for output in [
        &mut query_out,
        &mut keys_out,
        &mut values_out,
        &mut weights_out,
        &mut heads_out,
        &mut rows_out,
    ] {
        output.flush()?;
    }
    drop((
        query_out,
        keys_out,
        values_out,
        weights_out,
        heads_out,
        rows_out,
    ));
    if natural_h1_mismatches != 0 || replay_mismatches != 0 {
        return Err("GW-KEY-1 observation or natural source replay parity failed".into());
    }
    let artifacts = [
        (&query_path, "f32-le", json!([ROWS, HEAD_DIM])),
        (&keys_path, "f32-le", json!([source_offset, HEAD_DIM])),
        (&values_path, "f32-le", json!([source_offset, HEAD_DIM])),
        (&weights_path, "f32-le", json!([source_offset])),
        (&heads_path, "f32-le", json!([ROWS, HEAD_DIM])),
        (&rows_path, "jsonl", json!([ROWS])),
    ].into_iter().map(|(path,dtype,shape)| Ok(json!({"path":path.file_name().and_then(|v|v.to_str()).ok_or("bad path")?,"dtype":dtype,"shape":shape,"bytes":std::fs::metadata(path)?.len(),"sha256":file_sha(path)?}))).collect::<Result<Vec<_>>>()?;
    let manifest = json!({
        "schema":OUTPUT_SCHEMA,"status":"natural_source_capture_complete_pre_search",
        "preregistration_sha256":prereg["preregistration_sha256"],
        "gwhead1_natural_capture_sha256":file_sha(head_capture_path)?,
        "site":{"layer":LAYER,"head":HEAD,"head_dim":HEAD_DIM,"source_rows":source_offset},
        "parity":{"gwhead_natural_h1_bit_mismatches":natural_h1_mismatches,"source_replay_bit_mismatches":replay_mismatches},
        "authorities":{"plan_sha256":plan_identity,"backend":backend.name()},
        "artifacts":artifacts,"train_search_performed":false,
    });
    atomic_json(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
