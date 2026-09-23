//! Lossless, bounded Standard recording over the canonical CPU decode tap.
//! Usage: observatory_record [--logit-lens | --heads] CONTAINER OUTPUT.json PROMPT [PROBE_TEXT ...]
//! Records one greedy output token; an independent noop pass judges bit parity.
//! This is a runner adapter, not a second interpreter or a live transport.
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use larql_inference::vindex3::{
    CarrierForm, CarrierWriteRecord, ExecutionProvenance, FixedBasis, HeadProbe, RunProvenance,
    StatsObserver, StepEvent, StepObserver, SublayerSite,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::{
    backend::PlanBackend,
    decode::DecodeSession,
    kv::RowKvState,
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

#[path = "gw0_batch.rs"]
mod gw0_batch;
#[path = "gw0_walk.rs"]
mod gw0_walk;
#[path = "gw0b_reconcile.rs"]
mod gw0b_reconcile;
#[path = "gw3af_postings.rs"]
mod gw3af_postings;
#[path = "gwconv1_readout.rs"]
mod gwconv1_readout;
#[path = "gwhead1_capture.rs"]
mod gwhead1_capture;
#[path = "gwhead1_heldout.rs"]
mod gwhead1_heldout;
#[path = "gwhead1_search.rs"]
mod gwhead1_search;
#[path = "gwkey1_capture.rs"]
mod gwkey1_capture;
#[path = "gwkey1_heldout.rs"]
mod gwkey1_heldout;
#[path = "gwkey1_search.rs"]
mod gwkey1_search;
#[path = "gwread1_capture.rs"]
mod gwread1_capture;
#[path = "gwread1_replay.rs"]
mod gwread1_replay;
#[path = "gwsup1_readout.rs"]
mod gwsup1_readout;
#[path = "gwv2.rs"]
mod gwv2;
#[path = "gwstate1.rs"]
mod gwstate1;
#[path = "observatory_heads.rs"]
mod heads;
#[path = "observatory_token_map.rs"]
mod token_map;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const SEED: u64 = 0x5eed;

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct CapturedCarrier {
    position: usize,
    layer: usize,
    site: SublayerSite,
    layer_scale: Option<f32>,
    after: Vec<f32>,
    delta: Vec<f32>,
}

struct Recorder {
    stats: StatsObserver,
    capture_lens: bool,
    capture_heads: bool,
    heads: Vec<heads::CapturedHead>,
    carriers: Vec<CapturedCarrier>,
    run_id: String,
    clock: Instant,
    events: Vec<Value>,
    position: usize,
    pending: Option<(usize, SublayerSite)>,
    writes: usize,
    boundaries: usize,
    error: Option<String>,
}

impl Recorder {
    fn new(stats: StatsObserver, run_id: String) -> Self {
        Self {
            stats,
            capture_lens: false,
            capture_heads: false,
            heads: Vec::new(),
            carriers: Vec::new(),
            run_id,
            clock: Instant::now(),
            events: Vec::new(),
            position: 0,
            pending: None,
            writes: 0,
            boundaries: 0,
            error: None,
        }
    }

    fn push(&mut self, kind: &str, mut body: Value) {
        body["kind"] = json!(kind);
        body["run_id"] = json!(self.run_id);
        body["sequence"] = json!(self.events.len().to_string());
        body["timestamp_ns"] = json!(self.clock.elapsed().as_nanos().to_string());
        self.events.push(body);
    }
}

impl StepObserver for Recorder {
    fn wants_attention_heads(&self) -> bool {
        self.capture_heads
    }
    fn attention_head(
        &mut self,
        layer: usize,
        record: larql_vindex::format::vindex3::opplan::exec::observe::AttentionHeadRecord<'_>,
    ) {
        self.heads.push(heads::CapturedHead::capture(layer, record));
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if self.pending.is_some() || record.position != self.position {
            self.error = Some("unpaired or out-of-position carrier callback".into());
        }
        self.pending = Some((record.layer, record.site));
        if self.capture_lens {
            self.carriers.push(CapturedCarrier {
                position: record.position,
                layer: record.layer,
                site: record.site,
                layer_scale: record.layer_scale,
                after: record.after.to_vec(),
                delta: if self.capture_heads {
                    record.delta.to_vec()
                } else {
                    Vec::new()
                },
            });
        }
        self.stats.carrier_write(record);
        let row = self.stats.rows.pop().expect("stats observer emits one row");
        if !row.norm.is_finite()
            || !row.delta_norm.is_finite()
            || row
                .projection
                .iter()
                .chain(&row.probe)
                .any(|v| !v.is_finite())
            || row.layer_scale.is_some_and(|v| !v.is_finite())
        {
            self.error = Some("non-finite capture cannot be represented as JSON evidence".into());
        }
        self.push(
            "CarrierWrite",
            json!({ "stats": {
                "layer": row.layer, "site": format!("{:?}", row.site), "position": row.position,
                "norm": row.norm, "delta_norm": row.delta_norm,
                "layer_scale": row.layer_scale.map(f64::from),
                "projection": row.projection.into_iter().map(f64::from).collect::<Vec<_>>(),
                "probe": row.probe.into_iter().map(f64::from).collect::<Vec<_>>()
            }}),
        );
    }

    fn event(&mut self, event: StepEvent) {
        match event {
            StepEvent::Embedded { position } => self.position = position,
            StepEvent::CarrierWrite {
                layer,
                site,
                carrier,
            } => {
                if carrier != CarrierForm::Single || self.pending.take() != Some((layer, site)) {
                    self.error = Some("carrier structure and captured values disagree".into());
                }
                self.writes += 1;
            }
            StepEvent::AttentionDone { layer } | StepEvent::FfnDone { layer } => {
                let ffn = matches!(event, StepEvent::FfnDone { .. });
                self.boundaries += usize::from(ffn);
                self.push(
                    "SiteCompleted",
                    json!({"position": self.position, "layer": layer,
                    "site": if ffn { "FfnDone" } else { "AttentionDone" }}),
                );
            }
            StepEvent::Logits { .. } => {}
            _ => self.error = Some("unsupported structural event; adapter upgrade required".into()),
        }
    }
}

/// Probe the predeclared STORED head rows. Only these rows are widened, not the
/// full vocabulary matrix. These are raw probes, independent of head lowering.
fn head_probe(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    ids: &[u32],
    hidden: usize,
) -> Result<Option<HeadProbe>> {
    if ids.is_empty() {
        return Ok(None);
    }
    Ok(Some(HeadProbe::new(
        ids.to_vec(),
        head_rows(plan, store, ids, hidden)?,
        hidden,
    )?))
}

fn head_rows(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    ids: &[u32],
    hidden: usize,
) -> Result<Vec<Vec<f32>>> {
    let operand = &plan.output.as_ref().ok_or("plan has no head")?.projection;
    if operand.shape.len() != 2 || operand.shape[1] != hidden {
        return Err("unsupported head geometry".into());
    }
    let dtype = store.stored_dtype(operand).ok_or("missing head dtype")?;
    let stride = match dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        _ => return Err("probe requires stored BF16/F16/F32 head rows".into()),
    };
    let region = store.map_region(operand, (operand.shape[0] * hidden * stride) as u64)?;
    let mut rows = Vec::new();
    for &id in ids {
        if id as usize >= operand.shape[0] {
            return Err("probe token outside head vocabulary".into());
        }
        let start = id as usize * hidden * stride;
        let bytes = &region.bytes()[start..start + hidden * stride];
        rows.push(match dtype {
            "BF16" => larql_models::quant::half::decode_bf16(bytes),
            "F16" => larql_models::quant::half::decode_f16(bytes),
            _ => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        });
    }
    Ok(rows)
}

fn atomic_json(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}

fn bit_parity(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
        })
}

fn vocabulary_row(
    logits: &[f32],
    tokenizer: &Tokenizer,
    targets: &[(u32, String)],
) -> Result<Value> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite or empty vocabulary readout".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = logits
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>();
    let log_z = z.ln();
    let probability = |id: usize| (f64::from(logits[id]) - max).exp() / z;
    let entropy = logits
        .iter()
        .enumerate()
        .map(|(id, &v)| probability(id) * (log_z - (f64::from(v) - max)))
        .sum::<f64>();
    let order = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b));
    let mut top: Vec<_> = (0..logits.len()).collect();
    let keep = 20.min(top.len());
    if keep < top.len() {
        top.select_nth_unstable_by(keep, order);
        top.truncate(keep);
    }
    top.sort_unstable_by(order);
    let prediction = |id: usize, rank: usize| -> Result<Value> {
        let decoded = tokenizer.decode(&[id as u32], false)?;
        let label = targets
            .iter()
            .find(|(target, _)| *target as usize == id)
            .map(|(_, label)| label.clone())
            .unwrap_or_else(|| {
                if decoded.is_empty() {
                    format!("<token:{id}>")
                } else {
                    decoded
                }
            });
        Ok(
            json!({ "token_id": id, "token": label, "logit": f64::from(logits[id]), "probability": probability(id), "rank": rank }),
        )
    };
    let top = top
        .iter()
        .enumerate()
        .map(|(rank, &id)| prediction(id, rank + 1))
        .collect::<Result<Vec<_>>>()?;
    let selected = targets
        .iter()
        .map(|(id, _)| {
            let id = *id as usize;
            if id >= logits.len() {
                return Err("lens target outside vocabulary".into());
            }
            let rank = (0..logits.len())
                .filter(|other| order(other, &id).is_lt())
                .count()
                + 1;
            prediction(id, rank)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"top": top, "targets": selected, "entropy": entropy}))
}

fn lens_rows(
    carriers: &[CapturedCarrier],
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    tokenizer: &Tokenizer,
    targets: &[(u32, String)],
    canonical: &[Vec<f32>],
) -> Result<(Vec<Value>, String, Vec<u8>)> {
    let mut rows = Vec::new();
    let mut exits = Vec::new();
    let mut payload = Vec::new();
    for (i, c) in carriers.iter().enumerate() {
        payload.extend(c.after.iter().flat_map(|v| v.to_le_bytes()));
        let mut state = c.after.clone();
        if let Some(scale) = c.layer_scale {
            backend.scale_row(&mut state, scale);
        }
        let logits = ops.readout_carrier(backend, &state)?;
        let last_at_position = carriers
            .get(i + 1)
            .is_none_or(|next| next.position != c.position);
        if last_at_position {
            let expected = canonical
                .get(c.position)
                .ok_or("missing canonical position")?;
            if !bit_parity(
                std::slice::from_ref(expected),
                std::slice::from_ref(&logits),
            ) {
                return Err(
                    "last-state lens differs from canonical output; publication refused".into(),
                );
            }
            exits.extend(logits.iter().flat_map(|v| v.to_le_bytes()));
        }
        let mut row = vocabulary_row(&logits, tokenizer, targets)?;
        let role = if c.site == SublayerSite::Attention {
            "attention_write"
        } else {
            "ffn_write"
        };
        row["site"] = json!(format!("{}:{}:{role}:main:residual", c.position, c.layer));
        rows.push(row);
        if (i + 1).is_multiple_of(40) {
            eprintln!("Vocabulary readout {}/{}", i + 1, carriers.len());
        }
    }
    if carriers.is_empty()
        || sha(&exits)
            != sha(&canonical
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>())
    {
        return Err("lens exit coverage mismatch".into());
    }
    Ok((rows, sha(&exits), payload))
}

fn main() -> Result<()> {
    let v2_args: Vec<String> = std::env::args().skip(1).collect();
    if v2_args.first().is_some_and(|arg| arg == "--gwv2") {
        return gwv2::run(&v2_args[1..]);
    }
    if v2_args.first().is_some_and(|arg| arg == "--gwstate1") {
        return gwstate1::run(&v2_args[1..]);
    }
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--gw0-batch") {
        return gw0_batch::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gw0-walk") {
        return gw0_walk::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gw0b-attribute") {
        return gw0b_reconcile::attribute(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gw0b-promote") {
        return gw0b_reconcile::promote(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gw3af-postings") {
        return gw3af_postings::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwsup1-readout") {
        return gwsup1_readout::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwconv1-readout") {
        return gwconv1_readout::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwhead1-capture") {
        return gwhead1_capture::run(&args[1..]);
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--gwhead1-train-search")
    {
        return gwhead1_search::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwhead1-heldout") {
        return gwhead1_heldout::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwkey1-capture") {
        return gwkey1_capture::run(&args[1..]);
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--gwkey1-train-search")
    {
        return gwkey1_search::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwkey1-heldout") {
        return gwkey1_heldout::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwread1-capture") {
        return gwread1_capture::run(&args[1..]);
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--gwread1-train-replay")
    {
        return gwread1_replay::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--gwread1-heldout") {
        return gwread1_replay::run_heldout(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "--token-map") {
        return token_map::run(&args[1..]);
    }
    let capture_heads = args.first().is_some_and(|arg| arg == "--heads");
    let capture_lens = capture_heads || args.first().is_some_and(|arg| arg == "--logit-lens");
    if capture_lens {
        args.remove(0);
    }
    if args.len() < 3 || args.len() > 11 {
        return Err("Usage: observatory_record [--logit-lens | --heads] CONTAINER OUTPUT.json PROMPT [PROBE_TEXT ...] (up to 8 single-token probes)".into());
    }
    let root = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let parity_path = output.with_extension("parity.json");
    let lens_path = output.with_extension("lenses.json");
    let carrier_path = output.with_extension("carriers.f32");
    let head_path = output.with_extension("heads.json");
    if capture_heads && head_path.exists() {
        return Err("head capture output exists".into());
    }
    if capture_lens
        && (lens_path.exists()
            || carrier_path.exists()
            || output == lens_path
            || output == carrier_path)
    {
        return Err("lens/carrier output exists or paths collide".into());
    }
    if output == parity_path || output.exists() || parity_path.exists() {
        return Err("record/parity output already exists or paths collide".into());
    }
    let tokenizer_bytes = std::fs::read(root.join("tokenizer.json"))?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
    let encoding = tokenizer.encode(args[2].as_str(), true)?;
    let tokens = encoding.get_ids();
    if tokens.is_empty() || tokens.len() > 32 {
        return Err("capture supports 1..32 prompt tokens".into());
    }
    let mut probe_ids = Vec::new();
    for text in &args[3..] {
        let encoded = tokenizer.encode(text.as_str(), false)?;
        if encoded.len() != 1 {
            return Err(format!("probe {text:?} must encode as exactly one token").into());
        }
        let id = encoded.get_ids()[0];
        if probe_ids.contains(&id) {
            return Err("duplicate probe token".into());
        }
        probe_ids.push(id);
    }
    eprintln!("Verifying container payloads and binding canonical target plan...");
    let inspection = inspect_container(root, true)?;
    if !inspection.is_coherent() {
        return Err(format!("container defects: {:?}", inspection.defects).into());
    }
    let outcome = plan_component_ops(&inspection, root, "target")?;
    if !outcome.closed() {
        return Err(format!("plan refused: {:?}", outcome.defects).into());
    }
    let plan = outcome.plan.ok_or("no closed plan")?;
    if !plan.residual_topology.is_single_stream()
        || plan.layers.is_empty()
        || plan.layers.len() > 256
    {
        return Err("adapter supports nonempty Single-carrier plans up to 256 layers".into());
    }
    let plan_hash = sha(&serde_json::to_vec(&plan)?);
    let container_hash = sha(&serde_json::to_vec(
        &json!({ "index": inspection.index, "graph": inspection.graph }),
    )?);
    let store = OperandStore::open(root, &inspection)?;
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)?;
    if capture_lens
        && ops
            .hidden()
            .checked_mul(tokens.len() * plan.layers.len() * 2 * 4)
            .is_none_or(|bytes| bytes > 64 * 1024 * 1024)
    {
        return Err("lens carrier capture exceeds the 64 MiB bound".into());
    }
    let expected_heads = if capture_heads {
        heads::preflight(&plan, tokens.len(), ops.hidden())?
    } else {
        0
    };
    let basis = FixedBasis::seeded(ops.hidden(), 3, SEED)?;
    let probe = head_probe(&plan, &store, &probe_ids, ops.hidden())?;
    let stats = StatsObserver::new(basis, probe);
    let provenance = RunProvenance::new(ExecutionProvenance::of(&ops), Some(&stats));
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_id = format!("standard-{stamp}-{}", std::process::id());
    let mut recorder = Recorder::new(stats, run_id.clone());
    recorder.capture_lens = capture_lens;
    recorder.capture_heads = capture_heads;
    recorder.push("RunStarted", json!({}));
    recorder.push("Tokenized", json!({}));
    let mut observed = Vec::new();
    eprintln!(
        "Recording {} positions through {} declared layers...",
        tokens.len(),
        plan.layers.len()
    );
    {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &token in tokens {
            observed.push(
                session
                    .step_observed(token, &mut recorder)?
                    .logits
                    .ok_or("missing output logits")?,
            );
            if let Some(error) = &recorder.error {
                return Err(error.clone().into());
            }
        }
    }
    let expected = plan
        .layers
        .iter()
        .map(|l| 1 + usize::from(l.ffn.is_some()))
        .sum::<usize>()
        * tokens.len();
    if recorder.pending.is_some()
        || recorder.writes != expected
        || recorder.boundaries != tokens.len() * plan.layers.len()
    {
        return Err("capture coverage differs from the bound program".into());
    }
    if recorder.heads.len() != expected_heads {
        return Err("head capture coverage mismatch".into());
    }
    let last = observed.last().ok_or("empty observed output")?;
    if last.iter().any(|v| !v.is_finite()) {
        return Err("non-finite output logits".into());
    }
    let best = last
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .ok_or("empty vocabulary")?
        .0 as u32;
    let text = tokenizer.decode(&[best], false)?;
    recorder.push(
        "TokenProduced",
        json!({"token_id": best, "predicting_position": tokens.len() - 1, "text": text}),
    );
    recorder.push("RunCompleted", json!({"reason": "one_greedy_token"}));

    eprintln!("Checking unobserved control (separate session, same prepared operands)...");
    let mut control = Vec::new();
    {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv)?;
        for &token in tokens {
            control.push(
                session
                    .step(token)?
                    .logits
                    .ok_or("missing control logits")?,
            );
        }
    }
    if !bit_parity(&observed, &control) {
        return Err("observed/unobserved logits differ; witness refused".into());
    }
    let mut basis = serde_json::to_value(recorder.stats.basis().identity())?;
    basis["source"] = json!(format!(
        "FixedBasis::seeded(seed={SEED}); axes have no semantic labels"
    ));
    let program: Vec<_> = plan.layers.iter().map(|l| {
        let mut sites = vec![json!({"site": "Attention", "operation_id": format!("{plan_hash}:L{}:Attention", l.layer)})];
        if l.ffn.is_some() { sites.push(json!({"site": "Ffn", "operation_id": format!("{plan_hash}:L{}:Ffn", l.layer)})); }
        json!({"layer": l.layer, "sites": sites})
    }).collect();
    let labels: Vec<_> = tokens.iter().enumerate().map(|(position, id)| json!({"position": position, "token_id": id, "label": encoding.get_tokens()[position]})).collect();
    let probes: Vec<_> = probe_ids
        .iter()
        .zip(&args[3..])
        .map(|(id, text)| json!({"token_id": id, "label": text}))
        .collect();
    let record = json!({
        "schema": "larql.observatory.standard.v1", "provenance": "executor", "run_id": run_id,
        "model": root.file_name().ok_or("container name missing")?.to_string_lossy(), "prompt": args[2],
        "identity": { "container": format!("sha256:{container_hash}"), "plan": format!("sha256:{plan_hash}"),
            "lowering": provenance.execution.fingerprint(), "tokenizer": format!("sha256:{}", sha(&tokenizer_bytes)), "session": run_id },
        "capture": "standard", "topology": "Single", "norm_method": provenance.norm_method, "probe_method": provenance.probe_method,
        "layers": plan.layers.len(), "program": program, "tokens": labels, "observed_positions": (0..tokens.len()).collect::<Vec<_>>(),
        "probe_tokens": probes, "basis": basis, "run_provenance": provenance,
        "probe_source": "predeclared stored output-head rows; no final norm, output multiplier or softcap",
        "runtime": { "timing_intrusive": true, "device_readbacks": 0 }, "coverage": "complete", "events": recorder.events,
        "supplemental_capture": if capture_heads { "intrusive per-head source weights and weighted V; owned post-add carriers; derived head content offline" } else if capture_lens { "owned post-add carriers for offline vocabulary readout; timing intrusive" } else { "none" },
        "recording_policy": "bounded in-memory lossless recorder; no live queue; timestamps include observer/serialization work",
        "runner": { "name": "observatory_record", "version": env!("CARGO_PKG_VERSION"),
            "executable_sha256": sha(&std::fs::read(std::env::current_exe()?)?),
            "adapter_source_sha256": sha(include_bytes!("observatory_record.rs")) },
        "container_identity_method": "sha256 of serialized inspected index and graph; all payload hashes verified before execution",
        "generation": { "strategy": "greedy", "max_new_tokens": 1, "tie_break": "lowest token id", "chat_template": false }
    });
    let bytes = serde_json::to_vec_pretty(&record)?;
    if bytes.len() > 10_000_000 {
        return Err("record exceeds the Observatory import limit".into());
    }
    let logits_bytes = |rows: &[Vec<f32>]| {
        rows.iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let parity = json!({"schema": "larql.observatory.parity-witness.v1", "run_id": run_id,
        "source_sha256": sha(&bytes), "criterion": "all finite vocabulary logits bit-identical at every prompt position",
        "positions": tokens.len(), "writes": expected, "observed_sha256": sha(&logits_bytes(&observed)),
        "unobserved_sha256": sha(&logits_bytes(&control)), "result": "pass",
        "scope": "same production CPU backend, same prepared image, fresh KV per arm; not HF parity, cross-backend parity, or a timing witness"});
    // Vocabulary analysis is a head-only readout of captured states, after the
    // canonical run and parity control. Never another transformer traversal.
    if capture_lens {
        let mut targets: Vec<(u32, String)> = probe_ids
            .iter()
            .copied()
            .zip(args[3..].iter().cloned())
            .collect();
        if !targets.iter().any(|(id, _)| *id == best) {
            targets.push((best, text.clone()));
        }
        let (rows, exit_hash, payload) = lens_rows(
            &recorder.carriers,
            &ops,
            &backend,
            &tokenizer,
            &targets,
            &observed,
        )?;
        let mut lens = json!({
            "schema": "larql.observatory.lenses.v1", "run_id": run_id,
            "source_sha256": sha(&bytes), "provenance": "executor",
            "provider": "observatory_record / prepared CPU output-head readout v1",
            "vocabulary": { "method": "Recorded post-add carrier → recorded layer scale (when present) → prepared final norm → pinned output head, multiplier and softcap → full-vocabulary f64 softmax. Intermediate readouts are projections, not future predictions.",
                "basis": format!("sha256:{container_hash}/sha256:{plan_hash}/{}", provenance.execution.fingerprint()),
                "vocab_size": last.len(), "rows": rows },
            "readout_witness": { "result": "pass", "positions": tokens.len(), "criterion": "last captured carrier readout equals canonical logits bit-for-bit at every prompt position", "canonical_logits_sha256": sha(&logits_bytes(&observed)), "readout_logits_sha256": exit_hash },
            "carrier_payload": { "file": carrier_path.file_name().unwrap().to_string_lossy(), "sha256": sha(&payload), "format": "f32-le", "rows": recorder.carriers.len(), "width": ops.hidden(), "order": "vocabulary.rows order; raw post-add/pre-scale values; scale taken from source CarrierWrite" },
            "capture": { "timing_intrusive": true, "device_readbacks": 0, "analysis_phase": "after execution", "model_traversals_for_visuals": 1 }
        });
        if capture_heads {
            let head_payload = serde_json::to_vec(&recorder.heads)?;
            atomic_json(&head_path, &head_payload)?;
            lens["head_payload"] = json!({"file": head_path.file_name().unwrap().to_string_lossy(), "sha256": sha(&head_payload), "rows": expected_heads, "values": "pre-W_O weighted V; actual source weights with window offset; f32 round-trip JSON"});
            lens["heads"] = heads::analyse(
                &recorder.heads,
                &recorder.carriers,
                &plan,
                &store,
                &targets,
                ops.hidden(),
                &format!("sha256:{container_hash}/sha256:{plan_hash}/stored-W_O-and-head-v1"),
            )?;
            lens["head_capture_witness"] = json!({"heads": expected_heads, "coverage": "complete",
                "tap": "canonical production softmax aggregation, before gate/W_O",
                "parity": "all prompt-position vocabulary logits bit-identical against fresh noop control",
                "analysis": "post-execution stored-weight direction probes; no additional model traversal"});
        }
        let lens_bytes = serde_json::to_vec(&lens)?;
        if lens_bytes.len() > 10_000_000 {
            return Err(format!("lens exceeds UI import limit: {} bytes", lens_bytes.len()).into());
        }
        atomic_json(&carrier_path, &payload)?;
        atomic_json(&lens_path, &lens_bytes)?;
        eprintln!(
            "Normalized lens: {} sites; final readout parity PASS; {}",
            recorder.carriers.len(),
            lens_path.display()
        );
    }
    atomic_json(output, &bytes)?;
    atomic_json(&parity_path, &serde_json::to_vec_pretty(&parity)?)?;
    println!(
        "{} writes; output {text:?}; parity PASS\n{}\n{}",
        expected,
        output.display(),
        parity_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_detects_one_bit_nonfinite_shape_and_signed_zero() {
        let a = vec![vec![1.0, 0.0]];
        assert!(bit_parity(&a, &a));
        for b in [
            vec![vec![f32::from_bits(1.0f32.to_bits() + 1), 0.0]],
            vec![vec![1.0, -0.0]],
            vec![vec![1.0]],
            vec![vec![f32::NAN, 0.0]],
        ] {
            assert!(!bit_parity(&a, &b));
        }
    }

    #[test]
    fn vocabulary_uses_full_mass_stable_ties_and_declared_targets() {
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let row =
            vocabulary_row(&[1000., 1000., -1000.], &tokenizer, &[(1, "target".into())]).unwrap();
        assert_eq!(row["top"][0]["token_id"], 0);
        assert_eq!(row["targets"][0]["rank"], 2);
        assert_eq!(row["targets"][0]["probability"], 0.5);
        assert_eq!(row["top"][1], row["targets"][0]);
        assert!((row["entropy"].as_f64().unwrap() - 2.0f64.ln()).abs() < 1e-12);
        assert_eq!(row["top"][2]["probability"], 0.0);
        assert!(vocabulary_row(&[f32::NAN], &tokenizer, &[]).is_err());
        assert!(vocabulary_row(&[1.], &tokenizer, &[(4, "bad".into())]).is_err());
    }

    #[test]
    fn callbacks_pair_once_and_ffn_boundary_does_not_invent_a_write() {
        let mut r = Recorder::new(
            StatsObserver::new(FixedBasis::seeded(3, 3, SEED).unwrap(), None),
            "test".into(),
        );
        r.event(StepEvent::Embedded { position: 0 });
        r.carrier_write(CarrierWriteRecord {
            layer: 0,
            site: SublayerSite::Attention,
            position: 0,
            delta: &[1., 2., 3.],
            after: &[2., 3., 4.],
            layer_scale: None,
        });
        r.event(StepEvent::CarrierWrite {
            layer: 0,
            site: SublayerSite::Attention,
            carrier: CarrierForm::Single,
        });
        r.event(StepEvent::FfnDone { layer: 0 });
        assert!(r.error.is_none());
        assert_eq!(r.writes, 1);
        assert_eq!(
            r.events
                .iter()
                .filter(|e| e["kind"] == "CarrierWrite")
                .count(),
            1
        );
        r.event(StepEvent::CarrierWrite {
            layer: 0,
            site: SublayerSite::Ffn,
            carrier: CarrierForm::Single,
        });
        assert!(r.error.is_some());
    }
}
