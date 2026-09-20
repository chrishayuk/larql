//! Real-subject witnesses for the frozen V3-HEAD-OBS-1 HP5 and HP7 gates.
//!
//! This is deliberately a witness, not another execution path. It opens one
//! prepared production-CPU image and drives the canonical tokenwise session.
//! `bench` compares `NoopObserver`, a borrowed tap which only consumes the
//! records, and the shipped `HeadStats` HP3 reconstruction. `denmark` computes
//! the one pre-registered HEAD-2 reading in process; no full vectors are added
//! to the persisted observation schema.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use larql_inference::vindex3::{HeadReader, HeadStats, LogitsSession, Vindex3Runtime};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, NoopObserver, StepEvent, StepObserver, SublayerSite,
};
use larql_vindex::format::vindex3::opplan::exec::observe_heads::HeadWrite;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use serde_json::{json, Value};

type BoxErr = Box<dyn std::error::Error>;

const PROMPT: &str = "The capital of France is";
const RECIPIENT: &str = "The currency of Denmark is";
const DONOR: &str = "The capital of Denmark is";
const TARGET_LAYER: usize = 14;
const GENERATED: usize = 8;
const MAX_PLATEAU_RUNS: usize = 12;
const MAX_BRACKETS: usize = 10;
// One valid A/B/A block is the repository's measurement unit. More can be
// requested in a later characterization run; HP5 asks for a measured cost,
// not a variance estimate.
const REQUIRED_VALID_BRACKETS: usize = 1;

fn main() -> Result<(), BoxErr> {
    let mut args = std::env::args_os().skip(1);
    let mode = args
        .next()
        .ok_or("usage: head_obs_witness <bench|denmark> CONTAINER [OUTPUT]")?;
    let container = PathBuf::from(
        args.next()
            .ok_or("usage: head_obs_witness <bench|denmark> CONTAINER [OUTPUT]")?,
    );
    let backend = ProductionBackend::new();
    let image = Vindex3Runtime::open(&container, "target", backend)?.prepare()?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&container)?;

    match mode.to_string_lossy().as_ref() {
        "bench" => {
            let output = args.next().map(PathBuf::from);
            bench(&image, &tokenizer, output.as_deref())
        }
        "denmark" => {
            let output = args.next().map(PathBuf::from);
            denmark(&image, &tokenizer, output.as_deref())
        }
        other => Err(format!("unknown mode `{other}`; expected `bench` or `denmark`").into()),
    }
}

fn encode(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>, BoxErr> {
    Ok(tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("tokenising {prompt:?}: {e}"))?
        .get_ids()
        .to_vec())
}

fn greedy_fixture(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<u32>, BoxErr> {
    let mut tokens = encode(tokenizer, PROMPT)?;
    let mut kv = RowKvState::default();
    let mut session = image.session_with_kv(&mut kv)?;
    let mut logits = session.prefill(&tokens)?;
    for _ in 0..GENERATED {
        let next = argmax(&logits);
        tokens.push(next);
        logits = session.step(next)?;
    }
    Ok(tokens)
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate().skip(1) {
        if value > &values[best] {
            best = index;
        }
    }
    u32::try_from(best).expect("a vocabulary index fits in u32")
}

#[derive(Default)]
struct ReadOnlyTap {
    records: usize,
}

impl StepObserver for ReadOnlyTap {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.records += 1;
        // Force the borrowed fields to be consumed without copying, projecting,
        // sorting, or otherwise turning this arm into an attribution reader.
        black_box(layer);
        black_box(record.head);
        black_box(record.kv_head);
        black_box(record.source_start);
        black_box(record.weights);
        black_box(record.sink);
        black_box(record.values);
        black_box(record.gate);
        black_box(record.source_values);
    }
}

#[derive(Debug, Clone, Copy)]
enum Arm {
    Noop,
    ReadOnly,
    Reconstruct,
}

fn timed(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokens: &[u32],
    arm: Arm,
) -> Result<(Duration, usize, f64), BoxErr> {
    let mut kv = RowKvState::default();
    let mut session = image.session_with_kv(&mut kv)?;
    let started = Instant::now();
    let (records, residual) = match arm {
        Arm::Noop => {
            let mut observer = NoopObserver;
            for &token in tokens {
                black_box(session.step_observed(token, &mut observer)?);
            }
            (0, 0.0)
        }
        Arm::ReadOnly => {
            let mut observer = ReadOnlyTap::default();
            for &token in tokens {
                black_box(session.step_observed(token, &mut observer)?);
            }
            (observer.records, 0.0)
        }
        Arm::Reconstruct => {
            let mut observer =
                HeadStats::new(image.operands(), image.plan(), image.backend(), None, 0);
            for &token in tokens {
                black_box(session.step_observed(token, &mut observer)?);
            }
            if let Some(error) = &observer.failure {
                return Err(format!("head reconstruction failed: {error}").into());
            }
            let residual = observer
                .writes
                .iter()
                .map(|write| write.residual)
                .fold(0.0f64, f64::max);
            (observer.records, residual)
        }
    };
    Ok((started.elapsed(), records, residual))
}

fn bench(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokenizer: &tokenizers::Tokenizer,
    output: Option<&Path>,
) -> Result<(), BoxErr> {
    let tokens = greedy_fixture(image, tokenizer)?;
    println!(
        "model={} family={} positions={} ids={:?}",
        image.model_name(),
        image.family(),
        tokens.len(),
        tokens
    );

    // Warm every arm over the same resident image and fixed token stream,
    // then require the baseline itself to reach the repository's 1% plateau.
    for arm in [Arm::Noop, Arm::ReadOnly, Arm::Reconstruct] {
        black_box(timed(image, &tokens, arm)?);
    }
    let plateau = warm_to_plateau(image, &tokens)?;
    let read_only = bracket(image, &tokens, Arm::ReadOnly, "read-only")?;
    let reconstruct = bracket(image, &tokens, Arm::Reconstruct, "reconstruct")?;
    if let Some(path) = output {
        let result = json!({
            "gate": "HP5",
            "model": image.model_name(),
            "family": image.family(),
            "backend": image.backend().name(),
            "cpu_max_format": std::env::var("LARQL_CPU_MAX_FORMAT").ok(),
            "prompt": PROMPT,
            "positions": tokens.len(),
            "token_ids": tokens,
            "plateau_seconds": plateau,
            "arms": {
                "read_only": read_only,
                "reconstruct": reconstruct
            }
        });
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        println!("wrote={}", path.display());
    }
    Ok(())
}

fn warm_to_plateau(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokens: &[u32],
) -> Result<Vec<f64>, BoxErr> {
    let mut previous: Option<Duration> = None;
    let mut walls = Vec::new();
    for attempt in 1..=MAX_PLATEAU_RUNS {
        let (wall, _, _) = timed(image, tokens, Arm::Noop)?;
        walls.push(wall.as_secs_f64());
        if let Some(before) = previous {
            let midpoint = (before.as_secs_f64() + wall.as_secs_f64()) / 2.0;
            let drift = (before.as_secs_f64() - wall.as_secs_f64()).abs() / midpoint;
            println!(
                "plateau attempt={attempt} wall_s={:.9} prior_drift={drift:.6}",
                wall.as_secs_f64()
            );
            if drift <= 0.01 {
                return Ok(walls);
            }
        } else {
            println!("plateau attempt={attempt} wall_s={:.9}", wall.as_secs_f64());
        }
        previous = Some(wall);
    }
    Err(format!("baseline did not reach a 1% plateau in {MAX_PLATEAU_RUNS} runs").into())
}

fn bracket(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokens: &[u32],
    candidate: Arm,
    name: &str,
) -> Result<Vec<Value>, BoxErr> {
    let mut valid = 0usize;
    let mut attempts = Vec::new();
    for attempt in 1..=MAX_BRACKETS {
        let (a1, _, _) = timed(image, tokens, Arm::Noop)?;
        let (b, records, residual) = timed(image, tokens, candidate)?;
        let (a2, _, _) = timed(image, tokens, Arm::Noop)?;
        let baseline = (a1.as_secs_f64() + a2.as_secs_f64()) / 2.0;
        let drift = (a1.as_secs_f64() - a2.as_secs_f64()).abs() / baseline;
        let overhead = b.as_secs_f64() / baseline - 1.0;
        let accepted = drift <= 0.01;
        println!(
            "arm={name} attempt={attempt} accepted={accepted} a1_s={:.9} b_s={:.9} \
             a2_s={:.9} baseline_drift={:.6} overhead={:.6} records={records} \
             worst_residual={residual:.9e}",
            a1.as_secs_f64(),
            b.as_secs_f64(),
            a2.as_secs_f64(),
            drift,
            overhead,
        );
        attempts.push(json!({
            "attempt": attempt,
            "accepted": accepted,
            "baseline_before_seconds": a1.as_secs_f64(),
            "candidate_seconds": b.as_secs_f64(),
            "baseline_after_seconds": a2.as_secs_f64(),
            "baseline_drift": drift,
            "overhead": overhead,
            "records": records,
            "worst_residual": residual
        }));
        if accepted {
            valid += 1;
            if valid == REQUIRED_VALID_BRACKETS {
                return Ok(attempts);
            }
        }
    }
    Err(format!("{name}: obtained {valid} valid brackets, need {REQUIRED_VALID_BRACKETS}").into())
}

#[derive(Clone)]
struct OwnedHead {
    head: usize,
    kv_head: usize,
    source_start: usize,
    weights: Vec<f32>,
    values: Vec<f32>,
    gate: Option<Vec<f32>>,
    source_values: Vec<Vec<f32>>,
}

struct PairObserver<'a> {
    stats: HeadStats<'a, ProductionBackend>,
    target_position: usize,
    raw: Vec<OwnedHead>,
    write: Option<HeadWrite>,
}

impl HeadReader for PairObserver<'_> {
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer == TARGET_LAYER && record.position == self.target_position {
            self.raw.push(OwnedHead {
                head: record.head,
                kv_head: record.kv_head,
                source_start: record.source_start,
                weights: record.weights.to_vec(),
                values: record.values.to_vec(),
                gate: record.gate.map(<[f32]>::to_vec),
                source_values: record
                    .source_values
                    .iter()
                    .map(|row| row.to_vec())
                    .collect(),
            });
        }
        HeadReader::attention_head(&mut self.stats, layer, record);
    }

    fn finish_write(&mut self, record: &CarrierWriteRecord<'_>) -> Option<HeadWrite> {
        self.stats.finish_write(record)
    }

    fn records(&self) -> usize {
        self.stats.records()
    }

    fn failure(&self) -> Option<&larql_vindex::error::VindexError> {
        self.stats.failure()
    }
}

impl StepObserver for PairObserver<'_> {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        HeadReader::attention_head(self, layer, record);
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let target = record.layer == TARGET_LAYER
            && record.site == SublayerSite::Attention
            && record.position == self.target_position;
        if let Some(write) = HeadReader::finish_write(self, &record) {
            if target {
                self.write = Some(write);
            }
        }
    }
}

struct PromptCapture {
    ids: Vec<u32>,
    raw: Vec<OwnedHead>,
    write: HeadWrite,
}

fn capture(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<PromptCapture, BoxErr> {
    let ids = encode(tokenizer, prompt)?;
    let target_position = ids.len() - 1;
    let stats = HeadStats::new(image.operands(), image.plan(), image.backend(), None, 0)
        .retaining_children();
    let mut observer = PairObserver {
        stats,
        target_position,
        raw: Vec::new(),
        write: None,
    };
    let mut kv = RowKvState::default();
    let mut session = image.session_with_kv(&mut kv)?;
    for &token in &ids {
        black_box(session.step_observed(token, &mut observer)?);
    }
    if let Some(error) = observer.failure() {
        return Err(format!("head reader failed: {error}").into());
    }
    let write = observer.write.ok_or("target head write was not observed")?;
    Ok(PromptCapture {
        ids,
        raw: observer.raw,
        write,
    })
}

fn denmark(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    tokenizer: &tokenizers::Tokenizer,
    output: Option<&Path>,
) -> Result<(), BoxErr> {
    let recipient = capture(image, tokenizer, RECIPIENT)?;
    let donor = capture(image, tokenizer, DONOR)?;
    if recipient.ids.len() != donor.ids.len() {
        return Err("the sealed pair no longer has aligned token positions".into());
    }
    // BOS / RELATION / ENTITY / LAST for the six-token sealed templates.
    let entity = recipient.ids.len() - 2;
    let last = recipient.ids.len() - 1;
    let roles: Vec<&str> = (0..recipient.ids.len())
        .map(|position| match position {
            0 => "BOS",
            p if p == entity => "ENTITY",
            p if p == last => "LAST",
            _ => "RELATION",
        })
        .collect();

    let a = role_vectors(image, &recipient, &roles)?;
    let b = role_vectors(image, &donor, &roles)?;
    let mut rows = Vec::new();
    let labels = [
        "carrier TWO_ROLES(RELATION,ENTITY)",
        "SINGLE_ROLE(ENTITY)",
        "opposer MIXED_ROLES",
        "carrier SINGLE_ROLE(RELATION)",
        "SINGLE_ROLE(RELATION)[DIVERGE]",
        "SINGLE_ROLE(RELATION)[DIVERGE]",
        "carrier TWO_ROLES(RELATION,BOS)",
        "opposer SINGLE_ROLE(LAST)",
    ];
    println!("recipient_ids={:?}", recipient.ids);
    println!("donor_ids={:?}", donor.ids);
    println!("head  kv  BOS       RELATION  ENTITY    LAST      sum       sealed");
    for head in 0..a.len() {
        let delta = sub(&b[head].whole, &a[head].whole);
        let denom = dot(&delta, &delta);
        let mut shares = Vec::new();
        for role in 0..4 {
            let d = sub(&b[head].roles[role], &a[head].roles[role]);
            shares.push(dot(&d, &delta) / denom);
        }
        let sum: f64 = shares.iter().sum();
        println!(
            "H{head:<3} {:<3} {:+.6} {:+.6} {:+.6} {:+.6} {:+.6}  {}",
            recipient.raw[head].kv_head,
            shares[0],
            shares[1],
            shares[2],
            shares[3],
            sum,
            labels[head]
        );
        rows.push(json!({
            "head": head,
            "kv_head": recipient.raw[head].kv_head,
            "sealed_label": labels[head],
            "delta_norm_sq": denom,
            "projection_share": {
                "BOS": shares[0],
                "RELATION": shares[1],
                "ENTITY": shares[2],
                "LAST": shares[3],
                "sum": sum
            },
            "source_sum_relative": {
                "recipient": a[head].source_residual,
                "donor": b[head].source_residual
            },
            "attention_mass": {
                "recipient": a[head].mass,
                "donor": b[head].mass
            }
        }));
    }
    let result = json!({
        "gate": "HP7",
        "model": image.model_name(),
        "family": image.family(),
        "backend": image.backend().name(),
        "cpu_max_format": std::env::var("LARQL_CPU_MAX_FORMAT").ok(),
        "layer": TARGET_LAYER,
        "recipient": RECIPIENT,
        "donor": DONOR,
        "recipient_ids": recipient.ids,
        "donor_ids": donor.ids,
        "roles_by_position": roles,
        "worst_head_sum_residual": {
            "recipient": recipient.write.residual,
            "donor": donor.write.residual
        },
        "heads": rows
    });
    if let Some(path) = output {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        println!("wrote={}", path.display());
    } else {
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    Ok(())
}

struct RoleVectors {
    whole: Vec<f32>,
    roles: [Vec<f32>; 4],
    mass: Value,
    source_residual: f64,
}

fn role_vectors(
    image: &larql_inference::vindex3::PreparedVindex3<ProductionBackend>,
    capture: &PromptCapture,
    roles: &[&str],
) -> Result<Vec<RoleVectors>, BoxErr> {
    let children = capture
        .write
        .children
        .as_ref()
        .ok_or("the witness reader did not retain head children")?;
    let op = image.plan().layers[TARGET_LAYER]
        .attention
        .softmax()
        .ok_or("target layer is not softmax attention")?;
    let mut raw_whole = Vec::new();
    for head in &capture.raw {
        let input = gated(&head.values, head.gate.as_deref());
        raw_whole.push(image.operands().head_projection(
            image.backend(),
            TARGET_LAYER,
            head.head,
            op.head_dim,
            op.num_q_heads,
            &input,
        )?);
    }
    // The post-attention RMS transform is diagonal and shared by every
    // head at this write. Recover each diagonal entry from the head with
    // the largest raw magnitude there, then validate it by rebuilding
    // every normalized head from its sources below.
    let hidden = children[0].len();
    let mut scale = vec![0.0f32; hidden];
    for i in 0..hidden {
        let head = (0..raw_whole.len())
            .max_by(|a, b| {
                raw_whole[*a][i]
                    .abs()
                    .partial_cmp(&raw_whole[*b][i].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("at least one head");
        let raw = raw_whole[head][i];
        scale[i] = if raw.abs() > 1e-12 {
            children[head][i] / raw
        } else {
            0.0
        };
    }

    capture
        .raw
        .iter()
        .enumerate()
        .map(|(head_index, head)| {
            let mut grouped: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; hidden]);
            let mut mass = [0.0f64; 4];
            for (source, (&weight, value)) in
                head.weights.iter().zip(&head.source_values).enumerate()
            {
                let role = role_index(roles[head.source_start + source]);
                mass[role] += f64::from(weight);
                let weighted: Vec<f32> = value.iter().map(|v| weight * v).collect();
                let input = gated(&weighted, head.gate.as_deref());
                let raw = image.operands().head_projection(
                    image.backend(),
                    TARGET_LAYER,
                    head.head,
                    op.head_dim,
                    op.num_q_heads,
                    &input,
                )?;
                for i in 0..hidden {
                    grouped[role][i] += raw[i] * scale[i];
                }
            }
            let rebuilt: Vec<f32> = (0..hidden)
                .map(|i| grouped.iter().map(|role| role[i]).sum())
                .collect();
            let source_residual = relative(&rebuilt, &children[head_index]);
            Ok(RoleVectors {
                whole: children[head_index].clone(),
                roles: grouped,
                mass: json!({
                    "BOS": mass[0],
                    "RELATION": mass[1],
                    "ENTITY": mass[2],
                    "LAST": mass[3]
                }),
                source_residual,
            })
        })
        .collect()
}

fn gated(values: &[f32], gate: Option<&[f32]>) -> Vec<f32> {
    match gate {
        Some(gate) => values
            .iter()
            .zip(gate)
            .map(|(value, gate)| value * gate)
            .collect(),
        None => values.to_vec(),
    }
}

fn role_index(role: &str) -> usize {
    match role {
        "BOS" => 0,
        "RELATION" => 1,
        "ENTITY" => 2,
        "LAST" => 3,
        _ => unreachable!("the roles are closed"),
    }
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn relative(a: &[f32], b: &[f32]) -> f64 {
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = b
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    error / norm.max(1e-30)
}
