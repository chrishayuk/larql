//! BLOCK-TIME-1 trial: one arm, one process, one pass over supplied ids.
//!
//! Runs a supplied token stream through remote routed experts either one
//! position at a time (`sequential`) or through the routed block session at
//! a fixed width (`block-W`), and writes what happened: wall time of the
//! measured region and of each step or block in it, every exchange's
//! coordinator-side round trip and frame bytes, frame counts per route, and
//! SHA-256 and argmax of every position's logits.
//!
//! It judges nothing. Parity, frame-count, bracket and relay gates and all
//! analysis belong to the driver (`bench/v3-routed-experts/block_time.py`),
//! which reads this file's output. No token is sampled: the stream is
//! supplied, so every arm executes the same positions.
//!
//! The container is opened as `larql run` and the worker open it: stored
//! bytes, nothing requantised at load. With `--profile yes` each exchange
//! also asks the worker for its own timing; the frame is the same either
//! way, and the driver checks the logits are.
//!
//! Protocol: `bench/v3-routed-experts/BLOCK-TIME-1.md`.
//!
//! ```text
//! v3_block_time --model <container> --shards <url[,url]> --ids <ids.json> \
//!     --arm sequential|block-1|block-2|block-4|block-8 \
//!     --measure-from <position> --profile yes|no --out <trial.json>
//! ```
use larql_inference::vindex3::routed_block::RoutedBlockSession;
use larql_inference::vindex3::routed_experts::{self, ExpertOutput, ExpertTransport};
use larql_inference::vindex3::{OpenPolicy, Vindex3Runtime};
use larql_router::vindex3_experts::HttpExpertShards;
use larql_router_protocol::provider_calls::{self, ProviderCallObserver};
use larql_router_protocol::{
    vindex3_expert_rows as rows_wire, vindex3_experts as wire, vindex3_experts::Binding,
};
use larql_vindex::format::vindex3::opplan::exec::{
    continuation::plan_continuation_geometry, continuation_authority::ContinuationConfig,
    decode::DecodeSession, kv::RowKvState, operands::RepresentationSource,
    production::ProductionBackend,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type BoxErr = Box<dyn std::error::Error>;

/// The component a text container encodes its stack under.
const COMPONENT: &str = "target";
const SEQUENTIAL: &str = "sequential";
const BLOCK_PREFIX: &str = "block-";
/// Block widths the protocol admits.
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
const SINGLE_ROW: &str = "single-row";
const MULTI_ROW: &str = "multi-row";
/// `wire` values of the transport's per-request profile records.
const PROFILE_WIRES: [&str; 2] = ["experts-binary-f32-v1", "expert-rows-f32-v1"];

/// One exchange as the coordinator saw it.
struct Exchange {
    route: &'static str,
    layer: usize,
    rows: usize,
    pairs: usize,
    request_bytes: usize,
    response_bytes: usize,
    roundtrip_ns: u64,
}

impl Exchange {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "route": self.route, "layer": self.layer, "rows": self.rows, "pairs": self.pairs,
            "request_bytes": self.request_bytes, "response_bytes": self.response_bytes,
            "roundtrip_ns": self.roundtrip_ns,
        })
    }
}

/// The HTTP transport, timing each exchange and noting its frame sizes.
struct Timed {
    inner: HttpExpertShards,
    exchanges: Arc<Mutex<Vec<Exchange>>>,
}

impl ExpertTransport for Timed {
    fn bindings(&self) -> Vec<Binding> {
        self.inner.bindings()
    }

    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        let started = Instant::now();
        let outputs = self.inner.forward(shard, layer, experts, row)?;
        let roundtrip_ns = started.elapsed().as_nanos() as u64;
        self.exchanges.lock().unwrap().push(Exchange {
            route: SINGLE_ROW,
            layer,
            rows: 1,
            pairs: experts.len(),
            request_bytes: wire::HEADER_BYTES + 4 * experts.len() + 4 * row.len(),
            response_bytes: wire::response_len(row.len(), experts.len())?,
            roundtrip_ns,
        });
        Ok(outputs)
    }

    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[rows_wire::Row],
    ) -> Result<Vec<rows_wire::Output>, String> {
        let started = Instant::now();
        let outputs = self.inner.forward_rows(shard, layer, rows)?;
        let roundtrip_ns = started.elapsed().as_nanos() as u64;
        let pairs: usize = rows.iter().map(|r| r.experts.len()).sum();
        let hidden = rows.first().map_or(0, |r| r.input.len());
        self.exchanges.lock().unwrap().push(Exchange {
            route: MULTI_ROW,
            layer,
            rows: rows.len(),
            pairs,
            request_bytes: rows_wire::request_len(hidden, rows.len(), pairs)?,
            response_bytes: rows_wire::response_len(hidden, pairs)?,
            roundtrip_ns,
        });
        Ok(outputs)
    }
}

/// Collects the transport's own profile records, which carry the worker's
/// timing. Installed only for a profiled trial.
struct Recorder {
    on: AtomicBool,
    records: Mutex<Vec<serde_json::Value>>,
}

impl ProviderCallObserver for Recorder {
    fn enabled(&self) -> bool {
        self.on.load(Ordering::SeqCst)
    }

    fn record(&self, call: serde_json::Value) {
        if PROFILE_WIRES.iter().any(|w| call["wire"] == *w) {
            self.records.lock().unwrap().push(call);
        }
    }
}

struct Args {
    model: PathBuf,
    shards: Vec<String>,
    ids: PathBuf,
    arm: String,
    /// `None` for `sequential`.
    width: Option<usize>,
    measure_from: usize,
    profile: bool,
    out: PathBuf,
}

fn parse_args() -> Result<Args, BoxErr> {
    let mut values = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        values.insert(flag, value);
    }
    let mut take = |flag: &str| values.remove(flag).ok_or(format!("{flag} is required"));
    let arm = take("--arm")?;
    let width = match arm.strip_prefix(BLOCK_PREFIX) {
        Some(width) => Some(width.parse::<usize>()?),
        None if arm == SEQUENTIAL => None,
        None => return Err(format!("unknown arm {arm}").into()),
    };
    if width.is_some_and(|w| !WIDTHS.contains(&w)) {
        return Err(format!("block width must be one of {WIDTHS:?}").into());
    }
    let profile = match take("--profile")?.as_str() {
        "yes" => true,
        "no" => false,
        other => return Err(format!("--profile takes yes or no, not {other}").into()),
    };
    let parsed = Args {
        model: take("--model")?.into(),
        shards: take("--shards")?.split(',').map(str::to_string).collect(),
        ids: take("--ids")?.into(),
        measure_from: take("--measure-from")?.parse()?,
        out: take("--out")?.into(),
        arm,
        width,
        profile,
    };
    match values.keys().next() {
        Some(unknown) => Err(format!("unknown flag {unknown}").into()),
        None => Ok(parsed),
    }
}

fn ids_of(value: &serde_json::Value, key: &str) -> Result<Vec<u32>, BoxErr> {
    value[key]
        .as_array()
        .ok_or(format!("ids file has no `{key}` list"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|id| id as u32)
                .ok_or_else(|| "a token id is not a number".into())
        })
        .collect()
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best as u32
}

fn sha256_bits(row: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in row {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() -> Result<(), BoxErr> {
    let args = parse_args()?;
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.ids)?)?;
    let prompt = ids_of(&saved, "prompt")?;
    let mut stream = prompt.clone();
    stream.extend(ids_of(&saved, "generated")?);
    let unit = args.width.unwrap_or(1);
    if args.measure_from >= stream.len()
        || args.measure_from % unit != 0
        || (stream.len() - args.measure_from) % unit != 0
    {
        return Err("the untimed and measured regions must each be whole blocks".into());
    }

    // Before the coordinator is prepared, so this observer is the one the
    // transport reports to. Left uninstalled for an unprofiled trial.
    let recorder: &'static Recorder = Box::leak(Box::new(Recorder {
        on: AtomicBool::new(args.profile),
        records: Mutex::new(Vec::new()),
    }));
    if args.profile && !provider_calls::install_observer(recorder) {
        return Err("another profile observer is already installed".into());
    }

    let policy = OpenPolicy {
        want: None,
        source: RepresentationSource::Stored,
    };
    let runtime =
        Vindex3Runtime::open_with(&args.model, COMPONENT, ProductionBackend::new(), policy)?;
    let exchanges = Arc::new(Mutex::new(Vec::new()));
    let transport = Timed {
        inner: HttpExpertShards::connect(&args.shards, None)?,
        exchanges: exchanges.clone(),
    };
    let ops = routed_experts::prepare_coordinator(
        &args.model,
        runtime.plan(),
        runtime.operands(),
        runtime.backend(),
        transport,
    )?;
    let geometry = plan_continuation_geometry(runtime.plan())?;
    let continuation = larql_kv::shipped_continuations().select(
        &RowKvState::identity(),
        &ContinuationConfig::empty(),
        &geometry,
    )?;

    // Every position's logits, kept until the clock has stopped.
    let mut logits: Vec<Vec<f32>> = Vec::with_capacity(stream.len());
    let mut unit_ns: Vec<u64> = Vec::new();
    let (mut exchanges_before, mut records_before) = (0, 0);
    let mut measured = None;
    let mark = |exchanges_before: &mut usize, records_before: &mut usize| {
        *exchanges_before = exchanges.lock().unwrap().len();
        *records_before = recorder.records.lock().unwrap().len();
        Instant::now()
    };
    match args.width {
        None => {
            let mut state = continuation.build();
            let mut session =
                DecodeSession::over_prepared(runtime.plan(), &ops, runtime.backend(), &mut *state)?;
            for (position, token) in stream.iter().enumerate() {
                if position == args.measure_from {
                    measured = Some(mark(&mut exchanges_before, &mut records_before));
                }
                let started = Instant::now();
                let row = session.step(*token)?.logits;
                if position >= args.measure_from {
                    unit_ns.push(started.elapsed().as_nanos() as u64);
                }
                logits.push(row.ok_or("a position produced no logits")?);
            }
        }
        Some(width) => {
            let mut session =
                RoutedBlockSession::new(runtime.plan(), &ops, runtime.backend(), &continuation)?;
            for (index, block) in stream.chunks(width).enumerate() {
                let position = index * width;
                if position == args.measure_from {
                    measured = Some(mark(&mut exchanges_before, &mut records_before));
                }
                let started = Instant::now();
                let rows = session.verify(block)?;
                session.commit(block.len())?;
                if position >= args.measure_from {
                    unit_ns.push(started.elapsed().as_nanos() as u64);
                }
                logits.extend(rows);
            }
        }
    }
    let wall_ns = measured
        .ok_or("the measured region never started")?
        .elapsed()
        .as_nanos() as u64;
    recorder.on.store(false, Ordering::SeqCst);

    let exchanges = exchanges.lock().unwrap();
    let count = |from: usize, route: &str| {
        exchanges[from..]
            .iter()
            .filter(|e| e.route == route)
            .count()
    };
    let records = recorder.records.lock().unwrap();
    let trial = serde_json::json!({
        "schema": "larql.v3.block-time-trial.v1",
        "arm": args.arm,
        "width": args.width,
        "profiled": args.profile,
        "open_policy": { "want": null, "source": "stored" },
        "model": args.model.display().to_string(),
        "shards": args.shards,
        "prompt_tokens": prompt.len(),
        "positions": stream.len(),
        "measure_from": args.measure_from,
        "measured_positions": stream.len() - args.measure_from,
        "clock": "monotonic wall time around the measured region; load, the untimed positions \
                  and hashing excluded",
        "wall_ns": wall_ns,
        "unit": if args.width.is_some() { "block" } else { "position" },
        "unit_ns": unit_ns,
        "frames": {
            "whole_run": { SINGLE_ROW: count(0, SINGLE_ROW), MULTI_ROW: count(0, MULTI_ROW) },
            "measured": {
                SINGLE_ROW: count(exchanges_before, SINGLE_ROW),
                MULTI_ROW: count(exchanges_before, MULTI_ROW),
            },
        },
        "exchanges": exchanges[exchanges_before..].iter().map(Exchange::json).collect::<Vec<_>>(),
        "worker_records": &records[records_before..],
        "argmax": logits.iter().map(|row| argmax(row)).collect::<Vec<_>>(),
        "logits_sha256": logits.iter().map(|row| sha256_bits(row)).collect::<Vec<_>>(),
    });
    std::fs::write(&args.out, serde_json::to_vec(&trial)?)?;
    Ok(())
}
