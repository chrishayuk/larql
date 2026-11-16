//! VERIFY-REMOTE-1 trial: one arm, one process, one greedy generation.
//!
//! Prefills a workload's prompt through remote routed experts in untimed
//! blocks of eight, then generates greedily, either one position at a time
//! (`sequential`) or by prompt-lookup speculation verified in blocks of at
//! most `W` positions (`verify-W`). It writes what happened: the generated
//! ids; every unit of target work (a verify block or a single step) with
//! its width, committed count, wall time and the exchanges and worker
//! records it caused; for speculative arms, the event ledger, the blocks
//! and whether the ledger reconciles; and the generation's wall time.
//!
//! It judges nothing. Identity with `sequential`, the ledger, frame
//! accounting, bracket and relay gates and all analysis belong to the
//! driver (`bench/v3-routed-experts/verify_remote.py`).
//!
//! `sequential` is a separate loop from the speculative one, so that gate
//! 1 (every arm's ids equal `sequential`'s) does not compare a loop with
//! itself. It steps the same session type through
//! [`RoutedBlockSession::step`], which is `DecodeSession::step` over the
//! session's own continuation (the single-row route), so both arms share
//! one prefill method.
//!
//! `--corrupt first|middle|last` is a negative control: every proposal,
//! once cut before any stop id as the loop would cut it, has one token
//! changed at that offset to a non-stop id, and `corrupted_offsets` records
//! the offset per verify block (every block has a non-empty proposal; single
//! steps have none). The ids must still equal `sequential`'s and
//! acceptance must fall. Its timing is not a result.
//!
//! One worker only (`--shards` takes exactly one url), so every exchange
//! recorded is with the configured worker.
//!
//! `--relay-mark <control address>` marks the relay's generation boundary
//! (VERIFY-REMOTE-1 revision 2): after prefill has committed and before the
//! clock starts, the trial sends one `mark` and waits for the answer, which
//! it records with the number of exchanges completed. No generation exchange
//! is sent before the answer or the end of the wait. `--relay-mark none`
//! runs without a relay.
//!
//! Protocol: `bench/v3-routed-experts/VERIFY-REMOTE-1.md`.
//!
//! ```text
//! v3_verify_remote --model <container> --shards <url> \
//!     --workload <workload.json> --arm sequential|verify-2|verify-4|verify-8 \
//!     --generate <n> --stop <id[,id]> --corrupt none|first|middle|last \
//!     --profile yes|no --relay-mark <host:port>|none --out <trial.json>
//! ```
mod support;

use larql_inference::speculation::propose;
use larql_inference::vindex3::routed_block::RoutedBlockSession;
use larql_inference::vindex3::routed_experts;
use larql_inference::vindex3::speculate::{greedy_speculate, prefill, BlockTarget};
use larql_inference::vindex3::{OpenPolicy, Vindex3Runtime};
use larql_router::vindex3_experts::HttpExpertShards;
use larql_vindex::format::vindex3::opplan::exec::{
    continuation::plan_continuation_geometry, continuation_authority::ContinuationConfig,
    kv::RowKvState, operands::RepresentationSource, production::ProductionBackend,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use support::timing::{Exchange, Recorder, Timed, MULTI_ROW, SINGLE_ROW};
use support::verify::{ledger_json, sequential, Corrupt, Tagged, TrialProposer, Unit};

type BoxErr = Box<dyn std::error::Error>;

/// The component a text container encodes its stack under.
const COMPONENT: &str = "target";
const SEQUENTIAL: &str = "sequential";
const VERIFY_PREFIX: &str = "verify-";
/// Speculative widths the protocol admits.
const WIDTHS: [usize; 3] = [2, 4, 8];
/// Prompt prefill width, identical in every arm and untimed.
const PREFILL_WIDTH: usize = 8;

struct Args {
    model: PathBuf,
    shards: Vec<String>,
    workload: PathBuf,
    arm: String,
    /// `None` for `sequential`.
    width: Option<usize>,
    generate: usize,
    stop: Vec<u32>,
    corrupt: Corrupt,
    profile: bool,
    /// The relay's control address; `None` runs without a relay.
    relay_mark: Option<String>,
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
    let width = match arm.strip_prefix(VERIFY_PREFIX) {
        Some(width) => Some(width.parse::<usize>()?),
        None if arm == SEQUENTIAL => None,
        None => return Err(format!("unknown arm {arm}").into()),
    };
    if width.is_some_and(|w| !WIDTHS.contains(&w)) {
        return Err(format!("verify width must be one of {WIDTHS:?}").into());
    }
    let corrupt = Corrupt::parse(&take("--corrupt")?)?;
    if width.is_none() && !matches!(corrupt, Corrupt::None) {
        return Err("--corrupt applies only to a verify arm".into());
    }
    let profile = match take("--profile")?.as_str() {
        "yes" => true,
        "no" => false,
        other => return Err(format!("--profile takes yes or no, not {other}").into()),
    };
    let stop = take("--stop")?
        .split(',')
        .map(|id| id.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    let parsed = Args {
        model: take("--model")?.into(),
        shards: take("--shards")?.split(',').map(str::to_string).collect(),
        workload: take("--workload")?.into(),
        relay_mark: Some(take("--relay-mark")?).filter(|m| m != support::mark::NONE),
        generate: take("--generate")?.parse()?,
        out: take("--out")?.into(),
        arm,
        width,
        stop,
        corrupt,
        profile,
    };
    match values.keys().next() {
        Some(unknown) => Err(format!("unknown flag {unknown}").into()),
        None => Ok(parsed),
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() -> Result<(), BoxErr> {
    let args = parse_args()?;
    let workload_bytes = std::fs::read(&args.workload)?;
    let workload: serde_json::Value = serde_json::from_slice(&workload_bytes)?;
    let prompt: Vec<u32> = workload["prompt"]
        .as_array()
        .ok_or("the workload has no `prompt` list")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or("a prompt id is not a token id")
        })
        .collect::<Result<_, _>>()?;
    let stop: HashSet<u32> = args.stop.iter().copied().collect();
    // Gate 3 names the one configured worker; a record of several could not show which was contacted.
    if args.shards.len() != 1 {
        return Err("VERIFY-REMOTE-1 runs one worker: give exactly one --shards url".into());
    }

    let recorder = Recorder::install(args.profile)?;
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
    let mut target = Tagged {
        inner: RoutedBlockSession::new(runtime.plan(), &ops, runtime.backend(), &continuation)?,
        exchanges: exchanges.clone(),
        recorder,
        units: Vec::new(),
    };

    let first = prefill(&mut target, &prompt, PREFILL_WIDTH)?;
    let generation_unit = target.units.len();
    let (generation_exchange, generation_record) = target.counts();

    // The generation boundary: prefill has committed and its every exchange
    // has returned (the transport is synchronous). Nothing is sent until the
    // relay answers or the wait ends.
    let relay_mark = args.relay_mark.as_deref().map(|control| {
        support::mark::mark(control, support::mark::ANSWER_TIMEOUT, generation_exchange)
    });

    // The clock covers generation only: proposing, every verify block and
    // single step, commits, argmax and the loop itself. The first id is
    // prefill's and was produced untimed.
    let started = Instant::now();
    let (ids, ledger, changed) = match args.width {
        None => (
            sequential(&mut target, first, args.generate, &stop)?,
            None,
            None,
        ),
        Some(width) => {
            let mut proposer = TrialProposer {
                lookup: propose,
                corrupt: args.corrupt,
                stop: &stop,
                changed: Vec::new(),
            };
            let run = greedy_speculate(
                &mut target,
                &prompt,
                first,
                args.generate,
                width,
                &stop,
                &mut proposer,
            )?;
            (run.ids.clone(), Some(run), Some(proposer.changed))
        }
    };
    let wall_ns = started.elapsed().as_nanos() as u64;
    recorder.stop();

    // Requested, emitted, prefill-produced and timed ids, kept apart: the
    // first emitted id is prefill's and produced before the clock started.
    let prefill_produced = usize::from(!ids.is_empty());
    let ended_by = match ids.last() {
        None => "nothing requested",
        Some(last) if stop.contains(last) => "stop id",
        Some(_) => "generation limit",
    };
    let id_counts = serde_json::json!({
        "requested": args.generate,
        "emitted": ids.len(),
        "prefill_produced": prefill_produced,
        "timed": ids.len() - prefill_produced,
        "ended_by": ended_by,
    });

    let exchanges = exchanges.lock().unwrap();
    let count = |from: usize, route: &str| {
        exchanges[from..]
            .iter()
            .filter(|e| e.route == route)
            .count()
    };
    let records = recorder.records.lock().unwrap();
    let trial = serde_json::json!({
        "schema": "larql.v3.verify-remote-trial.v2",
        "arm": args.arm,
        "width": args.width,
        "corrupt": args.corrupt.name(),
        "profiled": args.profile,
        "open_policy": { "want": null, "source": "stored" },
        "model": args.model.display().to_string(),
        "shards": args.shards,
        "workload": {
            "name": workload["name"],
            "path": args.workload.display().to_string(),
            "sha256": hex_sha256(&workload_bytes),
        },
        "prompt_tokens": prompt.len(),
        "prefill_width": PREFILL_WIDTH,
        "generate": args.generate,
        "stop": args.stop,
        "timed_ids": ids.len() - prefill_produced,
        "id_counts": id_counts,
        "ids": ids,
        "first_id_source": "prefill, untimed; the clock covers the other `timed_ids`",
        "clock": "monotonic wall time from after prefill to the last generated id; load, \
                  prefill and writing excluded",
        "wall_ns": wall_ns,
        "final_position": target.position(),
        "relay_mark": relay_mark.as_ref().map(support::mark::Outcome::json),
        "generation_from": {
            "unit": generation_unit,
            "exchange": generation_exchange,
            "worker_record": generation_record,
        },
        "units": target.units.iter().map(Unit::json).collect::<Vec<_>>(),
        "ledger": ledger.as_ref().map(ledger_json),
        // Per verify block (one per ledger block, in order; single steps have
        // none): the proposal offset the control changed.
        "corrupted_offsets": changed,
        "frames": {
            "whole_run": { SINGLE_ROW: count(0, SINGLE_ROW), MULTI_ROW: count(0, MULTI_ROW) },
            "generation": {
                SINGLE_ROW: count(generation_exchange, SINGLE_ROW),
                MULTI_ROW: count(generation_exchange, MULTI_ROW),
            },
        },
        "exchanges": exchanges.iter().map(Exchange::json).collect::<Vec<_>>(),
        "worker_records": &*records,
    });
    std::fs::write(&args.out, serde_json::to_vec(&trial)?)?;
    Ok(())
}
