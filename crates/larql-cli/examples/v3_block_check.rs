//! Real-model check of the routed block session against sequential decode.
//!
//! Correctness only. Connects to running expert workers, and for each
//! workload prompt:
//!
//! 1. decodes the prompt and a greedy continuation one position at a time
//!    through the remote experts (single-row frames), keeping every
//!    position's logits as the reference;
//! 2. replays that token stream through `RoutedBlockSession` in blocks of
//!    each width, and compares every position's logits bit for bit;
//! 3. replays it with wrong tokens appended to each block and only the true
//!    prefix committed, once with a single rejected suffix and once as a
//!    chain of blocks, and compares again.
//!
//! It also checks itself and the reference:
//!
//! 4. two negative controls: one bit flipped in one returned logit must be
//!    reported at exactly that position, and one expert output negated in
//!    one multi-row frame must be reported as a mismatch;
//! 5. with `--local yes`, the remote sequential reference against the same
//!    decode with resident experts;
//! 6. with `--control`, the greedy token ids against a saved control.
//!
//! The container is opened as `larql run` and the worker open it: stored
//! bytes only, nothing requantised at load.
//!
//! It counts the frames the transport carried in each phase. It measures no
//! time and makes no performance claim.
//!
//! ```text
//! cargo run --release -p larql-cli --example v3_block_check -- \
//!     --model <container> --shards http://127.0.0.1:9100 \
//!     --prompts bench/v3-routed-experts/prompts-v2.json --out receipt.json
//! ```
use larql_inference::vindex3::routed_block::RoutedBlockSession;
use larql_inference::vindex3::routed_experts::{self, ExpertOutput, ExpertTransport};
use larql_inference::vindex3::{OpenPolicy, Vindex3Runtime};
use larql_router::vindex3_experts::HttpExpertShards;
use larql_router_protocol::{vindex3_expert_rows as rows_wire, vindex3_experts::Binding};
use larql_vindex::format::vindex3::opplan::exec::{
    continuation::plan_continuation_geometry,
    continuation_authority::ContinuationConfig,
    continuation_registry::SelectedContinuation,
    decode::DecodeSession,
    kv::RowKvState,
    operands::RepresentationSource,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

type BoxErr = Box<dyn std::error::Error>;

/// The component a text container encodes its stack under.
const COMPONENT: &str = "target";
/// Block widths replayed.
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// Greedy tokens decoded after each prompt, so streams outlast one block.
const DEFAULT_NEW_TOKENS: usize = 16;
/// Chained partial commit: verify `KEEP + REJECT` positions, commit `KEEP`.
const CHAIN: [(usize, usize); 3] = [(3, 5), (2, 2), (4, 3)];
/// Single partial commit: wrong positions appended to each kept block.
const REJECTED_SUFFIX: usize = 4;
/// Kept positions per block in the single partial-commit replay.
const PARTIAL_KEEP: usize = 4;
/// Block width the negative controls run at.
const CONTROL_WIDTH: usize = 8;

#[derive(Default)]
struct Counts {
    single_frames: AtomicU64,
    row_frames: AtomicU64,
    rows: AtomicU64,
    pairs: AtomicU64,
    /// Negative control: negate one value of the next multi-row answer.
    corrupt_next: AtomicBool,
    corrupted: AtomicU64,
}

impl Counts {
    fn snapshot(&self) -> [u64; 4] {
        [
            self.single_frames.load(Ordering::SeqCst),
            self.row_frames.load(Ordering::SeqCst),
            self.rows.load(Ordering::SeqCst),
            self.pairs.load(Ordering::SeqCst),
        ]
    }
}

fn delta(after: [u64; 4], before: [u64; 4]) -> serde_json::Value {
    serde_json::json!({
        "single_row_frames": after[0] - before[0],
        "multi_row_frames": after[1] - before[1],
        "rows": after[2] - before[2],
        "pairs": after[3] - before[3],
    })
}

/// The HTTP transport, counting each frame it carries.
struct Counting {
    inner: HttpExpertShards,
    counts: Arc<Counts>,
}

impl ExpertTransport for Counting {
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
        self.counts.single_frames.fetch_add(1, Ordering::SeqCst);
        self.inner.forward(shard, layer, experts, row)
    }

    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[rows_wire::Row],
    ) -> Result<Vec<rows_wire::Output>, String> {
        self.counts.row_frames.fetch_add(1, Ordering::SeqCst);
        self.counts
            .rows
            .fetch_add(rows.len() as u64, Ordering::SeqCst);
        let pairs: usize = rows.iter().map(|r| r.experts.len()).sum();
        self.counts.pairs.fetch_add(pairs as u64, Ordering::SeqCst);
        let mut outputs = self.inner.forward_rows(shard, layer, rows)?;
        if self.counts.corrupt_next.swap(false, Ordering::SeqCst) {
            if let Some(value) = outputs.first_mut().and_then(|o| o.values.first_mut()) {
                *value = if *value == 0.0 { 1.0 } else { -*value };
                self.counts.corrupted.fetch_add(1, Ordering::SeqCst);
            }
        }
        Ok(outputs)
    }
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
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

/// A token that is not `token`: a rejected draft at that position.
fn wrong(token: u32) -> u32 {
    if token == 0 {
        1
    } else {
        token - 1
    }
}

struct Model<'a> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a ProductionBackend,
    continuation: &'a SelectedContinuation,
    counts: &'a Counts,
}

/// Positions whose logits differ from the reference, as `[position, …]`.
fn mismatches(reference: &[Vec<u32>], start: usize, rows: &[Vec<f32>]) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(i, row)| reference.get(start + i) != Some(&bits(row)))
        .map(|(i, _)| start + i)
        .collect()
}

impl Model<'_> {
    /// One position at a time: the prompt, then `new_tokens` greedy tokens.
    fn sequential(
        &self,
        ops: &PreparedOperands,
        prompt: &[u32],
        new_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>), BoxErr> {
        let mut state = self.continuation.build();
        let mut session = DecodeSession::over_prepared(self.plan, ops, self.backend, &mut *state)?;
        let mut stream = prompt.to_vec();
        let mut reference = Vec::new();
        let mut index = 0;
        while index < stream.len() {
            let logits = session
                .step(stream[index])?
                .logits
                .ok_or("a position produced no logits")?;
            if index + 1 == stream.len() && stream.len() < prompt.len() + new_tokens {
                stream.push(argmax(&logits));
            }
            reference.push(bits(&logits));
            index += 1;
        }
        Ok((stream, reference))
    }

    /// The stream in blocks of `width`, every block committed whole.
    fn whole_blocks(
        &self,
        stream: &[u32],
        reference: &[Vec<u32>],
        width: usize,
    ) -> Result<serde_json::Value, BoxErr> {
        let before = self.counts.snapshot();
        let mut session =
            RoutedBlockSession::new(self.plan, self.ops, self.backend, self.continuation)?;
        let mut bad = Vec::new();
        for (index, block) in stream.chunks(width).enumerate() {
            let rows = session.verify(block)?;
            bad.extend(mismatches(reference, index * width, &rows));
            session.commit(block.len())?;
        }
        Ok(serde_json::json!({
            "width": width,
            "blocks": stream.len().div_ceil(width),
            "positions": stream.len(),
            "mismatched_positions": bad,
            "frames": delta(self.counts.snapshot(), before),
        }))
    }

    /// The stream through blocks that each carry rejected positions after
    /// their true ones; `plan` yields `(keep, reject)` for each block.
    fn partial(
        &self,
        stream: &[u32],
        reference: &[Vec<u32>],
        plan: &mut dyn FnMut(usize) -> (usize, usize),
    ) -> Result<serde_json::Value, BoxErr> {
        let before = self.counts.snapshot();
        let mut session =
            RoutedBlockSession::new(self.plan, self.ops, self.backend, self.continuation)?;
        let mut bad = Vec::new();
        let (mut at, mut block_index, mut rejected) = (0, 0, 0);
        while at < stream.len() {
            let (keep, reject) = plan(block_index);
            let keep = keep.min(stream.len() - at);
            let mut block = stream[at..at + keep].to_vec();
            // Rejected drafts: wrong at the first rejected position onwards.
            block.extend((0..reject).map(|i| {
                let true_token = stream.get(at + keep + i).copied().unwrap_or(0);
                wrong(true_token)
            }));
            let rows = session.verify(&block)?;
            // Only the kept positions have a reference: a rejected row
            // consumed a token the reference stream never saw.
            bad.extend(mismatches(reference, at, &rows[..keep]));
            session.commit(keep)?;
            if session.position() != at + keep {
                return Err("the committed position did not follow the commit".into());
            }
            at += keep;
            rejected += reject;
            block_index += 1;
        }
        Ok(serde_json::json!({
            "blocks": block_index,
            "positions_kept": at,
            "positions_rejected": rejected,
            "mismatched_positions": bad,
            "frames": delta(self.counts.snapshot(), before),
        }))
    }
}

impl Model<'_> {
    /// Both negative controls on the stream's first block. Each must be
    /// reported; a control the checker misses fails the run.
    fn negative_controls(
        &self,
        stream: &[u32],
        reference: &[Vec<u32>],
    ) -> Result<serde_json::Value, BoxErr> {
        let block = &stream[..CONTROL_WIDTH.min(stream.len())];
        let target = block.len() - 1;

        // The comparator alone: one bit of one true logit.
        let mut session =
            RoutedBlockSession::new(self.plan, self.ops, self.backend, self.continuation)?;
        let mut rows = session.verify(block)?;
        let clean = mismatches(reference, 0, &rows);
        rows[target][0] = f32::from_bits(rows[target][0].to_bits() ^ 1);
        let flipped = mismatches(reference, 0, &rows);

        // The execution path: one expert output of one frame, negated.
        let before = self.counts.corrupted.load(Ordering::SeqCst);
        let mut session =
            RoutedBlockSession::new(self.plan, self.ops, self.backend, self.continuation)?;
        self.counts.corrupt_next.store(true, Ordering::SeqCst);
        let corrupted_rows = session.verify(block)?;
        let injected = self.counts.corrupted.load(Ordering::SeqCst) - before;
        let corrupted = mismatches(reference, 0, &corrupted_rows);

        let detected =
            clean.is_empty() && flipped == vec![target] && injected == 1 && !corrupted.is_empty();
        Ok(serde_json::json!({
            "block_positions": block.len(),
            "before_any_change": clean,
            "one_logit_bit_flipped_at": target,
            "one_logit_bit_reported_at": flipped,
            "expert_outputs_negated": injected,
            "negated_output_reported_at": corrupted,
            "detected": detected,
        }))
    }
}

struct Args {
    model: PathBuf,
    shards: Vec<String>,
    prompts: PathBuf,
    out: PathBuf,
    new_tokens: usize,
    /// Also decode with resident experts and compare with the remote reference.
    local: bool,
    /// Saved control token ids (`{"prompt": [...], "generated": [...]}`) and
    /// the workload prompt they belong to.
    control: Option<(PathBuf, String)>,
}

fn parse_args() -> Result<Args, BoxErr> {
    let mut values = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        values.insert(flag, value);
    }
    let control = match (
        values.remove("--control"),
        values.remove("--control-prompt"),
    ) {
        (Some(path), Some(prompt)) => Some((path.into(), prompt)),
        (None, None) => None,
        _ => return Err("--control and --control-prompt go together".into()),
    };
    let local = match values.remove("--local").as_deref() {
        None | Some("no") => false,
        Some("yes") => true,
        Some(other) => return Err(format!("--local takes yes or no, not {other}").into()),
    };
    let mut take = |flag: &str| values.remove(flag).ok_or(format!("{flag} is required"));
    let parsed = Args {
        local,
        control,
        model: take("--model")?.into(),
        shards: take("--shards")?.split(',').map(str::to_string).collect(),
        prompts: take("--prompts")?.into(),
        out: take("--out")?.into(),
        new_tokens: match values.remove("--new-tokens") {
            Some(n) => n.parse()?,
            None => DEFAULT_NEW_TOKENS,
        },
    };
    match values.keys().next() {
        Some(unknown) => Err(format!("unknown flag {unknown}").into()),
        None => Ok(parsed),
    }
}

fn clean(value: &serde_json::Value) -> bool {
    value["mismatched_positions"]
        .as_array()
        .is_some_and(Vec::is_empty)
}

fn main() -> Result<(), BoxErr> {
    let args = parse_args()?;
    let workload: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.prompts)?)?;
    // As `larql run` and the expert worker open it on the CPU production
    // arm: the stored bytes, nothing manufactured at load.
    let policy = OpenPolicy {
        want: None,
        source: RepresentationSource::Stored,
    };
    let runtime =
        Vindex3Runtime::open_with(&args.model, COMPONENT, ProductionBackend::new(), policy)?;
    let local_ops = match args.local {
        true => Some(PreparedOperands::load(
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            ExecutionSlice::Full,
        )?),
        false => None,
    };
    let control: Option<(String, serde_json::Value)> = match &args.control {
        Some((path, prompt)) => Some((
            prompt.clone(),
            serde_json::from_slice(&std::fs::read(path)?)?,
        )),
        None => None,
    };
    let counts = Arc::new(Counts::default());
    let transport = Counting {
        inner: HttpExpertShards::connect(&args.shards, None)?,
        counts: counts.clone(),
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
    let model = Model {
        plan: runtime.plan(),
        ops: &ops,
        backend: runtime.backend(),
        continuation: &continuation,
        counts: &counts,
    };
    let routed_layers = runtime
        .plan()
        .layers
        .iter()
        .filter(|l| l.ffn.as_ref().and_then(|f| f.routed()).is_some())
        .count();

    let mut prompts = serde_json::Map::new();
    let mut pass = true;
    for (name, prompt) in workload.as_object().ok_or("workload is not an object")? {
        let Some(ids) = prompt.get("ids").and_then(|v| v.as_array()) else {
            continue;
        };
        let ids: Vec<u32> = ids
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|id| id as u32)
                    .ok_or("a token id is not a number")
            })
            .collect::<Result<_, _>>()?;
        eprintln!("{name}: sequential reference");
        let before = counts.snapshot();
        let (stream, reference) = model.sequential(&ops, &ids, args.new_tokens)?;
        let sequential_frames = delta(counts.snapshot(), before);

        let local = match &local_ops {
            Some(local_ops) => {
                eprintln!("{name}: local reference");
                let (local_stream, local_reference) =
                    model.sequential(local_ops, &ids, args.new_tokens)?;
                let bad: Vec<usize> = (0..reference.len().max(local_reference.len()))
                    .filter(|i| reference.get(*i) != local_reference.get(*i))
                    .collect();
                pass &= bad.is_empty() && local_stream == stream;
                serde_json::json!({
                    "positions": local_stream.len(),
                    "stream_equal": local_stream == stream,
                    "mismatched_positions": bad,
                })
            }
            None => serde_json::Value::Null,
        };
        let saved = match &control {
            Some((prompt_name, saved)) if prompt_name == name => {
                let ids_of = |key: &str| -> Vec<u32> {
                    let values = saved[key].as_array().cloned().unwrap_or_default();
                    values
                        .iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as u32)
                        .collect()
                };
                let (saved_prompt, saved_generated) = (ids_of("prompt"), ids_of("generated"));
                let generated = &stream[ids.len()..];
                let compared = generated.len().min(saved_generated.len());
                let equal = saved_prompt == ids
                    && compared > 0
                    && generated[..compared] == saved_generated[..compared];
                pass &= equal;
                serde_json::json!({
                    "prompt_equal": saved_prompt == ids,
                    "generated_compared": compared,
                    "generated_equal": equal,
                })
            }
            _ => serde_json::Value::Null,
        };
        eprintln!("{name}: negative controls");
        let controls = model.negative_controls(&stream, &reference)?;
        pass &= controls["detected"] == true;

        let mut whole = Vec::new();
        for width in WIDTHS {
            eprintln!("{name}: blocks of {width}");
            whole.push(model.whole_blocks(&stream, &reference, width)?);
        }
        eprintln!("{name}: partial commit");
        let single = model.partial(&stream, &reference, &mut |_| {
            (PARTIAL_KEEP, REJECTED_SUFFIX)
        })?;
        eprintln!("{name}: chained partial commit");
        let chained = model.partial(&stream, &reference, &mut |i| CHAIN[i % CHAIN.len()])?;

        pass &= whole.iter().all(clean) && clean(&single) && clean(&chained);
        prompts.insert(
            name.clone(),
            serde_json::json!({
                "prompt_tokens": ids.len(),
                "stream": stream,
                "sequential": { "positions": stream.len(), "frames": sequential_frames },
                "local_reference": local,
                "saved_control": saved,
                "negative_controls": controls,
                "whole_blocks": whole,
                "partial_commit": single,
                "chained_partial_commit": chained,
            }),
        );
    }
    let receipt = serde_json::json!({
        "check": "v3-block-check",
        "purpose": "correctness only; no timing and no performance claim",
        "comparison": "every position's logits, bit for bit, against one-position-at-a-time \
                       decode through the same remote experts",
        "model": args.model.display().to_string(),
        "workload": workload.get("workload"),
        "shards": args.shards,
        "open_policy": { "want": null, "source": "stored" },
        "routed_layers": routed_layers,
        "new_tokens": args.new_tokens,
        "pass": pass,
        "prompts": prompts,
    });
    std::fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
    eprintln!("pass: {pass}; receipt at {}", args.out.display());
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
