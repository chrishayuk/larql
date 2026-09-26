//! **CONTINUATION-MEM-1: what continuation bytes exist, which are read,
//! when, and what does the current API force into existence that nothing
//! needs?**
//!
//! The forecast is `docs/represent/forecasts/continuation-mem-1.json`
//! (frozen); the instrument deviations recorded before any measurement
//! are in `continuation-mem-1-notes.json`. Measurement only: nothing in
//! production changes. Controls assert; forecasts are computed and
//! written as records (a falsified forecast is a result, not a failure).
//!
//! Every test takes [`SERIAL`]: the allocator's scope state is process
//! global, and a scope's thread-integrity check would otherwise see a
//! neighbouring test's allocations as foreign.

mod alloc;
mod counting;
mod measured;
mod metrics;
mod retaining;
mod subjects;

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, hybrid_lllf_f32_model, miniature_glimmer, G_TOKENS, G_WINDOW,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;

use measured::{Inspect, Measured};
use subjects::{Journey, Subject};

#[global_allocator]
static ALLOCATOR: alloc::MeasuringAllocator = alloc::MeasuringAllocator;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("LARQL_MEM1_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("continuation-mem-1"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

// ---- controls -------------------------------------------------------------

/// Negative: with no scope open, allocations are never attributed.
/// Recorder silence: an empty scope attributes exactly nothing.
#[test]
fn control_negative_and_recorder_silence() {
    let _serial = serial();
    let before = alloc::attributed_totals();
    let planted: Vec<Vec<u8>> = (0..64).map(|i| vec![0u8; 16 + i]).collect();
    drop(planted);
    assert_eq!(
        alloc::attributed_totals(),
        before,
        "no scope open, nothing attributed"
    );

    let empty = alloc::enter().leave();
    assert_eq!(empty.allocs, 0, "an empty scope attributes no allocation");
    assert_eq!(empty.frees, 0);
    assert_eq!(empty.events, 0);
}

/// Positive: a scope does see an allocation made inside it, with its size.
#[test]
fn control_positive_attribution() {
    let _serial = serial();
    let scope = alloc::enter();
    let v: Vec<f32> = Vec::with_capacity(33);
    let delta = scope.leave();
    assert_eq!(delta.allocs, 1);
    assert_eq!(delta.alloc_bytes, 33 * 4);
    assert_eq!(alloc::live_size(v.as_ptr() as usize), Some(33 * 4));
    let ptr = v.as_ptr() as usize;
    drop(v);
    assert_eq!(
        alloc::live_size(ptr),
        None,
        "a free forgets the allocation wherever it happens"
    );
}

/// Thread integrity: an allocation on another thread while a scope is
/// open is counted foreign (the run would be INVALID).
#[test]
fn control_thread_integrity() {
    let _serial = serial();
    let handle = std::thread::spawn(|| {
        std::thread::park();
        let planted = vec![7u8; 4096];
        std::hint::black_box(&planted);
    });
    let scope = alloc::enter();
    handle.thread().unpark();
    // `join` itself does not allocate; the planted Vec is on the worker.
    let joined = handle.join();
    let delta = scope.leave();
    joined.unwrap();
    assert!(
        delta.foreign >= 1,
        "the planted foreign allocation must be seen: {delta:?}"
    );

    // The owner's own allocation is attributed to the owner — counted,
    // sized and logged. (Whether some other live thread also allocates
    // during the scope is not this control's claim: it is exactly what the
    // foreign counter exists to report, and a process has other threads.)
    let clean = alloc::enter();
    let local = vec![1u8; 64];
    let delta = clean.leave();
    let events = alloc::events(&delta);
    drop(local);
    assert_eq!(
        delta.allocs, 1,
        "the owner's allocation is counted once: {delta:?}"
    );
    assert_eq!(delta.alloc_bytes, 64);
    assert!(
        events
            .iter()
            .any(|e| e.kind == alloc::EventKind::Alloc && e.new_size == 64),
        "the owner's allocation is in the owner's event log"
    );
}

/// Tags survive address reuse: an allocation born in a tagged scope, freed,
/// and its address reused by an untagged scope's allocation, is no longer
/// reported as the tagged one (VIEW-1 V3: the append-born anti-cheat).
#[test]
fn control_tag_survives_address_reuse() {
    let _serial = serial();
    let tagged = alloc::enter_tagged(measured::APPEND_TAG);
    let first = vec![1u8; 96];
    let _ = tagged.leave();
    let address = first.as_ptr() as usize;
    assert_eq!(
        alloc::live_size_tagged(address, measured::APPEND_TAG),
        Some(96)
    );
    drop(first);
    assert_eq!(alloc::live_size_tagged(address, measured::APPEND_TAG), None);
    let untagged = alloc::enter();
    let second = vec![2u8; 96];
    let _ = untagged.leave();
    if second.as_ptr() as usize == address {
        assert_eq!(alloc::live_size(address), Some(96), "the reuse is live");
        assert_eq!(
            alloc::live_size_tagged(address, measured::APPEND_TAG),
            None,
            "a reused address must not answer as the append-born allocation"
        );
    }
    drop(second);
}

/// Realloc classification: every growth is classified moved or in place
/// by pointer comparison, and moved bytes are the old sizes of the moves.
#[test]
fn control_realloc_classification() {
    let _serial = serial();
    let mut v: Vec<u8> = Vec::with_capacity(1);
    v.push(0);
    let scope = alloc::enter();
    for cap in (1..=22).map(|s| 1usize << s) {
        v.reserve_exact(cap - v.len());
        v.resize(cap, 0);
    }
    let delta = scope.leave();
    let events = alloc::events(&delta);
    let mut moved_bytes = 0u64;
    let (mut moved, mut in_place) = (0, 0);
    for e in &events {
        match e.kind {
            alloc::EventKind::ReallocMoved => {
                assert_ne!(e.old_ptr, e.new_ptr);
                moved += 1;
                moved_bytes += e.old_size as u64;
            }
            alloc::EventKind::ReallocInPlace => {
                assert_eq!(e.old_ptr, e.new_ptr);
                in_place += 1;
            }
            _ => {}
        }
    }
    assert_eq!(moved + in_place, 22, "every growth is a realloc");
    assert_eq!(delta.moved_bytes, moved_bytes);
    assert!(moved > 0, "a small-to-large growth must move at least once");
    eprintln!("realloc control: {moved} moved, {in_place} in place");
}

// ---- journeys -------------------------------------------------------------

struct Run {
    /// Invalid solely because another thread allocated inside a provider
    /// scope: the run is discarded and repeated, never adjudicated.
    foreign_only: bool,
    record: Value,
    logits: Vec<(&'static str, Vec<f32>)>,
    storage: Vec<(&'static str, metrics::Storage)>,
    instrument_ok: bool,
}

fn measure<P: Inspect, B: PlanBackend>(
    subject: &Subject,
    backend: &B,
    provider: P,
    label: &str,
    journey: &Journey,
    watch_recurrent: bool,
) -> Run {
    let ops = subject.prepare(backend);
    let mut kv = Measured::new(provider).with_conv_qkv(subject.conv_qkv_layers());
    if watch_recurrent {
        kv = kv.watching_recurrent(subject.conv_history_bytes());
    }
    let outcome = subjects::run(subject, &ops, backend, &mut kv, journey);
    let (calls, intervals) = kv.take_records();

    let foreign_calls: u64 = calls.iter().map(|c| c.delta.foreign).sum();
    let foreign_windows: u64 = intervals.iter().map(|i| i.delta.foreign).sum();
    let mut phases = Vec::new();
    let mut storage = Vec::new();
    for (phase, inventory) in &outcome.inventories {
        let s = metrics::Storage::of(inventory);
        storage.push((*phase, s));
        let born = outcome
            .append_born
            .iter()
            .find(|(p, _)| p == phase)
            .map(|(_, b)| *b);
        phases.push(json!({
            "phase": phase,
            "logits_digest": subjects::logits_digest(&outcome.logits, phase),
            "append_born_live_bytes": born,
            "storage": s.json(),
            "append_traffic": metrics::appends(&calls, phase),
            "rows_offered": metrics::rows_offered(&calls, &subject.geometry, phase),
            "conv_qkv_windows": metrics::conv_qkv_windows(&intervals, phase),
            "recurrent_windows": metrics::recurrent_windows(&intervals, phase),
        }));
    }
    let unresolved: usize = storage.iter().map(|(_, s)| s.unresolved).sum();
    let unclassified: u64 = calls
        .iter()
        .filter_map(|c| c.append)
        .map(|t| t.unclassified)
        .sum();
    let dropped = metrics::events_dropped(&calls, &intervals);
    let instrument_ok = foreign_calls == 0
        && unresolved == 0
        && unclassified == 0
        && !dropped
        && !alloc::table_overflowed();
    let foreign_only = foreign_calls > 0
        && unresolved == 0
        && unclassified == 0
        && !dropped
        && !alloc::table_overflowed();
    Run {
        foreign_only,
        record: json!({
            "provider": label,
            "instrument": {
                "ok": instrument_ok,
                "foreign_allocs_in_provider_scopes": foreign_calls,
                "foreign_allocs_in_windows": foreign_windows,
                "unresolved_capacities": unresolved,
                "unclassified_append_events": unclassified,
                "events_dropped": dropped,
                "live_table_overflowed": alloc::table_overflowed(),
            },
            "phases": phases,
        }),
        logits: outcome.logits,
        storage,
        instrument_ok,
    }
}

/// Attempts before a run whose only defect is foreign allocations is
/// reported as invalid rather than repeated.
const MAX_ATTEMPTS: usize = 5;

/// [`measure`], repeated while the run is invalid ONLY because another
/// thread allocated inside a provider scope. The integrity rule is kept —
/// such a run is discarded, never read — and the record says how many were.
fn measure_valid<P: Inspect, B: PlanBackend>(
    subject: &Subject,
    backend: &B,
    provider: fn() -> P,
    label: &str,
    journey: &Journey,
    watch_recurrent: bool,
) -> Run {
    let mut discarded = 0;
    loop {
        let mut run = measure(
            subject,
            backend,
            provider(),
            label,
            journey,
            watch_recurrent,
        );
        if !run.foreign_only || discarded + 1 == MAX_ATTEMPTS {
            run.record["instrument"]["discarded_foreign_runs"] = json!(discarded);
            return run;
        }
        discarded += 1;
    }
}

/// Both shipped providers over one journey; parity (the C3 gate) and the
/// cross-provider forecasts (M1, M7) computed here.
fn measure_subject<B: PlanBackend>(
    subject: &Subject,
    backend: &B,
    journey: &Journey,
    watch_recurrent: bool,
    extra: Value,
) -> Value {
    let row = measure_valid(
        subject,
        backend,
        RowKvState::default,
        "row/v1",
        journey,
        watch_recurrent,
    );
    let canonical = measure_valid(
        subject,
        backend,
        CanonicalKvState::new,
        "canonical/v1",
        journey,
        watch_recurrent,
    );
    let parity = row.logits.len() == canonical.logits.len()
        && row
            .logits
            .iter()
            .zip(&canonical.logits)
            .all(|((_, a), (_, b))| {
                a.iter()
                    .map(|x| x.to_bits())
                    .eq(b.iter().map(|x| x.to_bits()))
            });

    let cross: Vec<Value> = row
        .storage
        .iter()
        .zip(&canonical.storage)
        .map(|((phase, r), (_, c))| {
            let m1 =
                (r.kv_allocated() > 0).then(|| c.kv_allocated() as f64 / r.kv_allocated() as f64);
            let m7_canonical = (c.kv_allocated() > 0)
                .then(|| c.row_storage as f64 / (c.row_storage + c.matrix) as f64);
            let m7_row = (r.kv_allocated() > 0).then_some(0.0);
            json!({
                "phase": phase,
                "payload_identical": r.kv_payload == c.kv_payload,
                "M1_allocated_ratio_canonical_over_row": m1,
                "M7_contract_duplicate_fraction_canonical": m7_canonical,
                "M7_contract_duplicate_fraction_row": m7_row,
            })
        })
        .collect();

    let record = json!({
        "subject": subject.name,
        "git_sha": git_sha(),
        "backend": backend.name(),
        "journey": {
            "prefill": journey.prefill.len(),
            "resume": journey.resume.len(),
            "decode": journey.decode.len(),
        },
        "geometry": metrics::geometry_json(&subject.geometry),
        "parity_row_vs_canonical_bit_identical": parity,
        "cross_provider": cross,
        "runs": [row.record, canonical.record],
        "extra": extra,
    });
    let path = out_dir().join(format!("{}.json", subject.name));
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
    assert!(
        parity,
        "C3 parity: row/v1 and canonical/v1 must be bit-identical on {}",
        subject.name
    );
    assert!(
        row.instrument_ok && canonical.instrument_ok,
        "instrument invalid on {}: see {}",
        subject.name,
        path.display()
    );
    record
}

// ---- I2: output dependence across a sliding window ------------------------

/// Perturb one held row outside the layer's selected range (bit-identical
/// logits required) and one inside it (a change required), on a state
/// that has crossed the window. Returns the witness record.
fn output_dependence<B: PlanBackend>(
    subject: &Subject,
    backend: &B,
    prefix: &[u32],
    next: u32,
) -> Value {
    let ops = subject.prepare(backend);
    let mut base = RowKvState::default();
    subjects_prefill(subject, &ops, backend, &mut base, prefix);
    let position = base.position_of();
    let mut witnesses = Vec::new();
    for (layer, g) in subject.geometry.iter().enumerate() {
        let Some(window) = g.kv_side().and_then(|k| k.window) else {
            continue;
        };
        // The step at `position` selects rows position+1-window ..= position
        // (the row being appended is the last); held rows below the start
        // are outside it.
        let start = (position + 1).saturating_sub(window);
        assert!(
            start > 0,
            "layer {layer}: the state must have crossed its window"
        );
        let step = |perturbed: Option<usize>| {
            let mut kv = Measured::new(base.clone());
            if let Some(row) = perturbed {
                kv.perturb(layer, row);
            }
            let mut session =
                DecodeSession::over_prepared(&subject.plan, &ops, backend, &mut kv).unwrap();
            session.step(next).unwrap().logits.unwrap()
        };
        let reference = step(None);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let outside_row = 0;
        let inside_row = position - 1;
        let outside_equal = bits(&step(Some(outside_row))) == bits(&reference);
        let inside_changed = bits(&step(Some(inside_row))) != bits(&reference);
        witnesses.push(json!({
            "layer": layer,
            "window": window,
            "position": position,
            "selected_range": [start, position],
            "outside_row": outside_row,
            "outside_bit_identical": outside_equal,
            "inside_row": inside_row,
            "inside_changed": inside_changed,
            "holds_both_directions": outside_equal && inside_changed,
        }));
    }
    Value::Array(witnesses)
}

trait PositionOf {
    fn position_of(&self) -> usize;
}

impl PositionOf for RowKvState {
    fn position_of(&self) -> usize {
        use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationProvider;
        self.position()
    }
}

fn subjects_prefill<B: PlanBackend>(
    subject: &Subject,
    ops: &larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands,
    backend: &B,
    kv: &mut RowKvState,
    tokens: &[u32],
) {
    larql_vindex::format::vindex3::opplan::exec::prefill_prepared(
        &subject.plan,
        ops,
        tokens,
        backend,
        kv,
    )
    .unwrap();
}

// ---- mechanism subjects (fixtures) ----------------------------------------

#[test]
fn mechanism_dense_softmax() {
    let _serial = serial();
    let subject = subjects::fixture(dense_f32_model, "mem1-dense");
    let journey = Journey {
        prefill: (1..9).collect(),
        resume: vec![20, 21, 22, 23],
        decode: vec![30, 31, 32, 33],
    };
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        false,
        Value::Null,
    );
}

#[test]
fn mechanism_sliding_window() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let backend = ReferenceBackend::new();
    let mut prefix = G_TOKENS.to_vec();
    prefix.extend([5, 9]);
    let witness = output_dependence(&subject, &backend, &prefix, 1);
    assert!(
        prefix.len() > G_WINDOW,
        "the witness state must cross the window"
    );
    measure_subject(
        &subject,
        &backend,
        &journey,
        false,
        json!({ "I2_output_dependence": witness }),
    );
}

#[test]
fn mechanism_kda_mla_hybrid() {
    let _serial = serial();
    let subject = subjects::fixture(hybrid_kda_mla_f32_model, "mem1-kda-mla");
    let journey = Journey {
        prefill: vec![1, 2, 3, 4, 5, 6],
        resume: vec![7, 8, 9],
        decode: vec![10, 11, 12],
    };
    let mla = mla_projection_counts(&subject, &journey);
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        true,
        json!({ "I5_mla": mla }),
    );
}

/// Gated DeltaNet (D6): the fourth recurrent operator, for M8's
/// per-call copy clause on the serial reference backend.
#[test]
fn mechanism_gated_delta_hybrid() {
    let _serial = serial();
    let subject = subjects::fixture(hybrid_lllf_f32_model, "mem1-gated-delta");
    let journey = Journey {
        prefill: vec![1, 2, 3, 4, 5, 6],
        resume: vec![7, 8, 9],
        decode: vec![10, 11, 12],
    };
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        true,
        Value::Null,
    );
}

/// I5 / M5: kv_b_proj calls per MLA call, counted at the projector.
fn mla_projection_counts(subject: &Subject, journey: &Journey) -> Value {
    use larql_vindex::format::vindex3::opplan::LayerAttention;
    let (layer, op) = subject
        .plan
        .layers
        .iter()
        .enumerate()
        .find_map(|(i, l)| match &l.attention {
            LayerAttention::Mla(op) => Some((i, op.clone())),
            _ => None,
        })
        .expect("the hybrid has an MLA layer");
    let g = op.geometry();
    let kv_b = (
        g.kv_lora_rank,
        g.num_heads * (g.qk_nope_head_dim + g.v_head_dim),
    );
    let backend = counting::Counting::new(ReferenceBackend::new());
    let ops = subject.prepare(&backend);
    let mut kv = RowKvState::default();

    let count = |b: &counting::Counting<ReferenceBackend>| b.projections.count(kv_b.0, kv_b.1);
    let mut phases = Vec::new();
    let before = count(&backend);
    subjects_prefill(subject, &ops, &backend, &mut kv, &journey.prefill);
    let n = journey.prefill.len();
    phases.push(json!({
        "phase": subjects::BATCHED,
        "positions": n,
        "kv_b_calls": count(&backend) - before,
        "forecast_n_n_plus_1_over_2": n * (n + 1) / 2,
    }));
    let mut steps = Vec::new();
    {
        let mut session =
            DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv).unwrap();
        for (i, &token) in journey.decode.iter().enumerate() {
            let h = n + i;
            let before = count(&backend);
            session.step(token).unwrap();
            steps.push(json!({ "h": h, "kv_b_calls": count(&backend) - before, "forecast_h_plus_1": h + 1 }));
        }
    }
    phases.push(json!({ "phase": subjects::DECODE, "steps": steps }));
    let shapes: Vec<Value> = backend
        .projections
        .shapes()
        .into_iter()
        .map(|((i, o), c)| json!({ "in": i, "out": o, "calls": c }))
        .collect();
    json!({
        "mla_layer": layer,
        "kv_b_shape": [kv_b.0, kv_b.1],
        "kv_a_out_width": g.compressed_kv_width(),
        "phases": phases,
        "all_projection_shapes": shapes,
    })
}

// ---- real containers (magnitude) ------------------------------------------

fn real(env: &str, name: &str, journey: Journey, watch_recurrent: bool, windowed: bool) {
    let _serial = serial();
    let Some(dir) = std::env::var_os(env) else {
        panic!("set {env} to the container directory");
    };
    let subject = subjects::open(std::path::Path::new(&dir), name);
    let backend = ProductionBackend::new();
    let mut extra = json!({ "container": dir.to_string_lossy() });
    if windowed {
        let mut prefix = journey.prefill.clone();
        prefix.extend(&journey.resume);
        extra["I2_output_dependence"] =
            output_dependence(&subject, &backend, &prefix, journey.decode[0]);
    }
    measure_subject(&subject, &backend, &journey, watch_recurrent, extra);
}

fn tokens(start: u32, n: usize) -> Vec<u32> {
    (start..start + n as u32).collect()
}

#[test]
#[ignore = "real container: LARQL_MEM1_QWEN"]
fn real_dense_qwen3_0_6b() {
    real(
        "LARQL_MEM1_QWEN",
        "qwen3-0.6b",
        Journey {
            prefill: tokens(1000, 128),
            resume: tokens(3000, 32),
            decode: tokens(4000, 16),
        },
        false,
        false,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_GEMMA (sequence crosses the sliding window)"]
fn real_sliding_gemma3_4b() {
    let n: usize = std::env::var("LARQL_MEM1_GEMMA_PREFILL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1100);
    real(
        "LARQL_MEM1_GEMMA",
        "gemma3-4b-it",
        Journey {
            prefill: tokens(1000, n),
            resume: tokens(3000, 16),
            decode: tokens(4000, 8),
        },
        false,
        true,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_OUTE"]
fn real_convqkv_mamba2_oute_250m() {
    real(
        "LARQL_MEM1_OUTE",
        "oute-mamba2attn-250m",
        Journey {
            prefill: tokens(1000, 64),
            resume: tokens(3000, 16),
            decode: tokens(4000, 16),
        },
        true,
        false,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_KIMI (short sequences only)"]
fn real_kda_mla_kimi_s7() {
    real(
        "LARQL_MEM1_KIMI",
        "kimi-linear-48b-s7",
        Journey {
            prefill: tokens(1000, 16),
            resume: tokens(3000, 4),
            decode: tokens(4000, 4),
        },
        true,
        false,
    );
}

/// D6: oute's recurrent-operator windows on the serial reference backend,
/// the integrity-clean source for M8's per-call copy clause.
#[test]
#[ignore = "real container: LARQL_MEM1_OUTE (serial reference backend)"]
fn real_recurrent_windows_oute_reference() {
    let _serial = serial();
    let dir = std::env::var_os("LARQL_MEM1_OUTE").expect("set LARQL_MEM1_OUTE");
    let subject = subjects::open(std::path::Path::new(&dir), "oute-mamba2attn-250m.reference");
    let journey = Journey {
        prefill: tokens(1000, 16),
        resume: tokens(3000, 4),
        decode: tokens(4000, 4),
    };
    let extra = json!({ "container": dir.to_string_lossy(), "purpose": "D6: integrity-clean recurrent windows" });
    measure_subject(&subject, &ReferenceBackend::new(), &journey, true, extra);
}

// ---- VIEW-1 V4: B (bounded retention) and S3 end to end --------------------

/// B: an exact-retention provider over one journey, beside row/v1. All four
/// conditions are asserted together: base == the plan's required start and
/// end == position + 1 after every append; every dropped row's own K and V
/// allocation freed, by pointer, inside the append that dropped it; logits
/// bit-identical to row/v1 on every phase.
fn retention_proof<B: PlanBackend>(subject: &Subject, backend: &B, journey: &Journey) -> Value {
    use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationProvider;
    let ops = subject.prepare(backend);
    let mut rows = Measured::new(RowKvState::default());
    let reference = subjects::run(subject, &ops, backend, &mut rows, journey);
    let mut kept = Measured::new(retaining::ExactRetention::default());
    let retained = subjects::run(subject, &ops, backend, &mut kept, journey);
    let (calls, _) = kept.take_records();

    let checks = &kept.inner.checks;
    let failures: Vec<_> = checks.iter().filter(|c| !c.holds()).collect();
    let dropped: u64 = checks.iter().map(|c| c.dropped as u64).sum();
    let traffic: Vec<_> = calls.iter().filter_map(|c| c.append).collect();
    let evicted: u64 = traffic.iter().map(|t| t.evicted_frees).sum();
    let unclassified: u64 = traffic.iter().map(|t| t.unclassified).sum();
    let bits = |o: &subjects::Outcome| -> Vec<Vec<u32>> {
        o.logits
            .iter()
            .map(|(_, l)| l.iter().map(|x| x.to_bits()).collect())
            .collect()
    };
    let identical = bits(&reference) == bits(&retained);
    let resident: Vec<Value> = subject
        .geometry
        .iter()
        .enumerate()
        .filter_map(|(layer, g)| {
            let kv = g.kv_side()?;
            Some(json!({
                "layer": layer,
                "window": kv.window,
                "resident_rows": kept.inner.resident(layer),
                "end": kept.inner.rows(layer).end(),
            }))
        })
        .collect();
    let record = json!({
        "subject": subject.name,
        "git_sha": git_sha(),
        "appends": checks.len(),
        "appends_off_the_plan_floor": failures.len(),
        "rows_dropped": dropped,
        "row_allocations_freed_by_pointer": evicted,
        "unclassified_append_events": unclassified,
        "logits_bit_identical_to_row_v1": identical,
        "resident": resident,
    });
    let path = out_dir().join(format!("{}.retention.json", subject.name));
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
    assert!(
        failures.is_empty(),
        "base/end off the plan: {:?}",
        &failures[..failures.len().min(3)]
    );
    assert!(
        dropped > 0,
        "the journey must cross a window, or B proves nothing"
    );
    assert_eq!(
        evicted,
        2 * dropped,
        "every dropped K and V row must be freed, by pointer"
    );
    assert_eq!(unclassified, 0, "an append event nobody accounts for");
    assert!(identical, "retention changed the logits");
    record
}

#[test]
fn b_exact_retention_on_the_sliding_fixture() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let record = retention_proof(&subject, &ReferenceBackend::new(), &journey);
    let sliding = &record["resident"][0];
    assert_eq!(sliding["window"], G_WINDOW);
    assert_eq!(
        sliding["resident_rows"], G_WINDOW,
        "a sliding layer holds its window"
    );
    assert_eq!(
        record["resident"][1]["resident_rows"], record["resident"][1]["end"],
        "a full layer holds everything"
    );
}

/// S3 end to end: a provider that drops a row the plan still requires is
/// refused by name at the caller of the step — and no attention kernel runs.
#[test]
fn s3_a_retention_bug_is_refused_before_any_kernel() {
    use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
    use std::sync::atomic::Ordering;
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let backend = counting::Counting::new(ReferenceBackend::new());
    let ops = subject.prepare(&backend);
    let mut kv = retaining::ExactRetention::default();
    prefill_prepared(&subject.plan, &ops, &G_TOKENS, &backend, &mut kv).unwrap();
    // A healthy step, then the seeded bug on the next step's layer-0 append.
    DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(1)
        .unwrap();
    kv.arm_violation();
    DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(2)
        .unwrap();
    assert!(
        kv.checks.iter().any(|c| !c.holds()),
        "the seed must have dropped a required row"
    );

    // Decode: the refusal reaches the caller; no kernel ran.
    let before = backend.attention_steps.load(Ordering::SeqCst);
    let err = DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(3)
        .err()
        .expect("a step lacking a required row must be refused");
    assert_eq!(
        backend.attention_steps.load(Ordering::SeqCst),
        before,
        "no attention kernel may run"
    );
    let message = err.to_string();
    assert!(
        message.contains("the plan requires"),
        "the refusal must be named: {message}"
    );

    // Resumed prefill takes the same door.
    let before = backend.attention_steps.load(Ordering::SeqCst);
    let err = prefill_prepared(&subject.plan, &ops, &[7, 8], &backend, &mut kv)
        .expect_err("a resumed prefill lacking a required row must be refused");
    assert_eq!(
        backend.attention_steps.load(Ordering::SeqCst),
        before,
        "no attention kernel may run"
    );
    assert!(err.to_string().contains("the plan requires"));
}

#[test]
#[ignore = "real container: LARQL_MEM1_GEMMA (B past the sliding window)"]
fn real_retention_gemma3_4b() {
    let _serial = serial();
    let dir = std::env::var_os("LARQL_MEM1_GEMMA").expect("set LARQL_MEM1_GEMMA");
    let subject = subjects::open(std::path::Path::new(&dir), "gemma3-4b-it");
    let journey = Journey {
        prefill: tokens(1000, 1100),
        resume: tokens(3000, 16),
        decode: tokens(4000, 8),
    };
    let record = retention_proof(&subject, &ProductionBackend::new(), &journey);
    for layer in record["resident"].as_array().unwrap() {
        if let Some(w) = layer["window"].as_u64() {
            assert_eq!(layer["resident_rows"].as_u64(), Some(w), "{layer}");
        }
    }
}
