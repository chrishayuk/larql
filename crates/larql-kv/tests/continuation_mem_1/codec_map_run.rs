//! CONTINUATION-CODEC-MAP-1 reconnaissance: the arms, resumable.
//!
//! Stage 0, once: R (logits kept), C (Q4_K, logits kept), CH (codec/v1
//! on the Q4_K store, logits kept, traced) and SIM-0 — MappedCodec with
//! every row compressed on the same store must reproduce CH's logits bit
//! for bit and its trace exactly, or the run stops. Then each requested
//! map (adaptive: the maps are named per invocation, a checkpoint per map,
//! and a name is never reused for another spec). The report is recomputed
//! from checkpoints on every invocation. Calibration text only: bank 1.

use std::path::Path;

use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;

use super::codec_1_c3::{bank, score, yardstick_guard, YARDSTICK_ENCODING};
use super::codec_2::full_precision_guard;
use super::codec_2_store::{logits_base, scores_name, Checkpoints};
use super::codec_map_report::{incremental, line, tail_persistence, worst, TAIL_WIDE};
use super::codec_map_sim::{CompressMap, MappedCodec, Recorder, Retained};
use super::measured::Inspect;
use super::*;

const BITS: u8 = 4;
const REPORT: &str = "report.json";

/// One rung: prefill rung − resume − decode, resume, decode scored.
pub struct Recon {
    pub rung: usize,
    pub resume: usize,
    pub decode: usize,
}

/// The reconnaissance ladder: bank 1's first 2,048 positions, the last
/// 1,024 scored.
pub const RECON: Recon = Recon {
    rung: 2_048,
    resume: 16,
    decode: 1_024,
};

impl Recon {
    pub(super) fn journey<'a>(&self, ids: &'a [u32]) -> (Journey, usize, &'a [u32]) {
        let start = self.rung - self.decode;
        let journey = Journey {
            prefill: ids[..start - self.resume].to_vec(),
            resume: ids[start - self.resume..start].to_vec(),
            decode: ids[start..self.rung].to_vec(),
        };
        (journey, start, &ids[start + 1..=self.rung])
    }
}

/// Decode logits of one provider's journey, and `read` of it after.
fn decode_with<P: Inspect, B: PlanBackend, R>(
    subject: (&Subject, &PreparedOperands),
    backend: &B,
    inner: P,
    journey: &Journey,
    read: impl Fn(&P) -> R,
) -> (Vec<Vec<f32>>, R) {
    let mut kv = Measured::new(inner);
    let out = subjects::run(subject.0, subject.1, backend, &mut kv, journey);
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect();
    (rows, read(&kv.inner))
}

fn traced<P: Inspect + Retained, B: PlanBackend>(
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    inner: P,
    journey: &Journey,
) -> (Vec<Vec<f32>>, (u64, u64)) {
    decode_with(yard, backend, Recorder::new(inner), journey, |r| r.trace())
}

fn bits(rows: &[Vec<f32>]) -> Vec<u32> {
    rows.iter().flatten().map(|x| x.to_bits()).collect()
}

/// Run stage 0 and every map the store lacks, then write the report.
pub fn run_recon<B: PlanBackend>(
    exact: (&Subject, &PreparedOperands),
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    ids: &[u32],
    recon: &Recon,
    maps: &[(CompressMap, String)],
    store: &Checkpoints,
) -> Value {
    let (journey, start, next) = recon.journey(ids);
    let n = recon.rung;
    let timed = |arm: &str, t: std::time::Instant| {
        eprintln!("{arm} done in {:.0}s", t.elapsed().as_secs_f64());
        t.elapsed().as_secs_f64()
    };
    let r = store.read_logits(&logits_base(n, "r")).unwrap_or_else(|| {
        let t = std::time::Instant::now();
        let (r, _) = decode_with(exact, backend, RowKvState::default(), &journey, |_| ());
        store.write_logits(&logits_base(n, "r"), &r);
        timed("R", t);
        r
    });
    if store.read::<Value>(&scores_name(n, "c")).is_none() {
        let t = std::time::Instant::now();
        let (c, _) = decode_with(yard, backend, RowKvState::default(), &journey, |_| ());
        store.write_logits(&logits_base(n, "c"), &c);
        store.write(
            &scores_name(n, "c"),
            &json!({"seconds": timed("C", t),
            "positions": score(&r, &c, next, n, start)}),
        );
    }
    if store.read::<Value>(&scores_name(n, "sim0")).is_none() {
        let t = std::time::Instant::now();
        let (ch, ch_trace) = traced(yard, backend, CodecKvState::new(BITS), &journey);
        store.write(&scores_name(n, "ch"), &json!({"seconds": timed("CH", t),
            "trace": [ch_trace.0.to_string(), ch_trace.1], "positions": score(&r, &ch, next, n, start)}));
        let t = std::time::Instant::now();
        let (sim, sim_trace) = traced(
            yard,
            backend,
            MappedCodec::new(BITS, CompressMap::all()),
            &journey,
        );
        let (logits, trace) = (bits(&sim) == bits(&ch), sim_trace == ch_trace);
        store.write(
            &scores_name(n, "sim0"),
            &json!({"passed": logits && trace,
            "logits_bit_identical": logits, "trace_identical": trace,
            "trace": [sim_trace.0.to_string(), sim_trace.1], "seconds": timed("SIM-0", t)}),
        );
    }
    let sim0: Value = store.read(&scores_name(n, "sim0")).unwrap();
    assert_eq!(sim0["passed"], true, "SIM-0 failed — stop: {sim0}");

    for (map, spec) in maps {
        let name = format!("map-{}", map.name);
        if let Some(held) = store.read::<Value>(&scores_name(n, &name)) {
            assert_eq!(
                held["spec"],
                spec.as_str(),
                "map `{}` was checkpointed with another spec",
                map.name
            );
            continue;
        }
        let t = std::time::Instant::now();
        let (rows, exact_fraction) = decode_with(
            yard,
            backend,
            MappedCodec::new(BITS, map.clone()),
            &journey,
            MappedCodec::exact_fraction,
        );
        store.write(
            &scores_name(n, &name),
            &json!({"spec": spec, "exact_fraction": exact_fraction,
            "seconds": timed(&name, t), "positions": score(&r, &rows, next, n, start)}),
        );
    }
    let report = report(store, n);
    store.write(REPORT, &report);
    report
}

fn positions(store: &Checkpoints, n: usize, arm: &str) -> Vec<PositionMetrics> {
    let v: Value = store
        .read(&scores_name(n, arm))
        .unwrap_or_else(|| panic!("{arm} missing"));
    serde_json::from_value(v["positions"].clone()).unwrap()
}

/// Every map checkpointed so far, against C and full CH.
fn report(store: &Checkpoints, n: usize) -> Value {
    let (c, ch) = (positions(store, n, "c"), positions(store, n, "ch"));
    let d_ch = incremental(&ch, &c);
    let mut lines = vec![line("ch_full", "all:kv:all", 0.0, &ch, &c, &d_ch)];
    let mut ds = Vec::new();
    for file in store.files() {
        let Some(arm) = file
            .strip_prefix(&format!("n{n}-"))
            .and_then(|f| f.strip_suffix(".json"))
        else {
            continue;
        };
        let Some(name) = arm.strip_prefix("map-") else {
            continue;
        };
        let v: Value = store.read(&file).unwrap();
        let p: Vec<PositionMetrics> = serde_json::from_value(v["positions"].clone()).unwrap();
        ds.push(incremental(&p, &c));
        lines.push(line(
            name,
            v["spec"].as_str().unwrap_or(""),
            v["exact_fraction"].as_f64().unwrap_or(0.0),
            &p,
            &c,
            &d_ch,
        ));
    }
    let persistence: Vec<Value> = tail_persistence(&d_ch, &ds)
        .into_iter()
        .map(|(i, count)| json!({"position": ch[i].position, "maps_sharing": count, "of": ds.len(),
            "d_ch": d_ch[i], "reference_margin": ch[i].reference_margin, "reference_entropy": ch[i].reference_entropy}))
        .collect();
    json!({"programme": "CONTINUATION-CODEC-MAP-1 reconnaissance", "rung": n,
        "tail_wide_positions": worst(&d_ch, TAIL_WIDE).len(), "maps": lines, "ch_tail_persistence": persistence,
        "sim0": store.read::<Value>(&scores_name(n, "sim0"))})
}

/// The reconnaissance on gemma3-4b-it, bank 1: containers from
/// LARQL_CMAP_F32 / LARQL_CMAP_Q4K, store in LARQL_CMAP_RUN_DIR, maps in
/// LARQL_CMAP_MAPS (`;`-separated `name=<layers>:<tensors>:<age>`).
#[test]
#[ignore = "real containers: LARQL_CMAP_F32 + LARQL_CMAP_Q4K + LARQL_CMAP_RUN_DIR + LARQL_CMAP_MAPS (CODEC-MAP-1 reconnaissance)"]
fn real_codec_map_1_reconnaissance() {
    let _serial = serial();
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let ids = bank().expect("bank 1 (calibration)");
    let exact = subjects::open(Path::new(&env("LARQL_CMAP_F32")), "gemma3-4b-it");
    let yard = subjects::open_bound(
        Path::new(&env("LARQL_CMAP_Q4K")),
        "gemma3-4b-it.q4k",
        Some(YARDSTICK_ENCODING),
    );
    full_precision_guard(exact.store.selection()).unwrap_or_else(|s| panic!("{s}"));
    yardstick_guard(yard.store.selection()).unwrap_or_else(|s| panic!("{s}"));
    let specs: Vec<String> = env("LARQL_CMAP_MAPS")
        .split(';')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let maps: Vec<(CompressMap, String)> = specs
        .iter()
        .map(|s| {
            (
                CompressMap::parse(s).unwrap_or_else(|e| panic!("{e}")),
                s.clone(),
            )
        })
        .collect();
    let stamp =
        json!({"implementation_git_sha": git_sha(), "bank": "codec-1 bank 1 (calibration)"});
    let store = Checkpoints::open(Path::new(&env("LARQL_CMAP_RUN_DIR")), &stamp);
    let backend = ProductionBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let report = run_recon(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids,
        &RECON,
        &maps,
        &store,
    );
    eprintln!(
        "report: {}",
        serde_json::to_string_pretty(&report["maps"]).unwrap()
    );
}
