//! CONTINUATION-CODEC-3 reconnaissance on gemma3-4b-it, bank 1
//! (calibration only): the real recent-K provider, resumable.
//!
//! Stage 0, once, in its own store: R (logits kept), C (Q4_K, logits
//! kept), CH (codec/v1, traced, residency) and W0 — codec-recent/v1 with a
//! window of 0 must reproduce CH's logits bit for bit and its trace
//! exactly, or the run stops. Then the CODEC-MAP-1 store is read as a
//! reference: this run's CH must score exactly as that store's CH.
//!
//! Then each window w, one checkpoint each: scores against R, the trace
//! (retention and reads must be CH's), and live-allocation residency
//! against what w declares. At w = 256 the scores must equal the
//! simulator's `p_recent256` map in the CODEC-MAP-1 store exactly (SIM-W),
//! or the run stops.
//!
//! The window is CHOSEN by the rule declared here before the run
//! ([`choose_window`]); the curve is reported whatever it picks. Decode
//! cost is out of scope until quality holds.

use std::path::Path;

use larql_kv::{CodecKvState, CodecRecentKvState};
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;

use super::codec_1_c3::{bank, score, yardstick_guard, YARDSTICK_ENCODING};
use super::codec_2::full_precision_guard;
use super::codec_2_store::{logits_base, scores_name, Checkpoints};
use super::codec_3::protection_spec;
use super::codec_map_report::{incremental, line, MapLine};
use super::codec_map_run::{Recon, RECON};
use super::codec_map_sim::{Recorder, Retained};
use super::*;

const BITS: u8 = 4;
const REPORT: &str = "report.json";
const STAGE0: &str = "stage0.json";
/// The calibration sweep, and the window CODEC-MAP-1's protection map used.
pub const SWEEP: [usize; 4] = [64, 128, 256, 512];
pub const PARITY_WINDOW: usize = 256;
/// The selection rule's saturation share (see [`choose_window`]).
pub const SATURATION: f64 = 0.9;

/// One traced arm: decode logits, the trace, and `read` of the measured
/// provider after the journey.
fn traced<P: Inspect + Retained, B: PlanBackend, R>(
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    inner: P,
    journey: &Journey,
    read: impl Fn(&Measured<Recorder<P>>) -> R,
) -> (Vec<Vec<f32>>, (u64, u64), R) {
    let mut kv = Measured::new(Recorder::new(inner));
    let out = subjects::run(yard.0, yard.1, backend, &mut kv, journey);
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect();
    (rows, kv.inner.trace(), read(&kv))
}

/// Decode logits of one untraced arm (R and C hold rows, not codes).
fn plain<P: Inspect, B: PlanBackend>(
    subject: (&Subject, &PreparedOperands),
    backend: &B,
    inner: P,
    journey: &Journey,
) -> Vec<Vec<f32>> {
    let mut kv = Measured::new(inner);
    subjects::run(subject.0, subject.1, backend, &mut kv, journey)
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect()
}

/// The CODEC-MAP-1 checkpoint of the simulated protection map at `w`.
pub fn map_parity_arm(window: usize) -> String {
    format!("map-p_recent{window}")
}

fn bits(rows: &[Vec<f32>]) -> Vec<u32> {
    rows.iter().flatten().map(|x| x.to_bits()).collect()
}

fn positions(v: &Value) -> Vec<PositionMetrics> {
    serde_json::from_value(v["positions"].clone()).expect("a scored checkpoint")
}

fn trace_json(t: (u64, u64)) -> Value {
    json!([t.0.to_string(), t.1])
}

fn timed(arm: &str, t: std::time::Instant) -> f64 {
    let s = t.elapsed().as_secs_f64();
    eprintln!("{arm} done in {s:.0}s");
    s
}

/// The pre-declared selection rule: the smallest swept window whose
/// removal of CH's worst-5% AND worst-1% incremental damage is each at
/// least [`SATURATION`] of the best removal any swept window reached.
/// `lines` are (window, line); removal is 1 − the share reproduced.
pub fn choose_window(lines: &[(usize, MapLine)]) -> Option<usize> {
    let removed = |l: &MapLine| (1.0 - l.explained_wide, 1.0 - l.explained_narrow);
    let best = lines.iter().fold((f64::MIN, f64::MIN), |b, (_, l)| {
        let r = removed(l);
        (b.0.max(r.0), b.1.max(r.1))
    });
    let mut sorted: Vec<&(usize, MapLine)> = lines.iter().collect();
    sorted.sort_by_key(|(w, _)| *w);
    sorted
        .into_iter()
        .find(|(_, l)| {
            let r = removed(l);
            r.0 >= SATURATION * best.0 && r.1 >= SATURATION * best.1
        })
        .map(|(w, _)| *w)
}

/// Run stage 0, the cross-store check, and every window the store lacks;
/// then write the report.
pub fn run_codec_3_recon<B: PlanBackend>(
    exact: (&Subject, &PreparedOperands),
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    ids: &[u32],
    recon: &Recon,
    (windows, parity_window): (&[usize], usize),
    (store, map_store): (&Checkpoints, &Checkpoints),
) -> Value {
    let (journey, start, next) = recon.journey(ids);
    let n = recon.rung;
    let r = store.read_logits(&logits_base(n, "r")).unwrap_or_else(|| {
        let t = std::time::Instant::now();
        let r = plain(exact, backend, RowKvState::default(), &journey);
        store.write_logits(&logits_base(n, "r"), &r);
        timed("R", t);
        r
    });
    if store.read::<Value>(&scores_name(n, "c")).is_none() {
        let t = std::time::Instant::now();
        let c = plain(yard, backend, RowKvState::default(), &journey);
        store.write_logits(&logits_base(n, "c"), &c);
        store.write(
            &scores_name(n, "c"),
            &json!({"seconds": timed("C", t), "positions": score(&r, &c, next, n, start)}),
        );
    }
    if store.read::<Value>(STAGE0).is_none() {
        let t = std::time::Instant::now();
        let (ch, ch_trace, residency) =
            traced(yard, backend, CodecKvState::new(BITS), &journey, |kv| {
                kv.codec_residency().expect("codec/v1 has codes")
            });
        store.write(
            &scores_name(n, "ch"),
            &json!({"seconds": timed("CH", t), "trace": trace_json(ch_trace),
                "residency": format!("{residency:?}"), "residency_holds": residency.holds(),
                "append_born_live": residency.append_born_live,
                "positions": score(&r, &ch, next, n, start)}),
        );
        let t = std::time::Instant::now();
        let (w0, w0_trace, _) = traced(
            yard,
            backend,
            CodecRecentKvState::new(BITS, 0),
            &journey,
            |_| (),
        );
        let (logits, trace) = (bits(&w0) == bits(&ch), w0_trace == ch_trace);
        store.write(
            STAGE0,
            &json!({"passed": logits && trace, "w0_logits_bit_identical_to_ch": logits,
                "w0_trace_identical_to_ch": trace, "ch_trace": trace_json(ch_trace),
                "seconds_w0": timed("W0", t)}),
        );
    }
    let stage0: Value = store.read(STAGE0).unwrap();
    assert_eq!(stage0["passed"], true, "W0 failed — stop: {stage0}");
    let ch: Value = store.read(&scores_name(n, "ch")).unwrap();
    let held_ch: Value = map_store
        .read(&scores_name(n, "ch"))
        .expect("the CODEC-MAP-1 store holds CH");
    assert_eq!(
        ch["positions"], held_ch["positions"],
        "this run's CH does not score as CODEC-MAP-1's CH — stop: the environment moved"
    );

    for &w in windows {
        let name = format!("w{w}");
        if store.read::<Value>(&scores_name(n, &name)).is_some() {
            continue;
        }
        let t = std::time::Instant::now();
        let (rows, trace, residency) = traced(
            yard,
            backend,
            CodecRecentKvState::new(BITS, w),
            &journey,
            |kv| {
                kv.mixed_residency(w)
                    .expect("codec-recent/v1 has a mixed layout")
            },
        );
        let seconds = timed(&name, t);
        let scored = json!(score(&r, &rows, next, n, start));
        let sim_w = (w == parity_window).then(|| {
            let sim: Value = map_store
                .read(&scores_name(n, &map_parity_arm(w)))
                .expect("the CODEC-MAP-1 store holds the protection map");
            // CODEC-MAP-1 checkpoints a map's spec as given on its command
            // line, `<name>=<clauses>`; the clauses are what must match.
            let held = sim["spec"].as_str().unwrap_or_default();
            let clauses = held.split_once('=').map_or(held, |(_, c)| c);
            assert_eq!(
                clauses,
                protection_spec(w),
                "the reference arm `{held}` is not the protection map at w {w}"
            );
            sim["positions"] == scored
        });
        store.write(
            &scores_name(n, &name),
            &json!({"window": w, "seconds": seconds,
                "trace": trace_json(trace), "trace_identical_to_ch": trace_json(trace) == stage0["ch_trace"],
                "sim_w_scores_identical": sim_w,
                "residency_holds": residency.holds(),
                "residency": {"append_born_live": residency.append_born_live, "expected": residency.expected,
                    "v_code_bytes": residency.v_code_bytes, "k_code_bytes": residency.k_code_bytes,
                    "k_exact_bytes": residency.k_exact_bytes, "list_bytes": residency.list_bytes,
                    "window_mismatches": residency.window_mismatches, "strays": residency.strays,
                    "scratch_bytes": residency.scratch_bytes, "scratch_bound": residency.scratch_bound},
                "positions": scored}),
        );
        assert_ne!(sim_w, Some(false), "SIM-W failed at w {w} — stop");
    }
    let report = report(store, n, &ch);
    store.write(REPORT, &report);
    report
}

/// Every window checkpointed so far, against C and full CH, with its
/// residency against CH's, and the pre-declared choice.
fn report(store: &Checkpoints, n: usize, ch: &Value) -> Value {
    let c = positions(&store.read::<Value>(&scores_name(n, "c")).unwrap());
    let ch_p = positions(ch);
    let d_ch = incremental(&ch_p, &c);
    let ch_bytes = ch["append_born_live"].as_u64().unwrap_or(0) as f64;
    let mut lines = Vec::new();
    let mut rows = vec![
        json!({"arm": "ch", "line": line("ch", "codec/v1", 0.0, &ch_p, &c, &d_ch),
        "resident_bytes": ch_bytes}),
    ];
    for file in store.files() {
        let Some(w) = file
            .strip_prefix(&format!("n{n}-w"))
            .and_then(|f| f.strip_suffix(".json"))
            .and_then(|w| w.parse::<usize>().ok())
        else {
            continue;
        };
        let v: Value = store.read(&file).unwrap();
        let l = line(
            &format!("w{w}"),
            &protection_spec(w),
            0.0,
            &positions(&v),
            &c,
            &d_ch,
        );
        let bytes = v["residency"]["append_born_live"].as_u64().unwrap_or(0) as f64;
        rows.push(json!({"arm": format!("w{w}"), "window": w, "line": l,
            "removed_wide": 1.0 - l.explained_wide, "removed_narrow": 1.0 - l.explained_narrow,
            "resident_bytes": bytes, "resident_vs_ch": bytes / ch_bytes.max(1.0),
            "k_exact_share_of_resident": v["residency"]["k_exact_bytes"].as_f64().unwrap_or(0.0) / bytes.max(1.0),
            "residency_holds": v["residency_holds"], "trace_identical_to_ch": v["trace_identical_to_ch"],
            "sim_w_scores_identical": v["sim_w_scores_identical"]}));
        lines.push((w, l));
    }
    lines.sort_by_key(|(w, _)| *w);
    json!({"programme": "CONTINUATION-CODEC-3 reconnaissance", "rung": n,
        "bank": "codec-1 bank 1 (calibration)", "selection_rule":
            format!("smallest swept w with removal of CH's worst-5% and worst-1% each >= {SATURATION} x the sweep's best"),
        "chosen_window": choose_window(&lines), "arms": rows,
        "stage0": store.read::<Value>(STAGE0)})
}

/// The reconnaissance on gemma3-4b-it, bank 1: containers from
/// LARQL_CR3_F32 / LARQL_CR3_Q4K, store in LARQL_CR3_RUN_DIR, the
/// CODEC-MAP-1 protection store (read only) in LARQL_CR3_MAP_STORE,
/// windows in LARQL_CR3_WINDOWS (comma-separated; default the sweep).
#[test]
#[ignore = "real containers: LARQL_CR3_F32 + LARQL_CR3_Q4K + LARQL_CR3_RUN_DIR + LARQL_CR3_MAP_STORE (CODEC-3 reconnaissance)"]
fn real_codec_3_reconnaissance() {
    let _serial = serial();
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let ids = bank().expect("bank 1 (calibration)");
    let exact = subjects::open(Path::new(&env("LARQL_CR3_F32")), "gemma3-4b-it");
    let yard = subjects::open_bound(
        Path::new(&env("LARQL_CR3_Q4K")),
        "gemma3-4b-it.q4k",
        Some(YARDSTICK_ENCODING),
    );
    full_precision_guard(exact.store.selection()).unwrap_or_else(|s| panic!("{s}"));
    yardstick_guard(yard.store.selection()).unwrap_or_else(|s| panic!("{s}"));
    let windows: Vec<usize> = std::env::var("LARQL_CR3_WINDOWS").map_or_else(
        |_| SWEEP.to_vec(),
        |s| {
            s.split(',')
                .map(|w| {
                    w.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("bad window {w}"))
                })
                .collect()
        },
    );
    let stamp =
        json!({"implementation_git_sha": git_sha(), "bank": "codec-1 bank 1 (calibration)"});
    let store = Checkpoints::open(Path::new(&env("LARQL_CR3_RUN_DIR")), &stamp);
    let (map_store, map_stamp) = Checkpoints::open_as_run(Path::new(&env("LARQL_CR3_MAP_STORE")));
    eprintln!("reference store: {map_stamp}");
    let backend = ProductionBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let report = run_codec_3_recon(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids,
        &RECON,
        (&windows, PARITY_WINDOW),
        (&store, &map_store),
    );
    eprintln!(
        "report: {}",
        serde_json::to_string_pretty(&report["arms"]).unwrap()
    );
}
