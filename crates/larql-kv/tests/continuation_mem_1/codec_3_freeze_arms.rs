//! CONTINUATION-CODEC-3 W1: the frozen arms over a ladder, and the record.
//!
//! One orchestration serves the frozen run and the executed fixture
//! control. Order per rung: R (logits kept) → NULL (guard) → C (scores,
//! logits kept under their own base — CODEC-2 W2's overwrite fault) → the
//! compressed arms in `ARMS` order, each traced, CH first. Every compressed
//! arm is a stop guard twice over: its residency must equal what it
//! declares, and each CH-recent arm's retention-and-read trace must equal
//! CH's at that rung.

use std::collections::BTreeMap;

use larql_kv::{CodecKvState, CodecRecentKvState};
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    summarise, PositionMetrics,
};

use super::codec_1_c3::{decode_rows, null_guard, score};
use super::codec_2_arms::{strata, Ladder};
use super::codec_2_store::{logits_base, scores_name, Checkpoints};
use super::codec_3_freeze::{adjudicate_bound, ArmSpec, Binding, Provider, ARMS, CODEC_BITS};
use super::codec_map_sim::{Recorder, Retained};
use super::measured::{CodecResidency, MixedResidency};
use super::*;

/// A CH-recent arm's trace must be CH's: the window changes bytes per row,
/// never which rows are held or read.
pub(super) fn trace_guard(arm: &str, n: usize, ch: &Value, got: &Value) -> Result<(), String> {
    if ch == got {
        Ok(())
    } else {
        Err(format!(
            "rung {n}: {arm} trace {got} is not CH's {ch} — provider fault"
        ))
    }
}

/// codec/v1's residency, CODEC-1's equation.
pub(super) fn codec_residency_guard(
    arm: &str,
    n: usize,
    r: Option<CodecResidency>,
) -> Result<Value, String> {
    let r = r.ok_or_else(|| format!("rung {n}: {arm} reported no codec residency"))?;
    if !r.holds() {
        return Err(format!("rung {n}: {arm} residency regression — {r:?}"));
    }
    Ok(
        json!({"holds": true, "append_born_live": r.append_born_live, "expected": r.expected,
        "strays": r.strays, "scratch_bytes": r.scratch_bytes, "scratch_bound": r.scratch_bound}),
    )
}

/// codec-recent/v1's residency against what its window declares.
pub(super) fn mixed_residency_guard(
    arm: &str,
    n: usize,
    r: Option<MixedResidency>,
) -> Result<Value, String> {
    let r = r.ok_or_else(|| format!("rung {n}: {arm} reported no mixed layout"))?;
    if !r.holds() {
        return Err(format!("rung {n}: {arm} residency regression — {r:?}"));
    }
    Ok(
        json!({"holds": true, "append_born_live": r.append_born_live, "expected": r.expected,
        "v_code_bytes": r.v_code_bytes, "k_code_bytes": r.k_code_bytes,
        "k_exact_bytes": r.k_exact_bytes, "list_bytes": r.list_bytes,
        "strays": r.strays, "window_mismatches": r.window_mismatches,
        "scratch_bytes": r.scratch_bytes, "scratch_bound": r.scratch_bound}),
    )
}

/// A stored residency, recomputed from its recorded fields — never the
/// recorded `holds` flag alone, so an inconsistent checkpoint is refused.
pub(super) fn stored_residency_guard(arm: &str, n: usize, r: &Value) -> Result<(), String> {
    let field = |k: &str| r[k].as_u64();
    // Only a mixed provider records window mismatches; CH has none.
    let zero = |k: &str| field(k).is_none_or(|v| v == 0);
    let holds = r["holds"] == true
        && field("append_born_live").is_some()
        && field("append_born_live") == field("expected")
        && field("strays") == Some(0)
        && zero("window_mismatches")
        && matches!((field("scratch_bytes"), field("scratch_bound")), (Some(b), Some(m)) if b <= m);
    if holds {
        Ok(())
    } else {
        Err(format!(
            "rung {n}: stored {arm} residency does not hold — {r}"
        ))
    }
}

/// A checkpoint is evidence only while its guards still hold: every
/// stored compressed arm's residency, and each CH-recent arm's trace
/// against CH's stored trace, are re-checked on resume and before the
/// record — a checkpoint's existence is never taken as proof.
pub(super) fn revalidate(spec: ArmSpec, n: usize, store: &Checkpoints) -> Result<(), String> {
    let stored: Value = store
        .read(&scores_name(n, &spec.key()))
        .ok_or_else(|| format!("rung {n}: {} has no checkpoint", spec.name))?;
    stored_residency_guard(spec.name, n, &stored["residency"])?;
    if spec.provider != Provider::Codec {
        let ch: Value = store
            .read(&scores_name(n, &ARMS[0].key()))
            .ok_or_else(|| format!("rung {n}: CH has no checkpoint"))?;
        trace_guard(spec.name, n, &ch["trace"], &stored["trace"])?;
    }
    Ok(())
}

/// One traced arm: decode logits, the trace, and `read` of the measured
/// provider after the journey.
fn traced<P: Inspect + Retained, B: PlanBackend, R>(
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    inner: P,
    journey: &Journey,
    read: impl Fn(&Measured<Recorder<P>>) -> R,
) -> (Vec<Vec<f32>>, Value, R) {
    let mut kv = Measured::new(Recorder::new(inner));
    let out = subjects::run(yard.0, yard.1, backend, &mut kv, journey);
    let rows = out
        .logits
        .into_iter()
        .filter(|(phase, _)| *phase == subjects::DECODE)
        .map(|(_, r)| r)
        .collect();
    let (hash, events) = kv.inner.trace();
    (rows, json!([hash.to_string(), events]), read(&kv))
}

/// Run one compressed arm: logits, trace, residency (a stop guard).
fn compressed<B: PlanBackend>(
    spec: ArmSpec,
    n: usize,
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    journey: &Journey,
) -> (Vec<Vec<f32>>, Value, Value) {
    let stop = |r: Result<Value, String>| r.unwrap_or_else(|stop| panic!("{stop}"));
    match spec.provider {
        Provider::Codec => {
            let (rows, trace, r) = traced(
                yard,
                backend,
                CodecKvState::new(CODEC_BITS),
                journey,
                |kv| kv.codec_residency(),
            );
            (rows, trace, stop(codec_residency_guard(spec.name, n, r)))
        }
        Provider::Recent { window } => {
            let inner = CodecRecentKvState::new(CODEC_BITS, window);
            let (rows, trace, r) = traced(yard, backend, inner, journey, |kv| {
                kv.mixed_residency(window)
            });
            (rows, trace, stop(mixed_residency_guard(spec.name, n, r)))
        }
    }
}

/// Run every arm the store lacks. `exact` must run the full-precision
/// stack and `yard` the Q4_K pack — the caller guards both.
pub(super) fn run_arms<B: PlanBackend>(
    exact: (&Subject, &PreparedOperands),
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    ids: &[u32],
    ladder: &Ladder,
    store: &Checkpoints,
) {
    for &n in &ladder.rungs {
        let (journey, start, next) = ladder.journey(ids, n);
        let timed = |arm: &str, t: std::time::Instant| {
            eprintln!("rung {n}: {arm} done in {:.0}s", t.elapsed().as_secs_f64());
            t.elapsed().as_secs_f64()
        };
        let r = store.read_logits(&logits_base(n, "r")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (r, _) = decode_rows(exact.0, exact.1, backend, RowKvState::default(), &journey);
            store.write_logits(&logits_base(n, "r"), &r);
            timed("R", t);
            r
        });
        match store.read::<Value>(&scores_name(n, "null")) {
            Some(null) if null["passed"] != true => {
                panic!("rung {n}: recorded NULL stop: {}", null["stop"])
            }
            Some(_) => {}
            None => {
                let t = std::time::Instant::now();
                let (null, _) =
                    decode_rows(exact.0, exact.1, backend, WindowKvState::new(), &journey);
                let verdict = null_guard(&r, &null);
                store.write(
                    &scores_name(n, "null"),
                    &json!({"passed": verdict.is_ok(),
                    "stop": verdict.as_ref().err(), "seconds": timed("NULL", t)}),
                );
                if let Err(stop) = verdict {
                    panic!("rung {n}: {stop}");
                }
            }
        }
        let c = store.read_logits(&logits_base(n, "c")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (c, _) = decode_rows(yard.0, yard.1, backend, RowKvState::default(), &journey);
            store.write(
                &scores_name(n, "c"),
                &json!({"arm": "c", "n": n, "seconds": timed("C", t),
                "positions": score(&r, &c, next, n, start)}),
            );
            store.write_logits(&logits_base(n, "c"), &c);
            c
        });
        for spec in ARMS {
            let key = spec.key();
            if store.read::<Value>(&scores_name(n, &key)).is_some() {
                revalidate(spec, n, store).unwrap_or_else(|stop| panic!("{stop}"));
                continue;
            }
            let t = std::time::Instant::now();
            let (rows, trace, residency) = compressed(spec, n, yard, backend, &journey);
            if spec.provider != Provider::Codec {
                let ch: Value = store
                    .read(&scores_name(n, &ARMS[0].key()))
                    .expect("CH runs first at every rung");
                trace_guard(spec.name, n, &ch["trace"], &trace)
                    .unwrap_or_else(|stop| panic!("{stop}"));
            }
            store.write(
                &scores_name(n, &key),
                &json!({"arm": spec.name, "n": n, "seconds": timed(spec.name, t),
                "trace": trace, "residency": residency,
                "positions": score(&r, &rows, next, n, start),
                "positions_vs_c": score(&c, &rows, next, n, start)}),
            );
        }
    }
}

/// The record: the bound verdict (the primary's alone), the report-only
/// adjudications, summaries, and per-rung guards, residency and time.
pub(super) fn record(
    stamp: &Value,
    binding: &Binding,
    store: &Checkpoints,
    ladder: &Ladder,
) -> Value {
    for &n in &ladder.rungs {
        for spec in ARMS {
            revalidate(spec, n, store).unwrap_or_else(|stop| panic!("{stop}"));
        }
    }
    let c = strata(store, ladder, "c", "positions");
    let arms: BTreeMap<String, Vec<Vec<PositionMetrics>>> = ARMS
        .iter()
        .map(|a| {
            (
                a.name.to_string(),
                strata(store, ladder, &a.key(), "positions"),
            )
        })
        .collect();
    let summary = |v: &[Vec<PositionMetrics>]| summarise(&v.concat()).unwrap();
    let mut reported = serde_json::Map::new();
    reported.insert("r_c".into(), json!(summary(&c)));
    for a in ARMS {
        let vs_c = strata(store, ladder, &a.key(), "positions_vs_c");
        reported.insert(
            a.name.into(),
            json!({"r_arm": summary(&arms[a.name]), "c_arm_kl": summary(&vs_c)}),
        );
    }
    let rungs: Vec<Value> = ladder
        .rungs
        .iter()
        .map(|&n| {
            let arm = |k: &str| {
                store
                    .read::<Value>(&scores_name(n, k))
                    .unwrap_or(Value::Null)
            };
            let per = |field: &str| -> Value {
                ARMS.iter()
                    .map(|a| (a.name.to_string(), arm(&a.key())[field].clone()))
                    .collect()
            };
            json!({"n": n, "null": arm("null"), "c_seconds": arm("c")["seconds"],
                "residency": per("residency"), "trace": per("trace"), "seconds": per("seconds")})
        })
        .collect();
    json!({
        "programme": "CONTINUATION-CODEC-3",
        "authorities": stamp,
        "adjudication": adjudicate_bound(binding, &c, &arms),
        "reported": reported,
        "rungs": rungs,
    })
}
