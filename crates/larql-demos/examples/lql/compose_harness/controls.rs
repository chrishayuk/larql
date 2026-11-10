//! Controls: cases whose outcome was observed before the harness existed. The harness must
//! reproduce them under the arm it is built for, or nothing it reports can be trusted.

use std::path::Path;

use larql_lql::ast::{InsertMode, Statement};
use larql_lql::balance_trace::{BalanceStep, TraceEvent, TraceRecorder};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::consts::CONTROL_TOLERANCE_PCT;
use crate::infer::{predict, top1};

#[derive(Debug, Deserialize)]
struct Expect {
    prompt: String,
    top1: String,
    pct: f64,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    tag: String,
    arm: String,
    alpha: Option<f32>,
    entity: String,
    relation: String,
    target: String,
    expect: Vec<Expect>,
}

#[derive(Debug, Deserialize)]
struct File {
    cases: Vec<Case>,
}

/// Runs every control recorded for `tag` and `arm`; true when all of them reproduce.
pub fn run(
    controls_path: &Path,
    container: &Path,
    tag: &str,
    arm: &str,
) -> Result<(bool, Value), String> {
    let bytes =
        std::fs::read(controls_path).map_err(|e| format!("{}: {e}", controls_path.display()))?;
    let file: File = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let cases: Vec<&Case> = file
        .cases
        .iter()
        .filter(|c| c.tag == tag && c.arm == arm)
        .collect();
    if cases.is_empty() {
        return Err(format!(
            "no control is recorded for tag {tag:?} and arm {arm:?}"
        ));
    }
    let mut all_pass = true;
    let mut results = Vec::new();
    for case in cases {
        let recorder = TraceRecorder::new();
        let mut session = crate::fact::new_session(container)?;
        session.set_balance_observer(Some(Box::new(recorder.clone())));
        let statement = Statement::Insert {
            entity: case.entity.clone(),
            relation: case.relation.clone(),
            target: case.target.clone(),
            layer: None,
            confidence: None,
            alpha: case.alpha,
            mode: InsertMode::Compose,
        };
        session
            .execute(&statement)
            .map_err(|e| format!("{}: {e}", case.id))?;
        let mut checks = Vec::new();
        for expect in &case.expect {
            let preds = predict(&mut session, &expect.prompt)?;
            let got = top1(&preds);
            let got_pct = got.p * 100.0;
            let pass = got.text.trim() == expect.top1.trim()
                && (got_pct - expect.pct).abs() <= CONTROL_TOLERANCE_PCT;
            all_pass &= pass;
            checks.push(json!({"prompt": expect.prompt, "expected": {"top1": expect.top1, "pct": expect.pct},
                               "got": {"top1": got.text, "pct": got_pct}, "pass": pass}));
        }
        // What the controller itself read: a failure to match shows here, apart from strength.
        let controller_reads: Vec<f64> = recorder
            .events()
            .iter()
            .filter_map(|e| {
                if let TraceEvent::Step(s) = e {
                    Some(s)
                } else {
                    None
                }
            })
            .map(|s: &BalanceStep| s.controller_target_prob)
            .collect();
        results.push(json!({"id": case.id, "alpha": case.alpha, "checks": checks, "controller_reads": controller_reads}));
    }
    Ok((all_pass, json!(results)))
}
