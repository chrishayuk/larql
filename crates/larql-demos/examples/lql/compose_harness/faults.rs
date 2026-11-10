//! Failure atomicity as a measured outcome (prereg §7.1): inject each fault point of
//! `larql_lql::install_fault` into an install of a development fact, in a fresh session, and report
//! which components of the session fingerprint the failed install changed and whether any
//! prediction on the atomicity probe set moved.
//!
//! This measures the implementation as it is. It never feeds the balancing algorithm, and a session
//! in which an install failed is discarded after its one measurement.

use std::path::Path;

use larql_lql::ast::Statement;
use larql_lql::install_fault::InstallFaultPoint;
use larql_lql::{parse, Session};
use serde_json::{json, Value};

use crate::bank::{Bank, Fact};
use crate::infer::{predict, Pred};

/// The seven points of PR #706, in the order an install reaches them. The balance point is the
/// first iteration: a later one fires only if the controller runs that long.
const POINTS: [InstallFaultPoint; 7] = [
    InstallFaultPoint::BeforeInstall,
    InstallFaultPoint::AfterCachesCommitted,
    InstallFaultPoint::AfterSlotStaged,
    InstallFaultPoint::AfterRefine,
    InstallFaultPoint::DuringBalance { iteration: 0 },
    InstallFaultPoint::AfterBalance,
    InstallFaultPoint::BeforeRecord,
];

fn install(session: &mut Session, fact: &Fact) -> Result<(), String> {
    let statement = Statement::Insert {
        entity: fact.entity.clone(),
        relation: fact.relation.clone(),
        target: fact.target.clone(),
        layer: None,
        confidence: None,
        alpha: None,
        mode: larql_lql::ast::InsertMode::Compose,
    };
    session
        .execute(&statement)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn begin_patch(session: &mut Session, path: &Path) -> Result<(), String> {
    let text = format!(
        "BEGIN PATCH \"{}\";",
        path.display().to_string().replace('\\', "\\\\")
    );
    let statement = parse(&text).map_err(|e| e.to_string())?;
    session
        .execute(&statement)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The printed top candidates for each probe prompt: ids with probabilities as the session prints
/// them (two decimals). Equality of these is the harness's "predictions unchanged".
fn probe(session: &mut Session, prompts: &[String]) -> Result<Vec<Vec<Pred>>, String> {
    prompts.iter().map(|p| predict(session, p)).collect()
}

fn moved(before: &[Vec<Pred>], after: &[Vec<Pred>], prompts: &[String]) -> Vec<String> {
    prompts
        .iter()
        .zip(before.iter().zip(after))
        .filter(|(_, (b, a))| b != a)
        .map(|(p, _)| p.clone())
        .collect()
}

/// One cell: a session, optionally with `earlier` already installed, then `fact` forced to fail at
/// `point`.
fn cell(
    container: &Path,
    scratch: &Path,
    fact: &Fact,
    earlier: Option<&Fact>,
    point: InstallFaultPoint,
    probes: &[String],
) -> Result<Value, String> {
    let mut session = crate::fact::new_session(container)?;
    let patch = scratch.join(format!("fault-{}-{}.vlp", fact.entity, point));
    begin_patch(&mut session, &patch)?;
    if let Some(e) = earlier {
        install(&mut session, e)?;
    }
    let fp_before = session.state_fingerprint();
    let preds_before = probe(&mut session, probes)?;
    let earlier_before = earlier
        .map(|e| probe(&mut session, std::slice::from_ref(&e.canonical_prompt)))
        .transpose()?;

    session.set_install_fault(Some(point));
    let result = install(&mut session, fact);
    // A fault that was never reached stays armed; clear it so it cannot fire later.
    session.set_install_fault(None);

    let fp_after = session.state_fingerprint();
    let preds_after = probe(&mut session, probes)?;
    let earlier_after = earlier
        .map(|e| probe(&mut session, std::slice::from_ref(&e.canonical_prompt)))
        .transpose()?;

    let (fired, error) = match result {
        Err(e) => (e.contains("injected fault"), Some(e)),
        Ok(()) => (false, None),
    };
    Ok(json!({
        "point": point.to_string(),
        "fired": fired,
        "error": error,
        "changed_components": fp_after.changed_since(&fp_before),
        "probe_prompts_moved": moved(&preds_before, &preds_after, probes),
        "earlier_install_prediction_preserved": earlier_before.map(|b| Some(b) == earlier_after),
        "state_unchanged": fp_after == fp_before && preds_before == preds_after,
    }))
}

/// Every development fact × every fault point, on a fresh session, and again with another
/// development fact installed first. Needs at least two development facts for the second form.
pub fn run(container: &Path, bank: &Bank) -> Result<Value, String> {
    let scratch = std::env::temp_dir().join(format!("compose-faults-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let mut cells = Vec::new();
    for (i, fact) in bank.development.iter().enumerate() {
        let earlier = bank
            .development
            .iter()
            .enumerate()
            .find(|(j, _)| *j != i)
            .map(|(_, f)| f);
        for scenario in ["fresh", "after_earlier_success"] {
            let prior = if scenario == "fresh" { None } else { earlier };
            if scenario != "fresh" && prior.is_none() {
                continue;
            }
            for point in POINTS {
                eprintln!(
                    "[faults] {} {} {scenario} {point}",
                    fact.relation, fact.entity
                );
                let mut c = cell(
                    container,
                    &scratch,
                    fact,
                    prior,
                    point,
                    &bank.atomicity_probe,
                )?;
                c["fact"] = json!({"relation": fact.relation, "entity": fact.entity});
                c["scenario"] = json!(scenario);
                cells.push(c);
            }
        }
    }
    let violated = cells
        .iter()
        .filter(|c| c["fired"] == true && c["state_unchanged"] == false)
        .count();
    let fired = cells.iter().filter(|c| c["fired"] == true).count();
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(json!({
        "probe_prompts": bank.atomicity_probe.len(),
        "probe_comparison": "printed top-20 ids and probabilities (two decimals); not raw logits",
        "cells": cells,
        "fired": fired,
        "violations": violated,
    }))
}
