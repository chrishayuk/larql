//! One fact: baseline, an observed install, post-install measurements, and the §6 outcome.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use larql_lql::ast::{InsertMode, Statement};
use larql_lql::{parse, Session};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::bank::{hex, Fact};
use crate::consts::*;
use crate::infer::{predict, prob_of, top1, Pred};
use crate::observer::{Baseline, Trajectory};

pub struct Settings {
    pub container: PathBuf,
    pub alpha: Option<f32>,
    pub unrelated: usize,
    pub check_determinism: bool,
    pub out_dir: PathBuf,
}

fn quoted(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

pub fn new_session(container: &Path) -> Result<Session, String> {
    let mut session = Session::new();
    let statement = parse(&format!("USE \"{}\";", quoted(container))).map_err(|e| e.to_string())?;
    session.execute(&statement).map_err(|e| e.to_string())?;
    Ok(session)
}

fn insert(fact: &Fact, alpha: Option<f32>) -> Statement {
    Statement::Insert {
        entity: fact.entity.clone(),
        relation: fact.relation.clone(),
        target: fact.target.clone(),
        layer: None,
        confidence: None,
        alpha,
        mode: InsertMode::Compose,
    }
}

/// Runs the install between BEGIN PATCH and SAVE PATCH and returns the saved bytes' digest: the
/// patch carries the final post-refine, post-balance vectors.
fn install_saving(
    session: &mut Session,
    fact: &Fact,
    alpha: Option<f32>,
    patch: &Path,
) -> Result<String, String> {
    let begin = parse(&format!("BEGIN PATCH \"{}\";", quoted(patch))).map_err(|e| e.to_string())?;
    session.execute(&begin).map_err(|e| e.to_string())?;
    session
        .execute(&insert(fact, alpha))
        .map_err(|e| e.to_string())?;
    session
        .execute(&Statement::SavePatch)
        .map_err(|e| e.to_string())?;
    let bytes = std::fs::read(patch).map_err(|e| format!("{}: {e}", patch.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

fn measure(
    session: &mut Session,
    prompts: &[String],
) -> Result<HashMap<String, Vec<Pred>>, String> {
    prompts
        .iter()
        .map(|p| Ok((p.clone(), predict(session, p)?)))
        .collect()
}

fn pick(all: &[String], n: usize) -> Vec<String> {
    let stride = (all.len() / n.max(1)).max(1);
    all.iter().step_by(stride).take(n).cloned().collect()
}

pub fn run(
    fact: &Fact,
    unrelated_all: &[String],
    settings: &Settings,
    tokenizer: &Tokenizer,
) -> Result<Value, String> {
    let unrelated = pick(unrelated_all, settings.unrelated);
    let mut prompts = vec![fact.canonical_prompt.clone()];
    prompts.extend(fact.same_template.iter().cloned());
    prompts.extend(fact.related.iter().cloned());
    prompts.extend(unrelated.iter().cloned());

    let target_ids = tokenizer
        .encode(format!(" {}", fact.target), false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let target_id = *target_ids.first().ok_or("target has no tokens")?;
    let target_pieces: Vec<String> = target_ids
        .iter()
        .map(|i| tokenizer.decode(&[*i], false).unwrap_or_default())
        .collect();

    // Baseline, on a session with nothing installed.
    let mut clean = new_session(&settings.container)?;
    let base = measure(&mut clean, &prompts)?;
    drop(clean);
    let baseline: Baseline = base
        .iter()
        .map(|(p, v)| (p.clone(), (top1(v).id, top1(v).p)))
        .collect();

    // The observed install.
    let events = Arc::new(Mutex::new(Vec::new()));
    let watched: Vec<String> = fact
        .same_template
        .iter()
        .take(TRAJECTORY_NEIGHBOURS)
        .chain(unrelated.iter().take(TRAJECTORY_UNRELATED))
        .cloned()
        .collect();
    let mut session = new_session(&settings.container)?;
    session.set_balance_observer(Some(Box::new(Trajectory {
        events: events.clone(),
        baseline: Arc::new(baseline.clone()),
        target_id,
        watched,
    })));
    std::fs::create_dir_all(settings.out_dir.join("patches")).map_err(|e| e.to_string())?;
    let patch = settings
        .out_dir
        .join("patches")
        .join(format!("{}-{}.vlp", fact.relation, fact.entity));
    let fingerprint_before = session.state_fingerprint();
    let installed = install_saving(&mut session, fact, settings.alpha, &patch);
    let fingerprint_after = session.state_fingerprint();
    let observer_failures = session.balance_observer_failures();

    let trajectory = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
    // The prompt the install captured its residual from, as the install itself reports it.
    let install_prompt = trajectory
        .iter()
        .find_map(|e| e["canonical"]["prompt"].as_str().map(str::to_string));
    let mut out = json!({
        "fact": {"relation": fact.relation, "entity": fact.entity, "target": fact.target},
        "canonical_install_prompt": install_prompt,
        "evaluation_prompt": fact.canonical_prompt,
        "install_and_evaluation_prompts_equal": install_prompt.as_deref() == Some(fact.canonical_prompt.as_str()),
        "target": {"first_token_id": target_id, "token_ids": target_ids, "pieces": target_pieces},
        "alpha": settings.alpha,
        "trajectory": trajectory,
        "observer_failures": observer_failures,
        "state_fingerprint": {
            "before_install": fingerprint_before.components(),
            "after_install": fingerprint_after.components(),
            "changed_by_install": fingerprint_after.changed_since(&fingerprint_before),
        },
        "atomicity": "measured by --set faults (prereg §7.1), not by an ordinary install",
    });
    let digest = match installed {
        Ok(d) => d,
        Err(e) => {
            out["outcome"] =
                json!({"class": "refused", "error": e, "state_unchanged_verified": false});
            return Ok(out);
        }
    };
    out["install_digest"] = json!(digest);

    let post = measure(&mut session, &prompts)?;
    out["measurements"] = score(fact, &unrelated, target_id, &base, &post);
    out["outcome"] = json!({"class": out["measurements"]["class"]});

    if settings.check_determinism {
        let mut again = new_session(&settings.container)?;
        let second = settings
            .out_dir
            .join("patches")
            .join(format!("{}-{}-again.vlp", fact.relation, fact.entity));
        let again_digest = install_saving(&mut again, fact, settings.alpha, &second)?;
        out["determinism"] = json!({
            "install_digest_equal_without_observer": again_digest == digest,
            "second_digest": again_digest,
        });
    }
    Ok(out)
}

fn score(
    fact: &Fact,
    unrelated: &[String],
    target_id: u32,
    base: &HashMap<String, Vec<Pred>>,
    post: &HashMap<String, Vec<Pred>>,
) -> Value {
    let kept = |prompt: &String| {
        let (b, p) = (top1(&base[prompt]), &post[prompt]);
        let same_top1 = top1(p).id == b.id;
        let delta = (prob_of(p, b.id) - b.p).abs();
        (same_top1, delta, b.p)
    };
    let canonical = &post[&fact.canonical_prompt];
    let canonical_ok = top1(canonical).id == target_id && top1(canonical).p >= TARGET_P_MIN;

    let unrelated_changed: Vec<&String> = unrelated.iter().filter(|p| !kept(p).0).collect();
    let scored: Vec<&String> = fact
        .same_template
        .iter()
        .filter(|p| kept(p).2 >= NEIGHBOUR_BASE_P_MIN)
        .collect();
    let neighbours_kept = scored
        .iter()
        .filter(|p| {
            let (same, delta, _) = kept(p);
            same && delta <= NEIGHBOUR_DELTA_P_MAX
        })
        .count();
    let neighbour_share = if scored.is_empty() {
        1.0
    } else {
        neighbours_kept as f64 / scored.len() as f64
    };
    let bleed: Vec<&String> = fact
        .same_template
        .iter()
        .filter(|p| top1(&post[*p]).id == target_id && top1(&base[*p]).id != target_id)
        .collect();

    let safe = canonical_ok
        && unrelated_changed.is_empty()
        && neighbour_share >= NEIGHBOUR_KEEP
        && bleed.is_empty();
    json!({
        "class": if safe { "installed_safe" } else { "installed_unsafe" },
        "canonical_top1_is_target": top1(canonical).id == target_id,
        "canonical_top1_p": top1(canonical).p,
        "canonical_target_p": prob_of(canonical, target_id),
        "canonical_ok": canonical_ok,
        "unrelated_measured": unrelated.len(),
        "unrelated_top1_changed": unrelated_changed,
        "same_template_scored": scored.len(),
        "same_template_kept": neighbours_kept,
        "same_template_keep_share": neighbour_share,
        "same_template_target_bleed": bleed,
        "per_prompt": fact.same_template.iter().chain(fact.related.iter()).chain(unrelated.iter()).map(|p| {
            let (same, delta, bp) = kept(p);
            json!({"prompt": p, "baseline_top1_id": top1(&base[p]).id, "baseline_top1_p": bp,
                   "post_top1_id": top1(&post[p]).id, "post_top1_p": top1(&post[p]).p,
                   "top1_kept": same, "delta_p_of_baseline_top1": delta})
        }).collect::<Vec<_>>(),
        "above_prob_ceiling": prob_of(canonical, target_id) > REPORT_CEILING,
    })
}
