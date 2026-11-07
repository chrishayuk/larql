//! LQL `INFER … TOP k` against the canonical V3 decode session, on a
//! container that declares what a real Granite checkpoint declares.
//!
//! The gate: for the same user prompt, LQL's top-k ids **and** printed
//! probabilities equal the full-vocabulary softmax of the logits the
//! canonical session (`prefill_into` → the provider `session_with_kv`
//! resumes — the stack `larql run`, `vindex3 observe` and the server
//! drive) produces from the prompt's bare tokenizer encoding.
//!
//! Why the fixture is shaped the way it is. Granite 4.1 3B answered
//! " is" (32%) to "The capital of France is" under LQL while every other
//! surface answered " Paris": its checkpoint declares `bos_token_id`
//! 100257 *and* `add_bos_token: false`, and LQL prepended the id anyway.
//! The existing LQL/V3 parity gates could not see that — their containers
//! carry no generation or tokenizer config, so both arms agreed by
//! accident. This container carries both declarations, plus the four
//! Granite scaling scalars (embedding, residual, attention multipliers
//! and logits scaling), so the parity also covers the program's
//! non-identity scalars over a multi-position prompt (a single position
//! would make the attention multiplier invisible: softmax over one key
//! is 1 whatever the scale).

use std::path::Path;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_inference::vindex3::Vindex3Runtime;
use larql_kv::CanonicalKvState;
use larql_lql::{parse, Session};
use larql_vindex::format::filenames::{GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

const COMPONENT: &str = "target";
const CONTAINER_NAME: &str = "granite-scalars-fixture";
const TOP_K: usize = 5;
/// Four whitespace-separated word-level tokens.
const PROMPT: &str = "[3] [17] [28] [5]";
/// The id the container declares as BOS — like Granite's
/// `<|end_of_text|>`, a real vocabulary entry, and not one the prompt uses.
const DECLARED_BOS: u32 = 1;

/// Granite-style declared scalars, each a non-identity op in the plan.
/// The first three are Granite 4.1 3B's own values.
const EMBEDDING_MULTIPLIER: f64 = 12.0;
const RESIDUAL_MULTIPLIER: f64 = 0.22;
const ATTENTION_MULTIPLIER: f64 = 0.015625;
/// `logits_scaling` becomes the head's `1 / scaling` multiplier. Granite
/// declares 10 (×0.1); on this miniature's ±0.05 weights that flattens
/// the head to within 0.01% of uniform over 128 ids, where no printed
/// probability can move and the control below fails. 0.1 (×10) keeps
/// the scalar non-identity and gives the distribution a shape.
const LOGITS_SCALING: f64 = 0.1;

/// LQL prints probabilities as percentages to two decimals; a faithful
/// print is within half a unit in the last place of the exact value.
const PERCENT_DISPLAY_HALF_ULP: f64 = 0.005;
/// Headroom for f32 summation order between the two softmaxes.
const PERCENT_SUM_ORDER_SLACK: f64 = 1e-4;

/// The dense miniature re-declared as a Granite checkpoint.
fn granite_scalars_model(dir: &Path) {
    dense_f32_model(dir);
    let path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    config["architectures"] = serde_json::json!(["GraniteForCausalLM"]);
    config["model_type"] = serde_json::json!("granite");
    config["embedding_multiplier"] = serde_json::json!(EMBEDDING_MULTIPLIER);
    config["residual_multiplier"] = serde_json::json!(RESIDUAL_MULTIPLIER);
    config["attention_multiplier"] = serde_json::json!(ATTENTION_MULTIPLIER);
    config["logits_scaling"] = serde_json::json!(LOGITS_SCALING);
    config["bos_token_id"] = serde_json::json!(DECLARED_BOS);
    std::fs::write(&path, config.to_string()).unwrap();
}

/// The synthetic word-level tokenizer, split on whitespace so a prompt
/// spans several positions. Its post-processor adds nothing — like
/// Granite's and Qwen3's, BOS is not part of the encoding.
fn whitespace_tokenizer_json() -> String {
    let mut json: serde_json::Value =
        serde_json::from_str(&synthetic_tokenizer_json(DENSE_VOCAB)).unwrap();
    json["pre_tokenizer"] = serde_json::json!({"type": "WhitespaceSplit"});
    json.to_string()
}

/// Encode the container and place beside it what the M2 capability
/// snapshot carries from a real Granite checkpoint: a generation config
/// naming the BOS id, a tokenizer config declaring it is not added.
fn granite_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        granite_scalars_model,
        checkpoint.path(),
        container.path(),
        CONTAINER_NAME,
    );
    let root = container.path();
    std::fs::write(root.join("tokenizer.json"), whitespace_tokenizer_json()).unwrap();
    std::fs::write(
        root.join(GENERATION_CONFIG_JSON),
        serde_json::json!({"bos_token_id": DECLARED_BOS, "eos_token_id": DECLARED_BOS}).to_string(),
    )
    .unwrap();
    std::fs::write(
        root.join(TOKENIZER_CONFIG_JSON),
        serde_json::json!({"add_bos_token": false, "bos_token": "[1]"}).to_string(),
    )
    .unwrap();
    container
}

fn lql_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

/// The prompt as the canonical surfaces encode it: the tokenizer's own
/// encoding, special tokens on, nothing prepended.
fn canonical_prompt_ids(container: &Path) -> Vec<u32> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec()
}

/// The canonical decode session's next-token logits for `ids`.
fn canonical_logits(container: &Path, ids: &[u32]) -> Vec<f32> {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let mut kv = CanonicalKvState::new();
    runtime.prefill_into(ids, &mut kv).unwrap()
}

/// Full-vocabulary softmax in f64, ranked; ties keep the lower id.
fn ranked_probabilities(logits: &[f32]) -> Vec<(u32, f64)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut ranked: Vec<(u32, f64)> = exps
        .iter()
        .enumerate()
        .map(|(id, e)| (id as u32, e / sum))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked
}

/// `(id, percent)` from each `  n. token (p%)  [id x]` row.
fn lql_rows(out: &[String]) -> Vec<(u32, f64)> {
    out.iter()
        .filter_map(|line| {
            let id = line.split("[id ").nth(1)?.trim_end_matches(']').trim();
            let percent = line.split('(').nth(1)?.split('%').next()?.trim();
            Some((id.parse().ok()?, percent.parse().ok()?))
        })
        .collect()
}

fn lql_infer(container: &Path) -> Vec<(u32, f64)> {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(container)));
    let out = run(&mut session, &format!("INFER \"{PROMPT}\" TOP {TOP_K};"));
    let rows = lql_rows(&out);
    assert_eq!(rows.len(), TOP_K, "one row per requested rank: {out:?}");
    rows
}

fn assert_rows_match(lql: &[(u32, f64)], canonical: &[(u32, f64)]) {
    let tolerance = PERCENT_DISPLAY_HALF_ULP + PERCENT_SUM_ORDER_SLACK;
    for (rank, ((lql_id, lql_percent), (id, prob))) in lql.iter().zip(canonical).enumerate() {
        assert_eq!(
            lql_id, id,
            "rank {rank}: LQL {lql:?} vs canonical {canonical:?}"
        );
        let percent = prob * 100.0;
        assert!(
            (lql_percent - percent).abs() <= tolerance,
            "rank {rank} (id {id}): LQL prints {lql_percent}% but the canonical logits give \
             {percent}%"
        );
    }
}

/// The fixture is the Granite-scalar program it claims to be: every
/// scalar reached the plan as a non-identity op.
#[test]
fn the_fixture_carries_the_granite_scalars_into_the_plan() {
    let container = granite_container();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let plan = runtime.plan();
    let embedding = plan.embedding.as_ref().expect("embedding op");
    assert_eq!(embedding.scale, Some(EMBEDDING_MULTIPLIER as f32));
    let output = plan.output.as_ref().expect("output head");
    let multiplier = output
        .multiplier
        .expect("logits scaling becomes a head multiplier");
    assert!((multiplier - 1.0 / LOGITS_SCALING).abs() < f64::EPSILON);
    for layer in &plan.layers {
        assert_eq!(layer.residual_scale, Some(RESIDUAL_MULTIPLIER as f32));
        let attention = layer.attention.softmax().expect("softmax attention");
        assert!((attention.score_scale - ATTENTION_MULTIPLIER).abs() < f64::EPSILON);
    }
}

/// **The control — keep it with the gate below.** Prepending the
/// declared BOS must move this fixture's distribution, or the parity
/// gate could not see the defect it exists to rule out.
#[test]
fn the_fixture_can_see_a_prepended_bos() {
    let container = granite_container();
    let ids = canonical_prompt_ids(container.path());
    assert!(ids.len() > 1, "a multi-position prompt: {ids:?}");
    assert_ne!(ids[0], DECLARED_BOS, "the tokenizer adds no BOS: {ids:?}");

    let mut with_bos = vec![DECLARED_BOS];
    with_bos.extend_from_slice(&ids);
    let bare = ranked_probabilities(&canonical_logits(container.path(), &ids));
    let prefixed = ranked_probabilities(&canonical_logits(container.path(), &with_bos));
    let tolerance = PERCENT_DISPLAY_HALF_ULP + PERCENT_SUM_ORDER_SLACK;
    let distinguishable = bare
        .iter()
        .zip(&prefixed)
        .take(TOP_K)
        .any(|((a, pa), (b, pb))| a != b || ((pa - pb) * 100.0).abs() > tolerance);
    assert!(
        distinguishable,
        "a prepended BOS leaves the printed top-{TOP_K} unchanged — the gate is blind"
    );
}

/// The gate: LQL `INFER TOP k` prints the canonical session's top-k ids
/// with their full-vocabulary probabilities.
#[test]
fn lql_infer_top_k_equals_the_canonical_session() {
    let container = granite_container();
    let ids = canonical_prompt_ids(container.path());
    let canonical = ranked_probabilities(&canonical_logits(container.path(), &ids));
    let lql = lql_infer(container.path());
    assert_rows_match(&lql, &canonical[..TOP_K]);
}
