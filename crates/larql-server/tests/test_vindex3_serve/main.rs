//! VI3-SERVE-1 gates: a VINDEX3 container over the normal server API.
//!
//! The authoritative arm (A) is the direct runtime stack —
//! `Vindex3Runtime` → `CanonicalKvState` → `prefill_into` →
//! `session_with_kv` → `continue_session` — assembled by hand in this
//! file. Arm B is an HTTP request through the server's model registry
//! into `/v1/completions`. The gate demands the streamed tokens match
//! arm A token-for-token: same first token, same ordering, same
//! count, same finish behaviour.
//!
//! The negative control pins the architectural regression this rung
//! exists to prevent: the served container **cannot** be opened by
//! the V2 path at all (`load_vindex_config` refuses the generation,
//! `load_single_vindex` errors), and the serving state holds zero V2
//! models while requests succeed — so the server provably did not
//! reconstitute an old-style model behind the scenes.

#[path = "../common/mod.rs"]
mod common;

use std::path::Path;
use std::sync::Arc;

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_server::bootstrap::{
    load_artifact, load_single_vindex, LoadVindexOptions, LoadedArtifact,
};
use larql_server::state::AppState;
use larql_server::vindex3::generate_v3;
use larql_vindex::format::load::load_vindex_config;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

const NEW_TOKENS: usize = 16;
const PROMPT: &str = "[3]";
const COMPONENT: &str = "target";

fn row_continuation(
    plan: &larql_vindex::format::vindex3::opplan::ComponentOpPlan,
) -> larql_inference::vindex3::SelectedContinuation {
    use larql_vindex::format::vindex3::opplan::exec::{
        continuation::plan_continuation_geometry, continuation_authority::ContinuationConfig,
        kv::RowKvState,
    };
    larql_kv::shipped_continuations()
        .select(
            &RowKvState::identity(),
            &ContinuationConfig::empty(),
            &plan_continuation_geometry(plan).unwrap(),
        )
        .unwrap()
}

/// Encode the miniature container and give it a servable tokenizer
/// (`[N]` ↔ id N, no pre-tokenizer).
fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "serve-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

/// Arm A: the direct runtime stack, by hand. Returns per-token
/// `(id, text)` pairs in emission order.
fn direct_arm(container: &Path, max_tokens: usize) -> Vec<(u32, String)> {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    let prompt_ids: Vec<u32> = tokenizer.encode(PROMPT, true).unwrap().get_ids().to_vec();
    assert!(!prompt_ids.is_empty());

    let mut kv = CanonicalKvState::new();
    let prefill = runtime.prefill_into(&prompt_ids, &mut kv).unwrap();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    let mut detok = Detokenizer::new(&tokenizer);
    detok.seed(&prompt_ids);
    let mut pairs = Vec::new();
    continue_session(
        &mut session,
        prefill,
        max_tokens,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| {
            let text = detok.push(id);
            pairs.push((id, text));
        },
    )
    .unwrap();
    pairs
}

/// A serving state holding ONLY the V3 model — bound through the same
/// `load_artifact` the real bootstrap uses.
fn v3_state(container: &Path) -> Arc<AppState> {
    let artifact =
        load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap();
    let v3 = match artifact {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    };
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models: vec![v3],
            aliases: Default::default(),
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: None,
        sessions: larql_server::session::SessionManager::new(3600),
        describe_cache: larql_server::cache::DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(60),
        patch_sources: Default::default(),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            larql_server::response_kv::DEFAULT_MAX_ENTRIES,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

/// Parse an SSE body into its JSON data chunks (excluding `[DONE]`).
fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE chunk is JSON"))
        .collect()
}

/// Ids named by `[N]` tokens in a synthetic-tokenizer surface string, in
/// order — the chat route returns text, and on this fixture the text IS
/// the id sequence.
fn ids_in_surface(text: &str) -> Vec<u32> {
    text.split('[')
        .filter_map(|piece| piece.split(']').next())
        .filter_map(|n| n.parse().ok())
        .collect()
}

/// Encode the fixture and declare `eos_id` as its end-of-turn token in
/// `generation_config.json`, the file the CLI's V3 arm already reads.
fn v3_container_declaring_eos(eos_id: u32) -> tempfile::TempDir {
    let container = v3_container();
    std::fs::write(
        container
            .path()
            .join(larql_vindex::format::filenames::GENERATION_CONFIG_JSON),
        serde_json::json!({ "eos_token_id": eos_id }).to_string(),
    )
    .unwrap();
    container
}

/// An `AppState` with nothing bound: the control for "absent".
fn empty_state() -> Arc<AppState> {
    let state = v3_state(v3_container().path());
    state
        .model_set
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .v3_models
        .clear();
    state
}

mod completions_and_loading;
mod dense_ffn_workers;
mod eos_backends_and_layer_workers;
mod routed_expert_grid;
mod routed_expert_rows;
