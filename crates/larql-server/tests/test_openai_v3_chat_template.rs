//! VINDEX3 chat renders through the container's own chat template.
//!
//! The server's leg of the `larql serve` / `larql run` parity gate: the
//! conversation is rendered by the container's `chat_template.jinja`
//! (never a template guessed from the family or id), encoded without the
//! tokenizer's post-processor, and the ids are the ones
//! `larql_inference::chat::render_conversation` + `encode` give — the
//! calls `larql run` makes. The CLI's leg
//! (`larql-cli/tests/test_vindex3_chat_template_ids.rs`) runs the `larql`
//! binary on the same fixture and pins its `--emit-ids` prompt ids to the
//! same [`SINGLE_TURN_IDS`].

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use larql_inference::chat::{render_conversation, ChatTurn};
use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::routes::openai::chat::ChatMessage;
use larql_server::routes::openai::v3_prompt::encode_v3_conversation;
use larql_server::state::AppState;
use larql_server::vindex3::V3Model;
use larql_vindex::format::filenames::TOKENIZER_JSON;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use tower::ServiceExt;

/// A template whose rendering is a run of fixture tokens: a `[9]`
/// opener, every turn's content, then a `[8]` generation marker. The
/// CLI leg uses the identical text.
const MARKED_TEMPLATE: &str = "[9] {% for m in messages %}{{ m.content }} {% endfor %}[8]";
/// The user turn of the single-turn conversation.
const SINGLE_TURN_USER: &str = "[3] [4]";
/// Its prompt ids under [`MARKED_TEMPLATE`] — pinned in both legs.
const SINGLE_TURN_IDS: [u32; 4] = [9, 3, 4, 8];
/// System + user + assistant + user, and its prompt ids.
const CONVERSATION: [(&str, &str); 4] = [
    ("system", "[5]"),
    ("user", "[3]"),
    ("assistant", "[6]"),
    ("user", "[4]"),
];
const CONVERSATION_IDS: [u32; 6] = [9, 5, 3, 6, 4, 8];
/// Tokens generated per request.
const NEW_TOKENS: usize = 4;
const COMPONENT: &str = "target";

/// A WordLevel tokenizer whose token `[i]` is id `i`, split on
/// whitespace only, so a prompt is its own id list (the CLI fixture's).
fn word_level_tokenizer(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

/// The glimmer fixture with a word-level tokenizer and, optionally, a
/// chat template.
fn container(template: Option<&str>) -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "chat-template-fixture",
    );
    std::fs::write(
        container.path().join(TOKENIZER_JSON),
        word_level_tokenizer(G_VOCAB),
    )
    .unwrap();
    if let Some(t) = template {
        std::fs::write(container.path().join("chat_template.jinja"), t).unwrap();
    }
    container
}

fn bind(container: &Path) -> Arc<V3Model> {
    match load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap() {
        LoadedArtifact::V3(m) => Arc::new(*m),
        LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
    }
}

fn state(model: Arc<V3Model>) -> Arc<AppState> {
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models: vec![model],
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
            0,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

fn wire(pairs: &[(&str, &str)]) -> Vec<serde_json::Value> {
    pairs
        .iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect()
}

fn messages(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
    serde_json::from_value(serde_json::Value::Array(wire(pairs))).unwrap()
}

/// What `larql run` / `larql chat` compute for these turns.
fn cli_ids(model: &V3Model, pairs: &[(&str, &str)]) -> Vec<u32> {
    let turns: Vec<ChatTurn> = pairs
        .iter()
        .map(|(role, content)| ChatTurn {
            role: (*role).to_string(),
            content: (*content).to_string(),
        })
        .collect();
    render_conversation(&model.path, &model.family, &turns)
        .unwrap()
        .encode(&model.tokenizer, model.prompt_bos)
        .unwrap()
}

/// The direct runtime fed `ids`: the text the route must answer with.
fn direct_text(container: &Path, ids: &[u32]) -> String {
    let runtime = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new()).unwrap();
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    let mut kv = CanonicalKvState::new();
    let prefill = runtime.prefill_into(ids, &mut kv).unwrap();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    let mut detok = Detokenizer::new(&tokenizer);
    detok.seed(ids);
    let mut text = String::new();
    continue_session(
        &mut session,
        prefill,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| text.push_str(&detok.push(id)),
    )
    .unwrap();
    text
}

async fn post(
    model: Arc<V3Model>,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let app = larql_server::routes::single_model_router(state(model));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[test]
fn server_and_cli_give_a_single_turn_the_same_prompt_ids() {
    let c = container(Some(MARKED_TEMPLATE));
    let model = bind(c.path());
    let turn = [("user", SINGLE_TURN_USER)];
    let server = encode_v3_conversation(&model, &messages(&turn)).unwrap();
    assert_eq!(server, SINGLE_TURN_IDS);
    assert_eq!(server, cli_ids(&model, &turn));
}

#[test]
fn every_turn_of_a_conversation_reaches_the_template() {
    let c = container(Some(MARKED_TEMPLATE));
    let model = bind(c.path());
    let server = encode_v3_conversation(&model, &messages(&CONVERSATION)).unwrap();
    assert_eq!(server, CONVERSATION_IDS);
    assert_eq!(server, cli_ids(&model, &CONVERSATION));
}

#[tokio::test]
async fn the_chat_route_generates_from_the_templated_ids() {
    let c = container(Some(MARKED_TEMPLATE));
    let expected = direct_text(c.path(), &CONVERSATION_IDS);
    let (status, v) = post(
        bind(c.path()),
        "/v1/chat/completions",
        serde_json::json!({"messages": wire(&CONVERSATION), "max_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["prompt_tokens"], CONVERSATION_IDS.len(), "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], expected.as_str());
}

#[tokio::test]
async fn the_responses_route_uses_the_same_template() {
    let c = container(Some(MARKED_TEMPLATE));
    let (status, v) = post(
        bind(c.path()),
        "/v1/responses",
        serde_json::json!({"input": SINGLE_TURN_USER, "max_output_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["input_tokens"], SINGLE_TURN_IDS.len(), "{v}");
}

#[tokio::test]
async fn without_a_template_a_lone_user_message_is_sent_verbatim() {
    let c = container(None);
    let model = bind(c.path());
    let ids = encode_v3_conversation(&model, &messages(&[("user", SINGLE_TURN_USER)])).unwrap();
    assert_eq!(ids, [3, 4], "the user text, verbatim");
    let (status, v) = post(
        model,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": wire(&[("user", SINGLE_TURN_USER)]),
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn without_a_template_a_conversation_is_refused() {
    let c = container(None);
    let (status, v) = post(
        bind(c.path()),
        "/v1/chat/completions",
        serde_json::json!({"messages": wire(&CONVERSATION), "max_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(message.contains("carries no chat template"), "{message}");
}

#[test]
fn a_template_that_raises_is_a_client_error() {
    let c = container(Some("{{ raise_exception('roles must alternate') }}"));
    let model = bind(c.path());
    let err = encode_v3_conversation(&model, &messages(&[("user", "[1]")])).unwrap_err();
    assert!(
        matches!(err, larql_server::error::ServerError::BadRequest(ref m) if m.contains("roles must alternate")),
        "{err}"
    );
}

#[test]
fn a_conversation_ending_with_the_assistant_is_refused() {
    let c = container(Some(MARKED_TEMPLATE));
    let model = bind(c.path());
    let err = encode_v3_conversation(&model, &messages(&[("user", "[1]"), ("assistant", "[2]")]))
        .unwrap_err();
    assert!(
        matches!(err, larql_server::error::ServerError::BadRequest(_)),
        "{err}"
    );
}

/// The CLI's raw-prompt switch, read by the shared renderer.
const RAW_PROMPT_ENV: &str = "LARQL_RAW_PROMPT";

/// Clears this thread's env overrides on drop, panic or not.
struct EnvOverride;

impl Drop for EnvOverride {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
    }
}

#[test]
fn a_raw_prompt_request_refuses_a_conversation_and_names_why() {
    let c = container(Some(MARKED_TEMPLATE));
    let model = bind(c.path());
    let _restore = EnvOverride;
    larql_compute::options::set_env_override(RAW_PROMPT_ENV, Some("1"));
    let lone = encode_v3_conversation(&model, &messages(&[("user", SINGLE_TURN_USER)])).unwrap();
    assert_eq!(lone, [3, 4], "raw: the user text, verbatim");
    let err = encode_v3_conversation(&model, &messages(&CONVERSATION)).unwrap_err();
    assert!(err.to_string().contains(RAW_PROMPT_ENV), "{err}");
}
