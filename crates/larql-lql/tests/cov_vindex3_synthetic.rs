//! VINDEX3 statement coverage over synthetic containers the other
//! `vindex3_*` suites do not bind: recurrent and latent-cache hybrids,
//! a sink/bias miniature, a compiled pack, and a derived deployment
//! image — plus the refusals a user meets on the wrong binding.
//!
//! Every assertion is on what the user reads: the rendered line, or the
//! refusal message. No real model is downloaded.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::{synthetic_tokenizer_json, write_synthetic_model_dir};
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    bf16_container_with_q8_pack, encode_fixture_container, hybrid_lllf_f32_model,
    miniature_glimmer, miniature_glimmer_with, MiniatureExtras, G_VOCAB,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::represent::policy::{Protections, RolePolicy};
use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

/// The prompt every text statement here runs — one in-vocabulary token
/// of the synthetic tokenizer.
const PROMPT: &str = "[3]";
/// The encoding the derived deployment image is compiled into.
const DEPLOY_ENCODING: &str = "Q8_0";
/// The compiled representation in [`bf16_container_with_q8_pack`].
const COMPILED_REP: &str = "target.decoder_stack@Q8_0";
/// The canonical representation it was compiled from.
const CANONICAL_REP: &str = "target.decoder_stack@BF16";
/// The weight tensor the UPDATE WEIGHTS cases edit (`[64, 256]`).
const EDIT_TENSOR: &str = "1.mlp.down_proj.weight";
const EDIT_ROW: usize = 5;
const EDIT_DELTA: &str = "0.0625";
/// How many elements a staging reply lists before it summarises — the
/// executor's `SHOWN_ELEMENTS`.
const SHOWN_ELEMENTS: usize = 8;
/// Columns beyond the listing limit, so the summary line must appear.
const EXTRA_ELEMENTS: usize = 3;
/// `hybrid_lllf_f32_model`: three Gated DeltaNet layers, one softmax.
const LLLF_RECURRENT_LAYERS: usize = 3;
/// `hybrid_kda_mla_f32_model`: three KDA layers, one MLA layer.
const KKKM_KDA_LAYERS: usize = 3;
const KKKM_MLA_LAYERS: usize = 1;

// ── plumbing ────────────────────────────────────────────────────────

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session.execute(&parsed).map_err(|e| e.to_string())
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    try_run(session, stmt).unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn bound(container: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(container)));
    session
}

fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l.contains(needle))
}

/// Encode `checkpoint` into a container, optionally with a tokenizer.
fn encode(
    write_checkpoint: impl FnOnce(&Path),
    name: &str,
    tokenizer_vocab: Option<usize>,
) -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write_checkpoint, checkpoint.path(), container.path(), name);
    if let Some(vocab) = tokenizer_vocab {
        std::fs::write(
            container.path().join("tokenizer.json"),
            synthetic_tokenizer_json(vocab),
        )
        .unwrap();
    }
    container
}

/// The BF16 + compiled-Q8_0 container weight edits run against.
struct Bf16Fixture {
    dir: tempfile::TempDir,
    container: PathBuf,
}

fn bf16_fixture() -> Bf16Fixture {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    Bf16Fixture { dir, container }
}

fn update_weights(cols: &[usize]) -> String {
    let list: Vec<String> = cols.iter().map(usize::to_string).collect();
    format!(
        "UPDATE WEIGHTS SET DELTA = {EDIT_DELTA} WHERE TENSOR = \"{EDIT_TENSOR}\" \
         AND ROW = {EDIT_ROW} AND COL IN ({});",
        list.join(", ")
    )
}

// ── vindex3.rs: STATS / SHOW LAYERS on non-softmax stacks ───────────

#[test]
fn stats_and_layers_name_the_recurrent_layers_of_a_hybrid() {
    let container = encode(hybrid_lllf_f32_model, "lllf", None);
    let mut session = bound(container.path());

    let stats = run(&mut session, "STATS;");
    assert!(
        has(&stats, &format!("{LLLF_RECURRENT_LAYERS} recurrent")),
        "the attention line counts the recurrent layers: {stats:?}"
    );
    assert!(
        has(
            &stats,
            &format!("recurrent state on {LLLF_RECURRENT_LAYERS} layer(s)")
        ) && has(&stats, "constant in sequence length"),
        "the continuation line states the recurrence's constant state: {stats:?}"
    );

    let layers = run(&mut session, "SHOW LAYERS;");
    let gated_delta = layers.iter().filter(|l| l.contains("gated-delta")).count();
    assert_eq!(gated_delta, LLLF_RECURRENT_LAYERS, "{layers:?}");
}

#[test]
fn stats_and_layers_name_the_latent_cache_of_an_mla_hybrid() {
    let container = encode(hybrid_kda_mla_f32_model, "kkkm", None);
    let mut session = bound(container.path());

    let stats = run(&mut session, "STATS;");
    assert!(
        has(&stats, &format!("{KKKM_MLA_LAYERS} latent-cache")),
        "the attention line counts the latent-cache layers: {stats:?}"
    );
    assert!(
        has(
            &stats,
            &format!("latent cache on {KKKM_MLA_LAYERS} layer(s)")
        ) && has(&stats, "growing with the prefix"),
        "an MLA cache grows with the prefix and must say so: {stats:?}"
    );

    let layers = run(&mut session, "SHOW LAYERS;");
    assert_eq!(
        layers.iter().filter(|l| l.contains(" kda ")).count(),
        KKKM_KDA_LAYERS,
        "{layers:?}"
    );
    assert_eq!(
        layers.iter().filter(|l| l.contains(" mla ")).count(),
        KKKM_MLA_LAYERS,
        "{layers:?}"
    );
}

/// INFER after a KNN insert runs the observed pass with residual capture
/// on — the taps the stored keys are matched against.
#[test]
fn infer_after_a_knn_insert_captures_residuals_and_still_predicts() {
    let container = encode(miniature_glimmer, "knn-infer", Some(G_VOCAB));
    let mut session = bound(container.path());
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
    );
    let out = run(&mut session, &format!("INFER \"{PROMPT}\" TOP 3;"));
    assert_eq!(
        out.first().map(String::as_str),
        Some("Predictions (VINDEX3 program):"),
        "{out:?}"
    );
    assert!(
        out.iter().any(|l| l.trim_start().starts_with("1.")),
        "{out:?}"
    );
}

// ── session.rs: EXPLAIN / TRACE ─────────────────────────────────────

#[test]
fn explain_states_a_recurrences_constant_state() {
    let container = encode(hybrid_lllf_f32_model, "lllf-explain", None);
    let mut session = bound(container.path());
    let out = run(&mut session, &format!("EXPLAIN INFER \"{PROMPT}\";"));
    let state_lines = out
        .iter()
        .filter(|l| l.trim_start().starts_with("state") && l.ends_with("elements/layer"))
        .count();
    assert_eq!(state_lines, LLLF_RECURRENT_LAYERS, "{out:?}");
}

#[test]
fn explain_names_projection_bias_as_an_attention_extra() {
    // Bias only: Glimmer's judged family carries no sink semantics, so an
    // encode with sink operands refuses at closure.
    let extras = MiniatureExtras {
        attention_bias: true,
        ..MiniatureExtras::default()
    };
    let container = encode(
        |dir| miniature_glimmer_with(dir, extras),
        "bias",
        Some(G_VOCAB),
    );
    let mut session = bound(container.path());
    let out = run(&mut session, &format!("EXPLAIN INFER \"{PROMPT}\";"));
    assert!(
        out.iter().any(|l| l.trim_start().starts_with("extras")
            && l.contains("qkvo_bias")
            && !l.contains("sinks")),
        "{out:?}"
    );
}

#[test]
fn trace_on_a_tokenizerless_container_names_the_missing_tokenizer() {
    let container = encode(miniature_glimmer, "ids-only-trace", None);
    let mut session = bound(container.path());
    let err = try_run(&mut session, &format!("TRACE \"{PROMPT}\";")).unwrap_err();
    assert!(
        err.contains("TRACE needs a tokenizer — this container carries none"),
        "{err}"
    );
}

// ── session.rs: directory statements ────────────────────────────────

const DIRECTORY_STATEMENTS: [&str; 4] = [
    "SHOW COMPONENTS;",
    "SHOW REPRESENTATIONS;",
    "SHOW PROVENANCE;",
    "SHOW AUTHORITY;",
];

#[test]
fn directory_statements_without_a_binding_ask_for_one() {
    let mut session = Session::new();
    for stmt in DIRECTORY_STATEMENTS {
        let err = try_run(&mut session, stmt).unwrap_err();
        assert!(err.contains("No backend loaded"), "{stmt}: {err}");
    }
}

#[test]
fn directory_statements_on_a_vindex2_binding_refuse_by_concept() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_model_dir(dir.path()).unwrap();
    let mut session = bound(dir.path());
    for stmt in DIRECTORY_STATEMENTS {
        let err = try_run(&mut session, stmt).unwrap_err();
        assert!(
            err.contains("This binding is not a VINDEX3 container"),
            "{stmt}: {err}"
        );
    }
}

#[test]
fn show_components_counts_recurrent_layers() {
    let container = encode(hybrid_lllf_f32_model, "lllf-components", None);
    let mut session = bound(container.path());
    let out = run(&mut session, "SHOW COMPONENTS;");
    assert!(
        has(&out, &format!("{LLLF_RECURRENT_LAYERS} recurrent")),
        "{out:?}"
    );
}

#[test]
fn compiled_representations_are_marked_and_filterable() {
    let fx = bf16_fixture();
    let mut session = bound(&fx.container);

    let all = run(&mut session, "SHOW REPRESENTATIONS;");
    let compiled = all
        .iter()
        .find(|l| l.starts_with(COMPILED_REP))
        .unwrap_or_else(|| panic!("{COMPILED_REP} listed: {all:?}"));
    assert!(compiled.ends_with("(compiled)"), "{compiled}");

    // The filter keeps the compiled pack and drops the canonical source.
    let filtered = run(
        &mut session,
        &format!("SHOW REPRESENTATIONS \"{COMPILED_REP}\";"),
    );
    assert!(has(&filtered, COMPILED_REP), "{filtered:?}");
    assert!(!has(&filtered, CANONICAL_REP), "{filtered:?}");
}

#[test]
fn provenance_names_what_a_compiled_pack_derives_from() {
    let fx = bf16_fixture();
    let mut session = bound(&fx.container);

    let prov = run(
        &mut session,
        &format!("SHOW PROVENANCE \"{COMPILED_REP}\";"),
    );
    assert_eq!(
        prov.first().map(String::as_str),
        Some(COMPILED_REP),
        "{prov:?}"
    );
    assert!(
        has(&prov, &format!("compiled_from:   {CANONICAL_REP}")),
        "{prov:?}"
    );
    assert!(has(&prov, "source_digest:"), "{prov:?}");
    assert!(
        !prov.iter().any(|l| l == CANONICAL_REP),
        "the filter excludes the source entry: {prov:?}"
    );

    let none = run(&mut session, r#"SHOW PROVENANCE "no-such-object";"#);
    assert!(
        has(&none, "(no entries match \"no-such-object\")"),
        "{none:?}"
    );
}

#[test]
fn a_deployment_image_declares_itself_derived() {
    let fx = bf16_fixture();
    let canonical = fx.dir.path().join("canonical.vindex3");
    let image = fx.dir.path().join("deploy.vindex3");
    let spec = RepresentSpec {
        encoding: DEPLOY_ENCODING.to_string(),
        objects: Vec::new(),
        roles: RolePolicy::default(),
        deployment: true,
        protect: Protections::default(),
    };
    compile_representation(&canonical, &image, &spec).unwrap();
    let mut session = bound(&image);

    let auth = run(&mut session, "SHOW AUTHORITY;");
    assert!(
        has(
            &auth,
            "Authority:   derived  (executable; not re-compilable)"
        ),
        "{auth:?}"
    );
    assert!(
        has(&auth, "Source:"),
        "a derived image names its source: {auth:?}"
    );

    let prov = run(&mut session, "SHOW PROVENANCE;");
    assert!(
        prov.first()
            .is_some_and(|l| l.starts_with("Container derives from: ")),
        "{prov:?}"
    );
}

#[test]
fn an_empty_directory_says_so_instead_of_printing_nothing() {
    let container = encode(miniature_glimmer, "empty-dir", None);
    let mut session = bound(container.path());
    // The directory statements read the index fresh from disk, so a
    // container whose directory has been emptied answers from that.
    let index_path = container.path().join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    index.as_object_mut().unwrap().remove("representations");
    std::fs::write(&index_path, index.to_string()).unwrap();

    for stmt in ["SHOW REPRESENTATIONS;", "SHOW PROVENANCE;"] {
        let out = run(&mut session, stmt);
        assert!(has(&out, "(the directory is empty)"), "{stmt}: {out:?}");
    }
    // The graph still declares objects the directory no longer holds:
    // SHOW COMPONENTS lists them and warns that the container is not
    // coherent rather than presenting it as sound.
    let components = run(&mut session, "SHOW COMPONENTS;");
    assert!(
        components
            .last()
            .is_some_and(|l| l.starts_with("warning:") && l.contains("coherence defect(s)")),
        "{components:?}"
    );
}

// ── weights.rs: UPDATE WEIGHTS / COMPILE refusals and replies ──────

#[test]
fn update_weights_without_a_binding_asks_for_one() {
    let mut session = Session::new();
    let err = try_run(&mut session, &update_weights(&[0])).unwrap_err();
    assert!(err.contains("No backend loaded"), "{err}");
}

#[test]
fn update_weights_on_a_vindex2_binding_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_model_dir(dir.path()).unwrap();
    let mut session = bound(dir.path());
    let err = try_run(&mut session, &update_weights(&[0])).unwrap_err();
    assert!(
        err.contains("this session is bound to a VINDEX2 vindex"),
        "{err}"
    );
}

#[test]
fn a_long_edit_lists_the_first_elements_and_summarises_the_rest() {
    let fx = bf16_fixture();
    let mut session = bound(&fx.container);
    let cols: Vec<usize> = (0..SHOWN_ELEMENTS + EXTRA_ELEMENTS).collect();
    let out = run(&mut session, &update_weights(&cols));
    let listed = out
        .iter()
        .filter(|l| l.trim_start().starts_with(&format!("[{EDIT_ROW}, ")))
        .count();
    assert_eq!(listed, SHOWN_ELEMENTS, "{out:?}");
    assert!(
        has(&out, &format!("… and {EXTRA_ELEMENTS} more")),
        "{out:?}"
    );
}

#[test]
fn weight_edits_join_the_anonymous_patch_and_refuse_to_compile_with_knowledge() {
    let fx = bf16_fixture();
    std::fs::write(
        fx.container.join("tokenizer.json"),
        synthetic_tokenizer_json(larql_vindex::format::vindex3::fixtures::DENSE_VOCAB),
    )
    .unwrap();
    let mut session = bound(&fx.container);
    // A knowledge edit opens the anonymous patch session.
    run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    let staged = run(&mut session, &update_weights(&[EDIT_ROW]));
    assert!(
        has(&staged, "recorded into the anonymous patch"),
        "{staged:?}"
    );

    let out = fx.dir.path().join("mixed.vindex3");
    let err = try_run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    )
    .unwrap_err();
    assert!(
        err.contains("COMPILE cannot combine stored-weight edits with knowledge edits"),
        "{err}"
    );
    assert!(!out.exists(), "a refused COMPILE writes nothing");
}
