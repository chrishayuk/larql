//! VI3-INF-0 and VI3-INF-2 gates.
//!
//! Two layers of evidence, deliberately separate:
//!
//! - **Driver semantics** against a scripted [`LogitsSession`] double —
//!   sampling, EOS, callback order, budget handling — with no container
//!   in sight, so a failure names the driver.
//! - **Seam parity** against the direct [`DecodeSession`] harness on a
//!   real encoded container: the runtime path must reproduce the
//!   existing V3 decode harness **bit-for-bit** (logits) and id-for-id
//!   (greedy tokens). The harness side opens its own plan and store —
//!   the two arms share only the container bytes and the backend
//!   arithmetic.
//!
//! A control precedes the parity claim (the instrument must fail on
//! known-different input): a diverged prompt stream must produce
//! diverged logits, or bit-equality above proves nothing.

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, G_TOKENS,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;

use super::{
    continue_session_masked, generate_session, plan_kv_geometry, KvState, LogitsSession,
    RecordingObserver, RowKvState, StepEvent, Vindex3Runtime, Vindex3Session,
};
use crate::error::InferenceError;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::sampling::SamplingConfig;

/// The rung's target: sixteen greedy tokens through the runtime.
const NEW_TOKENS: usize = 16;
/// Component id the miniature systems encode their text stack under.
const COMPONENT: &str = "target";

// ── Driver semantics against a scripted session ──

/// Scripted [`LogitsSession`]: returns pre-baked logit rows in order,
/// so driver tests control exactly what the sampler sees.
struct ScriptedSession {
    rows: Vec<Vec<f32>>,
    cursor: usize,
    position: usize,
}

impl ScriptedSession {
    fn new(rows: Vec<Vec<f32>>) -> Self {
        Self {
            rows,
            cursor: 0,
            position: 0,
        }
    }

    fn next_row(&mut self) -> Result<Vec<f32>, InferenceError> {
        let row =
            self.rows.get(self.cursor).cloned().ok_or_else(|| {
                InferenceError::Parse("scripted session ran out of rows".to_string())
            })?;
        self.cursor += 1;
        Ok(row)
    }
}

impl LogitsSession for ScriptedSession {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Parse("empty prompt".to_string()));
        }
        self.position += tokens.len();
        self.next_row()
    }

    fn step(&mut self, _token: u32) -> Result<Vec<f32>, InferenceError> {
        self.position += 1;
        self.next_row()
    }

    fn position(&self) -> usize {
        self.position
    }
}

/// A logit row whose argmax is `id` (out of 4 vocabulary entries).
fn row_peaking_at(id: usize) -> Vec<f32> {
    let mut row = vec![0.0f32; 4];
    row[id] = 1.0;
    row
}

#[test]
fn the_driver_emits_greedy_ids_in_callback_order() {
    let mut session = ScriptedSession::new(vec![
        row_peaking_at(2),
        row_peaking_at(0),
        row_peaking_at(3),
    ]);
    let mut streamed = Vec::new();
    let result = generate_session(
        &mut session,
        &[7, 8],
        3,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id| streamed.push(id),
    )
    .unwrap();
    assert_eq!(result.tokens, vec![2, 0, 3]);
    assert_eq!(streamed, result.tokens);
    assert_eq!(result.prompt_len, 2);
    // Prompt (2) + steps for all but the last emitted token (2).
    assert_eq!(session.position(), 4);
}

#[test]
fn a_stop_token_ends_generation_before_being_emitted() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(1), row_peaking_at(3)]);
    let result = generate_session(
        &mut session,
        &[7],
        4,
        SamplingConfig::greedy(),
        &EosConfig::empty().with_eos_id(3),
        |_| {},
    )
    .unwrap();
    // Token 1 emitted; the stop token 3 ended the run without appearing.
    assert_eq!(result.tokens, vec![1]);
}

// ── the masked driver (N0.6) ────────────────────────────────────────

#[test]
fn the_masked_driver_obeys_the_mask_over_greedy_preference() {
    // Every scripted row peaks at id 2, but the mask only ever admits
    // id 1 — the mask must win, and it must see the generated-so-far
    // history grow.
    let mut session = ScriptedSession::new(vec![row_peaking_at(2), row_peaking_at(2)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut histories = Vec::new();
    let mut mask = |generated: &[u32], logits: &mut Vec<f32>| {
        histories.push(generated.to_vec());
        for (id, logit) in logits.iter_mut().enumerate() {
            if id != 1 {
                *logit = f32::NEG_INFINITY;
            }
        }
    };
    let result = continue_session_masked(
        &mut session,
        logits,
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, vec![1, 1]);
    assert_eq!(histories, vec![vec![], vec![1]]);
}

#[test]
fn mask_exhaustion_before_first_emission_is_an_error() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(0)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut mask =
        |_: &[u32], logits: &mut Vec<f32>| logits.iter_mut().for_each(|l| *l = f32::NEG_INFINITY);
    let err = continue_session_masked(
        &mut session,
        logits,
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("sampler produced no token"),
        "{err}"
    );
}

#[test]
fn mask_exhaustion_after_emission_is_a_natural_stop() {
    // The grammar completing (nothing admissible any more) mirrors the
    // V2 constrained driver: a clean stop, not an error.
    let mut session = ScriptedSession::new(vec![row_peaking_at(0), row_peaking_at(0)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut calls = 0usize;
    let mut mask = |_: &[u32], logits: &mut Vec<f32>| {
        calls += 1;
        if calls > 1 {
            logits.iter_mut().for_each(|l| *l = f32::NEG_INFINITY);
        }
    };
    let result = continue_session_masked(
        &mut session,
        logits,
        4,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, vec![0], "one emission, then a clean stop");
}

#[test]
fn a_zero_token_budget_prefills_but_emits_nothing() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(1)]);
    let mut fired = false;
    let result = generate_session(
        &mut session,
        &[7, 8, 9],
        0,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| fired = true,
    )
    .unwrap();
    assert!(result.tokens.is_empty());
    assert!(!fired);
    // The prompt was still consumed — a follow-up call can continue.
    assert_eq!(session.position(), 3);
}

#[test]
fn non_finite_logits_surface_as_an_error_not_a_token() {
    let mut session = ScriptedSession::new(vec![vec![f32::NAN, f32::NAN]]);
    let err = generate_session(
        &mut session,
        &[7],
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("sampler produced no token"),
        "{err}"
    );
}

// ── Seam parity against the direct DecodeSession harness ──

/// Encode `write_checkpoint`'s model into a fresh container.
fn container_with(write_checkpoint: impl FnOnce(&Path)) -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        write_checkpoint,
        checkpoint.path(),
        container.path(),
        "seam-fixture",
    );
    container
}

/// Ties keep the first index — the same rule the V3 exec harness and
/// the greedy sampler use.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0 as u32
}

/// The existing V3 decode harness, verbatim in shape: a direct
/// [`DecodeSession`] fed the prompt tokenwise, then greedy argmax.
/// Returns the emitted ids and every logits row the greedy loop saw
/// (prompt-final first).
fn harness_decode<B: PlanBackend>(
    container: &Path,
    backend: &B,
    prompt: &[u32],
    new_tokens: usize,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, COMPONENT).unwrap();
    assert!(outcome.closed(), "harness fixture must close");
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    let mut session = DecodeSession::new(&plan, &store, backend).unwrap();

    let mut logits = None;
    for &token in prompt {
        logits = session.step(token).unwrap().logits;
    }
    let mut logits = logits.expect("plan carries an output head");
    let mut rows = vec![logits.clone()];
    let mut ids = Vec::new();
    while ids.len() < new_tokens {
        let next = argmax(&logits);
        ids.push(next);
        if ids.len() == new_tokens {
            break;
        }
        logits = session.step(next).unwrap().logits.unwrap();
        rows.push(logits.clone());
    }
    (ids, rows)
}

/// The parity gate per backend: prefill+step logits bit-for-bit, then
/// sixteen greedy ids through `generate_session` id-for-id.
fn assert_seam_parity<B: PlanBackend>(backend_for_harness: &B, backend_for_runtime: B) {
    let container = container_with(miniature_glimmer);
    let (harness_ids, harness_rows) =
        harness_decode(container.path(), backend_for_harness, &G_TOKENS, NEW_TOKENS);
    assert_eq!(harness_ids.len(), NEW_TOKENS);

    let runtime = Vindex3Runtime::open(container.path(), COMPONENT, backend_for_runtime).unwrap();

    // Logits stream, bit-for-bit: prefill equals the harness's
    // prompt-final row; each subsequent step equals the harness's row
    // for the same emitted id.
    let mut session = runtime.session().unwrap();
    let prefill = session.prefill(&G_TOKENS).unwrap();
    assert_eq!(prefill, harness_rows[0], "prefill logits diverge");
    assert_eq!(session.position(), G_TOKENS.len());
    for (row, &id) in harness_rows[1..].iter().zip(&harness_ids) {
        let stepped = session.step(id).unwrap();
        assert_eq!(&stepped, row, "step logits diverge after id {id}");
    }

    // Sixteen greedy tokens through the generation driver, on a fresh
    // session from the same runtime.
    let mut streamed = Vec::new();
    let mut session = runtime.session().unwrap();
    let result = generate_session(
        &mut session,
        &G_TOKENS,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id| streamed.push(id),
    )
    .unwrap();
    assert_eq!(result.tokens, harness_ids, "greedy ids diverge");
    assert_eq!(streamed, harness_ids);
    assert_eq!(result.prompt_len, G_TOKENS.len());
    assert_eq!(session.position(), G_TOKENS.len() + NEW_TOKENS - 1);
}

#[test]
fn reference_runtime_matches_the_decode_harness_bit_for_bit() {
    assert_seam_parity(&ReferenceBackend::new(), ReferenceBackend::new());
}

#[test]
fn production_runtime_matches_the_decode_harness_bit_for_bit() {
    assert_seam_parity(&ProductionBackend::new(), ProductionBackend::new());
}

/// Control: the instrument must fail on known-different input. A
/// reversed prompt is a different program input, so its prefill logits
/// must differ from the harness's — otherwise the bit-equality above
/// is vacuous.
#[test]
fn the_parity_instrument_detects_a_diverged_prompt() {
    let container = container_with(miniature_glimmer);
    let backend = ReferenceBackend::new();
    let (_, harness_rows) = harness_decode(container.path(), &backend, &G_TOKENS, 1);

    let runtime = Vindex3Runtime::open(container.path(), COMPONENT, backend).unwrap();
    let mut session = runtime.session().unwrap();
    let mut reversed = G_TOKENS.to_vec();
    reversed.reverse();
    assert_ne!(reversed, G_TOKENS.to_vec());
    let prefill = session.prefill(&reversed).unwrap();
    assert_ne!(
        prefill, harness_rows[0],
        "instrument cannot distinguish different prompts"
    );
}

/// The dense Llama-shaped anatomy through the same seam — a second,
/// independent plan shape (two norms, RoPE everywhere, no gates).
#[test]
fn dense_runtime_matches_the_decode_harness_bit_for_bit() {
    let container = container_with(dense_f32_model);
    let backend = ReferenceBackend::new();
    let prompt = [5u32, 99, 42];
    let (harness_ids, _) = harness_decode(container.path(), &backend, &prompt, NEW_TOKENS);

    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    assert_eq!(runtime.plan().component, COMPONENT);
    assert!(!runtime.backend().name().is_empty());
    let mut session = runtime.session().unwrap();
    let result = generate_session(
        &mut session,
        &prompt,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, harness_ids);
}

// ── Continuation state behind the KvState seam (VI3-INF-2) ──

/// Caller-owned continuation state must change nothing about the
/// generated ids, and must still hold every row after the session is
/// gone — with per-layer geometry matching the plan's own statement
/// ([`plan_kv_geometry`]), which is the whole point of the seam: KV
/// policy reads the program, not a family registry.
#[test]
fn a_caller_owned_kv_state_generates_identically_and_outlives_the_session() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut default_session = runtime.session().unwrap();
    let baseline = generate_session(
        &mut default_session,
        &G_TOKENS,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap();

    let mut kv = RowKvState::default();
    {
        let mut session = runtime.session_with_kv(&mut kv).unwrap();
        let provided = generate_session(
            &mut session,
            &G_TOKENS,
            NEW_TOKENS,
            SamplingConfig::greedy(),
            &EosConfig::empty(),
            |_| {},
        )
        .unwrap();
        assert_eq!(provided, baseline, "caller-owned KV diverged the ids");
    }

    // The session is gone; the caller still holds the conversation's
    // continuation state, one row pair per consumed position, at the
    // width the plan declares per layer.
    let geometry = plan_kv_geometry(runtime.plan());
    let consumed = G_TOKENS.len() + NEW_TOKENS - 1;
    for (layer, geo) in geometry.iter().enumerate() {
        assert_eq!(kv.keys(layer).len(), consumed);
        assert_eq!(kv.values(layer).len(), consumed);
        assert!(kv.keys(layer).iter().all(|row| row.len() == geo.kv_dim));
    }
    // …and the miniature's sliding+full split is explicit in that
    // geometry — no ModelArchitecture consulted anywhere in this test.
    assert_eq!(geometry[0].window, Some(3));
    assert_eq!(geometry[1].window, None);
}

/// VI3-INF-3 through the runtime, per backend: batch prefill into the
/// caller's provider, sample the first token from the prefill logits,
/// resume decode over the SAME provider — and the emitted ids must
/// match the tokenwise harness id-for-id, with the prefill logits
/// bit-identical to the harness's prompt-final row. The provider owns
/// the logical position throughout; nothing passes a start position.
fn assert_prefill_resume_matches_harness<B: PlanBackend>(
    backend_for_harness: &B,
    backend_for_runtime: B,
) {
    let container = container_with(miniature_glimmer);
    let (harness_ids, harness_rows) =
        harness_decode(container.path(), backend_for_harness, &G_TOKENS, NEW_TOKENS);

    let runtime = Vindex3Runtime::open(container.path(), COMPONENT, backend_for_runtime).unwrap();
    let mut kv = RowKvState::default();
    let prefill_logits = runtime.prefill_into(&G_TOKENS, &mut kv).unwrap();
    assert_eq!(prefill_logits, harness_rows[0], "prefill logits diverge");
    assert_eq!(kv.position(), G_TOKENS.len());

    // The exact SERVE-1 stack: prefill_into → session_with_kv →
    // continue_session, streaming callback and all.
    let mut streamed = Vec::new();
    let mut session = runtime.session_with_kv(&mut kv).unwrap();
    assert_eq!(session.position(), G_TOKENS.len(), "resume position");
    let result = super::continue_session(
        &mut session,
        prefill_logits,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id| streamed.push(id),
    )
    .unwrap();
    assert_eq!(result.tokens, harness_ids, "prefill+resume ids diverge");
    assert_eq!(streamed, harness_ids);
    assert_eq!(result.prompt_len, G_TOKENS.len());
    drop(session);
    assert_eq!(kv.position(), G_TOKENS.len() + NEW_TOKENS - 1);
}

#[test]
fn reference_batch_prefill_resumes_to_the_harness_ids() {
    assert_prefill_resume_matches_harness(&ReferenceBackend::new(), ReferenceBackend::new());
}

#[test]
fn production_batch_prefill_resumes_to_the_harness_ids() {
    assert_prefill_resume_matches_harness(&ProductionBackend::new(), ProductionBackend::new());
}

// ── Refusals ──

#[test]
fn an_empty_prompt_is_refused() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let mut session = runtime.session().unwrap();
    let err = session.prefill(&[]).unwrap_err();
    assert!(
        err.to_string().contains("at least one prompt token"),
        "{err}"
    );
}

#[test]
fn an_unknown_component_refuses_to_open() {
    let container = container_with(miniature_glimmer);
    let err = match Vindex3Runtime::open(
        container.path(),
        "no-such-component",
        ReferenceBackend::new(),
    ) {
        Ok(_) => panic!("an unknown component must refuse to open"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no-such-component"), "{err}");
}

/// An unclosed component must refuse to open with the defects in the
/// error — never "best-effort" execute. Provoked the same way the
/// vindex closure gates do it: strip the component's execution surface
/// from the container's own graph, which planning reports as a
/// `MissingSurface` defect.
#[test]
fn an_unclosed_component_refuses_to_open_naming_its_defects() {
    let container = container_with(miniature_glimmer);
    let graph_path = container
        .path()
        .join(larql_vindex::format::vindex3::encode::SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    for component in graph["components"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("execution");
    }
    std::fs::write(&graph_path, graph.to_string()).unwrap();

    let err = match Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()) {
        Ok(_) => panic!("an unclosed component must refuse to open"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("does not close"), "{err}");
}

/// The step path's defensive refusal (a step yielding no logits) is
/// unreachable while construction refuses headless plans — pinned here
/// so the fail-closed message stays a work item, not a mystery, if a
/// future change ever reaches it.
#[test]
fn the_defensive_missing_logits_refusal_names_the_invariant() {
    let err = super::session::missing_logits_error();
    assert!(err.to_string().contains("output head"), "{err}");
}

/// Same discipline for the prefill path's defensive refusal.
#[test]
fn the_defensive_headless_prefill_refusal_names_the_invariant() {
    let err = super::runtime::headless_prefill_error();
    assert!(err.to_string().contains("no output head"), "{err}");
}

#[test]
fn a_plan_without_an_output_head_is_refused() {
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let mut plan = outcome.plan.unwrap();
    plan.output = None;
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let err = match Vindex3Session::new(&plan, &store, &backend) {
        Ok(_) => panic!("a headless plan must be refused"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no output head"), "{err}");
}

// ── EXPLAIN (LQL-2): the structured explanation IS the authority ──

#[test]
fn explain_is_stable_and_reads_the_plan() {
    let container = container_with(miniature_glimmer);
    let a = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let b = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let explain = super::ExplainPlan::from_runtime(&a);
    // Stability: same container, identical structured explanation.
    assert_eq!(explain, super::ExplainPlan::from_runtime(&b));

    assert_eq!(explain.generation, 3);
    assert_eq!(explain.component, COMPONENT);
    assert!(explain.execution_closed);
    assert_eq!(explain.layers.len(), 2);
    // The miniature's sliding(3)/full split, from the plan alone.
    assert_eq!(explain.layers[0].attention.mode, "sliding");
    assert_eq!(explain.layers[0].attention.window, Some(3));
    assert_eq!(explain.layers[1].attention.mode, "full");
    // Four-norm placement shows as explicit ops, in execution order.
    assert_eq!(
        explain.layers[0].ops,
        vec![
            "pre_attention_norm",
            "attention",
            "post_attention_norm",
            "residual_add",
            "pre_ffn_norm",
            "ffn",
            "post_ffn_norm",
            "residual_add",
        ]
    );
    // Continuation geometry matches what a provider is prepared with.
    let geometry = plan_kv_geometry(a.plan());
    assert_eq!(explain.continuation.len(), geometry.len());
    for (e, g) in explain.continuation.iter().zip(&geometry) {
        assert_eq!(e.kv_dim, g.kv_dim);
        assert_eq!(e.window, g.window);
    }
    assert!(explain.output.is_some());
    assert!(explain.final_norm);
}

/// The negative control: mutate the plan and the explanation must
/// change with it — the instrument sees the program, not a cached
/// family notion.
#[test]
fn explain_changes_when_the_plan_changes() {
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();
    let baseline = super::ExplainPlan::from_plan(&plan, "m");

    let mut widened = plan.clone();
    widened.layers[0].attention.softmax_mut().unwrap().window = None;
    let explained = super::ExplainPlan::from_plan(&widened, "m");
    assert_ne!(baseline, explained);
    assert_eq!(explained.layers[0].attention.mode, "full");

    let mut headless = plan.clone();
    headless.output = None;
    assert!(super::ExplainPlan::from_plan(&headless, "m")
        .output
        .is_none());
}

/// Provenance closure: the coordinates the explanation quotes are
/// sufficient, alone, to reach the exact bytes execution loads — the
/// explain chain and the execution chain cannot name different
/// operands.
#[test]
fn explain_operand_coordinates_resolve_to_the_executed_bytes() {
    use larql_vindex::format::vindex3::opplan::OperandRef;
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let explain = super::ExplainPlan::from_plan(&plan, "m");

    // One attention operand and one FFN operand, per the gate.
    let q = &explain.layers[0].attention.operands[0];
    assert_eq!(q.role, "q");
    let ffn_up = explain.layers[0]
        .ffn
        .operands
        .iter()
        .find(|o| o.role == "up")
        .unwrap();
    for (quoted, executed) in [
        (q, &plan.layers[0].attention.softmax().unwrap().q),
        (
            ffn_up,
            &plan.layers[0].ffn.as_ref().unwrap().dense().unwrap().up,
        ),
    ] {
        let rebuilt = OperandRef {
            object: quoted.object.clone(),
            tensor: quoted.tensor.clone(),
            dtype: quoted.dtype.clone(),
            shape: quoted.shape.clone(),
        };
        let via_explain = store.load(&rebuilt).unwrap();
        let via_execution = store.load(executed).unwrap();
        assert_eq!(via_explain, via_execution, "{} bytes diverge", quoted.role);
        assert!(!via_explain.is_empty());
    }
}

/// The session-level observation seam: observed and unobserved steps
/// are bit-identical, and the recorder sees the step's boundaries —
/// one execution path, many consumers.
#[test]
fn an_observed_session_step_is_bit_identical_and_records_boundaries() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut plain = runtime.session().unwrap();
    let mut observed = runtime.session().unwrap();
    let mut recorder = RecordingObserver::default();
    for &token in G_TOKENS.iter() {
        let a = plain.step(token).unwrap();
        let b = observed.step_observed(token, &mut recorder).unwrap();
        assert_eq!(a, b, "observation changed the arithmetic");
    }
    // Per position: embed + (attention, ffn) per layer + logits.
    let per_position = 1 + 2 * runtime.plan().layers.len() + 1;
    assert_eq!(recorder.events.len(), G_TOKENS.len() * per_position);
    assert!(matches!(
        recorder.events[0],
        StepEvent::Embedded { position: 0 }
    ));
}

/// The analysis pass (V3-LQL-3B): `execute_streaming` runs the SAME
/// traversal prefill does — bit-identical logits — while streaming
/// each layer's residual taps to the sink. Observation is
/// subscription, never a second executor.
#[test]
fn execute_streaming_matches_prefill_and_taps_every_layer() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut kv = RowKvState::default();
    let prefill_logits = runtime.prefill_into(&G_TOKENS, &mut kv).unwrap();

    let mut tapped: Vec<(usize, usize)> = Vec::new();
    let output = runtime
        .execute_streaming(&G_TOKENS, &mut |event| {
            if let super::PlaneEvent::Layer { index, trace } = event {
                tapped.push((index, trace.post_layer.len()));
            }
            Ok(())
        })
        .unwrap();

    assert_eq!(
        output.logits.as_deref(),
        Some(&prefill_logits[..]),
        "the observed pass must price the same logits bit-for-bit"
    );
    let layers = runtime.plan().layers.len();
    let expected: Vec<(usize, usize)> = (0..layers).map(|l| (l, G_TOKENS.len())).collect();
    assert_eq!(
        tapped, expected,
        "one tap per layer, one residual row per position"
    );
}

/// A sink error aborts the pass and surfaces — the observer can stop
/// an analysis, though it can never change what execution computes.
#[test]
fn execute_streaming_surfaces_a_sink_error() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let err = runtime
        .execute_streaming(&G_TOKENS, &mut |_| {
            Err(larql_vindex::VindexError::Parse("stop here".into()))
        })
        .expect_err("the sink's error must surface");
    assert!(err.to_string().contains("stop here"), "{err}");
}

/// The browse view binds through the runtime, off the same plan and
/// operand store execution uses.
#[test]
fn knowledge_view_binds_from_the_runtime() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let tok_json = crate::test_utils::synthetic_tokenizer_json(
        larql_vindex::format::vindex3::fixtures::G_VOCAB,
    );
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let view = runtime.knowledge_view(&tokenizer).unwrap();
    assert_eq!(view.num_layers(), runtime.plan().layers.len());
    assert!(view.max_features() > 0, "the miniature carries dense FFNs");
}

/// The overlaid runtime surface (V3-LQL-3B compose): with EMPTY
/// overrides every overlaid entry point is its plain counterpart bit
/// for bit, and a real edit changes what execution computes — the
/// runtime-level statement of the operand-source seam's contract.
#[test]
fn overlaid_entry_points_are_bit_identical_when_empty_and_observe_edits() {
    use super::{OperandEdit, OperandOverrides};
    use larql_vindex::format::vindex3::opplan::LayerFfn;

    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    // Plain arms.
    let mut kv = RowKvState::default();
    let prefill = runtime.prefill_into(&G_TOKENS, &mut kv).unwrap();
    let streamed = runtime
        .execute_streaming(&G_TOKENS, &mut |_| Ok(()))
        .unwrap();

    // Empty overlay: bit-identical on every entry point.
    let empty = OperandOverrides::new();
    let mut kv2 = RowKvState::default();
    assert_eq!(
        runtime
            .prefill_into_overlaid(&G_TOKENS, &empty, &mut kv2)
            .unwrap(),
        prefill
    );
    assert_eq!(
        runtime
            .execute_streaming_overlaid(&G_TOKENS, &empty, &mut |_| Ok(()))
            .unwrap()
            .logits,
        streamed.logits
    );
    let step_plain = {
        let mut session = runtime.session().unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    let step_overlaid = {
        let mut session = runtime.session_overlaid(&empty).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    assert_eq!(step_plain, step_overlaid);
    let resumed = {
        let mut session = runtime.session_with_kv_overlaid(&mut kv2, &empty).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    let resumed_plain = {
        let mut session = runtime.session_with_kv(&mut kv).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    assert_eq!(resumed, resumed_plain);

    // A real edit is observed.
    let Some(LayerFfn::Dense(ffn)) = &runtime.plan().layers[0].ffn else {
        panic!("miniature layer 0 is dense");
    };
    let gate = ffn.gate.as_ref().unwrap().clone();
    let mut edited = OperandOverrides::new();
    edited.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![5.0; gate.shape[1]],
        },
    );
    let out = runtime
        .execute_streaming_overlaid(&G_TOKENS, &edited, &mut |_| Ok(()))
        .unwrap();
    assert_ne!(out.logits, streamed.logits, "the edit must be observed");
    // …and the resolver serves the effective row.
    let effective = runtime.operands();
    let base_gate = effective.load(&gate).unwrap();
    assert_ne!(&base_gate[..gate.shape[1]], &vec![5.0; gate.shape[1]][..]);
}
