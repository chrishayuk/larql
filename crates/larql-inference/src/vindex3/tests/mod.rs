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

mod inputs;
mod opener;
mod prompt;
mod record;

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, G_TOKENS,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationRegistry;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{RowFactory, RowKvState};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::plan_component_ops;

use super::{
    continue_session_masked, generate_session, plan_kv_geometry, KvState, LogitsSession,
    RecordingObserver, SelectedContinuation, StepEvent, Vindex3Runtime, Vindex3Session,
};
use crate::error::InferenceError;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

/// `row/v1` selected for `plan` — the explicit continuation these tests
/// name now that sessions have no default (CONTINUATION-PLUGIN-1, C3).
pub(super) fn row(plan: &ComponentOpPlan) -> SelectedContinuation {
    let mut registry = ContinuationRegistry::new();
    registry.register(Box::new(RowFactory)).unwrap();
    let geometry = plan_continuation_geometry(plan).unwrap();
    registry
        .select(
            &RowKvState::identity(),
            &ContinuationConfig::empty(),
            &geometry,
        )
        .unwrap()
}
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::sampling::SamplingConfig;

/// The rung's target: sixteen greedy tokens through the runtime.
const NEW_TOKENS: usize = 16;
/// Component id the miniature systems encode their text stack under.
const COMPONENT: &str = "target";

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
    let mut session = DecodeSession::new(
        &plan,
        &store,
        backend,
        Box::new(larql_vindex::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();

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
    let mut session = runtime.session(&row(runtime.plan())).unwrap();
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
    let mut session = runtime.session(&row(runtime.plan())).unwrap();
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

mod dense_ffn;

mod driver_semantics_against_a_scripted_sess;
mod explain_lql_2_the_structured_explanation;
mod refusals;
