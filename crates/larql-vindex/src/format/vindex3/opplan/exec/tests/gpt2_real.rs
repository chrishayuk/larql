//! GPT-2 Small (`openai-community/gpt2`) end to end: encode the Hugging
//! Face checkpoint, close the plan, and hold both executors' next-token
//! log-probabilities to goldens recorded from `GPT2LMHeadModel` in f64.
//! Ignored by default — it needs the checkpoint:
//!
//! ```sh
//! LARQL_GPT2_CHECKPOINT=<snapshot dir of openai-community/gpt2> \
//!   cargo test --release -p larql-vindex --lib gpt2_real -- --ignored --nocapture
//! ```

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::plan_component_ops;

const CHECKPOINT_ENV: &str = "LARQL_GPT2_CHECKPOINT";
const COMPONENT: &str = "target";
/// `The capital of France is`, GPT-2's own BPE.
const PROMPT: [u32; 5] = [464, 3139, 286, 4881, 318];
/// `GPT2LMHeadModel` in float64 (transformers 5.x, eager attention): the
/// five most likely next tokens and their log-probabilities.
const GOLDEN_TOP5: [(usize, f64); 5] = [
    (262, -2.469917),
    (783, -3.037685),
    (257, -3.075654),
    (4881, -3.430314),
    (6342, -3.434413),
];
/// f32 execution against an f64 reference; HF's own f32 forward sits
/// within 2e-4 of the same goldens over the whole vocabulary.
const TOLERANCE: f64 = 1e-4;

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum();
    logits
        .iter()
        .map(|&v| f64::from(v - max) - sum.ln())
        .collect()
}

fn check<B: PlanBackend>(container: &std::path::Path, backend: &B, name: &str) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    let logits = execute_plan(&plan, &store, &PROMPT, backend)
        .unwrap()
        .logits
        .unwrap();
    let logprobs = log_softmax(&logits);
    for (id, want) in GOLDEN_TOP5 {
        let got = logprobs[id];
        println!("{name}: id {id} logp {got:.6} (golden {want:.6})");
        assert!(
            (got - want).abs() < TOLERANCE,
            "{name}: id {id} logp {got} against golden {want}"
        );
    }
}

#[test]
#[ignore = "needs the openai-community/gpt2 checkpoint (LARQL_GPT2_CHECKPOINT)"]
fn gpt2_small_matches_hugging_face_on_both_executors() {
    let checkpoint = std::env::var(CHECKPOINT_ENV)
        .unwrap_or_else(|_| panic!("set {CHECKPOINT_ENV} to the GPT-2 snapshot directory"));
    let inventory = larql_models::inventory::build_inventory(checkpoint.as_ref()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[(COMPONENT.to_string(), inventory)], container.path()).unwrap();
    check(container.path(), &ReferenceBackend, "reference");
    check(container.path(), &ProductionBackend::new(), "production");
}
