//! TinyStories-1M (`roneneldan/TinyStories-1M`, `GPTNeoForCausalLM`) end
//! to end: encode a safetensors checkpoint of it, close the plan, and hold
//! both executors' next-token log-probabilities to goldens recorded from
//! `GPTNeoForCausalLM` in f64 — on a short prompt, and on a 300-token one
//! that crosses the local layers' 256-token window (run with every layer
//! global, the same weights move the top token's log-probability by 0.09,
//! so a window not applied cannot pass). Ignored by default — it needs the
//! checkpoint, as `save_pretrained` writes it (the hub repo ships only
//! `pytorch_model.bin`):
//!
//! ```sh
//! LARQL_TINYSTORIES_CHECKPOINT=<dir with config.json + model.safetensors> \
//!   cargo test --release -p larql-vindex --lib gpt_neo_real -- --ignored --nocapture
//! ```

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::plan_component_ops;

const CHECKPOINT_ENV: &str = "LARQL_TINYSTORIES_CHECKPOINT";
const COMPONENT: &str = "target";
/// `Once upon a time there was a little girl named Lily. She loved to`.
const PROMPT: [u32; 15] = [
    7454, 2402, 257, 640, 612, 373, 257, 1310, 2576, 3706, 20037, 13, 1375, 6151, 284,
];
/// [`PROMPT`] repeated to this many tokens — past the 256-token window.
const LONG_TOKENS: usize = 300;
/// `GPTNeoForCausalLM` in float64 (transformers 5.x, eager attention).
const GOLDEN_SHORT: [(usize, f64); 5] = [
    (711, -0.827804),
    (4483, -2.520456),
    (7301, -2.6181),
    (467, -2.719448),
    (1057, -3.630239),
];
const GOLDEN_LONG: [(usize, f64); 5] = [
    (711, -0.747461),
    (4483, -2.750773),
    (1702, -3.452655),
    (16225, -3.685491),
    (9280, -3.735072),
];
/// f32 execution against an f64 reference; HF's own f32 forward sits
/// within 1e-4 of the same goldens over the whole vocabulary.
const TOLERANCE: f64 = 1e-4;

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum();
    logits
        .iter()
        .map(|&v| f64::from(v - max) - sum.ln())
        .collect()
}

fn check<B: PlanBackend>(
    container: &std::path::Path,
    backend: &B,
    name: &str,
    tokens: &[u32],
    golden: &[(usize, f64)],
) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    let logits = execute_plan(&plan, &store, tokens, backend)
        .unwrap()
        .logits
        .unwrap();
    let logprobs = log_softmax(&logits);
    for &(id, want) in golden {
        let got = logprobs[id];
        println!("{name}: id {id} logp {got:.6} (golden {want:.6})");
        assert!(
            (got - want).abs() < TOLERANCE,
            "{name}: id {id} logp {got} against golden {want}"
        );
    }
}

#[test]
#[ignore = "needs a TinyStories-1M safetensors checkpoint (LARQL_TINYSTORIES_CHECKPOINT)"]
fn tinystories_1m_matches_hugging_face_on_both_executors_past_the_window() {
    let checkpoint = std::env::var(CHECKPOINT_ENV).unwrap_or_else(|_| {
        panic!("set {CHECKPOINT_ENV} to the TinyStories-1M checkpoint directory")
    });
    let inventory = larql_models::inventory::build_inventory(checkpoint.as_ref()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[(COMPONENT.to_string(), inventory)], container.path()).unwrap();
    let long: Vec<u32> = PROMPT.iter().copied().cycle().take(LONG_TOKENS).collect();
    for (tokens, golden, what) in [
        (&PROMPT[..], &GOLDEN_SHORT, "short"),
        (&long[..], &GOLDEN_LONG, "long"),
    ] {
        check(
            container.path(),
            &ReferenceBackend,
            &format!("reference/{what}"),
            tokens,
            golden,
        );
        let production = ProductionBackend::new();
        check(
            container.path(),
            &production,
            &format!("production/{what}"),
            tokens,
            golden,
        );
    }
}
