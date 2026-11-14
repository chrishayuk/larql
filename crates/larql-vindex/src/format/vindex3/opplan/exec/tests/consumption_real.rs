//! Runtime consumption of an edited stored weight, on REAL Gemma 3 4B containers, on the production CPU backend,
//! over NON-COHORT prompts only (the same three as the I5 qualification).
//!
//! The question is narrower than "is the edit right" (the ledger contract's) and than "does the behaviour change
//! as intended" (no behavioural claim is made): does the production run of the EDITED container compute with the
//! edited tensor, at the layer the edit names, and with nothing else different?
//!
//! Two containers: a base, and an edit of one FFN `down_proj` compiled from it. This harness establishes, by bits:
//!
//! C1 upstream identity: every carrier write below the edited layer's FFN, at every position, is bit-identical
//!    between the two runs, and so is the `down_proj` INPUT the backend fed the edited tensor. The two runs
//!    therefore hand the same vector to a tensor that differs only by the edit.
//! C2 first divergence: the edited layer's FFN write differs at every position, and the final logits differ.
//! C3 control (optional, a third container): a container whose edit is declared not to reach execution is
//!    bit-identical to the base at every write and every logit.
//!
//! It then writes a record (`LARQL_CONSUMPTION_RECORD`) holding, per run and position, the captured `down_proj`
//! input and the carrier on both sides of the FFN write. An INDEPENDENT reference
//! (`scripts/consumption_reference.py`) recomputes the FFN write from each container's own tensor bytes and that
//! input, and judges which container's weights each run's write is the product of. This file computes no
//! expected value.
//!
//! Instrument evidence only. Ignored by default:
//!
//! ```sh
//! LARQL_V3_CONTAINER=~/chris-models/gemma3-4b-it.vindex3 \
//! LARQL_V3_EDITED=~/chris-models/sg-edit-i2-scale.vindex3 \
//! LARQL_V3_EDITED_LAYER=12 \
//! LARQL_CONSUMPTION_RECORD=/path/to/record.json \
//!   cargo test --release -p larql-vindex --lib consumption_real -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use super::i5_support::{same_bits, Recorder, WriteKey};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;
use crate::format::vindex3::opplan::exec::intervene::{vector_sha256, InterventionPlan};
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::SublayerSite;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

const BASE_VAR: &str = "LARQL_V3_CONTAINER";
const EDITED_VAR: &str = "LARQL_V3_EDITED";
const LAYER_VAR: &str = "LARQL_V3_EDITED_LAYER";
const CONTROL_VAR: &str = "LARQL_V3_CONTROL";
const RECORD_VAR: &str = "LARQL_CONSUMPTION_RECORD";
const RECORD_SCHEMA: &str = "larql-consumption-record/1";
const INDEX_JSON: &str = "index.json";

const PROMPTS_VAR: &str = "LARQL_CONSUMPTION_PROMPTS";
/// Settings that select the arithmetic a production run performs, recorded with every run: a record that does
/// not say which policy executed cannot be read as evidence about any one of them.
const POLICY_VARS: [&str; 4] = [
    "LARQL_CPU_MAX_FORMAT",
    "LARQL_CPU_ARITHMETIC",
    "LARQL_KQUANT_EXEC",
    "RAYON_NUM_THREADS",
];

/// `<bos>The capital of France is`, pinned Gemma 3 tokenizer.
const FRANCE: [u32; 6] = [2, 818, 5279, 529, 7001, 563];
/// The same template with the one-token subject swapped to Germany.
const GERMANY: [u32; 6] = [2, 818, 5279, 529, 9405, 563];
/// `<bos>The weather in the mountains today is`.
const GENERIC: [u32; 8] = [2, 818, 7606, 528, 506, 13254, 3124, 563];
/// The exploratory run's prompts: what is used when no prompt file is named.
const EXPLORATORY_PROMPTS: [(&str, &[u32]); 3] = [
    ("france", &FRANCE),
    ("germany", &GERMANY),
    ("generic", &GENERIC),
];

/// The prompts to run: a JSON file `[{"name": ..., "tokens": [...]}, ...]` when [`PROMPTS_VAR`] names one (its
/// SHA-256 goes in the record), otherwise the exploratory three.
fn prompts() -> (Vec<(String, Vec<u32>)>, Option<String>) {
    let Ok(path) = std::env::var(PROMPTS_VAR) else {
        let fixed = EXPLORATORY_PROMPTS
            .iter()
            .map(|(name, tokens)| (name.to_string(), tokens.to_vec()))
            .collect();
        return (fixed, None);
    };
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(&path).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let listed: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
    let parsed = listed
        .iter()
        .map(|p| {
            let tokens = p["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect();
            (p["name"].as_str().unwrap().to_string(), tokens)
        })
        .collect();
    (parsed, Some(digest))
}

/// The largest elementwise difference between two logit vectors, and each one's argmax. Descriptive only.
fn logit_comparison(a: &[f32], b: &[f32]) -> serde_json::Value {
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
    };
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    serde_json::json!({
        "vocab": a.len(),
        "bit_identical": same_bits(a, b),
        "elements_differing": a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count(),
        "max_abs_difference": max_abs,
        "argmax": [argmax(a), argmax(b)],
    })
}

struct Image {
    path: String,
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: ProductionBackend,
}

/// One run's observations: every carrier write, the captured `down_proj` inputs, and each step's logits.
struct Run {
    writes: BTreeMap<WriteKey, Vec<f32>>,
    down: DownInputCapture,
    logits: Vec<Vec<f32>>,
}

impl Image {
    fn open(var: &str) -> Self {
        let path = std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a container"));
        let root = std::path::Path::new(&path);
        let inspection = inspect_container(root, false).unwrap();
        let outcome = plan_component_ops(&inspection, root, "target").unwrap();
        assert!(outcome.closed(), "defects: {:?}", outcome.defects);
        let plan = outcome.plan.unwrap();
        let store = OperandStore::open(root, &inspection).unwrap();
        let backend = ProductionBackend::new();
        let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
        Self {
            path,
            plan,
            ops,
            backend,
        }
    }

    fn run(&self, tokens: &[u32], layer: usize) -> Run {
        let mut recorder = Recorder {
            down: Some(DownInputCapture::at(
                (0..tokens.len()).map(|position| (layer, position)),
            )),
            ..Recorder::default()
        };
        let mut kv = RowKvState::default();
        let mut session =
            DecodeSession::over_prepared(&self.plan, &self.ops, &self.backend, &mut kv).unwrap();
        let none = (InterventionPlan::none(), HeadInterventionPlan::none());
        let logits = tokens
            .iter()
            .map(|&t| {
                session
                    .step_intervened(t, &mut recorder, &none.0, &none.1)
                    .unwrap()
                    .logits
                    .unwrap()
            })
            .collect();
        Run {
            writes: recorder.writes,
            down: recorder.down.unwrap(),
            logits,
        }
    }

    fn index_sha256(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(std::path::Path::new(&self.path).join(INDEX_JSON)).unwrap();
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Whether a write is upstream of `(layer, Ffn)`: nothing an edit of that layer's FFN can reach.
fn upstream(key: &WriteKey, layer: usize) -> bool {
    key.0 < layer || (key.0 == layer && key.1 == SublayerSite::Attention)
}

fn hex(values: &[f32]) -> String {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// C1 and C2: the edited run differs from the base at the edited layer's FFN write and nowhere before it.
fn check_edit(name: &str, layer: usize, base: &Run, edited: &Run) {
    assert!(
        base.logits.len() > 1,
        "{name}: a prompt needs more than its first token"
    );
    assert_eq!(
        base.writes.keys().collect::<Vec<_>>(),
        edited.writes.keys().collect::<Vec<_>>(),
        "{name}: the runs wrote different addresses"
    );
    let positions = base.logits.len();
    let mut upstream_writes = 0;
    for (key, after) in base.writes.iter().filter(|(k, _)| upstream(k, layer)) {
        assert!(
            same_bits(after, &edited.writes[key]),
            "{name} C1: the write at {key:?} is upstream of the edit and differs"
        );
        upstream_writes += 1;
    }
    for position in 0..positions {
        let (a, b) = (
            base.down.get(layer, position).unwrap(),
            edited.down.get(layer, position).unwrap(),
        );
        assert!(
            same_bits(a, b),
            "{name} C1: the down_proj input at layer {layer} position {position} differs"
        );
        let key = (layer, SublayerSite::Ffn, position);
        assert!(
            !same_bits(&base.writes[&key], &edited.writes[&key]),
            "{name} C2: the edited layer's FFN write at position {position} is unchanged"
        );
    }
    let last = positions - 1;
    assert!(
        !same_bits(&base.logits[last], &edited.logits[last]),
        "{name} C2: the final logits are unchanged"
    );
    println!(
        "{name}: C1 {upstream_writes} upstream writes and {positions} down_proj inputs bit-identical; \
         C2 the layer-{layer} FFN write differs at all {positions} positions and the logits differ"
    );
}

/// C3: a container whose edit does not reach execution runs bit-identically to the base.
fn check_control(name: &str, base: &Run, control: &Run) {
    assert_eq!(base.writes.len(), control.writes.len());
    for (key, after) in &base.writes {
        assert!(
            same_bits(after, &control.writes[key]),
            "{name} C3: the control's write at {key:?} differs from the base"
        );
    }
    for (step, (a, b)) in base.logits.iter().zip(&control.logits).enumerate() {
        assert!(same_bits(a, b), "{name} C3: logits differ at step {step}");
    }
    println!(
        "{name}: C3 all {} writes and {} logit vectors bit-identical to the base",
        base.writes.len(),
        base.logits.len()
    );
}

fn run_record(run: &Run, layer: usize) -> serde_json::Value {
    let positions: Vec<serde_json::Value> = (0..run.logits.len())
        .map(|position| {
            let before = &run.writes[&(layer, SublayerSite::Attention, position)];
            let after = &run.writes[&(layer, SublayerSite::Ffn, position)];
            serde_json::json!({
                "position": position,
                "down_input_f32le_hex": hex(run.down.get(layer, position).unwrap()),
                "carrier_before_ffn_f32le_hex": hex(before),
                "carrier_after_ffn_f32le_hex": hex(after),
            })
        })
        .collect();
    let last = run.logits.last().unwrap();
    serde_json::json!({
        "positions": positions,
        "final_logits_sha256": vector_sha256(last),
        "final_logits_vocab": last.len(),
    })
}

#[test]
#[ignore = "needs real Gemma 3 4B containers: LARQL_V3_CONTAINER and LARQL_V3_EDITED"]
fn consumption_real_an_edited_down_proj_is_what_the_production_run_computes_with() {
    let layer: usize = std::env::var(LAYER_VAR)
        .unwrap_or_else(|_| panic!("set {LAYER_VAR} to the edited layer"))
        .parse()
        .unwrap();
    let base = Image::open(BASE_VAR);
    let edited = Image::open(EDITED_VAR);
    let control = std::env::var(CONTROL_VAR)
        .is_ok()
        .then(|| Image::open(CONTROL_VAR));

    let (listed, prompts_sha256) = prompts();
    let mut prompts = Vec::new();
    for (name, tokens) in &listed {
        let (a, b) = (base.run(tokens, layer), edited.run(tokens, layer));
        check_edit(name, layer, &a, &b);
        let mut entry = serde_json::json!({
            "name": name,
            "tokens": tokens,
            "base": run_record(&a, layer),
            "edited": run_record(&b, layer),
            "final_logits_base_vs_edited": logit_comparison(
                a.logits.last().unwrap(), b.logits.last().unwrap()),
        });
        if let Some(control) = &control {
            let c = control.run(tokens, layer);
            check_control(name, &a, &c);
            entry["control_bit_identical_to_base"] = serde_json::json!(true);
            entry["final_logits_base_vs_control"] =
                logit_comparison(a.logits.last().unwrap(), c.logits.last().unwrap());
        }
        prompts.push(entry);
    }
    let environment: serde_json::Map<String, serde_json::Value> = POLICY_VARS
        .iter()
        .map(|var| {
            let value =
                std::env::var(var).map_or(serde_json::Value::Null, serde_json::Value::String);
            (var.to_string(), value)
        })
        .collect();

    let record = serde_json::json!({
        "schema": RECORD_SCHEMA,
        "backend": "production",
        "component": "target",
        "edited_layer": layer,
        "environment": environment,
        "prompts_file_sha256": prompts_sha256,
        "base": {"path": base.path, "index_sha256": base.index_sha256()},
        "edited": {"path": edited.path, "index_sha256": edited.index_sha256()},
        "control": control.as_ref().map(|c| serde_json::json!({
            "path": c.path, "index_sha256": c.index_sha256()})),
        "checks": {
            "C1_upstream_writes_and_down_inputs_bit_identical": true,
            "C2_edited_layer_ffn_write_and_logits_differ": true,
            "C3_control_bit_identical": control.is_some().then_some(true),
        },
        "prompts": prompts,
    });
    if let Ok(path) = std::env::var(RECORD_VAR) {
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();
        println!("record written to {path}");
    }
}
