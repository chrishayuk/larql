//! Pre-2C probe: do the two attention realisations agree *at the
//! attention boundary*, per layer and per position?
//!
//! V3-SERVE-2 wants `traverse`'s batched realisation to populate the KV
//! provider, using the K/V rows it already computes and currently
//! discards. The whole plan rests on those rows being the same rows the
//! per-position path appends — so that is worth proving **before** the
//! trait changes shape around them.
//!
//! # What this can and cannot observe today
//!
//! It cannot compare the K/V rows directly: `PlanBackend::attention`
//! returns outputs only, and making it return the rows *is* the
//! refactor. Adding a peephole for the probe would mean testing a code
//! path the refactor then replaces.
//!
//! What it does instead is compare the two realisations where they are
//! observable and where a K/V disagreement must surface — the attention
//! outputs, per position, per layer, isolated from the rest of the
//! stack. Position `p`'s batched output consumes the batched path's own
//! K/V for positions `0..=p`; the stepped output consumes the rows the
//! provider accumulated. If those rows disagreed anywhere, the outputs
//! would have to disagree from that position on.
//!
//! This is strictly sharper than the existing decode-vs-batch gates,
//! which compare **logits** at the end of the stack, where an attention
//! difference has a whole FFN and residual chain in which to be masked.
//!
//! The direct row-equality gate belongs to the refactor commit, where
//! the rows become returnable. This probe is what says the refactor is
//! worth starting.

use super::super::backend::{AttentionStepCall, PlanBackend};
use super::super::kv::{KvState, RowKvState};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::reference::ReferenceBackend;
use super::super::{execute_prepared_streaming, PlaneEvent};
use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Enough positions that a K/V disagreement has somewhere to show.
const TOKENS: [u32; 6] = [1, 2, 3, 4, 5, 6];

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "attention-parity",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

/// Discrepancy between two row sets, in the two shapes that matter: the
/// worst single element, and the error relative to the signal's own
/// scale (so a large-magnitude row is not flattered by an absolute
/// bound).
struct Divergence {
    max_abs: f32,
    rel_rms: f32,
    bit_identical: bool,
}

fn diverge(a: &[Vec<f32>], b: &[Vec<f32>]) -> Divergence {
    assert_eq!(a.len(), b.len(), "row counts differ");
    let mut max_abs = 0.0f32;
    let mut sq_err = 0.0f64;
    let mut sq_sig = 0.0f64;
    let mut bit_identical = true;
    for (ra, rb) in a.iter().zip(b) {
        assert_eq!(ra.len(), rb.len(), "row widths differ");
        for (x, y) in ra.iter().zip(rb) {
            if x.to_bits() != y.to_bits() {
                bit_identical = false;
            }
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            sq_err += f64::from(d) * f64::from(d);
            sq_sig += f64::from(*x) * f64::from(*x);
        }
    }
    Divergence {
        max_abs,
        rel_rms: if sq_sig > 0.0 {
            (sq_err / sq_sig).sqrt() as f32
        } else {
            0.0
        },
        bit_identical,
    }
}

/// The per-layer hidden states the batch traversal actually produces,
/// so each layer's attention is probed on its real inputs rather than
/// on synthetic rows.
fn hidden_per_layer<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
) -> Vec<Vec<Vec<f32>>> {
    let mut per_layer = Vec::new();
    execute_prepared_streaming(plan, ops, &TOKENS, backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(rows) => per_layer.push(rows.to_vec()),
            PlaneEvent::Layer { trace, .. } => per_layer.push(trace.post_layer.clone()),
        }
        Ok(())
    })
    .unwrap();
    per_layer.pop(); // the last layer's output feeds no further attention
    per_layer
}

/// What one realisation produced: outputs, and the conditioned rows a
/// provider would cache.
struct Realisation {
    outputs: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// Run both realisations of one layer's attention over the same inputs.
fn both_realisations<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    layer_index: usize,
    hidden: &[Vec<f32>],
) -> (Realisation, Realisation) {
    let layer = &plan.layers[layer_index];
    let prepared = &ops.layers()[layer_index];
    let width = ops.hidden();
    let eps = layer.declared_norm_eps;

    // Exactly what `execute_layer` feeds attention.
    let inputs: Vec<Vec<f32>> = hidden
        .iter()
        .map(|row| {
            prepared
                .pre_attention
                .as_ref()
                .expect("this fixture is a pre-norm stack")
                .apply(backend, row)
        })
        .collect();

    // This probe is about the KV realisations of SOFTMAX attention, so
    // it takes the softmax operands directly and would not compile for a
    // recurrence — which is the honest shape: a recurrence keeps no rows
    // and has no batched-vs-stepped KV question to answer.
    let super::super::prepared::PreparedAttention::Softmax(attn_ops) = &prepared.attention else {
        panic!("this probe is for softmax layers");
    };

    // Realisation A: one batched call over every position.
    let out = backend
        .attention(attn_ops.call(layer.attention.softmax().unwrap(), &inputs, eps, width))
        .unwrap();
    let batched = Realisation {
        outputs: out.outputs,
        keys: out.keys,
        values: out.values,
    };

    // Realisation B: position by position into a provider — the code
    // `attention_into_kv` runs, transcribed so the probe cannot drift
    // from it silently.
    let mut kv = RowKvState::default();
    kv.prepare(&super::super::kv::plan_kv_geometry(plan));
    let mut stepped = Realisation {
        outputs: Vec::with_capacity(inputs.len()),
        keys: Vec::with_capacity(inputs.len()),
        values: Vec::with_capacity(inputs.len()),
    };
    for offset in 0..inputs.len() {
        let call = attn_ops.call(
            layer.attention.softmax().unwrap(),
            &inputs[offset..=offset],
            eps,
            width,
        );
        let out = backend
            .attention_step(AttentionStepCall {
                op: call,
                position: offset,
                keys: kv.keys(layer_index),
                values: kv.values(layer_index),
            })
            .unwrap();
        stepped.keys.push(out.key.clone());
        stepped.values.push(out.value.clone());
        kv.append(layer_index, out.key, out.value);
        stepped.outputs.push(out.output);
    }
    (batched, stepped)
}

fn probe<B: PlanBackend>(name: &str, backend: &B) {
    let (_container, plan, store) = fixture();
    let ops = PreparedOperands::load(&plan, &store, backend, ExecutionSlice::Full).unwrap();
    let hidden = hidden_per_layer(&plan, &ops, backend);
    assert_eq!(hidden.len(), plan.layers.len(), "one input set per layer");

    for (layer_index, rows) in hidden.iter().enumerate() {
        let (batched, stepped) = both_realisations(&plan, &ops, backend, layer_index, rows);
        for (what, a, b) in [
            ("K", &batched.keys, &stepped.keys),
            ("V", &batched.values, &stepped.values),
            ("output", &batched.outputs, &stepped.outputs),
        ] {
            let d = diverge(a, b);
            println!(
                "{name} layer {layer_index} {what}: max_abs={:.3e} rel_rms={:.3e} \
                 bit_identical={}",
                d.max_abs, d.rel_rms, d.bit_identical
            );
            assert!(
                d.bit_identical,
                "{name} layer {layer_index}: the batched and per-position realisations \
                 disagree on {what} (max_abs={:.3e}, rel_rms={:.3e}). The rows the \
                 batched pass returns are not the rows the provider would accumulate, \
                 so populating from them changes what the model computes.",
                d.max_abs, d.rel_rms
            );
        }
    }
}

/// The semantic anchor: naive f32, sharing no arithmetic with
/// `larql-compute`. If any backend is going to disagree with itself,
/// this is the one where it matters most.
#[test]
fn reference_batched_and_stepped_attention_agree_per_position() {
    probe("reference", &ReferenceBackend::new());
}

/// The backend the server actually runs.
#[test]
fn production_batched_and_stepped_attention_agree_per_position() {
    probe("production", &ProductionBackend::new());
}

/// The rows must be a function of (operands, input, position) and
/// nothing else — no hidden state carried between calls. If a second
/// identical step produced different rows, returning them from a
/// batched traversal would be unsound however well the parity above
/// held.
#[test]
fn stepping_the_same_position_twice_yields_identical_rows() {
    let backend = ProductionBackend::new();
    let (_container, plan, store) = fixture();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let hidden = hidden_per_layer(&plan, &ops, &backend);

    let layer = &plan.layers[0];
    let prepared = &ops.layers()[0];
    let width = ops.hidden();
    let inputs: Vec<Vec<f32>> = hidden[0]
        .iter()
        .map(|row| {
            prepared
                .pre_attention
                .as_ref()
                .expect("this fixture is a pre-norm stack")
                .apply(&backend, row)
        })
        .collect();

    let super::super::prepared::PreparedAttention::Softmax(attn_ops) = &prepared.attention else {
        panic!("this probe is for softmax layers");
    };
    let step = |position: usize| {
        let call = attn_ops.call(
            layer.attention.softmax().unwrap(),
            &inputs[position..=position],
            layer.declared_norm_eps,
            width,
        );
        backend
            .attention_step(AttentionStepCall {
                op: call,
                position,
                keys: &[],
                values: &[],
            })
            .unwrap()
    };

    let first = step(0);
    let second = step(0);
    assert_eq!(first.key, second.key, "K row is not deterministic");
    assert_eq!(first.value, second.value, "V row is not deterministic");
    assert_eq!(first.output, second.output, "output is not deterministic");
}

/// Control: the probe must be able to *fail*.
///
/// A parity gate that reports `max_abs = 0` on everything looks the
/// same whether the paths agree or the harness is inert. This feeds the
/// two realisations deliberately different inputs — one position's row
/// perturbed — and requires the comparison to notice. Without this, the
/// green result above is not evidence.
#[test]
fn the_probe_reports_a_divergence_when_the_paths_are_fed_different_inputs() {
    let backend = ProductionBackend::new();
    let (_container, plan, store) = fixture();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let hidden = hidden_per_layer(&plan, &ops, &backend);

    let (batched, _) = both_realisations(&plan, &ops, &backend, 0, &hidden[0]);

    // Perturb one element of one position's input and re-run the
    // stepped side only.
    let mut skewed = hidden[0].clone();
    skewed[2][0] += 1e-3;
    let (_, stepped_skewed) = both_realisations(&plan, &ops, &backend, 0, &skewed);

    let d = diverge(&batched.keys, &stepped_skewed.keys);
    assert!(
        !d.bit_identical && d.max_abs > 0.0,
        "the comparison did not notice a perturbed input — the parity result above \
         would be meaningless (max_abs={:.3e}, rel_rms={:.3e})",
        d.max_abs,
        d.rel_rms
    );
    println!(
        "control: perturbed input gives max_abs={:.3e} rel_rms={:.3e}",
        d.max_abs, d.rel_rms
    );
}

/// The same probe against a **real** container, so the result is not a
/// statement about a two-layer miniature fixture.
///
/// Ignored by default because it needs a container on disk:
///
/// ```sh
/// LARQL_V3_CONTAINER=~/chris-models/granite-4.1-3b.vindex3 \
///   cargo test -p larql-vindex --lib attention_kv_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a real VINDEX3 container in LARQL_V3_CONTAINER"]
fn real_container_batched_and_stepped_attention_agree_per_position() {
    let Ok(path) = std::env::var("LARQL_V3_CONTAINER") else {
        panic!("set LARQL_V3_CONTAINER to a container directory");
    };
    let root = std::path::Path::new(&path);
    let inspection = inspect_container(root, false).unwrap();
    let outcome = plan_component_ops(&inspection, root, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(root, &inspection).unwrap();

    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let hidden = hidden_per_layer(&plan, &ops, &backend);

    let mut worst = 0.0f32;
    for (layer_index, rows) in hidden.iter().enumerate() {
        let (batched, stepped) = both_realisations(&plan, &ops, &backend, layer_index, rows);
        for (what, a, b) in [
            ("K", &batched.keys, &stepped.keys),
            ("V", &batched.values, &stepped.values),
            ("output", &batched.outputs, &stepped.outputs),
        ] {
            let d = diverge(a, b);
            worst = worst.max(d.max_abs);
            assert!(
                d.bit_identical,
                "layer {layer_index} of {path}: realisations disagree on {what} \
                 (max_abs={:.3e}, rel_rms={:.3e})",
                d.max_abs, d.rel_rms
            );
        }
    }
    println!(
        "{} layers, all bit-identical (worst max_abs {worst:.3e})",
        hidden.len()
    );
}
