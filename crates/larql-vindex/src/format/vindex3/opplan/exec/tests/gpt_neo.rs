//! **GPT-Neo through the generic plan**: the GPT-2 primitives (learned
//! absolute positions, LayerNorm with bias, a non-gated biased MLP, a tied
//! head) reused unchanged, plus three of its own — an output-only
//! attention bias, unscaled attention scores, and a `global`/`local`
//! interleave whose local layers see a trailing window.
//!
//! The executors are held to an independent forward over the stored
//! tensors, transcribed from `GPTNeoForCausalLM`. The fixture's window is
//! shorter than its prompt, so a local layer run globally, or a scale
//! applied that the model never had, moves the logits past tolerance.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures_gpt_neo::{
    miniature_gpt_neo_with, GptNeoForm, GptNeoMiniature, NEO_EPS, NEO_HEADS, NEO_HEAD_DIM,
    NEO_HIDDEN, NEO_INNER, NEO_LAYERS, NEO_LAYER_KINDS, NEO_TOKENS, NEO_VOCAB, NEO_WINDOW,
};
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use crate::format::vindex3::plan::plan_system;
use larql_models::config::PositionPolicy;

const COMPONENT: &str = "target";
const PARITY_TOLERANCE: f32 = 2e-5;

fn inventory(dir: &std::path::Path) -> larql_models::inventory::ArchitectureInventory {
    larql_models::inventory::build_inventory(dir).unwrap()
}

fn encoded(form: GptNeoForm) -> (GptNeoMiniature, tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    let model = miniature_gpt_neo_with(checkpoint.path(), form);
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[(COMPONENT.to_string(), inventory(checkpoint.path()))],
        container.path(),
    )
    .unwrap();
    (model, checkpoint, container)
}

fn plan_of(container: &std::path::Path) -> ComponentOpPlan {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    outcome.plan.unwrap()
}

fn last_logits<B: PlanBackend>(container: &std::path::Path, backend: &B) -> Vec<f32> {
    let plan = plan_of(container);
    let inspection = inspect_container(container, false).unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    execute_plan(&plan, &store, &NEO_TOKENS, backend)
        .unwrap()
        .logits
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ── The independent forward, over the stored tensors ──────────────────

fn layer_norm(x: &[f64], w: &[f32], b: &[f32]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + NEO_EPS).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(v, (w, b))| (v - mean) * inv * f64::from(*w) + f64::from(*b))
        .collect()
}

/// `x @ W.T (+ b)` for an `nn.Linear` weight `[out, in]`.
fn linear(x: &[f64], w: &[f32], b: Option<&[f32]>, out: usize) -> Vec<f64> {
    (0..out)
        .map(|o| {
            b.map_or(0.0, |b| f64::from(b[o]))
                + x.iter()
                    .enumerate()
                    .map(|(i, v)| v * f64::from(w[o * x.len() + i]))
                    .sum::<f64>()
        })
        .collect()
}

fn gelu_new(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// `GPTNeoForCausalLM.forward`, last position's logits, transcribed:
/// no score scaling, and a local layer keeps key `j` iff `j > p - window`.
fn independent_forward(m: &GptNeoMiniature, tokens: &[u32], kinds: &[&str]) -> Vec<f32> {
    let h = NEO_HIDDEN;
    let mut xs: Vec<Vec<f64>> = tokens
        .iter()
        .enumerate()
        .map(|(p, &t)| {
            (0..h)
                .map(|i| f64::from(m.wte[t as usize * h + i]) + f64::from(m.wpe[p * h + i]))
                .collect()
        })
        .collect();
    for (l, kind) in m.layers.iter().zip(kinds) {
        let normed: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| layer_norm(x, &l.ln_1_w, &l.ln_1_b))
            .collect();
        let q: Vec<Vec<f64>> = normed.iter().map(|x| linear(x, &l.q, None, h)).collect();
        let k: Vec<Vec<f64>> = normed.iter().map(|x| linear(x, &l.k, None, h)).collect();
        let v: Vec<Vec<f64>> = normed.iter().map(|x| linear(x, &l.v, None, h)).collect();
        let mut ctx = vec![vec![0.0f64; h]; xs.len()];
        for (p, row) in ctx.iter_mut().enumerate() {
            let first = match *kind {
                "local" => (p + 1).saturating_sub(NEO_WINDOW),
                _ => 0,
            };
            for head in 0..NEO_HEADS {
                let o = head * NEO_HEAD_DIM;
                let scores: Vec<f64> = (first..=p)
                    .map(|j| (0..NEO_HEAD_DIM).map(|d| q[p][o + d] * k[j][o + d]).sum())
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (e, j) in exps.iter().zip(first..=p) {
                    for d in 0..NEO_HEAD_DIM {
                        row[o + d] += e / sum * v[j][o + d];
                    }
                }
            }
        }
        for (x, c) in xs.iter_mut().zip(&ctx) {
            let attn = linear(c, &l.out, Some(&l.out_b), h);
            x.iter_mut().zip(attn).for_each(|(x, a)| *x += a);
            let up: Vec<f64> = linear(
                &layer_norm(x, &l.ln_2_w, &l.ln_2_b),
                &l.c_fc_w,
                Some(&l.c_fc_b),
                NEO_INNER,
            )
            .into_iter()
            .map(gelu_new)
            .collect();
            let down = linear(&up, &l.c_proj_w, Some(&l.c_proj_b), h);
            x.iter_mut().zip(down).for_each(|(x, d)| *x += d);
        }
    }
    let last = layer_norm(xs.last().unwrap(), &m.ln_f_w, &m.ln_f_b);
    (0..NEO_VOCAB)
        .map(|t| {
            last.iter()
                .enumerate()
                .map(|(i, v)| v * f64::from(m.wte[t * h + i]))
                .sum::<f64>() as f32
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────

#[test]
fn gpt_neo_plans_global_and_local_layers_with_no_rotation_and_no_score_scale() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt_neo_with(checkpoint.path(), GptNeoForm::default());
    let plan = plan_system(&[(COMPONENT.to_string(), inventory(checkpoint.path()))]);
    assert!(plan.admissible, "{:?}", plan.summary);
    let component = &plan.graph.components[0];
    let table = component.attention.as_ref().unwrap();
    assert!(table.iter().all(|l| l.position == PositionPolicy::None));
    assert_eq!(table[0].span, Some(AttentionSpan::Full));
    assert_eq!(table[1].span, Some(AttentionSpan::Sliding));
    assert_eq!(table[1].window, Some(NEO_WINDOW));
    let attention = component
        .execution
        .as_ref()
        .unwrap()
        .attention
        .as_ref()
        .unwrap();
    assert_eq!(attention.score_scale, 1.0);
    assert_eq!(attention.output_bias, Some(true));
}

#[test]
fn the_plan_binds_an_output_only_attention_bias() {
    let (_m, _c, container) = encoded(GptNeoForm::default());
    let plan = plan_of(container.path());
    assert!(plan.embedding.as_ref().unwrap().position.is_some());
    for layer in &plan.layers {
        let attn = layer.attention.softmax().unwrap();
        assert!(attn.q_bias.is_none() && attn.k_bias.is_none() && attn.v_bias.is_none());
        assert!(attn.o_bias.is_some());
        assert_eq!(attn.score_scale, 1.0);
    }
}

#[test]
fn both_executors_match_an_independent_forward_over_the_stored_tensors() {
    let (model, _c, container) = encoded(GptNeoForm::default());
    let want = independent_forward(&model, &NEO_TOKENS, &NEO_LAYER_KINDS);
    // The fixture distinguishes the window: the same weights run all
    // global move the logits far past tolerance.
    let global = independent_forward(&model, &NEO_TOKENS, &["global"; NEO_LAYERS]);
    assert!(max_abs_diff(&want, &global) > 100.0 * PARITY_TOLERANCE);
    for (name, got) in [
        (
            "reference",
            last_logits(container.path(), &ReferenceBackend),
        ),
        (
            "production",
            last_logits(container.path(), &ProductionBackend::new()),
        ),
    ] {
        let drift = max_abs_diff(&want, &got);
        assert!(drift < PARITY_TOLERANCE, "{name} drifts by {drift}");
    }
}

#[test]
fn decode_steps_agree_with_the_independent_forward_past_the_window() {
    let (model, _c, container) = encoded(GptNeoForm::default());
    let plan = plan_of(container.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend;
    let mut session = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut last = Vec::new();
    for &token in &NEO_TOKENS {
        last = session.step(token).unwrap().logits.unwrap();
    }
    let want = independent_forward(&model, &NEO_TOKENS, &NEO_LAYER_KINDS);
    let drift = max_abs_diff(&want, &last);
    assert!(drift < PARITY_TOLERANCE, "decode drifts by {drift}");
}

#[test]
fn a_q_bias_gpt_neo_does_not_declare_is_refused_by_name() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt_neo_with(
        checkpoint.path(),
        GptNeoForm {
            q_bias: true,
            ..Default::default()
        },
    );
    let container = tempfile::tempdir().unwrap();
    let err = encode_system(
        &[(COMPONENT.to_string(), inventory(checkpoint.path()))],
        container.path(),
    )
    .unwrap_err()
    .to_string();
    assert_eq!(
        err.matches("OperandImpliesAbsentOp").count(),
        NEO_LAYERS,
        "{err}"
    );
    assert!(err.contains("q_proj.bias"), "{err}");
}

#[test]
fn attention_layers_that_contradict_attention_types_block_the_plan() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt_neo_with(
        checkpoint.path(),
        GptNeoForm {
            contradictory_layers: true,
            ..Default::default()
        },
    );
    let plan = plan_system(&[(COMPONENT.to_string(), inventory(checkpoint.path()))]);
    assert!(!plan.admissible);
    assert!(plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .any(|f| f.blocks() && f.subject == "attention_layers"));
}
