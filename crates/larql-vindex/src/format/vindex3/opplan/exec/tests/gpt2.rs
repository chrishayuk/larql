//! **GPT-2 through the generic plan**: a learned absolute position table,
//! LayerNorm with bias, a non-gated biased MLP, Q/K/V split out of the
//! fused `Conv1D` `c_attn` at the inventory boundary, and a tied head.
//!
//! The executors are held to an independent forward written here over the
//! checkpoint's RAW tensors — input-major, fused, exactly as stored — so a
//! wrong transpose, a dropped bias or a mis-indexed position row cannot
//! pass by agreeing with itself.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures_gpt2::{
    miniature_gpt2_with, Gpt2Form, Gpt2Miniature, GPT2_EPS, GPT2_HEADS, GPT2_HEAD_DIM, GPT2_HIDDEN,
    GPT2_INNER, GPT2_LAYERS, GPT2_POSITIONS, GPT2_TOKENS, GPT2_VOCAB,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};
use crate::format::vindex3::plan::plan_system;
use larql_models::config::{InputPositionEncoding, PositionPolicy};

const COMPONENT: &str = "target";
/// f32 executors against an f64 reference over a 2-layer, 16-wide model.
const PARITY_TOLERANCE: f32 = 2e-5;

fn inventory(dir: &std::path::Path) -> larql_models::inventory::ArchitectureInventory {
    larql_models::inventory::build_inventory(dir).unwrap()
}

fn encoded(form: Gpt2Form) -> (Gpt2Miniature, tempfile::TempDir, tempfile::TempDir) {
    let checkpoint = tempfile::tempdir().unwrap();
    let model = miniature_gpt2_with(checkpoint.path(), form);
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

fn last_logits<B: PlanBackend>(
    container: &std::path::Path,
    tokens: &[u32],
    backend: &B,
) -> Vec<f32> {
    let plan = plan_of(container);
    let inspection = inspect_container(container, false).unwrap();
    let store = OperandStore::open(container, &inspection).unwrap();
    execute_plan(&plan, &store, tokens, backend)
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
    let inv = 1.0 / (var + GPT2_EPS).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(v, (w, b))| (v - mean) * inv * f64::from(*w) + f64::from(*b))
        .collect()
}

/// `x @ W + b` for an input-major (`Conv1D`) `W` of `[in, out]`.
fn conv1d(x: &[f64], w: &[f32], b: &[f32], out: usize) -> Vec<f64> {
    (0..out)
        .map(|o| {
            f64::from(b[o])
                + x.iter()
                    .enumerate()
                    .map(|(i, v)| v * f64::from(w[i * out + o]))
                    .sum::<f64>()
        })
        .collect()
}

fn gelu_new(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// `GPT2LMHeadModel.forward`, last position's logits, transcribed.
fn independent_forward(m: &Gpt2Miniature, tokens: &[u32]) -> Vec<f32> {
    let h = GPT2_HIDDEN;
    let mut xs: Vec<Vec<f64>> = tokens
        .iter()
        .enumerate()
        .map(|(p, &t)| {
            (0..h)
                .map(|i| f64::from(m.wte[t as usize * h + i]) + f64::from(m.wpe[p * h + i]))
                .collect()
        })
        .collect();
    let scale = match m.scale_attn_weights {
        Some(false) => 1.0,
        _ => 1.0 / (GPT2_HEAD_DIM as f64).sqrt(),
    };
    for l in &m.layers {
        let qkv: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                conv1d(
                    &layer_norm(x, &l.ln_1_w, &l.ln_1_b),
                    &l.c_attn_w,
                    &l.c_attn_b,
                    3 * h,
                )
            })
            .collect();
        let mut ctx = vec![vec![0.0f64; h]; xs.len()];
        for (p, row) in ctx.iter_mut().enumerate() {
            for head in 0..GPT2_HEADS {
                let off = head * GPT2_HEAD_DIM;
                let q = &qkv[p][off..off + GPT2_HEAD_DIM];
                let scores: Vec<f64> = (0..=p)
                    .map(|j| {
                        let k = &qkv[j][h + off..h + off + GPT2_HEAD_DIM];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    let v = &qkv[j][2 * h + off..2 * h + off + GPT2_HEAD_DIM];
                    for d in 0..GPT2_HEAD_DIM {
                        row[off + d] += e / sum * v[d];
                    }
                }
            }
        }
        for (x, c) in xs.iter_mut().zip(&ctx) {
            let o = conv1d(c, &l.attn_c_proj_w, &l.attn_c_proj_b, h);
            x.iter_mut().zip(o).for_each(|(x, o)| *x += o);
            let up: Vec<f64> = conv1d(
                &layer_norm(x, &l.ln_2_w, &l.ln_2_b),
                &l.c_fc_w,
                &l.c_fc_b,
                GPT2_INNER,
            )
            .into_iter()
            .map(gelu_new)
            .collect();
            let down = conv1d(&up, &l.mlp_c_proj_w, &l.mlp_c_proj_b, h);
            x.iter_mut().zip(down).for_each(|(x, d)| *x += d);
        }
    }
    let last = layer_norm(xs.last().unwrap(), &m.ln_f_w, &m.ln_f_b);
    (0..GPT2_VOCAB)
        .map(|t| {
            last.iter()
                .enumerate()
                .map(|(i, v)| v * f64::from(m.wte[t * h + i]))
                .sum::<f64>() as f32
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────

/// The planner graded GPT-2 `{kind: rope, theta: 10000}` off the parser's
/// DEFAULT base: its config declares no rope key and its positions are a
/// learned `wpe` table. Pinned at the graph, the surface and the plan.
#[test]
fn a_learned_position_table_with_no_rope_declaration_is_never_graded_rope() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt2_with(checkpoint.path(), Gpt2Form::default());
    let plan = plan_system(&[(COMPONENT.to_string(), inventory(checkpoint.path()))]);
    assert!(plan.admissible, "the miniature admits");
    let component = &plan.graph.components[0];
    let table = component.attention.as_ref().expect("attention table");
    assert_eq!(table.len(), GPT2_LAYERS);
    assert!(
        table.iter().all(|l| l.position == PositionPolicy::None),
        "{table:?}"
    );
    let head = component.execution.as_ref().unwrap().head.as_ref().unwrap();
    assert_eq!(
        head.input_position,
        Some(InputPositionEncoding::LearnedAbsolute {
            max_positions: GPT2_POSITIONS
        })
    );
    assert!(plan
        .graph
        .objects
        .iter()
        .any(|o| o.id == "target.position_embedding"));
}

#[test]
fn the_plan_binds_every_gpt2_operand_by_its_generic_role() {
    for prefix in ["transformer.", ""] {
        let (_m, _c, container) = encoded(Gpt2Form {
            prefix,
            ..Default::default()
        });
        let plan = plan_of(container.path());
        let embedding = plan.embedding.as_ref().unwrap();
        assert_eq!(
            embedding.position.as_ref().unwrap().table.shape,
            vec![GPT2_POSITIONS, GPT2_HIDDEN]
        );
        assert!(plan.final_norm.as_ref().unwrap().bias.is_some());
        for layer in &plan.layers {
            let attn = layer.attention.softmax().expect("softmax attention");
            assert_eq!(attn.position, PositionPolicy::None);
            // Split out of `c_attn` and transposed to `[out, in]`.
            assert_eq!(attn.q.shape, vec![GPT2_HIDDEN, GPT2_HIDDEN]);
            assert!(attn.q_bias.is_some() && attn.o_bias.is_some());
            assert!(layer.pre_attention_norm.as_ref().unwrap().bias.is_some());
            assert!(layer.pre_ffn_norm.as_ref().unwrap().bias.is_some());
            let Some(LayerFfn::Dense(ffn)) = &layer.ffn else {
                panic!("dense FFN");
            };
            assert!(ffn.gate.is_none(), "GPT-2's MLP is not gated");
            assert_eq!(ffn.up.shape, vec![GPT2_INNER, GPT2_HIDDEN]);
            assert!(ffn.up_bias.is_some() && ffn.down_bias.is_some());
        }
    }
}

#[test]
fn both_executors_match_an_independent_forward_over_the_stored_tensors() {
    for scale_attn_weights in [Some(true), None, Some(false)] {
        let (model, _c, container) = encoded(Gpt2Form {
            scale_attn_weights,
            ..Default::default()
        });
        let want = independent_forward(&model, &GPT2_TOKENS);
        for (name, got) in [
            (
                "reference",
                last_logits(container.path(), &GPT2_TOKENS, &ReferenceBackend),
            ),
            (
                "production",
                last_logits(container.path(), &GPT2_TOKENS, &ProductionBackend::new()),
            ),
        ] {
            let drift = max_abs_diff(&want, &got);
            assert!(
                drift < PARITY_TOLERANCE,
                "{name} (scale_attn_weights {scale_attn_weights:?}) drifts by {drift}"
            );
        }
    }
}

/// The decode path embeds one token at a time; it must add the row of the
/// position it is AT, not row 0, and refuse a position the table lacks.
#[test]
fn decode_steps_read_each_tokens_own_position_row_and_refuse_past_the_table() {
    let (model, _c, container) = encoded(Gpt2Form::default());
    let plan = plan_of(container.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ProductionBackend::new();
    let fresh = || Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default());
    let mut session = DecodeSession::new(&plan, &store, &backend, fresh()).unwrap();
    let mut last = Vec::new();
    for &token in &GPT2_TOKENS {
        last = session.step(token).unwrap().logits.unwrap();
    }
    let drift = max_abs_diff(&independent_forward(&model, &GPT2_TOKENS), &last);
    assert!(drift < PARITY_TOLERANCE, "decode drifts by {drift}");

    let mut session = DecodeSession::new(&plan, &store, &backend, fresh()).unwrap();
    for _ in 0..GPT2_POSITIONS {
        session.step(1).unwrap();
    }
    let err = session.step(1).err().expect("refused").to_string();
    assert!(err.contains("past the learned position table"), "{err}");
}

#[test]
fn the_mask_buffers_are_recorded_as_excluded_never_dropped_silently() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt2_with(checkpoint.path(), Gpt2Form::default());
    let inv = inventory(checkpoint.path());
    assert_eq!(inv.tensors.excluded.len(), 2 * GPT2_LAYERS);
    assert!(inv
        .tensors
        .excluded
        .iter()
        .all(|e| e.reason.contains("causal-mask buffer")));
    assert!(!inv
        .tensors
        .tensors
        .iter()
        .any(|t| t.name.ends_with("attn.c_attn.weight")));
}

#[test]
fn a_forward_changing_switch_declared_on_is_refused_at_plan_time() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt2_with(
        checkpoint.path(),
        Gpt2Form {
            inverse_layer_idx: true,
            ..Default::default()
        },
    );
    let plan = plan_system(&[(COMPONENT.to_string(), inventory(checkpoint.path()))]);
    assert!(!plan.admissible);
    assert!(plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .any(|f| f.blocks() && f.subject == "scale_attn_by_inverse_layer_idx"));
}

/// The guard is keyed on the DECLARATION, not on the family: a GPT-2
/// config that does declare a rotary base is graded rotary, and the plan
/// then carries both facts rather than choosing between them.
#[test]
fn a_declared_rope_base_is_still_read_as_rotary() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt2_with(
        checkpoint.path(),
        Gpt2Form {
            declares_rope: true,
            ..Default::default()
        },
    );
    let plan = plan_system(&[(COMPONENT.to_string(), inventory(checkpoint.path()))]);
    let table = plan.graph.components[0].attention.clone().unwrap();
    assert!(table.iter().all(|l| l.position.is_rotary()), "{table:?}");
}

/// G4 on a container whose decoder stack is DERIVED (split + transposed):
/// both hash ends agree, and each derived tensor is additionally held,
/// element by element and from the container side, to the stored tensor.
#[test]
fn g4_holds_derived_tensors_to_their_stored_source_element_by_element() {
    use crate::format::vindex3::verify_system::verify_system;
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_gpt2_with(checkpoint.path(), Gpt2Form::default());
    let honest = inventory(checkpoint.path());
    let named = [(COMPONENT.to_string(), honest.clone())];
    let container = tempfile::tempdir().unwrap();
    encode_system(&named, container.path()).unwrap();
    let verification = verify_system(&named, container.path()).unwrap();
    assert!(verification.verified, "{:?}", verification.payloads);
    let stack = verification
        .payloads
        .iter()
        .find(|p| p.representation.starts_with("target.decoder_stack"))
        .unwrap();
    assert!(
        stack.detail.contains("derived tensor(s) match"),
        "{}",
        stack.detail
    );

    // A container written with Q and K's slices exchanged — the shapes
    // close, the plan runs — is refused against the honest source, and the
    // audit names the element it caught.
    let mut doctored = honest;
    let slice_of = |inv: &larql_models::inventory::ArchitectureInventory, leaf: &str| {
        inv.tensors
            .tensors
            .iter()
            .position(|t| t.name.ends_with(leaf))
            .unwrap()
    };
    let (q, k) = (
        slice_of(&doctored, "h.0.self_attn.q_proj.weight"),
        slice_of(&doctored, "h.0.self_attn.k_proj.weight"),
    );
    let q_slice = doctored.tensors.tensors[q]
        .derivation
        .as_ref()
        .unwrap()
        .slice;
    let k_slice = doctored.tensors.tensors[k]
        .derivation
        .as_ref()
        .unwrap()
        .slice;
    doctored.tensors.tensors[q]
        .derivation
        .as_mut()
        .unwrap()
        .slice = k_slice;
    doctored.tensors.tensors[k]
        .derivation
        .as_mut()
        .unwrap()
        .slice = q_slice;
    let swapped = tempfile::tempdir().unwrap();
    encode_system(&[(COMPONENT.to_string(), doctored)], swapped.path()).unwrap();
    let verification = verify_system(&named, swapped.path()).unwrap();
    assert!(!verification.verified);
    let stack = verification
        .payloads
        .iter()
        .find(|p| p.representation.starts_with("target.decoder_stack"))
        .unwrap();
    assert!(
        stack.detail.contains("derived tensor audit failed"),
        "{}",
        stack.detail
    );
}

/// An excluded mask buffer is a claim that the declared span reproduces
/// it. The claim is checked from the stored bytes: a non-causal mask, or a
/// fill that would leave a masked score weight, refuses the encode.
#[test]
fn a_stored_mask_the_declared_span_does_not_reproduce_refuses_the_encode() {
    for (what, form) in [
        (
            "a non-causal mask",
            Gpt2Form {
                corrupt_mask: true,
                ..Default::default()
            },
        ),
        (
            "a weak fill",
            Gpt2Form {
                mask_fill: -1.0,
                ..Default::default()
            },
        ),
    ] {
        let checkpoint = tempfile::tempdir().unwrap();
        miniature_gpt2_with(checkpoint.path(), form);
        let container = tempfile::tempdir().unwrap();
        let err = encode_system(
            &[(COMPONENT.to_string(), inventory(checkpoint.path()))],
            container.path(),
        )
        .expect_err(what)
        .to_string();
        assert!(err.contains("refusing to exclude"), "{what}: {err}");
    }
    // And G4 re-checks the honest one.
    let (_m, checkpoint, container) = encoded(Gpt2Form::default());
    let named = [(COMPONENT.to_string(), inventory(checkpoint.path()))];
    let verification =
        crate::format::vindex3::verify_system::verify_system(&named, container.path()).unwrap();
    let check = verification
        .semantic
        .iter()
        .find(|c| c.subject == "excluded registered buffers")
        .expect("the buffer check runs at G4");
    assert!(check.pass, "{}", check.detail);
}
