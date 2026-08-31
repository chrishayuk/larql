//! **The lift-2 witness, in miniature: a KDA + MLA stack executes
//! through the ordinary plan path.**
//!
//! Kimi Linear's two operators were represented long before they were
//! executable, and what stood between was never a missing kernel —
//! `exec::kda` and `exec::mla` have been parity-proven against banked
//! oracles since P3d. It was two facts the container could not carry
//! (the drill's F5 and F6): the precision KDA's recurrence is held at,
//! which no checkpoint declares, and the epsilon MLA's latent norm runs
//! at, which is NOT the layer's `rms_norm_eps`. With both carried, the
//! generic path binds these layers by `OperandRole`, sizes their state
//! from `plan_continuation_geometry`, and runs them — no family lookup,
//! no `model_type` dispatch, no hardcoded tensor spellings.
//!
//! What this file proves, end to end and in CI:
//!
//! 1. a KDA/MLA checkpoint admits and ENCODES (operand closure holds
//!    over both operand programs);
//! 2. the continuation declares TWO state species in ONE program — a
//!    fixed-size recurrence on the KDA layers and a per-position latent
//!    cache on the MLA layers, neither of which the schema could state
//!    before this lift (the third species, KV rows, rides the same seam
//!    and is witnessed by the softmax and conv-QKV stacks);
//! 3. it executes, and the decode path agrees with batch prefill
//!    BITWISE across the step boundary — which holds only if the
//!    recurrent matrix, the three convolution windows AND the latent
//!    cache all cross it.
//!
//! Awkward miniature widths, every one distinct, so a transposition or
//! an off-by-one cannot coincide: hidden 12; KDA 2 heads × 3 (width 6),
//! conv 3, gate rank 4; MLA 2 heads, latent 5, nope 3, rope 2, v 4;
//! FFN 16. `head_dim` in the config is deliberately a value belonging to
//! NEITHER operator (7) — reading it for either would fail closure,
//! which is the same trap `kimi_mla_closure` sets one rung down.

use std::path::Path;

use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::plan::plan_system;
use larql_models::inventory::build_inventory;

const HIDDEN: usize = 12;
const LAYERS: usize = 4;
const VOCAB: usize = 23;
const INTER: usize = 16;
/// A width belonging to NEITHER operator: the config's own `head_dim`,
/// which an MLA layer must never read (its widths are the four beside
/// it) and a KDA layer must never read (its width is
/// `linear_attn_config.head_dim`).
const HEAD_DIM: usize = 7;

// KDA.
const K_HEADS: usize = 2;
const K_HEAD_DIM: usize = 3;
const K_WIDTH: usize = K_HEADS * K_HEAD_DIM; // 6
const K_CONV: usize = 3;
const K_GATE_RANK: usize = 4;

// MLA.
const M_HEADS: usize = 2;
const M_LATENT: usize = 5;
const M_NOPE: usize = 3;
const M_ROPE: usize = 2;
const M_V: usize = 4;
const M_Q_HEAD_DIM: usize = M_NOPE + M_ROPE; // 5
const M_CACHE_WIDTH: usize = M_LATENT + M_ROPE; // 7

/// Zero-based: layers 0 and 2 run KDA, layers 1 and 3 run MLA — the
/// checkpoint's own alternation, and a PARTITION of the stack, because
/// this family's declaration is explicit-set-with-complement: every
/// layer is in exactly one set, and the sets choose between two
/// operators rather than between an operator and a default. The config
/// spells them ONE-BASED, as Kimi Linear's own does — and `[2, 4]` over
/// four layers is out of range zero-based, which is what settles the
/// base.
const KDA_LAYERS: [usize; 2] = [0, 2];
const MLA_LAYERS: [usize; 2] = [1, 3];

fn miniature_kimi(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "torch_dtype": "float32",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        // MLA's head count IS the config's attention head count — the
        // decompressed K/V side produces this many heads' worth of
        // output. KDA's head count is declared separately, inside
        // `linear_attn_config`, and the two are different facts.
        "num_attention_heads": M_HEADS,
        "num_key_value_heads": M_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "tie_word_embeddings": false,
        "kv_lora_rank": M_LATENT,
        "qk_nope_head_dim": M_NOPE,
        "qk_rope_head_dim": M_ROPE,
        "v_head_dim": M_V,
        "mla_use_nope": true,
        "linear_attn_config": {
            "num_heads": K_HEADS,
            "head_dim": K_HEAD_DIM,
            "short_conv_kernel_size": K_CONV,
            // One-based, the checkpoint's own convention.
            "kda_layers": KDA_LAYERS.iter().map(|l| l + 1).collect::<Vec<_>>(),
            "full_attn_layers": MLA_LAYERS.iter().map(|l| l + 1).collect::<Vec<_>>(),
        },
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut shard = ShardBuilder::new();
    let mut push = |name: String, shape: &[usize], values: Vec<f32>| {
        shard.push(&name, shape, &values);
    };
    push(
        "model.embed_tokens.weight".into(),
        &[VOCAB, HIDDEN],
        lcg_values(VOCAB * HIDDEN, 1),
    );
    push(
        "model.norm.weight".into(),
        &[HIDDEN],
        norm_values(HIDDEN, 2),
    );
    push(
        "lm_head.weight".into(),
        &[VOCAB, HIDDEN],
        lcg_values(VOCAB * HIDDEN, 3),
    );
    for layer in 0..LAYERS {
        let seed = 400 + layer as u64 * 40;
        let p = format!("model.layers.{layer}.");
        push(
            format!("{p}input_layernorm.weight"),
            &[HIDDEN],
            norm_values(HIDDEN, seed),
        );
        push(
            format!("{p}post_attention_layernorm.weight"),
            &[HIDDEN],
            norm_values(HIDDEN, seed + 1),
        );
        push(
            format!("{p}mlp.gate_proj.weight"),
            &[INTER, HIDDEN],
            lcg_values(INTER * HIDDEN, seed + 2),
        );
        push(
            format!("{p}mlp.up_proj.weight"),
            &[INTER, HIDDEN],
            lcg_values(INTER * HIDDEN, seed + 3),
        );
        push(
            format!("{p}mlp.down_proj.weight"),
            &[HIDDEN, INTER],
            lcg_values(HIDDEN * INTER, seed + 4),
        );
        if KDA_LAYERS.contains(&layer) {
            for (n, s) in [("q", 5u64), ("k", 6), ("v", 7)] {
                push(
                    format!("{p}self_attn.{n}_proj.weight"),
                    &[K_WIDTH, HIDDEN],
                    lcg_values(K_WIDTH * HIDDEN, seed + s),
                );
                push(
                    format!("{p}self_attn.{n}_conv1d.weight"),
                    &[K_WIDTH, 1, K_CONV],
                    lcg_values(K_WIDTH * K_CONV, seed + s + 10),
                );
            }
            push(
                format!("{p}self_attn.o_proj.weight"),
                &[HIDDEN, K_WIDTH],
                lcg_values(HIDDEN * K_WIDTH, seed + 8),
            );
            for (n, s) in [("f", 20u64), ("g", 22)] {
                push(
                    format!("{p}self_attn.{n}_a_proj.weight"),
                    &[K_GATE_RANK, HIDDEN],
                    lcg_values(K_GATE_RANK * HIDDEN, seed + s),
                );
                push(
                    format!("{p}self_attn.{n}_b_proj.weight"),
                    &[K_WIDTH, K_GATE_RANK],
                    lcg_values(K_WIDTH * K_GATE_RANK, seed + s + 1),
                );
            }
            push(
                format!("{p}self_attn.b_proj.weight"),
                &[K_HEADS, HIDDEN],
                lcg_values(K_HEADS * HIDDEN, seed + 24),
            );
            push(
                format!("{p}self_attn.A_log"),
                &[K_HEADS],
                lcg_values(K_HEADS, seed + 25),
            );
            push(
                format!("{p}self_attn.dt_bias"),
                &[K_WIDTH],
                lcg_values(K_WIDTH, seed + 26),
            );
            push(
                format!("{p}self_attn.o_norm.weight"),
                &[K_HEAD_DIM],
                norm_values(K_HEAD_DIM, seed + 27),
            );
        } else if MLA_LAYERS.contains(&layer) {
            push(
                format!("{p}self_attn.q_proj.weight"),
                &[M_HEADS * M_Q_HEAD_DIM, HIDDEN],
                lcg_values(M_HEADS * M_Q_HEAD_DIM * HIDDEN, seed + 5),
            );
            push(
                format!("{p}self_attn.kv_a_proj_with_mqa.weight"),
                &[M_CACHE_WIDTH, HIDDEN],
                lcg_values(M_CACHE_WIDTH * HIDDEN, seed + 6),
            );
            push(
                format!("{p}self_attn.kv_a_layernorm.weight"),
                &[M_LATENT],
                norm_values(M_LATENT, seed + 7),
            );
            push(
                format!("{p}self_attn.kv_b_proj.weight"),
                &[M_HEADS * (M_NOPE + M_V), M_LATENT],
                lcg_values(M_HEADS * (M_NOPE + M_V) * M_LATENT, seed + 8),
            );
            push(
                format!("{p}self_attn.o_proj.weight"),
                &[HIDDEN, M_HEADS * M_V],
                lcg_values(HIDDEN * M_HEADS * M_V, seed + 9),
            );
        } else {
            unreachable!("the two sets partition the stack");
        }
    }
    shard.write(dir);
}

/// The stack admits with all three operators declared per layer — never
/// plain softmax standing in for a KDA or MLA layer.
#[test]
fn a_kda_mla_checkpoint_admits_with_all_three_operators_declared() {
    use crate::format::vindex3::graph::LayerOperator;

    let dir = tempfile::tempdir().unwrap();
    miniature_kimi(dir.path());
    let inventory = build_inventory(dir.path()).unwrap();
    let plan = plan_system(&[("kimi-mini".to_string(), inventory)]);
    let blocking: Vec<String> = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking: {blocking:?}");

    let component = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .expect("target component");
    let operators: Vec<LayerOperator> = component
        .attention
        .as_ref()
        .expect("per-layer table")
        .iter()
        .map(|l| l.operator)
        .collect();
    assert_eq!(
        operators,
        vec![
            LayerOperator::Kda,
            LayerOperator::Mla,
            LayerOperator::Kda,
            LayerOperator::Mla,
        ],
        "the declared sets choose the operator per layer; softmax is not a fallback here"
    );
    assert!(
        operators.iter().all(LayerOperator::has_executor),
        "every operator in this stack is executable since lift 2"
    );

    // F6: the epsilon MLA's latent norm runs at is CARRIED, and it is
    // not the layer's. A container that lost this fact could not be
    // executed from at all — which is the whole point of carrying it.
    let surface = component.execution.as_ref().expect("surface");
    let mla = surface.mla.expect("mla surface");
    assert_eq!(mla.kv_a_norm_eps, Some(1e-6));
    assert!(
        (surface.norm.pre.eps - 1e-5).abs() < 1e-12,
        "the layer's own epsilon is the other value: {:?}",
        surface.norm.pre.eps
    );
}

/// **The stack encodes, declares three continuation species, and
/// executes — with the decode path bitwise-identical to batch prefill.**
#[test]
fn the_kda_mla_stack_executes_and_every_state_species_survives_the_step() {
    use crate::format::vindex3::inspect::inspect_container;
    use crate::format::vindex3::opplan::exec::continuation::{
        plan_continuation_geometry, LayerContinuationGeometry,
    };
    use crate::format::vindex3::opplan::exec::decode::DecodeSession;
    use crate::format::vindex3::opplan::exec::kv::{ContinuationProvider, RowKvState};
    use crate::format::vindex3::opplan::exec::operands::OperandStore;
    use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
    use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};
    use larql_models::inventory::report::RecurrentStateDtype;

    let dir = tempfile::tempdir().unwrap();
    miniature_kimi(dir.path());
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("kimi-mini.vindex3");
    encode_checkpoint(dir.path(), &container).expect("closure holds over all three programs");

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("closure held");
    for (index, layer) in plan.layers.iter().enumerate() {
        if KDA_LAYERS.contains(&index) {
            let LayerAttention::Kda(op) = &layer.attention else {
                panic!("layer {index} must be KDA: {:?}", layer.attention);
            };
            assert_eq!(
                op.gate_rank, K_GATE_RANK,
                "resolved from the bound operands"
            );
        } else if MLA_LAYERS.contains(&index) {
            let LayerAttention::Mla(op) = &layer.attention else {
                panic!("layer {index} must be MLA: {:?}", layer.attention);
            };
            assert_eq!(
                op.kv_a_norm_eps,
                Some(1e-6),
                "the latent norm's epsilon rides the op into the executor"
            );
        } else {
            unreachable!("the two sets partition the stack");
        }
    }

    // TWO state species in one program, neither of them KV — the shape
    // lift 2 exists for, and the shape a KV-only provider must refuse.
    let geometry = plan_continuation_geometry(&plan).expect("every layer is stateable");
    for (index, layer) in geometry.iter().enumerate() {
        if KDA_LAYERS.contains(&index) {
            let LayerContinuationGeometry::Recurrent(state) = layer else {
                panic!("layer {index} keeps a recurrence: {layer:?}");
            };
            assert_eq!(state.buffers.len(), 4);
            assert_eq!(
                state.buffers[0].shape,
                vec![K_HEADS, K_HEAD_DIM, K_HEAD_DIM]
            );
            assert!(state
                .buffers
                .iter()
                .all(|b| b.dtype == RecurrentStateDtype::Float32));
        } else if MLA_LAYERS.contains(&index) {
            assert_eq!(
                layer.latent_kv().map(|l| l.width),
                Some(M_CACHE_WIDTH),
                "one compressed row per position: {layer:?}"
            );
        } else {
            unreachable!("the two sets partition the stack");
        }
        assert!(
            layer.kv().is_none(),
            "no layer in this stack keeps a K/V pair, so the KV-only \
             projection must answer for none of them: {layer:?}"
        );
    }

    let store = OperandStore::open(&container, &inspection).unwrap();
    let prefill = |tokens: &[u32], provider: &mut RowKvState| -> Vec<f32> {
        crate::format::vindex3::opplan::exec::prefill_plan(
            &plan,
            &store,
            tokens,
            &ReferenceBackend,
            provider,
        )
        .expect("the stack executes")
        .logits
        .expect("the fixture carries an output head")
    };
    let fresh = || {
        let mut p = RowKvState::default();
        p.prepare_continuation(&geometry).unwrap();
        p
    };

    let prompt = [3u32, 17, 5, 9, 12, 1];
    let mut provider = fresh();
    let logits = prefill(&prompt, &mut provider);
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "finite logits");

    // Deterministic from a fresh state — state reset is real.
    let mut again = fresh();
    assert_eq!(prefill(&prompt, &mut again), logits);

    // And the decode path lands on the batch answer BITWISE. Nothing
    // else in this file could catch a state region that silently
    // restarted at every step: a KDA layer whose conv window reset, or
    // an MLA layer that attended only to its own position, still
    // produces finite logits of the right length.
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    let mut stepped = None;
    for token in prompt {
        stepped = session.step(token).unwrap().logits;
    }
    assert_eq!(session.position(), prompt.len());
    assert_eq!(
        stepped.expect("head present"),
        logits,
        "the decode step path diverged from batch prefill"
    );
}
