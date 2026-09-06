//! **K3-ATTNRES-1, transition 1 — declare, own, address.**
//!
//! The claim, kept as narrow as the freeze that scoped it: LARQL can
//! read the attention-residual topology from a checkpoint's own
//! declaration, own the exit pair the topology requires as ONE object,
//! and give the four per-layer operands semantic identity — and it
//! refuses to execute what it has thereby represented, by name, through
//! one authority both the report and the executor read.
//!
//! **What this file does NOT claim.** No traversal: nothing here carries
//! a snapshot history, and no arithmetic of `_apply_attn_res` is
//! implemented or checked. There is no oracle yet — a torch
//! transcription of the reference is the next artefact of the rung — so
//! an implementation would have nothing to be judged against, and the
//! honest state is a topology that is represented and refused. It also
//! does not make Kimi-K3 plannable: K3 stays blocked by
//! `self_attn.g_proj` (K3-REP-GATE-1) and its `routed_expert_*` bank
//! (K3-LATENTMOE-1), neither of which is this rung's.
//!
//! # Evidence
//!
//! ```text
//! synthetic   a two-layer dense stack that DECLARES the period and
//!             ships the four site operands per layer plus the exit
//!             pair — the one place closure can be watched holding, and
//!             each refusal made to fire on its own
//! Kimi-K3     REAL headers, in `plan::tests::k3_representable`: the
//!             four `*_res_*` operands on a real KDA layer move from
//!             unaddressed to the four site roles under K3's own
//!             declared `attn_res_block_size` of 12
//! ```
//!
//! The synthetic geometry is deliberately free of coincidences: hidden
//! 64, vocabulary 128, FFN 256, and a block period of 3 that equals no
//! dimension anywhere in the fixture — so a period read out of a shape,
//! or a shape read out of the period, cannot pass by accident.

use crate::format::vindex3::encode::encode_graph;
use crate::format::vindex3::graph::{build_from_inventories, ObjectKind, OperandRole};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};
use crate::format::vindex3::plan::carriage::Carriage;
use crate::format::vindex3::plan::plan_system;
use crate::format::vindex3::plan::tests_support::{custom_artifact, glimmer_shaped_target_with};
use larql_models::config::ResidualTopology;

const HIDDEN: usize = 64;
const LAYERS: usize = 2;
/// The declared snapshot period. Equal to no dimension in this fixture,
/// on purpose.
const BLOCK: usize = 3;
const ARTIFACT: &str = "attn-res-artifact";

/// A dense two-layer stack, with or without the period declared.
fn config(attention_residual: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    if attention_residual {
        config["attn_res_block_size"] = serde_json::json!(BLOCK);
    }
    config
}

type Tensors = Vec<(String, Vec<usize>)>;

/// The ordinary two-norm estate: 9 stack operands per layer, plus
/// embedding, final norm and head.
fn dense_tensors() -> Tensors {
    let mut tensors: Tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![128, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![128, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for (leaf, shape) in [
            ("self_attn.q_proj.weight", vec![64, HIDDEN]),
            ("self_attn.k_proj.weight", vec![16, HIDDEN]),
            ("self_attn.v_proj.weight", vec![16, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, 64]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("mlp.gate_proj.weight", vec![256, HIDDEN]),
            ("mlp.up_proj.weight", vec![256, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, 256]),
        ] {
            tensors.push((format!("{stack}.{leaf}"), shape));
        }
    }
    tensors
}

/// The four site operands per layer, spelled as Kimi-K3 spells them:
/// a `[hidden]` norm and a `[1, hidden]` projection at the attention
/// site and again at the mlp site.
fn site_tensors() -> Tensors {
    let mut tensors = Tensors::new();
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for site in ["self_attention", "mlp"] {
            tensors.push((format!("{stack}.{site}_res_norm.weight"), vec![HIDDEN]));
            tensors.push((format!("{stack}.{site}_res_proj.weight"), vec![1, HIDDEN]));
        }
    }
    tensors
}

/// The exit pair, at the stack's end, in the same two spellings K3
/// writes under its own model prefix.
fn exit_tensors() -> Tensors {
    vec![
        (
            "model.output_attn_res_norm.weight".to_string(),
            vec![HIDDEN],
        ),
        (
            "model.output_attn_res_proj.weight".to_string(),
            vec![1, HIDDEN],
        ),
    ]
}

/// The whole attention-residual estate: stack, sites, exit.
fn full_tensors() -> Tensors {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(exit_tensors());
    tensors
}

/// The encoded container and its plan outcome, sources kept alive.
struct Planned {
    _source: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: crate::format::vindex3::inspect::SystemInspection,
    outcome: OpPlanOutcome,
}

/// Encode through the graph seam rather than the production writer: an
/// attention-residual estate is inadmissible by the topology's own
/// refusal, and the point of these tests is to prove what happens BELOW
/// that refusal — closure, ownership, addressing — so the container has
/// to be constructible.
fn plan(config: serde_json::Value, tensors: Tensors) -> Planned {
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config, &borrowed);
    let named = vec![(ARTIFACT.to_string(), inventory)];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    Planned {
        _source: source,
        container,
        inspection,
        outcome,
    }
}

/// Every object of the built graph, as `(id, tensors bound)`.
fn objects(config: serde_json::Value, tensors: Tensors) -> Vec<(String, usize)> {
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config, &borrowed);
    let built = build_from_inventories(&[(ARTIFACT.to_string(), inventory)]);
    built
        .graph
        .objects
        .iter()
        .map(|o| {
            (
                o.id.clone(),
                o.source_bindings.iter().map(|b| b.tensors).sum(),
            )
        })
        .collect()
}

fn bound(objects: &[(String, usize)], id: &str) -> Option<usize> {
    objects.iter().find(|(o, _)| o == id).map(|(_, n)| *n)
}

/// **The ownership witness, and the sweep it replaces.**
///
/// Under the declaration the exit pair is ONE object binding exactly two
/// tensors, and the component's final norm binds exactly ONE. Both
/// halves are asserted because the defect had two halves: the `[hidden]`
/// exit norm was swallowed by the generic `norm` name fragment, leaving
/// the "single final norm" object holding two tensors, while the
/// `[1, hidden]` projection beside it matched no fragment at all and
/// surfaced as an unplaced group. Byte placement was complete and
/// ownership was wrong in both directions at once.
#[test]
fn the_exit_pair_is_one_object_and_the_final_norm_binds_one_tensor() {
    let placed = objects(config(true), full_tensors());
    assert_eq!(
        bound(&placed, "target.attention_residual_exit"),
        Some(2),
        "the exit is one object binding its norm and its projection: {placed:?}"
    );
    assert_eq!(
        bound(&placed, "target.final_norm"),
        Some(1),
        "the swept norm has left the final-norm object: {placed:?}"
    );

    // The baseline the second assertion is measured against: the same
    // stack with no exit pair at all also binds one. Without this arm a
    // final norm that had lost its own tensor would read as a pass.
    let plain = objects(config(false), dense_tensors());
    assert_eq!(bound(&plain, "target.final_norm"), Some(1));
    assert_eq!(bound(&plain, "target.attention_residual_exit"), None);
}

/// **The control that makes the ownership witness mean something.** The
/// same two names on a component that declares NO period are unplaced,
/// each naming the disagreement — they do not fall back into the final
/// norm, and they are not owned by an object the declaration never
/// licensed. Recognition is not ownership.
#[test]
fn the_exit_pair_without_the_declaration_is_unplaced_and_named() {
    let source = tempfile::tempdir().unwrap();
    let tensors = {
        let mut t = dense_tensors();
        t.extend(exit_tensors());
        t
    };
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(false), &borrowed);
    let built = build_from_inventories(&[(ARTIFACT.to_string(), inventory)]);

    let mut unplaced: Vec<&str> = built
        .unplaced
        .iter()
        .map(|u| u.prefix.as_str())
        .filter(|p| p.contains("output_attn_res"))
        .collect();
    unplaced.sort_unstable();
    assert_eq!(
        unplaced,
        ["model.output_attn_res_norm", "model.output_attn_res_proj"],
        "both halves of the pair are unplaced, not just the projection"
    );
    for group in &built.unplaced {
        if group.prefix.contains("output_attn_res") {
            assert!(
                group.reason.contains("attn_res_block_size"),
                "the refusal names the declaration that is missing: {}",
                group.reason
            );
        }
    }
    // And the final norm is still exactly its own tensor — the exit norm
    // did not silently fall back into it.
    let final_norm = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::FinalNorm)
        .expect("the stack has a final norm");
    assert_eq!(
        final_norm
            .source_bindings
            .iter()
            .map(|b| b.tensors)
            .sum::<usize>(),
        1
    );
}

/// **The positive addressing witness.** A component that declares the
/// period and ships every operand CLOSES: the four site operands per
/// layer classify, are required, are checked at `[hidden]` and
/// `[1, hidden]`, and the exit object's two tensors pass their own
/// closure. The plan carries the topology with the declared period
/// inside it.
#[test]
fn a_declared_period_with_every_operand_closes_and_carries_the_topology() {
    let planned = plan(config(true), full_tensors());
    assert!(planned.outcome.closed(), "{:?}", planned.outcome.defects);
    let plan = planned.outcome.plan.as_ref().unwrap();

    assert_eq!(
        plan.residual_topology,
        ResidualTopology::AttentionResidual { block_size: BLOCK }
    );
    assert_eq!(plan.layers.len(), LAYERS);
    for layer in &plan.layers {
        // 9 ordinary operands + 4 site operands, all accounted for.
        assert_eq!(layer.operands_accounted, 13, "layer {}", layer.layer);
        assert_eq!(layer.operands_present, 13, "layer {}", layer.layer);
    }
}

/// **The control that makes the addressing witness mean something.** The
/// same four operands on a component declaring no period are refused by
/// name, and the refusal names the operation they imply rather than
/// reporting them as spellings nobody has judged. Without this a
/// vocabulary that classified the four unconditionally would pass the
/// test above — and would have let any checkpoint acquire a residual
/// topology by shipping four tensor names.
#[test]
fn site_operands_without_the_declaration_are_refused_by_name() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let planned = plan(config(false), tensors);
    assert!(planned.outcome.plan.is_none());

    let mut refused: Vec<String> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } if required_primitive.contains("attention-residual sites") => Some(tensor.clone()),
            _ => None,
        })
        .collect();
    refused.sort();
    assert_eq!(
        refused,
        [
            "0.mlp_res_norm.weight",
            "0.mlp_res_proj.weight",
            "0.self_attention_res_norm.weight",
            "0.self_attention_res_proj.weight",
            "1.mlp_res_norm.weight",
            "1.mlp_res_proj.weight",
            "1.self_attention_res_norm.weight",
            "1.self_attention_res_proj.weight",
        ],
        "{:?}",
        planned.outcome.defects
    );
    // And none of them was reported merely as an unrecognised spelling:
    // the two readings send a reader to different places.
    assert!(
        !planned.outcome.defects.iter().any(
            |d| matches!(d, ClosureDefect::UnclassifiedOperand { tensor, .. }
                if tensor.contains("_res_"))
        ),
        "{:?}",
        planned.outcome.defects
    );
}

/// A layer missing one of its four operands is a named closure defect,
/// not a layer that quietly runs one site. Closure requires all four on
/// every transformer layer, and the role is named so a reader knows
/// which factor of which site's score vector is absent.
#[test]
fn a_layer_missing_one_site_operand_names_the_role() {
    let dropped = "model.layers.1.mlp_res_proj.weight";
    let tensors: Tensors = full_tensors()
        .into_iter()
        .filter(|(name, _)| name != dropped)
        .collect();
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    assert!(
        planned.outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                layer: 1,
                role: OperandRole::AttnResMlpProj
            }
        )),
        "{:?}",
        planned.outcome.defects
    );
}

/// The pair's asymmetry is the contract, and swapping it refuses.
///
/// **Caught by ONE half, and the reason is worth stating.**
/// [`shape_satisfies`](super::super::build::shape_satisfies) treats a
/// one-dimensional contract as satisfied by any shape holding the same
/// values in the same order, so `[1, hidden]` stored under the norm's
/// name does satisfy `[hidden]` — deliberately, because a broadcast
/// spelling of a vector is that vector. The projection's contract is
/// two-dimensional and has no such latitude: `[hidden]` under its name
/// fails.
///
/// So the pair is checkable precisely BECAUSE its two halves have
/// different ranks. Had the reference stored the projection as a bare
/// `[hidden]` vector — the same numbers — a swap would satisfy both
/// contracts and this test could not exist. That is a fact about what
/// this transition can prove, and it belongs on the record beside the
/// part that works: geometry separates these two operands, and nothing
/// here proves an executor would read them in the right order. Only the
/// oracle can.
#[test]
fn a_swapped_pair_refuses_at_the_projection_s_own_contract() {
    let tensors: Tensors = full_tensors()
        .into_iter()
        .map(|(name, shape)| {
            if name == "model.layers.0.self_attention_res_norm.weight" {
                (name, vec![1, HIDDEN])
            } else if name == "model.layers.0.self_attention_res_proj.weight" {
                (name, vec![HIDDEN])
            } else {
                (name, shape)
            }
        })
        .collect();
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    let mismatched: Vec<(&str, &[usize])> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::GeometryMismatch {
                tensor, expected, ..
            } => Some((tensor.as_str(), expected.as_slice())),
            _ => None,
        })
        .collect();
    assert_eq!(
        mismatched,
        [(
            "target.decoder_stack/0.self_attention_res_proj.weight",
            [1, HIDDEN].as_slice()
        )],
        "the projection's two-dimensional contract is what catches the swap: {:?}",
        planned.outcome.defects
    );
}

/// **The exit is required by the declaration.** A component that
/// declares the period and ships no exit pair keeps a blocking
/// execution-surface finding naming the object, because the stack's last
/// layer leaves a prefix sum and a history that something has to
/// collapse before the final norm — and this build does not invent one.
/// The analogue of the hyper-connection head's case for the bundle.
#[test]
fn a_declaration_with_no_exit_pair_blocks_by_the_exit_s_own_name() {
    let source = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let system = plan_system(&[(ARTIFACT.to_string(), inventory)]);
    assert!(!system.admissible);

    let blockers: Vec<_> = system
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .collect();
    assert_eq!(blockers.len(), 1, "{blockers:?}");
    assert!(
        blockers[0].subject.ends_with("execution_surface")
            && blockers[0].detail.contains("attention_residual_exit"),
        "{:?}",
        blockers[0]
    );
}

/// **One authority, two readers — and at the lift both stop refusing
/// together.**
///
/// Through K3-ATTNRES-1's first transition this test asserted the
/// mirror image: the report called a complete attention-residual surface
/// not executable and the executor refused the same plan, both reading
/// `ResidualTopology::unimplemented_reason`. That function is now
/// deleted, along with both readers, because the decode traversal (2a)
/// and the batch traversal (2b) were each witnessed against a Torch
/// oracle transcribed from the reference.
///
/// So the test is inverted rather than removed, and it is the same
/// claim from the other side: what one reader admits, the other must
/// prepare AND run. A lift that moved the report without the executor
/// would be exactly the drift the single-authority rule exists to
/// prevent, and it would be invisible to a test that only checked the
/// report.
///
/// Two arms keep this from having been implemented as "stop refusing
/// attention residuals":
///
///   - the SINGLE-STREAM control, unchanged from the refusing version:
///     the same estate minus the topology still executes, so what is
///     being witnessed is the topology and not a fixture that got
///     easier.
///   - the EXIT-PAIR arm, in the test immediately above this one: the
///     same estate without an `attention_residual_exit` object is still
///     blocked by that object's absence. Capability was granted to a
///     traversal, not to the declaration.
#[test]
fn the_report_admits_the_topology_and_the_executor_runs_it() {
    let source = tempfile::tempdir().unwrap();
    let tensors = full_tensors();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let system = plan_system(&[(ARTIFACT.to_string(), inventory)]);
    let blockers: Vec<_> = system
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .collect();
    assert!(
        blockers.is_empty(),
        "a complete attention-residual surface must no longer block: {blockers:?}"
    );
    assert!(system.admissible, "{:?}", system.summary);

    // The executor, on the very same facts: it prepares, and it runs the
    // whole stack rather than merely accepting the plan.
    let planned = plan(config(true), full_tensors());
    let attn_res_plan = planned.outcome.plan.as_ref().unwrap();
    assert!(matches!(
        attn_res_plan.residual_topology,
        ResidualTopology::AttentionResidual { .. }
    ));
    let store = OperandStore::open(planned.container.path(), &planned.inspection).unwrap();
    execute_text(attn_res_plan, &store, &[1, 2, 3])
        .expect("an attention-residual plan prepares and executes");

    // The control: one stream, the same estate minus the topology and
    // its operands, executes.
    let single = plan(config(false), dense_tensors());
    let single_plan = single.outcome.plan.as_ref().unwrap();
    assert_eq!(
        single_plan.residual_topology,
        ResidualTopology::SingleStream
    );
    let store = OperandStore::open(single.container.path(), &single.inspection).unwrap();
    execute_text(single_plan, &store, &[1, 2, 3]).expect("a single-stream plan executes");
}

/// **A component runs ONE residual programme.** A checkpoint declaring
/// the attention-residual period BESIDE the Sinkhorn keys is refused,
/// because reading either would discard what the other declares — the
/// same rule that refuses a half-declared Sinkhorn rather than
/// completing it with one stream, applied to a whole second topology.
///
/// The positive arm is what keeps the refusal from having been
/// implemented as "refuse anything that declares a period": the period
/// alone resolves, and it resolves to the value the checkpoint wrote.
#[test]
fn a_period_declared_beside_the_sinkhorn_keys_refuses_rather_than_choosing() {
    let surface_of = |mutate: fn(&mut serde_json::Value)| {
        let dir = tempfile::tempdir().unwrap();
        let inventory = glimmer_shaped_target_with(dir.path(), mutate);
        let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
        let surface = built
            .graph
            .components
            .iter()
            .find(|c| c.id == "target")
            .and_then(|c| c.execution.clone());
        (surface, built.incomplete_surfaces)
    };

    // The period alone resolves, to the declared value.
    let (declared, _) = surface_of(|config| {
        config["text_config"]["attn_res_block_size"] = serde_json::json!(BLOCK);
    });
    assert_eq!(
        declared.expect("the surface builds").residual_topology,
        ResidualTopology::AttentionResidual { block_size: BLOCK }
    );

    // Both declared: neither is chosen, and the refusal says why.
    let (surface, incomplete) = surface_of(|config| {
        config["text_config"]["attn_res_block_size"] = serde_json::json!(BLOCK);
        config["text_config"]["hc_mult"] = serde_json::json!(4);
        config["text_config"]["hc_sinkhorn_iters"] = serde_json::json!(20);
        config["text_config"]["hc_eps"] = serde_json::json!(1e-6);
    });
    assert!(
        surface.is_none(),
        "a two-topology declaration must not build"
    );
    let reason = incomplete
        .iter()
        .find(|s| s.component == "target")
        .and_then(|s| s.missing.iter().find(|m| m.contains("residual topology")))
        .unwrap_or_else(|| panic!("{incomplete:?}"));
    assert!(reason.contains("attn_res_block_size"), "{reason}");
    assert!(reason.contains("hc_mult"), "{reason}");
    // Not "incomplete": both halves are complete declarations of
    // different things, and calling it partial would send a reader to
    // finish something nothing is missing from.
    assert!(reason.contains("ONE residual programme"), "{reason}");
}

/// **Carriage is attested from the BUILT graph, never from the
/// declaration.** A component whose surface does not build cannot answer
/// the period's probe, so the key stays a blocker there however
/// faithfully the parser read it.
///
/// This is wave 19's falsified P8 applied as a rule rather than
/// repeated: DeepSeek-V4's three topology keys changed CLASS and not
/// COUNT, because their probe reads a surface that never builds. The
/// estate here is the same shape — a period declared beside the Sinkhorn
/// keys, so the surface refuses — and the paired arm is a component
/// whose surface does build, where the same key is carried.
#[test]
fn the_period_is_carried_only_where_the_surface_actually_builds() {
    let finding_for = |mutate: fn(&mut serde_json::Value)| {
        let dir = tempfile::tempdir().unwrap();
        let inventory = glimmer_shaped_target_with(dir.path(), mutate);
        let plan = plan_system(&[("target-artifact".to_string(), inventory)]);
        plan.artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .find(|f| f.subject.ends_with("attn_res_block_size"))
            .cloned()
            .expect("the declared key is reported")
    };

    let builds = finding_for(|config| {
        config["text_config"]["attn_res_block_size"] = serde_json::json!(BLOCK);
    });
    assert!(!builds.blocks(), "{builds:?}");
    assert_eq!(builds.resolved, Some(serde_json::json!(BLOCK)));
    // **`Lowered` since the lift, and the stage is asserted rather than
    // matched as a substring of the prose.** The period now reaches a
    // traversal that reads it, so `Represented` would understate what
    // this build does with the key.
    //
    // The key did NOT change whether it blocks, and that half is scored
    // here too: it was already Representable/ExecutionSemantic and
    // non-blocking at `Represented`, so a COUNT that moved when the
    // stage did would mean the stage name was doing work it should not.
    assert_eq!(builds.carriage, Some(Carriage::Lowered), "{builds:?}");

    let refuses = finding_for(|config| {
        config["text_config"]["attn_res_block_size"] = serde_json::json!(BLOCK);
        config["text_config"]["hc_mult"] = serde_json::json!(4);
    });
    assert!(
        refuses.blocks(),
        "a key whose component has no surface must not read as carried: {refuses:?}"
    );
    assert!(refuses.resolved.is_none(), "{refuses:?}");
    assert!(
        refuses
            .detail
            .contains("no built component answered the probe"),
        "{refuses:?}"
    );
}
