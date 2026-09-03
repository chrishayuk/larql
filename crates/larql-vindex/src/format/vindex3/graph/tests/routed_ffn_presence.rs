//! An FFN is present when the component declares a width for one — and a
//! wholly-routed family declares it in the ROUTED spelling.
//!
//! `Qwen3_5MoeTextConfig` is `@strict` and has no `intermediate_size`
//! field at all: every one of its layers is a `Qwen3_5MoeSparseMoeBlock`,
//! so there is no dense FFN to size. Reading only the dense spelling
//! graded a 397B mixture of experts as running no FFN op, which sent both
//! `hidden_act` and `num_experts_per_tok` to "no built component answered
//! the probe" — the routed block was declared, judged, and then had
//! nowhere to be read back from.
//!
//! The falsifier below is the case the old rule got RIGHT: a component
//! declaring neither width still has no FFN, so this is a wider reading
//! of "declares a width", not a decision to always write one.

use crate::format::vindex3::graph::build_from_inventories;
use crate::format::vindex3::plan::tests_support::glimmer_shaped_target_with;

/// The routed width this fixture declares, and a shared-branch width that
/// is NOT a small multiple of it — so a derivation cannot pass for a
/// reading of the declared key.
const ROUTED_WIDTH: usize = 32;
const DECLARED_SHARED_WIDTH: usize = ROUTED_WIDTH * 4 + 7;

/// The fixture as a mixture of experts: routed experts declared, and the
/// dense width withdrawn, which is exactly what `Qwen3_5MoeTextConfig`
/// does — it has no `intermediate_size` field to declare.
fn routed_moe(config: &mut serde_json::Value) {
    let text = &mut config["text_config"];
    text["num_experts"] = serde_json::json!(8);
    text["num_experts_per_tok"] = serde_json::json!(2);
    text["moe_intermediate_size"] = serde_json::json!(ROUTED_WIDTH);
    text.as_object_mut()
        .expect("text config")
        .remove("intermediate_size");
}

fn surface_of(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Option<crate::format::vindex3::graph::surface::ExecutionSurface> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), mutate);
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.execution.clone())
}

#[test]
fn a_wholly_routed_component_has_an_ffn_sized_by_its_experts() {
    let surface = surface_of(routed_moe).expect("the component builds");
    let ffn = surface
        .ffn
        .as_ref()
        .expect("every layer runs a routed FFN, so the component runs an FFN");
    // Absence is stated, not written as a width of zero: nothing
    // downstream may take a zero here for a dense FFN's size.
    assert_eq!(ffn.intermediate_size, None);
    let moe = ffn.moe.expect("the routed judgment survives");
    assert_eq!(moe.expert_intermediate_size, ROUTED_WIDTH);
}

/// The rule this widened, still holding: neither width declared means no
/// FFN op to describe (the mixer-only case).
#[test]
fn a_component_declaring_neither_width_still_has_no_ffn() {
    let surface = surface_of(|config| {
        let text = config["text_config"].as_object_mut().expect("text config");
        text.remove("intermediate_size");
    })
    .expect("the component builds");
    assert!(surface.ffn.is_none(), "{:?}", surface.ffn);
}

/// The shared branch's width reaches the surface as declared, and is not
/// recomputed from the routed width on the way.
#[test]
fn the_declared_shared_expert_width_reaches_the_surface() {
    let surface = surface_of(|config| {
        routed_moe(config);
        config["text_config"]["n_shared_experts"] = serde_json::json!(1);
        config["text_config"]["shared_expert_intermediate_size"] =
            serde_json::json!(DECLARED_SHARED_WIDTH);
    })
    .expect("the component builds");
    let moe = surface
        .ffn
        .as_ref()
        .expect("the fixture runs an FFN")
        .moe
        .expect("the fixture declares experts");
    assert_eq!(moe.shared_experts, 1);
    assert_eq!(
        moe.shared_expert_intermediate_size,
        Some(DECLARED_SHARED_WIDTH)
    );
    assert_ne!(
        moe.shared_expert_intermediate_size,
        Some(moe.expert_intermediate_size * moe.shared_experts)
    );
}

/// A shared branch declared only by COUNT keeps the DeepSeek/Kimi
/// convention — one wider FFN — so widening the field did not quietly
/// unsize the lineage that has no key for it.
#[test]
fn a_count_only_shared_branch_keeps_the_one_wider_ffn_convention() {
    let surface = surface_of(|config| {
        routed_moe(config);
        config["text_config"]["n_shared_experts"] = serde_json::json!(2);
    })
    .expect("the component builds");
    let moe = surface
        .ffn
        .as_ref()
        .expect("the fixture runs an FFN")
        .moe
        .expect("the fixture declares experts");
    assert_eq!(moe.shared_expert_intermediate_size, Some(ROUTED_WIDTH * 2));
}
