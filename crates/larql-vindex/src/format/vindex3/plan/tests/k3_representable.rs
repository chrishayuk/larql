//! **K3 REPRESENTABLE probe — can REPRESENT describe one K3 KDA layer?**
//!
//! The capability matrix's second cell, asked as narrowly as it can be:
//!
//! > Can a truthful representation action be constructed for ONE K3 KDA
//! > layer, BF16 -> Q8_0, from K3's own descriptor and graph semantics?
//!
//! Not "does K3 support quantization". Not "can Kimi-Linear do KDA".
//!
//! # Weight-free, and honest about how
//!
//! Two planes have to be witnessed here and they fail independently: what
//! the CONFIG declares, and whether the tensor estate the checkpoint
//! actually spells is ADDRESSABLE. Declaration passing tells you nothing
//! about the second — that is the silent middle plane.
//!
//! So both inputs are real:
//!
//! ```text
//! config   moonshotai/Kimi-K3's public `config.json`, including its
//!          actual `linear_attn_config` layer lists, so family
//!          membership is config-derived and never assumed
//!
//! tensors  the VERBATIM safetensors headers of two shards, exported by
//!          `scripts/k3_header_fixture_export.py` over HTTP range
//!          requests — names, dtypes, shapes and byte counts as the
//!          checkpoint writes them
//! ```
//!
//! **The two reductions, and nothing may claim otherwise:**
//!
//! ```text
//! layers   2 of 93 — tensor layer 0 (KDA) and tensor layer 3 (MLA,
//!          config layer 4). Every name and shape within them is exact.
//!
//! payload  none. `header_only_shards` writes the length prefix and the
//!          header and stops; K3 is 1.56 TB and no byte of it is read.
//!          Byte COUNTS are still real — they come from `data_offsets`.
//! ```
//!
//! # The negative discriminator
//!
//! A positive case alone would pass if the planner classified every layer
//! as KDA. Layer 3 is full-attention in K3's own `full_attn_layers`, so it
//! must NOT be discovered as KDA — that is what makes the family logic a
//! yes/no witness rather than a yes.
//!
//! The two layers are a genuine trap and not a formality: they share
//! `self_attn.g_proj` at `[12288, 7168]` and `self_attn.o_proj` at
//! `[7168, 12288]`, **identical name and identical shape**. Anything
//! keying on those alone classifies layer 3 as KDA and fails here.

use super::support::header_only_shards;
use crate::format::vindex3::graph::roles::classify_stack_tensor_under;
use crate::format::vindex3::graph::{
    build_from_inventories, most_specific_owner, ComponentRole, LayerOperator, OperandRole,
};
use crate::format::vindex3::plan::{plan_system, PlannedFinding};
use larql_models::inventory::ArchitectureInventory;

/// K3's `linear_attn_config` layer lists, VERBATIM from the public
/// config and 1-indexed as the checkpoint writes them.
///
/// Held as arrays rather than inside the `json!` literal because 93
/// elements exceed that macro's recursion limit — and NOT regenerated
/// from a stride: the `..., 88, 92, 93` tail is irregular, so any
/// periodic reconstruction is right only by luck.
const FULL_ATTN_LAYERS: [u64; 24] = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93,
];
const KDA_LAYERS: [u64; 69] = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33,
    34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65,
    66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89, 90, 91,
];

fn layers(list: &[u64]) -> serde_json::Value {
    serde_json::Value::Array(list.iter().map(|l| serde_json::json!(l)).collect())
}

/// Hidden size and the KDA projection width, from the public config.
const HIDDEN: usize = 7168;
const KDA_PROJ: usize = 12288;

/// K3's real text config, trimmed to the fields the planner reads.
///
/// `kda_layers` and `full_attn_layers` are verbatim: 69 + 24 = 93, with
/// the irregular `..., 88, 92, 93` tail that no stride reproduces.
fn k3_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "dtype": "bfloat16",
        "text_config": {
            "model_type": "kimi_linear",
            "dtype": "bfloat16",
            "num_hidden_layers": 93,
            "hidden_size": HIDDEN,
            "intermediate_size": 33792,
            "moe_intermediate_size": 3072,
            "routed_expert_hidden_size": 3584,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "first_k_dense_replace": 1,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "vocab_size": 163840,
            "rms_norm_eps": 1e-5,
            "attn_res_block_size": 12,
            "linear_attn_config": {
                "full_attn_layers": layers(&FULL_ATTN_LAYERS),
                "kda_layers": layers(&KDA_LAYERS),
                "head_dim": 128,
                "num_heads": 96,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": true
            }
        }
    })
}

/// The two real shard headers, exported header-only from the public
/// checkpoint. Committed rather than fetched at test time: a fixture that
/// needs the network is a test that is absent whenever the network is.
const HEADERS: &str = include_str!("fixtures/k3_two_layer_headers.json");

/// Tensor layer carrying KDA, and the one carrying MLA — as the CONFIG
/// classifies them, 0-indexed into the tensor names.
///
/// Config layer 1 is KDA and config layer 4 is full-attention; the config
/// lists are 1-indexed, the tensor names 0-indexed.
/// The shard carrying the KDA layer, and the tensor-name prefix K3 puts
/// in front of every one of its layers — a multimodal container's text
/// tower, not a bare decoder.
const KDA_SHARD: &str = "model-00001-of-000096.safetensors";
const KDA_SHARD_PREFIX: &str = "language_model.model";

const KDA_LAYER: usize = 0;
const MLA_LAYER: usize = 3;

/// K3's real estate, built through the real inventory pipeline.
fn k3_inventory(dir: &std::path::Path) -> ArchitectureInventory {
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    let shards = fixture["shards"].as_object().unwrap();
    header_only_shards(dir, &k3_config(), shards)
}

/// The fixture carries what it claims to carry.
///
/// Without this the probe below could pass on an empty estate: a planner
/// that resolves everything from the config alone reports a complete
/// surface whether or not one tensor was ever addressed, and the failure
/// would be silent. This is the check that makes `KDA_PROJ` load-bearing
/// — the width is asserted against the header, so a fixture regenerated
/// against a different checkpoint cannot quietly pass.
#[test]
fn k3_fixture_carries_the_real_two_layer_estate() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = k3_inventory(dir.path());

    let shape_of = |name: &str| -> Vec<usize> {
        inventory
            .tensors
            .tensors
            .iter()
            .find(|t| t.name.ends_with(name))
            .unwrap_or_else(|| panic!("fixture has no tensor ending `{name}`"))
            .shape
            .clone()
    };

    // KDA layer 0: the recurrence's own operands, at K3's real widths.
    assert_eq!(
        shape_of("layers.0.self_attn.q_proj.weight"),
        [KDA_PROJ, HIDDEN]
    );
    assert_eq!(
        shape_of("layers.0.self_attn.k_proj.weight"),
        [KDA_PROJ, HIDDEN]
    );
    assert_eq!(
        shape_of("layers.0.self_attn.v_proj.weight"),
        [KDA_PROJ, HIDDEN]
    );
    assert_eq!(
        shape_of("layers.0.self_attn.o_proj.weight"),
        [HIDDEN, KDA_PROJ]
    );
    assert_eq!(shape_of("layers.0.self_attn.dt_bias"), [KDA_PROJ]);
    assert_eq!(
        shape_of("layers.0.self_attn.q_conv1d.weight"),
        [KDA_PROJ, 1, 4]
    );

    // MLA layer 3: the latent pair, which the KDA layer does not have.
    assert_eq!(
        shape_of("layers.3.self_attn.kv_a_proj_with_mqa.weight"),
        [576, HIDDEN]
    );
    assert_eq!(
        shape_of("layers.3.self_attn.kv_b_proj.weight"),
        [24576, 512]
    );

    // The trap: shared spelling AND shared shape across the two layers.
    // If this ever stops holding, the negative discriminator below has
    // become easy and no longer proves what it claims.
    for shared in ["self_attn.g_proj.weight", "self_attn.o_proj.weight"] {
        assert_eq!(
            shape_of(&format!("layers.{KDA_LAYER}.{shared}")),
            shape_of(&format!("layers.{MLA_LAYER}.{shared}")),
            "`{shared}` is supposed to collide across the two layers"
        );
    }
}

/// The per-layer operator the graph builder discovered, by tensor layer.
fn discovered_operators(inventory: ArchitectureInventory) -> Vec<LayerOperator> {
    let built = build_from_inventories(&[("k3".to_string(), inventory)]);
    let text = built
        .graph
        .components
        .iter()
        .find(|c| c.role != ComponentRole::Perception)
        .expect("K3 has a text component");
    text.attention
        .as_ref()
        .expect("the text component resolves a per-layer attention table")
        .iter()
        .map(|policy| policy.operator)
        .collect()
}

/// **The two-sided witness.** Layer 0 is KDA; layer 3 is not.
///
/// Both halves are required. The positive alone passes on a builder that
/// answers KDA for everything, which is exactly the mistake the shared
/// `g_proj`/`o_proj` spelling invites.
#[test]
fn k3_kda_layer_is_discovered_and_the_mla_layer_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let operators = discovered_operators(k3_inventory(dir.path()));

    assert_eq!(
        operators.len(),
        93,
        "the config declares 93 layers; the table must cover all of them"
    );
    assert_eq!(
        operators[KDA_LAYER],
        LayerOperator::Kda,
        "config layer 1 is in `kda_layers`"
    );
    assert_ne!(
        operators[MLA_LAYER],
        LayerOperator::Kda,
        "config layer 4 is in `full_attn_layers` — classifying it KDA is \
         the failure the shared g_proj/o_proj spelling invites"
    );
}

/// Every real tensor, paired with the role the operand vocabulary gives
/// it — through the graph's own membership and prefix-strip rules, and
/// the layer's own discovered operator. This is what `opplan` does; here
/// it needs no bytes, because the classifier is a pure function of the
/// name and the operator.
fn classified(inventory: ArchitectureInventory) -> Vec<(String, Option<OperandRole>)> {
    let names: Vec<String> = inventory
        .tensors
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let built = build_from_inventories(&[("k3".to_string(), inventory)]);
    let text = built
        .graph
        .components
        .iter()
        .find(|c| c.role != ComponentRole::Perception)
        .expect("K3 has a text component");
    let table = text
        .attention
        .clone()
        .expect("K3 resolves a per-layer attention table");
    // The component's own declared residual topology, not a default: the
    // four `*_res_*` operands are site operands exactly when K3 declares
    // `attn_res_block_size`, and asking with a single-stream default
    // would report them unaddressed on a checkpoint that declares 12.
    let topology = text
        .execution
        .as_ref()
        .map(|e| e.residual_topology)
        .expect("K3's text surface builds");

    names
        .iter()
        .map(|name| {
            let Some(object) = most_specific_owner(&built.graph.objects, name) else {
                return (name.clone(), None);
            };
            let prefix = object
                .source_bindings
                .iter()
                .filter(|b| b.covers(name))
                .map(|b| b.tensor_prefix.clone())
                .max_by_key(String::len)
                .expect("the owner covers the name");
            let relative = name
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_prefix('.'))
                .unwrap_or(name)
                .to_string();
            let operator = relative
                .split_once('.')
                .and_then(|(index, _)| index.parse::<usize>().ok())
                .and_then(|layer| table.get(layer))
                .map_or(LayerOperator::Softmax, |policy| policy.operator);
            (
                name.clone(),
                classify_stack_tensor_under(&relative, operator, topology).map(|(_, role)| role),
            )
        })
        .collect()
}

/// The KDA layer's operands that the role vocabulary DOES name — K3's
/// recurrence, addressed at its real spellings.
const KDA_ROLES_ADDRESSED: [OperandRole; 13] = [
    OperandRole::KdaQProj,
    OperandRole::KdaKProj,
    OperandRole::KdaVProj,
    OperandRole::KdaOutProj,
    OperandRole::KdaQConv1d,
    OperandRole::KdaKConv1d,
    OperandRole::KdaVConv1d,
    OperandRole::KdaFAProj,
    OperandRole::KdaFBProj,
    OperandRole::KdaBProj,
    OperandRole::KdaALog,
    OperandRole::KdaDtBias,
    OperandRole::KdaONorm,
];

/// The KDA layer's operand the vocabulary does NOT name, exactly.
///
/// One left, and it is a K3 delta with a config twin rather than a shape
/// problem:
///
/// ```text
/// self_attn.g_proj        the FULL-RANK output gate. The table carries
///                         `g_a_proj`/`g_b_proj`, Kimi-Linear's low-rank
///                         pair. Twin of the config blocker
///                         `use_full_rank_gate` — the same fact refused
///                         on both planes, which is the agreement that
///                         makes it a real gap and not a parser slip.
///                         K3-REP-GATE-1, not started.
/// ```
///
/// # This list was a cross-programme tripwire, and it fired twice
///
/// **Wave 18, the "premise was wrong" firing.** Before it, this comment
/// called the four `*_res_*` entries hyper-connection operands that wave
/// 18's generic `hc_*` roles would address, and laid out how to read the
/// failure it expected. Wave 18 landed (2026-09-05), this witness was
/// rerun UNCHANGED, and it **passed**: zero of the four moved. That is
/// the "fewer than 4 gone" arm — and it was not accommodation, because
/// no K3-shaped name entered wave 18's vocabulary. The abstraction had
/// nothing to transfer to.
///
/// The shapes settled it. A Sinkhorn hyper-connection site's mix
/// projection is `[(2 + hc)·hc, hc·hidden]`, which equals `[1, 7168]` for
/// NO stream count (hc = 1 gives `[3, 7168]`); its base is
/// `[(2 + hc)·hc]`, never `[hidden]`. Pinned both ways in
/// `opplan::tests::wave18_hc_carriage::k3s_residual_operands_are_not_sinkhorn_sites_under_any_stream_count`,
/// which this rung leaves untouched: it says what these operands are
/// NOT, and that is still true.
///
/// **K3-ATTNRES-1, the intended firing.** The four moved when the third
/// residual topology was given its own name, its own roles and its own
/// declaration — `attn_res_block_size`, which K3 declares as 12 and this
/// fixture's config carries verbatim. The `[hidden]` norm and the
/// `[1, hidden]` projection at each of a layer's two sites are the two
/// factors of one learned score vector, and they classify HERE and only
/// here: the operator-only classifier still answers nothing for them, so
/// a checkpoint shipping the spellings without the period gains no
/// topology from its tensor names.
///
/// 5 -> 1, and the one that stays is its own rung. `K3-LATENTMOE-1`
/// (`routed_expert_hidden_size` ↔ the `routed_expert_*` operands) is off
/// this layer and unaffected.
const KDA_LAYER_UNADDRESSED: [&str; 1] = ["self_attn.g_proj.weight"];

/// **The tensor-address witness.** Which of K3's real operands does the
/// role vocabulary actually name?
///
/// Asked HERE and not through [`plan_system`], because the plan stage
/// cannot answer it — see
/// [`the_plan_stage_places_bytes_and_classifies_no_operand`].
///
/// The result is pinned both ways. Pinning only the unaddressed set would
/// pass if the vocabulary lost every KDA role tomorrow; pinning only the
/// addressed set would pass if a later change quietly began naming
/// `g_proj` something wrong. The gap is a fact about K3 and belongs in
/// the record with the same weight as the part that works.
#[test]
fn k3_kda_operands_are_addressed_except_the_one_named_delta() {
    let dir = tempfile::tempdir().unwrap();
    let rows = classified(k3_inventory(dir.path()));

    let layer_prefix = format!("{KDA_SHARD_PREFIX}.layers.{KDA_LAYER}.");
    let on_kda_layer = |name: &String| name.starts_with(&layer_prefix);

    let mut addressed: Vec<OperandRole> = rows
        .iter()
        .filter(|(name, _)| on_kda_layer(name) && name.contains(".self_attn."))
        .filter_map(|(_, role)| *role)
        .collect();
    addressed.sort_unstable();
    let mut expected = KDA_ROLES_ADDRESSED;
    expected.sort_unstable();
    assert_eq!(
        addressed, expected,
        "the KDA recurrence's addressed operand set changed"
    );

    let mut unaddressed: Vec<&str> = rows
        .iter()
        .filter(|(name, role)| on_kda_layer(name) && role.is_none())
        .map(|(name, _)| name.strip_prefix(&layer_prefix).unwrap())
        .collect();
    unaddressed.sort_unstable();
    assert_eq!(
        unaddressed, KDA_LAYER_UNADDRESSED,
        "the KDA layer's unaddressed operands are the named K3 deltas, \
         no more and no fewer"
    );
}

/// The whole estate, reported and counted.
///
/// The count is what keeps the pin above honest about scope: 5,376 of the
/// unclassified are the 896-way expert bank repeating six MXFP4 spellings
/// (`weight_packed`/`weight_scale`, `compressed-tensors`), which is its
/// own rung and not the KDA layer's problem.
///
/// K3-ATTNRES-1 moved eight of these: the four `*_res_*` operands on each
/// of the two layers now classify to the attention-residual site roles,
/// under K3's own declared `attn_res_block_size`. 5,392 -> 5,384, and
/// eight distinct spellings out of the list.
///
/// Deliberately OUT of the KDA-Q8 critical path, exposed by the fixture
/// and left alone on purpose — implementing what a fixture happens to
/// reveal is how a capability cell turns back into broad architectural
/// support:
///
/// ```text
/// self_attn.q_a_proj / q_b_proj / q_a_layernorm
///     K3 factorises MLA's Q where Kimi-Linear did not. MLA is a
///     DIFFERENT capability-matrix cell; it gets its own.
///
/// block_sparse_moe.experts.N.w{1,2,3}.weight_{packed,scale}
///     the compressed-tensors MXFP4 dialect. Exercised when bank reuse
///     or encoding actually asks for it, not before.
/// ```
///
/// These counts move whenever either is addressed. That is intended: the
/// number failing is the notification that scope changed.
#[test]
fn k3_estate_reports_every_unaddressed_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let rows = classified(k3_inventory(dir.path()));

    let mut spellings: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut unclassified = 0usize;
    for (name, role) in &rows {
        if role.is_none() {
            unclassified += 1;
            spellings.insert(regex_free_expert_elide(name));
        }
    }
    eprintln!(
        "=== {} tensors, {unclassified} unclassified ===",
        rows.len()
    );
    for spelling in &spellings {
        eprintln!("  UNADDRESSED  {spelling}");
    }

    assert_eq!(rows.len(), 5421, "the fixture's two real layers");
    assert_eq!(
        unclassified, 5384,
        "5,376 expert-bank operands + 8 distinct dense spellings"
    );
    assert_eq!(
        spellings.len(),
        14,
        "distinct unaddressed spellings across both layers"
    );
}

/// `...experts.417.w1...` -> `...experts.N.w1...`, so 896 identical
/// spellings report once.
fn regex_free_expert_elide(name: &str) -> String {
    let Some((head, rest)) = name.split_once(".experts.") else {
        return name.to_string();
    };
    let tail = rest.split_once('.').map_or("", |(_, t)| t);
    format!("{head}.experts.N.{tail}")
}

/// **The plan stage is not the tensor-address witness.**
///
/// [`k3_representable_probe_reports_what_the_planner_finds`] reports no
/// complaint about a single one of K3's 5,421 tensors. That reads as
/// "every operand was addressed" — but it reads identically to "no
/// operand was ever classified", and those have opposite consequences.
/// This is the control that separates them: an operand no role table can
/// name, injected on the KDA layer, and the plan stage still returns the
/// same two config-plane blockers.
///
/// So plan-stage silence about tensors is **structural, not evidence**.
/// Byte placement is total — every tensor lands in `decoder_stack` or
/// `expert_bank` by prefix — and role classification happens later, in
/// `opplan`. Any REPRESENTABLE verdict that leans on this stage's silence
/// is unearned.
///
/// If this test ever fails because the stray IS refused, the plan stage
/// has gained operand closure: delete the test and move the witness here.
///
/// The third blocker in the expected list is NOT an operand refusal and
/// must not be read as one. It is a component-level object absence: this
/// fixture is two shards of ninety-six, and K3's `output_attn_res_
/// {norm,proj}` pair lives in the model-level tail that neither exports.
/// A component declaring `attn_res_block_size` and shipping no exit
/// object is refused by name, so the slice reports what the slice
/// actually holds. The whole checkpoint ships the pair and is refused
/// one step later instead — for the traversal, which does not exist. A
/// fixture cannot witness a model-level object it does not contain, and
/// pretending otherwise by adding tensors the two shards never held
/// would break the one property that makes this fixture evidence.
#[test]
fn the_plan_stage_places_bytes_and_classifies_no_operand() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    let shards = fixture["shards"].as_object_mut().unwrap();
    let stray = format!("{KDA_SHARD_PREFIX}.layers.{KDA_LAYER}.self_attn.zzz_proj.weight");
    shards[KDA_SHARD].as_object_mut().unwrap().insert(
        stray.clone(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": [KDA_PROJ, HIDDEN],
            "data_offsets": [0, KDA_PROJ * HIDDEN * 2],
        }),
    );
    let inventory = header_only_shards(dir.path(), &k3_config(), shards);

    // The stray really is in the estate, and really is unclassifiable.
    assert!(
        classified(inventory.clone())
            .iter()
            .any(|(name, role)| *name == stray && role.is_none()),
        "the injected operand must be present and classify to nothing"
    );

    let blocking: Vec<String> = plan_system(&[("k3".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .filter(|f| f.blocks())
        .map(|f| f.subject.clone())
        .collect();
    assert_eq!(
        blocking,
        [
            "text_config.linear_attn_config.use_full_rank_gate",
            "text_config.routed_expert_hidden_size",
            // Not an operand refusal: the two-shard slice carries no
            // model-level exit pair for the topology it declares. See
            // this test's own doc.
            "target.execution_surface",
        ],
        "the plan stage refuses on config semantics and component-level \
         objects only — an unclassifiable operand changes nothing here"
    );
}

/// The findings the planner raises for K3's real shape.
fn k3_findings() -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    plan_system(&[("k3".to_string(), k3_inventory(dir.path()))])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// **The probe.** What does the planner actually say about K3?
///
/// Exploratory by design: it reports rather than asserts an outcome, so
/// the first run states what K3 needs instead of what was hoped. The
/// specific expectations get pinned once the answer is known — writing
/// them first would only pin a guess.
#[test]
fn k3_representable_probe_reports_what_the_planner_finds() {
    let findings = k3_findings();
    eprintln!("=== K3 planner findings: {} ===", findings.len());
    for f in &findings {
        eprintln!(
            "  {:?}  {:?}  subject={}  component={}",
            f.category, f.class, f.subject, f.component
        );
        if !f.detail.is_empty() {
            eprintln!("      {}", f.detail.chars().take(220).collect::<String>());
        }
    }
    let blocking: Vec<&PlannedFinding> = findings.iter().filter(|f| f.blocks()).collect();
    eprintln!("=== blocking: {} ===", blocking.len());
    for f in &blocking {
        eprintln!("  BLOCKS  {}  ({:?})", f.subject, f.category);
    }

    // The one thing K3-ARCH-1 already earned: identity must no longer be
    // the blocker. Anything else blocking is the next named rung.
    assert!(
        !blocking
            .iter()
            .any(|f| f.subject == "architecture_identity"),
        "K3-ARCH-1 closed the identity gate; a conflict here is a regression"
    );
}
