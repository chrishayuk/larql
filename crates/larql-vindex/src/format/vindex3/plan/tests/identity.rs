//! Plan schema 4: who judged, what was judged, and whether two verdicts
//! are comparable.

use larql_models::inventory::build_inventory;
use larql_models::inventory::ArchitectureInventory;

use super::support::{custom_artifact, glimmer_shaped_target, glimmer_shaped_target_with};
use crate::format::vindex3::plan::{
    plan_system, plan_system_with_sources, ArtifactSource, PlannerIdentity, SystemPlan,
    VerdictCacheKey, PLANNER_SEMANTICS_VERSION, PLAN_SCHEMA,
};

const ARTIFACT: &str = "target-artifact";

fn one_glimmer(dir: &std::path::Path) -> Vec<(String, ArchitectureInventory)> {
    vec![(ARTIFACT.to_string(), glimmer_shaped_target(dir))]
}

#[test]
fn a_plan_names_the_build_that_judged_it() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_system(&one_glimmer(dir.path()));
    assert_eq!(plan.schema, PLAN_SCHEMA);
    assert_eq!(plan.planner, PlannerIdentity::current());
    assert_eq!(plan.planner.package, env!("CARGO_PKG_NAME"));
    assert_eq!(plan.planner.package_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(plan.planner.semantics_version, PLANNER_SEMANTICS_VERSION);
}

#[test]
fn a_local_artifacts_source_is_the_path_its_inventory_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let named = one_glimmer(dir.path());
    let plan = plan_system(&named);
    assert_eq!(
        plan.artifacts[0].source,
        ArtifactSource::local(named[0].1.path.clone())
    );
    assert!(plan.artifacts[0].source.revision.is_none());
}

#[test]
fn a_stated_source_carries_its_revision_and_changes_no_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let named = one_glimmer(dir.path());
    let sources = [ArtifactSource {
        path: "hf://org/model@abc123".to_string(),
        revision: Some("abc123".to_string()),
        unpinned_revision: None,
    }];
    let plan = plan_system_with_sources(&named, &sources).unwrap();
    assert_eq!(plan.artifacts[0].source, sources[0]);
    assert_eq!(plan.summary, plan_system(&named).summary);
}

#[test]
fn sources_must_pair_one_to_one_with_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let named = one_glimmer(dir.path());
    let err = plan_system_with_sources(
        &named,
        &[ArtifactSource::local("a"), ArtifactSource::local("b")],
    )
    .err()
    .map(|e| e.to_string())
    .expect("two sources for one artifact must be refused");
    assert!(err.contains("1 artifact(s) but 2 source(s)"), "{err}");
}

#[test]
fn identity_survives_a_round_trip_and_parse_refuses_other_schemas_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_system(&one_glimmer(dir.path()));
    let json = serde_json::to_string_pretty(&plan).unwrap();
    let back = SystemPlan::parse(&json).unwrap();
    assert_eq!(back.planner, plan.planner);
    assert_eq!(back.artifacts[0].source, plan.artifacts[0].source);

    // A plan written by the previous schema: no planner, schema 3.
    let mut old: serde_json::Value = serde_json::from_str(&json).unwrap();
    old["schema"] = serde_json::json!(3);
    old.as_object_mut().unwrap().remove("planner");
    let err = SystemPlan::parse(&old.to_string()).unwrap_err().to_string();
    assert!(err.contains("plan schema 3"), "{err}");
    assert!(
        err.contains(&format!("reads plan schema {PLAN_SCHEMA}")),
        "{err}"
    );

    old.as_object_mut().unwrap().remove("schema");
    let err = SystemPlan::parse(&old.to_string()).unwrap_err().to_string();
    assert!(err.contains("declares no schema"), "{err}");

    let err = SystemPlan::parse("not json").unwrap_err().to_string();
    assert!(err.contains("not a JSON object"), "{err}");
}

/// The semantics version is a promise about verdicts. This pins two
/// verdicts the fixtures are known to give — an admissible one and a
/// blocked one — beside the version, so a rule change that flips either
/// fails here until `PLANNER_SEMANTICS_VERSION` is bumped and this
/// witness is re-recorded.
#[test]
fn the_semantics_version_is_pinned_to_known_verdicts() {
    assert_eq!(PLANNER_SEMANTICS_VERSION, 21);

    let dir = tempfile::tempdir().unwrap();
    let admissible = plan_system(&one_glimmer(dir.path()));
    assert!(admissible.admissible, "{:?}", admissible.summary);
    assert_eq!(admissible.summary.blocking, 0);

    // Version 2's verdict: an identity no registry entry matches blocks,
    // where it used to pass into `GenericArch` unremarked. Pinned as a
    // verdict rather than as a unit of the gate, because that is what the
    // version number promises is comparable between builds.
    let dir = tempfile::tempdir().unwrap();
    let unjudged = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["model_type"] = serde_json::json!("no_such_family");
    });
    let refused = plan_system(&[(ARTIFACT.to_string(), unjudged)]);
    assert!(!refused.admissible, "{:?}", refused.summary);

    // Version 3's verdict: a decoding-policy default no longer blocks,
    // and `pretraining_tp` is judged against its value rather than its
    // name. Both arms are pinned — the second is what keeps the first
    // from having been implemented as "stop reading these keys".
    let dir = tempfile::tempdir().unwrap();
    let inert = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["do_sample"] = serde_json::json!(true);
        config["text_config"]["pretraining_tp"] = serde_json::json!(1);
    });
    let passed = plan_system(&[(ARTIFACT.to_string(), inert)]);
    assert!(passed.admissible, "{:?}", passed.summary);

    let dir = tempfile::tempdir().unwrap();
    let sliced = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["pretraining_tp"] = serde_json::json!(2);
    });
    let blocked = plan_system(&[(ARTIFACT.to_string(), sliced)]);
    assert!(
        !blocked.admissible,
        "pretraining_tp above 1 slices every projection and must block: {:?}",
        blocked.summary
    );

    // Version 4's verdict: a Llama-3 wavelength-band block is carried,
    // where every one of its four keys used to block.
    let dir = tempfile::tempdir().unwrap();
    let llama3 = glimmer_shaped_target_with(dir.path(), |config| {
        // Declared in the block this fixture already uses, so the
        // checkpoint states ONE rope type. Adding a second block would
        // test a conflicting declaration, which is a different gate.
        config["text_config"]["rope_parameters"] = serde_json::json!({
            "rope_theta": 500000.0,
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        });
    });
    let carried = plan_system(&[(ARTIFACT.to_string(), llama3)]);
    assert!(carried.admissible, "{:?}", carried.summary);

    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["some_future_field_nobody_reviewed"] = serde_json::json!(1.5);
    });
    let blocked = plan_system(&[(ARTIFACT.to_string(), inventory)]);
    assert!(!blocked.admissible, "{:?}", blocked.summary);
    assert!(blocked.summary.blocking > 0);

    // Version 15's verdict (wave 18): the hyper-connection head's three
    // bare groups are PLACED on a component that declares the topology
    // — no longer three unplaced-group blockers — and stay unplaced on
    // one that does not. Both arms pinned: the second is what keeps the
    // first from having been implemented as "own anything named hc_".
    // Neither plan is admissible (the estate is a stub with no layer
    // operands), so the verdict pinned is the groups' fate, read off the
    // findings by subject.
    let head_groups = ["hc_head_fn", "hc_head_base", "hc_head_scale"];
    let head_fate = |declares_topology: bool| -> Vec<String> {
        let dir = tempfile::tempdir().unwrap();
        let mut config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        if declares_topology {
            config["hc_mult"] = serde_json::json!(4);
            config["hc_sinkhorn_iters"] = serde_json::json!(20);
            config["hc_eps"] = serde_json::json!(1e-6);
        }
        let tensors: Vec<(&str, &[usize])> = vec![
            ("model.embed_tokens.weight", &[128, 64]),
            ("model.norm.weight", &[64]),
            ("model.layers.0.input_layernorm.weight", &[64]),
            ("hc_head_fn", &[4, 256]),
            ("hc_head_base", &[4]),
            ("hc_head_scale", &[1]),
        ];
        let inventory = custom_artifact(dir.path(), &config, &tensors);
        let plan = plan_system(&[(ARTIFACT.to_string(), inventory)]);
        plan.artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .filter(|f| head_groups.contains(&f.subject.as_str()) && f.blocks())
            .map(|f| f.subject.clone())
            .collect()
    };
    assert_eq!(
        head_fate(true),
        Vec::<String>::new(),
        "under the declaration the head groups are placed, not blockers"
    );
    let mut undeclared = head_fate(false);
    undeclared.sort();
    assert_eq!(
        undeclared,
        ["hc_head_base", "hc_head_fn", "hc_head_scale"],
        "without the declaration the same groups block as unplaced"
    );

    // Version 16's verdict (wave 19): a hyper-connected stack with every
    // site operand AND a head object is ADMISSIBLE — the topology's keys
    // are carried and the traversal runs — while the same stack with no
    // head stays blocked, by the head's own finding and not the
    // topology's. Both arms pinned: the second is what keeps the first
    // from having been implemented as "stop refusing hc_".
    let hc_stack = |head: bool| -> SystemPlan {
        let dir = tempfile::tempdir().unwrap();
        let mut config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        config["hc_mult"] = serde_json::json!(4);
        config["hc_sinkhorn_iters"] = serde_json::json!(20);
        config["hc_eps"] = serde_json::json!(1e-6);
        let mut tensors: Vec<(&str, &[usize])> = vec![
            ("model.embed_tokens.weight", &[128, 64]),
            ("model.norm.weight", &[64]),
            ("lm_head.weight", &[128, 64]),
            ("model.layers.0.self_attn.q_proj.weight", &[64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", &[16, 64]),
            ("model.layers.0.self_attn.v_proj.weight", &[16, 64]),
            ("model.layers.0.self_attn.o_proj.weight", &[64, 64]),
            ("model.layers.0.input_layernorm.weight", &[64]),
            ("model.layers.0.post_attention_layernorm.weight", &[64]),
            ("model.layers.0.mlp.gate_proj.weight", &[256, 64]),
            ("model.layers.0.mlp.up_proj.weight", &[256, 64]),
            ("model.layers.0.mlp.down_proj.weight", &[64, 256]),
            ("model.layers.0.hc_attn_fn", &[24, 256]),
            ("model.layers.0.hc_attn_base", &[24]),
            ("model.layers.0.hc_attn_scale", &[3]),
            ("model.layers.0.hc_ffn_fn", &[24, 256]),
            ("model.layers.0.hc_ffn_base", &[24]),
            ("model.layers.0.hc_ffn_scale", &[3]),
        ];
        if head {
            tensors.push(("hc_head_fn", &[4, 256]));
            tensors.push(("hc_head_base", &[4]));
            tensors.push(("hc_head_scale", &[1]));
        }
        let inventory = custom_artifact(dir.path(), &config, &tensors);
        plan_system(&[(ARTIFACT.to_string(), inventory)])
    };
    let with_head = hc_stack(true);
    assert!(with_head.admissible, "{:?}", with_head.summary);
    let headless = hc_stack(false);
    assert!(!headless.admissible, "{:?}", headless.summary);
    let blockers: Vec<_> = headless
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .collect();
    assert_eq!(blockers.len(), 1, "{blockers:?}");
    assert!(
        blockers[0].subject.ends_with("execution_surface")
            && blockers[0].detail.contains("hyper_connection_head"),
        "{:?}",
        blockers[0]
    );

    // Version 18's verdict (K3-ATTNRES-1, transition 2): an
    // attention-residual stack with every site operand AND the exit pair
    // is ADMISSIBLE — its period is carried to a traversal that reads
    // it, its exit is one placed object, its four per-layer operands are
    // addressed, and the decode (2a) and batch (2b) traversals have each
    // been witnessed against a Torch oracle. At version 17 this same
    // estate was blocked by ONE finding, the topology's own refusal,
    // because no traversal existed; that refusal and both its readers
    // are now deleted.
    //
    // The same stack WITHOUT the exit pair is still blocked, by the
    // exit's own finding. That arm is what keeps the first from having
    // been implemented as "stop refusing anything that declares a
    // period": capability was granted to a traversal, not to the
    // declaration. A plan that blocked on NEITHER would be the fail-open
    // this rung exists to make impossible.
    let attn_res_stack = |exit: bool| -> SystemPlan {
        let dir = tempfile::tempdir().unwrap();
        let mut config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "bfloat16",
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        config["attn_res_block_size"] = serde_json::json!(3);
        let mut tensors: Vec<(&str, &[usize])> = vec![
            ("model.embed_tokens.weight", &[128, 64]),
            ("model.norm.weight", &[64]),
            ("lm_head.weight", &[128, 64]),
            ("model.layers.0.self_attn.q_proj.weight", &[64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", &[16, 64]),
            ("model.layers.0.self_attn.v_proj.weight", &[16, 64]),
            ("model.layers.0.self_attn.o_proj.weight", &[64, 64]),
            ("model.layers.0.input_layernorm.weight", &[64]),
            ("model.layers.0.post_attention_layernorm.weight", &[64]),
            ("model.layers.0.mlp.gate_proj.weight", &[256, 64]),
            ("model.layers.0.mlp.up_proj.weight", &[256, 64]),
            ("model.layers.0.mlp.down_proj.weight", &[64, 256]),
            ("model.layers.0.self_attention_res_norm.weight", &[64]),
            ("model.layers.0.self_attention_res_proj.weight", &[1, 64]),
            ("model.layers.0.mlp_res_norm.weight", &[64]),
            ("model.layers.0.mlp_res_proj.weight", &[1, 64]),
        ];
        if exit {
            tensors.push(("model.output_attn_res_norm.weight", &[64]));
            tensors.push(("model.output_attn_res_proj.weight", &[1, 64]));
        }
        let inventory = custom_artifact(dir.path(), &config, &tensors);
        plan_system(&[(ARTIFACT.to_string(), inventory)])
    };
    // Version 19's verdict (K3-REP-GATE-1): the two K3 attention output
    // gates are DECLARED facts the plan carries. A Kimi-shaped estate
    // declaring `linear_attn_config.use_full_rank_gate` and
    // `mla_use_output_gate` is ADMISSIBLE — both keys judged
    // ExecutionSemantic and carried to the KDA op's gate form and the MLA
    // op's gate operand — where at version 18 the same estate was blocked
    // by two Unknown findings. The control beside it: Kimi Linear's own
    // forms, neither key declared, admissible at both versions.
    //
    // The same two keys on a component with NO KDA and no MLA block stay
    // blocked, uncarried: a gate form with no gate to describe reaches
    // nothing, and the probe says so. That arm is what keeps the first
    // from having been implemented as "stop reading these keys".
    let kimi_shaped = |forms: crate::format::vindex3::fixtures_kimi::HybridGateForms| {
        let dir = tempfile::tempdir().unwrap();
        crate::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model_with(dir.path(), forms);
        let inventory = build_inventory(dir.path()).unwrap();
        plan_system(&[(ARTIFACT.to_string(), inventory)])
    };
    let gated = kimi_shaped(crate::format::vindex3::fixtures_kimi::HybridGateForms::KIMI_K3);
    assert!(gated.admissible, "{:?}", gated.summary);
    let ungated = kimi_shaped(crate::format::vindex3::fixtures_kimi::HybridGateForms::KIMI_LINEAR);
    assert!(ungated.admissible, "{:?}", ungated.summary);
    let dir = tempfile::tempdir().unwrap();
    let homeless = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["linear_attn_config"] =
            serde_json::json!({ "use_full_rank_gate": true });
        config["text_config"]["mla_use_output_gate"] = serde_json::json!(true);
    });
    let blocked = plan_system(&[(ARTIFACT.to_string(), homeless)]);
    assert!(!blocked.admissible, "{:?}", blocked.summary);
    let blocking: Vec<String> = blocked
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| f.subject.clone())
        .collect();
    for key in [
        "text_config.linear_attn_config.use_full_rank_gate",
        "text_config.mla_use_output_gate",
    ] {
        assert!(
            blocking.iter().any(|s| s == key),
            "`{key}` on a component with no KDA/MLA block must stay blocked, uncarried: {blocking:?}"
        );
    }

    // Version 20's verdict (K3-ACT-1): `hidden_act: "situ"` with its two
    // softcaps is a carried declaration, where at version 19 the name
    // resolved to `silu` (a MISMATCH that blocked) and both softcaps were
    // Unknown. The control beside it is the second specimen of the class:
    // `relu2` names no activation this build has judged and no gate
    // policy, so it must STILL block — which is what keeps the first arm
    // from having been implemented as "resolve every unknown name to
    // itself".
    let situ_declared = |act: &'static str, betas: bool| {
        let dir = tempfile::tempdir().unwrap();
        let inventory = glimmer_shaped_target_with(dir.path(), |config| {
            config["text_config"]["hidden_act"] = serde_json::json!(act);
            if betas {
                config["text_config"]["activation_situ_beta"] = serde_json::json!(4.0);
                config["text_config"]["activation_situ_linear_beta"] = serde_json::json!(25.0);
            }
        });
        plan_system(&[(ARTIFACT.to_string(), inventory)])
    };
    let situ = situ_declared("situ", true);
    assert!(situ.admissible, "{:?}", situ.summary);
    let unjudged_act = situ_declared("relu2", false);
    assert!(
        !unjudged_act.admissible,
        "an activation this build has never judged must still block: {:?}",
        unjudged_act.summary
    );

    // Version 21's verdict (K3-MLA-Q-LORA-1): a Kimi-shaped estate that
    // declares `q_lora_rank` and ships the q-LoRA triple is admissible,
    // where at version 20 the triple was unaddressed and `MlaQProj` was
    // required unconditionally. The control beside it is the SAME estate
    // declaring the rank while shipping a dense `q_proj` — refused,
    // because the declaration chooses the form and the operand plane is
    // held to it. Kimi Linear's own direct form, declaring no rank, is
    // admissible at both versions and is the third arm.
    //
    // Deliberately three arms and not two: an implementation that
    // required the triple unconditionally would pass the first, fail the
    // third, and never be caught by the second.
    let q_lora = |forms: crate::format::vindex3::fixtures_kimi::HybridQueryForms| {
        let dir = tempfile::tempdir().unwrap();
        crate::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model_with_query(
            dir.path(),
            forms,
        );
        let inventory = build_inventory(dir.path()).unwrap();
        plan_system(&[(ARTIFACT.to_string(), inventory)])
    };
    let factorised = q_lora(crate::format::vindex3::fixtures_kimi::HybridQueryForms::KIMI_K3);
    assert!(factorised.admissible, "{:?}", factorised.summary);
    let direct = q_lora(crate::format::vindex3::fixtures_kimi::HybridQueryForms::KIMI_LINEAR);
    assert!(direct.admissible, "{:?}", direct.summary);
    // And the arm this version does NOT move: an estate declaring the
    // rank while shipping a dense `q_proj` still PLANS admissibly,
    // because `q_lora_rank` is a representable config leaf either way.
    // The disagreement is an OPERAND fact and closure is what refuses it
    // (`opplan::tests::k3_q_lora_closure`). Pinned here so the plan
    // stage's silence is a recorded property rather than an oversight —
    // the same distinction `the_plan_stage_places_bytes_and_classifies_
    // no_operand` draws for K3 itself.
    let disagreeing =
        q_lora(crate::format::vindex3::fixtures_kimi::HybridQueryForms::DECLARED_BUT_DENSE);
    assert!(
        disagreeing.admissible,
        "the PLAN stage judges config leaves, not operands: {:?}",
        disagreeing.summary
    );

    let complete = attn_res_stack(true);
    assert!(
        complete.admissible,
        "a complete attention-residual estate is admissible at semantics 18: {:?}",
        complete.summary
    );
    assert_eq!(complete.summary.blocking, 0, "{:?}", complete.summary);

    let no_exit = attn_res_stack(false);
    assert!(!no_exit.admissible, "{:?}", no_exit.summary);
    let blockers: Vec<_> = no_exit
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
    // ...and it is blocked by the EXIT, not by a traversal refusal that
    // should no longer exist anywhere in the report.
    assert!(
        !blockers[0].detail.contains("NOT executable"),
        "the traversal refusal must be gone from the report: {:?}",
        blockers[0]
    );
}

fn pinned(commit: &str) -> ArtifactSource {
    ArtifactSource {
        path: format!("hf://org/model@{commit}"),
        revision: Some(commit.to_string()),
        unpinned_revision: None,
    }
}

fn unpinned(name: &str) -> ArtifactSource {
    ArtifactSource {
        path: format!("hf://org/model@{name}"),
        revision: None,
        unpinned_revision: Some(name.to_string()),
    }
}

#[test]
fn a_pinned_source_yields_a_cache_key_of_its_commit_and_the_semantics_version() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_system_with_sources(&one_glimmer(dir.path()), &[pinned("abc123")]).unwrap();
    assert_eq!(
        plan.cache_key(),
        Some(VerdictCacheKey {
            revisions: vec!["abc123".to_string()],
            semantics_version: PLANNER_SEMANTICS_VERSION,
        })
    );
}

#[test]
fn an_unpinned_or_local_source_is_provenance_not_cache_identity() {
    let dir = tempfile::tempdir().unwrap();
    let named = one_glimmer(dir.path());
    // A revision NAME can point at different facts tomorrow.
    let plan = plan_system_with_sources(&named, &[unpinned("main")]).unwrap();
    assert_eq!(
        plan.artifacts[0].source.unpinned_revision.as_deref(),
        Some("main")
    );
    assert_eq!(plan.cache_key(), None);
    // A local path has no revision at all.
    assert_eq!(plan_system(&named).cache_key(), None);
}

#[test]
fn one_unpinned_artifact_makes_the_whole_plan_uncacheable() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let named = vec![
        ("a".to_string(), glimmer_shaped_target(dir_a.path())),
        ("b".to_string(), glimmer_shaped_target(dir_b.path())),
    ];
    let both = plan_system_with_sources(&named, &[pinned("aaa"), pinned("bbb")]).unwrap();
    assert_eq!(
        both.cache_key().map(|k| k.revisions),
        Some(vec!["aaa".to_string(), "bbb".to_string()])
    );
    let mixed = plan_system_with_sources(&named, &[pinned("aaa"), unpinned("main")]).unwrap();
    assert_eq!(mixed.cache_key(), None);
}
