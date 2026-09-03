//! Plan schema 4: who judged, what was judged, and whether two verdicts
//! are comparable.

use larql_models::inventory::ArchitectureInventory;

use super::support::{glimmer_shaped_target, glimmer_shaped_target_with};
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
    assert_eq!(PLANNER_SEMANTICS_VERSION, 13);

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
