//! Public preflight refusal and private stage-orchestration tests.
//!
//! Stage tests call `run_stages` to cover orchestration independently of the
//! public verification-capability refusal. Every stage is mocked through [`MockRunner`], so these assert the
//! orchestration contract — stage order, what each failure reports, and
//! what lands in the [`BuildRecord`] — without spawning a process or
//! needing credentials. Per-stage argument construction is tested in
//! each stage module.

use super::*;
use crate::build::runner::{Invocation, MockRunner};
use crate::test_support::{sample_recipe, single_output_recipe};
use crate::OutputSpec;

fn ok() -> CommandOutput {
    CommandOutput {
        status: 0,
        ..Default::default()
    }
}

fn failed(stderr: &str) -> CommandOutput {
    CommandOutput {
        status: 1,
        stderr: stderr.into(),
        ..Default::default()
    }
}

/// A scratch dir with a populated `full.vindex` — MANIFEST does a real
/// filesystem walk even when every subprocess is mocked, so it needs
/// bytes on disk to measure.
fn scratch_with_extract_output(recipe: &Recipe) -> tempfile::TempDir {
    let scratch = tempfile::tempdir().unwrap();
    for output in &recipe.spec.outputs {
        let dir = if stages::slice::needs_slicing(output) {
            scratch.path().join(format!("{}.vindex", output.preset))
        } else {
            scratch.path().join(FULL_OUTPUT_SUBDIR)
        };
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weights.bin"), vec![0u8; 128]).unwrap();
    }
    scratch
}

/// Every invocation the driver makes for `recipe`, in the order
/// [`run`] makes them. Pinning the sequence in one place is what lets
/// each failure test say only *where* it fails.
fn expected_sequence(recipe: &Recipe, scratch: &Path) -> Vec<Invocation> {
    let hf_cache = scratch.join(HF_CACHE_SUBDIR);
    let full_dir = scratch.join(FULL_OUTPUT_SUBDIR);
    let repo = |output: &OutputSpec| {
        stages::publish::repo_name(&recipe.metadata, &recipe.spec.publish, output)
    };
    let output_dir = |output: &OutputSpec| {
        if stages::slice::needs_slicing(output) {
            scratch.join(format!("{}.vindex", output.preset))
        } else {
            full_dir.clone()
        }
    };
    let repo_type = &recipe.spec.publish.hub.repo_type;

    let mut seq = vec![
        stages::fetch::invocation(recipe, &hf_cache),
        stages::extract::invocation(recipe, &full_dir, &hf_cache),
    ];
    for output in &recipe.spec.outputs {
        if stages::slice::needs_slicing(output) {
            seq.push(stages::slice::invocation(
                &full_dir,
                &output_dir(output),
                output,
            ));
        }
    }
    // MANIFEST runs between SLICE and VERIFY but spawns nothing.
    for output in &recipe.spec.outputs {
        seq.push(stages::verify::invocation(&output_dir(output)));
    }
    for output in &recipe.spec.outputs {
        seq.push(stages::publish::invocation(
            &output_dir(output),
            &repo(output),
            repo_type,
        ));
    }
    for output in &recipe.spec.outputs {
        seq.push(stages::release::invocation(&repo(output), repo_type));
    }
    seq
}

/// A runner that succeeds for the first `n` invocations of `seq` and
/// fails the one after, with `stderr`. Later invocations are never
/// queued, so a driver that keeps going past a failure trips
/// `MockRunner`'s "unexpected invocation" panic.
fn runner_failing_after(seq: &[Invocation], n: usize, stderr: &str) -> MockRunner {
    let mut runner = MockRunner::new();
    for inv in &seq[..n] {
        runner = runner.expect(inv.clone(), ok());
    }
    runner.expect(seq[n].clone(), failed(stderr))
}

/// The stage a failed `record` stopped at, or a panic if it passed.
fn failed_stage(record: &BuildRecord) -> (Stage, String) {
    match &record.status {
        BuildStatus::Failed { stage, message } => (*stage, message.clone()),
        BuildStatus::Passed => panic!("expected a failed build, got {:?}", record.status),
    }
}

#[test]
fn full_pipeline_passes_when_every_stage_succeeds() {
    let recipe = single_output_recipe();
    let scratch = tempfile::tempdir().unwrap();
    let hf_cache = scratch.path().join(HF_CACHE_SUBDIR);
    let full_dir = scratch.path().join(FULL_OUTPUT_SUBDIR);
    let output = &recipe.spec.outputs[0];
    // FETCH/EXTRACT are mocked, so nothing actually writes the
    // output directory — MANIFEST still does a real filesystem
    // walk, so it needs something on disk to measure.
    std::fs::create_dir_all(&full_dir).unwrap();
    std::fs::write(full_dir.join("weights.bin"), vec![0u8; 128]).unwrap();

    let runner = MockRunner::new()
        .expect(stages::fetch::invocation(&recipe, &hf_cache), ok())
        .expect(
            stages::extract::invocation(&recipe, &full_dir, &hf_cache),
            ok(),
        )
        // Single "full" output: no SLICE call.
        .expect(stages::verify::invocation(&full_dir), ok())
        .expect(
            stages::publish::invocation(
                &full_dir,
                &stages::publish::repo_name(&recipe.metadata, &recipe.spec.publish, output),
                &recipe.spec.publish.hub.repo_type,
            ),
            ok(),
        )
        .expect(
            stages::release::invocation(
                &stages::publish::repo_name(&recipe.metadata, &recipe.spec.publish, output),
                &recipe.spec.publish.hub.repo_type,
            ),
            ok(),
        );

    let record = run_stages(&runner, &recipe, scratch.path());

    assert!(record.passed(), "{record:?}", record = record.status);
    assert_eq!(record.outputs.len(), 1);
    assert_eq!(
        record.outputs[0].repo.as_deref(),
        Some("chrishayuk/tiny-model-vindex")
    );
    assert!(record.outputs[0].released);
    assert_eq!(record.build_id, crate::build_id(&recipe));
    assert!(runner.all_consumed());
}

#[test]
fn stops_at_fetch_on_failure_without_running_later_stages() {
    let recipe = single_output_recipe();
    let scratch = tempfile::tempdir().unwrap();
    let hf_cache = scratch.path().join(HF_CACHE_SUBDIR);

    let runner = MockRunner::new().expect(
        stages::fetch::invocation(&recipe, &hf_cache),
        CommandOutput {
            status: 1,
            stderr: "404 not found".into(),
            ..Default::default()
        },
    );

    let record = run_stages(&runner, &recipe, scratch.path());

    assert!(!record.passed());
    match record.status {
        BuildStatus::Failed { stage, message } => {
            assert_eq!(stage, Stage::Fetch);
            assert!(message.contains("404"));
        }
        BuildStatus::Passed => panic!("expected Failed"),
    }
    // Output records exist but weren't touched — FETCH failed
    // before MANIFEST/PUBLISH/RELEASE could run.
    assert!(record.outputs[0].repo.is_none());
    assert!(runner.all_consumed());
}

#[test]
fn preflight_failure_never_touches_the_runner() {
    let recipe = single_output_recipe();
    // A file, not a directory — create_dir_all fails on it.
    let scratch_parent = tempfile::tempdir().unwrap();
    let not_a_dir = scratch_parent.path().join("blocked");
    std::fs::write(&not_a_dir, b"x").unwrap();
    let scratch = not_a_dir.join("nested");

    let runner = MockRunner::new(); // expects zero calls

    let record = run_stages(&runner, &recipe, &scratch);

    assert!(!record.passed());
    match record.status {
        BuildStatus::Failed { stage, .. } => assert_eq!(stage, Stage::Preflight),
        BuildStatus::Passed => panic!("expected Failed"),
    }
    assert!(runner.all_consumed());
}

// ── Every stage's failure arm ───────────────────────────────────────
// §8's contract is that a build stops at the first failing stage and
// says which one it was. One test per arm, each asserting the reported
// stage, that stderr survives into the message, and — via
// `MockRunner`'s unexpected-invocation panic — that nothing downstream
// ran.

#[test]
fn stops_at_extract_on_failure() {
    let recipe = single_output_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = runner_failing_after(&seq, 1, "safetensors header truncated");

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, message) = failed_stage(&record);
    assert_eq!(stage, Stage::Extract);
    assert!(
        message.contains("safetensors header truncated"),
        "{message}"
    );
    assert!(runner.all_consumed());
}

#[test]
fn stops_at_slice_on_failure() {
    // Needs a multi-output recipe: the single-output fixture is all
    // `full`, which the driver never slices.
    let recipe = sample_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = runner_failing_after(&seq, 2, "unknown part");

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, message) = failed_stage(&record);
    assert_eq!(stage, Stage::Slice);
    assert!(message.contains("unknown part"), "{message}");
    assert!(runner.all_consumed());
}

#[test]
fn stops_at_manifest_when_an_output_directory_is_missing() {
    // MANIFEST is the one stage that spawns nothing — it fails by
    // measuring a directory that EXTRACT never produced.
    let recipe = single_output_recipe();
    let scratch = tempfile::tempdir().unwrap(); // deliberately empty
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = MockRunner::new()
        .expect(seq[0].clone(), ok())
        .expect(seq[1].clone(), ok());

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, _) = failed_stage(&record);
    assert_eq!(stage, Stage::Manifest);
    assert!(runner.all_consumed());
}

#[test]
fn stops_at_verify_on_failure() {
    let recipe = single_output_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = runner_failing_after(&seq, 2, "checksum mismatch");

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, message) = failed_stage(&record);
    assert_eq!(stage, Stage::Verify);
    assert!(message.contains("checksum mismatch"), "{message}");
    // Nothing was published, so nothing needs un-publishing.
    assert!(record.outputs[0].repo.is_none());
    assert!(runner.all_consumed());
}

#[test]
fn stops_at_publish_on_failure() {
    let recipe = single_output_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = runner_failing_after(&seq, 3, "401 unauthorized");

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, message) = failed_stage(&record);
    assert_eq!(stage, Stage::Publish);
    assert!(message.contains("401"), "{message}");
    assert!(!record.outputs[0].released);
    assert!(runner.all_consumed());
}

#[test]
fn a_release_failure_leaves_the_repo_recorded_but_unreleased() {
    // The §8 case that matters most: PUBLISH succeeded, so a private
    // repo exists and the record must name it even though the build
    // failed — otherwise the private repo is orphaned.
    let recipe = single_output_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let runner = runner_failing_after(&seq, 4, "403 forbidden");

    let record = run_stages(&runner, &recipe, scratch.path());

    let (stage, message) = failed_stage(&record);
    assert_eq!(stage, Stage::Release);
    assert!(message.contains("403"), "{message}");
    assert_eq!(
        record.outputs[0].repo.as_deref(),
        Some("chrishayuk/tiny-model-vindex")
    );
    assert!(!record.outputs[0].released, "nothing went public");
    assert!(runner.all_consumed());
}

#[test]
fn a_multi_output_recipe_slices_every_non_full_preset_and_releases_all() {
    let recipe = sample_recipe();
    let scratch = scratch_with_extract_output(&recipe);
    let seq = expected_sequence(&recipe, scratch.path());
    let mut runner = MockRunner::new();
    for inv in &seq {
        runner = runner.expect(inv.clone(), ok());
    }

    let record = run_stages(&runner, &recipe, scratch.path());

    assert!(record.passed(), "{:?}", record.status);
    assert_eq!(record.outputs.len(), recipe.spec.outputs.len());
    assert!(record.outputs.iter().all(|o| o.released));
    assert!(record.outputs.iter().all(|o| o.repo.is_some()));
    assert!(record.outputs.iter().all(|o| o.size_bytes == Some(128)));
    assert!(runner.all_consumed());
}

#[test]
fn build_id_is_present_regardless_of_outcome() {
    let recipe = single_output_recipe();
    let scratch = tempfile::tempdir().unwrap();
    let hf_cache = scratch.path().join(HF_CACHE_SUBDIR);
    let runner = MockRunner::new().expect(
        stages::fetch::invocation(&recipe, &hf_cache),
        CommandOutput {
            status: 1,
            ..Default::default()
        },
    );
    let record = run_stages(&runner, &recipe, scratch.path());
    assert_eq!(record.build_id, crate::build_id(&recipe));
    assert_eq!(record.recipe_name, "tiny-model");
}

#[test]
fn public_build_refuses_unimplemented_verification_before_any_side_effect() {
    for from_hub in [false, true] {
        let mut recipe = single_output_recipe();
        recipe.spec.verify.from_hub = from_hub;
        let parent = tempfile::tempdir().unwrap();
        let scratch = parent.path().join("must-not-be-created");
        let runner = MockRunner::new();
        let record = run(&runner, &recipe, &scratch);
        let (stage, message) = failed_stage(&record);
        assert_eq!(stage, Stage::Preflight);
        assert!(message.contains("spec.verify.reconstruction"));
        assert!(message.contains("spec.verify.logit_match"));
        assert_eq!(message.contains("spec.verify.from_hub"), from_hub);
        assert!(!scratch.exists());
        assert!(record
            .outputs
            .iter()
            .all(|o| o.repo.is_none() && !o.released));
        assert!(runner.all_consumed());
        assert_eq!(record.build_id, crate::build_id(&recipe));
    }
}
