//! `larql recipe build` — the Vindex Factory build-stage driver
//! (docs/vindex-factory.md §7).
//!
//! Scope, decided after tracing the actual reusable tooling (none of
//! the spec's original reuse assumptions held as written — see each
//! stage module's doc comment for the specific correction): this
//! module runs PREFLIGHT through RELEASE for real, orchestrating
//! existing `larql` subcommands as subprocesses via [`runner`]. MIRROR
//! (R2) and REGISTER (chuk-experiments-server) are not implemented
//! here — nothing in this codebase talks to either today, and the
//! spec's own text assumes they're owned by the rig's worker
//! infrastructure, not the `larql` binary itself; [`record::BuildRecord`]
//! is the structured hand-off point for an external wrapper to do both,
//! the same way `dec0-loopback.sh` already wraps `dec-bench`'s JSON
//! output. Reconstruction-fidelity and logit-match numeric checks
//! (§8.1) are also not implemented — building them correctly needs
//! per-architecture tensor-naming knowledge this session has no way to
//! validate against real model weights, so VERIFY here covers checksum
//! integrity only (reusing the existing `larql verify` command) rather
//! than a confident-but-unvalidated numeric check.

mod record;
mod runner;
mod stages;

pub use record::{BuildRecord, BuildStatus, OutputRecord, Stage};
pub use runner::{CommandOutput, CommandRunner, SubprocessRunner};

use std::path::{Path, PathBuf};

use crate::Recipe;

const HF_CACHE_SUBDIR: &str = "hf-cache";
const FULL_OUTPUT_SUBDIR: &str = "full.vindex";

/// Run every stage for `recipe`, using `scratch_dir` as the build's
/// working directory (created if missing) and `runner` to execute each
/// `larql` subcommand. Always returns a [`BuildRecord`] — a stage
/// failure is encoded in [`BuildRecord::status`], not a Rust `Err`, so
/// a caller always has a JSON-printable result either way.
pub fn run(runner: &dyn CommandRunner, recipe: &Recipe, scratch_dir: &Path) -> BuildRecord {
    let build_id = crate::build_id(recipe);
    let mut output_records: Vec<OutputRecord> = recipe
        .spec
        .outputs
        .iter()
        .map(|o| OutputRecord::new(&o.preset))
        .collect();

    macro_rules! fail {
        ($stage:expr, $message:expr) => {
            return BuildRecord {
                build_id,
                recipe_name: recipe.metadata.name.clone(),
                outputs: output_records,
                status: BuildStatus::Failed {
                    stage: $stage,
                    message: $message,
                },
            }
        };
    }

    if let Err(e) = stages::preflight::check_scratch_dir_writable(scratch_dir) {
        fail!(Stage::Preflight, e);
    }

    let hf_cache_dir = scratch_dir.join(HF_CACHE_SUBDIR);
    let full_dir = scratch_dir.join(FULL_OUTPUT_SUBDIR);

    if let Err(e) = stages::fetch::run(runner, recipe, &hf_cache_dir) {
        fail!(Stage::Fetch, e);
    }
    if let Err(e) = stages::extract::run(runner, recipe, &full_dir, &hf_cache_dir) {
        fail!(Stage::Extract, e);
    }

    // SLICE: resolve each output's local directory, slicing non-full
    // presets from the just-extracted full vindex.
    let mut output_dirs: Vec<PathBuf> = Vec::with_capacity(output_records.len());
    for output in &recipe.spec.outputs {
        let dir = if stages::slice::needs_slicing(output) {
            let dst = scratch_dir.join(format!("{}.vindex", output.preset));
            if let Err(e) = stages::slice::run(runner, &full_dir, &dst, output) {
                fail!(Stage::Slice, e);
            }
            dst
        } else {
            full_dir.clone()
        };
        output_dirs.push(dir);
    }

    // MANIFEST: measure what each output produced.
    for (i, dir) in output_dirs.iter().enumerate() {
        match stages::manifest::summarise(&output_records[i].preset, dir) {
            Ok(summary) => output_records[i].size_bytes = Some(summary.size_bytes),
            Err(e) => fail!(Stage::Manifest, e),
        }
    }

    // VERIFY: checksum integrity (see stages::verify's doc comment for
    // what this deliberately doesn't cover).
    for dir in &output_dirs {
        if let Err(e) = stages::verify::run(runner, dir) {
            fail!(Stage::Verify, e);
        }
    }

    // PUBLISH: private first — RELEASE flips visibility once every
    // output has published and verified.
    for (i, dir) in output_dirs.iter().enumerate() {
        match stages::publish::run(
            runner,
            dir,
            &recipe.metadata,
            &recipe.spec.publish,
            &recipe.spec.outputs[i],
        ) {
            Ok(repo) => output_records[i].repo = Some(repo),
            Err(e) => fail!(Stage::Publish, e),
        }
    }

    // RELEASE: flip every published repo to public.
    for i in 0..output_records.len() {
        let repo = output_records[i]
            .repo
            .clone()
            .expect("PUBLISH set every output's repo before RELEASE runs");
        if let Err(e) = stages::release::run(runner, &repo, &recipe.spec.publish.hub.repo_type) {
            fail!(Stage::Release, e);
        }
        output_records[i].released = true;
    }

    BuildRecord {
        build_id,
        recipe_name: recipe.metadata.name.clone(),
        outputs: output_records,
        status: BuildStatus::Passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::MockRunner;

    fn sample_recipe() -> Recipe {
        // Single-output fixture keeps the mock call sequence short and
        // readable; the multi-output sample fixture is exercised via
        // the individual stage modules' own tests.
        Recipe::from_yaml(
            r#"
apiVersion: vindex.chukai.io/v1
kind: VindexBuild
metadata:
  name: tiny-model
spec:
  source:
    hf_repo: chrishayuk/tiny-model
    revision: 9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82
    licence: mit
  extractor:
    tool: larql
    version: "0.14.2"
    level: inference
    dtype: f16
  outputs:
    - preset: full
  verify:
    reconstruction:
      layers_sampled: 1
      seed: 1
      max_abs_diff: 0.0
      min_cosine: 1.0
    logit_match:
      prompt_set: verify/tiny-smoke.jsonl
      top1_agreement_min: 0.99
      bits_per_char_drift_max: 0.005
    from_hub: true
  publish:
    hub:
      owner: chrishayuk
      repo_template: "{model_slug}-vindex{slice_suffix}"
      repo_type: dataset
      visibility: private-until-verified
  budget:
    executor: auto
    max_wall_minutes: 30
    max_usd: 1
    requires:
      disk_gib: 8
      ram_gib: 8
      net_down_mbps: 100
"#,
        )
        .unwrap()
    }

    fn ok() -> CommandOutput {
        CommandOutput {
            status: 0,
            ..Default::default()
        }
    }

    #[test]
    fn full_pipeline_passes_when_every_stage_succeeds() {
        let recipe = sample_recipe();
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

        let record = run(&runner, &recipe, scratch.path());

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
        let recipe = sample_recipe();
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

        let record = run(&runner, &recipe, scratch.path());

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
        let recipe = sample_recipe();
        // A file, not a directory — create_dir_all fails on it.
        let scratch_parent = tempfile::tempdir().unwrap();
        let not_a_dir = scratch_parent.path().join("blocked");
        std::fs::write(&not_a_dir, b"x").unwrap();
        let scratch = not_a_dir.join("nested");

        let runner = MockRunner::new(); // expects zero calls

        let record = run(&runner, &recipe, &scratch);

        assert!(!record.passed());
        match record.status {
            BuildStatus::Failed { stage, .. } => assert_eq!(stage, Stage::Preflight),
            BuildStatus::Passed => panic!("expected Failed"),
        }
        assert!(runner.all_consumed());
    }

    #[test]
    fn build_id_is_present_regardless_of_outcome() {
        let recipe = sample_recipe();
        let scratch = tempfile::tempdir().unwrap();
        let hf_cache = scratch.path().join(HF_CACHE_SUBDIR);
        let runner = MockRunner::new().expect(
            stages::fetch::invocation(&recipe, &hf_cache),
            CommandOutput {
                status: 1,
                ..Default::default()
            },
        );
        let record = run(&runner, &recipe, scratch.path());
        assert_eq!(record.build_id, crate::build_id(&recipe));
        assert_eq!(record.recipe_name, "tiny-model");
    }
}
