//! EXTRACT — `larql extract` against the FETCH-scoped HF cache.
//!
//! Maps the recipe's free-form `extractor.options` to the two flags
//! that currently exist on `larql extract`: `down_q4k` (bool) implies
//! `--quant q4k --down-q4k` — `--down-q4k` is a modifier of `--quant
//! q4k`, invalid on its own — and `drop_gate_vectors` (bool) maps
//! directly. Any other option key (e.g. the sample recipe's
//! `preserve_mtp_head`) has no CLI flag yet and is silently ignored —
//! `options` is deliberately open-ended per docs/vindex-factory.md §4,
//! and a future extractor version may recognise more of them.

use std::path::Path;

use larql_vindex_spec::StorageDtype;

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};
use crate::Recipe;

const HF_HUB_CACHE_ENV: &str = "HF_HUB_CACHE";
const DOWN_Q4K_OPTION_KEY: &str = "down_q4k";
const DROP_GATE_VECTORS_OPTION_KEY: &str = "drop_gate_vectors";

fn option_bool(recipe: &Recipe, key: &str) -> bool {
    recipe
        .spec
        .extractor
        .options
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn level_str(level: larql_vindex_spec::ExtractLevel) -> &'static str {
    use larql_vindex_spec::ExtractLevel::*;
    match level {
        Browse => "browse",
        Attention => "attention",
        Inference => "inference",
        All => "all",
    }
}

/// Build the `larql extract <repo> -o <full_dir> ...` invocation.
pub fn invocation(recipe: &Recipe, full_dir: &Path, hf_cache_dir: &Path) -> Invocation {
    let mut args = vec![
        recipe.spec.source.hf_repo.clone(),
        "-o".to_string(),
        full_dir.to_string_lossy().into_owned(),
        "--level".to_string(),
        level_str(recipe.spec.extractor.level).to_string(),
    ];

    if recipe.spec.extractor.dtype == StorageDtype::F32 {
        args.push("--f32".to_string());
    }
    if option_bool(recipe, DOWN_Q4K_OPTION_KEY) {
        args.push("--quant".to_string());
        args.push("q4k".to_string());
        args.push("--down-q4k".to_string());
    }
    if option_bool(recipe, DROP_GATE_VECTORS_OPTION_KEY) {
        args.push("--drop-gate-vectors".to_string());
    }
    // Safe unconditionally: a no-op on a fresh build, resumes a
    // partially-completed one on retry (§7's "each stage independently
    // resumable").
    args.push("--resume".to_string());

    Invocation::new(&["extract"], args).with_env(HF_HUB_CACHE_ENV, &hf_cache_dir.to_string_lossy())
}

/// Run EXTRACT via `runner`.
pub fn run(
    runner: &dyn CommandRunner,
    recipe: &Recipe,
    full_dir: &Path,
    hf_cache_dir: &Path,
) -> Result<CommandOutput, String> {
    let inv = invocation(recipe, full_dir, hf_cache_dir);
    let out = runner.run(&inv)?;
    if !out.success() {
        return Err(format!("larql extract failed: {}", out.stderr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::MockRunner;

    fn sample_recipe() -> Recipe {
        Recipe::from_yaml(include_str!("../../../testdata/gemma-3-4b-it.yaml")).unwrap()
    }

    fn dirs() -> (std::path::PathBuf, std::path::PathBuf) {
        (
            std::path::PathBuf::from("/scratch/full.vindex"),
            std::path::PathBuf::from("/scratch/hf-cache"),
        )
    }

    #[test]
    fn invocation_carries_repo_output_and_level() {
        let recipe = sample_recipe();
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert_eq!(inv.subcommand, vec!["extract"]);
        assert!(inv.args.contains(&"google/gemma-3-4b-it".to_string()));
        assert!(inv.args.contains(&"-o".to_string()));
        assert!(inv.args.contains(&"/scratch/full.vindex".to_string()));
        assert!(inv.args.contains(&"--level".to_string()));
        assert!(inv.args.contains(&"inference".to_string()));
    }

    #[test]
    fn invocation_shares_the_fetch_scoped_hf_cache() {
        let recipe = sample_recipe();
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert_eq!(
            inv.env,
            vec![("HF_HUB_CACHE".to_string(), "/scratch/hf-cache".to_string())]
        );
    }

    #[test]
    fn down_q4k_option_implies_quant_and_down_q4k_flags() {
        let recipe = sample_recipe();
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        // Sample recipe sets down_q4k: true.
        assert!(inv.args.contains(&"--quant".to_string()));
        assert!(inv.args.contains(&"q4k".to_string()));
        assert!(inv.args.contains(&"--down-q4k".to_string()));
    }

    #[test]
    fn down_q4k_false_omits_quant_flags() {
        let mut recipe = sample_recipe();
        recipe
            .spec
            .extractor
            .options
            .insert(DOWN_Q4K_OPTION_KEY.into(), serde_json::Value::Bool(false));
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(!inv.args.contains(&"--quant".to_string()));
        assert!(!inv.args.contains(&"--down-q4k".to_string()));
    }

    #[test]
    fn drop_gate_vectors_option_maps_to_flag() {
        let mut recipe = sample_recipe();
        recipe.spec.extractor.options.insert(
            DROP_GATE_VECTORS_OPTION_KEY.into(),
            serde_json::Value::Bool(true),
        );
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(inv.args.contains(&"--drop-gate-vectors".to_string()));
    }

    #[test]
    fn f32_dtype_adds_the_flag() {
        let mut recipe = sample_recipe();
        recipe.spec.extractor.dtype = StorageDtype::F32;
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(inv.args.contains(&"--f32".to_string()));
    }

    #[test]
    fn f16_dtype_omits_the_f32_flag() {
        let recipe = sample_recipe(); // f16 in the fixture
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(!inv.args.contains(&"--f32".to_string()));
    }

    #[test]
    fn always_passes_resume() {
        let recipe = sample_recipe();
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(inv.args.contains(&"--resume".to_string()));
    }

    #[test]
    fn unrecognised_option_keys_are_ignored_not_erroring() {
        let recipe = sample_recipe(); // has preserve_mtp_head: true, no CLI mapping
        let (full, cache) = dirs();
        let inv = invocation(&recipe, &full, &cache);
        assert!(!inv.args.iter().any(|a| a.contains("mtp")));
    }

    #[test]
    fn run_errors_with_stderr_on_failure() {
        let recipe = sample_recipe();
        let (full, cache) = dirs();
        let runner = MockRunner::new().expect(
            invocation(&recipe, &full, &cache),
            CommandOutput {
                status: 1,
                stderr: "out of disk".into(),
                ..Default::default()
            },
        );
        let err = run(&runner, &recipe, &full, &cache).unwrap_err();
        assert!(err.contains("out of disk"), "{err}");
    }
}
