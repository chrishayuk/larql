//! `larql recipe` — Vindex Factory recipe tooling (docs/vindex-factory.md
//! §3.1). This is the single implementation of recipe parsing,
//! structural validation, and `build_id` canonicalisation; the
//! `chuk-vindex-recipes` GitHub Action and the rig worker both call this
//! binary rather than reimplementing either, so they can't drift apart.

use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Subcommand)]
pub enum RecipeCommand {
    /// Structurally validate a recipe file (schema shape, required
    /// fields, value ranges). Does not check upstream existence, the
    /// licence allowlist, or Hub name collisions — those need network
    /// access / the recipes repo's `policy/` and run as later
    /// `validate.yml` steps.
    Validate(RecipeArgs),

    /// Compute a recipe's `build_id` — the content hash of the fields
    /// that determine its produced bytes (`source`, `extractor`,
    /// `outputs`). Prints the 64-char hex digest to stdout.
    BuildId(RecipeArgs),

    /// Estimate upstream download size, per-output size, executor
    /// class, and cost band (§6.1 step 4), so a recipe PR's approval is
    /// informed. Fetches the upstream repo's file listing and
    /// `config.json` over the network — the first `larql recipe`
    /// subcommand that does.
    Estimate(RecipeArgs),

    /// Run PREFLIGHT through RELEASE for a recipe (§7): fetch the
    /// pinned revision, extract, slice each declared output, verify
    /// checksums, publish private, then flip to public. Orchestrates
    /// this same `larql` binary's other subcommands as subprocesses.
    /// MIRROR and REGISTER are not run here — pipe the printed
    /// `BuildRecord` JSON to whatever external harness owns those.
    Build(BuildArgs),
}

#[derive(Args)]
pub struct RecipeArgs {
    /// Path to the recipe YAML file.
    pub recipe: PathBuf,
}

#[derive(Args)]
pub struct BuildArgs {
    /// Path to the recipe YAML file.
    pub recipe: PathBuf,

    /// Scratch working directory for the build (HF cache, extracted
    /// and sliced outputs). Defaults to a fresh temporary directory,
    /// removed once the build finishes — pass this to inspect the
    /// intermediate output after a failure.
    #[arg(long)]
    pub scratch_dir: Option<PathBuf>,
}

pub fn run(cmd: RecipeCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        RecipeCommand::Validate(args) => run_validate(args),
        RecipeCommand::BuildId(args) => run_build_id(args),
        RecipeCommand::Estimate(args) => run_estimate(args),
        RecipeCommand::Build(args) => run_build(args),
    }
}

fn load_recipe(path: &PathBuf) -> Result<larql_factory::Recipe, Box<dyn std::error::Error>> {
    let src =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let recipe = larql_factory::Recipe::from_yaml(&src)
        .map_err(|e| format!("parsing {}: {e}", path.display()))?;
    Ok(recipe)
}

fn run_validate(args: RecipeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recipe = load_recipe(&args.recipe)?;
    let errors = larql_factory::validate(&recipe);

    if errors.is_empty() {
        println!(
            "✓ {} is structurally valid ({})",
            args.recipe.display(),
            recipe.metadata.name
        );
        return Ok(());
    }

    println!("✗ {} — {} problem(s):", args.recipe.display(), errors.len());
    for err in &errors {
        println!("  - {err}");
    }
    std::process::exit(1);
}

fn run_build_id(args: RecipeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recipe = load_recipe(&args.recipe)?;
    println!("{}", larql_factory::build_id(&recipe));
    Ok(())
}

fn run_estimate(args: RecipeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recipe = load_recipe(&args.recipe)?;
    let estimate = larql_factory::estimate_size(&recipe)?;
    println!("{}", serde_json::to_string_pretty(&estimate)?);
    Ok(())
}

fn run_build(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recipe = load_recipe(&args.recipe)?;
    let runner = larql_factory::SubprocessRunner::current_exe()
        .map_err(|e| format!("resolving the larql binary to self-invoke: {e}"))?;

    let record = match &args.scratch_dir {
        Some(dir) => larql_factory::run_build(&runner, &recipe, dir),
        None => {
            let tmp =
                tempfile::tempdir().map_err(|e| format!("creating a scratch directory: {e}"))?;
            larql_factory::run_build(&runner, &recipe, tmp.path())
        }
    };

    println!("{}", serde_json::to_string_pretty(&record)?);
    if !record.passed() {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    // Only the two non-exiting paths are testable in-process:
    // `run_validate`'s failure branch calls `std::process::exit(1)`,
    // which would kill the test runner rather than return a `Result`.
    const VALID_RECIPE: &str = r#"
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
"#;

    fn write_temp_recipe(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp recipe file");
        file.write_all(contents.as_bytes())
            .expect("write temp recipe file");
        file
    }

    #[test]
    fn load_recipe_parses_a_valid_file() {
        let file = write_temp_recipe(VALID_RECIPE);
        let recipe = load_recipe(&file.path().to_path_buf()).unwrap();
        assert_eq!(recipe.metadata.name, "tiny-model");
    }

    #[test]
    fn load_recipe_errors_on_missing_file() {
        let err = load_recipe(&PathBuf::from("/nonexistent/recipe.yaml")).unwrap_err();
        assert!(err.to_string().contains("reading"));
    }

    #[test]
    fn load_recipe_errors_on_malformed_yaml() {
        let file = write_temp_recipe("not: [valid, yaml: shape");
        let err = load_recipe(&file.path().to_path_buf()).unwrap_err();
        assert!(err.to_string().contains("parsing"));
    }

    #[test]
    fn run_validate_succeeds_on_a_valid_recipe() {
        let file = write_temp_recipe(VALID_RECIPE);
        let args = RecipeArgs {
            recipe: file.path().to_path_buf(),
        };
        assert!(run_validate(args).is_ok());
    }

    #[test]
    fn run_build_id_prints_and_succeeds() {
        let file = write_temp_recipe(VALID_RECIPE);
        let args = RecipeArgs {
            recipe: file.path().to_path_buf(),
        };
        assert!(run_build_id(args).is_ok());
    }

    #[test]
    fn run_dispatches_build_id_variant() {
        let file = write_temp_recipe(VALID_RECIPE);
        let args = RecipeArgs {
            recipe: file.path().to_path_buf(),
        };
        assert!(run(RecipeCommand::BuildId(args)).is_ok());
    }

    #[test]
    fn run_dispatches_validate_variant() {
        let file = write_temp_recipe(VALID_RECIPE);
        let args = RecipeArgs {
            recipe: file.path().to_path_buf(),
        };
        assert!(run(RecipeCommand::Validate(args)).is_ok());
    }

    #[test]
    #[ignore = "hits the real HuggingFace API — network-dependent, not for CI; run with --ignored to smoke-test the live path"]
    fn run_estimate_against_a_real_public_repo() {
        // sshleifer/tiny-gpt2 — a small, stable, unauthenticated public
        // test fixture repo. VALID_RECIPE's chrishayuk/tiny-model
        // target doesn't exist on the real Hub, so this test points at
        // a real one instead.
        let recipe_yaml = VALID_RECIPE
            .replace("chrishayuk/tiny-model", "sshleifer/tiny-gpt2")
            .replace(
                "9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82",
                "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
            );
        let file = write_temp_recipe(&recipe_yaml);
        let args = RecipeArgs {
            recipe: file.path().to_path_buf(),
        };
        assert!(run_estimate(args).is_ok());
    }
}
