//! YAML front-end contract (#663).
//!
//! Recipes are parsed with `serde-saphyr`, which replaced the deprecated
//! `serde_yaml` 0.9 (its `unsafe-libyaml` backend is RUSTSEC-2024-0449).
//! These tests pin what the swap must not change: the checked-in
//! fixtures produce the same `build_id` they produced under `serde_yaml`,
//! the recipe a card inlines parses back to the recipe it came from, and
//! malformed input is refused with a located error rather than a panic.

use larql_factory::{
    build_id, render_card, CardInputs, LogitMatchResult, Recipe, ReconstructionResult,
    VerificationReport,
};

const GEMMA_RECIPE_YAML: &str = include_str!("../testdata/gemma-3-4b-it.yaml");
const SINGLE_OUTPUT_RECIPE_YAML: &str = include_str!("../testdata/tiny-model-single-output.yaml");

/// `build_id` of `testdata/gemma-3-4b-it.yaml`, recorded with `serde_yaml`
/// 0.9 before the parser swap.
const GEMMA_BUILD_ID: &str = "398f1b8e29adecf2ef3838748558ae6b82b292b96ed8ab412725886e4194e30a";

/// `build_id` of `testdata/tiny-model-single-output.yaml`, recorded with
/// `serde_yaml` 0.9 before the parser swap.
const SINGLE_OUTPUT_BUILD_ID: &str =
    "b9534b92339988183bdde6a3b31aca3e05ccf5d57856c5cd1ebf704fe6adfb35";

/// Fence that opens the inlined recipe in a rendered card body.
const RECIPE_FENCE_OPEN: &str = "## Recipe\n\n```yaml\n";
/// Fence that closes it.
const RECIPE_FENCE_CLOSE: &str = "```";

/// Every parser error must say where the problem is.
const LOCATION_MARKER: &str = "line ";

/// Revision tag passed to the card renderer; its value is irrelevant here.
const BUILD_ID_PLACEHOLDER: &str = "deadbeef";

fn fixtures() -> [(&'static str, &'static str, &'static str); 2] {
    [
        ("gemma-3-4b-it.yaml", GEMMA_RECIPE_YAML, GEMMA_BUILD_ID),
        (
            "tiny-model-single-output.yaml",
            SINGLE_OUTPUT_RECIPE_YAML,
            SINGLE_OUTPUT_BUILD_ID,
        ),
    ]
}

fn as_json(recipe: &Recipe) -> serde_json::Value {
    serde_json::to_value(recipe).expect("a parsed recipe is JSON-serialisable")
}

fn passing_verification() -> VerificationReport {
    VerificationReport {
        reconstruction: ReconstructionResult {
            layers_sampled: 1,
            max_abs_diff: 0.0,
            min_cosine: 1.0,
        },
        logit_match: LogitMatchResult {
            top1_agreement: 1.0,
            bits_per_char_drift: 0.0,
        },
        verified_from_hub: true,
    }
}

#[test]
fn fixtures_keep_their_pre_swap_build_ids() {
    for (name, src, expected) in fixtures() {
        let recipe = Recipe::from_yaml(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(build_id(&recipe), expected, "{name}");
    }
}

#[test]
fn the_recipe_inlined_in_a_card_parses_back_to_the_same_recipe() {
    let manifest = larql_vindex_spec::test_fixtures::sample_manifest();
    let verification = passing_verification();
    for (name, src, _) in fixtures() {
        let recipe = Recipe::from_yaml(src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let card = render_card(&CardInputs {
            recipe: &recipe,
            manifest: &manifest,
            verification: &verification,
            slices: &[],
            build_id: BUILD_ID_PLACEHOLDER,
        });
        let start = card
            .find(RECIPE_FENCE_OPEN)
            .unwrap_or_else(|| panic!("{name}: card has no recipe block"))
            + RECIPE_FENCE_OPEN.len();
        let len = card[start..]
            .find(RECIPE_FENCE_CLOSE)
            .unwrap_or_else(|| panic!("{name}: recipe block is not closed"));
        let inlined = &card[start..start + len];
        assert!(!inlined.is_empty(), "{name}: recipe block is empty");

        let reparsed = Recipe::from_yaml(inlined)
            .unwrap_or_else(|e| panic!("{name}: inlined recipe does not parse: {e}"));
        assert_eq!(as_json(&reparsed), as_json(&recipe), "{name}");
    }
}

#[test]
fn malformed_yaml_is_refused_with_a_located_error() {
    let cases = [
        (
            "unclosed flow sequence",
            "not: [valid, yaml: shape".to_owned(),
        ),
        (
            "tab indentation",
            SINGLE_OUTPUT_RECIPE_YAML.replace("  name: tiny-model\n", "\tname: tiny-model\n"),
        ),
        (
            "duplicate key",
            SINGLE_OUTPUT_RECIPE_YAML.replace(
                "  name: tiny-model\n",
                "  name: tiny-model\n  name: other\n",
            ),
        ),
        (
            "two documents",
            format!("{SINGLE_OUTPUT_RECIPE_YAML}---\n{SINGLE_OUTPUT_RECIPE_YAML}"),
        ),
        (
            "wrong type",
            SINGLE_OUTPUT_RECIPE_YAML.replace("    - preset: full\n", "    5\n"),
        ),
        (
            "missing required field",
            SINGLE_OUTPUT_RECIPE_YAML.replace("    licence: mit\n", ""),
        ),
        (
            // `serde_yaml` silently turned this into `null` in `options`.
            "non-finite float",
            SINGLE_OUTPUT_RECIPE_YAML.replace(
                "    dtype: f16\n",
                "    dtype: f16\n    options:\n      scale: .inf\n",
            ),
        ),
        ("empty input", String::new()),
    ];
    for (name, src) in cases {
        let err = match Recipe::from_yaml(&src) {
            Ok(recipe) => panic!("{name}: accepted as {recipe:?}"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains(LOCATION_MARKER),
            "{name}: error has no location: {err}"
        );
    }
}
