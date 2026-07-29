//! `larql card render` — §9's Hub model-card generator: frontmatter
//! plus body, both generated from the manifest and recipe, never
//! hand-written.
//!
//! Scope today: this crate only has `Recipe` and the vindex-v1
//! manifest to draw on. [`VerificationReport`] and [`SliceSummary`]
//! (in [`types`]) are a best-guess shape for what §7/§8's build driver
//! will eventually produce — there's no driver yet to validate them
//! against, so treat their fields as provisional.

mod body;
mod frontmatter;
pub(crate) mod naming;
mod revision_tag;
mod types;

pub use revision_tag::revision_tag;
pub use types::{
    CardInputs, LogitMatchResult, ReconstructionResult, SliceSummary, VerificationReport,
};

/// Render a full Hub model card: YAML frontmatter delimited by `---`,
/// then the body.
pub fn render(inputs: &CardInputs) -> String {
    let tag = revision_tag::revision_tag(inputs.recipe, inputs.manifest, inputs.build_id);
    format!(
        "---\n{}\n---\n\n{}",
        frontmatter::render_frontmatter(inputs.recipe, inputs.manifest),
        body::render_body(inputs, &tag),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Recipe;

    #[test]
    fn render_produces_frontmatter_delimiters_and_a_body() {
        let recipe = Recipe::from_yaml(include_str!("../../testdata/gemma-3-4b-it.yaml")).unwrap();
        let manifest = larql_vindex_spec::test_fixtures::sample_manifest();
        let verification = VerificationReport {
            reconstruction: ReconstructionResult {
                layers_sampled: 8,
                max_abs_diff: 0.0,
                min_cosine: 1.0,
            },
            logit_match: LogitMatchResult {
                top1_agreement: 1.0,
                bits_per_char_drift: 0.0,
            },
            verified_from_hub: true,
        };
        let inputs = CardInputs {
            recipe: &recipe,
            manifest: &manifest,
            verification: &verification,
            slices: &[],
            build_id: "3f9a2c71aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        };

        let card = render(&inputs);

        assert!(card.starts_with("---\nlibrary_name: larql"));
        // Frontmatter block closes before the body starts.
        let mut parts = card.splitn(3, "---\n");
        assert_eq!(parts.next(), Some(""));
        assert!(parts.next().unwrap().contains("base_model:"));
        assert!(parts.next().unwrap().contains("# google/gemma-3-4b-it"));
    }
}
