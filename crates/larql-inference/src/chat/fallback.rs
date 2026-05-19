//! Hardcoded chat templates for instruct-tuned families whose upstream
//! `tokenizer_config.json` doesn't ship one.
//!
//! The primary path always tries the HF-published template first
//! ([`super::source::try_hf_template`]). This module only fires when that
//! path returns `applied=false` or errors, AND the caller supplied a
//! `model_hint` that clearly names a chat/instruct variant we recognise.
//!
//! Principle: **only match explicit instruct variants, never base models.**
//! Wrapping a base model like `Llama-2-7b-hf` in `[INST]` markers degrades
//! its output — those tokens aren't in the base model's training
//! distribution. The detection guard below requires both an instruct-tag
//! substring (`-chat`, `-Instruct`, `-it`) AND a family substring
//! (`llama-2`, `mistral`, …), so a hypothetical `random-base-it` wouldn't
//! trip it.
//!
//! Adding a family: pick up the model card's canonical template, port it
//! to Jinja using the standard context (`messages`, `add_generation_prompt`,
//! `bos_token`), and add an arm below plus a unit test. Keep it single-turn
//! — multi-turn rendering is orthogonal and lives in the render layer.

/// Return `(human_label, jinja_template)` for a recognised instruct family,
/// or `None` if the hint doesn't match anything we've hardcoded. The
/// template is rendered through the same minijinja pipeline as HF
/// templates, so it has access to the full context machinery (pycompat,
/// `bos_token`, …).
pub(crate) fn fallback_template_for(model_hint: &str) -> Option<(&'static str, &'static str)> {
    let hint = model_hint.to_ascii_lowercase();

    if !is_instruct_hint(&hint) {
        return None;
    }

    // Llama-2-chat — Meta's `[INST] … [/INST]` format.
    if hint.contains("llama-2") && hint.contains("-chat") {
        // Single-turn flavour. BOS is prepended by the tokenizer's
        // post-processor, not embedded in the template.
        return Some((
            "llama-2-chat",
            "[INST] {{ messages[0]['content'] }} [/INST]",
        ));
    }

    // Mistral-Instruct — same `[INST]…[/INST]` surface as Llama-2 for the
    // single-turn case. Differs in multi-turn (no `<<SYS>>` system wrap);
    // not relevant here.
    if hint.contains("mistral") && (hint.contains("-instruct") || hint.contains("_instruct")) {
        return Some((
            "mistral-instruct",
            "[INST] {{ messages[0]['content'] }} [/INST]",
        ));
    }

    None
}

/// Heuristic: does the hint name an instruct/chat variant? Requires one of
/// the common tag substrings. This is a gate, not a family matcher — the
/// per-family checks below still need to pass.
fn is_instruct_hint(hint_lc: &str) -> bool {
    hint_lc.contains("-chat")
        || hint_lc.contains("-instruct")
        || hint_lc.contains("_instruct")
        || hint_lc.contains("-it")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn matches_llama2_chat() {
        let (label, tmpl) = fallback_template_for("meta-llama/Llama-2-7b-chat-hf").unwrap();
        assert_eq!(label, "llama-2-chat");
        assert!(tmpl.contains("[INST]"));
    }

    #[test]
    fn matches_mistral_instruct() {
        let (label, tmpl) = fallback_template_for("mistralai/Mistral-7B-Instruct-v0.3").unwrap();
        assert_eq!(label, "mistral-instruct");
        assert!(tmpl.contains("[INST]"));
    }

    #[test]
    fn base_llama2_rejected() {
        assert!(fallback_template_for("meta-llama/Llama-2-7b-hf").is_none());
    }

    #[test]
    fn base_mistral_rejected() {
        assert!(fallback_template_for("mistralai/Mistral-7B-v0.1").is_none());
    }

    #[test]
    fn unknown_instruct_family_rejected() {
        // Instruct-tag satisfied but family doesn't match any arm.
        // Better to pass through raw than guess the wrong template.
        assert!(fallback_template_for("unknown/Random-7B-Instruct").is_none());
    }

    #[test]
    fn hint_is_case_insensitive() {
        // HF repo paths are mixed-case (`meta-llama/Llama-2-7b-Chat-HF`
        // for instance). The match logic lowercases first.
        assert!(fallback_template_for("META-LLAMA/LLAMA-2-7B-CHAT-HF").is_some());
    }
}
