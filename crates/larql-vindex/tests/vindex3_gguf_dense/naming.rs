//! The naming port against outputs of llama.cpp's own
//! `Metadata.get_model_id_components` / `id_to_title` / `size_label`
//! (gguf-py at b9430), recorded from the reference implementation.

use larql_vindex::format::vindex3::gguf::dense::general::naming_entries;
use larql_vindex::format::vindex3::gguf::dense::naming::{
    id_to_title, naming_from_id, size_label, ModelNaming,
};

fn n(
    name: &str,
    org: Option<&str>,
    basename: Option<&str>,
    finetune: Option<&str>,
    version: Option<&str>,
    size: Option<&str>,
) -> ModelNaming {
    let s = |v: Option<&str>| v.map(String::from);
    ModelNaming {
        name: Some(name.into()),
        organization: s(org),
        basename: s(basename),
        finetune: s(finetune),
        version: s(version),
        size_label: s(size),
    }
}

#[test]
fn naming_matches_the_reference_heuristic() {
    let hex = "093f9f388b31de276ce2de164bdc2081324b9767";
    let cases: Vec<(&str, i64, ModelNaming)> = vec![
        (
            "gemma-3-tiny-it",
            3_880_099_328,
            n("Gemma 3 Tiny It", None, None, None, None, None),
        ),
        (
            hex,
            3_880_099_328,
            n(hex, None, None, Some(hex), None, None),
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            6_738_415_616,
            n(
                "Llama 2 7b Hf",
                Some("Meta Llama"),
                Some("Llama-2"),
                Some("hf"),
                None,
                Some("7B"),
            ),
        ),
        (
            "Mixtral-8x7B-Instruct-v0.1",
            46_702_792_704,
            n(
                "Mixtral 8x7B Instruct v0.1",
                None,
                Some("Mixtral"),
                Some("Instruct"),
                Some("v0.1"),
                Some("8x7B"),
            ),
        ),
        (
            "bloom-7b1",
            7_069_016_064,
            n("Bloom 7b1", None, Some("bloom"), None, None, Some("7.1B")),
        ),
        (
            "phi-3-mini-128k-instruct",
            3_821_079_552,
            n(
                "Phi 3 Mini 128k Instruct",
                None,
                Some("phi-3"),
                Some("128k-instruct"),
                None,
                Some("mini"),
            ),
        ),
        (
            "my model name",
            0,
            n("My Model Name", None, None, None, None, None),
        ),
        (
            "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
            32_763_876_352,
            n(
                "Qwen2.5 Coder 32B Instruct Q4_K_M",
                None,
                Some("Qwen2.5-Coder"),
                Some("Instruct"),
                None,
                Some("32B"),
            ),
        ),
        (
            "./x/model-small-q8_0",
            100,
            n(
                "X/Model Small Q8_0",
                None,
                Some("x/model"),
                None,
                None,
                Some("small"),
            ),
        ),
        (
            "tiny-llama-1.1B-chat-v1.0-bf16",
            1_100_048_384,
            n(
                "Tiny Llama 1.1B Chat v1.0 Bf16",
                None,
                Some("tiny-llama"),
                Some("chat"),
                Some("v1.0"),
                Some("1.1B"),
            ),
        ),
        (
            "llama-2-7b-lora",
            -40_000_000,
            n(
                "Llama 2 7b Lora",
                None,
                Some("llama-2"),
                None,
                None,
                Some("7B"),
            ),
        ),
    ];
    for (id, params, want) in cases {
        assert_eq!(naming_from_id(id, params), want, "naming of `{id}`");
    }
}

#[test]
fn org_titles_follow_the_reference() {
    // `meta-llama` → "Meta Llama": hyphens become spaces, words title-cased.
    assert_eq!(id_to_title("meta-llama"), "Meta Llama");
    assert_eq!(id_to_title("v2-model"), "v2 Model");
    assert_eq!(id_to_title("iter3-x"), "Iter3 X");
}

#[test]
fn size_labels_match_the_reference() {
    for (params, want) in [
        (3_880_099_328u64, "3.9B"),
        (999, "1.0K"),
        (1_500_000, "1.5M"),
        (270_000_000, "270M"),
        (12_000_000_000, "12B"),
        (2_000_000_000_000, "2.0T"),
        (95, "0.10K"),
    ] {
        assert_eq!(size_label(params), want, "size label of {params}");
    }
}

#[test]
fn naming_entries_fall_back_to_the_parameter_count() {
    let keys: Vec<(String, String)> = naming_entries("gemma-3-tiny-it", 3_880_099_328)
        .into_iter()
        .map(|(k, v)| (k, v.as_str().unwrap().to_string()))
        .collect();
    assert_eq!(
        keys,
        vec![
            ("general.name".into(), "Gemma 3 Tiny It".into()),
            ("general.size_label".into(), "3.9B".into()),
        ]
    );
    let labelled = naming_entries("Mixtral-8x7B-Instruct-v0.1", 1);
    assert!(labelled
        .iter()
        .any(|(k, v)| k == "general.size_label" && v.as_str() == Some("8x7B")));
    assert!(naming_entries("x", 0)
        .iter()
        .all(|(k, _)| k != "general.size_label"));
}
