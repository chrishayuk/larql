//! The declared-text-features reader.

use super::*;

/// Both knobs are read verbatim and credited by full path; a config that
/// declares neither yields no reading and no credit.
#[test]
fn reads_the_declared_knobs_and_credits_exactly_those_paths() {
    let config = serde_json::json!({
        "text_config": { "use_double_wide_mlp": false, "vocab_size_per_layer_input": 262144 }
    });
    let reading = read_text_features(&config).expect("declares features");
    assert_eq!(reading.features.double_wide_mlp, Some(false));
    assert_eq!(reading.features.per_layer_input_vocab, Some(262144));
    assert_eq!(
        reading.consumed_paths,
        [
            "text_config.use_double_wide_mlp",
            "text_config.vocab_size_per_layer_input"
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );
    assert!(
        read_text_features(&serde_json::json!({ "text_config": { "hidden_size": 8 } })).is_none()
    );
    assert!(read_text_features(&serde_json::json!({ "hidden_size": 8 })).is_none());
}
