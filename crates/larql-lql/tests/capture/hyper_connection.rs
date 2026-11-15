//! A hyper-connected component carries a bundle of residual streams per position, not one `[hidden]` row: CAPTURE
//! of a per-layer residual plane refuses there rather than choosing what a bundle means, while the FFN's normed
//! input, one row per position, is captured.

use larql_vindex::format::vindex3::fixtures_hyper_connection::{
    hyper_connected_container, HC_HIDDEN, HC_VOCAB,
};

use crate::support::*;

/// A word-level tokenizer over the miniature's vocabulary: token `i` is spelt `t<i>`.
fn write_tokenizer(container: &std::path::Path) {
    let vocab: serde_json::Map<String, serde_json::Value> =
        (0..HC_VOCAB).map(|i| (format!("t{i}"), i.into())).collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    std::fs::write(container.join("tokenizer.json"), tokenizer.to_string()).unwrap();
}

#[test]
fn a_residual_of_streams_is_refused_and_nothing_is_written() {
    let work = tempfile::tempdir().unwrap();
    let container = hyper_connected_container(work.path());
    write_tokenizer(&container);
    let mut s = session_on(&container);
    for site in ["residual", "post_attention"] {
        let out = work.path().join(format!("{site}.npy"));
        let err =
            try_capture(&mut s, &format!("{site} AT LAYER 1 FOR \"t1 t2 t3\""), &out).unwrap_err();
        assert!(
            err.contains("layer 1's residual has no single [hidden] row per position to capture"),
            "{site}: {err}"
        );
        assert!(err.contains("hyper-connected"), "{site}: {err}");
        assert!(!out.exists(), "{site}: a refusal wrote the array");
        assert!(
            !work.path().join(format!("{site}.npy.json")).exists(),
            "{site}: a refusal wrote the sidecar"
        );
    }
}

#[test]
fn the_ffn_input_is_one_row_per_position_and_is_captured() {
    let work = tempfile::tempdir().unwrap();
    let container = hyper_connected_container(work.path());
    write_tokenizer(&container);
    let mut s = session_on(&container);
    let out = work.path().join("ffn_input.npy");
    capture(&mut s, "ffn.input AT LAYER 1 FOR \"t1 t2 t3\"", &out);
    let npy = read_npy(&out);
    assert_eq!(npy.shape, vec![1, HC_HIDDEN]);
    assert!(npy.values.iter().all(|v| v.is_finite()));
}

#[test]
fn per_position_logits_are_refused_by_the_executor() {
    let work = tempfile::tempdir().unwrap();
    let container = hyper_connected_container(work.path());
    write_tokenizer(&container);
    let mut s = session_on(&container);
    let out = work.path().join("logits.npy");
    let err = try_capture(&mut s, "logits FOR \"t1 t2 t3\"", &out).unwrap_err();
    assert!(err.contains("capture pass failed"), "{err}");
    assert!(err.contains("single-stream"), "{err}");
    assert!(!out.exists(), "a refusal wrote the array");
}
