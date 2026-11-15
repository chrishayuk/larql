//! What CAPTURE refuses, and that a refusal writes nothing.

use larql_lql::{RepresentationSource, Session};

use crate::support::*;

/// Run a capture that must be refused; the refusal must mention `needle` and leave no file behind.
fn refused(s: &mut Session, fx: &Fixture, clauses: &str, needle: &str) {
    let out = fx.path("refused.npy");
    let err = try_capture(s, clauses, &out).unwrap_err();
    assert!(err.contains(needle), "{clauses}: {err}");
    assert!(!out.exists(), "{clauses}: a refusal wrote the array");
    assert!(
        !fx.path("refused.npy.json").exists(),
        "{clauses}: a refusal wrote the sidecar"
    );
}

#[test]
fn layers_are_named_exactly_where_the_site_has_them() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let p = PROMPT;
    refused(&mut s, &fx, &format!("residual FOR \"{p}\""), "AT LAYER");
    refused(
        &mut s,
        &fx,
        &format!("logits AT LAYER 0 FOR \"{p}\""),
        "takes no AT LAYER",
    );
    refused(
        &mut s,
        &fx,
        &format!("final AT LAYERS (0) FOR \"{p}\""),
        "takes no AT LAYER",
    );
    refused(
        &mut s,
        &fx,
        &format!("residual AT LAYER 2 FOR \"{p}\""),
        "layer 2 does not exist",
    );
    refused(
        &mut s,
        &fx,
        &format!("ffn.input AT LAYERS (0, 0) FOR \"{p}\""),
        "listed twice",
    );
}

#[test]
fn positions_and_tokens_must_be_in_the_sequence() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let p = PROMPT;
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{p}\" POSITIONS (5)"),
        "outside the 5-token sequence",
    );
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{p}\" POSITIONS (1, 1)"),
        "listed twice",
    );
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{p}\" POSITIONS REPLY"),
        "needs a continuation",
    );
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{p}\" CONTINUE TOKENS ({VOCAB})"),
        "outside the embedding's vocabulary",
    );
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{p}\" CONTINUE \" \""),
        "encodes to no tokens",
    );
}

#[test]
fn nothing_is_overwritten() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("once.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &out);
    let before = std::fs::read(&out).unwrap();
    let err = try_capture(&mut s, &format!("final FOR \"{PROMPT}\""), &out).unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    assert_eq!(std::fs::read(&out).unwrap(), before);

    // A sidecar in the way is refused too, and so is an output not named .npy.
    let blocked = fx.path("blocked.npy");
    std::fs::write(fx.path("blocked.npy.json"), b"{}").unwrap();
    let err = try_capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &blocked).unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    assert!(!blocked.exists());
    let err = try_capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\""),
        &fx.path("x.bin"),
    )
    .unwrap_err();
    assert!(err.contains("is not a .npy file"), "{err}");
}

#[test]
fn it_refuses_where_infer_refuses() {
    // No tokenizer.
    let fx = Fixture::tokenless();
    let mut s = fx.session();
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{PROMPT}\""),
        "tokenizer",
    );

    // Pending stored-weight ops under a source that may quantise at load: INFER is refused, and so is CAPTURE.
    let fx = Fixture::new();
    let mut s = Session::new();
    s.set_representation_source(RepresentationSource::Auto);
    run(&mut s, &format!("USE \"{}\";", q(&fx.container)));
    let free = fx.path("auto.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &free);
    assert_eq!(
        sidecar(&free)["execution"]["binding"]["representation_source"],
        "auto"
    );
    run(&mut s, STAGE);
    let infer = try_run(&mut s, &format!("INFER \"{PROMPT}\" TOP 1;")).unwrap_err();
    assert!(infer.contains("cannot be executed"), "{infer}");
    refused(
        &mut s,
        &fx,
        &format!("logits FOR \"{PROMPT}\""),
        "cannot be executed",
    );
    refused(
        &mut s,
        &fx,
        &format!("ffn.down_input AT LAYER 0 FOR \"{PROMPT}\""),
        "cannot be executed",
    );
}

#[test]
fn it_needs_a_bound_vindex3_container() {
    let mut s = Session::new();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x.npy");
    let err = try_capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &out).unwrap_err();
    assert!(err.contains("No backend loaded"), "{err}");
    assert!(!out.exists());
}
