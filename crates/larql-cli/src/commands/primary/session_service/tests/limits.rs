//! The resource limits: each one refuses beyond it, before anything executes (or, for the array bytes of a
//! capture, before anything is sent or charged).

use serde_json::json;

use super::*;

fn with_limits(fx: &Fixture, limits: Value) -> Service {
    let mut policy = fx.policy(5);
    policy["limits"] = limits;
    fx.start_with(&policy).unwrap()
}

fn refused(svc: &mut Service, op: &str, args: Value, needle: &str) {
    let (kind, reason) = err(svc, op, args);
    assert_eq!(kind, "refused", "{reason}");
    assert!(reason.contains(needle), "{reason}");
}

#[test]
fn every_limit_refuses_beyond_it() {
    let fx = Fixture::new();
    let mut svc = with_limits(
        &fx,
        json!({
            "max_generate_tokens": 2, "max_capture_positions": 3, "max_capture_layers": 1,
            "max_sequence_tokens": 6, "max_infer_top": 3,
        }),
    );
    refused(
        &mut svc,
        "generate",
        json!({"prompt": PROMPT, "n": 3}),
        "max_generate_tokens",
    );
    refused(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "generate": 3}),
        "max_generate_tokens",
    );
    refused(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": "ALL"}),
        "max_capture_positions",
    );
    refused(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": [0, 1, 2, 3]}),
        "max_capture_positions",
    );
    refused(
        &mut svc,
        "capture",
        json!({"site": "residual", "layers": [0, 1], "prompt": PROMPT}),
        "max_capture_layers",
    );
    refused(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "continue_tokens": [1, 2]}),
        "max_sequence_tokens",
    );
    refused(
        &mut svc,
        "generate",
        json!({"prompt": "The capital of France is Paris", "n": 1}),
        "max_sequence_tokens",
    );
    refused(
        &mut svc,
        "infer",
        json!({"prompt": PROMPT, "top": 4}),
        "max_infer_top",
    );
    // At the limits, answered.
    ok(&mut svc, "generate", json!({"prompt": PROMPT, "n": 1}));
    ok(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "continue_text": CONTINUATION, "positions": "REPLY"}),
    );
    ok(
        &mut svc,
        "capture",
        json!({"site": "residual", "layers": [1], "prompt": PROMPT, "positions": "PROMPT_END"}),
    );
    ok(&mut svc, "infer", json!({"prompt": PROMPT, "top": 3}));
}

#[test]
fn the_response_bytes_limit_refuses_reads_and_captures_without_charging() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = with_limits(&fx, json!({"max_response_bytes": 512}));
    refused(
        &mut svc,
        "tensor",
        json!({"name": TENSOR}),
        "max_response_bytes",
    );
    ok(
        &mut svc,
        "tensor",
        json!({"name": TENSOR, "rows": [0], "cols": [0, 1]}),
    );
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": "ALL"}),
    );
    assert_eq!(r["error"]["kind"], "refused");
    assert_eq!(
        (r["edited"].clone(), r["counted"].clone()),
        (json!(true), json!(false))
    );
    assert_eq!(r["budget"]["remaining"], 5);
}

#[test]
fn the_patch_bytes_limit_refuses_large_patches() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = with_limits(&fx, json!({"max_patch_bytes": 16}));
    refused(
        &mut svc,
        "apply_patch",
        json!({"path": "edit.vlp"}),
        "max_patch_bytes",
    );
}
