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
        (json!(false), json!(false))
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

const ALL_SITES: [(&str, Option<usize>); 7] = [
    ("logits", None),
    ("final", None),
    ("residual", Some(1)),
    ("post_attention", Some(1)),
    ("ffn.input", Some(1)),
    ("ffn.down_input", Some(1)),
    ("ffn.down_output", Some(1)),
];

fn site_args(site: &str, layer: Option<usize>) -> Value {
    let mut args = json!({"site": site, "prompt": PROMPT, "positions": "ALL"});
    if let Some(l) = layer {
        args["layer"] = json!(l);
    }
    args
}

/// On a dense, single-stream program every site is bounded before it runs, so requiring a bound refuses none.
#[test]
fn every_site_of_a_dense_program_is_bounded_and_allowed_under_either_setting() {
    for require in [true, false] {
        let fx = Fixture::new();
        let mut policy = fx.policy(5);
        policy["require_capture_bound"] = json!(require);
        let mut svc = fx.start_with(&policy).unwrap();
        for (site, layer) in ALL_SITES {
            let r = call(&mut svc, "capture", site_args(site, layer));
            assert_eq!(r["ok"], true, "{site} require={require}: {r}");
        }
        assert_eq!(
            ok(&mut svc, "hello", json!({}))["require_capture_bound"],
            require
        );
    }
}

/// A site the plan gives no width: refused up front when a bound is required (nothing admitted, nothing
/// charged), left to the post-run check when it is not.
#[test]
fn an_unbounded_site_is_refused_up_front_when_required_and_checked_after_running_when_not() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut policy = fx.policy(5);
    policy["limits"] = json!({"max_response_bytes": 512});
    let mut svc = fx.start_with(&policy).unwrap();
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    svc.forget_capture_width = true;
    let r = call(&mut svc, "capture", site_args("residual", Some(1)));
    assert_eq!(r["error"]["kind"], "refused");
    let reason = r["error"]["reason"].as_str().unwrap();
    assert!(reason.contains("site residual"), "{reason}");
    assert!(reason.contains("require_capture_bound"), "{reason}");
    assert_eq!(
        (r["counted"].clone(), r["budget"]["remaining"].clone()),
        (json!(false), json!(5))
    );
    assert!(fx.records().iter().all(|r| r["event"] != "admitted"));

    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut policy = fx.policy(5);
    policy["limits"] = json!({"max_response_bytes": 512});
    policy["require_capture_bound"] = json!(false);
    let mut svc = fx.start_with(&policy).unwrap();
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    svc.forget_capture_width = true;
    let r = call(&mut svc, "capture", site_args("residual", Some(1)));
    // It ran (and was charged at admission), and its array was then refused on size.
    assert_eq!(r["error"]["kind"], "refused");
    assert!(
        r["error"]["reason"]
            .as_str()
            .unwrap()
            .contains("the response's array bytes"),
        "{r}"
    );
    assert_eq!(
        (r["counted"].clone(), r["budget"]["remaining"].clone()),
        (json!(true), json!(4))
    );
    // A small one still passes.
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "residual", "layer": 1, "prompt": PROMPT, "positions": [0]}),
    );
    assert_eq!(r["ok"], true, "{r}");
}

#[test]
fn the_bound_check_alone() {
    use super::super::ops::capture_bound;
    use larql_lql::operations::CaptureSite;
    assert!(capture_bound(CaptureSite::Residual, None, 1, 1, true, 10).is_err());
    assert!(capture_bound(CaptureSite::Residual, None, 1, 1, false, 10).is_ok());
    // 2 layers x 3 positions x 4 values x 4 bytes + 256 header bytes = 352.
    assert!(capture_bound(CaptureSite::Residual, Some(4), 2, 3, true, 352).is_ok());
    assert!(capture_bound(CaptureSite::Residual, Some(4), 2, 3, true, 351).is_err());
}
