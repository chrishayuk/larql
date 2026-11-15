//! The budget: executions of the base model are free; every infer, generate and capture while a patch is
//! applied or a weight op is staged counts one; once spent, those are refused (and logged), and base calls
//! are still answered. Removing the patch (or discarding the staged op) makes calls free again, and the
//! base capture after removal is bit-identical to the base before.

use serde_json::json;

use super::*;

fn logits_all(svc: &mut Service) -> Value {
    call(
        svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": "ALL"}),
    )
}

fn remaining(r: &Value) -> u64 {
    r["budget"]["remaining"].as_u64().unwrap()
}

#[test]
fn base_executions_are_free() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    for _ in 0..3 {
        let r = logits_all(&mut svc);
        assert_eq!(
            (r["ok"].clone(), r["edited"].clone(), r["counted"].clone()),
            (json!(true), json!(false), json!(false))
        );
        assert_eq!(remaining(&r), 1);
    }
    let r = call(&mut svc, "infer", json!({"prompt": PROMPT}));
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(false), 1));
    let r = call(&mut svc, "generate", json!({"prompt": PROMPT, "n": 2}));
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(false), 1));
}

#[test]
fn capture_infer_and_generate_under_a_patch_each_count_one_then_are_refused() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(3);
    let base = logits_all(&mut svc);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));

    // A capture with several positions and layers is still one execution.
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "residual", "layers": [0, 1], "prompt": PROMPT, "positions": "ALL"}),
    );
    assert_eq!(
        (r["ok"].clone(), r["edited"].clone(), r["counted"].clone()),
        (json!(true), json!(true), json!(true))
    );
    assert_eq!(remaining(&r), 2);
    let r = call(&mut svc, "infer", json!({"prompt": PROMPT}));
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(true), 1));
    let r = call(&mut svc, "generate", json!({"prompt": PROMPT, "n": 3}));
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(true), 0));

    // Spent: every edited execution is refused, and the refusal is logged.
    for (op, args) in [
        ("capture", json!({"site": "logits", "prompt": PROMPT})),
        ("infer", json!({"prompt": PROMPT})),
        ("generate", json!({"prompt": PROMPT, "n": 1})),
    ] {
        let r = call(&mut svc, op, args);
        assert_eq!(r["ok"], false, "{op}");
        assert_eq!(r["error"]["kind"], "budget_exhausted");
        assert_eq!(
            (r["edited"].clone(), r["counted"].clone()),
            (json!(true), json!(false))
        );
        let last = fx.records().pop().unwrap();
        assert_eq!(last["seq"], r["seq"]);
        assert_eq!(last["op"], op);
        assert_eq!(last["ok"], false);
        assert_eq!(last["error"]["kind"], "budget_exhausted");
        assert_eq!(last["edited"], true);
        assert_eq!(last["budget_remaining"], 0);
    }
    // Reads that do not execute stay answered.
    ok(&mut svc, "tensor", json!({"name": TENSOR, "rows": [0]}));
    ok(&mut svc, "tokenize", json!({"text": PROMPT}));

    // Removing the patch makes executions free again, and the base is the base, bit for bit.
    ok(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    let after = logits_all(&mut svc);
    assert_eq!(
        (
            after["ok"].clone(),
            after["edited"].clone(),
            after["counted"].clone()
        ),
        (json!(true), json!(false), json!(false))
    );
    assert_eq!(
        npy_bytes(&after["result"]["array"]),
        npy_bytes(&base["result"]["array"])
    );
    assert_eq!(after["result"]["provenance"]["executes"], "stored");
    assert_eq!(after["result"]["provenance"]["patches"], json!([]));
}

#[test]
fn a_staged_weight_op_counts_until_it_is_discarded() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": DELTA}),
    );
    let r = call(&mut svc, "infer", json!({"prompt": PROMPT}));
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(true), 0));
    let r = logits_all(&mut svc);
    assert_eq!(r["error"]["kind"], "budget_exhausted");
    ok(&mut svc, "discard_weight_edits", json!({}));
    let r = logits_all(&mut svc);
    assert_eq!(
        (r["ok"].clone(), r["counted"].clone()),
        (json!(true), json!(false))
    );
}

#[test]
fn a_zero_budget_still_answers_the_base_model_and_the_budget_op() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(0);
    assert_eq!(
        ok(&mut svc, "budget", json!({})),
        json!({"limit": 0, "used": 0, "remaining": 0})
    );
    assert_eq!(logits_all(&mut svc)["ok"], true);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    assert_eq!(logits_all(&mut svc)["error"]["kind"], "budget_exhausted");
}

#[test]
fn the_charge_refuses_once_spent_even_when_only_the_capture_says_edited() {
    // The second witness: a run the pre-check called base, but the capture's provenance did not.
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    assert!(svc.charge(false).is_ok());
    let refused = svc.charge(true).unwrap_err();
    assert_eq!(
        refused.kind,
        super::super::protocol::ErrorKind::BudgetExhausted
    );
    assert!(svc.turn.edited);
    assert!(!svc.turn.counted);
}
