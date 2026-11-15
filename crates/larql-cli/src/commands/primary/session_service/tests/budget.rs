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
fn the_charge_refuses_once_spent_and_is_made_only_once_per_request() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    svc.charge().unwrap();
    svc.charge().unwrap();
    assert_eq!(svc.budget_used, 1);
    svc.turn = Default::default();
    let refused = svc.charge().unwrap_err();
    assert_eq!(
        refused.kind,
        super::super::protocol::ErrorKind::BudgetExhausted
    );
    assert!(svc.turn.edited);
    assert!(!svc.turn.counted);
}

/// An edited request leaves three kinds of record: `admitted` (charged, synced before it runs), then its
/// request record with the outcome; a refused one leaves only its request record, uncharged.
#[test]
fn an_admitted_execution_is_logged_before_its_outcome() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(1);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let r = call(&mut svc, "infer", json!({"prompt": PROMPT}));
    let refused = call(&mut svc, "infer", json!({"prompt": PROMPT}));
    let records = fx.records();
    let admitted: Vec<&Value> = records
        .iter()
        .filter(|r| r["event"] == "admitted")
        .collect();
    assert_eq!(admitted.len(), 1);
    let outcome = records.iter().find(|x| x["seq"] == r["seq"]).unwrap();
    assert_eq!(outcome["admitted_seq"], admitted[0]["seq"]);
    assert!(admitted[0]["seq"].as_u64() < outcome["seq"].as_u64());
    assert_eq!(admitted[0]["budget_used"], 1);
    assert_eq!(admitted[0]["op"], "infer");
    assert_eq!(outcome["outcome"], "completed");
    let rejected = records.iter().find(|x| x["seq"] == refused["seq"]).unwrap();
    assert_eq!(rejected["outcome"], "rejected");
    assert_eq!(rejected["admitted_seq"], Value::Null);
    assert_eq!(rejected["counted"], false);
}

/// Every way the review found to run the edited model without paying: each is now charged (it ran), or
/// refused before anything runs (and not charged).
#[test]
fn every_failure_after_admission_is_charged_and_every_early_refusal_is_not() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(20);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let prompt_tokens = fx.session().tokenize(PROMPT, true).unwrap().len();

    // Refused before execution, nothing computed, nothing charged.
    for (args, needle) in [
        (
            json!({"site": "logits", "prompt": PROMPT, "positions": "REPLY"}),
            "needs a continuation",
        ),
        (
            json!({"site": "residual", "prompt": PROMPT, "layers": []}),
            "layers= is empty",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "positions": []}),
            "empty list",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "positions": [0, 0]}),
            "listed twice",
        ),
        (
            json!({"site": "residual", "prompt": PROMPT, "layers": [1, 1]}),
            "listed twice",
        ),
        (
            json!({"site": "residual", "prompt": PROMPT, "layer": 9}),
            "does not exist",
        ),
        (
            json!({"site": "residual", "prompt": PROMPT}),
            "inside a layer",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "layer": 0}),
            "takes no layer",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "positions": [prompt_tokens]}),
            "outside the",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "generate": 2, "positions": [prompt_tokens + 2]}),
            "may produce",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "generate": 0}),
            "at least 1",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "continue_tokens": []}),
            "is empty",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "continue_tokens": [1_000_000]}),
            "outside the vocabulary",
        ),
        (
            json!({"site": "logits", "prompt": PROMPT, "continue_text": " "}),
            "no tokens",
        ),
    ] {
        let r = call(&mut svc, "capture", args.clone());
        assert_eq!(r["ok"], false, "{args}");
        assert!(
            r["error"]["reason"].as_str().unwrap().contains(needle),
            "{args}: {r}"
        );
        assert_eq!(r["counted"], false, "{args}");
        assert_eq!(remaining(&r), 20, "{args}");
    }
    assert!(fx.records().iter().all(|r| r["event"] != "admitted"));

    // Admitted, then refused by LARQL after it ran (or while it ran): charged.
    let before = svc.budget_used;
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "generate": 2, "positions": [prompt_tokens + 1]}),
    );
    assert_eq!(r["counted"], true, "{r}");
    assert_eq!(svc.budget_used, before + 1);
}

#[test]
fn an_oversized_capture_is_refused_before_it_runs() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut policy = fx.policy(5);
    policy["limits"] = json!({"max_response_bytes": 2048});
    let mut svc = fx.start_with(&policy).unwrap();
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let r = logits_all(&mut svc);
    assert_eq!(r["error"]["kind"], "refused");
    assert!(
        r["error"]["reason"].as_str().unwrap().contains("at most"),
        "{r}"
    );
    assert_eq!((r["counted"].clone(), remaining(&r)), (json!(false), 5));
}

/// The one predicate: after every state-changing operation on the allow-list, the service calls the model
/// edited exactly when a capture says it did not execute the stored weights.
#[test]
fn the_edited_predicate_agrees_with_what_capture_says_executed_after_every_operation() {
    let fx = Fixture::new();
    fx.patch("applied.vlp");
    let mut svc = fx.start(1000);
    let rows = tensor_rows(&fx.session());
    let low_rank = json!({
        "tensor": TENSOR, "a": operand(&a_factor(rows, 4.0), [rows, 1]),
        "b": operand(&[1.0; 3], [1, 3]), "columns": COLUMNS, "numerics": F64_PROFILE,
    });
    let steps: Vec<(&str, Value)> = vec![
        ("begin_patch", json!({"path": "rec.vlp"})),
        ("add_low_rank", low_rank),
        ("save_patch", json!({})),
        ("discard_weight_edits", json!({})),
        ("apply_patch", json!({"path": "rec.vlp"})),
        ("remove_patch", json!({"path": "rec.vlp"})),
        ("apply_patch", json!({"path": "applied.vlp"})),
        (
            "update_weights",
            json!({"tensor": "0.mlp.down_proj.weight", "row": 1, "cols": [0], "delta": DELTA}),
        ),
        ("remove_patch", json!({"path": "applied.vlp"})),
        ("discard_weight_edits", json!({})),
    ];
    let mut covered: Vec<&str> = Vec::new();
    for (op, args) in steps {
        ok(&mut svc, op, args);
        covered.push(op);
        let predicate = svc.model_is_edited().unwrap();
        let r = call(
            &mut svc,
            "capture",
            json!({"site": "logits", "prompt": PROMPT}),
        );
        let executes = r["result"]["provenance"]["executes"].clone();
        assert_eq!(predicate, executes != "stored", "after {op}: {executes}");
        assert_eq!(r["counted"], json!(predicate), "after {op}");
    }
    // Every allow-listed operation that can change the session's state is exercised.
    for op in super::super::protocol::Op::ALL {
        let changes_state = !matches!(
            op.name(),
            "tensors"
                | "tensor"
                | "tokenize"
                | "decode"
                | "capture"
                | "infer"
                | "generate"
                | "patches"
        );
        if changes_state {
            assert!(
                covered.contains(&op.name()),
                "{} is not exercised",
                op.name()
            );
        }
    }
}
