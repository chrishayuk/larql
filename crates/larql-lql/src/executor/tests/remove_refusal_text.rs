//! Gate B, B2: how REMOVE PATCH's refusal names each blocking journal op, and
//! how it caps the list. A fact id appears only for an op that carries a fact
//! key; a physical or weight op is named by its kind and address.

use crate::executor::remove_refusal::{describe_op, refusal_text, LISTED_EDITS_MAX};
use crate::executor::PatchRecording;
use larql_vindex::{FactKey, PatchOp};

fn op(json: serde_json::Value) -> PatchOp {
    serde_json::from_value(json).expect("a patch op")
}

fn source() -> serde_json::Value {
    serde_json::json!({"model": "m", "index_sha256": "i", "representation": "r", "payload_sha256": "p"})
}

#[test]
fn each_op_is_named_by_its_key_or_its_address() {
    let id = FactKey::new("a", "b").fact_id();
    let cases = [
        (
            serde_json::json!({"op": "insert", "layer": 1, "feature": 2, "relation": "b", "entity": "a", "target": "t"}),
            format!("insert, fact {id}, at layer 1 feature 2"),
        ),
        (
            serde_json::json!({"op": "insert", "layer": 1, "feature": 2, "entity": "a", "target": "t"}),
            "insert at layer 1 feature 2".to_string(),
        ),
        (
            serde_json::json!({"op": "update", "layer": 3, "feature": 4}),
            "update at layer 3 feature 4".to_string(),
        ),
        (
            serde_json::json!({"op": "delete", "layer": 5, "feature": 6}),
            "delete at layer 5 feature 6".to_string(),
        ),
        (
            serde_json::json!({"op": "insert_knn", "layer": 7, "entity": "a", "relation": "b", "target": "t", "target_id": 9, "key_vector_b64": ""}),
            format!("insert_knn, fact {id}, at layer 7"),
        ),
        (
            serde_json::json!({"op": "update_knn_fact", "entity": "a", "relation": "b", "target": "t", "target_id": 9}),
            format!("update_knn_fact, fact {id}"),
        ),
        (
            serde_json::json!({"op": "delete_knn_fact", "entity": "a", "relation": "b"}),
            format!("delete_knn_fact, fact {id}"),
        ),
        (
            serde_json::json!({"op": "delete_knn", "entity": "a"}),
            "delete_knn of entity \"a\"".to_string(),
        ),
        (
            serde_json::json!({"op": "weight_delta", "rounding": "r", "source": source(), "tensor": "w.0", "edits": [], "digest": "d"}),
            "weight_delta on tensor w.0".to_string(),
        ),
        (
            serde_json::json!({"op": "structured_weight", "source": source(), "tensor": "w.1", "numerics": "n", "operand": [], "alpha": "0", "columns": [], "digest": "d"}),
            "structured_weight on tensor w.1".to_string(),
        ),
        (
            serde_json::json!({"op": "low_rank_columns_weight", "source": source(), "tensor": "w.2", "numerics": "n", "rank": 1, "a": [], "b": [], "columns": [], "digest": "d"}),
            "low_rank_columns_weight on tensor w.2".to_string(),
        ),
    ];
    for (json, want) in cases {
        assert_eq!(describe_op(&op(json.clone())), want, "{json}");
    }
}

#[test]
fn the_list_is_capped_in_journal_order() {
    let ops: Vec<PatchOp> = (0..LISTED_EDITS_MAX + 3)
        .map(|i| op(serde_json::json!({"op": "delete", "layer": 0, "feature": i})))
        .collect();
    let refs: Vec<&PatchOp> = ops.iter().collect();
    let text = refusal_text("p.vlp", &refs, None);
    let listed: Vec<&str> = text.lines().filter(|l| l.starts_with("  ")).collect();
    assert_eq!(listed.len(), LISTED_EDITS_MAX + 1, "{text}");
    for (i, line) in listed.iter().take(LISTED_EDITS_MAX).enumerate() {
        assert_eq!(*line, format!("  delete at layer 0 feature {i}"));
    }
    assert_eq!(listed[LISTED_EDITS_MAX], "  +3 more");
    assert!(text.contains("holds 11 knowledge edit(s)"), "{text}");
    assert!(text.contains("keep these edits: bind the vindex"), "{text}");
}

#[test]
fn an_empty_open_recording_needs_no_save() {
    let ops = [op(
        serde_json::json!({"op": "delete", "layer": 0, "feature": 0}),
    )];
    let empty = PatchRecording {
        path: String::new(),
        operations: Vec::new(),
    };
    let text = refusal_text("p.vlp", &[&ops[0]], Some(&empty));
    assert!(text.contains("keep these edits: bind the vindex"), "{text}");
}
