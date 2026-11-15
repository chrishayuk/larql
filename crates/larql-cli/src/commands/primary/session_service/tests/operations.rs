//! Every operation through the service equals the typed operation run in-process: arrays bit for bit (the
//! `.npy` bytes), provenance field for field, structured results value for value.

use serde_json::{json, Value};

use larql_lql::operations::{
    npy_file_bytes, CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite, CaptureSpec,
    NpyDescr,
};

use super::*;

fn capture_bytes(captured: &larql_lql::operations::Captured) -> Vec<u8> {
    let body: Vec<u8> = captured
        .values()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    npy_file_bytes(NpyDescr::F4, &captured.shape, &body).unwrap()
}

#[test]
fn tensors_equal_the_listing() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let s = fx.session();
    for filter in [None, Some("decoder_stack")] {
        let listed = ok(&mut svc, "tensors", json!({"filter": filter}));
        let expected = s.tensors(filter).unwrap();
        let listed = listed.as_array().unwrap();
        assert_eq!(listed.len(), expected.len());
        for (got, want) in listed.iter().zip(&expected) {
            assert_eq!(got["name"], want.qualified_name());
            assert_eq!(got["representation"], want.representation);
            assert_eq!(got["object"], want.object);
            assert_eq!(got["tensor"], want.tensor);
            assert_eq!(got["canonical"], want.canonical);
            assert_eq!(got["dtype"], want.dtype);
            assert_eq!(got["shape"], json!(want.shape));
            assert_eq!(got["bare"], want.bare_name.label());
        }
    }
}

#[test]
fn tensor_reads_equal_read_weight_bit_for_bit() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let s = fx.session();
    let cases: [(Option<Vec<usize>>, Option<Vec<usize>>, bool); 3] = [
        (None, None, false),
        (Some(vec![1, 3]), Some(vec![0, 5, 7]), false),
        (Some(vec![4]), None, true),
    ];
    for (rows, cols, bits) in cases {
        let got = ok(
            &mut svc,
            "tensor",
            json!({"name": TENSOR, "rows": rows, "cols": cols, "bits": bits}),
        );
        let want = s
            .read_weight(TENSOR, rows.as_deref(), cols.as_deref(), bits)
            .unwrap();
        assert_eq!(
            npy_bytes(&got["array"]),
            npy_file_bytes(want.descr, &want.shape, &want.bytes).unwrap()
        );
        assert_eq!(got["array"]["dtype"], want.descr.label());
        assert_eq!(got["array"]["shape"], json!(want.shape));
        assert_eq!(got["provenance"], redacted(want.provenance.clone()));
        assert_eq!(got["array"]["sha256"], want.provenance["npy_sha256"]);
    }
    let (kind, reason) = err(&mut svc, "tensor", json!({"name": "no.such.tensor"}));
    assert_eq!(kind, "failed");
    assert!(reason.contains("EXPORT WEIGHT refused"), "{reason}");
}

#[test]
fn tokens_equal_the_encoders() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let s = fx.session();
    assert_eq!(
        ok(&mut svc, "tokenize", json!({"text": PROMPT})),
        json!(s.tokenize(PROMPT, true).unwrap())
    );
    let raw = s.tokenize(CONTINUATION, false).unwrap();
    assert_eq!(
        ok(
            &mut svc,
            "tokenize",
            json!({"text": CONTINUATION, "raw": true})
        ),
        json!(raw)
    );
    assert_eq!(
        ok(&mut svc, "decode", json!({"ids": raw})),
        json!(s.decode(&raw).unwrap())
    );
}

#[test]
fn captures_equal_the_typed_capture_bit_for_bit() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let s = fx.session();
    let cases: Vec<(
        Value,
        CaptureSite,
        Option<CaptureLayers>,
        Option<CaptureContinuation>,
        CapturePositions,
    )> = vec![
        (
            json!({"site": "logits"}),
            CaptureSite::Logits,
            None,
            None,
            CapturePositions::Last,
        ),
        (
            json!({"site": "final", "positions": "all"}),
            CaptureSite::Final,
            None,
            None,
            CapturePositions::All,
        ),
        (
            json!({"site": "residual", "layer": 0, "positions": "PROMPT_END"}),
            CaptureSite::Residual,
            Some(CaptureLayers::One(0)),
            None,
            CapturePositions::PromptEnd,
        ),
        (
            json!({"site": "post_attention", "layers": [1, 0], "continue_text": CONTINUATION, "positions": "REPLY"}),
            CaptureSite::PostAttention,
            Some(CaptureLayers::Many(vec![1, 0])),
            Some(CaptureContinuation::Text(CONTINUATION.into())),
            CapturePositions::Reply,
        ),
        (
            json!({"site": "ffn.input", "layer": 1, "continue_tokens": [3, 7], "positions": [0, 6]}),
            CaptureSite::FfnInput,
            Some(CaptureLayers::One(1)),
            Some(CaptureContinuation::Tokens(vec![3, 7])),
            CapturePositions::List(vec![0, 6]),
        ),
        (
            json!({"site": "ffn.down_output", "layers": [0], "generate": 2, "positions": "LAST"}),
            CaptureSite::FfnDownOutput,
            Some(CaptureLayers::Many(vec![0])),
            Some(CaptureContinuation::Generate(2)),
            CapturePositions::Last,
        ),
        (
            json!({"site": "ffn.down_input", "layer": 1}),
            CaptureSite::FfnDownInput,
            Some(CaptureLayers::One(1)),
            None,
            CapturePositions::Last,
        ),
    ];
    for (mut args, site, layers, continuation, positions) in cases {
        args["prompt"] = json!(PROMPT);
        let got = ok(&mut svc, "capture", args.clone());
        let want = s
            .capture(&CaptureSpec {
                site,
                layers: layers.as_ref(),
                prompt: PROMPT,
                continuation: continuation.as_ref(),
                positions: &positions,
            })
            .unwrap();
        assert_eq!(npy_bytes(&got["array"]), capture_bytes(&want), "{args}");
        assert_eq!(
            got["provenance"],
            redacted(want.provenance.clone()),
            "{args}"
        );
        assert_eq!(got["array"]["sha256"], want.provenance["npy_sha256"]);
    }
}

#[test]
fn bad_capture_arguments_are_refused_before_running() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let bad = [
        (json!({"site": "nowhere"}), "unknown site"),
        (
            json!({"site": "residual", "layer": 0, "layers": [0]}),
            "not both",
        ),
        (
            json!({"site": "logits", "continue_text": "x", "generate": 1}),
            "at most one",
        ),
        (json!({"site": "logits", "positions": "FIRST"}), "positions"),
    ];
    for (mut args, needle) in bad {
        args["prompt"] = json!(PROMPT);
        let (kind, reason) = err(&mut svc, "capture", args);
        assert_eq!(kind, "invalid_request");
        assert!(reason.contains(needle), "{reason}");
    }
    let (kind, reason) = err(
        &mut svc,
        "capture",
        json!({"site": "residual", "prompt": PROMPT}),
    );
    assert_eq!(kind, "invalid_request");
    assert!(reason.contains("inside a layer"), "{reason}");
}

#[test]
fn infer_and_generate_equal_predict_and_generate() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let s = fx.session();
    let got = ok(&mut svc, "infer", json!({"prompt": PROMPT, "top": 4}));
    let want = s.predict(PROMPT, 4).unwrap();
    assert_eq!(got["prompt_ids"], json!(want.prompt_ids));
    assert_eq!(got["knn_override"], Value::Null);
    let preds = got["predictions"].as_array().unwrap();
    assert_eq!(preds.len(), 4);
    for (p, w) in preds.iter().zip(&want.top) {
        assert_eq!(p["token_id"], w.token_id);
        assert_eq!(p["token"], w.token);
        assert_eq!(p["probability"].as_f64().unwrap(), f64::from(w.probability));
    }
    assert_eq!(
        ok(&mut svc, "infer", json!({"prompt": PROMPT}))["predictions"]
            .as_array()
            .unwrap()
            .len(),
        5
    );

    let got = ok(&mut svc, "generate", json!({"prompt": PROMPT, "n": 3}));
    let want = s.generate(PROMPT, 3).unwrap();
    assert_eq!(got["prompt_ids"], json!(want.prompt_ids));
    assert_eq!(got["token_ids"], json!(want.token_ids));
    assert_eq!(got["text"], want.text);
}

#[test]
fn a_low_rank_op_from_arrays_saves_the_vlp_the_in_process_session_saves() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let mut s = fx.session();
    let rows = tensor_rows(&s);
    let a = a_factor(rows, 0.125);
    let b = vec![1.0; COLUMNS.len()];

    assert_eq!(
        ok(&mut svc, "begin_patch", json!({"path": "sub/edit.vlp"})),
        Value::Null
    );
    let staged = ok(
        &mut svc,
        "add_low_rank",
        json!({
            "tensor": TENSOR, "a": operand(&a, [rows, 1]), "b": operand(&b, [1, COLUMNS.len()]),
            "columns": COLUMNS, "numerics": F64_PROFILE,
        }),
    );
    let saved = ok(&mut svc, "save_patch", json!({}));
    assert_eq!(saved["path"], "sub/edit.vlp");
    assert_eq!(saved["weight_edits"], 1);

    let in_process = fx.logs.path().join("in-process.vlp");
    s.begin_patch(in_process.to_str().unwrap()).unwrap();
    let a_bits = larql_lql::operations::operand_bits_from_values(
        F64_PROFILE,
        larql_lql::operations::OperandValues::F64(&a),
        "a",
    )
    .unwrap();
    let b_bits = larql_lql::operations::operand_bits_from_values(
        F64_PROFILE,
        larql_lql::operations::OperandValues::F64(&b),
        "b",
    )
    .unwrap();
    let origin = larql_lql::operations::operand_origin(
        "python arrays",
        &a_bits,
        [rows, 1],
        &b_bits,
        [1, COLUMNS.len()],
    );
    let want = s
        .stage_low_rank(
            TENSOR,
            &COLUMNS,
            F64_PROFILE,
            1,
            &a_bits,
            &b_bits,
            Some(origin),
        )
        .unwrap();
    s.save_patch().unwrap();
    assert_eq!(staged["digest"], want.op.digest);
    assert_eq!(staged["changed_elements"], want.changed_elements);
    assert_eq!(staged["sham"], want.sham);
    assert_eq!(
        staged["pending_structured_ops"],
        want.pending_structured_ops
    );
    assert_eq!(staged["origin"], json!(want.op.origin));
    assert_eq!(
        std::fs::read(fx.workdir().join("sub/edit.vlp")).unwrap(),
        std::fs::read(&in_process).unwrap()
    );
}

#[test]
fn misshapen_operands_are_refused() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let rows = tensor_rows(&fx.session());
    let (kind, reason) = err(
        &mut svc,
        "add_low_rank",
        json!({
            "tensor": TENSOR, "a": operand(&a_factor(rows, 0.5), [rows, 1]),
            "b": operand(&[1.0; 6], [2, 3]), "columns": COLUMNS, "numerics": F64_PROFILE,
        }),
    );
    assert_eq!(kind, "invalid_request");
    assert!(reason.contains("must be 1x3"), "{reason}");
    let (kind, reason) = err(
        &mut svc,
        "add_low_rank",
        json!({
            "tensor": TENSOR, "a": operand(&a_factor(rows, 0.5), [rows, 1]),
            "b": operand(&[1.0; 3], [1, 3]), "columns": COLUMNS, "numerics": "f32-seq-rne1/v1",
        }),
    );
    assert_eq!(kind, "failed");
    assert!(reason.contains("narrowing"), "{reason}");
    let f32_b = {
        let body: Vec<u8> = [1.0f32; 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let file = npy_v1(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3), }",
            &body,
        );
        json!({"npy_base64": base64::engine::general_purpose::STANDARD.encode(file)})
    };
    let (kind, reason) = err(
        &mut svc,
        "add_low_rank",
        json!({
            "tensor": TENSOR, "a": operand(&a_factor(rows, 0.5), [rows, 1]),
            "b": f32_b, "columns": COLUMNS, "numerics": "nonsense",
        }),
    );
    assert_eq!(kind, "failed", "{reason}");
}

#[test]
fn update_weights_equals_stage_weight_delta() {
    let fx = Fixture::new();
    let mut svc = fx.start(0);
    let mut s = fx.session();
    let got = ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0, 1, 2], "delta": 2.0}),
    );
    let want = s.stage_weight_delta(TENSOR, ROW, &[0, 1, 2], 2.0).unwrap();
    assert_eq!(got["digest"], want.digest);
    assert_eq!(got["pending_elements"], want.pending_elements);
    let elements = got["elements"].as_array().unwrap();
    for (g, w) in elements.iter().zip(&want.elements) {
        assert_eq!(g["col"], w.col);
        assert_eq!(g["old_bits"], w.old_bits);
        assert_eq!(g["new_bits"], w.new_bits);
    }
}

#[test]
fn the_patch_lifecycle_through_the_service_equals_in_process() {
    let fx = Fixture::new();
    let vlp = fx.patch("edit.vlp");
    let mut svc = fx.start(5);
    let mut s = fx.session();

    let applied = ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let want = s.apply_patch(vlp.to_str().unwrap()).unwrap();
    assert_eq!(applied["path"], "edit.vlp");
    assert_eq!(applied["sha256"], want.sha256);
    for key in [
        "operations",
        "inserts",
        "updates",
        "deletes",
        "weight_edits",
        "structured_ops",
    ] {
        let w = match key {
            "operations" => want.operations,
            "inserts" => want.inserts,
            "updates" => want.updates,
            "deletes" => want.deletes,
            "weight_edits" => want.weight_edits,
            _ => want.structured_ops,
        };
        assert_eq!(applied[key], w, "{key}");
    }

    let spec = CaptureSpec {
        site: CaptureSite::Logits,
        layers: None,
        prompt: PROMPT,
        continuation: None,
        positions: &CapturePositions::All,
    };
    let edited = ok(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": "ALL"}),
    );
    let want_edited = s.capture(&spec).unwrap();
    assert_eq!(npy_bytes(&edited["array"]), capture_bytes(&want_edited));
    assert_eq!(edited["provenance"]["executes"], "patched");
    assert_eq!(edited["provenance"]["patches"][0]["sha256"], want.sha256);
    assert_eq!(edited["provenance"]["patches"][0]["path"], "edit.vlp");

    let listed = ok(&mut svc, "patches", json!({}));
    let want_listed = s.patches().unwrap();
    assert_eq!(listed["applied"][0]["path"], "edit.vlp");
    assert_eq!(listed["applied"][0]["name"], "edit.vlp");
    assert_eq!(
        listed["applied"][0]["sha256"],
        json!(want_listed.applied[0].sha256)
    );
    assert_eq!(listed["weight_ops"], want_listed.weight_ops);
    assert_eq!(listed["weight_elements"], want_listed.weight_elements);
    assert_eq!(listed["session_overrides"], want_listed.session_overrides);
    assert_eq!(listed["recording"], Value::Null);

    let removed = ok(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    assert_eq!(removed, json!({"number": 1, "carried_weight_ops": true}));
    s.remove_patch(vlp.to_str().unwrap()).unwrap();
    let base = ok(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT, "positions": "ALL"}),
    );
    assert_eq!(
        npy_bytes(&base["array"]),
        capture_bytes(&s.capture(&spec).unwrap())
    );
    assert_eq!(base["provenance"]["executes"], "stored");
    assert_eq!(ok(&mut svc, "patches", json!({}))["applied"], json!([]));

    let (kind, reason) = err(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    assert_eq!(kind, "failed");
    assert!(reason.contains("patch not found: edit.vlp"), "{reason}");
    assert!(!reason.contains(&svc.staging_prefix), "{reason}");
}

#[test]
fn a_recording_is_listed_and_saving_without_one_is_refused() {
    let fx = Fixture::new();
    let mut svc = fx.start(5);
    let (kind, reason) = err(&mut svc, "save_patch", json!({}));
    assert_eq!(kind, "failed");
    assert!(reason.contains("no active patch session"), "{reason}");

    ok(&mut svc, "begin_patch", json!({"path": "named.vlp"}));
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": 2.0}),
    );
    assert_eq!(
        ok(&mut svc, "patches", json!({}))["recording"],
        json!({"path": "named.vlp", "operations": 1})
    );
    let (_, reason) = err(&mut svc, "begin_patch", json!({"path": "other.vlp"}));
    assert!(reason.contains("already active"), "{reason}");
    assert_eq!(ok(&mut svc, "save_patch", json!({}))["path"], "named.vlp");
    assert!(fx.workdir().join("named.vlp").is_file());
}

#[test]
fn discard_weight_edits_equals_the_typed_discard() {
    let fx = Fixture::new();
    let mut svc = fx.start(5);
    let mut s = fx.session();
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": 2.0}),
    );
    s.stage_weight_delta(TENSOR, ROW, &[0], 2.0).unwrap();
    let got = ok(&mut svc, "discard_weight_edits", json!({}));
    let want = s.discard_weight_edits().unwrap();
    assert_eq!(
        got,
        json!({"discarded": want.discarded, "unrecorded": want.unrecorded, "from_patches": want.from_patches})
    );
    assert_eq!(got["discarded"], 1);
}

#[test]
fn a_saved_patch_overwrites_an_existing_workdir_file_and_applies() {
    let fx = Fixture::new();
    std::fs::write(
        fx.workdir().join("again.vlp"),
        "old contents that are longer than nothing",
    )
    .unwrap();
    let mut svc = fx.start(5);
    ok(
        &mut svc,
        "begin_patch",
        json!({"path": fx.workdir().join("again.vlp")}),
    );
    ok(
        &mut svc,
        "update_weights",
        json!({"tensor": TENSOR, "row": ROW, "cols": [0], "delta": DELTA}),
    );
    ok(&mut svc, "save_patch", json!({}));
    ok(&mut svc, "discard_weight_edits", json!({}));
    let applied = ok(&mut svc, "apply_patch", json!({"path": "again.vlp"}));
    assert_eq!(applied["weight_edits"], 1);
}
