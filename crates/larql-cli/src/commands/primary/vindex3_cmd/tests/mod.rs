//! CLI-level gates for the vindex3 verbs.

mod calibration_digest;
mod decode;
mod exec_resume;
mod generate;
mod realizations;
mod sizes;

use super::*;
use std::io::Write;

/// `judged` controls whether the config carries a key no registry has
/// seen — the difference between an admissible and a blocked plan.
fn fixture_dir(judged: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut text_config = serde_json::json!({
        "model_type": "muse_glimmer_text",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "qk_scale_factor": 3.87,
        // The operand estate below is four-norm, so the post-norms must
        // apply *some* epsilon. Declaring it keeps the fixture a faithful
        // model of the thing it stands in for; without it the stack is
        // under-specified and the plan is refused, which is the point.
        "post_norm_eps": 1e-8
    });
    if !judged {
        text_config["some_future_field_nobody_reviewed"] = serde_json::json!(1.5);
    }
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "model_type": "muse_glimmer",
            "text_config": text_config
        })
        .to_string(),
    )
    .unwrap();
    // A closable four-norm + QK-norm operand estate (13 tensors per
    // layer) plus a final norm, so encode's execution-surface gate and
    // the ops verb's operand closure both hold — and the detailed layer
    // print exercises every placement site.
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    let mut push =
        |header: &mut serde_json::Map<String, serde_json::Value>, name: String, shape: &[usize]| {
            let len = 2 * shape.iter().product::<usize>() as u64;
            header.insert(
                name,
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + len],
                }),
            );
            offset += len;
        };
    push(&mut header, "model.norm.weight".to_string(), &[64]);
    push(
        &mut header,
        "model.embed_tokens.weight".to_string(),
        &[128, 64],
    );
    push(&mut header, "lm_head.weight".to_string(), &[128, 64]);
    for layer in 0..2 {
        for (suffix, shape) in [
            ("self_attn.q_proj.weight", vec![64usize, 64]),
            ("self_attn.k_proj.weight", vec![16, 64]),
            ("self_attn.v_proj.weight", vec![16, 64]),
            ("self_attn.o_proj.weight", vec![64, 64]),
            // Judged on the registered family: the gate operand must ship.
            ("self_attn.gate_proj.weight", vec![64, 64]),
            ("self_attn.q_norm.weight", vec![8]),
            ("self_attn.k_norm.weight", vec![8]),
            ("input_layernorm.weight", vec![64]),
            ("post_attention_layernorm.weight", vec![64]),
            ("pre_feedforward_layernorm.weight", vec![64]),
            ("post_feedforward_layernorm.weight", vec![64]),
            ("mlp.gate_proj.weight", vec![256, 64]),
            ("mlp.up_proj.weight", vec![256, 64]),
            ("mlp.down_proj.weight", vec![64, 256]),
        ] {
            push(
                &mut header,
                format!("model.layers.{layer}.{suffix}"),
                &shape,
            );
        }
    }
    let payload_len = offset;
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    // Real payload bytes so encode/verify can stream them.
    f.write_all(&vec![7u8; payload_len as usize]).unwrap();
    dir
}

#[test]
fn inadmissible_plan_is_an_error_and_still_writes_the_report() {
    let dir = fixture_dir(false);
    let out = dir.path().join("plan.json");
    let result = run(Vindex3Command::Plan(PlanArgs {
        artifacts: vec![dir.path().to_path_buf()],
        output: Some(out.clone()),
    }));
    assert!(result.is_err(), "an unjudged key must block");
    let plan: larql_vindex::format::vindex3::plan::SystemPlan =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert!(!plan.admissible);
    assert!(plan.summary.blocking > 0);
}

#[test]
fn admissible_plan_exits_clean() {
    let dir = fixture_dir(true);
    let out = dir.path().join("plan.json");
    let result = run(Vindex3Command::Plan(PlanArgs {
        artifacts: vec![dir.path().to_path_buf()],
        output: Some(out.clone()),
    }));
    assert!(
        result.is_ok(),
        "fully judged artifact must pass: {result:?}"
    );
    let plan: larql_vindex::format::vindex3::plan::SystemPlan =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert!(plan.admissible);
    assert_eq!(plan.summary.blocking, 0);
}

#[test]
fn accepts_a_saved_inventory_json() {
    let dir = fixture_dir(false);
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let saved = dir.path().join("inventory.json");
    std::fs::write(&saved, serde_json::to_string(&inventory).unwrap()).unwrap();
    let out = dir.path().join("plan.json");
    let result = run(Vindex3Command::Plan(PlanArgs {
        artifacts: vec![saved],
        output: Some(out.clone()),
    }));
    assert!(result.is_err()); // same artifact, same verdict
    assert!(out.exists());
}

#[test]
fn verify_round_trips_an_encoded_fixture() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    run(Vindex3Command::Verify(VerifyArgs {
        artifacts: vec![dir.path().to_path_buf()],
        container: out,
        json: true,
    }))
    .expect("an untouched encode must verify");
}

#[test]
fn verify_exits_nonzero_on_a_corrupted_container() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    let segment = std::fs::read_dir(out.join("segments"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&segment).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&segment, bytes).unwrap();
    let err = run(Vindex3Command::Verify(VerifyArgs {
        artifacts: vec![dir.path().to_path_buf()],
        container: out,
        json: false,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("NOT verified"), "{err}");
}

#[test]
fn execution_complete_gate_passes_then_refuses_a_stripped_graph() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    let inspect_args = |p: &std::path::Path| InspectArgs {
        container: p.to_path_buf(),
        verify: false,
        execution_complete: true,
        json: false,
    };
    run(Vindex3Command::Inspect(inspect_args(&out))).expect("a fresh encode must be executable");

    // Strip every surface from the persisted graph — a pre-G5a
    // container — and the same gate must refuse it.
    let graph_path = out.join("system_graph.json");
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    for component in graph["components"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("execution");
    }
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let err = run(Vindex3Command::Inspect(inspect_args(&out))).unwrap_err();
    assert!(err.to_string().contains("not executable"), "{err}");
}

#[test]
fn ops_emits_a_closed_plan_for_an_encoded_fixture() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    run(Vindex3Command::Ops(OpsArgs {
        container: out.clone(),
        component: "target".to_string(),
        layer: Some(0),
        json: false,
        realizations: false,
        budget_gib: None,
    }))
    .expect("a closable estate must plan");
    run(Vindex3Command::Ops(OpsArgs {
        container: out.clone(),
        component: "target".to_string(),
        layer: None,
        json: true,
        realizations: false,
        budget_gib: None,
    }))
    .expect("json summary must also close");
    run(Vindex3Command::Ops(OpsArgs {
        container: out,
        component: "target".to_string(),
        layer: None,
        json: false,
        realizations: false,
        budget_gib: None,
    }))
    .expect("per-layer summary must also close");
}

#[test]
fn plan_without_an_output_path_prints_to_stdout() {
    let dir = fixture_dir(true);
    run(Vindex3Command::Plan(PlanArgs {
        artifacts: vec![dir.path().to_path_buf()],
        output: None,
    }))
    .expect("admissible plan on stdout");
}

#[test]
fn verify_prints_semantic_failures_on_a_tampered_graph() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    let graph_path = out.join("system_graph.json");
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    graph["components"][0]["hidden_size"] = serde_json::json!(9999);
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let err = run(Vindex3Command::Verify(VerifyArgs {
        artifacts: vec![dir.path().to_path_buf()],
        container: out,
        json: false,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("NOT verified"), "{err}");
}

#[test]
fn ops_refusals_and_bad_layer_exit_nonzero() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    let err = run(Vindex3Command::Ops(OpsArgs {
        container: out.clone(),
        component: "target".to_string(),
        layer: Some(99),
        json: false,
        realizations: false,
        budget_gib: None,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("no layer 99"), "{err}");

    // Strip the surfaces: closure must refuse and itemise.
    let graph_path = out.join("system_graph.json");
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    for component in graph["components"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("execution");
    }
    std::fs::write(&graph_path, graph.to_string()).unwrap();
    let err = run(Vindex3Command::Ops(OpsArgs {
        container: out,
        component: "target".to_string(),
        layer: None,
        json: false,
        realizations: false,
        budget_gib: None,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("operand closure failed"), "{err}");
}

#[test]
fn inspect_exits_nonzero_on_an_incoherent_container() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    let segment = std::fs::read_dir(out.join("segments"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&segment).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&segment, bytes).unwrap();
    // Structure is intact without --verify; the re-hash catches it.
    let err = run(Vindex3Command::Inspect(InspectArgs {
        container: out,
        verify: true,
        execution_complete: false,
        json: true,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("incoherent"), "{err}");
}

#[test]
fn missing_artifact_errors_before_planning() {
    let result = run(Vindex3Command::Plan(PlanArgs {
        artifacts: vec![PathBuf::from("/nonexistent/model")],
        output: None,
    }));
    assert!(result.is_err());
}
