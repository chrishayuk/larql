//! Every refusal the export makes before writing a byte, and the one
//! guarantee about what it leaves behind.

use larql_vindex::format::vindex3::gguf::dense::{export_dense, DenseExportOptions};

use super::common::Fixture;

fn refusal(fx: &Fixture, options: &DenseExportOptions) -> String {
    let out = fx.out("refused.gguf");
    let err = export_dense(&fx.container, &out, options).expect_err("the export must refuse");
    assert!(!out.exists(), "a refused export leaves no file");
    err.to_string()
}

/// Give the container a second, perception component, as Gemma 3 4B's
/// vision tower is one.
fn add_vision_component(fx: &Fixture) {
    fx.edit_json("system_graph.json", |g| {
        let comps = g["components"].as_array_mut().unwrap();
        let mut vision = comps[0].clone();
        vision["id"] = "vision".into();
        vision["role"] = "perception".into();
        comps.push(vision);
    });
}

#[test]
fn another_component_needs_text_only() {
    let fx = Fixture::new();
    add_vision_component(&fx);
    let msg = refusal(&fx, &DenseExportOptions::default());
    assert!(
        msg.contains("\"vision\"") && msg.contains("--text-only"),
        "{msg}"
    );

    let out = fx.out("text.gguf");
    let report = export_dense(
        &fx.container,
        &out,
        &DenseExportOptions {
            text_only: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.excluded_components, vec!["vision".to_string()]);
}

#[test]
fn an_unsupported_family_refuses_by_its_declared_name() {
    let fx = Fixture::new();
    fx.edit_json("index.json", |i| i["family"] = "llama".into());
    let msg = refusal(&fx, &DenseExportOptions::default());
    assert!(
        msg.contains("`llama`") && msg.contains("gemma3_text"),
        "{msg}"
    );
}

#[test]
fn a_non_bf16_canonical_refuses() {
    let fx = Fixture::new();
    fx.edit_json("index.json", |i| {
        let reps = i["representations"].as_object_mut().unwrap();
        let mut entry = reps.remove("target.final_norm@BF16").unwrap();
        entry["encoding"] = "Q8_0".into();
        reps.insert("target.final_norm@Q8_0".into(), entry);
    });
    fx.edit_json("system_graph.json", |g| {
        for o in g["objects"].as_array_mut().unwrap() {
            if o["id"] == "target.final_norm" {
                o["representations"][0]["encoding"] = "Q8_0".into();
            }
        }
    });
    let msg = refusal(&fx, &DenseExportOptions::default());
    assert!(
        msg.contains("Q8_0") && msg.contains("BF16 and F32"),
        "{msg}"
    );
}

#[test]
fn an_object_without_one_canonical_refuses() {
    let fx = Fixture::new();
    fx.edit_json("system_graph.json", |g| {
        for o in g["objects"].as_array_mut().unwrap() {
            if o["id"] == "target.final_norm" {
                o["representations"][0]["fidelity"] = "approximate".into();
            }
        }
    });
    let msg = refusal(&fx, &DenseExportOptions::default());
    assert!(msg.contains("0 canonical"), "{msg}");
}

#[test]
fn an_existing_output_is_never_overwritten() {
    let fx = Fixture::new();
    let out = fx.out("exists.gguf");
    std::fs::write(&out, b"keep me").unwrap();
    let err = export_dense(&fx.container, &out, &DenseExportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    assert_eq!(std::fs::read(&out).unwrap(), b"keep me");
}

#[test]
fn a_container_without_a_tokenizer_refuses() {
    let fx = Fixture::new();
    std::fs::remove_file(fx.container.join("tokenizer.json")).unwrap();
    let msg = refusal(&fx, &DenseExportOptions::default());
    assert!(msg.contains("no tokenizer.json"), "{msg}");
}

#[test]
fn a_graph_head_that_contradicts_the_plan_refuses() {
    let fx = Fixture::new();
    fx.edit_json("system_graph.json", |g| {
        g["components"][0]["execution"]["head"]["head_reuses_embedding"] = false.into();
    });
    let err = export_dense(
        &fx.container,
        &fx.out("x.gguf"),
        &DenseExportOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("OutputHead"), "{err}");
}
