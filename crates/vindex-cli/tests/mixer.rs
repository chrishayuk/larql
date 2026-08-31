//! The token mixer, adversarially: two hybrid vocabularies, four
//! operators, and the assertion that each one is named as itself.
//!
//! This file exists because of a specific regression. `mixer_label`
//! matched the plan's softmax op and treated *every* other layer as
//! Gated DeltaNet, so on the real Kimi-Linear-48B container — whose
//! graph records `KKKM KKKM …`, twenty KDA layers and seven MLA —
//! `vindex layers` printed twenty-seven Gated DeltaNet layers,
//! `describe layer.3.mixer` answered `GATED DELTANET` over an empty
//! operand table, and the q/k/v/o refusal asserted a false operator by
//! name. Three confident wrong answers from a container that recorded
//! the right one.
//!
//! So the assertions here are **exact sequences**, never
//! `assert_ne!(…, "GATED DELTANET")` or a disjunction over two
//! acceptable labels: a negative passes for the wrong reason the
//! moment a third operator appears, which is precisely how the
//! original defect survived a green suite.

use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, gated_q_f32_model, hybrid_lllf_f32_model,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use vindex_cli::{describe_facts, layers_facts, precision_matrix_facts};

fn encoded(write: impl FnOnce(&std::path::Path), name: &str) -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), dir.path(), name);
    dir
}

fn mixers(root: &std::path::Path) -> Vec<String> {
    let v = layers_facts(root).unwrap();
    v["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["mixer"].as_str().unwrap().to_string())
        .collect()
}

fn roles(root: &std::path::Path, address: &str) -> Vec<String> {
    let v = describe_facts(root, address, 2, None).unwrap();
    assert!(
        v["undescribed"].is_null(),
        "{address} rendered no operand table: {v}"
    );
    v["operands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["role"].as_str().unwrap().to_string())
        .collect()
}

/// Qwen3.8's vocabulary: a Gated DeltaNet recurrence against gated
/// softmax attention, at the LLLF cadence.
#[test]
fn the_gated_delta_hybrid_names_both_of_its_programmes() {
    let dir = encoded(hybrid_lllf_f32_model, "vindex-cli-lllf");
    assert_eq!(
        mixers(dir.path()),
        vec![
            "GATED DELTANET",
            "GATED DELTANET",
            "GATED DELTANET",
            "SOFTMAX ATTENTION",
        ]
    );

    assert_eq!(
        roles(dir.path(), "layer.0.mixer"),
        vec![
            "fused recurrent q|k|v",
            "decay projection",
            "write-strength projection",
            "output-gate projection",
            "causal conv over q|k|v",
            "log decay",
            "timestep bias",
            "gated norm",
            "output projection",
        ]
    );
    assert_eq!(
        roles(dir.path(), "layer.3.mixer"),
        vec!["query", "key", "value", "output"]
    );
}

/// The gate is an operand, not an operator: the same softmax mixer
/// answers `GATED ATTENTION` when the layer ships a fused output gate
/// and `SOFTMAX ATTENTION` when it does not. Qwen3.8's full-attention
/// layers are the gated kind, so the label the film shows is this one.
///
/// `SOFTMAX ATTENTION` rather than a bare `ATTENTION` for the ungated
/// case on purpose — gated attention is *also* softmax attention, and
/// a pair reading `ATTENTION` / `GATED ATTENTION` would suggest the
/// gate replaces the softmax rather than following it.
#[test]
fn the_output_gate_refines_the_softmax_label_without_replacing_it() {
    let dir = encoded(gated_q_f32_model, "vindex-cli-gated");
    let seen = mixers(dir.path());
    assert!(
        seen.iter().all(|m| m == "GATED ATTENTION"),
        "every layer of the fused-gate fixture is gated: {seen:?}"
    );
    assert!(
        roles(dir.path(), "layer.0.mixer").contains(&"output gate".to_string()),
        "the gate must appear as an operand, not only in the label"
    );
}

/// Kimi Linear's vocabulary: KDA against MLA, at the `KKKM` cadence.
/// The container that exposed the defect, in miniature.
#[test]
fn the_kda_mla_hybrid_names_both_of_its_programmes() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");
    assert_eq!(mixers(dir.path()), vec!["KDA", "KDA", "KDA", "MLA"]);
}

/// KDA is fifteen operands split where Gated DeltaNet fuses — the
/// structural difference that makes reading one as the other bind the
/// wrong tensors to the wrong roles at plausible shapes.
#[test]
fn a_kda_layer_describes_its_own_fifteen_operands() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");
    assert_eq!(
        roles(dir.path(), "layer.0.mixer"),
        vec![
            "query projection",
            "key projection",
            "value projection",
            "causal conv over q",
            "causal conv over k",
            "causal conv over v",
            "decay gate down",
            "decay gate up",
            "output gate down",
            "output gate up",
            "write-strength projection",
            "log decay",
            "timestep bias",
            "gated norm",
            "output projection",
        ]
    );
}

/// MLA's five, on a layer whose `q_proj`/`o_proj` are byte-identical
/// spellings to the softmax set. Naming them by role is the whole
/// distinction.
#[test]
fn an_mla_layer_describes_the_compressed_kv_set_not_qkvo() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");
    assert_eq!(
        roles(dir.path(), "layer.3.mixer"),
        vec![
            "query projection",
            "compressed kv projection",
            "kv latent norm",
            "kv decompression",
            "output projection",
        ]
    );
}

/// The regression in its purest form: no layer of either hybrid may
/// answer with an empty operand table. An empty table was the shape
/// the wrong answer took.
#[test]
fn no_layer_of_either_hybrid_answers_with_an_empty_table() {
    for (write, name) in [
        (hybrid_lllf_f32_model as fn(&std::path::Path), "lllf"),
        (hybrid_kda_mla_f32_model as fn(&std::path::Path), "kimi"),
    ] {
        let dir = encoded(write, name);
        for n in 0..4 {
            let ops = roles(dir.path(), &format!("layer.{n}.mixer"));
            assert!(!ops.is_empty(), "{name} layer {n} described no operands");
        }
    }
}

/// The q/k/v/o refusal must name the operator the graph declares —
/// the message that previously asserted GATED DELTANET over an MLA
/// layer.
#[test]
fn the_qkvo_refusal_names_the_operator_the_graph_declares() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");

    let kda = describe_facts(dir.path(), "layer.0.attention.q", 2, None).unwrap_err();
    assert!(kda.contains("token mixer is KDA"), "{kda}");
    assert!(kda.contains("layer.0.mixer"), "{kda}");

    let mla = describe_facts(dir.path(), "layer.3.attention.q", 2, None).unwrap_err();
    assert!(mla.contains("token mixer is MLA"), "{mla}");
    assert!(
        !mla.contains("GATED DELTANET"),
        "an MLA layer must never be described as a recurrence: {mla}"
    );
}

/// The precision matrix groups by the declared programme, so a hybrid
/// yields two groups whose columns are each operator's own — never one
/// group wearing the other's schema.
#[test]
fn the_precision_matrix_groups_a_hybrid_by_declared_programme() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");
    let v = precision_matrix_facts(dir.path()).unwrap();
    let programmes = v["programmes"].as_array().unwrap();
    let labels: Vec<&str> = programmes
        .iter()
        .map(|p| p["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, vec!["KDA", "MLA"], "{v}");
    assert_eq!(programmes[0]["layers"], 3, "{v}");
    assert_eq!(programmes[1]["layers"], 1, "{v}");

    let kda_cols: Vec<&str> = programmes[0]["roles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap())
        .collect();
    assert!(kda_cols.contains(&"qconv"), "{kda_cols:?}");
    assert!(kda_cols.contains(&"fa"), "{kda_cols:?}");
    let mla_cols: Vec<&str> = programmes[1]["roles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap())
        .collect();
    assert!(mla_cols.contains(&"kv_a"), "{mla_cols:?}");
    assert!(mla_cols.contains(&"kv_b"), "{mla_cols:?}");
    assert!(
        !mla_cols.contains(&"qconv"),
        "MLA must not carry KDA's columns: {mla_cols:?}"
    );
}

/// An out-of-range layer names an inclusive range. `0..4` is Rust's
/// half-open spelling and reads to everyone else as five layers.
#[test]
fn an_out_of_range_layer_names_an_inclusive_range() {
    let dir = encoded(hybrid_kda_mla_f32_model, "vindex-cli-kimi");
    let err = describe_facts(dir.path(), "layer.9.mixer", 2, None).unwrap_err();
    assert!(err.contains("the plan holds layers 0\u{2013}3"), "{err}");
}

/// What the gated-attention programme's matrix columns actually are —
/// pinned because the film shows this table and the script must quote
/// the columns that render, not the ones it expects.
#[test]
fn the_gated_attention_matrix_carries_a_zgate_column() {
    let dir = encoded(gated_q_f32_model, "vindex-cli-gated");
    let v = precision_matrix_facts(dir.path()).unwrap();
    let p = &v["programmes"][0];
    assert_eq!(p["label"], "GATED ATTENTION", "{v}");
    let cols: Vec<&str> = p["roles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap())
        .collect();
    assert_eq!(
        cols,
        vec!["gate", "up", "down", "q", "k", "v", "o", "zgate"],
        "{v}"
    );
}
