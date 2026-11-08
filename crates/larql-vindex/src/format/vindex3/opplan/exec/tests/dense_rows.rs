//! Every mixer's dense operands are checked against the backend's projector BEFORE the layer runs. The
//! reference projector consumes f32 only; a compact slab reaching it used to panic mid-layer (`metal-nvfp4`
//! on Qwen3.5). Now it is a typed refusal naming the layer, the operand and the representation.

use super::super::conv_qkv::ConvQkvWeights;
use super::super::cpu::physical::ExecutorProjections;
use super::super::cpu::WeightRows;
use super::super::dense_rows::{consuming_projector, DenseRowSet};
use super::super::gated_delta::{DenseProjections, GatedDeltaWeights, ScalarProjections};
use super::super::kda::{KdaOutputGateWeights, KdaWeights};
use super::super::mamba2::Mamba2Weights;
use super::super::mla::{MlaQueryWeights, MlaWeights};
use larql_models::config::KdaGateForm;

const F32_ROWS: [f32; 4] = [1.0; 4];
const BF16_ROWS: [u16; 4] = [0x3f80; 4];
const GLUE: [f32; 4] = [0.0; 4];
const LAYER: usize = 7;

fn f32_rows() -> WeightRows<'static> {
    WeightRows::F32(&F32_ROWS)
}

fn bf16_rows() -> WeightRows<'static> {
    WeightRows::Bf16(&BF16_ROWS)
}

fn gdn(out_proj: WeightRows<'static>) -> GatedDeltaWeights<'static> {
    GatedDeltaWeights {
        in_proj_qkv: f32_rows(),
        in_proj_a: f32_rows(),
        in_proj_b: f32_rows(),
        in_proj_z: f32_rows(),
        out_proj,
        conv1d: &GLUE,
        a_log: &GLUE,
        dt_bias: &GLUE,
        norm: &GLUE,
        norm_eps: 1e-6,
    }
}

/// The reference projector's refusal for `w`, as text.
fn refusal<W: DenseRowSet>(w: &W) -> String {
    match consuming_projector(&ScalarProjections, w, LAYER) {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e.to_string(),
    }
}

fn names<W: DenseRowSet>(w: &W) -> Vec<&'static str> {
    w.dense_rows().into_iter().map(|(n, _)| n).collect()
}

#[test]
fn the_reference_projector_consumes_f32_only() {
    assert!(ScalarProjections.consumes(f32_rows()));
    assert!(!ScalarProjections.consumes(bf16_rows()));
    assert!(ExecutorProjections.consumes(bf16_rows()));
}

#[test]
fn a_compact_operand_is_refused_by_name_before_the_layer_runs() {
    let err = match consuming_projector(&ScalarProjections, &gdn(bf16_rows()), LAYER) {
        Ok(_) => panic!("a bf16 slab must not reach the reference projector"),
        Err(e) => e.to_string(),
    };
    for part in ["layer 7", "Gated DeltaNet", "`out_proj`", "bf16"] {
        assert!(err.contains(part), "{part} missing from: {err}");
    }
}

#[test]
fn an_all_f32_layer_and_a_capable_projector_both_pass() {
    assert!(consuming_projector(&ScalarProjections, &gdn(f32_rows()), LAYER).is_ok());
    assert!(consuming_projector(&ExecutorProjections, &gdn(bf16_rows()), LAYER).is_ok());
}

#[test]
fn every_mixer_names_all_its_dense_operands() {
    assert_eq!(
        names(&gdn(f32_rows())),
        [
            "in_proj_qkv",
            "in_proj_a",
            "in_proj_b",
            "in_proj_z",
            "out_proj"
        ]
    );

    let mamba = Mamba2Weights {
        in_proj: f32_rows(),
        out_proj: bf16_rows(),
        conv1d: &GLUE,
        conv1d_bias: None,
        a_log: &GLUE,
        d: &GLUE,
        dt_bias: &GLUE,
        norm: None,
        norm_eps: 1e-6,
    };
    assert_eq!(names(&mamba), ["in_proj", "out_proj"]);
    assert!(refusal(&mamba).contains("Mamba2"));

    let conv = ConvQkvWeights {
        in_proj: bf16_rows(),
        out_proj: f32_rows(),
        conv1d: &GLUE,
        conv1d_bias: None,
    };
    assert_eq!(names(&conv), ["in_proj", "out_proj"]);
    assert!(refusal(&conv).contains("conv-QKV"));

    let kda = |output_gate| KdaWeights {
        q_proj: f32_rows(),
        k_proj: f32_rows(),
        v_proj: f32_rows(),
        q_conv1d: &GLUE,
        k_conv1d: &GLUE,
        v_conv1d: &GLUE,
        f_a_proj: &GLUE,
        f_b_proj: &GLUE,
        output_gate,
        b_proj: &GLUE,
        a_log: &GLUE,
        dt_bias: &GLUE,
        o_norm: &GLUE,
        o_proj: f32_rows(),
        norm_eps: 1e-6,
        gate_rank: 1,
        gate_form: KdaGateForm::Softplus,
    };
    let low_rank = kda(KdaOutputGateWeights::LowRank {
        g_a_proj: &GLUE,
        g_b_proj: &GLUE,
    });
    assert_eq!(names(&low_rank), ["q_proj", "k_proj", "v_proj", "o_proj"]);
    let full_rank = kda(KdaOutputGateWeights::FullRank {
        g_proj: bf16_rows(),
    });
    assert_eq!(
        names(&full_rank),
        ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"]
    );
    assert!(refusal(&full_rank).contains("`g_proj`"));

    let mla = |query, output_gate| MlaWeights {
        query,
        kv_a_proj: f32_rows(),
        kv_a_norm: &GLUE,
        kv_b_proj: f32_rows(),
        o_proj: f32_rows(),
        kv_a_norm_eps: 1e-6,
        output_gate,
    };
    let direct = mla(MlaQueryWeights::Direct { q_proj: f32_rows() }, None);
    assert_eq!(
        names(&direct),
        ["q_proj", "kv_a_proj", "kv_b_proj", "o_proj"]
    );
    let low = mla(
        MlaQueryWeights::LowRank {
            q_a_proj: f32_rows(),
            q_a_norm: &GLUE,
            q_b_proj: bf16_rows(),
            q_a_norm_eps: 1e-6,
        },
        Some(f32_rows()),
    );
    assert_eq!(
        names(&low),
        [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj",
            "kv_b_proj",
            "o_proj",
            "g_proj"
        ]
    );
    assert!(refusal(&low).contains("MLA"));
}
