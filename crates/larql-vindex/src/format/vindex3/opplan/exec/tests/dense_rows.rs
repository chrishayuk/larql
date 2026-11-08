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
/// Any norm epsilon: these weights never reach arithmetic.
const EPS: f32 = 1e-6;

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
        norm_eps: EPS,
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
        norm_eps: EPS,
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
        norm_eps: EPS,
        gate_rank: 1,
        gate_form: KdaGateForm::Softplus,
    };
    let low_rank = kda(KdaOutputGateWeights::LowRank {
        g_a_proj: &GLUE,
        g_b_proj: &GLUE,
    });
    assert_eq!(names(&low_rank), ["q_proj", "k_proj", "v_proj", "o_proj"]);
    // A full-rank g_proj is projected by the CPU executor directly, never by the backend's projector, so it
    // is not a dense row of this check: a compact one must not refuse a layer that runs.
    let full_rank = kda(KdaOutputGateWeights::FullRank {
        g_proj: bf16_rows(),
    });
    assert_eq!(names(&full_rank), ["q_proj", "k_proj", "v_proj", "o_proj"]);
    assert!(consuming_projector(&ScalarProjections, &full_rank, LAYER).is_ok());

    let mla = |query, output_gate| MlaWeights {
        query,
        kv_a_proj: f32_rows(),
        kv_a_norm: &GLUE,
        kv_b_proj: f32_rows(),
        o_proj: f32_rows(),
        kv_a_norm_eps: f64::from(EPS),
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
            q_a_norm_eps: f64::from(EPS),
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

/// A projector with no kernel at all, so the whole-plan check's reach can be shown on any real plan.
struct ConsumesNothing;

impl DenseProjections for ConsumesNothing {
    fn consumes(&self, _weight: WeightRows<'_>) -> bool {
        false
    }
    fn project(&self, _weight: WeightRows<'_>, _x: &[f32], _out_dim: usize) -> Vec<f32> {
        unreachable!("the whole-plan check refuses before any projection")
    }
}

/// The whole-plan form, over a REAL prepared plan (the Gated DeltaNet hybrid fixture under the production backend):
/// a projector that consumes nothing is refused for the whole plan before any layer runs, naming the first
/// mixer layer and operand; the reference projector, which consumes the f32 rows this backend prepared, and the
/// production executor's both pass.
#[test]
fn a_prepared_plan_is_checked_whole_before_any_layer_runs() {
    use super::super::dense_rows::refuse_unconsumable_plan;
    use super::super::operands::OperandStore;
    use super::super::prepared::{ExecutionSlice, PreparedOperands};
    use super::super::production::ProductionBackend;
    use crate::format::vindex3::encode::encode_system;
    use crate::format::vindex3::fixtures::hybrid_lllf_f32_model;
    use crate::format::vindex3::inspect::inspect_container;
    use crate::format::vindex3::opplan::plan_component_ops;

    let src = tempfile::tempdir().unwrap();
    hybrid_lllf_f32_model(src.path());
    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("hybrid".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    let err = refuse_unconsumable_plan(&plan, &ops, &ConsumesNothing)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("layer 0") && err.contains("Gated DeltaNet") && err.contains("`in_proj_qkv`"),
        "{err}"
    );
    refuse_unconsumable_plan(&plan, &ops, &ScalarProjections)
        .expect("the reference consumes the f32 rows");
    refuse_unconsumable_plan(&plan, &ops, &ExecutorProjections)
        .expect("the production projector consumes it");
}
