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

// ── #751: the whole-plan check's CALL PLACEMENT ────────────────────────
//
// `a_prepared_plan_is_checked_whole_before_any_layer_runs` tests the check
// itself; nothing tested that `traverse` and the decode session call it
// up front. These drive both entries with a backend that is production in
// every respect the preparation sees (name, identity, selection) but whose
// dense projector consumes nothing, and whose every compute method panics.
// The run must come back as the typed refusal: deleting either call lets
// it reach `embed` and panic, so the call placement is what is under test.

mod placement {
    use super::super::super::backend::{
        AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, FfnCall, NormCall,
        PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
    };
    use super::super::super::decode::DecodeSession;
    use super::super::super::gated_delta::DenseProjections;
    use super::super::super::kv::{KvState, RowKvState};
    use super::super::super::lowering::LoweringIdentity;
    use super::super::super::prepared::{ExecutionSlice, PreparedOperands};
    use super::super::super::production::ProductionBackend;
    use super::super::super::realization::{RepresentationFacts, Selection, SelectionRefusal};
    use super::super::hybrid_traversal::hybrid;
    use super::ConsumesNothing;
    use crate::error::VindexError;
    use crate::format::vindex3::opplan::planned::PlannedOperand;

    const NO_LAYER_RUNS: &str = "the whole-plan check must refuse before any layer runs";

    /// Production as far as preparation can tell; consumes no dense rows.
    struct UnconsumingProduction(ProductionBackend);

    impl PlanBackend for UnconsumingProduction {
        fn name(&self) -> &str {
            self.0.name()
        }
        fn identity(&self) -> LoweringIdentity {
            self.0.identity()
        }
        fn select(
            &self,
            operand: &PlannedOperand,
            facts: &RepresentationFacts,
        ) -> Result<Selection, Box<SelectionRefusal>> {
            self.0.select(operand, facts)
        }
        fn dense_projector(&self) -> &dyn DenseProjections {
            &ConsumesNothing
        }
        fn embed(&self, _: &[f32], _: usize, _: u32, _: Option<f32>) -> Vec<f32> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn norm(&self, _: NormCall<'_>) -> Vec<f32> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn project(&self, _: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn attention(&self, _: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn attention_step(
            &self,
            _: AttentionStepCall<'_>,
        ) -> Result<AttentionStepOut, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn ffn(&self, _: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn routed_ffn(&self, _: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn output_head(
            &self,
            _: WeightSlice<'_>,
            _: usize,
            _: usize,
            _: &[f32],
            _: Option<f64>,
            _: Option<f32>,
        ) -> Result<Vec<f32>, VindexError> {
            panic!("{NO_LAYER_RUNS}")
        }
        fn residual_add(&self, _: &mut [f32], _: &[f32]) {
            panic!("{NO_LAYER_RUNS}")
        }
    }

    fn assert_refused_naming_the_first_mixer(err: VindexError) {
        let msg = err.to_string();
        assert!(
            msg.contains("layer 0") && msg.contains("Gated DeltaNet"),
            "{msg}"
        );
    }

    #[test]
    fn the_traversal_refuses_before_any_plane_or_position() {
        let (_dir, plan, store) = hybrid();
        let ops = PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap();
        let backend = UnconsumingProduction(ProductionBackend::new());
        let mut kv = RowKvState::default();
        let mut events = 0usize;
        let err = super::super::super::prefill_prepared_observed(
            &plan,
            &ops,
            &[1, 2, 3],
            &backend,
            &mut kv,
            &mut |_| {
                events += 1;
                Ok(())
            },
        )
        .expect_err("a projector that consumes nothing is refused");
        assert_refused_naming_the_first_mixer(err);
        assert_eq!(events, 0, "no plane was emitted");
        assert_eq!(kv.position(), 0, "the provider did not advance");
    }

    #[test]
    fn the_decode_session_refuses_before_it_exists() {
        let (_dir, plan, store) = hybrid();
        let ops = PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap();
        let backend = UnconsumingProduction(ProductionBackend::new());
        let mut kv = RowKvState::default();
        let err = match DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv) {
            Ok(_) => panic!("a projector that consumes nothing is refused"),
            Err(e) => e,
        };
        assert_refused_naming_the_first_mixer(err);
        assert_eq!(kv.position(), 0, "the provider did not advance");
    }
}
