//! **Stage 4: the facade renders, and derives nothing.**
//!
//! Every test below runs against the Rung 5 record — the same facts the
//! 1d replay gate uses, reloaded from JSON so that what is rendered came
//! out of storage and not out of the object that built it.
//!
//! Two kinds of check, and both are needed:
//!
//! ```text
//! origin      every rendered FIELD names a substrate call, and every
//!             declared call is reached — the registry cannot rot
//! render      every rendered VALUE equals the substrate's own answer,
//!             and the real Rung 5 numbers survive the round trip
//! ```
//!
//! The first alone would pass on a view that declared honest origins and
//! then rendered nonsense. The second alone would pass on a view that
//! grew an undeclared field nobody thought to assert.

mod compare;
mod current;
mod describe;
mod evidence;
mod explain;
mod frontier;
mod next_experiment;
mod origin;
mod render;

use std::collections::BTreeSet;

use super::super::compiler::read_source_identity;
use super::super::diagnostic::DiagnosticPolicy;
use super::super::execution_cost::ExecutionCostModel;
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::TailSupportPolicy;
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::super::quality::kimi_logit_balanced_v1;
use super::super::search_evidence::SearchCalibrationRegistry;
use super::super::state::accounting::read_source_storage;
use super::super::state::action_space::{ActionVocabulary, MapEdit};
use super::super::state::candidate::Footprint;
use super::super::state::fixtures;
use super::super::state::graph::{RepresentationStateGraph, TransitionPolicy};
use super::super::state::identity::RepresentationState;
use super::super::state::key::MeasurementRegistry;
use super::super::state::realization::ResolvedState;
use super::super::state::resolved::{layout_admission, PACK_LAYOUT_ADMISSION};
use super::super::state::semantics::SearchSemantics;
use super::super::state::snapshot::{
    Objective, SearchConfig, SearchFacts, SearchSnapshot, SearchSpace,
};
use super::super::state::surface::{SurfaceTensor, TensorSurface};
use super::super::state::{PackCompiledBytes, RankingRule, RankingSemantics, SurfaceFootprint};
use super::OptimizerView;

/// The Rung 5 record, stored and read back.
pub(super) fn reloaded() -> SearchSnapshot {
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_str(&json).expect("deserialize");
    back.check_schema().expect("schema");
    back
}

/// A facade over the reloaded record.
pub(super) fn view(snapshot: &SearchSnapshot) -> OptimizerView<'_> {
    OptimizerView::new(snapshot)
}

/// **A search record over a REAL encoded container.**
///
/// Every fact traces to the container's own sealed authority: the
/// surface is what it stores, the model identity is read from its
/// index, and the accounting facts are read through the segment digests
/// that identity seals. Nothing derived is stored — no bind, no price
/// table, no footprint, no ranking.
pub(super) fn priced_record(container: &std::path::Path) -> SearchSnapshot {
    priced_record_from(container, BTreeSet::new())
}

/// The same record asked from a given applied set.
pub(super) fn priced_record_from(
    container: &std::path::Path,
    applied: BTreeSet<String>,
) -> SearchSnapshot {
    priced_record_under(
        container,
        applied,
        PACK_LAYOUT_ADMISSION,
        "logical-bytes/v1",
    )
}

/// The same record under DECLARED policies of the caller's choosing.
///
/// The point of the parameters is that they are the record's, not the
/// caller's: whatever is named here is what both state resolution and
/// price-table construction must resolve to, and nothing may quietly
/// fall back to this build's favourite.
/// The same record over a surface of a chosen SHAPE.
///
/// The shape decides whether the pack layout can hold the tensor, which
/// is what makes the declared layout policy observable.
pub(super) fn priced_record_shaped(
    container: &std::path::Path,
    shape: Vec<usize>,
    declared_layout: &str,
) -> SearchSnapshot {
    build(
        container,
        BTreeSet::new(),
        declared_layout,
        "logical-bytes/v1",
        shape,
        None,
        false,
    )
}

/// A record whose accounting facts were read from ANOTHER container.
pub(super) fn priced_record_with_foreign_accounting(
    container: &std::path::Path,
    other: &std::path::Path,
) -> SearchSnapshot {
    build(
        container,
        BTreeSet::new(),
        PACK_LAYOUT_ADMISSION,
        "logical-bytes/v1",
        vec![64, 64],
        Some(other.to_path_buf()),
        false,
    )
}

/// A record whose two edits resolve DIFFERENTLY, so the policy has more
/// than one opportunity to order.
pub(super) fn priced_record_ranked(container: &std::path::Path) -> SearchSnapshot {
    build(
        container,
        BTreeSet::new(),
        PACK_LAYOUT_ADMISSION,
        "logical-bytes/v1",
        vec![64, 64],
        None,
        true,
    )
}

pub(super) fn priced_record_under(
    container: &std::path::Path,
    applied: BTreeSet<String>,
    declared_layout: &str,
    declared_accounting: &str,
) -> SearchSnapshot {
    build(
        container,
        applied,
        declared_layout,
        declared_accounting,
        vec![64, 64],
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn build(
    container: &std::path::Path,
    applied: BTreeSet<String>,
    declared_layout: &str,
    declared_accounting: &str,
    shape: Vec<usize>,
    accounting_from: Option<std::path::PathBuf>,
    distinct_edits: bool,
) -> SearchSnapshot {
    let model = read_source_identity(container).expect("identity");
    let facts_root = accounting_from.unwrap_or_else(|| container.to_path_buf());
    let facts_model = read_source_identity(&facts_root).expect("identity");
    let accounting = read_source_storage(&facts_root, &facts_model).expect("storage facts");
    let priced = read_source_storage(container, &model).expect("storage facts");

    // The container's own tensors, at an NVFP4-admissible shape.
    let surface = TensorSurface::new(priced.tensors().map(|(id, _)| {
        SurfaceTensor::new(&id.object, &id.tensor, Role::DecoderLinear, shape.clone())
    }))
    .expect("one entry per stored tensor");

    // The role is in the map's domain and a blanket exception protects
    // it, so the base state presents source bytes throughout. Each edit
    // lifts that protection, which is the direction the objective wants.
    let compile_everything = || Exception {
        projection: None,
        layers: None,
        encoding: Some(DTYPE_NVFP4.into()),
    };
    let base_map = PrecisionMap {
        name: "protect-everything".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![Exception {
            projection: None,
            layers: None,
            encoding: None,
        }],
    };
    // Two edits that resolve identically: one physical state reached by
    // two routes, which is the case `MeasurementOpportunity` exists for
    // — two realizations, ONE experiment.
    let second = match distinct_edits {
        // Compiles only the q projections, so it resolves to a
        // different physical state and the policy has two opportunities
        // to order rather than two routes to one.
        true => Exception {
            projection: Some("q_proj".into()),
            layers: None,
            encoding: Some(DTYPE_NVFP4.into()),
        },
        false => compile_everything(),
    };
    let vocabulary = ActionVocabulary::new([
        MapEdit::new("compile-all", compile_everything()),
        MapEdit::new("compile-all-by-another-name", second),
    ])
    .expect("distinct names");

    // The root: the base map resolved, priced by the same procedure the
    // record declares. A writer may compute; the record stores only the
    // resulting node.
    let layout = layout_admission(PACK_LAYOUT_ADMISSION).expect("this build implements it");
    let root_state = RepresentationState::resolve(&model, &surface, &base_map, layout);
    let root_bytes = priced
        .bind(&model, &surface)
        .ok()
        .and_then(|bound| {
            SurfaceFootprint::new(
                &bound,
                &surface,
                layout,
                &PackCompiledBytes,
                &[DTYPE_NVFP4.to_string()],
            )
            .ok()
            .map(|f| f.logical_bytes(&root_state))
        })
        .unwrap_or_else(|| {
            // A shape the pack cannot hold prices as source throughout,
            // which is what the base map presents anyway.
            crate::format::vindex3::represent::state::realization::LogicalBytes::new(
                priced.tensors().map(|(_, f)| f.logical_bytes.get()).sum(),
            )
        });
    let root = ResolvedState::new(root_state.clone(), root_bytes);

    SearchSnapshot::new(
        SearchSpace {
            surface,
            base_map,
            vocabulary,
            applied,
        },
        SearchConfig {
            objective: Objective::MinimiseLogicalBytes,
            gate: kimi_logit_balanced_v1(),
            tail_support: TailSupportPolicy::route_cal_1(),
            calibrations: SearchCalibrationRegistry::default(),
            diagnostic_policy: DiagnosticPolicy::bs2_kimi_v1(),
            semantics: SearchSemantics::new(
                "exchange-1-out-1-in/v1",
                "ruling-1-three-prunes/v1",
                "search-evidence-ladder/v1",
                "kimi-balanced-v1-authority-only/v1",
                "physical-prize-first/v1",
                declared_accounting,
                declared_layout,
            ),
            ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
            standing_intent: fixtures::standing_intent(),
        },
        SearchFacts {
            graph: RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, root),
            measurements: MeasurementRegistry::default(),
            byte_ledgers: Default::default(),
            execution_cost: ExecutionCostModel::new(Vec::new()),
            accounting: Some(accounting),
        },
    )
}
