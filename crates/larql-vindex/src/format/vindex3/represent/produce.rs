//! **A plan-v1 search record from a real container** (MEASURE-PLAN-2,
//! amendment A2).
//!
//! Until now only a test fixture built a `SearchSnapshot`, and it filled
//! every judgement field from the Kimi programme. This producer builds a
//! **characterisation-only** plan record: readings can be taken, ingested
//! and replayed, and nothing is judged. The four fields that judge are
//! declared unearned rather than borrowed:
//!
//! ```text
//! gate                None
//! tail_support        requires more observations than any bank holds
//! diagnostic_policy   observes nothing
//! promotion_rule      promotes nothing
//! ```
//!
//! Slice 3's pre-registration installs all four together.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::auto_rep::group_vocabulary;
use super::compile::hash_bytes;
use super::compiler::read_source_identity;
use super::diagnostic::DiagnosticPolicy;
use super::execution_cost::ExecutionCostModel;
use super::map::PrecisionMap;
use super::measure::plan::PROCEDURE;
use super::measurement::{EvidenceScale, TailSupportPolicy};
use super::search_evidence::SearchCalibrationRegistry;
use super::state::accounting::{read_source_storage, PHYSICAL_ACCOUNTING_PROCEDURE};
use super::state::assess::{RankingRule, RankingSemantics};
use super::state::footprint::{compiled_bytes, SurfaceFootprint};
use super::state::instrument::{InstrumentSemantics, MetricSemantics};
use super::state::protocol::MeasurementProtocol;
use super::state::resolved::{layout_admission, PACK_LAYOUT_ADMISSION};
use super::state::semantics::SearchSemantics;
use super::state::snapshot::{Objective, SearchConfig, SearchFacts, SearchSnapshot, SearchSpace};
use super::state::{
    ActionVocabulary, EvidenceBank, MeasurementRegistry, RepresentationState,
    RepresentationStateGraph, ResolvedState, SurfaceTensor, TensorSurface, TransitionPolicy,
};
use super::token_bank::{container_tokenizer_sha256, TokenBank, MANIFEST_FILE, TOKEN_BANK_SCHEMA};
use super::{plan_roles, primary_text_objects, tensor_role, RepresentSpec};
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;

/// The promotion rule of a record that promotes nothing.
pub const NO_PROMOTION: &str = "none/characterisation-only";
/// The diagnostic policy of a record that observes nothing.
pub const NO_DIAGNOSTICS: &str = "none/characterisation-only";
/// How this producer generates candidates: AUTO-REP-1a's proposer.
pub const AUTO_REP_GENERATION: &str = "auto-rep-bnb/v1";

fn refused(message: impl Into<String>) -> VindexError {
    VindexError::Parse(format!("plan record producer: {}", message.into()))
}

/// The instrument plan-v1 is: KL(reference ‖ candidate) in nats over the
/// full vocabulary, every teacher-forced position.
pub fn plan_instrument() -> InstrumentSemantics {
    InstrumentSemantics::new(
        "kl(reference || candidate), nats",
        "mean, p50, p99, max over positions",
        "teacher-forced, every position",
        PROCEDURE,
    )
    .with_semantics(MetricSemantics::PLAN_V1)
    .expect("the plan-v1 instrument is untruncated and declares full-vocabulary support")
}

/// **Every tensor of the container, with the role the compiler gives
/// it**: the plan's own operand binding for primary-text objects, name
/// classification otherwise. The same helper `compile_representation`
/// uses, so the solver groups by the roles the compiler compiles.
pub fn plan_surface(source: &Path) -> Result<TensorSurface, VindexError> {
    let inspection = inspect_container(source, false)?;
    let primary_text = primary_text_objects(&inspection);
    let declared = plan_roles::plan_roles(source, &inspection);
    let raw = std::fs::read_to_string(source.join(INDEX_JSON))
        .map_err(|e| refused(format!("read {INDEX_JSON}: {e}")))?;
    let index: Vindex3Index =
        serde_json::from_str(&raw).map_err(|e| refused(format!("parse {INDEX_JSON}: {e}")))?;
    let mut entries = BTreeMap::new();
    for rep in index.representations.values() {
        let (header, _) = read_segment_header(&source.join(&rep.segment))?;
        for tensor in &header.tensors {
            let role = tensor_role(&declared, &primary_text, &rep.object, tensor);
            entries.insert(
                (rep.object.clone(), tensor.name.clone()),
                SurfaceTensor::new(&rep.object, &tensor.name, role, tensor.shape.clone()),
            );
        }
    }
    TensorSurface::new(entries.into_values())
}

/// What a plan record is produced from.
pub struct ProduceInputs<'a> {
    /// The source container every candidate is compiled from.
    pub source: &'a Path,
    /// Encoding and roles. Its protections must be empty: the solver owns
    /// every exception.
    pub spec: &'a RepresentSpec,
    /// A `teacher-forced-token-bank/v1` bank exported with this
    /// container's tokenizer.
    pub bank: &'a Path,
    /// Samples each measurement reads, from `seq-000` in bank order.
    pub sequences: usize,
}

/// **Produce a characterisation-only plan-v1 record.**
pub fn produce(inputs: &ProduceInputs<'_>) -> Result<SearchSnapshot, VindexError> {
    if !inputs.spec.protect.is_empty() {
        return Err(refused(
            "the spec protects tensors; the solver owns every exception",
        ));
    }
    let bank = TokenBank::open(inputs.bank).map_err(|e| refused(format!("token bank: {e}")))?;
    let tokenizer = container_tokenizer_sha256(inputs.source)
        .map_err(|e| refused(format!("container tokenizer: {e}")))?;
    bank.check_tokenizer(&tokenizer)
        .map_err(|e| refused(format!("the bank is not this model's: {e}")))?;
    let held = bank.sample_count();
    if inputs.sequences == 0 || inputs.sequences > held {
        return Err(refused(format!(
            "{} sequences requested; the bank holds {held}",
            inputs.sequences
        )));
    }
    let manifest = std::fs::read(inputs.bank.join(MANIFEST_FILE))
        .map_err(|e| refused(format!("bank manifest: {e}")))?;
    let evidence = EvidenceBank::new(
        TOKEN_BANK_SCHEMA,
        hash_bytes(&manifest),
        bank.manifest().samples[..inputs.sequences]
            .iter()
            .map(|s| s.id.clone()),
        // A variable-length bank declares no per-sample length.
        0,
    );
    let protocol = MeasurementProtocol::new(evidence, plan_instrument(), PROCEDURE);

    let model = read_source_identity(inputs.source)?;
    let accounting = read_source_storage(inputs.source, &model)?;
    let surface = plan_surface(inputs.source)?;
    let spec = inputs.spec;
    let base_map =
        PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);

    // The root is the base map resolved and priced by the procedure the
    // record declares.
    let layout = layout_admission(PACK_LAYOUT_ADMISSION)?;
    let root_state = RepresentationState::resolve(&model, &surface, &base_map, layout);
    let bound = accounting
        .bind(&model, &surface)
        .map_err(|e| refused(format!("accounting: {e}")))?;
    let footprint = SurfaceFootprint::new(
        &bound,
        &surface,
        layout,
        compiled_bytes(PHYSICAL_ACCOUNTING_PROCEDURE)?,
        std::slice::from_ref(&spec.encoding),
    )
    .map_err(|e| refused(format!("pricing: {e}")))?;
    let root_bytes = footprint
        .try_logical_bytes(&root_state)
        .map_err(|e| refused(format!("pricing the root: {e}")))?;

    let mut space = SearchSpace {
        surface,
        base_map,
        vocabulary: ActionVocabulary::new([])?,
        applied: BTreeSet::new(),
    };
    space.vocabulary = group_vocabulary(&space)?;

    let config = SearchConfig {
        objective: Objective::MinimiseLogicalBytes,
        gate: None,
        tail_support: TailSupportPolicy {
            // More observations than any bank holds: were anything to read
            // it, no percentile would price. Nothing does while the record
            // has no gate.
            min_tail_observations: f64::MAX,
            provenance: "not earned for plan-v1: a characterisation-only record adjudicates \
                         nothing; slice 3 pre-registers the tail policy with the gate"
                .into(),
        },
        calibrations: SearchCalibrationRegistry::default(),
        diagnostic_policy: DiagnosticPolicy {
            id: NO_DIAGNOSTICS.into(),
            observations: Vec::new(),
        },
        semantics: SearchSemantics::new(
            AUTO_REP_GENERATION,
            "ruling-1-three-prunes/v1",
            "search-evidence-ladder/v1",
            NO_PROMOTION,
            "physical-prize-first/v1",
            PHYSICAL_ACCOUNTING_PROCEDURE,
            PACK_LAYOUT_ADMISSION,
        ),
        ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
        standing_intent: protocol.intent(EvidenceScale::Authority),
        protocol: Some(protocol),
    };
    let facts = SearchFacts {
        graph: RepresentationStateGraph::new(
            TransitionPolicy::Unconstrained,
            ResolvedState::new(root_state, root_bytes),
        ),
        measurements: MeasurementRegistry::default(),
        byte_ledgers: BTreeMap::new(),
        execution_cost: ExecutionCostModel::new(Vec::new()),
        accounting: Some(accounting),
    };
    Ok(SearchSnapshot::new(space, config, facts))
}
