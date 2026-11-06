//! Rung 3b: the prepared plan resolves every planned operand's
//! representation, derives the candidates from what the registry declares,
//! pins one realization with its reason, and refuses — before any byte is
//! read — when nothing can serve. The old boolean ladder is kept here as
//! an oracle so the selector's answers are pinned to the pre-rung-3
//! behaviour, class by class, size by size, arm by arm.

use super::super::cpu::physical::{KQuantExecution, PhysicalProjectionPlan};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::{select_cpu, ProductionBackend};
use super::super::realization::{
    MappedAccess, RealizationForm, RealizationId, RepresentationFacts, SelectionReason,
};
use super::super::reference::ReferenceBackend;
use super::bf16_zlib_execution::{transcode, Transcode};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{MatrixClass, WeightFormat};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::{RepresentationExtent, ResidencyProfile};

/// Below and above the size policy's cache threshold.
const SMALL: usize = 64 * 64;
const LARGE: usize = 17408 * 5120;
const PROJECTION_SUFFIX: &str = "_proj.weight";

struct Prepared {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn dense(into: Option<Transcode>) -> Prepared {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
    if let Some(into) = into {
        let transcoded = transcode(
            container.path(),
            |name, shape| shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX),
            into,
        );
        assert!(!transcoded.is_empty());
    }
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Prepared {
        _src: src,
        _container: container,
        plan,
        store,
    }
}

fn is_projection(operation: Operation) -> bool {
    matches!(operation, Operation::Project(_) | Operation::OutputHead)
}

/// The pre-rung-3 policy, copied verbatim as a MIGRATION oracle: three
/// stored-dtype booleans and a size policy. The selector must land on the
/// same resident representation for every class, size, label and arm —
/// today. This ladder is transitional, not the new semantic authority:
/// once 3c's independent accounting and 3d's provider inventories exist,
/// new behaviour is allowed to diverge from it deliberately, and this
/// test is then narrowed or retired rather than preserving old mistakes.
fn old_ladder(
    class: MatrixClass,
    elements: usize,
    stored_bf16: bool,
    stored_nvfp4: bool,
    stored_kquant: bool,
    kquant: KQuantExecution,
) -> WeightFormat {
    match class {
        MatrixClass::RoutedExpertBank => WeightFormat::F32,
        _ if stored_nvfp4 => WeightFormat::Nvfp4,
        _ if stored_kquant && kquant == KQuantExecution::Direct => WeightFormat::KQuant,
        MatrixClass::AttentionProjection | MatrixClass::FfnProjection | MatrixClass::OutputHead => {
            PhysicalProjectionPlan::choose_for(Some(class), elements, stored_bf16).format()
        }
    }
}

fn synthetic(operation: Operation, elements: usize) -> PlannedOperand {
    PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.w".into(),
            dtype: String::new(),
            shape: vec![elements, 1],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: elements,
    }
}

use super::super::quantise::{Q4_BLOCK, Q8_BLOCK};
use super::super::realization::{
    class_of, common_selection, cpu_projection_candidates, resident_profile, RealizationBackend,
    RefusalKind, SelectionRefusal, SelectionRefusals,
};
use crate::format::vindex3::represent::codec::codecs::float::BF16;
use crate::format::vindex3::represent::codec::{CodecRegistry, RequiredAccess, ResidencyClass};

fn every_form() -> Vec<RealizationForm> {
    vec![
        RealizationForm::Direct(PhysicalProjectionPlan::FusedBf16),
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32),
        RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8),
        RealizationForm::SliceStored {
            convert: WeightFormat::F16,
        },
        RealizationForm::DecodedGather,
        RealizationForm::DeviceResident(WeightFormat::Nvfp4),
    ]
}

mod records_on_real_containers;
mod the_selector_answers_what_the_boolean_la;
mod the_stored_source_forbids_requantise;
mod the_vocabulary_itself;
