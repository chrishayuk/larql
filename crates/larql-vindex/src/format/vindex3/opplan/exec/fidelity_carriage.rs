//! Carriage of the composed fidelity bound into selection.
//!
//! [`attested_fidelity`](super::attested_fidelity) is the ALGEBRA — the
//! three phases, and what may influence a decision. This module is the
//! CARRIAGE: the one place that runs those phases over a plan's records
//! and replaces each extent option's declared certificate with the
//! derived one, so the floor that gates selection judges a bound a
//! caller may actually rely on.
//!
//! Kept apart from `prepared.rs` deliberately. The planner is already
//! large, and this is a self-contained pass with one entry point and no
//! knowledge of realizations, backends or budgets — it takes records and
//! a source, and it edits certificates.

use std::collections::BTreeMap;

use super::accounting::RepresentationFloor;
use super::attested_fidelity::{admit, VerifiedEvidence};
use super::operands::OperandSource;
use super::realization::{DependencyPin, RealizationRecord, RepresentationFacts};
use crate::error::VindexError;
use crate::format::vindex3::auxiliary_references::OperandAddress;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{
    CodecRegistry, FidelityCertificate, RepresentationCodec, RepresentationExtent,
};
use crate::format::vindex3::representation_attestations::terminal_baseline;
use crate::format::vindex3::representation_attestations::tuple::AttestedSubject;

/// **B3's carriage.** Replace every extent option's DECLARED certificate
/// with the DERIVED one, so the floor that gates selection judges the
/// bound a caller may actually rely on.
///
/// Until this ran, `shallowest_saving` read `option.certificate.radius`
/// straight from the codec's declaration — a bound that knows nothing
/// about the dependency the extent decodes through. A codebook read at a
/// shallow extent moves every value that indexes it, so the declared
/// number is an over-promise for any codec with an auxiliary, and it was
/// harmless only because no shipped codec both declared a radius and
/// required one.
///
/// The three phases are the module's, in order, and the order is the
/// security boundary rather than a preference:
///
/// 1. [`admit`] over every planned operand — metadata only, no payload
///    read, so a container that attests nothing costs nothing here.
/// 2. [`Admitted::verify`] — hashes exactly what admission bound. These
///    reads are PREPARATION WORK: they resolve through `load_raw`, so
///    they land in the store's consumption ledger (`load_count`,
///    `touched_objects`) like every other preparation read. The
///    `ResourceLedger`'s `prepare` figure is priced from declared
///    extents and does NOT yet include them — that is debt D1, named by
///    this wave's freeze and not paid here; what matters is that the
///    reads are counted somewhere real rather than absorbed.
/// 3. [`VerifiedEvidence::derive`] per option — the attested claim where
///    one was verified, the codec's declaration otherwise, composed with
///    each dependency's certificate.
///
/// UNAVAILABLE, NEVER OPTIMISTIC. A missing dependency pin, a dependency
/// whose codec declares no radius, and a composition the metrics refuse
/// all leave the option with no radius at all. `RepresentationFloor`
/// already reads that correctly — an extent declaring no radius is
/// admissible only as the terminal one, because an undeclared error is
/// not a small one — so every one of those cases refuses a shallower
/// extent rather than admitting it on a bound nobody computed. A
/// composition refusal is deliberately not fatal to the plan: a floor of
/// `Exact` never consults a radius, and a container whose metrics
/// disagree should still be plannable at full depth.
pub(super) fn compose_extent_certificates(
    records: &mut [(RealizationRecord, RepresentationFacts)],
    store: OperandSource<'_>,
) -> Result<(), VindexError> {
    let registry = store.registry();
    let table = store.store().attestations();
    let recognised = store.store().recognised();

    // Phase one's inputs, owned, because an `AttestedSubject` borrows
    // what the container says rather than copying it — which is the
    // point: a subject built from anything but the container would be
    // comparing the attestation against itself.
    struct Facts {
        depth: u32,
        address: OperandAddress,
        shape: Vec<usize>,
        family: String,
        revision: u32,
        baselines: BTreeMap<String, String>,
        operand: OperandRef,
    }

    let mut facts = Vec::new();
    for (record, _) in records.iter() {
        let Some(codec) = registry.by_label(&record.representation) else {
            continue;
        };
        let identity = codec.identity();
        let address = OperandAddress::new(
            &record.planned.operand.object,
            &record.planned.operand.tensor,
        );
        // Only depths this operand is actually attested at. `depths` is a
        // metadata read, so an unattested operand — every operand in
        // every container written before this wave — adds no work.
        for depth in table.depths(&address) {
            facts.push(Facts {
                depth,
                address: address.clone(),
                shape: record.planned.operand.shape.clone(),
                family: identity.family.clone(),
                revision: identity.revision,
                baselines: auxiliary_baselines(
                    registry,
                    codec,
                    RepresentationExtent { depth },
                    &record.dependencies,
                ),
                operand: record.planned.operand.clone(),
            });
        }
    }

    let subjects: Vec<(AttestedSubject<'_>, OperandRef)> = facts
        .iter()
        .map(|f| {
            (
                AttestedSubject {
                    operand: &f.address,
                    extent_depth: f.depth,
                    codec_family: &f.family,
                    codec_revision: f.revision,
                    shape: &f.shape,
                    auxiliary_baselines: f.baselines.clone(),
                    // A reader holding only the container knows neither.
                    // They are carried as identity for a verifier that
                    // does — see the module note on B4.
                    expected_source_digest: None,
                    expected_recipe: None,
                },
                f.operand.clone(),
            )
        })
        .collect();

    let evidence = if subjects.is_empty() {
        VerifiedEvidence::none(&store)
    } else {
        admit(table, &subjects, recognised, &store).verify(table, &store)?
    };

    // Phase three. Every option, attested or not: the composition is owed
    // to a declared certificate exactly as much as to a measured one.
    for (record, _) in records.iter_mut() {
        let Some(codec) = registry.by_label(&record.representation) else {
            continue;
        };
        let address = OperandAddress::new(
            &record.planned.operand.object,
            &record.planned.operand.tensor,
        );
        let tensor = record.planned.operand.tensor.clone();
        let label = record.representation.clone();
        for option in &mut record.extent.options {
            let extent = option.certificate.extent;
            let dependencies =
                match dependency_certificates(registry, codec, extent, &record.dependencies) {
                    Some(certificates) => certificates,
                    // A dependency that certifies nothing leaves the
                    // composition unavailable rather than optimistic.
                    None => {
                        option.certificate.radius = None;
                        continue;
                    }
                };
            option.certificate.radius = evidence
                .derive(
                    &(address.clone(), extent.depth),
                    option.certificate.radius.as_ref(),
                    &dependencies,
                    &tensor,
                    &label,
                )
                .unwrap_or(None);
        }
    }
    Ok(())
}

/// Each dependency's TERMINAL baseline identity at `extent`, keyed by the
/// name the owner's codec declared — what an attestation binds to, and
/// what a reader rebuilds to check that binding.
fn auxiliary_baselines(
    registry: &CodecRegistry,
    codec: &dyn RepresentationCodec,
    extent: RepresentationExtent,
    pins: &[DependencyPin],
) -> BTreeMap<String, String> {
    codec
        .required_auxiliaries(extent)
        .iter()
        .filter_map(|spec| {
            let pin = pins.iter().find(|d| d.name == spec.name)?;
            let identity = registry.by_label(&pin.label)?.identity();
            Some((
                spec.name.to_string(),
                terminal_baseline(&identity.family, identity.revision),
            ))
        })
        .collect()
}

/// The certificates of the dependency extents ACTUALLY selected, in the
/// order the owner's codec declares them.
///
/// `None` — meaning the composition is unavailable — as soon as any one
/// term is missing: a required auxiliary with no pin, a label no codec
/// claims, or a dependency whose own codec declares no radius. Adding up
/// the terms that happen to exist would state a bound narrower than the
/// truth, which is the exact failure this composition exists to prevent.
///
/// Dependencies are pinned WHOLE today, so the selected extent is the
/// dependency's terminal one. When auxiliary extent selection arrives —
/// [`AuxiliaryExtents`](super::operands::AuxiliaryExtents) is already the
/// shape it will take — this is the one place that has to learn about it.
fn dependency_certificates(
    registry: &CodecRegistry,
    codec: &dyn RepresentationCodec,
    extent: RepresentationExtent,
    pins: &[DependencyPin],
) -> Option<Vec<FidelityCertificate>> {
    codec
        .required_auxiliaries(extent)
        .iter()
        .map(|spec| {
            let pin = pins.iter().find(|d| d.name == spec.name)?;
            let dependency = registry.by_label(&pin.label)?;
            let terminal = dependency.extents().iter().map(|c| c.extent).max()?;
            dependency
                .certificate_at(terminal, &pin.tensor)
                .ok()?
                .radius
        })
        .collect()
}

/// Refuse a plan whose SELECTED extents do not satisfy `floor`.
///
/// The floor gates SELECTION, so it has to be asked about what was
/// selected — not only about the shallower extents a budget considers
/// moving to. Until this existed, `RepresentationFloor` was consulted in
/// exactly one place, `shallowest_saving`, which judges candidate MOVES
/// under preparation pressure. A caller could therefore declare
/// `CertifiedExact`, plan a Q4_K container under an unbounded budget,
/// and be refused nothing: the requirement was real, nothing ever asked
/// it, and the plan carried a quality claim it had never earned.
///
/// The asymmetry between the floors is the point, and it is the referent
/// that produces it:
///
/// * [`RepresentationFloor::TerminalExtent`] is STRUCTURAL and proceeds
///   when no composition is available. It asks how much of the artifact
///   is read, and every pin starts on the whole of it.
/// * [`RepresentationFloor::CertifiedExact`] and
///   [`RepresentationFloor::Within`] are EVIDENCE requirements and must
///   refuse. A certificate bounds decoded values against the typed
///   logical source tensor, so an operand carrying none has made no
///   claim about its source at all — and "no claim" is not "small".
///
/// Called once on the initial selection and again on whatever the budget
/// loop settles on, so a plan cannot reach a caller having quietly
/// re-selected past its own requirement.
pub(super) fn enforce_floor<'a>(
    records: impl IntoIterator<Item = &'a RealizationRecord>,
    floor: RepresentationFloor,
) -> Result<(), VindexError> {
    let mut refused = Vec::new();
    for record in records {
        // `selected_option` folds together the two ways a pin can offer
        // no certificate to judge: a label no codec in this build claims
        // (no declared extents at all), and — impossible today, because
        // the only two writers of `selected` take it from `options` — a
        // depth the codec does not declare. Both mean the same thing to a
        // floor, so they are one branch rather than two, and the second
        // is not given a refusal message it could never print.
        let Some(option) = record.extent.selected_option() else {
            // A structural floor has nothing to check here; an evidence
            // floor has nothing to rely on.
            if !matches!(floor, RepresentationFloor::TerminalExtent) {
                refused.push(format!(
                    "`{}` is stored as `{}`, which declares no extent this build can judge,                      so it certifies nothing",
                    record.planned.operand.tensor, record.representation,
                ));
            }
            continue;
        };
        let terminal = record
            .extent
            .options
            .iter()
            .map(|o| o.certificate.extent)
            .max()
            .unwrap_or(option.certificate.extent);
        if floor.admits(option, terminal) {
            continue;
        }
        refused.push(match &option.certificate.radius {
            Some(radius) => format!(
                "`{}` (`{}`, depth {}) composes to {:.3e} in {} over {}",
                record.planned.operand.tensor,
                record.representation,
                option.certificate.extent.depth,
                radius.radius(),
                radius.metric(),
                radius.domain(),
            ),
            None => format!(
                "`{}` (`{}`, depth {}) states no source-fidelity bound — its error is a \
                 property of the instance that was encoded, not of the scheme, so only a \
                 verified attestation can supply one",
                record.planned.operand.tensor,
                record.representation,
                option.certificate.extent.depth,
            ),
        });
    }
    if refused.is_empty() {
        return Ok(());
    }
    Err(VindexError::Parse(format!(
        "the plan cannot meet its fidelity requirement of {}: {} operand(s) do not satisfy it \
         — {}. A certificate bounds decoded values against the typed logical source tensor, \
         so an operand that declares none has stated nothing about its source. Attest the \
         instances, or ask for {} instead, which requires the complete stored representation \
         and makes no source-fidelity claim.",
        floor.describe(),
        refused.len(),
        refused.join("; "),
        RepresentationFloor::TerminalExtent.describe(),
    )))
}
