//! Why a realization is refused.

use super::super::cpu::physical::PhysicalProjectionPlan;
use super::super::lowering::{LoweringIdentity, LoweringRegistry};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{
    CodecRegistry, ExtentCertificate, RepresentationExtent, RequiredAccess,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;
use serde::{Deserialize, Serialize};
use std::fmt;

#[allow(unused_imports)]
use super::*;

/// Why no realization could be selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalKind {
    /// The stored label names no registered codec, so nothing can decode
    /// it and no capability is declared for it.
    UnregisteredRepresentation,
    /// Every candidate needs access the stored representation does not
    /// provide.
    AccessRefused,
    /// The plan executes an operation no realization on this backend can
    /// bind — a planned operand with nowhere to go.
    MissingRealization,
}

impl RefusalKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::UnregisteredRepresentation => "unregistered representation",
            Self::AccessRefused => "access refused",
            Self::MissingRealization => "missing realization",
        }
    }
}

/// A refusal that names the operand, what it asked for, what the
/// representation is, and every candidate with the reason it was rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRefusal {
    pub operand: OperandRef,
    pub operation: Operation,
    pub representation: String,
    pub requested: RequiredAccess,
    pub kind: RefusalKind,
    pub considered: Vec<(RealizationId, String)>,
}

impl fmt::Display for SelectionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operand `{}` ({}, requires {} access) stored as `{}`: {}",
            self.operand.tensor,
            self.operation.name(),
            self.requested.name(),
            self.representation,
            self.kind.name()
        )?;
        if self.considered.is_empty() {
            write!(f, "; no realization to consider")?;
        }
        for (candidate, why) in &self.considered {
            write!(f, "; {} — {why}", candidate.name())?;
        }
        Ok(())
    }
}

/// Every refusal a plan raised, together, so a caller sees the whole
/// problem and not its first symptom.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRefusals(pub Vec<SelectionRefusal>);

impl fmt::Display for SelectionRefusals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} planned operand(s) have no admissible realization",
            self.0.len()
        )?;
        for refusal in &self.0 {
            write!(f, "\n  {refusal}")?;
        }
        Ok(())
    }
}

/// One planned operand's pinned realization — the record the prepared
/// plan keeps and the trace reads.
#[derive(Debug, Clone, PartialEq)]
pub struct RealizationRecord {
    pub planned: PlannedOperand,
    pub representation: String,
    /// The identity the stored label resolved to at preparation; `None`
    /// for a label no codec claims.
    ///
    /// The CODEC plane's authority: what the bytes are.
    pub codec_provider: Option<CodecIdentity>,
    /// The lowering provider that qualified this realization — the
    /// LOWERING plane's authority: what implementation decided the pin
    /// (LOWERING-PLUGIN-1, L4).
    ///
    /// Not optional, and not derived from the other: a record exists
    /// because some provider selected it, and since L1 every provider
    /// states an identity. The two authorities move independently — a
    /// codec revision can change under an unchanged lowering and the
    /// reverse — so the pin names both and invalidation says which.
    ///
    /// The provider's SEMANTIC identity, never its configuration: two
    /// device providers built with different format tables share
    /// `device-matmul/v1`, and what their configurations changed is
    /// pinned in [`Selection::realization`] instead.
    pub lowering_provider: LoweringIdentity,
    pub selection: Selection,
    /// How much of the stored representation this pin reads.
    ///
    /// On the PIN, not on the plan: a depth is a fact about one codec, and
    /// the plan is representation-independent. The artifact still holds
    /// every extent whatever this says — what the pin decides is how much
    /// of it execution opens.
    pub extent: ExtentPin,
    /// The other represented objects this pin will resolve, and what its
    /// realization does with each. Empty for every operand whose codec
    /// depends on nothing.
    /// Bytes physically read to verify this operand's attestations
    /// during selection. Zero for an operand that attests nothing, and
    /// zero for one whose claim was refused from metadata — admission
    /// costs no payload read, which is a property worth being able to
    /// observe rather than assert.
    pub verified_bytes: u64,
    pub dependencies: Vec<DependencyPin>,
}

impl RealizationRecord {
    /// Pin another realization on this record, and let every dependency's
    /// lifetime follow it: the lifetime is the realization's, so a re-pin
    /// that left it standing would price the old realization's retention
    /// against the new one's bytes.
    pub fn repin(&mut self, realization: RealizationId) {
        self.selection.realization = realization;
        let lifetime = if realization.retains_dependencies() {
            DependencyLifetime::Retained
        } else {
            DependencyLifetime::PreparationOnly
        };
        for dependency in &mut self.dependencies {
            dependency.lifetime = lifetime;
        }
    }
}

/// The authorities a prepared image was pinned under, both planes, as a
/// value that survives being written down.
///
/// Two independent authorities. The CODEC plane says what the stored
/// bytes are, keyed by the representation label each pin read; the
/// LOWERING plane says which implementation qualified the realization.
/// Neither implies the other, so an image carries both and a refusal
/// names which one moved.
///
/// Separated from [`PreparedOperands`](super::super::prepared::PreparedOperands)
/// so that the SAME code judges a live image and one that was serialized
/// and reloaded: an image is judged through the authorities it yields,
/// and a reloaded record of those authorities yields the same two
/// verdicts. Nothing here holds a codec or a provider — only what was
/// decided — because an image that held its providers would keep a
/// removed one alive and could never be invalidated by its
/// disappearance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinnedAuthorities {
    /// Every representation the image pinned, once each, with the
    /// identity its label resolved to at preparation. `None` is an
    /// overlay edit's f32-space fact, never a label a loader judged for
    /// itself.
    pub codecs: Vec<(String, Option<CodecIdentity>)>,
    /// Every lowering provider that qualified a pin, once each, in pin
    /// order. One today — a prepared image is lowered by one provider —
    /// and a list because nothing in the contract says it must stay one.
    pub lowerings: Vec<LoweringIdentity>,
}

impl PinnedAuthorities {
    /// Refuse an image whose codec authority has moved: a representation
    /// that resolves to a different identity than it did at preparation,
    /// or to none at all.
    pub fn ensure_codecs_in(&self, registry: &CodecRegistry) -> Result<(), VindexError> {
        let describe = |identity: &Option<CodecIdentity>| {
            identity
                .as_ref()
                .map(|i| format!("{} r{}", i.family, i.revision))
                .unwrap_or_else(|| "no registered codec".to_string())
        };
        for (label, prepared) in &self.codecs {
            let now = registry.by_label(label).map(|c| c.identity());
            if now != *prepared {
                return Err(VindexError::Parse(format!(
                    "representation `{label}` was prepared against {} and the registry now \
                     offers {}; re-prepare rather than execute a pin whose provider changed",
                    describe(prepared),
                    describe(&now)
                )));
            }
        }
        Ok(())
    }

    /// Refuse an image whose lowering authority has moved: the provider
    /// that qualified a pin is not in `registry` under exactly the
    /// identity the pin recorded.
    ///
    /// A family present at another revision is a different provider and
    /// is refused; so is a registry full of perfectly good alternatives.
    /// Nothing re-selects and nothing substitutes — the refusal names the
    /// identity the pin recorded and every identity the registry holds,
    /// and re-preparation is the only way forward.
    pub fn ensure_lowerings_in(&self, registry: &LoweringRegistry) -> Result<(), VindexError> {
        lowerings_stand_in(&self.lowerings, registry)
    }
}

/// The lowering plane's judgment itself — see
/// [`PinnedAuthorities::ensure_lowerings_in`], which is this over what was
/// written down.
///
/// A free function so that a prepared image can ask the question without
/// first building its codec half, and so that the live image and the
/// reloaded record cannot drift into two judgments of one contract.
pub(in super::super) fn lowerings_stand_in(
    pinned: &[LoweringIdentity],
    registry: &LoweringRegistry,
) -> Result<(), VindexError> {
    for identity in pinned {
        if registry.provider(identity).is_err() {
            let held = registry.identities();
            let held = if held.is_empty() {
                "none".to_string()
            } else {
                held.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(VindexError::Parse(format!(
                "lowering provider `{identity}` qualified this preparation and the registry \
                 now holds {held}; re-prepare rather than execute a pin whose provider is \
                 gone — no other provider stands in for it"
            )));
        }
    }
    Ok(())
}

/// What a realization does with a dependency once it has read it.
///
/// The distinction the accounting turns on, and it belongs to the
/// REALIZATION rather than to the operand: being an auxiliary says
/// nothing about lifetime. A canonical decode reads a codebook, produces
/// an f32 image and is finished with it; a direct kernel over codes would
/// have to keep it and touch it for every token. Same object, same
/// container, different cost — decided by what was pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyLifetime {
    /// Read to prepare the owner, then dropped. Nothing of it is resident
    /// afterwards and no token touches it.
    PreparationOnly,
    /// Kept resident and read while serving. NOTHING SELECTS THIS TODAY:
    /// no realization in this build declares it, and the accounting can
    /// price it so that a realization which did would be paid for
    /// honestly rather than silently.
    Retained,
}

/// One dependency a pinned realization will resolve, and what it will
/// cost once resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct DependencyPin {
    /// The name the owner's codec declared.
    pub name: String,
    pub object: String,
    pub tensor: String,
    /// The stored label the container records for the target — its
    /// representation, which is its own business and not its owner's.
    pub label: String,
    /// The identity that label resolved to when this pin was made.
    /// `None` for a label no codec claims, which admission refuses.
    pub provider: Option<CodecIdentity>,
    /// The container's recorded length for it — `None` when the container
    /// holds no such tensor, which admission refuses before this matters.
    pub stored_bytes: Option<u64>,
    /// Logical elements it holds, for pricing a retained image.
    pub elements: usize,
    pub lifetime: DependencyLifetime,
}

impl DependencyPin {
    /// The address, as the ledger keys deduplication on.
    pub fn address(&self) -> (String, String) {
        (self.object.clone(), self.tensor.clone())
    }
}

/// One extent a pin could take: what it certifies, and what it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtentOption {
    pub certificate: ExtentCertificate,
    /// Bytes of the stored operand this extent reads, where the codec
    /// prices a shape. `None` for an instance-sized encoding, whose
    /// authority is the container's recorded length.
    pub stored_bytes: Option<u64>,
}

/// The extent a pin selected, and every extent it could have taken.
///
/// Three things the vocabulary keeps apart: what the ARTIFACT contains
/// (every option here, because the container holds every plane), what
/// EXECUTION requires (a fidelity floor, which the budget carries), and
/// what the PIN chose (`selected`). A shallower selection does not shrink
/// the artifact and must never be accounted as if it had.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtentPin {
    pub selected: RepresentationExtent,
    pub options: Vec<ExtentOption>,
}

impl ExtentPin {
    /// A pin on the whole representation: the deepest extent declared.
    /// The default, so nothing changes for a plan that asks for nothing.
    pub fn whole(options: Vec<ExtentOption>) -> Self {
        let selected = options
            .iter()
            .map(|o| o.certificate.extent)
            .max()
            .unwrap_or(RepresentationExtent::BASE);
        Self { selected, options }
    }

    /// A pin for a representation whose extents are not known — an
    /// unregistered label, whose bytes nothing can price either.
    pub fn unknown() -> Self {
        Self {
            selected: RepresentationExtent::BASE,
            options: Vec::new(),
        }
    }

    /// The option the pin selected, when the extents are known.
    pub fn selected_option(&self) -> Option<&ExtentOption> {
        self.options
            .iter()
            .find(|o| o.certificate.extent == self.selected)
    }

    /// Bytes the selected extent reads, when the codec prices them.
    pub fn touch_bytes(&self) -> Option<u64> {
        self.selected_option().and_then(|o| o.stored_bytes)
    }

    /// Whether this pin has anything to choose between.
    pub fn is_progressive(&self) -> bool {
        self.options.len() > 1
    }
}

// ── The candidate sets, derived from declarations ─────────────────────

/// The candidates the CPU executor has for a projection-class operand:
/// every direct realization the codec declares, the universal decode
/// (`decode_plan` is the f32 projection that follows it), and — for a
/// source whose stored bytes the executor knows how to re-quantise, which
/// it declares by naming a direct bf16 kernel — the executor's own compact
/// forms. Nothing is added by label.
pub fn cpu_projection_candidates(
    facts: &RepresentationFacts,
    decode_plan: PhysicalProjectionPlan,
    requantise: &[PhysicalProjectionPlan],
) -> Vec<RealizationId> {
    let mut out: Vec<RealizationId> = facts
        .direct_cpu_plans()
        .into_iter()
        .map(|p| RealizationId::cpu(RealizationForm::Direct(p)))
        .collect();
    if facts.registered.is_some() {
        out.push(RealizationId::cpu(RealizationForm::Decode(decode_plan)));
        // A requantised image is manufactured at load, so it is no
        // candidate at all when the store forbids that: offering it would
        // let a later policy (the residency budget) pick what the loader
        // must then refuse.
        if !facts.requantise_forbidden
            && facts
                .direct_cpu_plans()
                .contains(&PhysicalProjectionPlan::FusedBf16)
        {
            out.extend(
                requantise
                    .iter()
                    .map(|p| RealizationId::cpu(RealizationForm::Requantise(*p))),
            );
        }
    }
    out
}
