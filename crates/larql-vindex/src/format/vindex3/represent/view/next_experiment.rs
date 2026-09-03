//! **What to measure next — and why this record cannot say.**
//!
//! The one tool of the seven that does not answer, and it is a finding
//! rather than an omission.
//!
//! [`SearchSnapshot::next_experiment`] derives the whole chain from
//! stored facts, but it takes two arguments that are CODE and not data:
//! a [`LayoutAdmission`] and a [`Footprint`]. The first has production
//! implementations. The second has none — `Footprint`'s own contract is
//! *"supplied, never derived here"*, and the only implementations in the
//! tree are three copies of a test fixture.
//!
//! Nor can the facade write one. Pricing a decision the map protects
//! needs the source dtype, and [`TensorSurface`] carries object, tensor,
//! role and shape and no dtype at all. The three fixtures close that gap
//! by multiplying by two, which is bf16 asserted rather than read, and
//! promoting an assertion about a dtype into production is exactly the
//! move that makes a search price bytes it never saved.
//!
//! So this tool refuses, and names what is missing. Serving a candidate
//! ranking built on an invented price would be the facade deriving the
//! one quantity the whole objective is measured in.
//!
//! ```text
//! declared   SearchSemantics.physical_accounting = "logical-bytes/v1"
//! held       nothing that can evaluate it
//! ```
//!
//! [`LayoutAdmission`]: super::super::state::resolved::LayoutAdmission
//! [`Footprint`]: super::super::state::candidate::Footprint
//! [`TensorSurface`]: super::super::state::surface::TensorSurface
//! [`SearchSnapshot::next_experiment`]: super::super::state::snapshot::SearchSnapshot::next_experiment

use serde::Serialize;

use super::super::state::action_space::ActionVocabulary;
use super::super::state::snapshot::SearchSnapshot;
use super::current::ScaleGap;
use super::origin::{Origin, Rendered};

/// A fact the record would have to carry for the chain to run, and what
/// is wrong without it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Missing {
    pub fact: String,
    pub because: String,
}

/// The declared accounting rule, what evaluating it would need, and the
/// facts the record still answers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Refusal {
    /// The rule the snapshot DECLARES bytes are counted under. A
    /// version id, which is a name for a procedure and not the
    /// procedure.
    pub declared_accounting: String,
    pub missing: Vec<Missing>,
    /// Every move that exists at all. Still worth showing: R5-F6 was a
    /// vocabulary failure and cost two ~430 MB moves, and a reader can
    /// see the whole move set here without any of it being priced.
    pub vocabulary: ActionVocabulary,
    /// States with no reading, per scale — a fact query that needs no
    /// footprint, because these states are already in the graph and
    /// already priced.
    pub unmeasured: Vec<ScaleGap>,
}

/// **What to measure next.**
///
/// One variant, and it is a refusal. Written as an enum so that the
/// day a footprint oracle becomes a stored fact, the answer arrives as
/// a visible schema change rather than as a field quietly filling in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum NextExperiment {
    /// No candidate can be priced, so none can be enumerated, ranked or
    /// pruned.
    NoFootprintOracle(Refusal),
}

/// Why the missing facts are missing, stated once so the refusal reads
/// the same everywhere.
const MISSING_FOOTPRINT: (&str, &str) = (
    "a Footprint oracle the snapshot can name",
    "the generator prices every candidate through one, and SearchSemantics \
     carries the accounting rule's NAME rather than anything that evaluates it",
);

const MISSING_SOURCE_DTYPE: (&str, &str) = (
    "a source dtype on TensorSurface",
    "a decision the map protects is carried verbatim, and its bytes cannot \
     be priced from shape and role alone",
);

impl NextExperiment {
    pub fn of(snapshot: &SearchSnapshot) -> Self {
        Self::NoFootprintOracle(Refusal {
            declared_accounting: snapshot.semantics().physical_accounting.clone(),
            missing: [MISSING_FOOTPRINT, MISSING_SOURCE_DTYPE]
                .into_iter()
                .map(|(fact, because)| Missing {
                    fact: fact.to_string(),
                    because: because.to_string(),
                })
                .collect(),
            vocabulary: snapshot.space().vocabulary.clone(),
            unmeasured: ScaleGap::all(snapshot),
        })
    }
}

impl Rendered for NextExperiment {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new(
                "NoFootprintOracle.declared_accounting",
                "SearchSemantics.physical_accounting",
            ),
            Origin::new("NoFootprintOracle.missing[].fact", "this module's refusal"),
            Origin::new(
                "NoFootprintOracle.missing[].because",
                "this module's refusal",
            ),
            Origin::new("NoFootprintOracle.vocabulary", "SearchSpace.vocabulary"),
        ];
        origins.extend(
            ScaleGap::origins()
                .iter()
                .map(|o| o.under("NoFootprintOracle.unmeasured[]")),
        );
        origins
    }
}
