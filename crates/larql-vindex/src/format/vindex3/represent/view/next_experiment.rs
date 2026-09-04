//! **What to measure next.**
//!
//! Three answers, because an agent takes three different actions:
//!
//! ```text
//! Available     the deterministic optimiser had the factual authority
//!               to select the next unresolved experiment, and here it is
//! Exhausted     it had the authority, and every move is already
//!               observed or pruned
//! Unavailable   it could not answer, and this is what is missing
//! ```
//!
//! Collapsing the middle one into the last would tell an agent "nothing
//! to do" and "I cannot tell you" in the same words, and those call for
//! opposite next steps.
//!
//! **`Available` means the optimiser could SELECT the experiment.** Not
//! that it has been run, admitted or promoted — those remain entirely
//! separate, and nothing here touches them.
//!
//! # What this used to be
//!
//! One variant, and it was a refusal. `SearchSnapshot::next_experiment`
//! derived the whole chain from stored facts but took two arguments
//! that were CODE and not data: a layout rule and a pricing routine.
//! The first had production implementations; the second had none, and
//! the facade could not write one, because pricing a protected decision
//! needs the source bytes and `TensorSurface` carries shape and role
//! and no storage facts at all.
//!
//! The enum was written as an enum for exactly this moment, so that the
//! day the missing authority arrived the answer would show up as a
//! visible schema change rather than as a field quietly filling in.
//! 4b-a…e supplied it: the container's own segment table, sealed into
//! the semantic identity, read through that seal, bound to the surface,
//! and summed under the record's own declared policies.
//!
//! **The transport did not change.** `optimizer.next_experiment` still
//! calls one view method and serialises whatever it returns. That is
//! the point: MCP was complete when it refused.

use std::collections::BTreeSet;

use serde::Serialize;

use super::super::state::accounting::PhysicalAccountingSemanticsId;
use super::super::state::action_space::ActionVocabulary;
use super::super::state::identity::RepresentationStateId;
use super::super::state::key::MeasurementKey;
use super::super::state::search_policy::Selection;
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

/// **Where the prices came from**, rendered beside every answer that
/// used them.
///
/// A reader must be able to tell which authority an experiment was
/// selected under without re-deriving it — and, when the answer is a
/// refusal, which authority was absent.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Accounting {
    /// The procedure the record declares bytes are counted under.
    pub procedure: String,
    /// The layout policy that decided what could be compiled at all.
    pub layout_admission: String,
    /// What that procedure MEANS, digested. Present only when the
    /// record carries accounting facts.
    pub semantics: Option<PhysicalAccountingSemanticsId>,
    /// The semantic source identity the prices were read from.
    pub source: Option<String>,
    /// How many tensors the container prices.
    pub priced_tensors: Option<usize>,
    /// Every encoding the base map and the vocabulary can select, each
    /// of which had to be priceable before the search could start.
    pub selectable_encodings: Vec<String>,
}

/// The experiment the deterministic optimiser selected.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Available {
    /// The applied set this question was asked from.
    pub applied: BTreeSet<String>,
    /// The experiment. One key, one run.
    pub experiment: MeasurementKey,
    pub state: RepresentationStateId,
    /// The best physical prize among the routes to it. Negative means
    /// bytes are removed, which is the direction the objective wants.
    pub physical_delta: i64,
    /// How many routes reach this one experiment. More than one is
    /// normal: two realizations can reach one physical state.
    pub routes: usize,
    /// How many opportunities the policy ordered. `1` when there was
    /// nothing to rank.
    pub considered: usize,
    pub accounting: Accounting,
}

/// The optimiser could answer, and there is nothing left to measure.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Exhausted {
    /// The applied set this question was asked from.
    pub applied: BTreeSet<String>,
    pub accounting: Accounting,
    /// Every move that exists at all — so a reader can see that the
    /// emptiness is over a real vocabulary. R5-F6 was a vocabulary
    /// failure and cost two ~430 MB moves.
    pub vocabulary: ActionVocabulary,
}

/// The optimiser could not answer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Unavailable {
    /// The applied set this question was asked from.
    pub applied: BTreeSet<String>,
    /// A stable, machine-readable reason. Distinct reasons stay
    /// distinct: re-reading a container fixes one of these and cannot
    /// fix another.
    pub reason: String,
    /// The substrate's own message, unedited.
    pub detail: String,
    pub missing: Vec<Missing>,
    pub accounting: Accounting,
    /// Still worth showing without a price: the whole move set, and the
    /// states already in the graph that carry no reading.
    pub vocabulary: ActionVocabulary,
    pub unmeasured: Vec<ScaleGap>,
}

/// **What to measure next.**
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum NextExperiment {
    Available(Available),
    Exhausted(Exhausted),
    Unavailable(Unavailable),
}

/// The reason a record with no accounting facts at all cannot answer.
const NO_ACCOUNTING_AUTHORITY: &str = "no-accounting-authority";
/// The reason a record whose facts do not price its surface cannot.
const ACCOUNTING_UNUSABLE: &str = "accounting-unusable";

const MISSING_ACCOUNTING: (&str, &str) = (
    "physical accounting facts the record can name",
    "the generator prices every candidate, and a candidate that is neither \
     eligible nor pruned would break the census conservation law",
);

impl NextExperiment {
    /// Derive the answer from the record and nothing else.
    ///
    /// No arguments, deliberately. The question — which applied set,
    /// which corpus, which scale, which instrument — is stored, so the
    /// answer is a property of the record rather than of whatever the
    /// caller passed in, and the transport supplies nothing at all.
    pub fn of(snapshot: &SearchSnapshot) -> Self {
        let accounting = Accounting::of(snapshot);
        let unavailable = |reason: &str, detail: String, missing: Vec<Missing>| {
            Self::Unavailable(Unavailable {
                applied: snapshot.applied().clone(),
                reason: reason.to_string(),
                detail,
                missing,
                accounting: accounting.clone(),
                vocabulary: snapshot.space().vocabulary.clone(),
                unmeasured: ScaleGap::all(snapshot),
            })
        };
        if snapshot.facts().accounting.is_none() {
            return unavailable(
                NO_ACCOUNTING_AUTHORITY,
                "this record carries no physical accounting authority".into(),
                vec![Missing {
                    fact: MISSING_ACCOUNTING.0.to_string(),
                    because: MISSING_ACCOUNTING.1.to_string(),
                }],
            );
        }
        match snapshot.next_experiment() {
            Err(e) => unavailable(ACCOUNTING_UNUSABLE, e.to_string(), Vec::new()),
            Ok(Selection::Exhausted) => Self::Exhausted(Exhausted {
                applied: snapshot.applied().clone(),
                accounting,
                vocabulary: snapshot.space().vocabulary.clone(),
            }),
            Ok(selection) => {
                let considered = match &selection {
                    Selection::Ranked { considered, .. } => *considered,
                    _ => 1,
                };
                let chosen = selection
                    .opportunity()
                    .expect("a non-exhausted selection holds an opportunity");
                Self::Available(Available {
                    applied: snapshot.applied().clone(),
                    experiment: chosen.key.clone(),
                    state: chosen.state.clone(),
                    physical_delta: chosen.physical_delta(),
                    routes: chosen.routes(),
                    considered,
                    accounting,
                })
            }
        }
    }
}

impl Accounting {
    fn of(snapshot: &SearchSnapshot) -> Self {
        let facts = snapshot.facts().accounting.as_ref();
        Self {
            procedure: snapshot.semantics().physical_accounting.clone(),
            layout_admission: snapshot.semantics().layout_admission.clone(),
            semantics: facts.map(|f| f.semantics().clone()),
            source: facts.map(|f| f.source_digest().to_string()),
            priced_tensors: facts.map(|f| f.len()),
            selectable_encodings: snapshot.selectable_encodings(),
        }
    }

    fn origins(under: &str) -> Vec<Origin> {
        [
            ("procedure", "SearchSemantics.physical_accounting"),
            ("layout_admission", "SearchSemantics.layout_admission"),
            ("semantics", "PhysicalAccountingFacts::semantics"),
            ("source", "PhysicalAccountingFacts::source_digest"),
            ("priced_tensors", "PhysicalAccountingFacts::len"),
            (
                "selectable_encodings",
                "SearchSnapshot::selectable_encodings",
            ),
        ]
        .into_iter()
        .map(|(field, call)| Origin::new(format!("{under}.accounting.{field}"), call))
        .collect()
    }
}

impl Rendered for NextExperiment {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new(
                "Available.experiment",
                "MeasurementOpportunity.key, via SearchSnapshot::next_experiment",
            ),
            Origin::new("Available.state", "MeasurementOpportunity.state"),
            Origin::new(
                "Available.physical_delta",
                "MeasurementOpportunity::physical_delta",
            ),
            Origin::new("Available.routes", "MeasurementOpportunity::routes"),
            Origin::new("Available.considered", "Selection::Ranked.considered"),
            Origin::new("Exhausted.vocabulary", "SearchSpace.vocabulary"),
            Origin::new("Unavailable.reason", "this module's refusal"),
            Origin::new(
                "Unavailable.detail",
                "the substrate error from SearchSnapshot::next_experiment",
            ),
            Origin::new("Unavailable.missing[].fact", "this module's refusal"),
            Origin::new("Unavailable.missing[].because", "this module's refusal"),
            Origin::new("Unavailable.vocabulary", "SearchSpace.vocabulary"),
            Origin::new("Available.applied", "SearchSpace.applied"),
            Origin::new("Exhausted.applied", "SearchSpace.applied"),
            Origin::new("Unavailable.applied", "SearchSpace.applied"),
        ];
        for under in ["Available", "Exhausted", "Unavailable"] {
            origins.extend(Accounting::origins(under));
        }
        origins.extend(
            ScaleGap::origins()
                .iter()
                .map(|o| o.under("Unavailable.unmeasured[]")),
        );
        origins
    }
}
