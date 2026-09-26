//! **AUTO-REP-1a: proposed precision maps, by search.** SEARCH only.
//!
//! Reads a surface, a base map and a price table, and proposes the K
//! cheapest distinct representation states the base map's grammar can
//! write. A proposal is **not authority**: it is a prediction that some
//! compiled candidate is worth a joint measurement, and only that
//! measurement and promotion can admit it (`docs/auto-rep-1.md`).
//!
//! ```text
//! surface + base map ──group──▶ variables  (projection, layer)
//!            footprint ──price─▶ Compile / Source cost per variable
//!   pins, cuts, ceiling ──────▶ unary filtering, bound
//!                  solver ────▶ K cheapest leaves, distinct states
//!               synthesis ────▶ Protections → PrecisionMap → state
//! ```
//!
//! There is no quality term. SENSITIVITY-1 falsified every cheap
//! per-component score tried, so an [`OrderingPrior`] may break ties
//! between equal-byte states and nothing more, and none ships here.

pub(crate) mod solver;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::super::compile::hash_bytes;
use super::super::compiler::SourceIdentity;
use super::super::map::PrecisionMap;
use super::super::policy::{layer_of, projection_of, Protections};
use super::super::search_evidence::SearchEvidence;
use super::footprint::SurfaceFootprint;
use super::identity::{RepresentationState, RepresentationStateId};
use super::realization::LogicalBytes;
use super::resolved::LayoutAdmission;
use super::surface::{TensorSurface, FIELD, RECORD};
use crate::error::VindexError;

/// Which solver produced a proposal. Moves with any change to what the
/// solver returns for a given problem.
pub const SOLVER_REVISION: &str = "auto-rep-bnb/v1";

/// The canonical form [`ProposalProblem::id`] digests.
pub const PROBLEM_ID_VERSION: &str = "auto-rep-problem/v1";

/// One variable: every eligible tensor one exception with a single
/// projection and a single layer addresses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GroupKey {
    pub projection: String,
    pub layer: u32,
}

impl GroupKey {
    pub fn new(projection: impl Into<String>, layer: u32) -> Self {
        Self {
            projection: projection.into(),
            layer,
        }
    }

    fn describe(&self) -> String {
        format!("{}@{}", self.projection, self.layer)
    }
}

/// Every group the base map's eligible roles present on `surface`, in
/// canonical order: the variables a [`ProposalProblem`] over the same
/// inputs has, and the edits a validate loop's vocabulary must name.
pub fn group_keys(surface: &TensorSurface, base: &PrecisionMap) -> Vec<GroupKey> {
    let keys: BTreeSet<GroupKey> = surface
        .entries()
        .iter()
        .filter(|t| base.roles.iter().any(|r| r == t.role.name()))
        .filter_map(|t| {
            Some(GroupKey::new(
                projection_of(&t.tensor)?,
                layer_of(&t.tensor)?,
            ))
        })
        .collect();
    keys.into_iter().collect()
}

/// A variable's domain in 1a: what `Protections` can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GroupChoice {
    /// The base map's default encoding.
    Compile,
    /// Held at source precision.
    Source,
}

impl GroupChoice {
    const ALL: [GroupChoice; 2] = [GroupChoice::Compile, GroupChoice::Source];

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            GroupChoice::Compile => "compile",
            GroupChoice::Source => "source",
        }
    }
}

/// Search knowledge that excludes assignments. Never quality authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cut {
    /// A choice that cannot execute or cannot mean anything, known
    /// without measuring.
    Structural {
        group: GroupKey,
        choice: GroupChoice,
        reason: String,
    },
    /// Exactly the state a failed joint measurement measured. Excludes
    /// every map that resolves to it and nothing else; it is never
    /// widened into a claim about which choice caused the failure.
    ExactNoGood { state: RepresentationStateId },
}

impl Cut {
    /// The name a record lists this cut under.
    pub fn id(&self) -> String {
        match self {
            Cut::Structural {
                group,
                choice,
                reason,
            } => format!("structural:{}:{}:{reason}", group.describe(), choice.name()),
            Cut::ExactNoGood { state } => format!("no-good:{}", state.as_str()),
        }
    }
}

/// An ordering among equal-byte states. Declares its evidence class,
/// and is refused when that class is [`SearchEvidence::Unusable`].
pub trait OrderingPrior {
    fn evidence(&self) -> SearchEvidence;
    /// What this prior is, for the record.
    fn identity(&self) -> String;
    /// Lower is preferred. Summed over groups; compared only between
    /// states of equal bytes.
    fn score(&self, group: &GroupKey, choice: GroupChoice) -> f64;
}

/// Everything a problem is built from.
pub struct ProposalInputs<'a> {
    pub model: &'a SourceIdentity,
    pub surface: &'a TensorSurface,
    /// Default encoding and eligible roles. Must carry no exceptions.
    pub base: &'a PrecisionMap,
    pub layout: &'a dyn LayoutAdmission,
    /// The name `layout` was resolved from, for the problem identity.
    pub layout_id: &'a str,
    pub footprint: &'a SurfaceFootprint,
    pub ceiling: Option<LogicalBytes>,
    pub pins: BTreeMap<GroupKey, GroupChoice>,
    pub cuts: Vec<Cut>,
    pub prior: Option<&'a dyn OrderingPrior>,
}

#[derive(Debug, Clone)]
struct Group {
    key: GroupKey,
    /// Cost per choice, indexed by [`GroupChoice::index`].
    cost: [u64; 2],
}

/// A problem ready to solve: variables, their priced and filtered
/// domains, and its identity.
pub struct ProposalProblem<'a> {
    inputs: ProposalInputs<'a>,
    groups: Vec<Group>,
    fixed: Vec<(String, String)>,
    base_bytes: u64,
    structural: Vec<Cut>,
    no_goods: BTreeSet<RepresentationStateId>,
    id: String,
}

fn refused(message: impl Into<String>) -> VindexError {
    VindexError::Parse(format!("AUTO-REP proposal problem: {}", message.into()))
}

impl<'a> ProposalProblem<'a> {
    /// Group, price and identify. Refuses a base map with exceptions, an
    /// `Unusable` prior, a price table for another surface, and pins or
    /// cuts naming groups this problem does not have.
    pub fn new(inputs: ProposalInputs<'a>) -> Result<Self, VindexError> {
        if !inputs.base.exceptions.is_empty() {
            return Err(refused(format!(
                "base map `{}` already has exceptions; the solver owns every exception",
                inputs.base.name
            )));
        }
        if let Some(prior) = inputs.prior {
            if prior.evidence() == SearchEvidence::Unusable {
                return Err(refused(format!(
                    "prior `{}` declares Unusable evidence",
                    prior.identity()
                )));
            }
        }
        if inputs.footprint.surface_identity() != inputs.surface.identity() {
            return Err(refused("the price table prices another surface"));
        }
        let default = inputs.base.encoding.as_str();
        let price = |object: &str, tensor: &str, encoding: Option<&str>| {
            inputs
                .footprint
                .presented(object, tensor, encoding)
                .map(LogicalBytes::get)
                .ok_or_else(|| refused(format!("`{object}`/`{tensor}` has no price")))
        };
        let mut groups: BTreeMap<GroupKey, ([u64; 2], bool)> = BTreeMap::new();
        let mut fixed = Vec::new();
        let mut base_bytes = 0u64;
        for t in inputs.surface.entries() {
            if !inputs.base.roles.iter().any(|r| r == t.role.name()) {
                base_bytes += price(&t.object, &t.tensor, None)?;
                continue;
            }
            match (projection_of(&t.tensor), layer_of(&t.tensor)) {
                (Some(projection), Some(layer)) => {
                    let entry = groups
                        .entry(GroupKey::new(projection, layer))
                        .or_insert(([0, 0], false));
                    entry.0[GroupChoice::Compile.index()] +=
                        price(&t.object, &t.tensor, Some(default))?;
                    entry.0[GroupChoice::Source.index()] += price(&t.object, &t.tensor, None)?;
                    entry.1 |= inputs.footprint.admits(&t.object, &t.tensor, default);
                }
                // No exception addresses it without addressing others,
                // so it takes the default and is reported.
                _ => {
                    base_bytes += price(&t.object, &t.tensor, Some(default))?;
                    fixed.push((t.object.clone(), t.tensor.clone()));
                }
            }
        }
        let mut structural = Vec::new();
        for (key, (_, admitted)) in &groups {
            if !admitted {
                structural.push(Cut::Structural {
                    group: key.clone(),
                    choice: GroupChoice::Compile,
                    reason: format!("layout refuses {default} for every tensor"),
                });
            }
        }
        let mut no_goods = BTreeSet::new();
        for cut in &inputs.cuts {
            match cut {
                Cut::Structural { group, .. } => {
                    if !groups.contains_key(group) {
                        return Err(refused(format!(
                            "cut names `{}`, which this problem does not have",
                            group.describe()
                        )));
                    }
                    structural.push(cut.clone());
                }
                Cut::ExactNoGood { state } => {
                    no_goods.insert(state.clone());
                }
            }
        }
        for group in inputs.pins.keys() {
            if !groups.contains_key(group) {
                return Err(refused(format!(
                    "pin names `{}`, which this problem does not have",
                    group.describe()
                )));
            }
        }
        let groups: Vec<Group> = groups
            .into_iter()
            .map(|(key, (cost, _))| Group { key, cost })
            .collect();
        let mut problem = Self {
            inputs,
            groups,
            fixed,
            base_bytes,
            structural,
            no_goods,
            id: String::new(),
        };
        problem.id = problem.canonical_id()?;
        Ok(problem)
    }

    fn canonical_id(&self) -> Result<String, VindexError> {
        let base = serde_json::to_string(self.inputs.base)
            .map_err(|e| refused(format!("base map does not serialise: {e}")))?;
        let mut records = vec![
            PROBLEM_ID_VERSION.to_string(),
            self.inputs.model.semantic_digest(),
            self.inputs.surface.identity(),
            base,
            self.inputs.layout_id.to_string(),
            self.base_bytes.to_string(),
            self.inputs
                .ceiling
                .map_or_else(|| "-".into(), |c| c.get().to_string()),
            self.inputs
                .prior
                .map_or_else(|| "-".into(), |p| p.identity()),
        ];
        records.extend(self.groups.iter().map(|g| {
            format!(
                "{}{FIELD}{}{FIELD}{}",
                g.key.describe(),
                g.cost[0],
                g.cost[1]
            )
        }));
        records.extend(
            self.inputs
                .pins
                .iter()
                .map(|(g, c)| format!("pin{FIELD}{}{FIELD}{}", g.describe(), c.name())),
        );
        records.extend(self.cut_ids());
        Ok(hash_bytes(records.join(&RECORD.to_string()).as_bytes()))
    }

    /// What this problem IS: every input that can move a proposal.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The variables, in canonical order.
    pub fn groups(&self) -> impl Iterator<Item = &GroupKey> {
        self.groups.iter().map(|g| &g.key)
    }

    /// Eligible tensors no single-group exception can address; they take
    /// the default encoding.
    pub fn fixed(&self) -> &[(String, String)] {
        &self.fixed
    }

    /// Every cut in force, derived and supplied, sorted.
    pub fn cut_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.structural.iter().map(Cut::id).collect();
        ids.extend(
            self.no_goods
                .iter()
                .map(|s| Cut::ExactNoGood { state: s.clone() }.id()),
        );
        ids.sort();
        ids.dedup();
        ids
    }

    fn excluded(&self, group: &GroupKey, choice: GroupChoice) -> bool {
        self.inputs.pins.get(group).is_some_and(|&pin| pin != choice)
            || self.structural.iter().any(|c| {
                matches!(c, Cut::Structural { group: g, choice: x, .. } if g == group && *x == choice)
            })
    }

    fn core(&self) -> solver::Core {
        let vars = self
            .groups
            .iter()
            .map(|g| solver::Var {
                options: GroupChoice::ALL
                    .into_iter()
                    .filter(|&c| !self.excluded(&g.key, c))
                    .map(|c| solver::Opt {
                        choice: c.index(),
                        cost: g.cost[c.index()],
                        score: self.inputs.prior.map_or(0.0, |p| p.score(&g.key, c)),
                    })
                    .collect(),
            })
            .collect();
        solver::Core {
            base: self.base_bytes,
            vars,
            ceiling: self.inputs.ceiling.map(LogicalBytes::get),
        }
    }

    /// The groups an assignment holds at source, in canonical order.
    fn protected(&self, assignment: &[usize]) -> Vec<GroupKey> {
        self.groups
            .iter()
            .zip(assignment)
            .filter(|(_, &c)| c == GroupChoice::Source.index())
            .map(|(g, _)| g.key.clone())
            .collect()
    }

    /// The map a set of protected groups is written as, checked against
    /// this surface. A map the check refuses is a synthesis bug.
    fn synthesize(
        &self,
        protected: &[GroupKey],
        name: String,
    ) -> Result<(PrecisionMap, Protections), VindexError> {
        let protections = protections_for(protected);
        let map = PrecisionMap {
            name,
            exceptions: protections.as_exceptions(),
            ..self.inputs.base.clone()
        };
        map.check_against(
            self.inputs
                .surface
                .entries()
                .iter()
                .map(|t| (t.role, t.tensor.as_str())),
        )
        .map_err(|refusal| refused(format!("synthesised a map the check refuses: {refusal}")))?;
        Ok((map, protections))
    }

    fn state_of(&self, map: &PrecisionMap) -> RepresentationState {
        RepresentationState::resolve(
            self.inputs.model,
            self.inputs.surface,
            map,
            self.inputs.layout,
        )
    }
}

/// One `projection_in` rule per maximal run of consecutive protected
/// layers of one projection. `protected` is in canonical order.
fn protections_for(protected: &[GroupKey]) -> Protections {
    let mut protections = Protections::default();
    let mut run: Option<(&str, u32, u32)> = None;
    for g in protected {
        run = match run {
            Some((p, lo, hi)) if p == g.projection && g.layer == hi + 1 => Some((p, lo, g.layer)),
            Some((p, lo, hi)) => {
                protections = protections.projection_in(p, lo, hi);
                Some((&g.projection, g.layer, g.layer))
            }
            None => Some((&g.projection, g.layer, g.layer)),
        };
    }
    if let Some((p, lo, hi)) = run {
        protections = protections.projection_in(p, lo, hi);
    }
    protections
}

/// Whether the search ran to completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchOutcome {
    Complete,
    /// Stopped at the node budget, the deterministic stand-in for a
    /// timeout.
    NodeLimit {
        explored: u64,
    },
}

/// The prior a proposal was ordered by, if any.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriorRecord {
    pub identity: String,
    pub evidence: SearchEvidence,
}

/// How a proposal came to be. Enough to name the search in a lockfile
/// and to rerun it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalRecord {
    pub solver_revision: String,
    pub problem_id: String,
    pub cuts: Vec<String>,
    pub prior: Option<PriorRecord>,
    pub outcome: SearchOutcome,
    /// No proposal of this problem presents fewer bytes.
    pub lower_bound: Option<u64>,
    /// 1-based.
    pub rank: usize,
    pub bytes: u64,
    pub state: RepresentationStateId,
    pub protected: Vec<GroupKey>,
}

/// One proposed map. A prediction to measure, not an admission.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub map: PrecisionMap,
    /// The same decisions as a compilation request.
    pub protections: Protections,
    pub state: RepresentationState,
    pub record: ProposalRecord,
}

/// The K proposals of one search.
#[derive(Debug, Clone)]
pub struct Proposals {
    pub proposals: Vec<Proposal>,
    pub outcome: SearchOutcome,
    pub lower_bound: Option<LogicalBytes>,
}

/// Anything that turns a problem into ranked proposals, so solvers can be
/// compared on one problem and one record schema.
pub trait Proposer {
    fn revision(&self) -> &'static str;
    fn propose(&self, problem: &ProposalProblem<'_>, k: usize) -> Result<Proposals, VindexError>;
}

/// The native branch-and-bound.
#[derive(Debug, Clone, Copy)]
pub struct BranchAndBound {
    pub node_limit: u64,
}

impl Proposer for BranchAndBound {
    fn revision(&self) -> &'static str {
        SOLVER_REVISION
    }

    fn propose(&self, problem: &ProposalProblem<'_>, k: usize) -> Result<Proposals, VindexError> {
        let mut failure: Option<VindexError> = None;
        let mut accept = |assignment: &[usize]| -> Option<String> {
            if failure.is_some() {
                return None;
            }
            let protected = problem.protected(assignment);
            match problem.synthesize(&protected, "candidate".into()) {
                Ok((map, _)) => {
                    let state = problem.state_of(&map);
                    (!problem.no_goods.contains(state.id()))
                        .then(|| state.id().as_str().to_string())
                }
                Err(e) => {
                    failure = Some(e);
                    None
                }
            }
        };
        let solved = solver::solve(&problem.core(), k, self.node_limit, &mut accept);
        if let Some(e) = failure {
            return Err(e);
        }
        let outcome = match solved.outcome {
            solver::Outcome::Complete => SearchOutcome::Complete,
            solver::Outcome::NodeLimit { explored } => SearchOutcome::NodeLimit { explored },
        };
        let cuts = problem.cut_ids();
        let prior = problem.inputs.prior.map(|p| PriorRecord {
            identity: p.identity(),
            evidence: p.evidence(),
        });
        let short = &problem.id[..problem.id.len().min(12)];
        let mut proposals = Vec::with_capacity(solved.leaves.len());
        for (i, leaf) in solved.leaves.iter().enumerate() {
            let rank = i + 1;
            let protected = problem.protected(&leaf.assignment);
            let (map, protections) =
                problem.synthesize(&protected, format!("auto-rep-{short}-r{rank}"))?;
            let state = problem.state_of(&map);
            proposals.push(Proposal {
                record: ProposalRecord {
                    solver_revision: SOLVER_REVISION.into(),
                    problem_id: problem.id.clone(),
                    cuts: cuts.clone(),
                    prior: prior.clone(),
                    outcome,
                    lower_bound: solved.lower_bound,
                    rank,
                    bytes: leaf.cost,
                    state: state.id().clone(),
                    protected,
                },
                map,
                protections,
                state,
            });
        }
        Ok(Proposals {
            proposals,
            outcome,
            lower_bound: solved.lower_bound.map(LogicalBytes::new),
        })
    }
}

#[cfg(test)]
#[path = "solver_tests.rs"]
mod solver_tests;

#[cfg(test)]
#[path = "propose_tests.rs"]
mod propose_tests;
