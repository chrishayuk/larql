//! **AUTO-REP-1b: the validate loop.** Propose, compile, measure, ingest,
//! cut, propose again.
//!
//! Adds no gate, no procedure and no authority. Every step is an existing
//! door (`docs/auto-rep-1.md`, 1b contract):
//!
//! ```text
//! record ──cuts──▶ 1a solver ──rank 1──▶ Protections ──compile──▶ candidate
//!   ▲                                                      │ establish = state?
//!   │                                   MeasurementRequest::of(key, applied)
//!   └──── ingest ◀── artifact ◀── ExecutorRegistry::execute ◀──┘
//!            │
//!       adjudicate(key): admitted → stop · refused → ExactNoGood next pass
//! ```
//!
//! An `ExactNoGood` comes only from an Authority-scale reading the gate
//! refused, and excludes that state alone. Admission here is the gate's
//! verdict on one reading; promotion stays a separate step.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::actuate::artifacts::DeclaredArtifacts;
use super::actuate::executor::ExecutorRegistry;
use super::actuate::prepare::{PreparedExperiment, Ready};
use super::actuate::request::MeasurementRequest;
use super::ingest::artifact::MeasurementArtifact;
use super::ingest::state_evidence::ArtifactStateEvidence;
use super::ingest::{ingest, IngestionSources};
use super::map::Exception;
use super::measurement::EvidenceScale;
use super::state::action_space::{ActionVocabulary, MapEdit};
use super::state::key::MeasurementKey;
use super::state::propose::{
    group_keys, BranchAndBound, Cut, GroupChoice, GroupKey, Proposal, ProposalInputs,
    ProposalProblem, ProposalRecord, Proposer,
};
use super::state::snapshot::SearchSnapshot;
use super::state::RepresentationStateId;
use super::{compile_representation, RepresentSpec};
use crate::error::VindexError;

/// Which loop produced a campaign record.
pub const CAMPAIGN_REVISION: &str = "auto-rep-validate/v1";

fn refused(message: impl Into<String>) -> VindexError {
    VindexError::Parse(format!("AUTO-REP campaign: {}", message.into()))
}

/// The vocabulary edit that protects one group.
pub fn edit_name(group: &GroupKey) -> String {
    format!("protect:{}@{}", group.projection, group.layer)
}

/// One source-precision edit per group the base map presents, so every
/// 1a proposal is an applied set of this vocabulary.
pub fn group_vocabulary(
    snapshot_space: &super::state::snapshot::SearchSpace,
) -> Result<ActionVocabulary, VindexError> {
    ActionVocabulary::new(
        group_keys(&snapshot_space.surface, &snapshot_space.base_map)
            .into_iter()
            .map(|g| {
                MapEdit::new(
                    edit_name(&g),
                    Exception {
                        projection: Some(g.projection.clone()),
                        layers: Some((g.layer, g.layer)),
                        encoding: None,
                    },
                )
            }),
    )
}

/// The applied set a proposal is, in its vocabulary.
pub fn applied_for(protected: &[GroupKey]) -> BTreeSet<String> {
    protected.iter().map(edit_name).collect()
}

/// Builds a candidate container. The production one calls
/// [`compile_representation`]; a test can substitute one that lies, to
/// prove the identity check refuses it.
pub trait CandidateCompiler {
    fn compile(&self, spec: &RepresentSpec, out: &Path) -> Result<(), VindexError>;
}

/// [`compile_representation`] from one source container.
pub struct RepresentCompiler<'a> {
    pub source: &'a Path,
}

impl CandidateCompiler for RepresentCompiler<'_> {
    fn compile(&self, spec: &RepresentSpec, out: &Path) -> Result<(), VindexError> {
        compile_representation(self.source, out, spec).map(drop)
    }
}

/// Everything a campaign needs besides the record it advances.
pub struct CampaignSetup<'a> {
    /// The source container the record describes.
    pub source: &'a Path,
    /// The corpus the protocol's bank was sealed from.
    pub corpus: &'a Path,
    /// Where candidates are compiled, one directory per state.
    pub workdir: &'a Path,
    /// Encoding and roles; must agree with the record's base map. Its
    /// protections are replaced by each proposal's.
    pub spec: &'a RepresentSpec,
    pub compiler: &'a dyn CandidateCompiler,
    pub executors: &'a ExecutorRegistry<'a>,
    /// Measurements this campaign may spend. Reused readings are free.
    pub budget: usize,
    pub node_limit: u64,
    /// Groups the caller fixes; the loop searches the rest.
    pub pins: BTreeMap<GroupKey, GroupChoice>,
}

/// The gate's verdict on one entry's reading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Admitted,
    /// The criteria it failed, in the gate's order.
    Refused {
        failures: Vec<String>,
    },
}

/// One proposal the loop acted on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignEntry {
    pub proposal: ProposalRecord,
    pub key: MeasurementKey,
    /// The reading was already in the record; nothing was compiled or run.
    pub reused: bool,
    pub verdict: Verdict,
}

/// How a campaign ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignOutcome {
    Admitted {
        state: RepresentationStateId,
    },
    /// Every state is cut or none fits the ceiling.
    Exhausted,
    BudgetSpent,
}

/// **What the campaign did**, rejected proposals included: the
/// rejection frontier is evidence too.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignRecord {
    pub revision: String,
    pub outcome: CampaignOutcome,
    pub measurements_spent: usize,
    pub entries: Vec<CampaignEntry>,
}

/// Refuse a record or setup the loop cannot honestly run.
fn check(snapshot: &SearchSnapshot, setup: &CampaignSetup<'_>) -> Result<(), VindexError> {
    if snapshot.gate().is_none() {
        return Err(refused(
            "the record is characterisation-only (no gate): no reading can be refused, so no \
             cut can be earned and nothing can be admitted",
        ));
    }
    if snapshot.standing_intent().scale != EvidenceScale::Authority {
        return Err(refused(
            "the standing intent is not at Authority scale; a refusal there is not a failed \
             joint measurement and cannot become a cut",
        ));
    }
    let space = snapshot.space();
    if !space.base_map.exceptions.is_empty() {
        return Err(refused(
            "the base map has exceptions; the solver owns every exception",
        ));
    }
    let expected = group_vocabulary(space)?;
    if space.vocabulary.edits() != expected.edits() {
        return Err(refused(
            "the vocabulary is not the group vocabulary of this surface",
        ));
    }
    let roles: Vec<String> = setup
        .spec
        .roles
        .roles()
        .iter()
        .map(|r| r.name().to_string())
        .collect();
    if setup.spec.encoding != space.base_map.encoding || roles != space.base_map.roles {
        return Err(refused(
            "the compile spec's encoding or roles disagree with the base map",
        ));
    }
    Ok(())
}

/// Every state an Authority reading in the record refused.
fn no_goods(snapshot: &SearchSnapshot) -> Vec<Cut> {
    let states: BTreeSet<RepresentationStateId> = snapshot
        .measurements()
        .keys()
        .filter(|k| k.scale() == EvidenceScale::Authority)
        .filter(|k| snapshot.adjudicate(k).is_some_and(|a| !a.admissible()))
        .map(|k| k.state().clone())
        .collect();
    states
        .into_iter()
        .map(|state| Cut::ExactNoGood { state })
        .collect()
}

fn verdict(snapshot: &SearchSnapshot, key: &MeasurementKey) -> Result<Verdict, VindexError> {
    let adjudication = snapshot
        .adjudicate(key)
        .ok_or_else(|| refused("an ingested reading is not in the record"))?;
    Ok(if adjudication.admissible() {
        Verdict::Admitted
    } else {
        Verdict::Refused {
            failures: adjudication
                .failures()
                .iter()
                .map(|m| format!("{:?}", m.criterion))
                .collect(),
        }
    })
}

/// The rank-1 proposal under the record's current cuts, if any.
fn propose(
    snapshot: &SearchSnapshot,
    setup: &CampaignSetup<'_>,
) -> Result<Option<Proposal>, VindexError> {
    let layout = snapshot.layout()?;
    let footprint = snapshot.footprint()?;
    let space = snapshot.space();
    let problem = ProposalProblem::new(ProposalInputs {
        model: snapshot.graph().model(),
        surface: &space.surface,
        base: &space.base_map,
        layout,
        layout_id: &snapshot.semantics().layout_admission,
        footprint: &footprint,
        ceiling: None,
        pins: setup.pins.clone(),
        cuts: no_goods(snapshot),
        prior: None,
    })?;
    Ok(BranchAndBound {
        node_limit: setup.node_limit,
    }
    .propose(&problem, 1)?
    .proposals
    .into_iter()
    .next())
}

/// Compile a proposal, or reuse a directory that already is it.
fn candidate_for(
    setup: &CampaignSetup<'_>,
    proposal: &Proposal,
) -> Result<(PathBuf, super::ingest::state_evidence::EstablishedState), VindexError> {
    let path = setup.workdir.join(proposal.state.id().short());
    if !path.exists() {
        let mut spec = setup.spec.clone();
        spec.protect = proposal.protections.clone();
        setup.compiler.compile(&spec, &path)?;
    }
    let established = ArtifactStateEvidence::establish(&path)
        .map_err(|e| refused(format!("candidate at {}: {e}", path.display())))?;
    if established.established() != proposal.state.id() {
        return Err(refused(format!(
            "the candidate at {} is state {}, not the proposed {}",
            path.display(),
            established.established().as_str(),
            proposal.state.id().as_str()
        )));
    }
    Ok((path, established))
}

/// Measure one proposal through the existing doors and ingest it.
fn measure(
    snapshot: &mut SearchSnapshot,
    setup: &CampaignSetup<'_>,
    proposal: &Proposal,
    key: &MeasurementKey,
) -> Result<(), VindexError> {
    let (candidate, established) = candidate_for(setup, proposal)?;
    let request = MeasurementRequest::of(snapshot, key, &applied_for(&proposal.record.protected))
        .map_err(|e| refused(format!("request: {e}")))?;
    let prepared = PreparedExperiment::Ready(Box::new(Ready {
        request,
        physical_delta: 0,
        routes: 1,
        considered: 1,
    }));
    let request = prepared.request().expect("just built");
    let locator = DeclaredArtifacts::new()
        .container_at(setup.source)
        .corpus_at(setup.corpus)
        .overlay_at(key.state(), &candidate);
    let observed = setup
        .executors
        .execute(request, &locator)
        .map_err(|e| refused(format!("execution: {e}")))?;
    let artifact = MeasurementArtifact::from_execution(&prepared, &observed, &established)
        .map_err(|e| refused(format!("artifact: {e}")))?;
    ingest(
        snapshot,
        &prepared,
        &artifact,
        &IngestionSources {
            container: setup.source,
            candidate: &candidate,
            corpus: setup.corpus,
        },
    )
    .map_err(|e| refused(format!("ingestion: {e}")))?;
    Ok(())
}

/// **Run the loop** until a proposal is admitted, none remain, or the
/// budget is spent. The record advances only through ingestion.
pub fn run(
    snapshot: &mut SearchSnapshot,
    setup: &CampaignSetup<'_>,
) -> Result<CampaignRecord, VindexError> {
    check(snapshot, setup)?;
    let mut entries = Vec::new();
    let mut spent = 0usize;
    let outcome = loop {
        let Some(proposal) = propose(snapshot, setup)? else {
            break CampaignOutcome::Exhausted;
        };
        let key = snapshot.standing_intent().key_for(proposal.state.id());
        let reused = snapshot.measurements().contains(&key);
        if !reused {
            if spent == setup.budget {
                break CampaignOutcome::BudgetSpent;
            }
            measure(snapshot, setup, &proposal, &key)?;
            spent += 1;
        }
        let verdict = verdict(snapshot, &key)?;
        let admitted = verdict == Verdict::Admitted;
        if reused && !admitted {
            // Unreachable while the cuts are derived from the record: a
            // refused reading excludes its state. Refusing here, rather
            // than asserting, is what guarantees the loop terminates.
            return Err(refused(format!(
                "proposed {}, which the record already refused",
                proposal.state.id().as_str()
            )));
        }
        entries.push(CampaignEntry {
            proposal: proposal.record.clone(),
            key,
            reused,
            verdict,
        });
        if admitted {
            break CampaignOutcome::Admitted {
                state: proposal.state.id().clone(),
            };
        }
    };
    Ok(CampaignRecord {
        revision: CAMPAIGN_REVISION.into(),
        outcome,
        measurements_spent: spent,
        entries,
    })
}

#[cfg(test)]
#[path = "auto_rep_tests.rs"]
mod tests;
