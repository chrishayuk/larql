//! **What this search IS**: the model, the contract, and every rule a
//! conclusion is drawn under.
//!
//! The first thing an agent should ask, and the answer is entirely
//! declarative — nothing here depends on what has been measured. It
//! exists so that a reader of any other response can tell what the
//! numbers in it were judged by, without being told separately.

use serde::Serialize;

use super::super::compiler::SourceIdentity;
use super::super::measurement::TailSupportPolicy;
use super::super::quality::QualityGate;
use super::super::state::action_space::ActionVocabulary;
use super::super::state::graph::TransitionPolicy;
use super::super::state::semantics::SearchSemantics;
use super::super::state::snapshot::{Objective, SearchSnapshot};
use super::origin::{Origin, Rendered};

/// The declared identity of a search.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Describe {
    /// The stored form's schema. A reader that does not know this
    /// string should not trust its reading of anything below it.
    pub schema: String,
    pub model: SourceIdentity,
    /// What the state-id digest binds to besides the decisions.
    pub surface_identity: String,
    pub surface_tensors: usize,
    pub objective: Objective,
    /// The FROZEN behavioural contract. Changing a threshold means a
    /// new gate id, so this pins what every verdict in this snapshot
    /// means.
    pub contract: QualityGate,
    /// When a percentile is thin enough to stop being one.
    pub tail_support: TailSupportPolicy,
    pub transition_policy: TransitionPolicy,
    /// Whether the policy makes the graph a DAG. A theorem under the
    /// declared policy, not a property of the domain.
    pub guarantees_acyclic: bool,
    /// The six decision procedures between a fact and a conclusion.
    pub semantics: SearchSemantics,
    /// Their joint identity, so a later replay can tell a changed
    /// PROCEDURE from changed data.
    pub semantics_id: String,
    pub ranking_rule: String,
    pub ranking_id: String,
    /// The complete order, stated once. Everything after the first
    /// element exists so that no answer depends on insertion order.
    pub tie_break_chain: Vec<String>,
    /// Every move the search may make. An input, never inferred from
    /// what happens to have been tried — R5-F6 was a vocabulary
    /// failure, and no policy can recover a move never declared.
    pub vocabulary: ActionVocabulary,
}

impl Describe {
    pub fn of(snapshot: &SearchSnapshot) -> Self {
        let graph = snapshot.graph();
        let ranking = &snapshot.config().ranking;
        Self {
            schema: snapshot.schema().to_string(),
            model: graph.model().clone(),
            surface_identity: graph.surface_identity().to_string(),
            surface_tensors: snapshot.space().surface.len(),
            objective: snapshot.objective(),
            contract: snapshot.gate().clone(),
            tail_support: snapshot.tail_support().clone(),
            transition_policy: graph.policy(),
            guarantees_acyclic: graph.policy().guarantees_acyclic(),
            semantics: snapshot.semantics().clone(),
            semantics_id: snapshot.semantics_id().as_str().to_string(),
            ranking_rule: ranking.rule.name().to_string(),
            ranking_id: ranking.id().as_str().to_string(),
            tie_break_chain: ranking
                .tie_break_chain()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            vocabulary: snapshot.space().vocabulary.clone(),
        }
    }
}

impl Rendered for Describe {
    fn origins() -> Vec<Origin> {
        vec![
            Origin::new("schema", "SearchSnapshot::schema()"),
            Origin::new("model", "RepresentationStateGraph::model()"),
            Origin::new(
                "surface_identity",
                "RepresentationStateGraph::surface_identity()",
            ),
            Origin::new("surface_tensors", "TensorSurface::len()"),
            Origin::new("objective", "SearchSnapshot::objective()"),
            Origin::new("contract", "SearchSnapshot::gate()"),
            Origin::new("tail_support", "SearchSnapshot::tail_support()"),
            Origin::new("transition_policy", "RepresentationStateGraph::policy()"),
            Origin::new(
                "guarantees_acyclic",
                "TransitionPolicy::guarantees_acyclic()",
            ),
            Origin::new("semantics", "SearchSnapshot::semantics()"),
            Origin::new("semantics_id", "SearchSnapshot::semantics_id()"),
            Origin::new("ranking_rule", "RankingRule::name()"),
            Origin::new("ranking_id", "RankingSemantics::id()"),
            Origin::new("tie_break_chain", "RankingSemantics::tie_break_chain()"),
            Origin::new("vocabulary", "SearchSpace.vocabulary"),
        ]
    }
}
