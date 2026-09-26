//! Gates 2–9 over a real container: the glimmer fixture's draft and
//! target decoder stacks share tensor names, so one `(projection, layer)`
//! group spans both objects, exactly as the map grammar addresses them.

use std::collections::BTreeMap;

use super::super::super::compiler::read_source_identity;
use super::super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::super::policy::{layer_of, Role};
use super::super::accounting::read_source_storage;
use super::super::footprint::PackCompiledBytes;
use super::super::resolved::{PackLayoutAdmission, PACK_LAYOUT_ADMISSION};
use super::super::surface::SurfaceTensor;
use super::super::tests::container;
use super::*;

/// The group whose shape no NVFP4 pack admits (k = 8 is not a multiple
/// of the 16-wide group).
const REFUSED: (&str, u32) = ("q_proj", 2);

struct Fixture {
    _dir: tempfile::TempDir,
    model: SourceIdentity,
    surface: TensorSurface,
    footprint: SurfaceFootprint,
    base: PrecisionMap,
}

fn is_linear(tensor: &str) -> bool {
    tensor.ends_with("_proj.weight") || tensor == "fc.weight"
}

/// A `[numel / 64, 64]` shape carrying the stored bytes (BF16), so
/// larger tensors cost more; the refused group gets k = 8.
fn shape(tensor: &str, bf16_bytes: u64) -> Vec<usize> {
    let numel = (bf16_bytes / 2) as usize;
    let refused = projection_of(tensor) == Some(REFUSED.0) && layer_of(tensor) == Some(REFUSED.1);
    if refused {
        vec![numel / 8, 8]
    } else {
        vec![numel / 64, 64]
    }
}

fn fixture() -> Fixture {
    let dir = container::glimmer();
    let model = read_source_identity(dir.path()).expect("identity");
    let facts = read_source_storage(dir.path(), &model).expect("facts");
    let surface = TensorSurface::new(facts.tensors().map(|(id, fact)| {
        let role = if is_linear(&id.tensor) {
            Role::DecoderLinear
        } else {
            Role::Norm
        };
        SurfaceTensor::new(
            &id.object,
            &id.tensor,
            role,
            shape(&id.tensor, fact.logical_bytes.get()),
        )
    }))
    .expect("one entry per tensor");
    let bound = facts.bind(&model, &surface).expect("bound");
    let footprint = SurfaceFootprint::new(
        &bound,
        &surface,
        &PackLayoutAdmission,
        &PackCompiledBytes,
        &[DTYPE_NVFP4.to_string()],
    )
    .expect("priced");
    Fixture {
        _dir: dir,
        model,
        surface,
        footprint,
        base: PrecisionMap {
            name: "base".into(),
            encoding: DTYPE_NVFP4.into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![],
        },
    }
}

impl Fixture {
    fn inputs(&self) -> ProposalInputs<'_> {
        ProposalInputs {
            model: &self.model,
            surface: &self.surface,
            base: &self.base,
            layout: &PackLayoutAdmission,
            layout_id: PACK_LAYOUT_ADMISSION,
            footprint: &self.footprint,
            ceiling: None,
            pins: BTreeMap::new(),
            cuts: vec![],
            prior: None,
        }
    }

    fn propose(&self, inputs: ProposalInputs<'_>, k: usize) -> Proposals {
        let problem = ProposalProblem::new(inputs).expect("problem");
        BranchAndBound {
            node_limit: u64::MAX,
        }
        .propose(&problem, k)
        .expect("proposals")
    }
}

fn states(p: &Proposals) -> Vec<RepresentationStateId> {
    p.proposals.iter().map(|p| p.state.id().clone()).collect()
}

#[test]
fn every_proposal_presents_the_bytes_the_footprint_prices() {
    let f = fixture();
    let out = f.propose(f.inputs(), 8);
    assert_eq!(out.proposals.len(), 8);
    assert_eq!(out.outcome, SearchOutcome::Complete);
    for (i, p) in out.proposals.iter().enumerate() {
        let priced = f
            .footprint
            .try_logical_bytes(&p.state)
            .expect("same surface");
        assert_eq!(p.record.bytes, priced.get(), "rank {}", i + 1);
        assert_eq!(p.record.rank, i + 1);
        assert_eq!(&p.record.state, p.state.id());
        let resolved =
            RepresentationState::resolve(&f.model, &f.surface, &p.map, &PackLayoutAdmission);
        assert_eq!(
            resolved.id(),
            p.state.id(),
            "the record names the map's state"
        );
    }
    assert!(out
        .proposals
        .windows(2)
        .all(|w| w[0].record.bytes <= w[1].record.bytes));
    // Compiling every eligible group is cheapest; the refused group is
    // held at source because compiling it is structurally meaningless.
    assert_eq!(
        out.proposals[0].record.protected,
        vec![GroupKey::new(REFUSED.0, REFUSED.1)]
    );
    assert_eq!(
        out.lower_bound.map(LogicalBytes::get),
        Some(out.proposals[0].record.bytes)
    );
    // A group spans objects: draft and target both hold `0.mlp.down_proj`.
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    assert!(problem
        .groups()
        .any(|g| g == &GroupKey::new("down_proj", 0)));
    assert!(problem.fixed().contains(&(
        "draft.feature_projector".to_string(),
        "fc.weight".to_string()
    )));
}

#[test]
fn every_proposal_passes_the_map_check_and_a_dead_rule_is_refused() {
    let f = fixture();
    let out = f.propose(f.inputs(), 8);
    for p in &out.proposals {
        p.map
            .check_against(
                f.surface
                    .entries()
                    .iter()
                    .map(|t| (t.role, t.tensor.as_str())),
            )
            .expect("a synthesised map decides every exception");
    }
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    let err = problem
        .synthesize(&[GroupKey::new("q_proj", 99)], "dead".into())
        .unwrap_err()
        .to_string();
    assert!(err.contains("check refuses"), "{err}");
}

#[test]
fn an_exact_no_good_removes_that_state_and_no_other() {
    let f = fixture();
    let before = f.propose(f.inputs(), 4);
    let cut = before.proposals[0].state.id().clone();
    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::ExactNoGood { state: cut.clone() }];
    let after = f.propose(inputs, 3);
    assert_eq!(states(&after), states(&before)[1..].to_vec());
    assert!(!states(&after).contains(&cut));
    assert!(after.proposals[0]
        .record
        .cuts
        .contains(&Cut::ExactNoGood { state: cut }.id()));
    // The states that differ from the cut state in one group survive:
    // the cut is not widened into a claim about any one choice.
    let cut_protected = &before.proposals[0].record.protected;
    for p in &after.proposals {
        let differ = p
            .record
            .protected
            .iter()
            .filter(|g| !cut_protected.contains(g))
            .count()
            + cut_protected
                .iter()
                .filter(|g| !p.record.protected.contains(g))
                .count();
        assert!(differ >= 1);
    }
    assert_eq!(
        after.proposals[0].record.protected.len(),
        cut_protected.len() + 1
    );
}

#[test]
fn a_structural_cut_never_appears_and_is_recorded() {
    let f = fixture();
    let group = GroupKey::new("o_proj", 1);
    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::Structural {
        group: group.clone(),
        choice: GroupChoice::Compile,
        reason: "no kernel".into(),
    }];
    let out = f.propose(inputs, 5);
    for p in &out.proposals {
        assert!(p.record.protected.contains(&group), "Compile was cut");
        assert!(p
            .record
            .cuts
            .contains(&"structural:o_proj@1:compile:no kernel".to_string()));
        assert!(p
            .record
            .cuts
            .iter()
            .any(|c| c.starts_with("structural:q_proj@2:compile:layout refuses")));
    }
}

#[test]
fn a_truncated_search_says_so_and_identical_inputs_give_identical_records() {
    let f = fixture();
    let complete = f.propose(f.inputs(), 3);
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    let truncated = BranchAndBound { node_limit: 5 }
        .propose(&problem, 3)
        .unwrap();
    assert!(matches!(truncated.outcome, SearchOutcome::NodeLimit { .. }));
    let bound = truncated.lower_bound.expect("bounded");
    assert!(bound.get() <= complete.proposals[0].record.bytes);

    let again = f.propose(f.inputs(), 3);
    let records = |p: &Proposals| {
        p.proposals
            .iter()
            .map(|p| serde_json::to_string(&p.record).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(records(&complete), records(&again));
    assert_eq!(
        complete.proposals[0].record.solver_revision,
        SOLVER_REVISION
    );
}

struct Prefers {
    projection: &'static str,
    evidence: SearchEvidence,
}

impl OrderingPrior for Prefers {
    fn evidence(&self) -> SearchEvidence {
        self.evidence.clone()
    }
    fn identity(&self) -> String {
        format!("prefers-protecting-{}", self.projection)
    }
    fn score(&self, group: &GroupKey, choice: GroupChoice) -> f64 {
        if group.projection == self.projection && choice == GroupChoice::Source {
            -1.0
        } else {
            0.0
        }
    }
}

#[test]
fn a_prior_reorders_only_equal_byte_states_and_unusable_is_refused() {
    let f = fixture();
    let plain = f.propose(f.inputs(), 12);
    let prior = Prefers {
        projection: "k_proj",
        evidence: SearchEvidence::OrderingProxy {
            calibration: "test".into(),
        },
    };
    let mut inputs = f.inputs();
    inputs.prior = Some(&prior);
    let ordered = f.propose(inputs, 12);
    let bytes = |p: &Proposals| {
        p.proposals
            .iter()
            .map(|p| p.record.bytes)
            .collect::<Vec<_>>()
    };
    assert_eq!(bytes(&plain), bytes(&ordered), "the byte order never moves");
    assert_ne!(
        states(&plain),
        states(&ordered),
        "a tie was broken differently"
    );
    // Within one byte level the preferred protection comes first; without
    // the prior the canonical assignment order puts `v_proj` first.
    let first_tie = |p: &Proposals| {
        let i = p
            .proposals
            .windows(2)
            .position(|w| w[0].record.bytes == w[1].record.bytes)
            .expect("k_proj and v_proj are the same size, so ties exist");
        p.proposals[i].record.protected.clone()
    };
    assert!(first_tie(&plain).iter().any(|g| g.projection == "v_proj"));
    assert!(first_tie(&ordered).iter().any(|g| g.projection == "k_proj"));
    assert_eq!(
        ordered.proposals[0]
            .record
            .prior
            .as_ref()
            .map(|p| p.identity.as_str()),
        Some("prefers-protecting-k_proj")
    );

    let unusable = Prefers {
        projection: "v_proj",
        evidence: SearchEvidence::Unusable,
    };
    let mut inputs = f.inputs();
    inputs.prior = Some(&unusable);
    assert!(ProposalProblem::new(inputs).is_err());
}

#[test]
fn malformed_problems_are_refused() {
    let f = fixture();
    let mut with_exception = f.base.clone();
    with_exception
        .exceptions
        .push(super::super::super::map::Exception {
            projection: Some("q_proj".into()),
            layers: None,
            encoding: None,
        });
    let mut inputs = f.inputs();
    inputs.base = &with_exception;
    assert!(ProposalProblem::new(inputs).is_err());

    let mut inputs = f.inputs();
    inputs.pins = BTreeMap::from([(GroupKey::new("q_proj", 99), GroupChoice::Source)]);
    assert!(ProposalProblem::new(inputs).is_err());

    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::Structural {
        group: GroupKey::new("nope", 0),
        choice: GroupChoice::Source,
        reason: "x".into(),
    }];
    assert!(ProposalProblem::new(inputs).is_err());

    // A ceiling below the cheapest state leaves nothing feasible.
    let mut inputs = f.inputs();
    inputs.ceiling = Some(LogicalBytes::new(1));
    let out = f.propose(inputs, 3);
    assert!(out.proposals.is_empty());
    assert_eq!(out.lower_bound, None);
}

#[test]
fn a_proposals_protections_compile_to_the_same_state() {
    let f = fixture();
    let mut inputs = f.inputs();
    inputs.pins = BTreeMap::from([
        (GroupKey::new("up_proj", 0), GroupChoice::Source),
        (GroupKey::new("up_proj", 1), GroupChoice::Source),
        (GroupKey::new("up_proj", 3), GroupChoice::Source),
    ]);
    let out = f.propose(inputs, 4);
    for p in &out.proposals {
        // What `compile_representation` would be handed: the base map's
        // encoding and roles with these protections, as `from_policy`
        // writes them.
        let compiled = PrecisionMap {
            exceptions: p.protections.as_exceptions(),
            ..f.base.clone()
        };
        let state =
            RepresentationState::resolve(&f.model, &f.surface, &compiled, &PackLayoutAdmission);
        assert_eq!(state.id(), p.state.id());
    }
    // Consecutive layers of one projection merge into one rule.
    let top = &out.proposals[0].map.exceptions;
    assert!(top
        .iter()
        .any(|e| e.projection.as_deref() == Some("up_proj") && e.layers == Some((0, 1))));
    assert!(top
        .iter()
        .any(|e| e.projection.as_deref() == Some("up_proj") && e.layers == Some((3, 3))));
}
