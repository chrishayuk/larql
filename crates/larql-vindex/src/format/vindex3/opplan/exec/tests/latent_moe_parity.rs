//! **K3-LATENTMOE-1 — the latent routed branch against its oracle.**
//!
//! The oracle was committed before any of this arithmetic existed
//! (`scripts/kimi_latent_moe_oracle_export.py`), so these are not
//! self-consistent comparisons: the reference is `transformers`' own
//! `KimiSparseMoeBlock` forward, and the numbers here had no chance to
//! be fitted to it.
//!
//! The functions under test are the ones execution runs —
//! [`enter_latent`] and [`exit_latent`] from `experts.rs`, not a second
//! copy written here. A test that reimplements the operator proves the
//! two implementations agree and says nothing about whether either is
//! right. Both take a backend, because both projections and the norm go
//! through one: the wrapper is not host glue, and running it here on a
//! path execution does not use would make these numbers about a
//! different program.
//!
//! # What the geometry is for
//!
//! `hidden = 10`, `latent = 7`, `intermediate = 4`. All three differ,
//! and `hidden / 2 = 5` is none of them, so a latent width silently
//! derived from half the hidden size or from the expert intermediate
//! width cannot pass. That is deliberate: those are the two derivations
//! a reader would reach for, and the fixture exists to refuse them.
//!
//! # What is asserted, and what cannot be
//!
//! The oracle found that of the operator's placement facts only the
//! NORM's is a silent defect. Routing on the latent and running the
//! shared branch through the bottleneck both fail on SHAPE at this
//! geometry — the router is `[experts, hidden]` and the shared expert's
//! `w1` is `[inter, hidden]`, so neither can accept a `[7]` vector.
//! Those two are therefore witnessed POSITIVELY, by cross-form
//! bit-identity, rather than by a mutant that cannot be built.

use serde_json::Value;

use super::super::backend::WeightSlice;
use super::super::experts::{enter_latent, exit_latent};
use super::super::production::ProductionBackend;

/// The committed oracle.
const ORACLE: &str = include_str!("kimi_latent_moe_oracle.json");

/// Every boundary must land this close to the reference. The arithmetic
/// is f32 over widths of 7 and 10, so the accumulated error is tiny; a
/// loose bound here would let a wrong epsilon through, and the epsilon
/// is the point of the rung.
const TOLERANCE: f32 = 2e-6;

struct Fixture {
    doc: Value,
    hidden: usize,
    latent: usize,
}

fn floats(node: &Value) -> Vec<f32> {
    node.as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

fn load() -> Fixture {
    let doc: Value = serde_json::from_str(ORACLE).expect("the oracle parses");
    let hidden = doc["hidden"].as_u64().expect("hidden") as usize;
    let latent = doc["latent"].as_u64().expect("latent") as usize;
    Fixture {
        doc,
        hidden,
        latent,
    }
}

impl Fixture {
    fn weight(&self, name: &str) -> Vec<f32> {
        floats(&self.doc["weights"][name])
    }

    fn latent_arm(&self, boundary: &str) -> Vec<f32> {
        floats(&self.doc["arms"]["latent"][boundary])
    }

    fn norm_eps(&self) -> f64 {
        self.doc["latent_norm_eps"].as_f64().expect("eps")
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "boundary widths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The fixture must be able to SEE the facts it is asked about before
/// any of them is scored: three distinct widths, and a latent width that
/// is neither of the two numbers a reader would derive it from.
#[test]
fn the_geometry_can_distinguish_the_derivations_it_must_refuse() {
    let f = load();
    let inter = f.doc["intermediate"].as_u64().unwrap() as usize;
    assert_ne!(
        f.latent, f.hidden,
        "a bottleneck that is not one proves nothing"
    );
    assert_ne!(
        f.latent,
        f.hidden / 2,
        "hidden/2 must not pass as the latent width"
    );
    assert_ne!(f.latent, inter, "the expert width must not pass either");
    assert_eq!(
        f.weight("routed_expert_down_proj").len(),
        f.latent * f.hidden
    );
    assert_eq!(f.weight("routed_expert_up_proj").len(), f.hidden * f.latent);
    assert_eq!(f.weight("routed_expert_norm").len(), f.latent);
}

/// **Entering the bottleneck.** `routed_expert_down_proj · x`.
#[test]
fn the_down_projection_matches_the_reference() {
    let f = load();
    let x = floats(&f.doc["input"]);
    let down = f.weight("routed_expert_down_proj");
    let got = enter_latent(
        &ProductionBackend::new(),
        WeightSlice::F32(&down),
        &x,
        f.latent,
        f.hidden,
    )
    .expect("the down projection executes");
    let want = f.latent_arm("latent");
    assert_eq!(got.len(), f.latent);
    assert!(
        max_abs_diff(&got, &want) < TOLERANCE,
        "down projection differs by {}",
        max_abs_diff(&got, &want)
    );
}

/// **The norm, on the weighted aggregate.** The oracle's `routed_sum`
/// in, its `routed_normed` out — one vector per token, after top-k
/// weighting and summation, before the expansion.
#[test]
fn the_norm_applies_to_the_weighted_aggregate() {
    let f = load();
    let aggregate = f.latent_arm("routed_sum");
    let weight = f.weight("routed_expert_norm");
    // `exit_latent` normalises and then expands; to see the norm alone,
    // expand with an identity of the right shape.
    let identity: Vec<f32> = (0..f.latent * f.latent)
        .map(|i| {
            if i / f.latent == i % f.latent {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let got = exit_latent(
        &ProductionBackend::new(),
        aggregate,
        Some((&weight, f.norm_eps())),
        WeightSlice::F32(&identity),
        f.latent,
        f.latent,
    )
    .expect("the norm and the identity expansion execute");
    let want = f.latent_arm("routed_normed");
    assert!(
        max_abs_diff(&got, &want) < TOLERANCE,
        "normed aggregate differs by {}",
        max_abs_diff(&got, &want)
    );
}

/// **Leaving the bottleneck**, norm and up-projection together — the
/// whole `routed_sum -> routed_out` segment as execution runs it.
#[test]
fn the_branch_exit_matches_the_reference() {
    let f = load();
    let up = f.weight("routed_expert_up_proj");
    let got = exit_latent(
        &ProductionBackend::new(),
        f.latent_arm("routed_sum"),
        Some((&f.weight("routed_expert_norm"), f.norm_eps())),
        WeightSlice::F32(&up),
        f.hidden,
        f.latent,
    )
    .expect("the branch exit executes");
    let want = f.latent_arm("routed_out");
    assert_eq!(
        got.len(),
        f.hidden,
        "the branch must return to the residual width"
    );
    assert!(
        max_abs_diff(&got, &want) < TOLERANCE,
        "routed_out differs by {}",
        max_abs_diff(&got, &want)
    );
}

/// **The anti-memory control.** K3-ACT-1 and K3-MLA-Q-LORA-1 both found
/// a `1e-6` class-default epsilon on a low-rank norm; this one runs at
/// the LAYER's `1e-5`, because the reference passes it explicitly.
///
/// Substituting the neighbours' value must move the boundary well
/// outside tolerance — otherwise the fixture cannot see the difference
/// and the epsilon assertions above are decoration.
#[test]
fn the_class_default_epsilon_is_visibly_wrong_here() {
    let f = load();
    let wrong_eps = f.doc["class_default_eps_not_used_here"]
        .as_f64()
        .expect("the oracle exports the epsilon it does NOT use");
    assert_ne!(wrong_eps, f.norm_eps());

    let up = f.weight("routed_expert_up_proj");
    let with_wrong = exit_latent(
        &ProductionBackend::new(),
        f.latent_arm("routed_sum"),
        Some((&f.weight("routed_expert_norm"), wrong_eps)),
        WeightSlice::F32(&up),
        f.hidden,
        f.latent,
    )
    .expect("the branch exit executes");
    let want = f.latent_arm("routed_out");
    let drift = max_abs_diff(&with_wrong, &want);
    assert!(
        drift > TOLERANCE * 10.0,
        "the fixture cannot see a 10x epsilon error (drift {drift}) — it cannot \
         witness the rung's own finding, and the parity assertions above prove nothing \
         about which epsilon was used"
    );
}

/// **The norm is not optional in disguise.** Omitting it must change the
/// answer; if it did not, every assertion about the norm's placement
/// would pass with the norm removed entirely.
#[test]
fn omitting_the_norm_changes_the_branch_output() {
    let f = load();
    let up = f.weight("routed_expert_up_proj");
    let without = exit_latent(
        &ProductionBackend::new(),
        f.latent_arm("routed_sum"),
        None,
        WeightSlice::F32(&up),
        f.hidden,
        f.latent,
    )
    .expect("the branch exit executes");
    let want = f.latent_arm("routed_out");
    assert!(
        max_abs_diff(&without, &want) > TOLERANCE * 10.0,
        "removing the norm left the output unchanged — the fixture cannot witness it"
    );
}

/// **The two placement facts no mutant can express**, witnessed
/// positively instead.
///
/// The oracle records these as `structurally_unreachable`: the router is
/// `[experts, hidden]` and the shared expert reads `hidden`, so feeding
/// either the `[7]` latent fails on shape rather than computing a
/// different model. Their evidence is that both arms agree bit-for-bit
/// on what the router saw and what the shared branch produced — which is
/// exactly the claim "the router reads the un-projected input, and the
/// shared branch stays outside the bottleneck".
#[test]
fn the_router_and_shared_branch_are_identical_across_both_forms() {
    let f = load();
    let bit_identical = f.doc["cross_form"]["bit_identical"]
        .as_array()
        .expect("the oracle records which boundaries it asserted equal");
    let names: Vec<&str> = bit_identical.iter().map(|v| v.as_str().unwrap()).collect();
    for required in ["router_weights", "shared_input", "shared_output"] {
        assert!(
            names.contains(&required),
            "{required} must be witnessed identical across the two forms"
        );
    }
    for boundary in ["shared_input", "shared_output"] {
        let latent = floats(&f.doc["arms"]["latent"][boundary]);
        let ordinary = floats(&f.doc["arms"]["ordinary"][boundary]);
        assert_eq!(
            latent, ordinary,
            "{boundary} differs between the forms — the shared branch entered the bottleneck"
        );
        assert_eq!(latent.len(), f.hidden, "the shared branch runs at hidden");
    }
    // And the router's input is the block input itself, at full width.
    assert_eq!(
        floats(&f.doc["arms"]["latent"]["router_input"]),
        floats(&f.doc["input"]),
        "the router reads the un-projected block input"
    );
}
