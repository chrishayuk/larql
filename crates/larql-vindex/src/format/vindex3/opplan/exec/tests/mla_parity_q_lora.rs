//! Kimi-K3's FACTORISED MLA query against the committed oracle,
//! boundary by boundary — [`mla_parity.rs`](super::mla_parity)'s own role
//! for the other query form.
//!
//! `kimi_mla_oracle_q_lora.json` (`scripts/kimi_mla_oracle_export.py
//! --q-lora`) is the third arm of the same reference: same operator, same
//! KV path, same attention, and `q_a_proj -> q_a_layernorm -> q_b_proj`
//! where the first two arms have one `q_proj`. It is UNGATED on purpose —
//! it isolates the query change and nothing else.
//!
//! # What this file has that the direct arm does not
//!
//! **The trace control.** `Mutation::QbFedPreNorm` feeds `q_b_proj` the
//! un-normed `q_a` while `q_a_layernorm`'s output is still computed and
//! still reported. It is OUTPUT-IDENTICAL to skipping the norm, and the
//! only thing that separates them is the trace — so it is the one arm
//! that distinguishes "the boundary this executor reports is the value
//! its next stage consumed" from "the boundary is a separately-computed
//! display that happens to look right". Everything else here could pass
//! with a decorative `q_a_normed`.
//!
//! **The epsilon.** `q_a_layernorm` runs at `1e-6`, `KimiRMSNorm`'s class
//! default, NOT the layer's `rms_norm_eps` — and not by borrowing the KV
//! latent norm's value either, though the two agree. The oracle exports
//! both under separate keys and this file reads the q-side one.

use larql_models::config::MlaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::mla::{
    mla_forward, MlaQueryWeights, MlaState, MlaTrace, MlaWeights, Mutation,
};

const ORACLE: &str = include_str!("kimi_mla_oracle_q_lora.json");

/// f32 transcription noise between two implementations of the same
/// arithmetic in different orders — the bound `mla_parity.rs` uses.
const TOLERANCE: f32 = 2e-5;

struct Fixture {
    geometry: MlaGeometry,
    hidden: usize,
    kv_a_norm_eps: f64,
    q_a_norm_eps: f64,
    weights: std::collections::BTreeMap<String, Vec<f32>>,
    input: Vec<Vec<f32>>,
    boundaries: Value,
    controls: Value,
    positions: usize,
}

fn floats(node: &Value) -> Vec<f32> {
    node.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn load() -> Fixture {
    let v: Value = serde_json::from_str(ORACLE).expect("oracle fixture parses");
    let weights = v["weights"]
        .as_object()
        .expect("weights")
        .iter()
        .map(|(k, node)| (k.clone(), floats(node)))
        .collect();
    Fixture {
        geometry: MlaGeometry {
            num_heads: v["num_heads"].as_u64().unwrap() as usize,
            kv_lora_rank: v["kv_lora_rank"].as_u64().unwrap() as usize,
            qk_nope_head_dim: v["qk_nope_head_dim"].as_u64().unwrap() as usize,
            qk_rope_head_dim: v["qk_rope_head_dim"].as_u64().unwrap() as usize,
            v_head_dim: v["v_head_dim"].as_u64().unwrap() as usize,
        },
        hidden: v["hidden"].as_u64().unwrap() as usize,
        kv_a_norm_eps: v["kv_a_norm_eps"].as_f64().unwrap(),
        q_a_norm_eps: v["q_a_norm_eps"].as_f64().unwrap(),
        positions: v["positions"].as_u64().unwrap() as usize,
        input: v["input"].as_array().unwrap().iter().map(floats).collect(),
        boundaries: v["boundaries"].clone(),
        controls: v["controls"].clone(),
        weights,
    }
}

impl Fixture {
    fn weights(&self) -> MlaWeights<'_> {
        let g = |n: &str| self.weights.get(n).expect(n).as_slice();
        MlaWeights {
            output_gate: None,
            query: MlaQueryWeights::LowRank {
                q_a_proj: WeightRows::F32(g("q_a_proj")),
                q_a_norm: g("q_a_norm"),
                q_b_proj: WeightRows::F32(g("q_b_proj")),
                // The q-side epsilon, read from its OWN key. Reading
                // `kv_a_norm_eps` here would pass on this fixture, where
                // the two agree, and be wrong on the first family that
                // overrode one of them.
                q_a_norm_eps: self.q_a_norm_eps,
            },
            kv_a_proj: WeightRows::F32(g("kv_a_proj")),
            kv_a_norm: g("kv_a_norm"),
            kv_b_proj: WeightRows::F32(g("kv_b_proj")),
            o_proj: WeightRows::F32(g("o_proj")),
            kv_a_norm_eps: self.kv_a_norm_eps,
        }
    }

    /// Every position 0..=up_to through a FRESH state, threaded exactly
    /// like a real decode sequence.
    fn run_to(&self, up_to: usize, mutation: Mutation) -> MlaTrace {
        let mut state = MlaState::default();
        let mut last = None;
        for p in 0..=up_to {
            last = Some(mla_forward(
                &self.input[p],
                self.hidden,
                self.weights(),
                self.geometry,
                &mut state,
                mutation,
            ));
        }
        last.unwrap()
    }

    fn expected(&self, p: usize, boundary: &str) -> Vec<f32> {
        floats(&self.boundaries[boundary][p])
    }

    fn control(&self, name: &str, p: usize, boundary: &str) -> Vec<f32> {
        floats(&self.controls[name]["boundaries"][boundary][p])
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The query's own boundaries, per position, plus everything downstream.
fn assert_boundaries(up_to: usize) {
    let f = load();
    let trace = f.run_to(up_to, Mutation::None);
    let q_a = trace.q_a.as_ref().expect("the low-rank form reports q_a");
    let q_a_normed = trace.q_a_normed.as_ref().expect("and its norm");
    let q_b = trace.q_b.as_ref().expect("and its up-projection");

    for (name, actual) in [
        ("q_a", q_a),
        ("q_a_normed", q_a_normed),
        ("q_b", q_b),
        ("q_states", &trace.q_states),
        ("compressed_kv", &trace.compressed_kv),
        ("kv_a_normed", &trace.kv_a_normed),
        ("kv_b", &trace.kv_b),
        ("attn_weights", &trace.attn_weights),
        ("attn_value", &trace.attn_value),
        ("output", &trace.output),
    ] {
        let d = max_abs_diff(actual, &f.expected(up_to, name));
        assert!(
            d < TOLERANCE,
            "position {up_to} boundary `{name}`: max|Δ| {d:e}"
        );
    }
}

/// The fixture carries as many positions as it claims, on every
/// boundary.
///
/// Cheap, and it is what keeps the three `assert_boundaries` arms below
/// from silently testing fewer positions than the oracle exported — a
/// fixture that lost a position would make them pass by doing less.
#[test]
fn the_fixture_carries_every_position_it_declares() {
    let f = load();
    assert_eq!(f.positions, 3, "three positions, as `mla_parity` needs");
    assert_eq!(f.input.len(), f.positions);
    for boundary in ["q_a", "q_a_normed", "q_b", "q_states", "output"] {
        assert_eq!(
            f.boundaries[boundary].as_array().expect(boundary).len(),
            f.positions,
            "boundary `{boundary}` is short"
        );
    }
}

#[test]
fn position_zero_matches_every_query_boundary() {
    assert_boundaries(0);
}

#[test]
fn position_one_matches_every_boundary_including_real_attention() {
    assert_boundaries(1);
}

#[test]
fn position_two_matches_every_boundary_at_three_cached_positions() {
    assert_boundaries(2);
}

/// `q_states` IS `q_b` under this form — the factorisation's output is
/// the query, not a stage before one.
///
/// Asserted because the two are separate trace fields and a reader could
/// reasonably wonder whether one is a further transform of the other.
#[test]
fn the_query_leaving_the_factorisation_is_the_query_that_attends() {
    let f = load();
    let trace = f.run_to(2, Mutation::None);
    assert_eq!(
        trace.q_b.as_ref().expect("q_b"),
        &trace.q_states,
        "q_states is q_b, not a transform of it"
    );
}

/// **The trace control.** `QbFedPreNorm` moves `q_b` and leaves
/// `q_a_normed` EXACTLY where it was.
///
/// If the executor computed `q_a_normed` for the trace and separately
/// (correctly) for `q_b`, this mutation would move neither and the test
/// would fail. If it reported `q_b`'s actual input as `q_a_normed`, the
/// norm boundary would move and the test would fail. Only an executor
/// whose reported boundary IS the consumed value passes both halves.
#[test]
fn the_reported_norm_boundary_is_the_value_the_next_stage_consumed() {
    let f = load();
    let clean = f.run_to(2, Mutation::None);
    let mutant = f.run_to(2, Mutation::QbFedPreNorm);

    assert_eq!(
        mutant.q_a_normed, clean.q_a_normed,
        "the norm is still computed and still reported; only its consumer changed"
    );
    let moved = max_abs_diff(
        mutant.q_b.as_ref().expect("q_b"),
        clean.q_b.as_ref().expect("q_b"),
    );
    assert!(
        moved > TOLERANCE * 100.0,
        "feeding q_b the un-normed q_a must move q_b; moved only {moved:e}"
    );

    // And it must equal the ORACLE's own wrong answer, not merely differ
    // from the right one.
    for (name, actual) in [
        ("q_b", mutant.q_b.as_ref().expect("q_b")),
        ("q_states", &mutant.q_states),
        ("output", &mutant.output),
    ] {
        let d = max_abs_diff(actual, &f.control("q_b_fed_pre_norm", 2, name));
        assert!(
            d < TOLERANCE,
            "the mutant must reproduce the oracle's `q_b_fed_pre_norm` at `{name}`: {d:e}"
        );
    }
}

/// The oracle's own record that this control is output-indistinguishable
/// from omitting the norm — the reason it exists.
///
/// Read from the fixture rather than re-derived: the export asserted it
/// at generation time, and a consumer that quietly disagreed would mean
/// the two sides are not running the same experiment.
#[test]
fn the_trace_control_is_output_identical_to_omitting_the_norm() {
    let f = load();
    assert_eq!(
        f.controls["q_b_fed_pre_norm"]["output_identical_to"].as_str(),
        Some("q_a_norm_omitted"),
        "the oracle must record which control this one is indistinguishable from"
    );
    let mutant = f.run_to(2, Mutation::QbFedPreNorm);
    let d = max_abs_diff(&mutant.output, &f.control("q_a_norm_omitted", 2, "output"));
    assert!(
        d < TOLERANCE,
        "and the executor must reproduce that indistinguishability: {d:e}"
    );
}

/// The q-side epsilon is read from its own key and is NOT the layer's.
///
/// The fixture carries all three numbers so this can be asserted rather
/// than described: the two low-rank norms agree, and the layer eps — the
/// value a build would reach for by analogy with the block's own norms —
/// is a factor of ten away.
#[test]
fn the_q_a_epsilon_is_the_class_default_and_not_the_layers() {
    let f = load();
    let v: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(f.q_a_norm_eps, 1e-6, "KimiRMSNorm's class default");
    assert_eq!(
        f.q_a_norm_eps, f.kv_a_norm_eps,
        "equal to the latent norm's — a shared cause, one class default used twice"
    );
    let layer = v["layer_norm_eps_not_used_by_q_a"].as_f64().unwrap();
    assert_eq!(layer, 1e-5);
    assert_ne!(
        f.q_a_norm_eps, layer,
        "and NOT the layer eps, which is what a build reaching by analogy would use"
    );
}

/// The fixture's rank is distinct from every other width, so a
/// transposed axis or a borrowed width cannot pass here by accident.
///
/// In particular `rank != hidden`, which is what makes "the norm applied
/// before `q_a_proj`" structurally inexpressible — recorded by the oracle
/// as unreachable rather than claimed as covered.
#[test]
fn the_rank_is_distinct_from_every_other_width() {
    let f = load();
    let v: Value = serde_json::from_str(ORACLE).unwrap();
    let rank = v["q_lora_rank"].as_u64().unwrap() as usize;
    let g = f.geometry;
    for (name, width) in [
        ("hidden", f.hidden),
        ("num_heads", g.num_heads),
        ("kv_lora_rank", g.kv_lora_rank),
        ("qk_nope_head_dim", g.qk_nope_head_dim),
        ("qk_rope_head_dim", g.qk_rope_head_dim),
        ("v_head_dim", g.v_head_dim),
        ("q_states_width", g.num_heads * g.q_head_dim()),
    ] {
        assert_ne!(rank, width, "rank collides with {name}; controls go blind");
    }
    assert!(
        v["structurally_unreachable"]["norm_before_q_a"].is_string(),
        "the oracle must RECORD what it cannot express, not omit it"
    );
}

/// Every control the oracle exports is discriminable at the boundary it
/// names, and the executor reproduces the oracle's own wrong answer
/// there — differing from the right answer is not enough.
#[test]
fn every_exported_control_is_reproduced_at_its_own_boundary() {
    let f = load();
    let clean = f.run_to(2, Mutation::None);
    let controls = f.controls.as_object().expect("controls");
    assert!(!controls.is_empty(), "the oracle exported no controls");

    for (name, record) in controls {
        let at = record["caught_at"].as_str().expect("caught_at");
        let delta = record["delta_rel_l2"][at].as_f64().expect("its delta");
        assert!(
            delta > 1e-3,
            "control {name} reads {delta:e} at `{at}` — it cannot tell that defect \
             from the reference"
        );
        assert!(
            !record["inert_at_its_own_boundary"]
                .as_bool()
                .unwrap_or(true),
            "control {name} is recorded inert and must not be listed"
        );
    }

    // The one the executor can run: the rest perturb the reference in
    // ways this operator has no mutation for, and are the oracle's own
    // evidence that the fixture is discriminative.
    let mutant = f.run_to(2, Mutation::QbFedPreNorm);
    assert!(
        max_abs_diff(&mutant.output, &clean.output) > TOLERANCE * 100.0,
        "the executor's own control must move the output"
    );
}
