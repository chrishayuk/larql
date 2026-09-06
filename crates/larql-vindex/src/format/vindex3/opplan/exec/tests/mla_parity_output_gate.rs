//! MLA execution with Kimi-K3's OUTPUT GATE, against its own pinned oracle
//! arm (K3-REP-GATE-1) — [`mla_parity.rs`](super::mla_parity)'s role with
//! the declared gate in the ladder.
//!
//! `kimi_mla_oracle_output_gate.json` is `kimi_mla_oracle.json`'s twin:
//! the same geometry, seed, input and every operand, plus one `g_proj` of
//! `[Hq·v_head_dim, hidden]` read from the block input, `sigmoid`, and
//! multiplied into the aggregated value before `o_proj`
//! (`modeling_kimi_linear.py` L398-401, L470-472). Every boundary up to
//! `attn_value` is bit-identical between the arms; the gated arm adds
//! `output_gate` and `gated_value`, and `output` changes.
//!
//! `hidden` (7) is not `Hq·v_head_dim` (10) on purpose: a gate applied
//! after `o_proj` cannot even be expressed at this geometry, so that
//! placement defect is recorded as unreachable rather than claimed caught.
//!
//! The band is asserted on the executor's own pre-activations before any
//! value is compared (freeze D10). The executor keeps the post-sigmoid
//! gate in its trace, so the raw pre-activation is read through
//! [`Mutation::SigmoidOmitted`] — the same projection, un-squashed.
use larql_models::config::MlaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::mla::{
    mla_forward, MlaQueryWeights, MlaState, MlaTrace, MlaWeights, Mutation,
};

const ORACLE: &str = include_str!("kimi_mla_oracle_output_gate.json");
/// The ungated arm, for the cross-arm control.
const UNGATED_ORACLE: &str = include_str!("kimi_mla_oracle.json");

/// Same bound as `mla_parity.rs`: f32 transcription noise.
const TOLERANCE: f32 = 2e-5;
/// A control must move its named boundary by at least this.
const CONTROL_FLOOR: f32 = 1e-3;

fn floats(node: &Value) -> Vec<f32> {
    node.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

struct Fixture {
    geometry: MlaGeometry,
    hidden: usize,
    kv_a_norm_eps: f64,
    weights: std::collections::BTreeMap<String, Vec<f32>>,
    input: Vec<Vec<f32>>,
    boundaries: Value,
    positions: usize,
    band: Value,
    controls: Value,
}

fn load_from(text: &str) -> Fixture {
    let v: Value = serde_json::from_str(text).expect("oracle fixture parses");
    let weights = v["weights"]
        .as_object()
        .expect("weights")
        .iter()
        .map(|(k, node)| (k.clone(), floats(node)))
        .collect();
    let positions = v["positions"].as_u64().unwrap() as usize;
    let input = v["input"].as_array().unwrap().iter().map(floats).collect();
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
        weights,
        input,
        boundaries: v["boundaries"].clone(),
        positions,
        band: v["gate_band"].clone(),
        controls: v["controls"].clone(),
    }
}

fn load() -> Fixture {
    load_from(ORACLE)
}

impl Fixture {
    fn weights(&self, gated: bool) -> MlaWeights<'_> {
        let g = |n: &str| self.weights.get(n).expect(n).as_slice();
        MlaWeights {
            query: MlaQueryWeights::Direct {
                q_proj: WeightRows::F32(g("q_proj")),
            },
            kv_a_proj: WeightRows::F32(g("kv_a_proj")),
            kv_a_norm: g("kv_a_norm"),
            kv_b_proj: WeightRows::F32(g("kv_b_proj")),
            o_proj: WeightRows::F32(g("o_proj")),
            kv_a_norm_eps: self.kv_a_norm_eps,
            output_gate: gated.then(|| WeightRows::F32(g("g_proj"))),
        }
    }

    /// Every position 0..=up_to through a FRESH state, threaded like a
    /// real decode; the trace at `up_to`.
    fn run_to(&self, up_to: usize, mutation: Mutation) -> MlaTrace {
        let mut state = MlaState::default();
        let mut last = None;
        for p in 0..=up_to {
            last = Some(mla_forward(
                &self.input[p],
                self.hidden,
                self.weights(true),
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

    fn control(&self, name: &str, boundary: &str, p: usize) -> Vec<f32> {
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

/// The band on the executor's OWN pre-activations (freeze D10), over
/// every position: nothing past the ceiling, every head at the floor.
fn assert_band(f: &Fixture) {
    let ceiling = f.band["limits"]["max_abs"].as_f64().unwrap() as f32;
    let floor = f.band["limits"]["min_head_max_abs"].as_f64().unwrap() as f32;
    let v_dim = f.geometry.v_head_dim;
    let mut per_head = vec![0.0f32; f.geometry.num_heads];
    for p in 0..f.positions {
        let raw = f
            .run_to(p, Mutation::SigmoidOmitted)
            .output_gate
            .expect("a gated layer emits its gate");
        for (h, m) in per_head.iter_mut().enumerate() {
            let head_max = raw[h * v_dim..(h + 1) * v_dim]
                .iter()
                .fold(0.0f32, |acc, g| acc.max(g.abs()));
            assert!(
                head_max <= ceiling,
                "position {p} head {h}: |g| {head_max} > {ceiling}"
            );
            *m = m.max(head_max);
        }
    }
    for (h, m) in per_head.iter().enumerate() {
        assert!(
            *m >= floor,
            "head {h} gate near-constant: max |g| {m} < {floor}"
        );
    }
}

fn assert_boundaries(up_to: usize) {
    let f = load();
    assert_band(&f);
    let trace = f.run_to(up_to, Mutation::None);
    let output_gate = trace.output_gate.as_ref().expect("gated");
    let gated_value = trace.gated_value.as_ref().expect("gated");
    let got: [(&str, &Vec<f32>); 9] = [
        // The fixture's key is `q_states` (see `mla_parity.rs`); the Rust
        // field is still `q_proj` because this executor still has only
        // the direct query form.
        ("q_states", &trace.q_states),
        ("compressed_kv", &trace.compressed_kv),
        ("kv_a_normed", &trace.kv_a_normed),
        ("kv_b", &trace.kv_b),
        ("attn_weights", &trace.attn_weights),
        ("attn_value", &trace.attn_value),
        ("output_gate", output_gate),
        ("gated_value", gated_value),
        ("output", &trace.output),
    ];
    for (name, actual) in got {
        let d = max_abs_diff(actual, &f.expected(up_to, name));
        assert!(
            d < TOLERANCE,
            "position {up_to} boundary `{name}`: max|Δ| {d:e}"
        );
    }
}

#[test]
fn the_fixture_declares_the_gate_at_a_width_that_is_not_hidden() {
    let v: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(v["output_gate"], true);
    let f = load();
    let width = f.geometry.num_heads * f.geometry.v_head_dim;
    assert_eq!(
        f.weights["g_proj"].len(),
        width * f.hidden,
        "[Hq·v_head_dim, hidden]"
    );
    assert_ne!(
        width, f.hidden,
        "a gate after o_proj must be structurally impossible here"
    );
}

#[test]
fn position_zero_matches_every_boundary_including_the_gate() {
    assert_boundaries(0);
}

#[test]
fn position_one_matches_every_boundary_including_the_gate() {
    assert_boundaries(1);
}

#[test]
fn position_two_matches_every_boundary_including_the_gate() {
    assert_boundaries(2);
}

/// **Only the gate is new.** Everything up to `attn_value` is bit-identical
/// between the gated and ungated arms — read from the two JSON files — and
/// `output` differs at every position.
#[test]
fn everything_before_the_gate_is_identical_to_the_ungated_arm() {
    let gated: Value = serde_json::from_str(ORACLE).unwrap();
    let plain: Value = serde_json::from_str(UNGATED_ORACLE).unwrap();
    assert_eq!(gated["input"], plain["input"]);
    for shared in ["q_proj", "kv_a_proj", "kv_a_norm", "kv_b_proj", "o_proj"] {
        assert_eq!(
            gated["weights"][shared], plain["weights"][shared],
            "operand `{shared}`"
        );
    }
    for before in [
        "q_states",
        "compressed_kv",
        "kv_a_normed",
        "kv_b",
        "attn_weights",
        "attn_value",
    ] {
        assert_eq!(
            gated["boundaries"][before], plain["boundaries"][before],
            "boundary `{before}` must not depend on the gate"
        );
    }
    let positions = gated["positions"].as_u64().unwrap() as usize;
    for p in 0..positions {
        let d = max_abs_diff(
            &floats(&gated["boundaries"]["output"][p]),
            &floats(&plain["boundaries"]["output"][p]),
        );
        assert!(
            d > CONTROL_FLOOR,
            "position {p}: output must differ under the gate: {d:e}"
        );
    }
}

/// An ungated layer through the same executor emits NO gate boundaries and
/// reproduces the ungated arm — `None` is a fact, not an empty vector.
#[test]
fn an_ungated_layer_emits_no_gate_and_matches_the_ungated_arm() {
    let f = load_from(UNGATED_ORACLE);
    let mut state = MlaState::default();
    let mut last = None;
    for p in 0..f.positions {
        last = Some(mla_forward(
            &f.input[p],
            f.hidden,
            f.weights(false),
            f.geometry,
            &mut state,
            Mutation::None,
        ));
    }
    let trace = last.unwrap();
    assert!(trace.output_gate.is_none() && trace.gated_value.is_none());
    let d = max_abs_diff(&trace.output, &f.expected(f.positions - 1, "output"));
    assert!(d < TOLERANCE, "ungated output: max|Δ| {d:e}");
}

// ── controls ────────────────────────────────────────────────────────
//
// Run at the last position (three cached positions). Each control must
// move the boundary named for it, leave the earlier one exactly alone
// where it does not touch it, and EQUAL the oracle's own wrong answer.

fn control(mutation: Mutation, name: &str, moves_output_gate: bool) {
    let f = load();
    assert_eq!(
        f.controls[name]["inert_on_this_fixture"], false,
        "the exporter found `{name}` inert; it cannot be a control here"
    );
    let p = f.positions - 1;
    let base = f.run_to(p, Mutation::None);
    let mutant = f.run_to(p, mutation);
    let (base_gate, base_value) = (base.output_gate.unwrap(), base.gated_value.unwrap());
    let (mut_gate, mut_value) = (
        mutant.output_gate.clone().unwrap(),
        mutant.gated_value.clone().unwrap(),
    );
    let d_gate = max_abs_diff(&base_gate, &mut_gate);
    if moves_output_gate {
        assert!(d_gate > CONTROL_FLOOR, "{name}: output_gate Δ {d_gate:e}");
    } else {
        assert_eq!(d_gate, 0.0, "{name}: must not touch the gate itself");
    }
    let d_value = max_abs_diff(&base_value, &mut_value);
    assert!(d_value > CONTROL_FLOOR, "{name}: gated_value Δ {d_value:e}");
    let d_out = max_abs_diff(&base.output, &mutant.output);
    assert!(d_out > CONTROL_FLOOR, "{name}: output Δ {d_out:e}");
    for (boundary, against) in [
        ("output_gate", &mut_gate),
        ("gated_value", &mut_value),
        ("output", &mutant.output),
    ] {
        let d = max_abs_diff(against, &f.control(name, boundary, p));
        assert!(
            d < TOLERANCE,
            "{name}: the Rust mutant is not the oracle's defect at `{boundary}`: max|Δ| {d:e}"
        );
    }
}

#[test]
fn omitting_the_gate_is_caught_at_the_gated_value() {
    control(Mutation::GateOmitted, "gate_omitted", false);
}

#[test]
fn omitting_the_sigmoid_is_caught_at_the_gate() {
    control(Mutation::SigmoidOmitted, "sigmoid_omitted", true);
}

/// The placement control — every cached position's value gated by its
/// OWN gate before the weighted sum — is measured by the oracle
/// (`gate_on_values_before_aggregation`, a real delta on this fixture)
/// but has no executor arm: this executor caches the compressed LATENT
/// and decompresses each position's value at read time, so it holds no
/// per-position gate to apply and cannot express the defect. Recorded
/// here as a fact about the reference's placement claim, never as a
/// mutant the executor caught.
#[test]
fn the_placement_control_is_measured_by_the_oracle_and_unreachable_here() {
    let f = load();
    let c = &f.controls["gate_on_values_before_aggregation"];
    assert_eq!(c["inert_on_this_fixture"], false);
    assert!(c["delta_rel_l2"]["output"].as_f64().unwrap() > CONTROL_FLOOR as f64);
    assert_eq!(c["delta_rel_l2"]["output_gate"].as_f64().unwrap(), 0.0);
}
