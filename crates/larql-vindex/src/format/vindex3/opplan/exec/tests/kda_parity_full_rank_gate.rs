//! KDA execution with Kimi-K3's FULL-RANK output gate, against its own
//! pinned oracle arm (K3-REP-GATE-1) — [`kda_parity.rs`](super::kda_parity)'s
//! role for the OTHER declared form of the same gate.
//!
//! `kda_oracle_full_rank_gate.json` is `kda_oracle.json`'s twin: the same
//! geometry, seed, input and every non-gate operand bit for bit, with one
//! `g_proj` of `[H·D, hidden]` in place of the `g_a_proj`/`g_b_proj` pair
//! (`modeling_kimi_linear.py` L531-537, L651-654). Only the gate's
//! PROJECTION differs between the two arms — the sigmoid, the gated RMS
//! norm and `o_proj` are the same code — and the cross-arm test below is
//! what makes that a measured fact rather than a description.
//!
//! The band is asserted on the executor's OWN emitted gate before any
//! value is compared (freeze D10): a saturated sigmoid blinds every gate
//! control, so a green read outside the band would report nothing.
//!
//! Each control is held to two things: it must MOVE the boundary named
//! for it, and it must EQUAL the oracle's own wrong answer for the same
//! defect — so a Rust mutant is the same defect the reference measured,
//! not merely "something different".
use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::continuation::RecurrentState;
use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda;
use crate::format::vindex3::opplan::exec::kda::{
    layer_forward, zero_state, KdaOutputGateWeights, KdaPlanes, KdaWeights, Mutation,
};

const ORACLE: &str = include_str!("kda_oracle_full_rank_gate.json");
/// The low-rank arm, for the cross-arm control: everything before the
/// gate must be IDENTICAL between the two fixtures.
const LOW_RANK_ORACLE: &str = include_str!("kda_oracle.json");

/// Same bound as `kda_parity.rs`, for the same reason: q/k/v/o_proj are
/// BF16 and their raw boundaries carry genuine bf16 quantisation error.
const TOLERANCE: f32 = 3e-3;
/// A control must move its named boundary by at least this — comfortably
/// above the transcription tolerance, far below every measured delta.
const CONTROL_FLOOR: f32 = 1e-3;
/// The longest run, where every control has room to act.
const N: usize = 8;

fn bf16_from_f32(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn floats(node: &Value) -> Vec<f32> {
    node.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

struct Fixture {
    geometry: KdaGeometry,
    hidden: usize,
    eps: f32,
    weights: std::collections::BTreeMap<String, Vec<f32>>,
    bf16: std::collections::BTreeMap<String, Vec<u16>>,
    runs: Value,
    band: Value,
    controls: Value,
}

fn load() -> Fixture {
    let v: Value = serde_json::from_str(ORACLE).expect("oracle fixture parses");
    let weights: std::collections::BTreeMap<String, Vec<f32>> = v["weights"]
        .as_object()
        .expect("weights")
        .iter()
        .map(|(k, node)| (k.clone(), floats(node)))
        .collect();
    let bf16 = ["q_proj", "k_proj", "v_proj", "o_proj"]
        .iter()
        .map(|&name| {
            let codes: Vec<u16> = weights[name].iter().map(|&f| bf16_from_f32(f)).collect();
            (name.to_string(), codes)
        })
        .collect();
    Fixture {
        geometry: KdaGeometry {
            num_heads: v["num_heads"].as_u64().unwrap() as usize,
            head_dim: v["head_dim"].as_u64().unwrap() as usize,
            conv_kernel: v["conv_kernel"].as_u64().unwrap() as usize,
        },
        hidden: v["hidden"].as_u64().unwrap() as usize,
        eps: v["rms_eps"].as_f64().unwrap() as f32,
        weights,
        bf16,
        runs: v["runs"].clone(),
        band: v["gate_band"].clone(),
        controls: v["controls"].clone(),
    }
}

impl Fixture {
    fn weights(&self) -> KdaWeights<'_> {
        let g = |n: &str| self.weights.get(n).expect(n).as_slice();
        let b = |n: &str| self.bf16.get(n).expect(n).as_slice();
        KdaWeights {
            q_proj: WeightRows::Bf16(b("q_proj")),
            k_proj: WeightRows::Bf16(b("k_proj")),
            v_proj: WeightRows::Bf16(b("v_proj")),
            q_conv1d: g("q_conv1d"),
            k_conv1d: g("k_conv1d"),
            v_conv1d: g("v_conv1d"),
            f_a_proj: g("f_a_proj"),
            f_b_proj: g("f_b_proj"),
            // The declared form under test: one projection, no pair.
            output_gate: KdaOutputGateWeights::FullRank {
                g_proj: WeightRows::F32(g("g_proj")),
            },
            b_proj: g("b_proj"),
            a_log: g("a_log"),
            dt_bias: g("dt_bias"),
            o_norm: g("o_norm"),
            o_proj: WeightRows::Bf16(b("o_proj")),
            norm_eps: self.eps,
            // The f gate's rank, unchanged by the output gate's form.
            gate_rank: self.weights.get("f_a_proj").expect("f_a_proj").len() / self.hidden,
        }
    }

    fn run(&self, n: usize, mutation: Mutation) -> (KdaPlanes, RecurrentState) {
        let x = floats(&self.runs[n.to_string()]["input"]);
        let mut state = zero_state(self.geometry);
        let planes = layer_forward(
            &x,
            self.hidden,
            self.weights(),
            self.geometry,
            &mut state,
            mutation,
        );
        (planes, state)
    }

    fn expected(&self, n: usize, boundary: &str) -> Vec<f32> {
        floats(&self.runs[n.to_string()]["boundaries"][boundary])
    }

    fn expected_state(&self, n: usize) -> Vec<f32> {
        floats(&self.runs[n.to_string()]["state"])
    }

    /// The oracle's own wrong answer for a named defect, on the full run.
    fn control(&self, name: &str, boundary: &str) -> Vec<f32> {
        floats(&self.controls[name]["boundaries"][boundary])
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The non-saturation band (freeze D10), on the EXECUTOR's emitted gate:
/// no pre-activation past the ceiling, and every head reaching the floor
/// somewhere in the run so the gate is not a near-constant scale.
fn assert_band(f: &Fixture, planes: &KdaPlanes) {
    let (heads, dim) = (f.geometry.num_heads, f.geometry.head_dim);
    let width = heads * dim;
    let ceiling = f.band["limits"]["max_abs"].as_f64().unwrap() as f32;
    let floor = f.band["limits"]["min_head_max_abs"].as_f64().unwrap() as f32;
    let positions = planes.o_gate.len() / width;
    let max_abs = planes.o_gate.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    assert!(
        max_abs <= ceiling,
        "gate saturated: max |g| {max_abs} > {ceiling}"
    );
    for h in 0..heads {
        let head_max = (0..positions)
            .flat_map(|p| (0..dim).map(move |d| p * width + h * dim + d))
            .map(|i| planes.o_gate[i].abs())
            .fold(0.0f32, f32::max);
        assert!(
            head_max >= floor,
            "head {h} gate near-constant: max |g| {head_max} < {floor}"
        );
    }
}

fn assert_boundaries(n: usize) {
    let f = load();
    let (planes, state) = f.run(n, Mutation::None);
    if n == N {
        assert_band(&f, &planes);
    }
    let got: [(&str, &Vec<f32>); 15] = [
        ("q_proj", &planes.q_proj),
        ("k_proj", &planes.k_proj),
        ("v_proj", &planes.v_proj),
        ("q_conv", &planes.q_conv),
        ("k_conv", &planes.k_conv),
        ("v_conv", &planes.v_conv),
        ("q_norm", &planes.q_norm),
        ("k_norm", &planes.k_norm),
        ("f_lowrank", &planes.f_lowrank),
        ("g_decay", &planes.g_decay),
        ("beta", &planes.beta),
        ("recurrent_out", &planes.recurrent_out),
        ("o_gate", &planes.o_gate),
        ("o_norm", &planes.o_norm),
        ("output", &planes.output),
    ];
    for (name, actual) in got {
        let d = max_abs_diff(actual, &f.expected(n, name));
        assert!(d < TOLERANCE, "N={n} boundary `{name}`: max|Δ| {d:e}");
    }
    let d = max_abs_diff(state.buffer(kda::RECURRENT).cells(), &f.expected_state(n));
    assert!(d < TOLERANCE, "N={n} recurrent state: max|Δ| {d:e}");
}

#[test]
fn the_fixture_is_the_full_rank_arm_and_ships_no_pair() {
    let v: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(v["output_gate_form"], "full_rank");
    let f = load();
    assert!(f.weights.contains_key("g_proj"));
    assert!(!f.weights.contains_key("g_a_proj") && !f.weights.contains_key("g_b_proj"));
    let (heads, dim, hidden) = (f.geometry.num_heads, f.geometry.head_dim, f.hidden);
    assert_eq!(
        f.weights["g_proj"].len(),
        heads * dim * hidden,
        "[H·D, hidden]"
    );
    // The band was measured by the exporter before any control was scored.
    assert!(f.band["max_abs"].as_f64().unwrap() <= f.band["limits"]["max_abs"].as_f64().unwrap());
}

#[test]
fn the_executors_own_gate_lies_inside_the_band() {
    let f = load();
    let (planes, _) = f.run(N, Mutation::None);
    assert_band(&f, &planes);
}

#[test]
fn one_position_matches_every_boundary() {
    assert_boundaries(1);
}

#[test]
fn two_positions_match_every_boundary() {
    assert_boundaries(2);
}

#[test]
fn eight_positions_match_every_boundary() {
    assert_boundaries(8);
}

/// **Only the projection changes.** The two oracle arms share every
/// operand and the input up to the gate, so every boundary BEFORE the
/// gate is bit-identical between them, and the three at or after it are
/// not. Read from the two JSON files directly: this is a fact about the
/// fixtures that the executor is then held to on both sides.
#[test]
fn everything_before_the_gate_is_identical_to_the_low_rank_arm() {
    let full: Value = serde_json::from_str(ORACLE).unwrap();
    let low: Value = serde_json::from_str(LOW_RANK_ORACLE).unwrap();
    let run = N.to_string();
    assert_eq!(full["runs"][&run]["input"], low["runs"][&run]["input"]);
    for shared in [
        "q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj",
        "b_proj", "a_log", "dt_bias", "o_norm", "o_proj",
    ] {
        assert_eq!(
            full["weights"][shared], low["weights"][shared],
            "operand `{shared}`"
        );
    }
    for before in [
        "q_proj",
        "k_proj",
        "v_proj",
        "q_conv",
        "k_conv",
        "v_conv",
        "q_norm",
        "k_norm",
        "f_lowrank",
        "g_decay",
        "beta",
        "recurrent_out",
    ] {
        assert_eq!(
            full["runs"][&run]["boundaries"][before], low["runs"][&run]["boundaries"][before],
            "boundary `{before}` must not depend on the gate's form"
        );
    }
    for after in ["o_gate", "o_norm", "output"] {
        let d = max_abs_diff(
            &floats(&full["runs"][&run]["boundaries"][after]),
            &floats(&low["runs"][&run]["boundaries"][after]),
        );
        assert!(
            d > CONTROL_FLOOR,
            "boundary `{after}` must differ between the forms: {d:e}"
        );
    }
}

// ── controls ────────────────────────────────────────────────────────
//
// Each perturbs the real function at one point. A control is held to the
// boundary named for it: the ones that leave the projection alone must
// leave `o_gate` EXACTLY unchanged and move `o_norm`; the one that zeroes
// the gate moves `o_gate` itself. And every Rust mutant must equal the
// oracle's own control values — the same defect, measured on both sides.

fn control(mutation: Mutation, name: &str, moves_o_gate: bool) {
    let f = load();
    assert_eq!(
        f.controls[name]["inert_on_this_fixture"], false,
        "the exporter found `{name}` inert; it cannot be a control here"
    );
    let (base, _) = f.run(N, Mutation::None);
    let (mutant, _) = f.run(N, mutation);
    let d_gate = max_abs_diff(&base.o_gate, &mutant.o_gate);
    if moves_o_gate {
        assert!(d_gate > CONTROL_FLOOR, "{name}: o_gate Δ {d_gate:e}");
    } else {
        assert_eq!(d_gate, 0.0, "{name}: must not touch the projection itself");
    }
    let d_norm = max_abs_diff(&base.o_norm, &mutant.o_norm);
    assert!(d_norm > CONTROL_FLOOR, "{name}: o_norm Δ {d_norm:e}");
    let d_out = max_abs_diff(&base.output, &mutant.output);
    assert!(d_out > CONTROL_FLOOR, "{name}: output Δ {d_out:e}");
    for boundary in ["o_gate", "o_norm", "output"] {
        let against = match boundary {
            "o_gate" => &mutant.o_gate,
            "o_norm" => &mutant.o_norm,
            _ => &mutant.output,
        };
        let d = max_abs_diff(against, &f.control(name, boundary));
        assert!(
            d < TOLERANCE,
            "{name}: the Rust mutant is not the oracle's defect at `{boundary}`: max|Δ| {d:e}"
        );
    }
}

#[test]
fn skipping_the_gate_is_caught_at_the_projection_and_the_norm() {
    control(Mutation::GateSkipped, "gate_skipped", true);
}

#[test]
fn gating_before_the_norm_is_caught_at_the_norm() {
    control(Mutation::GateBeforeNorm, "gate_before_norm", false);
}

#[test]
fn omitting_the_sigmoid_is_caught_at_the_norm() {
    control(Mutation::SigmoidOmitted, "sigmoid_omitted", false);
}

#[test]
fn gating_the_value_before_the_recurrence_is_caught_at_the_norm() {
    control(
        Mutation::GateOnValueBeforeRecurrence,
        "gate_on_value_before_recurrence",
        false,
    );
}

/// The low-rank arm's own controls still fire under the new form of the
/// weights type: the gate form changed one projection, not the
/// recurrence around it.
#[test]
fn the_recurrence_controls_still_fire_under_the_full_rank_form() {
    let f = load();
    let (base, base_state) = f.run(N, Mutation::None);
    for mutation in [Mutation::NoKNorm, Mutation::NoDecay, Mutation::NoBeta] {
        let (changed, changed_state) = f.run(N, mutation);
        assert!(
            max_abs_diff(&base.output, &changed.output) > CONTROL_FLOOR,
            "{mutation:?}"
        );
        assert!(
            max_abs_diff(
                base_state.buffer(kda::RECURRENT).cells(),
                changed_state.buffer(kda::RECURRENT).cells()
            ) > CONTROL_FLOOR,
            "{mutation:?}"
        );
    }
}
