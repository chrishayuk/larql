//! KDA execution against the pinned oracle, boundary by boundary.
//!
//! The fixture in `kda_oracle.json` is a tiny layer (2 heads × 4) built by
//! `scripts/kda_oracle_export.py` through `scripts/kda_reference.py` — a
//! transcription pinned to the call the Kimi Linear checkpoint's own
//! `modeling_kimi.py` makes, not to upstream `fla`'s current signature.
//! Small on purpose: the arithmetic is identical at any width, and a
//! fixture that fits in a repository is one that gets run.
//!
//! `q_proj`/`k_proj`/`v_proj`/`o_proj` are BF16 (P4c-4) — the fixture's own
//! Python generator rounds them before computing the oracle, so this loads
//! bf16-exact JSON floats and truncates them back to `u16` code units
//! ([`bf16_from_f32`]), losslessly, the same way a real checkpoint's
//! already-bf16 tensors round-trip.
//!
//! Comparison is per boundary, not on the output. A recurrence wrong in
//! one factor still produces a plausible final tensor, and the point of a
//! ladder is to name which stage moved.

use larql_models::config::KdaGateForm;
use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::continuation::RecurrentState;
use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda;
use crate::format::vindex3::opplan::exec::kda::{
    layer_forward, zero_state, KdaOutputGateWeights, KdaPlanes, KdaWeights, Mutation,
};

const ORACLE: &str = include_str!("kda_oracle.json");

/// f32 transcription noise between two implementations of the same
/// arithmetic in different orders, WIDENED from `2e-5` at P4c-4: q/k/v/
/// o_proj are now BF16, so `q_proj`'s own raw boundary carries genuine
/// bf16 quantisation error (~2e-4 measured). At this fixture's
/// deliberately tiny `hidden=6`/`head_dim=4`, the L2 normalisation
/// AMPLIFIES that error rather than averaging it out — a handful of
/// terms in both the numerator and the norm itself, so one rounded
/// element moves a larger fraction of both — measured worst case
/// `q_norm` at N=8: ~1.04e-3. Still comfortably below every control's
/// effect below (each moves things by `1e-3` or more, several by two to
/// three orders more).
const TOLERANCE: f32 = 3e-3;

/// The top 16 bits of an IEEE754 f32 — lossless truncation ONLY when the
/// value already has zero in its low mantissa bits, which the fixture's
/// own `bf16_exact()` round-trip on the Python side guarantees for
/// q/k/v/o_proj.
fn bf16_from_f32(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

struct Fixture {
    geometry: KdaGeometry,
    hidden: usize,
    eps: f32,
    weights: std::collections::BTreeMap<String, Vec<f32>>,
    /// q/k/v/o_proj only, pre-converted once at load — `KdaWeights`
    /// borrows these BF16-compact code units (P4c-4), so they must
    /// outlive every `weights()` call, not be rebuilt inside it.
    bf16: std::collections::BTreeMap<String, Vec<u16>>,
    runs: Value,
}

fn load() -> Fixture {
    let v: Value = serde_json::from_str(ORACLE).expect("oracle fixture parses");
    let floats = |node: &Value| -> Vec<f32> {
        node.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number") as f32)
            .collect()
    };
    let weights: std::collections::BTreeMap<String, Vec<f32>> = v["weights"]
        .as_object()
        .expect("weights")
        .iter()
        .map(|(k, node)| (k.clone(), floats(node)))
        .collect();
    let bf16 = ["q_proj", "k_proj", "v_proj", "o_proj"]
        .iter()
        .map(|&name| {
            let v: Vec<u16> = weights[name].iter().map(|&f| bf16_from_f32(f)).collect();
            (name.to_string(), v)
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
    }
}

impl Fixture {
    /// The same operands under a DECLARED gate form, so the form can be
    /// varied without varying anything else.
    fn weights_as(&self, gate_form: KdaGateForm) -> KdaWeights<'_> {
        let g = |n: &str| self.weights.get(n).expect(n).as_slice();
        let b = |n: &str| self.bf16.get(n).expect(n).as_slice();
        KdaWeights {
            gate_form,
            q_proj: WeightRows::Bf16(b("q_proj")),
            k_proj: WeightRows::Bf16(b("k_proj")),
            v_proj: WeightRows::Bf16(b("v_proj")),
            q_conv1d: g("q_conv1d"),
            k_conv1d: g("k_conv1d"),
            v_conv1d: g("v_conv1d"),
            f_a_proj: g("f_a_proj"),
            f_b_proj: g("f_b_proj"),
            output_gate: KdaOutputGateWeights::LowRank {
                g_a_proj: g("g_a_proj"),
                g_b_proj: g("g_b_proj"),
            },
            b_proj: g("b_proj"),
            a_log: g("a_log"),
            dt_bias: g("dt_bias"),
            o_norm: g("o_norm"),
            o_proj: WeightRows::Bf16(b("o_proj")),
            norm_eps: self.eps,
            // The rank the two gate factorisations meet at, read from this
            // fixture's own `f_a_proj` rather than assumed equal to the head
            // dim: on this checkpoint the two coincide, and the executor no
            // longer takes that coincidence as its definition.
            gate_rank: self.weights.get("f_a_proj").expect("f_a_proj").len() / self.hidden,
        }
    }

    /// This fixture's oracle is Kimi Linear, whose reference reads
    /// `gate_lower_bound` nowhere — so the parity runs and every control
    /// but the mirror declare [`KdaGateForm::Softplus`].
    fn run(&self, n: usize, mutation: Mutation) -> (KdaPlanes, RecurrentState) {
        self.run_as(n, mutation, KdaGateForm::Softplus)
    }

    fn run_as(
        &self,
        n: usize,
        mutation: Mutation,
        gate_form: KdaGateForm,
    ) -> (KdaPlanes, RecurrentState) {
        let run = &self.runs[n.to_string()];
        let x: Vec<f32> = run["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let mut state = zero_state(self.geometry);
        let planes = layer_forward(
            &x,
            self.hidden,
            self.weights_as(gate_form),
            self.geometry,
            &mut state,
            mutation,
        );
        (planes, state)
    }

    fn expected(&self, n: usize, boundary: &str) -> Vec<f32> {
        self.runs[n.to_string()]["boundaries"][boundary]
            .as_array()
            .unwrap_or_else(|| panic!("no boundary `{boundary}`"))
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }

    fn expected_state(&self, n: usize) -> Vec<f32> {
        self.runs[n.to_string()]["state"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_boundaries(n: usize) {
    let f = load();
    let (planes, state) = f.run(n, Mutation::None);
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
    // The state is the gate, not the output: an implementation can match
    // every token and still carry the wrong thing forward.
    let d = max_abs_diff(state.buffer(kda::RECURRENT).cells(), &f.expected_state(n));
    assert!(d < TOLERANCE, "N={n} recurrent state: max|Δ| {d:e}");
}

#[test]
fn one_position_matches_every_boundary() {
    assert_boundaries(1);
}

/// If `N=1` passes and `N=2` fails, the defect is state carry rather than
/// the recurrence formula — the two run the same arithmetic and differ
/// only in whether a second step reads what the first wrote.
#[test]
fn two_positions_match_every_boundary() {
    assert_boundaries(2);
}

#[test]
fn eight_positions_match_every_boundary() {
    assert_boundaries(8);
}

// ── controls ────────────────────────────────────────────────────────
//
// Each perturbs the real function. A fixture proves nothing until the
// things it exists to catch actually break it.

fn control(mutation: Mutation) -> (f32, f32) {
    const N: usize = 8;
    let f = load();
    let (base, base_state) = f.run(N, Mutation::None);
    let (changed, changed_state) = f.run(N, mutation);
    (
        max_abs_diff(&base.output, &changed.output),
        max_abs_diff(
            base_state.buffer(kda::RECURRENT).cells(),
            changed_state.buffer(kda::RECURRENT).cells(),
        ),
    )
}

/// **`gate_lower_bound` is provenance, executably.** Applying the declared
/// clamp changes the result, so the decision not to apply it is a real one
/// — and this is the test that argues with anyone who later wires it in.
#[test]
fn applying_the_declared_decay_clamp_changes_the_result() {
    let (out, state) = control(Mutation::ApplyGateLowerBound(-5.0));
    assert!(out > 1e-2, "output Δ {out:e}");
    assert!(state > 1e-2, "state Δ {state:e}");
}

/// **The mirror, and the half that matters for GLM-5.3-Flash.**
///
/// The control above proves the clamp is *not* applied on a checkpoint
/// declaring [`KdaGateForm::Softplus`]. This one starts from a checkpoint
/// declaring [`KdaGateForm::ClampedSigmoid`] — GLM's form, with GLM's
/// declared `-5.0` — and forces the softplus branch, which is exactly what
/// running the Kimi-shaped executor unchanged on GLM would compute.
///
/// Both directions are asserted because a one-sided control cannot
/// distinguish "the form is declared and honoured" from "the form is
/// declared and ignored in favour of the one hard-coded here". Only the
/// pair shows that the declaration is what the executor reads.
///
/// Measured on the real GLM layer 0 against the pinned `transformers`
/// reference (`scripts/glm_gate_control.py`), the same swap moves the
/// layer output by relative 2.50e-2 and the decay gate's mean from -0.906
/// to -2.528.
#[test]
fn forcing_softplus_on_a_clamped_checkpoint_changes_the_result() {
    const N: usize = 8;
    let f = load();
    let clamped = KdaGateForm::ClampedSigmoid { lower_bound: -5.0 };
    let (base, base_state) = f.run_as(N, Mutation::None, clamped);
    let (changed, changed_state) = f.run_as(N, Mutation::ForceSoftplusGate, clamped);
    let out = max_abs_diff(&base.output, &changed.output);
    let state = max_abs_diff(
        base_state.buffer(kda::RECURRENT).cells(),
        changed_state.buffer(kda::RECURRENT).cells(),
    );
    assert!(out > 1e-2, "output Δ {out:e}");
    assert!(state > 1e-2, "state Δ {state:e}");
}

/// The declared form is what the executor reads — nothing else in the
/// operand set changes with it.
///
/// Without this, the control above could pass simply because
/// `ForceSoftplusGate` perturbs *something*; this shows the two forms are
/// genuinely different functions of the same operands, and that declaring
/// the form (with no mutation at all) already selects between them.
#[test]
fn the_declared_gate_form_alone_selects_the_arithmetic() {
    const N: usize = 8;
    let f = load();
    let (soft, _) = f.run_as(N, Mutation::None, KdaGateForm::Softplus);
    let (clamped, _) = f.run_as(
        N,
        Mutation::None,
        KdaGateForm::ClampedSigmoid { lower_bound: -5.0 },
    );
    let delta = max_abs_diff(&soft.output, &clamped.output);
    assert!(
        delta > 1e-2,
        "declared form must change the output, Δ {delta:e}"
    );

    // And the identity arm: the same declared form twice must agree
    // exactly, so a non-zero above is the form and not run-to-run noise.
    let (again, _) = f.run_as(N, Mutation::None, KdaGateForm::Softplus);
    assert_eq!(
        max_abs_diff(&soft.output, &again.output),
        0.0,
        "the same declared form must be bit-identical across runs"
    );
}

/// The q normalisation moves the output and leaves the state **exactly**
/// untouched: q is read-only against the recurrence.
///
/// The zero is the useful half. It makes the fault-localisation rule
/// executable — a disagreement that moves the state cannot be in the q
/// path — rather than a note someone has to remember.
#[test]
fn omitting_the_query_normalisation_moves_the_output_but_not_the_state() {
    let (out, state) = control(Mutation::NoQNorm);
    assert!(out > 1e-2, "output Δ {out:e}");
    assert_eq!(state, 0.0, "q must not touch the recurrent state");
}

#[test]
fn omitting_the_key_normalisation_moves_both() {
    let (out, state) = control(Mutation::NoKNorm);
    assert!(out > 1e-2, "output Δ {out:e}");
    assert!(state > 1e-2, "state Δ {state:e}");
}

/// The recurrence is kept in f32. Rounding the state to bf16 each step
/// measurably diverges, so an "optimisation" that drops the promotion is
/// caught rather than merely disapproved of in a comment.
#[test]
fn a_bf16_recurrent_state_diverges() {
    let (out, state) = control(Mutation::Bf16Recurrence);
    assert!(out > 1e-4, "output Δ {out:e}");
    assert!(state > 1e-4, "state Δ {state:e}");
}

/// Writing `v` instead of the prediction error `v - kᵀS` is the most
/// plausible wrong transcription of a delta rule — and it agrees at one
/// position from a zero state, which is why the ladder does not stop at
/// `N = 1`.
#[test]
fn writing_the_value_instead_of_the_error_is_caught_only_past_one_position() {
    let f = load();
    let (base_1, _) = f.run(1, Mutation::None);
    let (wrong_1, _) = f.run(1, Mutation::WriteValueNotError);
    assert!(
        max_abs_diff(&base_1.output, &wrong_1.output) < TOLERANCE,
        "at one position from a zero state the two rules agree — that is the trap"
    );
    let (out, state) = control(Mutation::WriteValueNotError);
    // Comfortably above the 2e-5 transcription tolerance without pinning
    // the exact magnitude, which is a property of these fixture weights
    // rather than of the defect.
    assert!(
        out > 1e-3,
        "by 8 positions it must be caught: output Δ {out:e}"
    );
    assert!(state > 1e-3, "state Δ {state:e}");
}

#[test]
fn the_remaining_recurrence_controls_all_fire() {
    for mutation in [
        Mutation::ReadBeforeWrite,
        Mutation::NoDecay,
        Mutation::NoBeta,
    ] {
        let (out, state) = control(mutation);
        assert!(out > 1e-3, "{mutation:?}: output Δ {out:e}");
        let _ = state;
    }
}

/// **Genericity, without weights.** The same executor accepts GLM-5.3-Flash's
/// geometry — 64 heads × 128 against Kimi Linear's 32 × 128 — with no
/// family branch and no width constant. Construction only: GLM's weights
/// are not downloaded, and this is the rung where a width assumption would
/// first bite.
#[test]
fn the_executor_is_generic_over_both_observed_geometries() {
    for (name, heads, dim) in [("Kimi Linear", 32, 128), ("GLM-5.3-Flash", 64, 128)] {
        let g = KdaGeometry {
            num_heads: heads,
            head_dim: dim,
            conv_kernel: 4,
        };
        let state = zero_state(g);
        assert_eq!(
            state.buffer(kda::RECURRENT).cells().len(),
            heads * dim * dim,
            "{name}"
        );
        assert_eq!(
            state.buffer(kda::CONV_Q).cells().len(),
            heads * dim * (g.conv_kernel - 1),
            "{name}"
        );
        assert_eq!(g.value_width(), heads * dim, "{name}");
    }
}
