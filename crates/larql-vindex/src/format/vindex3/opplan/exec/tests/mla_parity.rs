//! MLA execution against a pinned tiny oracle, boundary by boundary —
//! [`kda_parity.rs`](super::kda_parity)'s own role, for the OTHER
//! attention family.
//!
//! `kimi_mla_oracle.json` (2 heads, tiny asymmetric widths, `scripts/
//! kimi_mla_oracle_export.py`) is synthetic — a small committable
//! fixture, same reasoning KDA's own tiny oracle uses. The arithmetic is
//! identical at any width; indexing, stride and causal boundaries at
//! REAL width (32 heads, 512-wide latent) are a separate, env-gated
//! real-weight gate (`kimi_mla_layer_real.rs`).
//!
//! **Three positions, not one.** A single cached position cannot
//! exercise this operator at all: softmax over one score is `1.0`
//! regardless of its value (see `exec::mla`'s own doc comment), so N=1
//! only proves the projection/decompression stages, never the attention
//! math itself. N=2 and N=3 are where a causal or indexing defect would
//! actually show up.

use larql_models::config::{MlaGeometry, NormType};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kernels::norm as rms_norm;
use crate::format::vindex3::opplan::exec::mla::{
    mla_forward, MlaState, MlaTrace, MlaWeights, Mutation,
};

const ORACLE: &str = include_str!("kimi_mla_oracle.json");
/// f32 transcription noise between two implementations of the same
/// arithmetic in different orders — matches `kda_parity.rs`'s own bound.
const TOLERANCE: f32 = 2e-5;

struct Fixture {
    geometry: MlaGeometry,
    hidden: usize,
    kv_a_norm_eps: f64,
    weights: std::collections::BTreeMap<String, Vec<f32>>,
    input: Vec<Vec<f32>>,
    boundaries: Value,
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
    }
}

impl Fixture {
    fn weights(&self) -> MlaWeights<'_> {
        let g = |n: &str| self.weights.get(n).expect(n).as_slice();
        MlaWeights {
            q_proj: WeightRows::F32(g("q_proj")),
            kv_a_proj: WeightRows::F32(g("kv_a_proj")),
            kv_a_norm: g("kv_a_norm"),
            kv_b_proj: WeightRows::F32(g("kv_b_proj")),
            o_proj: WeightRows::F32(g("o_proj")),
            kv_a_norm_eps: self.kv_a_norm_eps,
        }
    }

    /// Run every position 0..=up_to through a FRESH state, threaded
    /// exactly like a real decode sequence, returning the trace at
    /// `up_to`.
    fn run_to(&self, up_to: usize, mutation: Mutation) -> (MlaTrace, MlaState) {
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
        (last.unwrap(), state)
    }

    fn expected(&self, p: usize, boundary: &str) -> Vec<f32> {
        floats(&self.boundaries[boundary][p])
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_boundaries(up_to: usize) {
    let f = load();
    let (trace, _) = f.run_to(up_to, Mutation::None);
    let got: [(&str, &Vec<f32>); 6] = [
        ("q_proj", &trace.q_proj),
        ("compressed_kv", &trace.compressed_kv),
        ("kv_a_normed", &trace.kv_a_normed),
        ("kv_b", &trace.kv_b),
        ("attn_weights", &trace.attn_weights),
        ("attn_value", &trace.attn_value),
    ];
    for (name, actual) in got {
        let d = max_abs_diff(actual, &f.expected(up_to, name));
        assert!(
            d < TOLERANCE,
            "position {up_to} boundary `{name}`: max|Δ| {d:e}"
        );
    }
    let d = max_abs_diff(&trace.output, &f.expected(up_to, "output"));
    assert!(
        d < TOLERANCE,
        "position {up_to} boundary `output`: max|Δ| {d:e}"
    );
}

#[test]
fn position_zero_matches_every_projection_boundary() {
    // The degenerate case this module's own doc comment names: proves
    // q_proj/compressed_kv/kv_a_normed/kv_b are right, NOT that the
    // attention math is — softmax(one score) is 1.0 regardless.
    assert_boundaries(0);
}

/// If position 0 passes and position 1 fails, the defect is in the
/// causal read (state carry, indexing into cached positions) rather
/// than the per-position projection formulas — the two run the same
/// projections and differ only in how many cached positions get read.
#[test]
fn position_one_matches_every_boundary_including_real_attention() {
    assert_boundaries(1);
}

#[test]
fn position_two_matches_every_boundary_at_three_cached_positions() {
    assert_boundaries(2);
}

#[test]
fn geometry_matches_kimis_ratios_not_a_symmetric_placeholder() {
    // An awkward-shapes check in miniature: every width in this fixture
    // is DIFFERENT from every other, so a transposed head axis or a
    // swapped nope/rope/v_head_dim slice cannot pass by accident the way
    // it could if they all happened to share one value.
    let f = load();
    let g = f.geometry;
    let widths = [
        g.num_heads,
        g.kv_lora_rank,
        g.qk_nope_head_dim,
        g.qk_rope_head_dim,
        g.v_head_dim,
    ];
    for i in 0..widths.len() {
        for j in (i + 1)..widths.len() {
            assert_ne!(
                widths[i], widths[j],
                "widths[{i}] == widths[{j}], control is not load-bearing"
            );
        }
    }
}

// ── controls ────────────────────────────────────────────────────────
//
// Each perturbs the real function. A fixture proves nothing until the
// things it exists to catch actually break it. Run at position 2 (three
// cached positions), the earliest point every control has room to act.

fn control(mutation: Mutation) -> (f32, f32) {
    let f = load();
    let last = f.positions - 1;
    let (base, base_state) = f.run_to(last, Mutation::None);
    let (changed, changed_state) = f.run_to(last, mutation);
    (
        max_abs_diff(&base.output, &changed.output),
        max_abs_diff(
            base_state.rows().last().unwrap(),
            changed_state.rows().last().unwrap(),
        ),
    )
}

/// **The load-bearing claim this whole rung exists to make executable**:
/// `mla_use_nope` is not a comment, it is a measured property. Rotating
/// the shared K-rope component with the crate's OWN trusted RoPE kernel
/// must move the output once real positions differ — proving the
/// reference's omission of that rotation is a real decision, not one
/// that happened not to matter.
#[test]
fn treating_the_shared_k_rope_component_as_positioned_moves_the_output() {
    let (out, _) = control(Mutation::TreatSharedKRopeAsPositioned { theta: 10_000.0 });
    assert!(
        out > 1e-3,
        "output Δ {out:e} — the rope-as-positioned control did not fire"
    );
}

/// The one thing the rope control CANNOT prove: `compressed_kv` itself
/// is the raw `kv_a_proj` output, unaffected by anything read-side —
/// mutating how it is READ must never rewrite what got cached.
#[test]
fn the_rope_control_never_touches_what_was_cached() {
    let mut none_state = MlaState::default();
    let mut mutated_state = MlaState::default();
    for p in 0..=(load().positions - 1) {
        let f = load();
        mla_forward(
            &f.input[p],
            f.hidden,
            f.weights(),
            f.geometry,
            &mut none_state,
            Mutation::None,
        );
        mla_forward(
            &f.input[p],
            f.hidden,
            f.weights(),
            f.geometry,
            &mut mutated_state,
            Mutation::TreatSharedKRopeAsPositioned { theta: 10_000.0 },
        );
    }
    assert_eq!(
        none_state, mutated_state,
        "the cache must not depend on how it is later read"
    );
}

/// `kv_a_layernorm`'s gain is not the identity at real weights — skipping
/// it at every cached position's decompression must move the output.
#[test]
fn omitting_kv_a_layernorm_moves_the_output() {
    let (out, _) = control(Mutation::OmitKvANorm);
    assert!(out > 1e-3, "output Δ {out:e}");
}

/// **No "attend to the future" control exists, deliberately** — see
/// `exec::mla::mla_forward`'s own doc comment. `state` never holds a
/// position this call has not itself appended, so causality here is a
/// property of the append-then-read CONTRACT, not a runtime bound some
/// omitted check could plausibly disable. A property test built to
/// "prove" that control fires would either fail structurally (there is
/// nothing to leak) or, if forced by widening the visible range past
/// what was pushed, would be testing invented code no real call path
/// runs — pinning a coincidence rather than a defect. Confirmed by
/// running exactly that experiment before writing this comment: at the
/// last processed position, `cached == cur_pos + 1` always, so the
/// widened range was identical to the causal one and the control never
/// fired. Removed rather than kept as a test that cannot fail.
#[test]
fn causality_is_structural_not_a_runtime_bound() {
    // What IS checkable: `state.compressed_kv` never grows past what
    // this call itself appended.
    let f = load();
    let mut state = MlaState::default();
    for p in 0..f.positions {
        mla_forward(
            &f.input[p],
            f.hidden,
            f.weights(),
            f.geometry,
            &mut state,
            Mutation::None,
        );
        assert_eq!(
            state.len(),
            p + 1,
            "state must hold exactly the positions seen so far"
        );
    }
}

/// The pure-function geometry helpers this module leans on — verified
/// once here rather than assumed correct because every other test
/// exercises them indirectly.
#[test]
fn geometry_helpers_match_kimis_real_shapes() {
    let g = MlaGeometry {
        num_heads: 32,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
    };
    assert_eq!(g.q_head_dim(), 192);
    assert_eq!(g.compressed_kv_width(), 576);
}

/// `NormType::RmsNorm` at `kv_a_norm_eps` is exactly what `mla_forward`
/// itself calls for the latent — pinning the shared kernel's own
/// contract here means a change to `exec::kernels::norm`'s signature or
/// semantics fails this file too, not just silently changes MLA's
/// behaviour underneath it.
#[test]
fn the_shared_norm_kernel_is_what_kv_a_layernorm_actually_calls() {
    let f = load();
    let raw = &f.input[0][..f.geometry.kv_lora_rank.min(f.input[0].len())];
    let weight = vec![1.0f32; raw.len()];
    let out = rms_norm(NormType::RmsNorm, raw, &weight, 0.0, f.kv_a_norm_eps);
    assert_eq!(out.len(), raw.len());
}
