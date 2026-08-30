//! One Kimi MoE block, in isolation — the combine arithmetic (weighted
//! sum over the SELECTED experts, plus the shared branch, unscaled), not
//! yet against real weights (that gate is a real-container test, run
//! separately and env-gated).
//!
//! Reuses `kimi_router.rs`'s exact fixture (same logits, same bias, same
//! near-boundary construction) so the selected ids are already proven —
//! this file's own job is everything AFTER selection.

use std::sync::Mutex;

use crate::format::vindex3::opplan::exec::kimi_moe_block::{
    expert_ffn, moe_block_forward, ExpertWeights,
};

const EXPERTS: usize = 4;
const HIDDEN: usize = 3;
const INTER: usize = 2;
const TOP_K: usize = 2;
const BRANCH_SCALE: f64 = 2.446;

// Identical to kimi_router.rs's fixture: ids resolve to {0, 2}.
const LOGITS: [f32; EXPERTS] = [2.0, 1.0, 0.5, -1.0];
const BIAS: [f32; EXPERTS] = [0.0, 0.0, 0.3, 0.0];

fn x() -> Vec<f32> {
    vec![1.0, 0.0, 0.0]
}

fn router_weight() -> Vec<f32> {
    let mut w = vec![0.0f32; EXPERTS * HIDDEN];
    for e in 0..EXPERTS {
        w[e * HIDDEN] = LOGITS[e];
    }
    w
}

/// `ExpertWeights` is BF16 code units (P4a — see `kimi_moe_block.rs`'s
/// own doc comment): each `u16` is the top 16 bits of the f32 it
/// denotes. Every value this fixture uses (0, 1, 2, 3, 4, 5) is an
/// integer small enough to be EXACT in bf16 — truncation introduces no
/// error here, so the hand-derived tolerances below are unaffected by
/// the representation change.
fn bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

fn bf16_vec(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&f| bf16(f)).collect()
}

/// Expert `id`'s weights, distinct per id so two experts' outputs cannot
/// be mistaken for one another: `gate = [id+1, 0]` against `x = [1,0,0]`,
/// `up = [1, 0]`, `down` picks the single nonzero `h` entry straight
/// through to every hidden position — the point is not the numbers, it
/// is that expert `id`'s output is recognisably "shaped like `id`".
fn expert(id: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let scale = (id + 1) as f32;
    let mut gate = vec![0.0f32; INTER * HIDDEN];
    gate[0] = scale; // gate = [scale, 0] against x = [1, 0, 0]
    let mut up = vec![0.0f32; INTER * HIDDEN];
    up[0] = 1.0; // up = [1, 0]
                 // down: [HIDDEN, INTER], every row reads only h[0].
    let mut down = vec![0.0f32; HIDDEN * INTER];
    for row in 0..HIDDEN {
        down[row * INTER] = 1.0;
    }
    (bf16_vec(&gate), bf16_vec(&up), bf16_vec(&down))
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Every expert's weights, built once so the closures below can borrow
/// stable-address slices — no per-call allocation, no unsafe.
struct AllExperts(Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>);

impl AllExperts {
    fn build() -> Self {
        Self((0..EXPERTS).map(expert).collect())
    }
    fn get(&self, id: usize) -> ExpertWeights<'_> {
        let (gate, up, down) = &self.0[id];
        ExpertWeights { gate, up, down }
    }
}

/// `expert_ffn` alone, hand-checked: `h[0] = silu(scale) * 1`, `h[1] = 0`,
/// every output position equals `h[0]` (the `down` fixture reads only
/// `h[0]` into every row).
#[test]
fn expert_ffn_matches_the_hand_derived_value() {
    let (gate, up, down) = expert(2);
    let out = expert_ffn(
        &x(),
        ExpertWeights {
            gate: &gate,
            up: &up,
            down: &down,
        },
        HIDDEN,
        INTER,
    );
    let expected = silu(3.0) * 1.0; // scale = id + 1 = 3
    for &v in &out {
        assert!((v - expected).abs() < 1e-6, "{v} vs {expected}");
    }
}

/// `expert_weights` is called EXACTLY for the selected ids — never for an
/// unselected expert. This is the sparsity guarantee routing exists for,
/// checked as a real property rather than assumed.
#[test]
fn only_the_selected_experts_are_ever_loaded() {
    let all = AllExperts::build();
    // P4b-1: `expert_weights` now runs concurrently across the executor's
    // pool (see `moe_block_forward`'s own doc comment), so tracking calls
    // needs THREAD-SAFE interior mutability — `Mutex`, not `RefCell`.
    let queried: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    let trace = moe_block_forward(
        &x(),
        HIDDEN,
        INTER,
        &router_weight(),
        &BIAS,
        EXPERTS,
        TOP_K,
        true,
        BRANCH_SCALE,
        |id| {
            queried.lock().unwrap().push(id);
            all.get(id)
        },
        None,
    );

    let mut queried = queried.into_inner().unwrap();
    queried.sort_unstable();
    let mut selected = trace.router.selected_ids.clone();
    selected.sort_unstable();
    assert_eq!(
        queried, selected,
        "queried experts must equal selected ids, exactly"
    );
    assert_eq!(selected, vec![0, 2]);
}

/// The routed sum is exactly the weighted sum of the per-expert outputs
/// the trace itself reports — the combine arithmetic, checked against the
/// trace's OWN intermediate values rather than a hand-computed number, so
/// this test is about the RELATIONSHIP and not about re-deriving SiLU.
#[test]
fn routed_sum_is_the_weighted_sum_of_the_reported_expert_outputs() {
    let all = AllExperts::build();
    let trace = moe_block_forward(
        &x(),
        HIDDEN,
        INTER,
        &router_weight(),
        &BIAS,
        EXPERTS,
        TOP_K,
        true,
        BRANCH_SCALE,
        |id| all.get(id),
        None,
    );

    assert_eq!(trace.expert_outputs.len(), TOP_K);
    let mut expected = vec![0.0f32; HIDDEN];
    for (out, &w) in trace.expert_outputs.iter().zip(&trace.router.weights) {
        for (e, &v) in expected.iter_mut().zip(out) {
            *e += v * w;
        }
    }
    for (got, want) in trace.routed_sum.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    // With no shared expert, output IS the routed sum.
    assert_eq!(trace.output, trace.routed_sum);
    assert_eq!(trace.shared_output, vec![0.0f32; HIDDEN]);
}

/// The shared branch is ADDED, never multiplied by `routed_scaling_
/// factor` — the P3d-e finding this executor must not silently undo.
#[test]
fn the_shared_branch_is_summed_unscaled() {
    let all = AllExperts::build();
    let shared = expert(EXPERTS); // one past the routed set — its own branch
    let shared_weights = ExpertWeights {
        gate: &shared.0,
        up: &shared.1,
        down: &shared.2,
    };

    let trace = moe_block_forward(
        &x(),
        HIDDEN,
        INTER,
        &router_weight(),
        &BIAS,
        EXPERTS,
        TOP_K,
        true,
        BRANCH_SCALE,
        |id| all.get(id),
        Some((shared_weights, INTER)),
    );

    let expected_shared = expert_ffn(&x(), shared_weights, HIDDEN, INTER);
    assert_eq!(trace.shared_output, expected_shared);

    // output - routed_sum must equal shared_output EXACTLY — not scaled
    // by BRANCH_SCALE, not scaled by anything.
    for ((o, r), s) in trace
        .output
        .iter()
        .zip(&trace.routed_sum)
        .zip(&trace.shared_output)
    {
        assert!((o - (r + s)).abs() < 1e-6);
        // The sharpest negative check: applying the branch scale to the
        // shared output would have produced a DIFFERENT number (branch
        // scale is 2.446, and the shared output here is nonzero).
        assert!(
            (o - (r + s * BRANCH_SCALE as f32)).abs() > 1e-3,
            "shared branch looks scaled by routed_scaling_factor — it must not be"
        );
    }
}
