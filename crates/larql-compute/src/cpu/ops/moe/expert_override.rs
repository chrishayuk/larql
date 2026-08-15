//! BW-C: oracle single-expert-invocation substitution (research
//! instrument, **OFF by default**).
//!
//! Ablates exactly ONE `(layer, expert)` occurrence's contribution
//! inside `cpu_moe_forward`'s `add_expert` closure — the whole-operation
//! unit (an entire expert call disappears), not a sub-matrix. Contrast
//! [`super::within_expert`], BW-B's sibling instrument, which prunes
//! FEATURES inside a kept expert; this module skips the expert
//! entirely.
//!
//! Layer attribution mirrors [`super::within_expert`]'s
//! [`super::within_expert::set_current_layer`] exactly, and for the
//! SAME reason `within_expert` doesn't use
//! [`crate::moe_route_observe::LayerScope`]: `add_expert` runs on
//! rayon worker threads (`into_par_iter`/`par_chunks`/the spin-pool
//! path), and `LayerScope` is thread-local — set on the driving
//! thread, invisible from a worker. This module's `CURRENT_LAYER` is a
//! plain [`AtomicUsize`], set once per layer by the driver
//! (`moe_ffn_block_cpu_with_index`, right beside the existing
//! `within_expert::set_current_layer` call) before the per-position
//! loop, and read from whichever thread `add_expert` executes on.
//! Confirmed the hard way: an earlier version of this module used
//! `LayerScope` and silently observed zero calls on every run — the
//! thread-local read zero on every worker thread, and there was
//! nothing to fail loudly about it.
//!
//! # Substitute semantics
//!
//! The only substitute implemented here is ZERO — skip the expert's
//! contribution to the combine sum entirely. Because GPT-OSS's MoE
//! combine has no post-expert normalisation
//! (`MoePostExpertNormPolicy::None`, a pure additive sum), zero-
//! contribution IS residual/identity pass-through:
//! `h_out = h_post_attn + Σ_{i≠e} w_i·expert_i(x)`. This is NOT the
//! same claim as "the router had picked only the other k-1 experts" —
//! GPT-OSS renormalises the selected top-k weights to sum to 1 BEFORE
//! this sum (`MoeTopKWeightPolicy::RenormalizedSoftmax`), so removing
//! one selected expert's contribution post-hoc leaves the survivors
//! summing to `1 - w_e`, not renormalised to 1. State this precisely
//! in any result that cites it — it is a real, well-defined
//! perturbation, just a distinct one from "never routed there".
//!
//! # "Exactly one invocation" — BW-C3's SET generalisation, and
//! BW-C5.1's MULTI-LAYER generalisation
//!
//! [`arm_once`] targets a single `(layer, expert)` pair and fires on
//! the FIRST matching call only. [`arm_set`] generalises this to a
//! whole SET of experts at one layer — BW-C3's minimum-sufficient-set
//! question needs "ablate experts {2, 5, 9} simultaneously at this one
//! step", not just one at a time. [`arm_multi`] generalises AGAIN, to
//! up to [`MAX_TARGETS`] simultaneous `(layer, expert-set)` pairs —
//! BW-C5.1's multi-layer composability ladder needs "ablate the whole
//! top-4 group at layers {16, 18, 20, 22} all within the SAME forward
//! pass", not just one layer at a time. `arm_once(layer, expert)` and
//! `arm_set(layer, experts)` are both sugar for `arm_multi` with a
//! single-element target list — all three share one fixed-size array
//! of bitmasks (one per target layer, up to 64 experts each —
//! GPT-OSS-20B's ~32 is well under that), and each expert at each
//! targeted layer fires independently (its own bit cleared by its own
//! compare-exchange) the first time [`should_skip`] sees it, so a
//! repeat visit to an ALREADY-fired expert at a LATER decode step runs
//! normally — same one-shot-per-target contract throughout, now per
//! `(layer, expert)` pair instead of per-call. This is what makes "one
//! invocation, of however many experts at however many layers"
//! well-defined without a separate decode-step counter: the first
//! occurrence of each targeted `(layer, expert)` pair during a
//! deterministic (greedy) decode from a fixed prompt is itself
//! deterministic. [`fired`] (any target at any layer fired) and
//! [`fired_mask`] (exactly which experts fired at the FIRST-armed
//! layer — the `arm_once`/`arm_set` single-layer case) distinguish
//! "ablated" from "target never reached"; [`fired_mask_for`] gives the
//! same detail for any armed layer, needed once more than one is live.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// How many simultaneous `(layer, expert-set)` targets [`arm_multi`]
/// supports. 8 covers BW-C5.1's full "all candidate late layers" rung
/// with room to spare — a fixed-size array (not a `Vec`/`HashMap`)
/// keeps [`should_skip`]'s hot-path scan lock-free and allocation-free.
pub const MAX_TARGETS: usize = 8;

/// How many of the [`MAX_TARGETS`] slots are currently armed — `0`
/// means fully disarmed, matching the old `ARMED: AtomicBool`'s
/// semantics exactly (`should_skip` bails immediately when this is 0).
static ARMED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Slot `i`'s target layer, for `i < ARMED_COUNT`. `usize::MAX` in an
/// unused slot (`i >= ARMED_COUNT`) — never consulted, since
/// `should_skip` only scans `0..ARMED_COUNT`.
static TARGET_LAYER: [AtomicUsize; MAX_TARGETS] =
    [const { AtomicUsize::new(usize::MAX) }; MAX_TARGETS];
/// Slot `i`'s live ablation-target bitmask (bit `e` = expert `e` is
/// still armed at `TARGET_LAYER[i]`). [`should_skip`] clears a bit via
/// compare-exchange the first time it matches — see the module doc.
static TARGET_MASK: [AtomicU64; MAX_TARGETS] = [const { AtomicU64::new(0) }; MAX_TARGETS];
/// Slot `i`'s fired bitmask (bit `e` = expert `e`'s ablation actually
/// fired since the last arm call). Superset info beyond the boolean
/// [`fired`].
static FIRED_MASK: [AtomicU64; MAX_TARGETS] = [const { AtomicU64::new(0) }; MAX_TARGETS];
/// The layer currently executing, set by the driver loop. Cross-thread
/// (see the module doc for why this can't be `moe_route_observe`'s
/// thread-local `LayerScope`). `usize::MAX` = no layer declared.
static CURRENT_LAYER: AtomicUsize = AtomicUsize::new(usize::MAX);
/// L2 norm of the incoming residual stream at the position `add_expert`
/// is about to be called for — BW-C1's contribution/residual-norm
/// covariate. Set once per POSITION (not per layer, unlike
/// `CURRENT_LAYER`: the residual varies every position within a
/// layer's loop) by the driver, right where it already has the
/// pre-MoE-combine row in hand. Bit-cast `f32` in an `AtomicU32` —
/// there is no lock-free `AtomicF32` in `std`. `RESIDUAL_NORM_VALID`
/// distinguishes "never set" from a legitimate zero norm.
static RESIDUAL_NORM_BITS: AtomicU32 = AtomicU32::new(0);
static RESIDUAL_NORM_VALID: AtomicBool = AtomicBool::new(false);

/// Set the layer whose expert calls are about to run. Called once per
/// MoE layer by the driver, right beside the existing
/// `within_expert::set_current_layer` call — layers run sequentially,
/// so a single atomic store covers the whole per-position loop.
pub fn set_current_layer(layer: usize) {
    CURRENT_LAYER.store(layer, Ordering::Relaxed);
}

fn current_layer() -> Option<usize> {
    match CURRENT_LAYER.load(Ordering::Relaxed) {
        usize::MAX => None,
        l => Some(l),
    }
}

/// Set the incoming residual stream's L2 norm for the position about to
/// be processed. Called once per position by the driver
/// (`moe_ffn_block_cpu_with_index`), right after it computes that
/// position's pre-MoE row — cheap (the driver already holds the row;
/// this is one extra `sqrt(sum(v*v))` per position, not per expert
/// call).
pub fn set_current_residual_norm(norm: f32) {
    RESIDUAL_NORM_BITS.store(norm.to_bits(), Ordering::Relaxed);
    RESIDUAL_NORM_VALID.store(true, Ordering::Relaxed);
}

fn current_residual_norm() -> Option<f32> {
    RESIDUAL_NORM_VALID
        .load(Ordering::Relaxed)
        .then(|| f32::from_bits(RESIDUAL_NORM_BITS.load(Ordering::Relaxed)))
}

/// Expert index as a `TARGET_MASK`/`FIRED_MASK` bit, or `0` if it's
/// out of the 64-expert range this module supports — refuses to guess
/// rather than wrapping/panicking on a shift overflow. GPT-OSS-20B's
/// ~32 experts are well inside this; the guard is for whatever model
/// runs next.
fn expert_bit(expert: usize) -> u64 {
    if expert < 64 {
        1u64 << expert
    } else {
        0
    }
}

/// Arm a one-shot override on a single expert — sugar for
/// [`arm_multi`]`(&[(layer, &[expert])])`. See the module doc's
/// "Exactly one invocation" section for the fire-once contract.
pub fn arm_once(layer: usize, expert: usize) {
    arm_multi(&[(layer, &[expert])]);
}

/// Arm a one-shot override on a SET of experts at one layer — BW-C3's
/// minimum-sufficient-set question. Sugar for
/// [`arm_multi`]`(&[(layer, experts)])`. Each expert in `experts`
/// fires independently the first time [`should_skip`] sees it; a
/// repeat visit to an already-fired expert at a LATER decode step runs
/// normally, exactly like [`arm_once`].
pub fn arm_set(layer: usize, experts: &[usize]) {
    arm_multi(&[(layer, experts)]);
}

/// Arm a one-shot override across UP TO [`MAX_TARGETS`] simultaneous
/// `(layer, expert-set)` pairs — BW-C5.1's multi-layer composability
/// question needs "ablate the whole group at layers {16, 18, 20, 22}
/// within the SAME forward pass", not one layer at a time. Each
/// expert at each targeted layer fires independently, exactly like
/// [`arm_set`] generalised across layers. Overwrites any previous
/// target and clears every slot's [`fired`]/[`fired_mask_for`].
/// Experts `>= 64` are silently dropped from their layer's mask (see
/// [`expert_bit`]) rather than panicking. `targets.len() >
/// MAX_TARGETS` is truncated to the first `MAX_TARGETS` entries rather
/// than panicking — a caller building a ladder rung larger than this
/// module supports gets a smaller-than-requested arm, not a crash;
/// check `targets.len() <= MAX_TARGETS` in the caller if that
/// silent truncation would be a problem.
pub fn arm_multi(targets: &[(usize, &[usize])]) {
    let n = targets.len().min(MAX_TARGETS);
    for i in 0..MAX_TARGETS {
        if i < n {
            let (layer, experts) = targets[i];
            let mask = experts.iter().fold(0u64, |m, &e| m | expert_bit(e));
            TARGET_LAYER[i].store(layer, Ordering::Relaxed);
            TARGET_MASK[i].store(mask, Ordering::Relaxed);
        } else {
            TARGET_LAYER[i].store(usize::MAX, Ordering::Relaxed);
            TARGET_MASK[i].store(0, Ordering::Relaxed);
        }
        FIRED_MASK[i].store(0, Ordering::Relaxed);
    }
    ARMED_COUNT.store(n, Ordering::Relaxed);
}

/// Disarm — restores byte-exact parity with the un-instrumented path.
pub fn disarm() {
    ARMED_COUNT.store(0, Ordering::Relaxed);
}

/// Whether ANY targeted expert's ablation fired, at ANY armed layer,
/// since the last arm call. See the module doc: distinguishes a real
/// ablation from a target that was never reached. Prefer
/// [`fired_mask_for`] (or [`fired_mask`] in the single-layer case) to
/// confirm ALL targeted experts fired — this only tells you at least
/// one did, at least one layer.
pub fn fired() -> bool {
    FIRED_MASK.iter().any(|m| m.load(Ordering::Relaxed) != 0)
}

/// Which experts fired at the FIRST-armed layer (slot 0) — the
/// `arm_once`/`arm_set` single-layer case, kept as the zero-argument
/// form those callers already use. For a multi-layer [`arm_multi`]
/// arm, use [`fired_mask_for`] per layer instead.
pub fn fired_mask() -> u64 {
    FIRED_MASK[0].load(Ordering::Relaxed)
}

/// Which experts fired at `layer` since the last arm call, as a
/// bitmask over the SAME expert indices passed to `arm_multi`/
/// `arm_set` for that layer (bit `e` set = expert `e` fired). Returns
/// `0` if `layer` was never armed — indistinguishable from "armed but
/// nothing fired" by design (both mean "nothing to report for this
/// layer"); a caller checking a specific rung's targets already knows
/// which layers it armed.
pub fn fired_mask_for(layer: usize) -> u64 {
    let armed = ARMED_COUNT.load(Ordering::Relaxed);
    for i in 0..armed {
        if TARGET_LAYER[i].load(Ordering::Relaxed) == layer {
            return FIRED_MASK[i].load(Ordering::Relaxed);
        }
    }
    0
}

/// Consulted inside `add_expert`: `true` means "skip this expert's
/// contribution entirely" (the zero/residual-identity substitute).
/// Single relaxed atomic load in the common (disarmed) case — no
/// layer lookup, no scan.
pub(crate) fn should_skip(expert: usize) -> bool {
    let armed = ARMED_COUNT.load(Ordering::Relaxed);
    if armed == 0 {
        return false;
    }
    let Some(layer) = current_layer() else {
        // No layer attribution active on this call path — refuse
        // rather than guess which layer this is.
        return false;
    };
    let bit = expert_bit(expert);
    if bit == 0 {
        return false;
    }
    // Scan only the armed slots (<= MAX_TARGETS, typically far fewer)
    // — cheap linear search, no allocation, no lock. At most one slot
    // can match `layer`: `arm_multi` never stores the same layer
    // twice.
    for i in 0..armed {
        if TARGET_LAYER[i].load(Ordering::Relaxed) != layer {
            continue;
        }
        // One-shot PER EXPERT: the first call for a given expert at
        // this layer wins the compare-exchange and clears just that
        // bit; a second visit to the SAME (layer, expert) pair (a
        // later decode step) finds its bit already cleared and runs
        // normally, independent of every other targeted layer/expert.
        loop {
            let current = TARGET_MASK[i].load(Ordering::Relaxed);
            if current & bit == 0 {
                return false;
            }
            if TARGET_MASK[i]
                .compare_exchange(
                    current,
                    current & !bit,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                FIRED_MASK[i].fetch_or(bit, Ordering::Relaxed);
                return true;
            }
        }
    }
    false
}

// ── Observation: record which (layer, expert, router_weight,
// out_norm, residual_norm) quintuples actually fire, in call order, so
// a harness can pick REAL oracle targets from a real decode rather
// than guessing indices. `router_weight` is BW-C1's first covariate
// (cheapest — `add_expert` already has it); `out_norm` and
// `residual_norm` are BW-C1's second wave — GPT-OSS renormalises
// selected top-k weights before the combine (see the module doc),
// which decouples routing SCORE from anything about the expert's
// actual output MAGNITUDE, so `out_norm` (and the derived
// `out_norm / residual_norm` ratio) tests a genuinely different
// hypothesis from router weight, not a restatement of it.
// `residual_norm` is `NaN` when [`set_current_residual_norm`] was
// never called on this path (e.g. a unit test that only exercises
// `observe` directly) — an explicit "not measured", never a silently
// wrong zero. No-op unless enabled — same opt-in-instrument contract
// as the rest of this module. ──

/// One observed `add_expert` call — every covariate BW-C1 tests as a
/// candidate predictor of skippability. A named struct rather than a
/// positional tuple on purpose: this module has already grown from 2
/// fields (layer, expert) to 5 across BW-C1's two covariate waves, and
/// a positional tuple that size is exactly the kind of thing that
/// silently transposes two `f32` fields at a call site — every
/// consumer here matches by name instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertObservation {
    pub layer: usize,
    pub expert: usize,
    /// The router's own selection weight for this call (BW-C1's first,
    /// cheapest covariate — GPT-OSS's `RenormalizedSoftmax` selected
    /// weight, already in scope at every call site).
    pub router_weight: f32,
    /// L2 norm of the expert's raw (pre-weight) output vector — BW-C1's
    /// second covariate. GPT-OSS renormalises selected top-k weights
    /// before the combine (see the module doc), which decouples
    /// routing SCORE from anything about the expert's actual output
    /// MAGNITUDE, so this tests a genuinely different hypothesis from
    /// `router_weight`, not a restatement of it.
    pub out_norm: f32,
    /// L2 norm of the incoming residual stream at this position —
    /// `out_norm / residual_norm` is the contribution-relative-to-
    /// stream ratio, which normalises out the fact that raw activation
    /// norms grow substantially with depth on their own (a `NaN` here
    /// would otherwise confound `out_norm` with layer index). `NaN`
    /// when [`set_current_residual_norm`] was never called on this
    /// path (e.g. a unit test that only exercises `observe` directly)
    /// — an explicit "not measured", never a silently wrong zero.
    pub residual_norm: f32,
}

static OBSERVE: AtomicBool = AtomicBool::new(false);
static OBSERVED: Mutex<Vec<ExpertObservation>> = Mutex::new(Vec::new());

/// Start recording every [`ExpertObservation`] `add_expert` visits, in
/// call order. Clears any prior recording.
pub fn start_observing() {
    OBSERVED.lock().unwrap_or_else(|p| p.into_inner()).clear();
    OBSERVE.store(true, Ordering::Relaxed);
}

/// Stop recording and drain the observed call sequence.
pub fn stop_observing() -> Vec<ExpertObservation> {
    OBSERVE.store(false, Ordering::Relaxed);
    std::mem::take(&mut *OBSERVED.lock().unwrap_or_else(|p| p.into_inner()))
}

/// Consulted inside `add_expert`, AFTER the expert's raw (pre-weight)
/// output vector is computed — unlike [`should_skip`], which must gate
/// BEFORE that computation to actually save the work when armed. Takes
/// `out_norm` (the raw output's L2 norm) as a caller-computed value
/// rather than a slice, so this module never touches the expert-width
/// buffer itself: the norm is one `sqrt(sum(v*v))` the caller already
/// has cheap access to right where the vector exists, and passing it
/// in keeps `expert_override` architecture-agnostic (no `hidden`/
/// `inter` knowledge needed here). `residual_norm` comes from
/// [`current_residual_norm`], not a parameter — it is set once per
/// POSITION by the driver, not per expert call, so there is nothing
/// for a caller here to usefully pass. Single relaxed atomic load when
/// not observing.
pub(crate) fn observe(expert: usize, router_weight: f32, out_norm: f32) {
    if !OBSERVE.load(Ordering::Relaxed) {
        return;
    }
    let Some(layer) = current_layer() else {
        return;
    };
    let residual_norm = current_residual_norm().unwrap_or(f32::NAN);
    OBSERVED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(ExpertObservation {
            layer,
            expert,
            router_weight,
            out_norm,
            residual_norm,
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Global state (`ARMED`/`TARGET_LAYER`/`TARGET_MASK`/`FIRED_MASK`/
    /// `OBSERVE`/`OBSERVED`/`CURRENT_LAYER`/`RESIDUAL_NORM_*`) means
    /// every test here must be serialised against every other, exactly
    /// as `within_expert.rs`'s test module does.
    static LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        disarm();
        let _ = stop_observing();
        CURRENT_LAYER.store(usize::MAX, Ordering::Relaxed);
        RESIDUAL_NORM_VALID.store(false, Ordering::Relaxed);
    }

    /// With no layer declared, `current_layer()` is `None`, so
    /// `should_skip` refuses rather than guessing a layer — armed or
    /// not, it must return `false`.
    #[test]
    fn should_skip_refuses_without_a_declared_layer() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_once(0, 0);
        assert!(
            !should_skip(0),
            "no layer declared — must refuse, not guess"
        );
        assert!(!fired());
        reset();
    }

    /// The one load-bearing property: armed + matching layer + matching
    /// expert fires exactly once, and only once — a second visit to the
    /// same pair (simulating a later decode step) runs normally.
    #[test]
    fn arm_once_fires_exactly_once_on_the_matching_pair() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_once(3, 5);
        set_current_layer(3);
        assert!(!should_skip(4), "wrong expert must not fire");
        assert!(!fired());
        assert!(should_skip(5), "matching layer+expert must fire");
        assert!(fired());
        assert!(
            !should_skip(5),
            "the SAME pair on a later call must run normally — one-shot"
        );
        reset();
    }

    /// A different layer with the same expert index must not fire —
    /// layer and expert are a joint key, not independent filters.
    #[test]
    fn wrong_layer_does_not_fire_even_with_matching_expert() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_once(3, 5);
        set_current_layer(7);
        assert!(!should_skip(5));
        assert!(!fired());
        reset();
    }

    /// `disarm` restores byte-exact parity — no fire regardless of layer
    /// or expert.
    #[test]
    fn disarm_suppresses_every_call() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_once(0, 0);
        disarm();
        set_current_layer(0);
        assert!(!should_skip(0));
        reset();
    }

    /// Observation records the exact call sequence, in order, and clears
    /// on `start_observing` — a stale recording from an earlier test must
    /// never leak into the next. Sets a residual norm first so the
    /// observations are plain comparable floats (a never-set residual
    /// norm is covered separately, since NaN != NaN breaks
    /// `assert_eq!`).
    #[test]
    fn observation_records_call_order_and_clears_on_restart() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        start_observing();
        set_current_layer(0);
        set_current_residual_norm(4.0);
        observe(2, 0.7, 1.5);
        observe(9, 0.1, 2.25);
        set_current_layer(1);
        observe(4, 0.4, 0.75);
        let obs = |layer, expert, router_weight, out_norm, residual_norm| ExpertObservation {
            layer,
            expert,
            router_weight,
            out_norm,
            residual_norm,
        };
        assert_eq!(
            stop_observing(),
            vec![
                obs(0, 2, 0.7, 1.5, 4.0),
                obs(0, 9, 0.1, 2.25, 4.0),
                obs(1, 4, 0.4, 0.75, 4.0),
            ]
        );

        // A fresh start must not carry over the drained recording.
        start_observing();
        assert_eq!(stop_observing(), Vec::<ExpertObservation>::new());
        reset();
    }

    /// A residual norm that was never set is recorded as `NaN` — an
    /// explicit "not measured", never a silently-wrong zero that would
    /// masquerade as a real (degenerate) residual.
    #[test]
    fn observe_residual_norm_is_nan_when_never_set() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        start_observing();
        set_current_layer(0);
        observe(2, 0.7, 1.5);
        let recorded = stop_observing();
        assert_eq!(recorded.len(), 1);
        assert!(
            recorded[0].residual_norm.is_nan(),
            "residual_norm must be NaN when set_current_residual_norm was never called, got {}",
            recorded[0].residual_norm
        );
        reset();
    }

    /// `observe` is a no-op (single relaxed load) unless
    /// `start_observing` was called — it must not silently accumulate.
    #[test]
    fn observe_is_inert_when_not_observing() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        set_current_layer(0);
        observe(0, 0.5, 1.0);
        assert_eq!(stop_observing(), Vec::<ExpertObservation>::new());
        reset();
    }

    /// BW-C3: `arm_set` on {2, 5, 9} fires each of the three
    /// independently, in whatever order `should_skip` visits them —
    /// each targeted expert gets its own one-shot, not a single global
    /// fire-once shared across the set.
    #[test]
    fn arm_set_fires_every_targeted_expert_independently() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_set(3, &[2, 5, 9]);
        set_current_layer(3);
        assert!(!should_skip(4), "expert not in the set must not fire");
        assert!(should_skip(2));
        assert!(should_skip(5));
        assert!(should_skip(9));
        assert_eq!(
            fired_mask(),
            expert_bit(2) | expert_bit(5) | expert_bit(9),
            "fired_mask must reflect exactly the three targeted experts"
        );
        assert!(fired());
        reset();
    }

    /// A repeat visit to an ALREADY-fired member of the set (simulating
    /// a later decode step re-selecting the same expert) runs normally
    /// — one-shot per expert, independent of the other set members'
    /// state, exactly like `arm_once`'s single-target contract.
    #[test]
    fn arm_set_member_is_one_shot_independent_of_the_rest() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_set(0, &[1, 2]);
        set_current_layer(0);
        assert!(should_skip(1));
        assert!(
            !should_skip(1),
            "expert 1 already fired — a later visit must run normally"
        );
        assert!(
            should_skip(2),
            "expert 2's own one-shot is independent of expert 1's"
        );
        reset();
    }

    /// A partial fire (only SOME of the armed set actually reached at
    /// this layer) is visible via `fired_mask` — a caller comparing it
    /// against the mask it armed can tell a partial ablation from a
    /// complete one, not just "something fired".
    #[test]
    fn fired_mask_reveals_a_partial_fire() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_set(0, &[1, 2, 3]);
        set_current_layer(0);
        assert!(should_skip(1));
        // Expert 2 never actually gets called this "step" (simulates a
        // mis-specified target that wasn't part of the real routing).
        assert!(should_skip(3));
        assert_eq!(
            fired_mask(),
            expert_bit(1) | expert_bit(3),
            "expert 2 never fired — its bit must be absent, not silently assumed"
        );
        reset();
    }

    /// `arm_once` is exactly `arm_set` with a one-element set — the
    /// wrapper must not change externally-observable behaviour.
    #[test]
    fn arm_once_matches_arm_set_with_one_expert() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_once(2, 7);
        set_current_layer(2);
        assert!(should_skip(7));
        assert_eq!(fired_mask(), expert_bit(7));
        reset();
    }

    /// BW-C5.1: `arm_multi` on two DIFFERENT layers fires independently
    /// at each — a decode step visits layers sequentially
    /// (`set_current_layer` changes as the forward pass proceeds), and
    /// each layer's targets must fire without disturbing the other's
    /// state.
    #[test]
    fn arm_multi_fires_independently_across_layers() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_multi(&[(3, &[1, 2]), (7, &[5])]);

        set_current_layer(3);
        assert!(!should_skip(9), "expert 9 not targeted at layer 3");
        assert!(should_skip(1));
        assert!(should_skip(2));
        assert!(!should_skip(1), "layer 3 expert 1 already fired — one-shot");

        set_current_layer(7);
        assert!(
            !should_skip(1),
            "layer 3's targets must not leak into layer 7"
        );
        assert!(should_skip(5));

        assert_eq!(fired_mask_for(3), expert_bit(1) | expert_bit(2));
        assert_eq!(fired_mask_for(7), expert_bit(5));
        assert!(fired());
        reset();
    }

    /// A layer that was never armed always refuses, even with an
    /// otherwise-matching expert index at a DIFFERENT armed layer —
    /// `fired_mask_for` on it returns 0, indistinguishable from "armed
    /// but nothing fired" by design (see the function's doc).
    #[test]
    fn unarmed_layer_never_fires_and_reports_zero() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_multi(&[(3, &[1])]);
        set_current_layer(99);
        assert!(!should_skip(1));
        assert_eq!(fired_mask_for(99), 0);
        assert_eq!(fired_mask_for(3), 0, "layer 3 was armed but never visited");
        reset();
    }

    /// `disarm` clears every slot, not just the first — a stale
    /// multi-layer arm must not leak into a later single-layer
    /// `arm_once` call's unused slots.
    #[test]
    fn disarm_clears_every_multi_target_slot() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        arm_multi(&[(3, &[1]), (7, &[5]), (9, &[2])]);
        disarm();
        set_current_layer(3);
        assert!(!should_skip(1));
        set_current_layer(7);
        assert!(!should_skip(5));
        set_current_layer(9);
        assert!(!should_skip(2));
        assert!(!fired());
        reset();
    }
}
