//! PREC-DIAG-1 E0b gates: the RoPE-angle intervention.
//!
//! 1. Off is the shipped path: the default and an explicit
//!    `RopeAngles::Larql` are bit-identical to an image never touched, on
//!    both CPU backends.
//! 2. On is non-vacuous and confined to the rotation: the logits move;
//!    the E0a capture shows the unpositioned q/k rotate to the positioned
//!    q/k through the HF-f32 construction bit for bit; and the decode step
//!    and the batch traversal agree bit for bit under it (both read the
//!    mode from the same call).
//! 3. The construction is HF's, operation by operation, for plain and
//!    linear RoPE; every other rotary policy is refused.

use super::decode::fixture;
use super::golden::{G_HIDDEN, G_TOKENS};
use super::hybrid_traversal::hybrid;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_prepared_streaming;
use crate::format::vindex3::opplan::exec::kernels::rope_rotate;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::StepObserver;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::rope_angles::{
    hf_f32_cos_sin, hf_f32_inv_freq, hf_f32_policy_inv_freq, rope_rotate_hf_f32, rotate_or_defer,
    RopeAngles,
};
use crate::format::vindex3::opplan::exec::site_capture::{
    CaptureRequest, CaptureSite, CapturedSites, SiteCapture,
};
use crate::format::vindex3::opplan::ComponentOpPlan;
use larql_models::config::PositionPolicy;

const ALL_POSITIONS: [usize; 5] = [0, 1, 2, 3, 4];

fn prepared<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    angles: Option<RopeAngles>,
) -> PreparedOperands {
    let mut ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap();
    if let Some(angles) = angles {
        ops.set_rope_angles(angles).unwrap();
    }
    ops
}

fn decode<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    mut observer: Option<&mut dyn StepObserver>,
) -> Vec<Vec<u32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    G_TOKENS
        .iter()
        .map(|&t| {
            let out = match observer.as_deref_mut() {
                Some(o) => session.step_observed(t, o).unwrap(),
                None => session.step(t).unwrap(),
            };
            out.logits.unwrap().iter().map(|v| v.to_bits()).collect()
        })
        .collect()
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// Run one check on both CPU backends.
fn on_both(mut check: impl FnMut(&str, &dyn Fn(Option<RopeAngles>) -> Run)) {
    let (_c, plan, store) = fixture();
    check("reference", &|angles| {
        let backend = ReferenceBackend::new();
        let ops = prepared(&plan, &store, &backend, angles);
        Run::new(&plan, &ops, &backend)
    });
    check("production", &|angles| {
        let backend = ProductionBackend::new();
        let ops = prepared(&plan, &store, &backend, angles);
        Run::new(&plan, &ops, &backend)
    });
}

/// What one configuration produced: decode logits, a full E0a capture,
/// and the batch traversal's last-position logits.
struct Run {
    decode: Vec<Vec<u32>>,
    captured: CapturedSites,
    batch_last: Vec<u32>,
    plan: ComponentOpPlan,
}

impl Run {
    fn new<B: PlanBackend>(plan: &ComponentOpPlan, ops: &PreparedOperands, backend: &B) -> Self {
        let decode_rows = decode(plan, ops, backend, None);
        let request = CaptureRequest::for_plan(
            plan,
            0..plan.layers.len(),
            G_HIDDEN,
            ALL_POSITIONS.to_vec(),
            G_TOKENS.len(),
            u64::MAX,
        )
        .unwrap();
        let mut capture = SiteCapture::new(request).unwrap();
        let observed = decode(plan, ops, backend, Some(&mut capture));
        assert_eq!(decode_rows, observed, "capture changed the logits");
        let batch =
            execute_prepared_streaming(plan, ops, &G_TOKENS, backend, None, &mut |_| Ok(()))
                .unwrap();
        Self {
            decode: decode_rows,
            captured: capture.finish().unwrap(),
            batch_last: bits(&batch.logits.unwrap()),
            plan: plan.clone(),
        }
    }

    fn row(&self, site: CaptureSite, layer: usize, position: usize) -> Vec<f32> {
        let plane = self
            .captured
            .planes
            .iter()
            .find(|p| p.site == site && p.layer == Some(layer))
            .unwrap();
        let w = plane.width;
        plane.values[position * w..(position + 1) * w].to_vec()
    }
}

// ── 1. off is the shipped path ────────────────────────────────────────

#[test]
fn the_default_and_explicit_larql_are_bit_identical_to_an_untouched_image() {
    on_both(|name, run| {
        let untouched = run(None);
        let explicit = run(Some(RopeAngles::Larql));
        assert_eq!(untouched.decode, explicit.decode, "{name}: decode");
        assert_eq!(untouched.batch_last, explicit.batch_last, "{name}: batch");
        assert_eq!(RopeAngles::default(), RopeAngles::Larql);
    });
}

// ── 2. on is non-vacuous and confined to the rotation ────────────────

/// Positions the miniature is stepped through for the non-vacuity check:
/// long enough that the two constructions' cos/sin tables differ at the
/// fixture's own head width and bases (asserted, not assumed).
const LONG_SEQUENCE: usize = 64;

/// Whether the two constructions give different cos/sin bits for a
/// `head_dim`-wide rotary at `theta` at `position`.
fn tables_differ(head_dim: usize, theta: f64, position: usize) -> bool {
    hf_f32_inv_freq(head_dim, theta, None)
        .iter()
        .enumerate()
        .any(|(i, f)| {
            let (cos_hf, sin_hf) = hf_f32_cos_sin(position, *f);
            let mut shipped = vec![0.0f32; head_dim];
            shipped[i] = 1.0;
            rope_rotate(&mut shipped, position, theta);
            cos_hf.to_bits() != shipped[i].to_bits()
                || sin_hf.to_bits() != shipped[head_dim / 2 + i].to_bits()
        })
}

#[test]
fn hf_f32_angles_move_the_logits_exactly_where_the_tables_differ_first() {
    let (_c, plan, store) = fixture();
    let tokens: Vec<u32> = G_TOKENS
        .iter()
        .copied()
        .cycle()
        .take(LONG_SEQUENCE)
        .collect();
    // The first position at which ANY layer's tables differ: before it,
    // the two modes compute identical bits by construction.
    let first_difference = (0..LONG_SEQUENCE)
        .find(|&position| {
            plan.layers.iter().any(|lp| {
                let op = lp.attention.softmax().unwrap();
                matches!(op.position, PositionPolicy::Rope { theta } if tables_differ(op.head_dim, theta, position))
            })
        })
        .expect("the fixture's tables never differ within the long sequence");
    fn logits<B: PlanBackend>(
        plan: &ComponentOpPlan,
        store: &OperandStore,
        backend: &B,
        angles: RopeAngles,
        tokens: &[u32],
    ) -> Vec<Vec<u32>> {
        let ops = prepared(plan, store, backend, Some(angles));
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(plan, &ops, backend, &mut kv).unwrap();
        tokens
            .iter()
            .map(|&t| bits(&session.step(t).unwrap().logits.unwrap()))
            .collect()
    }
    for (name, larql, hf) in [
        (
            "reference",
            logits(
                &plan,
                &store,
                &ReferenceBackend::new(),
                RopeAngles::Larql,
                &tokens,
            ),
            logits(
                &plan,
                &store,
                &ReferenceBackend::new(),
                RopeAngles::HfF32,
                &tokens,
            ),
        ),
        (
            "production",
            logits(
                &plan,
                &store,
                &ProductionBackend::new(),
                RopeAngles::Larql,
                &tokens,
            ),
            logits(
                &plan,
                &store,
                &ProductionBackend::new(),
                RopeAngles::HfF32,
                &tokens,
            ),
        ),
    ] {
        assert_eq!(
            larql[..first_difference],
            hf[..first_difference],
            "{name}: logits moved before the tables differ"
        );
        assert!(
            larql[first_difference..] != hf[first_difference..],
            "{name}: the tables differ from position {first_difference} yet no logit moved"
        );
    }
}

#[test]
fn under_hf_f32_the_unpositioned_q_k_rotate_to_the_positioned_q_k_through_the_hf_construction() {
    let mut checked = 0;
    on_both(|name, run| {
        let hf = run(Some(RopeAngles::HfF32));
        let larql = run(None);
        for (layer, lp) in hf.plan.layers.iter().enumerate() {
            let op = lp.attention.softmax().unwrap();
            let PositionPolicy::Rope { theta } = op.position else {
                continue;
            };
            let inv_freq = hf_f32_inv_freq(op.head_dim, theta, None);
            for position in ALL_POSITIONS {
                for (un, pos) in [
                    (CaptureSite::QueryUnpositioned, CaptureSite::QueryPositioned),
                    (CaptureSite::KeyUnpositioned, CaptureSite::KeyPositioned),
                ] {
                    let mut rotated = hf.row(un, layer, position);
                    for head in rotated.chunks_exact_mut(op.head_dim) {
                        rope_rotate_hf_f32(head, position, &inv_freq);
                    }
                    assert_eq!(
                        bits(&rotated),
                        bits(&hf.row(pos, layer, position)),
                        "{name}: layer {layer} position {position} {un:?}"
                    );
                    checked += 1;
                }
            }
            // Layer 0's projection reads only the embedding, which no
            // rotation reaches: its unpositioned q/k are identical
            // across the two modes, so the intervention starts exactly
            // at the rotation.
            if layer == 0 {
                for position in ALL_POSITIONS {
                    for site in [CaptureSite::QueryUnpositioned, CaptureSite::KeyUnpositioned] {
                        assert_eq!(
                            bits(&hf.row(site, 0, position)),
                            bits(&larql.row(site, 0, position)),
                            "{name}: {site:?} at {position} moved before the rotation"
                        );
                    }
                }
            }
        }
    });
    assert!(checked > 0, "no rope layer was checked");
}

#[test]
fn under_hf_f32_decode_and_batch_agree_bit_for_bit() {
    on_both(|name, run| {
        let hf = run(Some(RopeAngles::HfF32));
        assert_eq!(
            hf.decode.last().unwrap(),
            &hf.batch_last,
            "{name}: decode and batch drifted under hf-f32"
        );
    });
}

#[test]
fn hf_f32_is_refused_on_an_image_with_non_softmax_attention() {
    let (_c, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let mut ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let err = ops.set_rope_angles(RopeAngles::HfF32).unwrap_err();
    assert!(err.to_string().contains("softmax attention only"), "{err}");
    // The default stays settable: refusing the intervention never
    // refuses the shipped path.
    ops.set_rope_angles(RopeAngles::Larql).unwrap();
}

// ── 3. the construction ──────────────────────────────────────────────

#[test]
fn the_hf_f32_frequencies_are_hf_s_formula_in_f32() {
    const HEAD_DIM: usize = 256;
    for theta in [10_000.0f64, 1_000_000.0] {
        let plain = hf_f32_inv_freq(HEAD_DIM, theta, None);
        assert_eq!(plain.len(), HEAD_DIM / 2);
        assert_eq!(plain[0].to_bits(), 1.0f32.to_bits(), "base^0 is exactly 1");
        for (i, f) in plain.iter().enumerate() {
            // Spelled out once more, step by step, as HF writes it.
            let exponent = (2 * i) as f32 / HEAD_DIM as f32;
            let expected = 1.0f32 / (theta as f32).powf(exponent);
            assert_eq!(f.to_bits(), expected.to_bits(), "theta {theta} i {i}");
            // And within a few f32 ulps of the exact value: the formula
            // is the right one, not merely the same code twice.
            let exact = theta.powf(-((2 * i) as f64) / HEAD_DIM as f64);
            assert!(
                ((*f as f64) - exact).abs() <= 4.0 * f32::EPSILON as f64 * exact,
                "{i}"
            );
        }
        // Linear divides the f32 table by the factor, in f32, in place.
        let linear = hf_f32_inv_freq(HEAD_DIM, theta, Some(8.0));
        for (p, l) in plain.iter().zip(&linear) {
            assert_eq!(l.to_bits(), (p / 8.0f32).to_bits());
        }
    }
}

#[test]
fn the_two_constructions_differ_somewhere_over_the_registered_range() {
    // Non-vacuity at the table level: over Gemma 3's two bases, its head
    // width and the registered positions, the HF-f32 cos/sin differ from
    // the shipped f64-derived ones in at least one entry.
    const HEAD_DIM: usize = 256;
    let mut differing = 0usize;
    for theta in [10_000.0f64, 1_000_000.0] {
        let hf = hf_f32_inv_freq(HEAD_DIM, theta, None);
        for position in [1usize, 511, 512, 1039] {
            for (i, f) in hf.iter().enumerate() {
                let (cos_hf, sin_hf) = hf_f32_cos_sin(position, *f);
                let mut shipped = vec![0.0f32; HEAD_DIM];
                shipped[i] = 1.0; // x0 = 1, x1 = 0: the rotation returns (cos, sin)
                rope_rotate(&mut shipped, position, theta);
                if cos_hf.to_bits() != shipped[i].to_bits()
                    || sin_hf.to_bits() != shipped[HEAD_DIM / 2 + i].to_bits()
                {
                    differing += 1;
                }
            }
        }
    }
    assert!(differing > 0);
}

#[test]
fn rotate_or_defer_leaves_the_shipped_path_alone_and_takes_over_under_hf_f32() {
    const HEAD_DIM: usize = 4;
    let policy = PositionPolicy::Rope { theta: 10_000.0 };
    let (q0, k0) = (vec![1.0f32, 2.0, 3.0, 4.0], vec![0.5f32, -1.0, 0.25, 2.0]);

    let (mut q, mut k) = (q0.clone(), k0.clone());
    let deferred = rotate_or_defer(RopeAngles::Larql, policy, HEAD_DIM, 3, &mut q, &mut k).unwrap();
    assert_eq!(deferred, policy, "the default hands the plan's policy back");
    assert_eq!(
        (q.clone(), k.clone()),
        (q0.clone(), k0.clone()),
        "and touches nothing"
    );

    let deferred = rotate_or_defer(RopeAngles::HfF32, policy, HEAD_DIM, 3, &mut q, &mut k).unwrap();
    assert_eq!(
        deferred,
        PositionPolicy::None,
        "the backend must not rotate again"
    );
    let inv_freq = hf_f32_inv_freq(HEAD_DIM, 10_000.0, None);
    let (mut eq, mut ek) = (q0.clone(), k0.clone());
    rope_rotate_hf_f32(&mut eq, 3, &inv_freq);
    rope_rotate_hf_f32(&mut ek, 3, &inv_freq);
    assert_eq!((bits(&q), bits(&k)), (bits(&eq), bits(&ek)));

    // Linear: the HF table divided by the factor.
    let linear = PositionPolicy::Linear {
        theta: 10_000.0,
        factor: 8.0,
    };
    let (mut q, mut k) = (q0.clone(), k0.clone());
    rotate_or_defer(RopeAngles::HfF32, linear, HEAD_DIM, 3, &mut q, &mut k).unwrap();
    let inv_freq = hf_f32_inv_freq(HEAD_DIM, 10_000.0, Some(8.0));
    let mut eq = q0.clone();
    rope_rotate_hf_f32(&mut eq, 3, &inv_freq);
    assert_eq!(bits(&q), bits(&eq));

    // No position encoding: nothing to rotate, nothing moves.
    let (mut q, mut k) = (q0.clone(), k0.clone());
    let deferred = rotate_or_defer(
        RopeAngles::HfF32,
        PositionPolicy::None,
        HEAD_DIM,
        3,
        &mut q,
        &mut k,
    )
    .unwrap();
    assert_eq!(deferred, PositionPolicy::None);
    assert_eq!((q, k), (q0, k0));
}

#[test]
fn every_other_rotary_policy_is_refused_under_hf_f32() {
    const HEAD_DIM: usize = 8;
    for policy in [
        PositionPolicy::PartialRope {
            theta: 10_000.0,
            rotary_fraction: 0.5,
            basis: larql_models::config::RotaryFrequencyBasis::HeadWidth,
        },
        PositionPolicy::Relative {
            d_rel: 4,
            extent: 8,
        },
    ] {
        let err = hf_f32_policy_inv_freq(policy, HEAD_DIM).unwrap_err();
        assert!(
            err.to_string().contains("plain and linear RoPE only"),
            "{policy:?}"
        );
        let (mut q, mut k) = (vec![1.0f32; HEAD_DIM], vec![1.0f32; HEAD_DIM]);
        assert!(rotate_or_defer(RopeAngles::HfF32, policy, HEAD_DIM, 1, &mut q, &mut k).is_err());
    }
    assert_eq!(RopeAngles::Larql.name(), "larql");
    assert_eq!(RopeAngles::HfF32.name(), "hf-f32");
}
