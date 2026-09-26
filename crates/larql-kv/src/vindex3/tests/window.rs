//! CONTINUATION-WINDOW-1 W2: `window/v1`, the horizon-bounded provider.
//!
//! Selection (shipped, KV-only), retention after every append (the frozen
//! `base == required_start(p)`, `end == p + 1`, at most `w` rows held on a
//! sliding layer, full layers whole, rows adopted not copied), exactness
//! against `row/v1` through prefill, resumed prefill and decode, and a
//! handoff that resumes at a NON-ZERO base: physical history starting past
//! position 0 still serves absolute positions and keeps advancing.
//! Physical release and the length ladder are W3's, on the MEM-1 harness.

use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_TOKENS, G_WINDOW};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationRegistryError;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{
    HistoryRange, KvState, LayerKvGeometry, RowKvState,
};
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;

use super::super::{shipped_continuations, WindowKvState};
use super::registry_parity::open;

/// Tokens after the fixture prompt: enough that layer 0 (window 3) drains.
const RESUME: [u32; 3] = [5, 9, 13];
const DECODE: [u32; 4] = [1, 2, 3, 4];

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn window_v1_is_shipped_and_holds_only_kv() {
    let registry = shipped_continuations();
    assert!(registry.identities().contains(&WindowKvState::identity()));
    let (_container, plan, _store) = open(hybrid_kda_mla_f32_model, "w2-hybrid");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let refused = registry
        .select(
            &WindowKvState::identity(),
            &ContinuationConfig::empty(),
            &geometry,
        )
        .err()
        .expect("a hybrid plan needs regions window/v1 does not hold");
    assert!(
        matches!(refused, ContinuationRegistryError::Unsupported { .. }),
        "refused at selection, by capability: {refused}"
    );
}

#[test]
fn retention_follows_the_plan_after_every_append() {
    const W: usize = 3;
    let mut state = WindowKvState::new();
    state.prepare(&[
        LayerKvGeometry {
            kv_dim: 2,
            window: Some(W),
            history: HistoryRange::Trailing(W),
        },
        LayerKvGeometry {
            kv_dim: 2,
            window: None,
            history: HistoryRange::Full,
        },
    ]);
    for p in 0..10 {
        for layer in 0..2 {
            let key = vec![p as f32, layer as f32];
            let adopted = key.as_ptr();
            state.append(layer, key, vec![-(p as f32), 0.5]);
            let view = state.rows(layer);
            assert_eq!(view.end(), p + 1, "end is logical: every row ever appended");
            let expected_base = if layer == 0 {
                (p + 1).saturating_sub(W)
            } else {
                0
            };
            assert_eq!(
                view.base(),
                expected_base,
                "layer {layer} after position {p}"
            );
            assert!(view.end() - view.base() <= if layer == 0 { W } else { p + 1 });
            assert_eq!(
                view.key(p).as_ptr(),
                adopted,
                "the row is adopted, never copied"
            );
            for q in view.base()..view.end() {
                assert_eq!(
                    view.key(q),
                    &[q as f32, layer as f32],
                    "absolute position {q}"
                );
                assert_eq!(view.value(q)[0], -(q as f32));
            }
        }
    }
}

#[test]
fn window_v1_is_bit_identical_to_row_v1_and_holds_its_window() {
    let (_container, plan, store) = open(miniature_glimmer, "w2-exact");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let registry = shipped_continuations();
    let backend = ReferenceBackend::new();
    let mut logits = Vec::new();
    for identity in [RowKvState::identity(), WindowKvState::identity()] {
        let selected = registry
            .select(&identity, &ContinuationConfig::empty(), &geometry)
            .unwrap();
        let mut state = selected.build();
        let mut run = Vec::new();
        run.push(bits(
            &prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *state)
                .unwrap()
                .logits
                .unwrap(),
        ));
        run.push(bits(
            &prefill_plan(&plan, &store, &RESUME, &backend, &mut *state)
                .unwrap()
                .logits
                .unwrap(),
        ));
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut *state).unwrap();
        for &token in &DECODE {
            run.push(bits(&session.step(token).unwrap().logits.unwrap()));
        }
        drop(session);
        if identity == WindowKvState::identity() {
            let n = G_TOKENS.len() + RESUME.len() + DECODE.len();
            let sliding = state.rows(0);
            assert_eq!(
                (sliding.base(), sliding.end()),
                (n - G_WINDOW, n),
                "the sliding layer holds its window"
            );
            let full = state.rows(1);
            assert_eq!(
                (full.base(), full.end()),
                (0, n),
                "the full layer holds everything"
            );
        }
        logits.push(run);
    }
    assert_eq!(
        logits[0], logits[1],
        "window/v1 must decode exactly as row/v1"
    );
}

#[test]
fn a_handoff_resumes_at_a_non_zero_base_and_keeps_advancing() {
    let (_container, plan, store) = open(miniature_glimmer, "w2-resume");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let registry = shipped_continuations();
    let backend = ReferenceBackend::new();
    let config = ContinuationConfig::empty();

    // Reference: row/v1, uninterrupted.
    let mut reference = registry
        .select(&RowKvState::identity(), &config, &geometry)
        .unwrap()
        .build();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *reference).unwrap();
    prefill_plan(&plan, &store, &RESUME, &backend, &mut *reference).unwrap();
    let mut expected = Vec::new();
    {
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut *reference).unwrap();
        for &token in &DECODE {
            expected.push(bits(&session.step(token).unwrap().logits.unwrap()));
        }
    }

    // window/v1 sealed in a handoff, filled until its sliding layer drains.
    let selected = registry
        .select(&WindowKvState::identity(), &config, &geometry)
        .unwrap();
    let authority = selected.authority().clone();
    let mut handoff = selected.begin();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, handoff.state_mut()).unwrap();
    prefill_plan(&plan, &store, &RESUME, &backend, handoff.state_mut()).unwrap();
    let base_at_seal = handoff.state().rows(0).base();
    assert!(
        base_at_seal > 0,
        "the sliding layer must have drained before the handoff"
    );

    // A different authority refuses (a refused resume consumes its
    // handoff, so this uses a second one filled the same way).
    let row_authority = registry
        .select(&RowKvState::identity(), &config, &geometry)
        .unwrap()
        .authority()
        .clone();
    let mut other = registry
        .select(&WindowKvState::identity(), &config, &geometry)
        .unwrap()
        .begin();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, other.state_mut()).unwrap();
    assert!(
        other.resume(&row_authority).is_err(),
        "row/v1's authority must refuse window state"
    );

    // The same authority resumes with the physical base intact.
    let mut resumed = handoff.resume(&authority).unwrap();
    assert_eq!(
        resumed.state().rows(0).base(),
        base_at_seal,
        "resume keeps the physical base"
    );

    let mut got = Vec::new();
    {
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, resumed.state_mut()).unwrap();
        for &token in &DECODE {
            got.push(bits(&session.step(token).unwrap().logits.unwrap()));
        }
    }
    assert_eq!(got, expected, "decoding from a non-zero base must be exact");
    let after = resumed.state().rows(0);
    assert!(
        after.base() > base_at_seal,
        "the base keeps advancing after resume"
    );
    assert_eq!(after.end() - after.base(), G_WINDOW);
}
