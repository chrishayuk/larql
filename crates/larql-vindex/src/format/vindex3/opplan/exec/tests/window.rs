//! STATE-3 gates: window-shadow retirement is EXACT and actually bounded.
//!
//! Two claims, each pinned as measurement. Exactness: a decode (and a
//! prefill) over [`WindowKvState`] is bit-identical to the full row
//! store at EVERY step — freed rows were provably unreachable, so not
//! holding them changes nothing. Boundedness: the sliding layer's
//! resident payload stops growing at its window while the full layer's
//! scales exactly with the context — the memory model as arithmetic at
//! two well-separated lengths, not a label.
//!
//! The miniature fixture is layer 0 Sliding(3) + layer 1 Full, so one
//! decode exercises both regimes side by side.

use super::super::backend::PlanBackend;
use super::super::decode::DecodeSession;
use super::super::kv::{KvState, RowKvState};
use super::super::prefill_plan;
use super::super::production::ProductionBackend;
use super::super::reference::ReferenceBackend;
use super::super::window::WindowKvState;
use super::decode::fixture;
use crate::format::vindex3::fixtures::{G_HEAD_DIM, G_KV_HEADS, G_VOCAB, G_WINDOW};

/// Long enough that the window (3) is far behind, and awkwardly not a
/// multiple of anything in the fixture.
const LONG: usize = 61;
const SHORT: usize = 17;

/// A deterministic non-repeating token stream inside the vocabulary.
fn stream(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + 3) % G_VOCAB) as u32).collect()
}

/// Decode `tokens` over `kv`, returning every step's logits.
fn decode_all(kv: &mut dyn KvState, tokens: &[u32], backend: &impl PlanBackend) -> Vec<Vec<f32>> {
    let (_c, plan, store) = fixture();
    let mut session = DecodeSession::with_kv_state(&plan, &store, backend, kv).unwrap();
    tokens
        .iter()
        .map(|&t| session.step(t).unwrap().logits.expect("output head"))
        .collect()
}

/// Exactness, per backend and per step: the bounded provider and the
/// full store produce the same bits at every position of a long decode.
#[test]
fn a_bounded_state_decodes_bit_identically_to_the_full_store() {
    let tokens = stream(LONG);
    for name in ["reference", "production"] {
        let (full, bounded) = match name {
            "reference" => {
                let backend = ReferenceBackend::new();
                let mut a = RowKvState::default();
                let mut b = WindowKvState::default();
                (
                    decode_all(&mut a, &tokens, &backend),
                    decode_all(&mut b, &tokens, &backend),
                )
            }
            _ => {
                let backend = ProductionBackend::new();
                let mut a = RowKvState::default();
                let mut b = WindowKvState::default();
                (
                    decode_all(&mut a, &tokens, &backend),
                    decode_all(&mut b, &tokens, &backend),
                )
            }
        };
        assert_eq!(
            full, bounded,
            "{name}: bounded and full stores diverged somewhere in {LONG} steps"
        );
    }
}

/// Boundedness as arithmetic: at two well-separated lengths the sliding
/// layer holds exactly its window while the full layer holds exactly the
/// context, and the payload accounting agrees to the byte.
#[test]
fn the_sliding_layer_is_bounded_and_the_full_layer_is_not() {
    let row_bytes = 2 * G_KV_HEADS * G_HEAD_DIM * std::mem::size_of::<f32>();
    let backend = ReferenceBackend::new();
    let mut residencies = Vec::new();
    for n in [SHORT, LONG] {
        let mut kv = WindowKvState::default();
        decode_all(&mut kv, &stream(n), &backend);
        assert_eq!(kv.resident_rows(0), G_WINDOW, "sliding layer at length {n}");
        assert_eq!(kv.resident_rows(1), n, "full layer at length {n}");
        assert_eq!(
            kv.resident_payload_bytes(),
            (G_WINDOW + n) * row_bytes,
            "payload bytes at length {n}"
        );
        residencies.push(kv.resident_payload_bytes());
    }
    // The whole point, stated once more as a difference: growing the
    // context moved ONLY the full layer's share.
    assert_eq!(residencies[1] - residencies[0], (LONG - SHORT) * row_bytes);
}

/// The batch prefill frees shadows exactly as the decode path does, and
/// a decode resumed over a prefilled bounded state matches the full
/// store bit for bit.
#[test]
fn prefill_over_a_bounded_state_matches_the_full_store() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let tokens = stream(LONG);
    let (last, prefix) = tokens.split_last().unwrap();

    let run = |kv: &mut dyn KvState| {
        prefill_plan(&plan, &store, prefix, &backend, kv).unwrap();
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, kv).unwrap();
        session.step(*last).unwrap().logits.expect("output head")
    };

    let mut full = RowKvState::default();
    let mut bounded = WindowKvState::default();
    let full_logits = run(&mut full);
    let bounded_logits = run(&mut bounded);
    assert_eq!(full_logits, bounded_logits);
    assert_eq!(
        bounded.resident_rows(0),
        G_WINDOW,
        "prefill must free the shadow as it goes, not leave it for decode"
    );
    assert_eq!(bounded.resident_rows(1), LONG);
}

/// KV-only, said loudly: a recurrent layer refuses this provider by
/// name instead of running stateless.
#[test]
fn a_recurrent_plan_refuses_the_bounded_provider() {
    let mut kv = WindowKvState::default();
    let err = kv.recurrent_state(0).unwrap_err().to_string();
    assert!(err.contains("WindowKvState"), "{err}");
    assert!(err.contains("refusing"), "{err}");
}
