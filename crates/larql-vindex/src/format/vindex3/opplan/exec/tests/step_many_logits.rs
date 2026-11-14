//! **Every position's logits from one block traversal.**
//!
//! `step_many_logits` must return, for each position of a block, exactly
//! the logits `step` returns at that position, and must leave exactly the
//! continuation state sequential decode leaves. `step_many` keeps its own
//! contract: last position only, unchanged.

use super::golden::{G_TOKENS, G_VOCAB, G_WINDOW};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{KvState, RowKvState};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// Block widths under test.
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// Longest prefix decoded before the block: blocks therefore start before
/// the sliding window fills, straddle its boundary and start beyond it.
const PREFIX_MAX: usize = 3 * G_WINDOW;
/// Positions decoded one at a time after the block, to prove the state the
/// block left behind and not only the logits it returned.
const FOLLOW_ON: usize = G_WINDOW + 2;

/// A deterministic token stream longer than the golden one, inside the vocabulary.
fn token(index: usize) -> u32 {
    (G_TOKENS[index % G_TOKENS.len()] as usize * (index / G_TOKENS.len() + 1) % G_VOCAB) as u32
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn gate<B: PlanBackend>(backend: &B) {
    let (_container, plan, store) = super::decode::fixture();
    for prefix in 0..=PREFIX_MAX {
        for width in WIDTHS {
            let before: Vec<u32> = (0..prefix).map(token).collect();
            let block: Vec<u32> = (prefix..prefix + width).map(token).collect();
            let after: Vec<u32> = (prefix + width..prefix + width + FOLLOW_ON)
                .map(token)
                .collect();

            // Sequential reference: one `step` per position throughout.
            let mut sequential_kv = RowKvState::default();
            let mut sequential =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut sequential_kv).unwrap();
            for t in &before {
                sequential.step(*t).unwrap();
            }
            let expected: Vec<Vec<u32>> = block
                .iter()
                .map(|t| bits(&sequential.step(*t).unwrap().logits.unwrap()))
                .collect();
            let expected_after: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&sequential.step(*t).unwrap().logits.unwrap()))
                .collect();

            // The block, in one traversal, every position's logits.
            let mut kv = RowKvState::default();
            let mut session =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut kv).unwrap();
            for t in &before {
                session.step(*t).unwrap();
            }
            let rows = session.step_many_logits(&block).unwrap();
            assert_eq!(rows.len(), width);
            let got: Vec<Vec<u32>> = rows.iter().map(|row| bits(row)).collect();
            assert_eq!(
                got, expected,
                "prefix {prefix}, width {width}: block logits"
            );
            assert_eq!(session.position(), prefix + width);
            // The state it left: every later position still agrees.
            let got_after: Vec<Vec<u32>> = after
                .iter()
                .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
                .collect();
            assert_eq!(
                got_after, expected_after,
                "prefix {prefix}, width {width}: follow-on"
            );

            // `step_many` keeps its contract: the last position's logits.
            let mut last_kv = RowKvState::default();
            let mut last =
                DecodeSession::with_kv_state(&plan, &store, backend, &mut last_kv).unwrap();
            for t in &before {
                last.step(*t).unwrap();
            }
            let only_last = last.step_many(&block).unwrap().logits.unwrap();
            assert_eq!(bits(&only_last), expected[width - 1], "step_many changed");
        }
    }
}

#[test]
fn every_position_matches_sequential_decode_production() {
    gate(&ProductionBackend::new());
}

#[test]
fn every_position_matches_sequential_decode_reference() {
    gate(&ReferenceBackend::new());
}

#[test]
fn a_refused_block_moves_no_state() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = ProductionBackend::new();
    let mut kv = RowKvState::default();
    {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        session.step(token(0)).unwrap();
        let error = session.step_many_logits(&[]).unwrap_err().to_string();
        assert!(error.contains("given none"), "{error}");
        // A bad id in the LAST place: the earlier tokens must not have run.
        let error = session
            .step_many_logits(&[token(1), token(2), G_VOCAB as u32])
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the embedding table"), "{error}");
        assert_eq!(session.position(), 1);
    }
    assert_eq!(kv.position(), 1);
    assert_eq!(kv.rows(0).end(), 1, "a refused block appended rows");
}
