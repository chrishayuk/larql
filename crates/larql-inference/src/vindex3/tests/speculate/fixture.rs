//! Exact ids and frame accounting against the routed fixture.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use super::super::routed_expert_rows::{topologies, transport, Probe};
use super::super::*;
use crate::speculation::{greedy_id, propose};
use crate::vindex3::routed_block::RoutedBlockSession;
use crate::vindex3::routed_experts;
use crate::vindex3::speculate::{greedy_speculate, prefill, Proposer};
use larql_vindex::format::vindex3::fixtures_routed::{miniature_routed, LAYERS, VOCAB, WINDOW};
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;

/// Widths under test.
const WIDTHS: [usize; 3] = [2, 4, 8];
/// Ids generated: well past the sliding window.
const GENERATE: usize = 3 * WINDOW + 5;
/// Prefill block width.
const PREFILL_WIDTH: usize = 8;

/// A prompt that repeats itself, so prompt lookup finds matches.
fn prompt() -> Vec<u32> {
    let motif: Vec<u32> = (0..5).map(|i| ((i * 3 + 1) % VOCAB) as u32).collect();
    motif
        .iter()
        .chain(&motif)
        .chain(&motif[..3])
        .copied()
        .collect()
}

fn coordinator(
    container: &Path,
    runtime: &Vindex3Runtime<ProductionBackend>,
    topology: &[(usize, usize)],
) -> (PreparedOperands, Probe) {
    let probe = Probe::default();
    let ops = routed_experts::prepare_coordinator(
        container,
        runtime.plan(),
        runtime.operands(),
        runtime.backend(),
        transport(container, topology, &probe),
    )
    .unwrap();
    (ops, probe)
}

/// Sequential greedy decoding over the remote experts, one `step` per id.
fn reference(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    prompt: &[u32],
    n: usize,
) -> Vec<u32> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    let mut last = None;
    for token in prompt {
        last = session.step(*token).unwrap().logits;
    }
    let mut out = vec![greedy_id(&last.unwrap()).unwrap()];
    while out.len() < n {
        let logits = session.step(*out.last().unwrap()).unwrap().logits.unwrap();
        out.push(greedy_id(&logits).unwrap());
    }
    out
}

#[test]
fn speculative_ids_equal_sequential_greedy_over_remote_experts() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let prompt = prompt();
    for topology in topologies() {
        let (ops, _probe) = coordinator(container.path(), &runtime, &topology);
        let expected = reference(runtime.plan(), &ops, &backend, &prompt, GENERATE);
        // The true continuation, for an oracle and a corrupted proposer.
        let full: Vec<u32> = prompt.iter().chain(&expected).copied().collect();
        let continuation = |ctx: &[u32], max: usize| -> Vec<u32> {
            full[ctx.len().min(full.len())..]
                .iter()
                .take(max)
                .copied()
                .collect()
        };
        for width in WIDTHS {
            type Make<'a> = Box<dyn Proposer + 'a>;
            let proposers: Vec<(&str, Make)> = vec![
                ("prompt lookup", Box::new(propose)),
                (
                    "oracle",
                    Box::new(|ctx: &[u32], _: usize, max: usize| continuation(ctx, max)),
                ),
                (
                    "corrupted last",
                    Box::new(|ctx: &[u32], _: usize, max: usize| {
                        let mut guess = continuation(ctx, max);
                        if let Some(token) = guess.last_mut() {
                            *token = (*token + 1) % VOCAB as u32;
                        }
                        guess
                    }),
                ),
                ("none", Box::new(|_: &[u32], _: usize, _: usize| Vec::new())),
            ];
            for (name, mut proposer) in proposers {
                let mut session =
                    RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                        .unwrap();
                let first = prefill(&mut session, &prompt, PREFILL_WIDTH).unwrap();
                let run = greedy_speculate(
                    &mut session,
                    &prompt,
                    first,
                    GENERATE,
                    width,
                    &HashSet::new(),
                    &mut *proposer,
                )
                .unwrap();
                run.reconcile().unwrap();
                assert_eq!(run.ids, expected, "{topology:?} width {width} {name}");
                if name == "oracle" {
                    assert!(run.blocks.iter().all(|b| b.rejected_at.is_none()));
                }
                if name == "corrupted last" {
                    assert!(run
                        .blocks
                        .iter()
                        .all(|b| b.rejected_at == Some(b.proposed - 1)));
                }
            }
        }
    }
}

#[test]
fn blocks_use_one_multi_row_frame_per_layer_and_steps_single_row_frames_only() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let prompt = prompt();
    // One worker owns every expert: exact counts per layer.
    let topology = topologies()[0].clone();
    let (ops, probe) = coordinator(container.path(), &runtime, &topology);
    for width in WIDTHS {
        let mut session =
            RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan())).unwrap();
        let first = prefill(&mut session, &prompt, PREFILL_WIDTH).unwrap();
        probe.single_calls.store(0, Ordering::SeqCst);
        probe.rows_calls.store(0, Ordering::SeqCst);
        probe.exchanges.lock().unwrap().clear();
        let mut calls = 0usize;
        // Alternate proposing and not, so both routes run.
        let mut alternate = |ctx: &[u32], source_len: usize, max: usize| {
            calls += 1;
            if calls.is_multiple_of(2) {
                Vec::new()
            } else {
                propose(ctx, source_len, max)
            }
        };
        let run = greedy_speculate(
            &mut session,
            &prompt,
            first,
            GENERATE,
            width,
            &HashSet::new(),
            &mut alternate,
        )
        .unwrap();
        run.reconcile().unwrap();
        assert_eq!(
            probe.rows_calls.load(Ordering::SeqCst),
            LAYERS * run.blocks.len(),
            "width {width}"
        );
        assert_eq!(
            probe.single_calls.load(Ordering::SeqCst),
            LAYERS * run.single_steps,
            "width {width}"
        );
        // Each block's frames carry exactly that block's positions.
        let exchanges = probe.exchanges.lock().unwrap().clone();
        let widths: Vec<usize> = exchanges.chunks(LAYERS).map(|layer| layer[0].2).collect();
        assert_eq!(
            widths,
            run.blocks.iter().map(|b| b.width).collect::<Vec<_>>()
        );
        assert!(exchanges
            .chunks(LAYERS)
            .all(|layer| layer.iter().all(|e| e.2 == layer[0].2)));
    }
}
