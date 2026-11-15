//! **A block of positions crosses to remote experts once per layer.**
//!
//! The block traversal (`step_many_logits`) over coordinator operands with
//! a bound provider must give, for every position, exactly the logits
//! sequential decode gives, local or remote, while sending each contacted
//! worker ONE multi-row frame per routed layer and no single-row frame.
//!
//! Real expert workers run in-process behind the stub transport of
//! `routed_expert_rows`, which counts each path separately.
use super::routed_expert_rows::{bits, topologies, transport, Fault, Probe};
use super::*;
use crate::vindex3::routed_block::RoutedBlockSession;
use crate::vindex3::routed_experts;
use larql_vindex::format::vindex3::fixtures_routed::{
    miniature_routed, LAYERS, TOP_K, VOCAB, WINDOW,
};
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

/// Block widths under test.
const WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// Longest prefix decoded before the block: blocks start before the sliding
/// window fills, straddle its boundary and start beyond it.
const PREFIX_MAX: usize = 3 * WINDOW;
/// Positions decoded one at a time after the block.
const FOLLOW_ON: usize = WINDOW + 2;
/// Constants of the deterministic token stream.
const TOKEN_STEP: usize = 7;
const TOKEN_OFFSET: usize = 3;

fn token(index: usize) -> u32 {
    ((index * TOKEN_STEP + TOKEN_OFFSET) % VOCAB) as u32
}

/// One `step` per token over `ops`, every position's logits.
fn sequential(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    kv: &mut RowKvState,
    tokens: &[u32],
) -> Vec<Vec<u32>> {
    let mut session = DecodeSession::over_prepared(plan, ops, backend, kv).unwrap();
    tokens
        .iter()
        .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
        .collect()
}

#[test]
fn every_block_position_matches_sequential_decode_with_one_exchange_per_worker_per_layer() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let local = Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new())
        .unwrap()
        .prepare_slice(ExecutionSlice::Full)
        .unwrap();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for topology in topologies() {
        let probe = Probe::default();
        let remote = routed_experts::prepare_coordinator(
            container.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            transport(container.path(), &topology, &probe),
        )
        .unwrap();
        for prefix in 0..=PREFIX_MAX {
            for width in WIDTHS {
                let before: Vec<u32> = (0..prefix).map(token).collect();
                let block: Vec<u32> = (prefix..prefix + width).map(token).collect();
                let after: Vec<u32> = (prefix + width..prefix + width + FOLLOW_ON)
                    .map(token)
                    .collect();
                let all: Vec<u32> = before.iter().chain(&block).chain(&after).copied().collect();

                // Reference 1: everything local, one position at a time.
                let mut local_kv = RowKvState::default();
                let local_logits = sequential(
                    local.plan(),
                    local.operands(),
                    &backend,
                    &mut local_kv,
                    &all,
                );
                // Reference 2: remote experts, one position at a time,
                // through the existing single-row frames.
                let mut remote_kv = RowKvState::default();
                let remote_logits =
                    sequential(runtime.plan(), &remote, &backend, &mut remote_kv, &all);
                assert_eq!(
                    remote_logits, local_logits,
                    "sequential remote differs from local"
                );

                // The block: prefix sequentially, then one traversal.
                let mut kv = RowKvState::default();
                let mut session =
                    DecodeSession::over_prepared(runtime.plan(), &remote, &backend, &mut kv)
                        .unwrap();
                for t in &before {
                    session.step(*t).unwrap();
                }
                probe.single_calls.store(0, Ordering::SeqCst);
                probe.rows_calls.store(0, Ordering::SeqCst);
                probe.exchanges.lock().unwrap().clear();
                let rows = session.step_many_logits(&block).unwrap();
                let got: Vec<Vec<u32>> = rows.iter().map(|row| bits(row)).collect();
                assert_eq!(
                    got,
                    local_logits[prefix..prefix + width],
                    "{topology:?} prefix {prefix} width {width}: block logits"
                );

                // Transport: no single-row frame, and per routed layer one
                // multi-row frame per CONTACTED worker.
                assert_eq!(
                    probe.single_calls.load(Ordering::SeqCst),
                    0,
                    "a block used the single-row route"
                );
                let exchanges = probe.exchanges.lock().unwrap().clone();
                let distinct: BTreeSet<(usize, usize)> = exchanges
                    .iter()
                    .map(|(layer, shard, _)| (*layer, *shard))
                    .collect();
                assert_eq!(
                    distinct.len(),
                    exchanges.len(),
                    "a worker was asked twice in a layer"
                );
                let layers: BTreeSet<usize> = exchanges.iter().map(|(layer, ..)| *layer).collect();
                assert_eq!(layers, (0..LAYERS).collect::<BTreeSet<_>>());
                assert!(exchanges.len() <= LAYERS * topology.len());
                assert!(exchanges.iter().all(|(_, _, rows)| *rows <= width));
                if topology.len() == 1 {
                    // One worker owns everything: exactly one frame per
                    // layer, carrying every position of the block.
                    assert_eq!(exchanges.len(), LAYERS);
                    assert!(exchanges.iter().all(|(_, _, rows)| *rows == width));
                }
                assert_eq!(probe.rows_calls.load(Ordering::SeqCst), exchanges.len());

                // The state the block left: later positions still agree.
                let follow: Vec<Vec<u32>> = after
                    .iter()
                    .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
                    .collect();
                assert_eq!(
                    follow,
                    local_logits[prefix + width..],
                    "{topology:?} prefix {prefix} width {width}: follow-on"
                );
            }
        }
        // Sanity on the workload: the widest block really shares experts.
        assert!(TOP_K * WIDTHS[WIDTHS.len() - 1] > routed_experts_in_fixture());
    }
}

/// Experts per layer in the routed fixture.
fn routed_experts_in_fixture() -> usize {
    larql_vindex::format::vindex3::fixtures_routed::EXPERTS
}

/// A coordinator over in-process workers, with its probe.
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

/// Tokens verified and then discarded: deliberately not the real stream,
/// so any trace of them left behind changes a later result.
fn rejected(index: usize) -> u32 {
    ((token(index) as usize + TOKEN_STEP + 1) % VOCAB) as u32
}

fn rows_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter().map(|row| bits(row)).collect()
}

#[test]
fn committing_any_prefix_of_a_block_equals_having_decoded_only_that_prefix() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let width = WIDTHS[WIDTHS.len() - 1];
    for topology in topologies() {
        let (ops, _probe) = coordinator(container.path(), &runtime, &topology);
        // Starts before, on and beyond the sliding-window boundary.
        for prefix in [0, WINDOW - 1, WINDOW, 2 * WINDOW + 1] {
            for kept in 0..=width {
                let before: Vec<u32> = (0..prefix).map(token).collect();
                // The first `kept` block tokens are real; the rest are not.
                let block: Vec<u32> = (0..width)
                    .map(|i| {
                        if i < kept {
                            token(prefix + i)
                        } else {
                            rejected(prefix + i)
                        }
                    })
                    .collect();
                let after: Vec<u32> = (prefix + kept..prefix + kept + FOLLOW_ON)
                    .map(token)
                    .collect();

                // Reference: a session that only ever saw the kept tokens.
                let clean: Vec<u32> = (0..prefix + kept + FOLLOW_ON).map(token).collect();
                let mut clean_kv = RowKvState::default();
                let expected = sequential(runtime.plan(), &ops, &backend, &mut clean_kv, &clean);

                let mut session =
                    RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                        .unwrap();
                if !before.is_empty() {
                    session.verify(&before).unwrap();
                    session.commit(before.len()).unwrap();
                }
                assert_eq!(session.position(), prefix);
                let rows = rows_bits(&session.verify(&block).unwrap());
                assert_eq!(
                    session.position(),
                    prefix,
                    "verify moved the committed position"
                );
                assert_eq!(
                    rows[..kept],
                    expected[prefix..prefix + kept],
                    "{topology:?} prefix {prefix} kept {kept}: kept rows"
                );
                session.commit(kept).unwrap();
                assert_eq!(session.position(), prefix + kept);
                // Decode on, one position per block.
                let mut follow = Vec::new();
                for t in &after {
                    follow.extend(rows_bits(&session.verify(&[*t]).unwrap()));
                    session.commit(1).unwrap();
                }
                assert_eq!(
                    follow,
                    expected[prefix + kept..],
                    "{topology:?} prefix {prefix} kept {kept}: decode after partial commit"
                );
            }
        }
    }
}

#[test]
fn chained_verify_and_partial_commit_leaves_no_trace_of_rejected_positions() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    // Verify eight, commit three; verify four, commit two; decode on.
    let (first, keep_first, second, keep_second) = (8usize, 3usize, 4usize, 2usize);
    let committed = keep_first + keep_second;
    for topology in topologies() {
        let (ops, probe) = coordinator(container.path(), &runtime, &topology);
        // The chain is run from two starts so it also crosses the window.
        for start in [0, WINDOW - 1] {
            let clean: Vec<u32> = (0..start + committed + FOLLOW_ON).map(token).collect();
            let mut clean_kv = RowKvState::default();
            let expected = sequential(runtime.plan(), &ops, &backend, &mut clean_kv, &clean);

            let mut session =
                RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                    .unwrap();
            if start > 0 {
                let before: Vec<u32> = (0..start).map(token).collect();
                session.verify(&before).unwrap();
                session.commit(start).unwrap();
            }
            probe.single_calls.store(0, Ordering::SeqCst);

            let block_one: Vec<u32> = (0..first)
                .map(|i| {
                    if i < keep_first {
                        token(start + i)
                    } else {
                        rejected(start + i)
                    }
                })
                .collect();
            let one = rows_bits(&session.verify(&block_one).unwrap());
            assert_eq!(one[..keep_first], expected[start..start + keep_first]);
            // A second block before the first is committed is refused.
            assert!(session.verify(&block_one).is_err());
            session.commit(keep_first).unwrap();

            let base = start + keep_first;
            let block_two: Vec<u32> = (0..second)
                .map(|i| {
                    if i < keep_second {
                        token(base + i)
                    } else {
                        rejected(base + i)
                    }
                })
                .collect();
            let two = rows_bits(&session.verify(&block_two).unwrap());
            assert_eq!(two[..keep_second], expected[base..base + keep_second]);
            session.commit(keep_second).unwrap();
            assert_eq!(session.position(), start + committed);

            let mut follow = Vec::new();
            for index in start + committed..start + committed + FOLLOW_ON {
                follow.extend(rows_bits(&session.verify(&[token(index)]).unwrap()));
                session.commit(1).unwrap();
            }
            assert_eq!(
                follow,
                expected[start + committed..],
                "{topology:?} start {start}"
            );
            assert_eq!(probe.single_calls.load(Ordering::SeqCst), 0);

            // Verifying and discarding the same block twice gives the same logits.
            let block: Vec<u32> = (0..first)
                .map(|i| rejected(start + committed + FOLLOW_ON + i))
                .collect();
            let once = rows_bits(&session.verify(&block).unwrap());
            session.commit(0).unwrap();
            let twice = rows_bits(&session.verify(&block).unwrap());
            session.commit(0).unwrap();
            assert_eq!(once, twice);
            assert_eq!(session.position(), start + committed + FOLLOW_ON);
        }
    }
}

#[test]
fn a_worker_failure_at_any_layer_commits_nothing_and_invalidates_the_session() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let prefix = WINDOW + 1;
    let block: Vec<u32> = (prefix..prefix + WIDTHS[2]).map(token).collect();
    for topology in topologies() {
        // The fixture's routed layers: first and last.
        for layer in 0..LAYERS {
            let (ops, probe) = coordinator(container.path(), &runtime, &topology);
            let before: Vec<u32> = (0..prefix).map(token).collect();
            let mut session =
                RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                    .unwrap();
            session.verify(&before).unwrap();
            session.commit(prefix).unwrap();

            *probe.fault_layer.lock().unwrap() = Some(layer);
            *probe.fault.lock().unwrap() = Some(Fault::Fail);
            assert!(
                session.verify(&block).is_err(),
                "layer {layer}: a failed block returned logits"
            );
            assert_eq!(
                session.position(),
                prefix,
                "layer {layer}: a failed block was committed"
            );
            // Invalid from here: nothing can be committed or verified.
            *probe.fault.lock().unwrap() = None;
            assert!(session.commit(0).is_err());
            assert!(session.commit(1).is_err());
            let error = session.verify(&block).unwrap_err().to_string();
            assert!(error.contains("invalid"), "{error}");
            assert_eq!(session.position(), prefix);

            // Recovery is a fresh session replaying the committed prefix.
            let all: Vec<u32> = before.iter().chain(&block).copied().collect();
            let mut clean_kv = RowKvState::default();
            let expected = sequential(runtime.plan(), &ops, &backend, &mut clean_kv, &all);
            let mut fresh =
                RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                    .unwrap();
            fresh.verify(&before).unwrap();
            fresh.commit(prefix).unwrap();
            assert_eq!(
                rows_bits(&fresh.verify(&block).unwrap()),
                expected[prefix..]
            );
        }
    }
}

#[test]
fn the_session_refuses_what_it_cannot_do_before_any_state_moves() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    // Operands that are not a routed-expert coordinator.
    let full = Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new())
        .unwrap()
        .prepare_slice(ExecutionSlice::Full)
        .unwrap();
    let error =
        match RoutedBlockSession::new(full.plan(), full.operands(), &backend, &row(full.plan())) {
            Ok(_) => panic!("a block session was built over resident experts"),
            Err(e) => e.to_string(),
        };
    assert!(error.contains("coordinator operands"), "{error}");

    let topology = topologies()[1].clone();
    let (ops, probe) = coordinator(container.path(), &runtime, &topology);
    // Every shipped provider but the row provider has no rollback, and is
    // refused before any state exists.
    let registry = larql_kv::shipped_continuations();
    let geometry = plan_continuation_geometry(runtime.plan()).unwrap();
    let mut refused = 0;
    for identity in registry.identities() {
        let Ok(selected) = registry.select(&identity, &ContinuationConfig::empty(), &geometry)
        else {
            continue;
        };
        if identity == RowKvState::identity() {
            continue;
        }
        let error = match RoutedBlockSession::new(runtime.plan(), &ops, &backend, &selected) {
            Ok(_) => panic!("a block session was built over {identity:?}"),
            Err(e) => e.to_string(),
        };
        assert!(error.contains("can be rewound"), "{error}");
        refused += 1;
    }
    assert!(refused > 0, "no other provider was selectable to refuse");
    assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);

    let mut session =
        RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan())).unwrap();
    assert!(session.commit(0).is_err(), "nothing verified yet");
    assert!(session.verify(&[]).is_err(), "empty block");
    assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);
    // A bad id is a caller error, found before any layer runs; it leaves
    // the session valid and nothing pending.
    session.verify(&[token(0), token(1)]).unwrap();
    assert!(session.commit(3).is_err(), "more than was verified");
    session.commit(2).unwrap();
    assert_eq!(session.position(), 2);
}

/// Positions stepped one at a time through the block session, around and
/// between blocks.
const STEPS: usize = WINDOW + 3;

#[test]
fn a_step_equals_sequential_decode_on_the_single_row_route_around_blocks() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    // Step, verify four and keep two, then step on across the window.
    let (width, kept) = (WIDTHS[2], 2usize);
    for topology in topologies() {
        let (ops, probe) = coordinator(container.path(), &runtime, &topology);
        let committed = STEPS + kept + STEPS;
        let clean: Vec<u32> = (0..committed).map(token).collect();
        let mut clean_kv = RowKvState::default();
        let expected = sequential(runtime.plan(), &ops, &backend, &mut clean_kv, &clean);

        let mut session =
            RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan())).unwrap();
        let mut got = Vec::new();
        probe.single_calls.store(0, Ordering::SeqCst);
        probe.rows_calls.store(0, Ordering::SeqCst);
        for index in 0..STEPS {
            got.push(bits(&session.step(token(index)).unwrap()));
            assert_eq!(session.position(), index + 1, "a step commits its position");
        }
        // Steps use the single-row route only.
        assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);
        let singles = probe.single_calls.load(Ordering::SeqCst);
        assert!(singles > 0);
        if topology.len() == 1 {
            assert_eq!(
                singles,
                STEPS * LAYERS,
                "one single-row frame per layer per step"
            );
        }

        let block: Vec<u32> = (0..width)
            .map(|i| {
                if i < kept {
                    token(STEPS + i)
                } else {
                    rejected(STEPS + i)
                }
            })
            .collect();
        let rows = rows_bits(&session.verify(&block).unwrap());
        got.extend(rows[..kept].iter().cloned());
        session.commit(kept).unwrap();

        probe.rows_calls.store(0, Ordering::SeqCst);
        for index in STEPS + kept..committed {
            got.push(bits(&session.step(token(index)).unwrap()));
        }
        assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);
        assert_eq!(session.position(), committed);
        assert_eq!(
            got, expected,
            "{topology:?}: steps around a partly committed block"
        );
    }
}

#[test]
fn a_step_is_refused_while_a_block_awaits_commit_and_leaves_it_pending() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let topology = topologies()[0].clone();
    let (ops, probe) = coordinator(container.path(), &runtime, &topology);
    let mut session =
        RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan())).unwrap();
    session.verify(&[token(0), token(1)]).unwrap();
    let singles = probe.single_calls.load(Ordering::SeqCst);
    let error = session.step(token(0)).unwrap_err().to_string();
    assert!(error.contains("awaiting commit"), "{error}");
    assert_eq!(
        probe.single_calls.load(Ordering::SeqCst),
        singles,
        "a refused step ran"
    );
    assert_eq!(session.position(), 0);
    // The block is still pending, and the session still usable.
    session.commit(2).unwrap();
    session.step(token(2)).unwrap();
    assert_eq!(session.position(), 3);
}

#[test]
fn a_failed_step_commits_nothing_and_invalidates_the_session() {
    let container = container_with(miniature_routed);
    let backend = ProductionBackend::new();
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for topology in topologies() {
        for layer in 0..LAYERS {
            let (ops, probe) = coordinator(container.path(), &runtime, &topology);
            let mut session =
                RoutedBlockSession::new(runtime.plan(), &ops, &backend, &row(runtime.plan()))
                    .unwrap();
            session.step(token(0)).unwrap();
            *probe.single_fault_layer.lock().unwrap() = Some(layer);
            assert!(
                session.step(token(1)).is_err(),
                "layer {layer}: a failed step returned logits"
            );
            assert_eq!(
                session.position(),
                1,
                "layer {layer}: a failed step was committed"
            );
            *probe.single_fault_layer.lock().unwrap() = None;
            let error = session.step(token(1)).unwrap_err().to_string();
            assert!(error.contains("invalid"), "{error}");
            assert!(session.verify(&[token(1)]).is_err());
            assert!(session.commit(0).is_err());
            assert_eq!(session.position(), 1);
        }
    }
}
