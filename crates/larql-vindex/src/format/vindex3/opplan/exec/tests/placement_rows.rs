//! **Multi-row routed placement is the single-row operation, several rows
//! at a time.**
//!
//! One provider call carries every row of a layer. Each row is routed and
//! reduced exactly as the single-row path does it, so a row's output is the
//! same bits whether it travelled alone or in a batch, whatever order the
//! provider answered in. A response that does not answer exactly what was
//! asked fails the whole call: there is no partial result to commit.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use larql_models::config::{ExpertRoutingPolicy, GateUpLayout, MoeRouterKind};
use larql_models::{Activation, ExpertGatePolicy};

use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::backend::{
    ExpertSlices, PlanBackend, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::routed_experts::{
    ExpertOutput, ExpertRow, ExpertTransformCall, ExpertWeights, RoutedExpertProvider,
    RowExpertOutput,
};

const HIDDEN: usize = 4;
const INTER: usize = 2;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const LAYER: usize = 0;
/// Rows chosen so the router sends them to different expert pairs that
/// still overlap: the batch must both differ per row and share experts.
const INPUTS: [[f32; HIDDEN]; 5] = [
    [0.75, -0.5, 0.25, 1.0],
    [-1.0, 0.5, 0.75, -0.25],
    [0.125, 0.875, -0.625, 0.5],
    [1.5, 1.0, -1.25, -0.75],
    [-0.375, -1.125, 0.625, 0.25],
];
/// Constants of the deterministic weight pattern: distinct per expert and
/// per element, small enough to stay finite through the activation.
const PATTERN_EXPERT: usize = 31;
const PATTERN_INDEX: usize = 7;
const PATTERN_MODULUS: usize = 13;
const PATTERN_SCALE: f32 = 8.0;
const PATTERN_OFFSET: f32 = 0.75;

fn pattern(expert: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            ((expert * PATTERN_EXPERT + i * PATTERN_INDEX) % PATTERN_MODULUS) as f32 / PATTERN_SCALE
                - PATTERN_OFFSET
        })
        .collect()
}

struct Bank {
    gate_up: Vec<Vec<f32>>,
    down: Vec<Vec<f32>>,
    router: Vec<f32>,
}

impl Bank {
    fn new() -> Self {
        Self {
            gate_up: (0..EXPERTS)
                .map(|e| pattern(e, 2 * INTER * HIDDEN))
                .collect(),
            down: (0..EXPERTS)
                .map(|e| pattern(e + EXPERTS, HIDDEN * INTER))
                .collect(),
            router: pattern(2 * EXPERTS, EXPERTS * HIDDEN),
        }
    }
}

fn slices(values: &[Vec<f32>]) -> Vec<WeightSlice<'_>> {
    values.iter().map(|v| WeightSlice::F32(v)).collect()
}

fn call<'a>(
    x: &'a [f32],
    router: &'a [f32],
    gate_up: &'a [WeightSlice<'a>],
    down: &'a [WeightSlice<'a>],
) -> RoutedFfnCall<'a> {
    RoutedFfnCall {
        x,
        hidden: HIDDEN,
        intermediate: INTER,
        experts: EXPERTS,
        top_k: TOP_K,
        router_kind: MoeRouterKind::default(),
        routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
        branch_scale: 1.0,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        router,
        router_bias: None,
        weights: ExpertSlices::Fused {
            gate_up,
            down,
            layout: GateUpLayout::Interleaved,
        },
        gate_up_bias: None,
        down_bias: None,
        router_input: None,
        router_scale: None,
        router_per_expert_scale: None,
        router_norm_eps: None,
    }
}

/// A provider that knows only the single-row operation, so its multi-row
/// answer is the trait's default.
struct SingleRow<'a>(&'a Bank);

impl SingleRow<'_> {
    fn expert(&self, expert: usize, x: &[f32]) -> Result<Vec<f32>, VindexError> {
        ProductionBackend::new().expert_transform(ExpertTransformCall {
            x,
            hidden: HIDDEN,
            intermediate: INTER,
            activation: Activation::Silu,
            gate_policy: ExpertGatePolicy::Gated,
            weights: ExpertWeights::Fused {
                gate_up: WeightSlice::F32(&self.0.gate_up[expert]),
                down: WeightSlice::F32(&self.0.down[expert]),
                layout: GateUpLayout::Interleaved,
                gate_up_bias: None,
                down_bias: None,
            },
        })
    }
}

impl RoutedExpertProvider for SingleRow<'_> {
    fn apply(
        &self,
        _layer: usize,
        input: &[f32],
        experts: &[usize],
    ) -> Result<Vec<ExpertOutput>, VindexError> {
        experts
            .iter()
            .map(|&expert| {
                Ok(ExpertOutput {
                    expert,
                    row: self.expert(expert, input)?,
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Fault {
    None,
    DropPair,
    DuplicatePair,
    SubstituteExpert,
    UnsolicitedRow,
    NonFinite,
    ShortRow,
    Refuse,
}

/// A provider with its own multi-row entry point. It answers in reverse
/// order, records what it was asked, and can be told to answer wrongly.
/// Its single-row entry point must never be reached by a multi-row call.
struct Batched<'a> {
    inner: SingleRow<'a>,
    fault: Fault,
    calls: AtomicUsize,
    asked: Mutex<Vec<(u32, Vec<usize>)>>,
}

impl<'a> Batched<'a> {
    fn new(bank: &'a Bank, fault: Fault) -> Self {
        Self {
            inner: SingleRow(bank),
            fault,
            calls: AtomicUsize::new(0),
            asked: Mutex::new(Vec::new()),
        }
    }
}

impl RoutedExpertProvider for Batched<'_> {
    fn apply(&self, _: usize, _: &[f32], _: &[usize]) -> Result<Vec<ExpertOutput>, VindexError> {
        unreachable!("a multi-row call must use the provider's multi-row entry point")
    }
    fn apply_rows(
        &self,
        _layer: usize,
        rows: &[ExpertRow<'_>],
    ) -> Result<Vec<RowExpertOutput>, VindexError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fault == Fault::Refuse {
            return Err(VindexError::Parse("injected worker failure".into()));
        }
        let mut out = Vec::new();
        for row in rows {
            self.asked
                .lock()
                .unwrap()
                .push((row.id, row.experts.to_vec()));
            for &expert in row.experts {
                out.push(RowExpertOutput {
                    row: row.id,
                    expert,
                    output: self.inner.expert(expert, row.input)?,
                });
            }
        }
        out.reverse();
        match self.fault {
            Fault::DropPair => {
                out.pop();
            }
            Fault::DuplicatePair => {
                let copy = RowExpertOutput {
                    row: out[0].row,
                    expert: out[0].expert,
                    output: out[0].output.clone(),
                };
                out[1] = copy;
            }
            Fault::SubstituteExpert => out[0].expert = EXPERTS,
            Fault::UnsolicitedRow => out[0].row = u32::MAX,
            Fault::NonFinite => out[0].output[0] = f32::NAN,
            Fault::ShortRow => {
                out[0].output.pop();
            }
            Fault::None | Fault::Refuse => {}
        }
        Ok(out)
    }
}

fn bits(rows: Result<Vec<Vec<f32>>, VindexError>) -> Result<Vec<Vec<u32>>, String> {
    rows.map(|rows| {
        rows.iter()
            .map(|row| row.iter().map(|v| v.to_bits()).collect())
            .collect()
    })
    .map_err(|e| e.to_string())
}

#[test]
fn one_row_or_many_reproduce_independent_single_row_execution_bit_for_bit() {
    let bank = Bank::new();
    let (gate_up, down) = (slices(&bank.gate_up), slices(&bank.down));
    let backend = ProductionBackend::new();
    let single = SingleRow(&bank);
    let make = |x| call(x, &bank.router, &gate_up, &down);

    // The reference each row is held to: it travelled alone, through the
    // existing single-row placed path, and equals fully local execution.
    let alone: Vec<Vec<u32>> = INPUTS
        .iter()
        .map(|x| {
            let placed = backend.routed_ffn_placed(make(x), LAYER, &single).unwrap();
            let local = backend.routed_ffn(make(x)).unwrap();
            let placed: Vec<u32> = placed.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                placed,
                local.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            placed
        })
        .collect();

    for count in [1, INPUTS.len()] {
        let calls: Vec<RoutedFfnCall<'_>> = INPUTS[..count].iter().map(|x| make(x)).collect();

        // A provider with a real multi-row entry point, answering reversed.
        let batched = Batched::new(&bank, Fault::None);
        let rows = bits(backend.routed_ffn_placed_rows(&calls, LAYER, &batched));
        assert_eq!(rows, Ok(alone[..count].to_vec()), "{count} rows, batched");
        assert_eq!(
            batched.calls.load(Ordering::SeqCst),
            1,
            "one provider call per layer"
        );

        // A provider that only has the single-row operation.
        let defaulted = bits(backend.routed_ffn_placed_rows(&calls, LAYER, &single));
        assert_eq!(
            defaulted,
            Ok(alone[..count].to_vec()),
            "{count} rows, default"
        );

        if count == INPUTS.len() {
            let asked = batched.asked.lock().unwrap();
            let pairs: usize = asked.iter().map(|(_, experts)| experts.len()).sum();
            let distinct: std::collections::BTreeSet<usize> = asked
                .iter()
                .flat_map(|(_, experts)| experts.iter().copied())
                .collect();
            let selections: std::collections::BTreeSet<&Vec<usize>> =
                asked.iter().map(|(_, experts)| experts).collect();
            assert_eq!(pairs, count * TOP_K);
            assert!(
                distinct.len() < pairs,
                "the batch shares experts across rows"
            );
            assert!(selections.len() > 1, "the batch routes rows differently");
        }
    }
}

#[test]
fn any_wrong_answer_fails_the_whole_call() {
    let bank = Bank::new();
    let (gate_up, down) = (slices(&bank.gate_up), slices(&bank.down));
    let backend = ProductionBackend::new();
    let calls: Vec<RoutedFfnCall<'_>> = INPUTS
        .iter()
        .map(|x| call(x, &bank.router, &gate_up, &down))
        .collect();
    for (name, fault) in [
        ("dropped pair", Fault::DropPair),
        ("duplicated pair", Fault::DuplicatePair),
        ("substituted expert", Fault::SubstituteExpert),
        ("unsolicited row", Fault::UnsolicitedRow),
        ("nonfinite output", Fault::NonFinite),
        ("short output", Fault::ShortRow),
        ("provider failure", Fault::Refuse),
    ] {
        let provider = Batched::new(&bank, fault);
        // `Result<Vec<_>>`: an error carries no rows, so nothing computed
        // for the healthy rows can be committed by a caller.
        let outcome = backend.routed_ffn_placed_rows(&calls, LAYER, &provider);
        assert!(outcome.is_err(), "{name} was accepted");
    }
}

#[test]
fn multi_row_placement_is_forwarded_refused_by_name_and_bounded() {
    let bank = Bank::new();
    let (gate_up, down) = (slices(&bank.gate_up), slices(&bank.down));
    let single = SingleRow(&bank);
    let calls: Vec<RoutedFfnCall<'_>> = INPUTS
        .iter()
        .map(|x| call(x, &bank.router, &gate_up, &down))
        .collect();

    let production = ProductionBackend::new();
    let shared = Arc::new(ProductionBackend::new());
    let direct = bits(production.routed_ffn_placed_rows(&calls, LAYER, &single));
    assert!(direct.is_ok(), "{direct:?}");
    assert_eq!(
        bits(shared.routed_ffn_placed_rows(&calls, LAYER, &single)),
        direct
    );

    let reference = Arc::new(ReferenceBackend);
    for backend in [&ReferenceBackend as &dyn PlanBackend, &reference] {
        let err = bits(backend.routed_ffn_placed_rows(&calls, LAYER, &single)).unwrap_err();
        assert!(
            err.contains("does not support multi-row routed expert placement"),
            "{err}"
        );
        assert!(err.contains(backend.name()), "{err}");
    }

    let err = bits(production.routed_ffn_placed_rows(&[], LAYER, &single)).unwrap_err();
    assert!(err.contains("carries no row"), "{err}");
    let mut mixed: Vec<RoutedFfnCall<'_>> = INPUTS[..2]
        .iter()
        .map(|x| call(x, &bank.router, &gate_up, &down))
        .collect();
    mixed[1].hidden = HIDDEN + 1;
    let err = bits(production.routed_ffn_placed_rows(&mixed, LAYER, &single)).unwrap_err();
    assert!(err.contains("mixes hidden widths"), "{err}");
}
