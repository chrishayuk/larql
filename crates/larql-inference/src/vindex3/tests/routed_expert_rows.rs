//! The coordinator's multi-row expert dispatch, driven directly.
//!
//! Real expert workers run in-process behind a stub transport, so nothing
//! here is a second implementation of dispatch: every call goes through the
//! provider `admit_expert_provider` returns. The transport records what it
//! was sent, counts each path separately and can answer wrongly on demand.
use super::*;
use crate::vindex3::routed_experts::{
    self, BoundExpertWorker, ExpertOutput, ExpertRow, ExpertTransport,
};
use larql_router_protocol::{vindex3_expert_rows as rows_wire, vindex3_experts as wire};
use larql_vindex::format::vindex3::fixtures_routed::{
    miniature_routed, EXPERTS, HIDDEN, LAYERS, TOP_K,
};
use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

const LAYER: usize = 1;
/// Three rows over all four experts. Expert 1 is selected by every row, and
/// under the split topology the first and last rows span both workers.
const SELECTIONS: [[usize; TOP_K]; 3] = [[0, 3], [1, 2], [3, 1]];
const ROW_IDS: [u32; 3] = [40, 7, 19];
/// Constants of the deterministic input pattern: distinct per row and per
/// element, and small.
const PATTERN_MODULUS: usize = 11;
const PATTERN_SCALE: f32 = 16.0;
const PATTERN_OFFSET: f32 = 0.3;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fault {
    None,
    Fail,
    DropPair,
    DuplicatePair,
    SubstituteRow,
    ForeignExpert,
    NonFinite,
    ShortOutput,
}
const FAULTS: [Fault; 7] = [
    Fault::Fail,
    Fault::DropPair,
    Fault::DuplicatePair,
    Fault::SubstituteRow,
    Fault::ForeignExpert,
    Fault::NonFinite,
    Fault::ShortOutput,
];

/// A change to the bindings a transport advertises, which admission must refuse.
type Mutation = dyn Fn(&mut Vec<wire::Binding>);

/// One `(row id, experts)` list per multi-row frame, tagged with its worker.
type Frames = Arc<Mutex<Vec<(usize, Vec<(u32, Vec<usize>)>)>>>;

#[derive(Clone, Default)]
struct Probe {
    single_calls: Arc<AtomicUsize>,
    rows_calls: Arc<AtomicUsize>,
    frames: Frames,
    fault: Arc<Mutex<Option<Fault>>>,
}

struct InProcess {
    workers: Vec<BoundExpertWorker<ProductionBackend>>,
    /// What `bindings()` reports; normally the workers' own.
    advertised: Vec<wire::Binding>,
    probe: Probe,
}

impl ExpertTransport for InProcess {
    fn bindings(&self) -> Vec<wire::Binding> {
        self.advertised.clone()
    }
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        self.probe.single_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.workers[shard]
            .apply(layer, experts, row)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|o| ExpertOutput {
                expert: o.expert,
                row: o.row,
            })
            .collect())
    }
    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[rows_wire::Row],
    ) -> Result<Vec<rows_wire::Output>, String> {
        self.probe.rows_calls.fetch_add(1, Ordering::SeqCst);
        self.probe.frames.lock().unwrap().push((
            shard,
            rows.iter().map(|r| (r.id, r.experts.clone())).collect(),
        ));
        let borrowed: Vec<ExpertRow<'_>> = rows
            .iter()
            .map(|r| ExpertRow {
                id: r.id,
                input: &r.input,
                experts: &r.experts,
            })
            .collect();
        let mut outputs: Vec<rows_wire::Output> = self.workers[shard]
            .apply_rows(layer, &borrowed, None)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|o| rows_wire::Output {
                row: o.row,
                expert: o.expert,
                values: o.output,
            })
            .collect();
        // Never the order it was asked in.
        outputs.reverse();
        // Only the last worker misbehaves, so with two workers the first
        // one's answer is always correct and must still be discarded.
        if shard + 1 == self.workers.len() {
            match self.probe.fault.lock().unwrap().unwrap_or(Fault::None) {
                Fault::Fail => return Err("injected worker timeout".into()),
                Fault::DropPair => {
                    outputs.pop();
                }
                Fault::DuplicatePair => {
                    let last = outputs.len() - 1;
                    outputs[last] = outputs[0].clone();
                }
                Fault::SubstituteRow => outputs[0].row = u32::MAX,
                Fault::ForeignExpert => outputs[0].expert = EXPERTS,
                Fault::NonFinite => outputs[0].values[0] = f32::NAN,
                Fault::ShortOutput => {
                    outputs[0].values.pop();
                }
                Fault::None => {}
            }
        }
        Ok(outputs)
    }
}

fn worker(container: &Path, experts: (usize, usize)) -> BoundExpertWorker<ProductionBackend> {
    let prepared = Vindex3Runtime::open(container, COMPONENT, ProductionBackend::new())
        .unwrap()
        .prepare_slice(ExecutionSlice::RoutedExperts {
            start: 0,
            end: LAYERS,
            expert_start: experts.0,
            expert_end: experts.1,
        })
        .unwrap();
    BoundExpertWorker::new(container, Arc::new(prepared)).unwrap()
}

fn transport(container: &Path, topology: &[(usize, usize)], probe: &Probe) -> InProcess {
    let workers: Vec<_> = topology.iter().map(|t| worker(container, *t)).collect();
    InProcess {
        advertised: workers.iter().map(|w| w.binding().clone()).collect(),
        workers,
        probe: probe.clone(),
    }
}

fn topologies() -> [Vec<(usize, usize)>; 2] {
    [
        vec![(0, EXPERTS)],
        vec![(0, EXPERTS / 2), (EXPERTS / 2, EXPERTS)],
    ]
}

fn inputs() -> Vec<Vec<f32>> {
    (0..SELECTIONS.len())
        .map(|row| {
            (0..HIDDEN)
                .map(|i| {
                    ((row * HIDDEN + i) % PATTERN_MODULUS) as f32 / PATTERN_SCALE - PATTERN_OFFSET
                })
                .collect()
        })
        .collect()
}

fn batch(inputs: &[Vec<f32>]) -> Vec<ExpertRow<'_>> {
    (0..SELECTIONS.len())
        .map(|i| ExpertRow {
            id: ROW_IDS[i],
            input: &inputs[i],
            experts: &SELECTIONS[i],
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn multi_row_dispatch_is_single_row_dispatch_per_pair_in_one_frame_per_worker() {
    let container = container_with(miniature_routed);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for topology in topologies() {
        let probe = Probe::default();
        let provider = routed_experts::admit_expert_provider(
            container.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            transport(container.path(), &topology, &probe),
        )
        .unwrap();
        let inputs = inputs();
        let rows = batch(&inputs);

        // The reference: every row alone, through the single-row path.
        let mut alone = BTreeMap::new();
        for row in &rows {
            for output in provider.apply(LAYER, row.input, row.experts).unwrap() {
                alone.insert((row.id, output.expert), bits(&output.row));
            }
        }
        assert_eq!(alone.len(), SELECTIONS.len() * TOP_K);
        let single_calls = probe.single_calls.load(Ordering::SeqCst);
        assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);

        for count in [1, rows.len()] {
            probe.rows_calls.store(0, Ordering::SeqCst);
            probe.frames.lock().unwrap().clear();
            let outputs = provider.apply_rows(LAYER, &rows[..count]).unwrap();

            // Identity and bits, whatever order the workers answered in.
            let got: BTreeMap<(u32, usize), Vec<u32>> = outputs
                .iter()
                .map(|o| ((o.row, o.expert), bits(&o.output)))
                .collect();
            assert_eq!(got.len(), outputs.len(), "a pair was answered twice");
            let want: BTreeMap<(u32, usize), Vec<u32>> = alone
                .iter()
                .filter(|((row, _), _)| ROW_IDS[..count].contains(row))
                .map(|(key, value)| (*key, value.clone()))
                .collect();
            assert_eq!(got, want, "{topology:?}, {count} rows");

            // One frame per contacted worker, each holding only what that
            // worker owns, and together exactly what was selected.
            let frames = probe.frames.lock().unwrap();
            let contacted: std::collections::BTreeSet<usize> =
                frames.iter().map(|(shard, _)| *shard).collect();
            assert_eq!(
                frames.len(),
                contacted.len(),
                "a worker was sent two frames"
            );
            assert_eq!(probe.rows_calls.load(Ordering::SeqCst), frames.len());
            let mut sent = BTreeMap::<u32, Vec<usize>>::new();
            for (shard, frame) in frames.iter() {
                let (first, last) = topology[*shard];
                for (row, experts) in frame {
                    assert!(experts.iter().all(|e| (first..last).contains(e)));
                    sent.entry(*row).or_default().extend(experts);
                }
            }
            for (index, row) in rows[..count].iter().enumerate() {
                let mut asked = sent.remove(&row.id).unwrap();
                asked.sort_unstable();
                let mut selected = SELECTIONS[index].to_vec();
                selected.sort_unstable();
                assert_eq!(asked, selected);
            }
            assert!(sent.is_empty(), "a row nobody asked for was sent");
            if count == rows.len() {
                assert_eq!(
                    frames.len(),
                    topology.len(),
                    "every worker is contacted once"
                );
            }
        }
        assert_eq!(
            probe.single_calls.load(Ordering::SeqCst),
            single_calls,
            "a multi-row call used the single-row route"
        );
    }
}

#[test]
fn one_wrong_worker_answer_rejects_the_whole_multi_row_call() {
    let container = container_with(miniature_routed);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    for topology in topologies() {
        let probe = Probe::default();
        let provider = routed_experts::admit_expert_provider(
            container.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            transport(container.path(), &topology, &probe),
        )
        .unwrap();
        let inputs = inputs();
        let rows = batch(&inputs);
        for fault in FAULTS {
            *probe.fault.lock().unwrap() = Some(fault);
            probe.rows_calls.store(0, Ordering::SeqCst);
            // `Result`: an error carries no rows, including the rows a
            // healthy worker answered correctly in the same call.
            assert!(
                provider.apply_rows(LAYER, &rows).is_err(),
                "{topology:?}: {fault:?} was accepted"
            );
            assert_eq!(
                probe.rows_calls.load(Ordering::SeqCst),
                topology.len(),
                "every worker was asked before the call was judged"
            );
        }
        *probe.fault.lock().unwrap() = None;
        assert!(provider.apply_rows(LAYER, &rows).is_ok(), "healthy again");

        // Caller errors are refused before any worker is contacted.
        probe.rows_calls.store(0, Ordering::SeqCst);
        assert!(provider.apply_rows(LAYERS, &rows).is_err(), "unowned layer");
        let foreign = [EXPERTS];
        let unowned = [ExpertRow {
            id: 1,
            input: &inputs[0],
            experts: &foreign,
        }];
        assert!(
            provider.apply_rows(LAYER, &unowned).is_err(),
            "unowned expert"
        );

        // Identity is judged over the whole call. These two rows share an
        // id but select experts on different workers under the split
        // topology, so each worker's frame would be well formed on its own.
        let (low, high) = ([0usize], [EXPERTS - 1]);
        let colliding = [
            ExpertRow {
                id: 5,
                input: &inputs[0],
                experts: &low,
            },
            ExpertRow {
                id: 5,
                input: &inputs[1],
                experts: &high,
            },
        ];
        assert!(
            provider.apply_rows(LAYER, &colliding).is_err(),
            "colliding row ids"
        );
        let repeated = [1usize, 1];
        let twice = [ExpertRow {
            id: 1,
            input: &inputs[0],
            experts: &repeated,
        }];
        assert!(
            provider.apply_rows(LAYER, &twice).is_err(),
            "expert selected twice"
        );
        let nothing = [ExpertRow {
            id: 1,
            input: &inputs[0],
            experts: &[],
        }];
        assert!(
            provider.apply_rows(LAYER, &nothing).is_err(),
            "row selecting nothing"
        );
        assert!(provider.apply_rows(LAYER, &[]).is_err(), "no rows");
        assert_eq!(probe.rows_calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn admitting_a_provider_runs_every_check_coordinator_preparation_ran() {
    let container = container_with(miniature_routed);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ProductionBackend::new()).unwrap();
    let topology = vec![(0, EXPERTS / 2), (EXPERTS / 2, EXPERTS)];
    let refusal = |mutate: &Mutation| {
        let mut stub = transport(container.path(), &topology, &Probe::default());
        mutate(&mut stub.advertised);
        let before = runtime.operands().store().bytes_read();
        let error = match routed_experts::admit_expert_provider(
            container.path(),
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            stub,
        ) {
            Ok(_) => panic!("a mutated binding was admitted"),
            Err(e) => e.to_string(),
        };
        // Admission decides from bindings alone: no local payload is read.
        assert_eq!(runtime.operands().store().bytes_read(), before);
        error
    };
    let cases: [(&str, &Mutation); 9] = [
        ("execution identity", &|b| {
            b[0].program.execution_identity = "0".repeat(64)
        }),
        ("incomplete", &|b| b.clear()),
        ("incomplete", &|b| {
            b.pop();
        }),
        ("overlapping", &|b| b.push(b[0].clone())),
        ("artifact", &|b| b[0].program.artifact = "0".repeat(64)),
        ("provider", &|b| {
            b[0].program.lowering = "cpu-production/v999".into()
        }),
        ("region", &|b| b[0].regions.push(' ')),
        ("dimensions", &|b| b[0].program.hidden += 1),
        ("representation", &|b| {
            b[0].operands[0].representation = "different".into()
        }),
    ];
    for (want, mutate) in cases {
        let error = refusal(mutate);
        assert!(error.contains(want), "expected {want}: {error}");
    }

    // A backend that is not CPU production lowering is refused outright.
    let reference = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend).unwrap();
    let error = match routed_experts::admit_expert_provider(
        container.path(),
        reference.plan(),
        reference.operands(),
        reference.backend(),
        transport(container.path(), &topology, &Probe::default()),
    ) {
        Ok(_) => panic!("a non-production backend was admitted"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("CPU production lowering"), "{error}");

    // The seam and coordinator preparation agree on a good set of workers.
    assert!(routed_experts::prepare_coordinator(
        container.path(),
        runtime.plan(),
        runtime.operands(),
        runtime.backend(),
        transport(container.path(), &topology, &Probe::default()),
    )
    .is_ok());
}
