//! V3-HEAD-OBS-1 acceptance on the golden plan
//! (`docs/v3-head-obs-1-per-head-observation.md`): HP1 parity, HP2 record
//! identity, HP3 the head-sum law through the post-norm, HP4 the source
//! split, A6 per-layer coverage on a mixed-family plan, A8 refusal before
//! the first token, and the event order.

use std::collections::BTreeMap;

use super::decode::fixture;
use super::device::LoopDevice;
use super::golden::{G_HEAD_DIM, G_KV_HEADS, G_LAYERS, G_Q_HEADS, G_TOKENS, G_WINDOW};
use super::hybrid_traversal::hybrid;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightFormat};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::observe_heads::{HeadReader, HeadStats};
use crate::format::vindex3::opplan::exec::observe_stats::FixedBasis;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;

type Key = (usize, SublayerSite, usize);

/// An owned copy of one head record.
#[derive(Debug, Clone, PartialEq)]
struct Head {
    layer: usize,
    position: usize,
    head: usize,
    kv_head: usize,
    source_start: usize,
    weights: Vec<f32>,
    sink: f32,
    values: Vec<f32>,
    gate: Option<Vec<f32>>,
    source_values: Vec<Vec<f32>>,
    query: Vec<f32>,
    source_keys: Vec<Vec<f32>>,
}

/// Everything a run says: writes, events with positions, logits, and —
/// when asked — every head record.
#[derive(Default)]
struct Witness {
    heads: bool,
    /// Narrow an armed capture to one `(layer, position)`.
    only: Option<(usize, usize)>,
    position: usize,
    delta: BTreeMap<Key, Vec<f32>>,
    after: BTreeMap<Key, Vec<f32>>,
    layer_scale: BTreeMap<Key, Option<f32>>,
    events: Vec<(usize, StepEvent)>,
    records: Vec<Head>,
}

impl StepObserver for Witness {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
        }
        self.events.push((self.position, event));
    }

    fn wants_attention_heads(&self) -> bool {
        self.heads
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        match self.only {
            Some(address) => self.heads && address == (layer, position),
            None => self.heads,
        }
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.records.push(Head {
            layer,
            position: record.position,
            head: record.head,
            kv_head: record.kv_head,
            source_start: record.source_start,
            weights: record.weights.to_vec(),
            sink: record.sink,
            values: record.values.to_vec(),
            gate: record.gate.map(<[f32]>::to_vec),
            source_values: record.source_values.iter().map(|v| v.to_vec()).collect(),
            query: record.query.to_vec(),
            source_keys: record.source_keys.iter().map(|k| k.to_vec()).collect(),
        });
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let key = (record.layer, record.site, record.position);
        self.delta.insert(key, record.delta.to_vec());
        self.after.insert(key, record.after.to_vec());
        self.layer_scale.insert(key, record.layer_scale);
    }
}

fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    heads: bool,
) -> (Vec<Vec<f32>>, Witness) {
    run_narrowed(plan, store, backend, heads, None)
}

fn run_narrowed<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    heads: bool,
    only: Option<(usize, usize)>,
) -> (Vec<Vec<f32>>, Witness) {
    let mut session = DecodeSession::new(
        plan,
        store,
        backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut witness = Witness {
        heads,
        only,
        ..Witness::default()
    };
    let logits = G_TOKENS
        .iter()
        .map(|&t| {
            session
                .step_observed(t, &mut witness)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    (logits, witness)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn structural(events: &[(usize, StepEvent)]) -> Vec<(usize, StepEvent)> {
    events
        .iter()
        .filter(|(_, e)| {
            !matches!(
                e,
                StepEvent::HeadsObserved { .. } | StepEvent::HeadsUncovered { .. }
            )
        })
        .cloned()
        .collect()
}

macro_rules! on_both_backends {
    (|$backend:ident| $body:block) => {{
        {
            let $backend: &ReferenceBackend = &ReferenceBackend::new();
            $body
        }
        {
            let $backend: &ProductionBackend = &ProductionBackend::new();
            $body
        }
    }};
}

fn prepared<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> PreparedOperands {
    PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap()
}

/// A synthetic head record over borrowed buffers, for driving the reader
/// without the executor: what the kernel would hand it, spelled by hand.
fn synthetic_record<'a>(
    head: usize,
    position: usize,
    weights: &'a [f32],
    values: &'a [f32],
    source_values: &'a [&'a [f32]],
) -> AttentionHeadRecord<'a> {
    AttentionHeadRecord {
        position,
        head,
        kv_head: head / (G_Q_HEADS / G_KV_HEADS),
        source_start: 0,
        weights,
        sink: 0.0,
        values,
        gate: None,
        source_values,
        // The reader never reads keys or the query; aligned stand-ins
        // keep the record's own shape contract.
        query: values,
        source_keys: source_values,
        query_unpositioned: None,
        key_unpositioned: None,
    }
}

mod events_coverage_refusal;
mod hp1;
mod hp3_hp4;
mod the_reader_s_refusals_and_degenerate_bra;
