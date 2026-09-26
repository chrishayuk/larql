//! Subjects (fixtures and real containers) and the four-phase journey
//! (I3): batched prefill → resumed prefill → decode, each under one
//! `Measured` provider. The handoff phase is the server target's.

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::encode_fixture_container;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::{conv_qkv, gated_delta, kda, mamba2};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerAttention};

use super::measured::{Backing, Inspect, Measured};

pub struct Subject {
    _container: Option<tempfile::TempDir>,
    pub name: String,
    pub plan: ComponentOpPlan,
    pub store: OperandStore,
    pub geometry: Vec<LayerContinuationGeometry>,
}

pub fn fixture(model: fn(&Path), name: &str) -> Subject {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(model, checkpoint.path(), container.path(), name);
    let mut subject = open(container.path(), name);
    subject._container = Some(container);
    subject
}

pub fn open(dir: &Path, name: &str) -> Subject {
    let inspection = inspect_container(dir, false).unwrap();
    let outcome = plan_component_ops(&inspection, dir, "target").unwrap();
    assert!(outcome.closed(), "{name} must close: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(dir, &inspection).unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    Subject {
        _container: None,
        name: name.to_string(),
        plan,
        store,
        geometry,
    }
}

impl Subject {
    pub fn prepare<B: PlanBackend>(&self, backend: &B) -> PreparedOperands {
        PreparedOperands::load(&self.plan, &self.store, backend, ExecutionSlice::Full).unwrap()
    }

    /// Layers whose attention is conv-QKV (KV rows AND a conv history).
    pub fn conv_qkv_layers(&self) -> Vec<bool> {
        self.plan
            .layers
            .iter()
            .map(|l| matches!(l.attention, LayerAttention::ConvQkv(_)))
            .collect()
    }

    /// M8: per layer, the byte sizes of the buffers its operator declares
    /// as conv history (the ones the forecast says are copied per call).
    pub fn conv_history_bytes(&self) -> Vec<Option<Vec<usize>>> {
        self.plan
            .layers
            .iter()
            .zip(&self.geometry)
            .map(|(layer, geometry)| {
                let recurrent = geometry.recurrent()?;
                let indices: &[usize] = match layer.attention {
                    LayerAttention::ConvQkv(_) => &[conv_qkv::CONV_HISTORY],
                    LayerAttention::Mamba2(_) => &[mamba2::CONV_HISTORY],
                    LayerAttention::GatedDelta(_) => &[gated_delta::CONV_HISTORY],
                    LayerAttention::Kda(_) => &[kda::CONV_Q, kda::CONV_K, kda::CONV_V],
                    _ => return None,
                };
                Some(
                    indices
                        .iter()
                        .map(|&i| recurrent.buffers[i].bytes())
                        .collect(),
                )
            })
            .collect()
    }
}

/// Token streams for the three in-process phases.
#[derive(Clone, Debug)]
pub struct Journey {
    pub prefill: Vec<u32>,
    pub resume: Vec<u32>,
    pub decode: Vec<u32>,
}

/// What the journey produced: logits at each phase boundary and the
/// storage inventory after each phase.
pub struct Outcome {
    pub logits: Vec<(&'static str, Vec<f32>)>,
    pub inventories: Vec<(&'static str, Vec<Backing>)>,
    /// Append-born live bytes at each phase end (VIEW-1 V3 anti-cheat).
    pub append_born: Vec<(&'static str, usize)>,
}

/// FNV-1a over the bits of every logit of `phase`, in order: two builds
/// computed the same thing iff their digests match.
pub fn logits_digest(logits: &[(&'static str, Vec<f32>)], phase: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (_, row) in logits.iter().filter(|(p, _)| *p == phase) {
        for x in row {
            for byte in x.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    format!("{hash:016x}")
}

pub const BATCHED: &str = "batched_prefill";
pub const RESUMED: &str = "resumed_prefill";
pub const DECODE: &str = "decode";

pub fn run<P: Inspect, B: PlanBackend>(
    subject: &Subject,
    ops: &PreparedOperands,
    backend: &B,
    kv: &mut Measured<P>,
    journey: &Journey,
) -> Outcome {
    let mut logits = Vec::new();
    let mut inventories = Vec::new();
    let mut append_born = Vec::new();

    kv.set_phase(BATCHED);
    let out = prefill_prepared(&subject.plan, ops, &journey.prefill, backend, kv).unwrap();
    logits.push((BATCHED, out.logits.expect("prefill logits")));
    kv.set_phase("inventory");
    inventories.push((BATCHED, kv.inventory()));
    append_born.push((BATCHED, kv.append_born_live_bytes()));

    if !journey.resume.is_empty() {
        kv.set_phase(RESUMED);
        let out = prefill_prepared(&subject.plan, ops, &journey.resume, backend, kv).unwrap();
        logits.push((RESUMED, out.logits.expect("resumed logits")));
        kv.set_phase("inventory");
        inventories.push((RESUMED, kv.inventory()));
        append_born.push((RESUMED, kv.append_born_live_bytes()));
    }

    kv.set_phase(DECODE);
    {
        let mut session = DecodeSession::over_prepared(&subject.plan, ops, backend, kv).unwrap();
        for &token in &journey.decode {
            let step = session.step(token).unwrap();
            logits.push((DECODE, step.logits.expect("decode logits")));
        }
    }
    kv.set_phase("inventory");
    inventories.push((DECODE, kv.inventory()));
    append_born.push((DECODE, kv.append_born_live_bytes()));
    Outcome {
        logits,
        inventories,
        append_born,
    }
}
