//! The coordinator-side view of remote expert exchanges, shared by the
//! BLOCK-TIME and VERIFY-REMOTE trials: a transport that times every
//! exchange and notes its frame sizes, and an observer that collects the
//! worker's own timing.

use larql_inference::vindex3::routed_experts::{ExpertOutput, ExpertTransport};
use larql_router::vindex3_experts::HttpExpertShards;
use larql_router_protocol::provider_calls::ProviderCallObserver;
use larql_router_protocol::{
    vindex3_expert_rows as rows_wire, vindex3_experts as wire, vindex3_experts::Binding,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const SINGLE_ROW: &str = "single-row";
pub const MULTI_ROW: &str = "multi-row";
/// `wire` values of the transport's per-request profile records.
const PROFILE_WIRES: [&str; 2] = ["experts-binary-f32-v1", "expert-rows-f32-v1"];

/// One exchange as the coordinator saw it.
pub struct Exchange {
    pub route: &'static str,
    pub layer: usize,
    pub rows: usize,
    pub pairs: usize,
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub roundtrip_ns: u64,
}

impl Exchange {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "route": self.route, "layer": self.layer, "rows": self.rows, "pairs": self.pairs,
            "request_bytes": self.request_bytes, "response_bytes": self.response_bytes,
            "roundtrip_ns": self.roundtrip_ns,
        })
    }
}

/// The HTTP transport, timing each exchange and noting its frame sizes.
pub struct Timed {
    pub inner: HttpExpertShards,
    pub exchanges: Arc<Mutex<Vec<Exchange>>>,
}

impl ExpertTransport for Timed {
    fn bindings(&self) -> Vec<Binding> {
        self.inner.bindings()
    }

    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        let started = Instant::now();
        let outputs = self.inner.forward(shard, layer, experts, row)?;
        let roundtrip_ns = started.elapsed().as_nanos() as u64;
        self.exchanges.lock().unwrap().push(Exchange {
            route: SINGLE_ROW,
            layer,
            rows: 1,
            pairs: experts.len(),
            request_bytes: wire::HEADER_BYTES + 4 * experts.len() + 4 * row.len(),
            response_bytes: wire::response_len(row.len(), experts.len())?,
            roundtrip_ns,
        });
        Ok(outputs)
    }

    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[rows_wire::Row],
    ) -> Result<Vec<rows_wire::Output>, String> {
        let started = Instant::now();
        let outputs = self.inner.forward_rows(shard, layer, rows)?;
        let roundtrip_ns = started.elapsed().as_nanos() as u64;
        let pairs: usize = rows.iter().map(|r| r.experts.len()).sum();
        let hidden = rows.first().map_or(0, |r| r.input.len());
        self.exchanges.lock().unwrap().push(Exchange {
            route: MULTI_ROW,
            layer,
            rows: rows.len(),
            pairs,
            request_bytes: rows_wire::request_len(hidden, rows.len(), pairs)?,
            response_bytes: rows_wire::response_len(hidden, pairs)?,
            roundtrip_ns,
        });
        Ok(outputs)
    }
}

/// Collects the transport's own profile records, which carry the worker's
/// timing. Installed only for a profiled trial.
pub struct Recorder {
    pub on: AtomicBool,
    pub records: Mutex<Vec<serde_json::Value>>,
}

impl Recorder {
    /// A recorder for the whole process, installed as the transport's
    /// observer when `profile` is set. Call before the coordinator is
    /// prepared, so this observer is the one the transport reports to.
    pub fn install(profile: bool) -> Result<&'static Recorder, String> {
        let recorder: &'static Recorder = Box::leak(Box::new(Recorder {
            on: AtomicBool::new(profile),
            records: Mutex::new(Vec::new()),
        }));
        if profile && !larql_router_protocol::provider_calls::install_observer(recorder) {
            return Err("another profile observer is already installed".into());
        }
        Ok(recorder)
    }

    pub fn stop(&self) {
        self.on.store(false, Ordering::SeqCst);
    }
}

impl ProviderCallObserver for Recorder {
    fn enabled(&self) -> bool {
        self.on.load(Ordering::SeqCst)
    }

    fn record(&self, call: serde_json::Value) {
        if PROFILE_WIRES.iter().any(|w| call["wire"] == *w) {
            self.records.lock().unwrap().push(call);
        }
    }
}
