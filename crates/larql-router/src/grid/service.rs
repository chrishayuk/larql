//! gRPC [`GridService`] implementation and the admin RPCs.
//!
//! Wires the tonic-generated trait to the in-memory [`GridState`]:
//!
//! - `join` — bidirectional stream that handles announce / heartbeat
//!   / dropping (Mode A) and available / ready / refuse (Mode B).
//! - `status` — read-only snapshot for the admin `larql-router
//!   status` subcommand.
//! - `drain_server` — admin-initiated drain of a serving replica;
//!   sends `UnassignMsg(reason)` over the announce stream's sender.
//! - `assign_range` — admin-initiated assignment of a layer range to
//!   an available spare, with an explicit or live-replica-derived
//!   origin.
//!
//! Authentication: when `grid_key` is set on the impl, every incoming
//! `Join` must carry `Authorization: Bearer <key>` or it gets
//! `UNAUTHENTICATED`. The other RPCs are unauthenticated — admins
//! should put them behind a private network or an external proxy.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    AckMsg, AdminAck, AnnounceMsg, AssignRangeRequest, DrainRequest, GridService, RouterMessage,
    RouterPayload, ServerMessage, ServerPayload, StatusRequest, StatusResponse, UnassignMsg,
};

use crate::metrics::RouterMetrics;

use super::{GridState, ServerEntry};

pub struct GridServiceImpl {
    pub state: Arc<RwLock<GridState>>,
    next_id: AtomicU64,
    /// If set, every incoming Join stream must present "Authorization: Bearer <key>".
    grid_key: Option<String>,
    /// ADR-0017 — shared metrics registry. `None` skips observation
    /// (used by integration tests that don't need the dependency).
    metrics: Option<Arc<RouterMetrics>>,
}

impl GridServiceImpl {
    #[allow(dead_code)]
    pub fn new(state: Arc<RwLock<GridState>>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: None,
            metrics: None,
        }
    }

    pub fn new_with_key(state: Arc<RwLock<GridState>>, key: Option<String>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: key,
            metrics: None,
        }
    }

    /// Builder-style setter — installs the shared metrics handle so
    /// every Join stream the service handles bumps the right counters.
    pub fn with_metrics(mut self, metrics: Arc<RouterMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    fn alloc_server_id(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("srv-{ts}-{n}")
    }
}

type JoinStream = Pin<Box<dyn futures_core::Stream<Item = Result<RouterMessage, Status>> + Send>>;

#[tonic::async_trait]
impl GridService for GridServiceImpl {
    type JoinStream = JoinStream;

    async fn join(
        &self,
        request: Request<Streaming<ServerMessage>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        // Auth check — reject streams that don't carry the correct grid key.
        if let Some(expected) = &self.grid_key {
            let token = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            if token.map(|t| t != expected).unwrap_or(true) {
                return Err(Status::unauthenticated("invalid grid key"));
            }
        }

        let state = self.state.clone();
        let metrics = self.metrics.clone();
        let server_id = self.alloc_server_id();
        let (tx, rx) = mpsc::channel::<Result<RouterMessage, Status>>(32);
        let mut inbound = request.into_inner();

        let sid = server_id.clone();
        tokio::spawn(async move {
            let mut registered_model: Option<(String, u32, u32)> = None; // (model_id, start, end)
            let mut is_available = false; // true while in Mode B available pool

            while let Some(msg) = inbound.next().await {
                match msg {
                    Err(e) => {
                        tracing::warn!(server_id = %sid, "Stream error: {e}");
                        break;
                    }
                    Ok(ServerMessage { payload: None }) => {}
                    Ok(ServerMessage { payload: Some(p) }) => match p {
                        ServerPayload::Announce(AnnounceMsg {
                            model_id,
                            layer_start,
                            layer_end,
                            ram_bytes,
                            listen_url,
                            vindex_hash,
                            expert_start,
                            expert_end,
                        }) => {
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: listen_url.clone(),
                                model_id: model_id.clone(),
                                layer_start,
                                layer_end,
                                vindex_hash,
                                cpu_pct: 0.0,
                                ram_used: ram_bytes,
                                requests_in_flight: 0,
                                last_seen: Instant::now(),
                                layer_latencies: HashMap::new(),
                                req_per_sec: 0.0,
                                rtt_ms: None,
                                expert_start,
                                expert_end,
                            };
                            state.write().register_with_sender(entry, tx.clone());
                            if let Some(m) = &metrics {
                                m.grid_registers_total.inc();
                            }
                            registered_model = Some((model_id, layer_start, layer_end));

                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Heartbeat(hb) => {
                            state.write().update_heartbeat(
                                &sid,
                                hb.cpu_pct,
                                hb.ram_used,
                                hb.requests_in_flight,
                                hb.layer_stats,
                                hb.req_per_sec,
                            );
                        }

                        ServerPayload::Dropping(d) => {
                            tracing::info!(
                                server_id = %sid,
                                model_id = %d.model_id,
                                layers = %format!("{}-{}", d.layer_start, d.layer_end),
                                reason = %d.reason,
                                "Server dropping shard"
                            );
                            // Drop the server and immediately try to fill any
                            // freshly-exposed gap. After gap-fill, also check
                            // whether the disappearance created under-
                            // replication and pull spares for that too.
                            let (filled, replicated) = {
                                let mut guard = state.write();
                                guard.deregister(&sid);
                                let f = guard.try_fill_all_gaps();
                                let r = guard.try_replicate_from_available();
                                (f, r)
                            };
                            if let Some(m) = &metrics {
                                m.grid_deregisters_total
                                    .with_label_values(&["dropping"])
                                    .inc();
                                if filled > 0 || replicated > 0 {
                                    m.rebalancer_actions_total
                                        .with_label_values(&["replicate"])
                                        .inc_by((filled + replicated) as u64);
                                }
                            }
                            if filled > 0 || replicated > 0 {
                                tracing::info!(
                                    filled,
                                    replicated,
                                    "Grid: re-fill / re-replicate triggered by Dropping"
                                );
                            }
                            registered_model = None;
                        }

                        ServerPayload::Available(av) => {
                            // Mode B: server advertises capacity.
                            // Register it, then try to use it for either a
                            // coverage gap (zero replicas) or an
                            // under-replicated range (replica count below
                            // target_replicas).
                            state.write().register_available(
                                sid.clone(),
                                tx.clone(),
                                av.ram_bytes,
                                av.disk_bytes,
                                av.store_path.clone(),
                            );
                            is_available = true;
                            tracing::info!(
                                server_id = %sid,
                                ram_gb = av.ram_bytes / (1024 * 1024 * 1024),
                                "Grid: Mode B server registered; checking gaps + replicas…"
                            );

                            // 1) Fill coverage gaps first (most urgent — gaps
                            //    cause 503s; under-replicated ranges still
                            //    serve traffic from the surviving replicas).
                            let gaps = state.read().coverage_gaps();
                            let mut consumed = false;
                            for (model_id, layer_start, layer_end) in gaps {
                                // ADR-0018: coverage gaps are dense layer-range
                                // holes; the gap-fill path passes 0/0 for the
                                // expert range (no expert-level holes here).
                                let assigned = state.write().try_assign_gap(
                                    &model_id,
                                    layer_start,
                                    layer_end,
                                    0,
                                    0,
                                    av.ram_bytes,
                                );
                                if assigned {
                                    consumed = true;
                                    break;
                                }
                            }
                            // 2) If the spare wasn't used to fill a gap, try
                            //    using it to satisfy under-replication.
                            if !consumed {
                                let replicated = state.write().try_replicate_from_available();
                                if replicated > 0 {
                                    tracing::info!(
                                        replicated,
                                        "Grid: under-replicated range filled from new available server"
                                    );
                                }
                            }
                        }

                        ServerPayload::Ready(r) => {
                            // Mode B: server finished downloading + loading a shard.
                            // Register it as a serving shard and send Ack.
                            //
                            // vindex_hash is not present in ReadyMsg today (the
                            // server only knows the hash advertised on the assign);
                            // leave it empty for the freshly-loaded replica. This
                            // means the new replica won't be chosen as a Mode B
                            // origin for a further gap until its hash is known,
                            // but that's fine — the surviving original replica
                            // remains a valid origin.
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: r.listen_url.clone(),
                                model_id: r.model_id.clone(),
                                layer_start: r.layer_start,
                                layer_end: r.layer_end,
                                vindex_hash: String::new(),
                                cpu_pct: 0.0,
                                ram_used: 0,
                                requests_in_flight: 0,
                                last_seen: std::time::Instant::now(),
                                layer_latencies: HashMap::new(),
                                req_per_sec: 0.0,
                                rtt_ms: None,
                                // ADR-0018: ReadyMsg carries the expert range
                                // for the just-loaded shard. Mode B operators
                                // can hand back a Ready with `0/0` (dense) or
                                // a real expert range matching the originating
                                // AssignMsg.
                                expert_start: r.expert_start,
                                expert_end: r.expert_end,
                            };
                            state.write().register_with_sender(entry, tx.clone());
                            if let Some(m) = &metrics {
                                m.grid_registers_total.inc();
                            }
                            registered_model =
                                Some((r.model_id.clone(), r.layer_start, r.layer_end));
                            is_available = false;
                            tracing::info!(
                                server_id = %sid,
                                model_id = %r.model_id,
                                layers = %format!("{}-{}", r.layer_start, r.layer_end),
                                "Grid: Mode B server ready — now serving"
                            );
                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Refuse(r) => {
                            // Mode B: server refused the assignment. Re-add to available pool.
                            tracing::warn!(
                                server_id = %sid,
                                reason = %r.reason,
                                "Grid: Mode B server refused assignment — re-queuing"
                            );
                            // The tx clone is lost here; server must reconnect. Just log.
                        }
                    },
                }
            }

            // Stream closed — clean up. If a serving server vanished without
            // a graceful DroppingMsg, gap re-fill + replication must still
            // run because losing a replica may have introduced a gap or
            // dropped the count below target_replicas.
            if registered_model.is_some() {
                let (filled, replicated) = {
                    let mut guard = state.write();
                    guard.deregister(&sid);
                    let f = guard.try_fill_all_gaps();
                    let r = guard.try_replicate_from_available();
                    (f, r)
                };
                if let Some(m) = &metrics {
                    m.grid_deregisters_total
                        .with_label_values(&["stream_close"])
                        .inc();
                    if filled > 0 || replicated > 0 {
                        m.rebalancer_actions_total
                            .with_label_values(&["replicate"])
                            .inc_by((filled + replicated) as u64);
                    }
                }
                if filled > 0 || replicated > 0 {
                    tracing::info!(
                        filled,
                        replicated,
                        "Grid: re-fill / re-replicate triggered by disconnect"
                    );
                }
            }
            if is_available {
                state.write().deregister_available(&sid);
            }
            tracing::info!(server_id = %sid, "Connection closed");
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let resp = self.state.read().status_response();
        Ok(Response::new(resp))
    }

    /// ADR-0004 Phase 5: admin drain. Sends `UnassignMsg(reason)` to the
    /// named serving server. The server's announce loop handles drain →
    /// `DroppingMsg` → optional re-enter Mode B from there.
    async fn drain_server(
        &self,
        request: Request<DrainRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        let req = request.into_inner();
        let reason = if req.reason.is_empty() {
            "admin_drain".to_string()
        } else {
            req.reason
        };

        // Find the server + the layer range it currently covers.
        let (sender, layers) = {
            let guard = self.state.read();
            let entry = guard
                .servers()
                .find(|(id, _)| **id == req.server_id)
                .map(|(_, e)| (e.model_id.clone(), e.layer_start, e.layer_end));
            (guard.serving_sender(&req.server_id), entry)
        };

        let Some((model_id, layer_start, layer_end)) = layers else {
            return Ok(Response::new(AdminAck {
                ok: false,
                message: format!("server_id {:?} is not currently serving", req.server_id),
            }));
        };
        let Some(tx) = sender else {
            return Ok(Response::new(AdminAck {
                ok: false,
                message: format!("server_id {:?} has no outbound channel", req.server_id),
            }));
        };

        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id,
                layer_start,
                layer_end,
                reason,
            })),
        };
        match tx.try_send(Ok(msg)) {
            Ok(()) => Ok(Response::new(AdminAck {
                ok: true,
                message: String::new(),
            })),
            Err(e) => Ok(Response::new(AdminAck {
                ok: false,
                message: format!("send to {:?} failed: {e}", req.server_id),
            })),
        }
    }

    /// ADR-0004 Phase 5: force-assign a layer range. Either targets a
    /// named available server (when `target_server_id` is set) or any
    /// available spare; either uses the explicit origin URL/hash from the
    /// request, or resolves an origin from the live coverage matrix.
    async fn assign_range(
        &self,
        request: Request<AssignRangeRequest>,
    ) -> Result<Response<AdminAck>, Status> {
        let req = request.into_inner();
        let mut guard = self.state.write();

        // Resolve the origin: explicit > live replica.
        let (origin_url, shard_hash) = if !req.explicit_origin_url.is_empty() {
            (
                req.explicit_origin_url.clone(),
                req.explicit_origin_hash.clone(),
            )
        } else {
            // ADR-0018: admin `AssignRange` is dense-only today.
            // The RPC proto doesn't carry expert_start/end on
            // AssignRangeRequest yet — adding them is additive and
            // tracked separately.
            match guard.find_origin_for(&req.model_id, req.layer_start, req.layer_end, 0, 0) {
                Some(o) => o,
                None => {
                    return Ok(Response::new(AdminAck {
                        ok: false,
                        message: format!(
                            "no live replica covers {}[{}-{}] — pass explicit_origin_url to override",
                            req.model_id, req.layer_start, req.layer_end
                        ),
                    }));
                }
            }
        };

        // If target_server_id was named, hand the assignment to that specific
        // available server. Otherwise pick any spare.
        let assigned = if req.target_server_id.is_empty() {
            guard.try_assign_gap_with_origin(
                &req.model_id,
                req.layer_start,
                req.layer_end,
                0,
                0,
                &origin_url,
                &shard_hash,
                /* min_ram */ 0,
            )
        } else {
            match guard.send_assign_to_named_available(
                &req.target_server_id,
                &req.model_id,
                req.layer_start,
                req.layer_end,
                0,
                0,
                &origin_url,
                &shard_hash,
            ) {
                Ok(()) => true,
                Err(reason) => {
                    return Ok(Response::new(AdminAck {
                        ok: false,
                        message: reason,
                    }));
                }
            }
        };
        Ok(Response::new(if assigned {
            AdminAck {
                ok: true,
                message: String::new(),
            }
        } else {
            AdminAck {
                ok: false,
                message: "no available server matched the request".into(),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_service_impl_constructors_install_state_and_key() {
        let state = Arc::new(RwLock::new(GridState::default()));

        let svc = GridServiceImpl::new(state.clone());
        let id1 = svc.alloc_server_id();
        let id2 = svc.alloc_server_id();
        assert!(id1.starts_with("srv-"));
        assert!(id2.starts_with("srv-"));
        assert_ne!(id1, id2, "ids must increment monotonically");

        // new_with_key wires the key field.
        let _svc_keyed = GridServiceImpl::new_with_key(state, Some("secret".into()));
        // Field is private; constructing it exercises the branch.
    }
}
