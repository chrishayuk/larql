//! gRPC `AttentionService` — thin shim over the same logic as the
//! HTTP `/v1/attention/*` and `/v1/kv-cache/*` routes.
//!
//! Wire layout: see `crates/larql-router-protocol/proto/attention.proto`
//! and `docs/attention-service-protocol.md`. The byte format for
//! snapshots is identical to the HTTP variant (no base64 wrapper,
//! since proto3 carries bytes natively).
//!
//! `attention-service-routes` change.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use tonic::{Request, Response, Status};

use larql_router_protocol::{
    AttCreateSessionRequest, AttCreateSessionResponse, AttDecodeRequest, AttDecodeResponse,
    AttDeleteSessionRequest, AttDeleteSessionResponse, AttFreeRequest, AttFreeResponse,
    AttGetSessionRequest, AttGetSessionResponse, AttPrefillRequest, AttRestoreRequest,
    AttRestoreResponse, AttSnapshotRequest, AttSnapshotResponse, AttentionService, PrefillEvent,
};

use crate::attention_session::{AttentionSession, SessionId, SessionMapError};
use crate::kv_snapshot;
use crate::routes::attention::{
    check_fp32_attention_loaded, deserialize_snapshot_bytes, kv_format_str, parse_kv_format,
};
use crate::state::AppState;

pub struct AttentionGrpcService {
    pub state: Arc<AppState>,
}

type PrefillStream = Pin<Box<dyn Stream<Item = Result<PrefillEvent, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl AttentionService for AttentionGrpcService {
    async fn create_session(
        &self,
        request: Request<AttCreateSessionRequest>,
    ) -> Result<Response<AttCreateSessionResponse>, Status> {
        let req = request.into_inner();
        let model = self
            .state
            .model(Some(&req.model_id))
            .ok_or_else(|| Status::not_found(format!("no_such_model: {}", req.model_id)))?;
        let num_layers = model.config.num_layers;

        let session = if !req.restore_from_snapshot.is_empty() {
            let cache = deserialize_snapshot_bytes(&req.restore_from_snapshot)
                .map_err(|(code, detail)| Status::invalid_argument(format!("{code}: {detail}")))?;
            let mut s = AttentionSession::new(SessionId::new(), &req.model_id, num_layers, None);
            s.seq_len = cache.next_position;
            s.prefilled = s.seq_len > 0;
            s.cache = cache;
            s
        } else {
            let fmt_str = if req.kv_format.is_empty() {
                "fp32"
            } else {
                req.kv_format.as_str()
            };
            let kv_format = parse_kv_format(fmt_str)
                .map_err(|d| Status::invalid_argument(format!("kv_format_unknown: {d}")))?;
            AttentionSession::new(SessionId::new(), &req.model_id, num_layers, kv_format)
        };

        let entry = self.state.attention_sessions.insert(session).map_err(
            |SessionMapError::AtCap { cap }| {
                Status::resource_exhausted(format!("session_map_full: at cap of {cap}"))
            },
        )?;
        let g = entry.read().await;
        Ok(Response::new(AttCreateSessionResponse {
            session_id: g.id.as_str().to_string(),
            layer_start: 0,
            layer_end: num_layers as u32,
            kv_format: kv_format_str(g.cache.kv_format).to_string(),
        }))
    }

    async fn get_session(
        &self,
        request: Request<AttGetSessionRequest>,
    ) -> Result<Response<AttGetSessionResponse>, Status> {
        let id = SessionId(request.into_inner().session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;
        let g = entry.read().await;
        Ok(Response::new(AttGetSessionResponse {
            session_id: g.id.as_str().to_string(),
            model_id: g.model_id.clone(),
            kv_format: kv_format_str(g.cache.kv_format).to_string(),
            seq_len: g.seq_len as u32,
            prefilled: g.prefilled,
            num_layers: g.cache.layers.len() as u32,
        }))
    }

    async fn delete_session(
        &self,
        request: Request<AttDeleteSessionRequest>,
    ) -> Result<Response<AttDeleteSessionResponse>, Status> {
        let id = SessionId(request.into_inner().session_id);
        if self.state.attention_sessions.remove(&id).is_none() {
            return Err(Status::not_found("no_such_session"));
        }
        Ok(Response::new(AttDeleteSessionResponse {}))
    }

    type PrefillStream = PrefillStream;

    async fn prefill(
        &self,
        request: Request<AttPrefillRequest>,
    ) -> Result<Response<Self::PrefillStream>, Status> {
        use larql_router_protocol::{Vec1d as PVec1d, Vec2d as PVec2d};

        let req = request.into_inner();
        let id = SessionId(req.session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;

        let token_embeddings = req
            .token_embeddings
            .ok_or_else(|| Status::invalid_argument("missing_token_embeddings"))?;
        let seq_len = token_embeddings.rows.len();
        if seq_len == 0 {
            return Err(Status::invalid_argument("empty_token_embeddings"));
        }
        let hidden_dim = token_embeddings.rows[0].values.len();
        if !token_embeddings
            .rows
            .iter()
            .all(|r| r.values.len() == hidden_dim)
        {
            return Err(Status::invalid_argument("ragged_token_embeddings"));
        }

        // Look up the model.
        let model = {
            let g = entry.read().await;
            let model_id = g.model_id.clone();
            self.state
                .model(Some(&model_id))
                .ok_or_else(|| Status::not_found(format!("no_such_model: {model_id}")))?
                .clone()
        };
        if model.infer_disabled {
            return Err(Status::unavailable(
                "inference_disabled: server started with --no-infer; weights are not loaded",
            ));
        }

        // Loud guard against Q4_K-quantised models (same as the HTTP
        // path; see routes::attention::check_fp32_attention_loaded).
        {
            let weights_guard = model
                .get_or_load_weights()
                .map_err(|e| Status::internal(format!("weights_load_failed: {e}")))?;
            if let Err(missing_layer) = check_fp32_attention_loaded(&weights_guard) {
                return Err(Status::unavailable(format!(
                    "quantized_model_unsupported: layer {missing_layer} has no FP32 attention tensor; \
                     use the OpenAI completions path instead until prefill_q4 wires in"
                )));
            }
        }

        // Reshape input.
        let mut flat = Vec::with_capacity(seq_len * hidden_dim);
        for row in &token_embeddings.rows {
            flat.extend_from_slice(&row.values);
        }
        let h0 = larql_inference::ndarray::Array2::from_shape_vec((seq_len, hidden_dim), flat)
            .map_err(|e| Status::internal(format!("embedding_shape_error: {e}")))?;

        let start = std::time::Instant::now();

        let model_for_blocking = Arc::clone(&model);
        type PrefillOk = (
            Vec<larql_inference::ndarray::Array2<f32>>,
            Vec<Option<larql_inference::attention::SharedKV>>,
        );
        let result: Result<PrefillOk, String> = tokio::task::spawn_blocking(move || {
            let weights_guard = model_for_blocking.get_or_load_weights()?;
            let weights: &larql_inference::ModelWeights = &weights_guard;
            let num_layers = weights.num_layers;
            let ffn = larql_inference::ffn::WeightFfn { weights };

            let mut h = h0;
            let mut residuals = Vec::with_capacity(num_layers);
            let mut kvs = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                match larql_inference::run_layer_with_ffn_capturing_h_post_attn(
                    weights, &h, layer, &ffn, false, None, None,
                ) {
                    Some((h_next, h_post_attn, _, kv_out)) => {
                        residuals.push(h_post_attn);
                        kvs.push(kv_out);
                        h = h_next;
                    }
                    None => {
                        residuals.push(larql_inference::ndarray::Array2::zeros((
                            seq_len, hidden_dim,
                        )));
                        kvs.push(None);
                    }
                }
            }
            Ok((residuals, kvs))
        })
        .await
        .unwrap_or_else(|e| Err(format!("blocking task panicked: {e}")));

        let (residuals, kvs) =
            result.map_err(|e| Status::internal(format!("weights_load_failed: {e}")))?;

        // Stash KVs into the session.
        {
            let mut g = entry.write().await;
            for (layer, kv) in kvs.into_iter().enumerate() {
                if let Some(kv) = kv {
                    g.cache.set_layer(layer, kv);
                }
            }
            g.cache.next_position = seq_len;
            g.seq_len = seq_len;
            g.prefilled = true;
            g.touch();
        }

        // Stream one PrefillEvent per layer + a terminal one with done=true.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<PrefillEvent, Status>>(32);
        let total = residuals.len();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tokens_processed = seq_len as u32;
        tokio::spawn(async move {
            for (layer, r) in residuals.into_iter().enumerate() {
                let post_attention = PVec2d {
                    cols: hidden_dim as u32,
                    rows: (0..r.nrows())
                        .map(|i| PVec1d {
                            values: r.row(i).to_vec(),
                        })
                        .collect(),
                };
                let evt = PrefillEvent {
                    layer: layer as u32,
                    post_attention: Some(post_attention),
                    done: layer + 1 == total,
                    tokens_processed,
                    latency_ms: if layer + 1 == total { latency_ms } else { 0.0 },
                };
                if tx.send(Ok(evt)).await.is_err() {
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            as tokio_stream::wrappers::ReceiverStream<_>;
        Ok(Response::new(Box::pin(stream) as Self::PrefillStream))
    }

    async fn decode(
        &self,
        request: Request<AttDecodeRequest>,
    ) -> Result<Response<AttDecodeResponse>, Status> {
        use larql_router_protocol::{Vec1d as PVec1d, Vec2d as PVec2d};

        let req = request.into_inner();
        let id = SessionId(req.session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;
        if req.query_token_embedding.is_empty() {
            return Err(Status::invalid_argument("empty_query_embedding"));
        }
        let hidden_dim = req.query_token_embedding.len();

        let (model, abs_position) = {
            let g = entry.read().await;
            if !g.prefilled {
                return Err(Status::failed_precondition(
                    "decode_before_prefill: call Prefill first",
                ));
            }
            let model_id = g.model_id.clone();
            let abs = g.cache.next_position;
            (
                self.state
                    .model(Some(&model_id))
                    .ok_or_else(|| Status::not_found(format!("no_such_model: {model_id}")))?
                    .clone(),
                abs,
            )
        };
        if model.infer_disabled {
            return Err(Status::unavailable(
                "inference_disabled: server started with --no-infer; weights are not loaded",
            ));
        }

        // Loud guard against Q4_K-quantised models (same as prefill).
        {
            let weights_guard = model
                .get_or_load_weights()
                .map_err(|e| Status::internal(format!("weights_load_failed: {e}")))?;
            if let Err(missing_layer) = check_fp32_attention_loaded(&weights_guard) {
                return Err(Status::unavailable(format!(
                    "quantized_model_unsupported: layer {missing_layer} has no FP32 attention tensor"
                )));
            }
        }

        let h_new_array = larql_inference::ndarray::Array2::from_shape_vec(
            (1, hidden_dim),
            req.query_token_embedding,
        )
        .map_err(|e| Status::internal(format!("embedding_shape_error: {e}")))?;

        let start = std::time::Instant::now();
        let mut g = entry.write().await;
        let blocking_inputs = std::mem::take(&mut g.cache.layers);
        drop(g);

        let model_for_blocking = Arc::clone(&model);
        type DecodeOk = (
            Vec<larql_inference::ndarray::Array2<f32>>,
            Vec<Option<larql_inference::attention::SharedKV>>,
        );
        let result: Result<DecodeOk, String> = tokio::task::spawn_blocking(move || {
            let weights_guard = model_for_blocking.get_or_load_weights()?;
            let weights: &larql_inference::ModelWeights = &weights_guard;
            let num_layers = weights.num_layers;
            let ffn = larql_inference::ffn::WeightFfn { weights };

            let mut layers = blocking_inputs;
            layers.resize_with(num_layers, || None);
            let mut h_step = h_new_array;
            let mut residuals = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let kv_entry = layers[layer].as_ref();
                match larql_inference::attention::run_attention_block_decode_step(
                    weights,
                    &h_step,
                    layer,
                    kv_entry,
                    abs_position,
                ) {
                    Some((h_post_attn, new_kv)) => {
                        layers[layer] = Some(new_kv);
                        let (h_out, _) = larql_inference::forward::run_ffn(
                            weights,
                            &h_post_attn,
                            layer,
                            &ffn,
                            false,
                        );
                        residuals.push(h_post_attn);
                        h_step = h_out;
                    }
                    None => {
                        residuals.push(larql_inference::ndarray::Array2::zeros((1, hidden_dim)));
                    }
                }
            }
            Ok((residuals, layers))
        })
        .await
        .unwrap_or_else(|e| Err(format!("blocking task panicked: {e}")));

        let (residuals, layers) =
            result.map_err(|e| Status::internal(format!("weights_load_failed: {e}")))?;

        {
            let mut g = entry.write().await;
            g.cache.layers = layers;
            g.cache.next_position += 1;
            g.seq_len += 1;
            g.touch();
        }

        let post_attention_residual = PVec2d {
            cols: hidden_dim as u32,
            rows: residuals
                .into_iter()
                .map(|r| PVec1d {
                    values: r.row(0).to_vec(),
                })
                .collect(),
        };

        Ok(Response::new(AttDecodeResponse {
            post_attention_residual: Some(post_attention_residual),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        }))
    }

    async fn snapshot(
        &self,
        request: Request<AttSnapshotRequest>,
    ) -> Result<Response<AttSnapshotResponse>, Status> {
        let id = SessionId(request.into_inner().session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;
        let g = entry.read().await;
        let bytes = kv_snapshot::serialize(&g.cache);
        let bytes_len = bytes.len() as u64;
        Ok(Response::new(AttSnapshotResponse {
            session_id: g.id.as_str().to_string(),
            snapshot: bytes,
            bytes_len,
        }))
    }

    async fn restore(
        &self,
        request: Request<AttRestoreRequest>,
    ) -> Result<Response<AttRestoreResponse>, Status> {
        let req = request.into_inner();
        let id = SessionId(req.session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;
        let cache = deserialize_snapshot_bytes(&req.snapshot)
            .map_err(|(code, detail)| Status::invalid_argument(format!("{code}: {detail}")))?;
        let mut g = entry.write().await;
        let num_layers = cache.layers.len();
        let seq_len = cache.next_position;
        g.cache = cache;
        g.seq_len = seq_len;
        g.prefilled = seq_len > 0;
        g.touch();
        Ok(Response::new(AttRestoreResponse {
            session_id: g.id.as_str().to_string(),
            seq_len: seq_len as u32,
            num_layers: num_layers as u32,
        }))
    }

    async fn free(
        &self,
        request: Request<AttFreeRequest>,
    ) -> Result<Response<AttFreeResponse>, Status> {
        let req = request.into_inner();
        let id = SessionId(req.session_id);
        let entry = self
            .state
            .attention_sessions
            .get(&id)
            .ok_or_else(|| Status::not_found("no_such_session"))?;
        let mut g = entry.write().await;
        let total = g.cache.layers.len();
        let layers_freed = if req.all {
            for layer in 0..total {
                g.cache.clear_layer(layer);
            }
            total as u32
        } else {
            let layer = req.layer as usize;
            if layer >= total {
                return Err(Status::invalid_argument(format!(
                    "layer_out_of_range: {layer} >= {total}"
                )));
            }
            g.cache.clear_layer(layer);
            1
        };
        g.touch();
        Ok(Response::new(AttFreeResponse {
            session_id: g.id.as_str().to_string(),
            layers_freed,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_session::AttentionSessionMap;
    use crate::cache::DescribeCache;
    use crate::session::SessionManager;
    use crate::state::AppState;
    use std::sync::atomic::AtomicU64;

    fn empty_state() -> Arc<AppState> {
        Arc::new(AppState {
            models: vec![],
            started_at: std::time::Instant::now(),
            requests_served: AtomicU64::new(0),
            api_key: None,
            sessions: SessionManager::new(60),
            describe_cache: DescribeCache::new(0),
            attention_sessions: Arc::new(AttentionSessionMap::new(60, 16)),
        })
    }

    fn svc() -> AttentionGrpcService {
        AttentionGrpcService {
            state: empty_state(),
        }
    }

    #[tokio::test]
    async fn delete_unknown_session_returns_not_found() {
        let s = svc();
        let r = s
            .delete_session(Request::new(AttDeleteSessionRequest {
                session_id: "01HMMISSING0000000000000000".into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn get_unknown_session_returns_not_found() {
        let s = svc();
        let r = s
            .get_session(Request::new(AttGetSessionRequest {
                session_id: "01HMMISSING0000000000000000".into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn snapshot_unknown_session_returns_not_found() {
        let s = svc();
        let r = s
            .snapshot(Request::new(AttSnapshotRequest {
                session_id: "01HMMISSING0000000000000000".into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn free_unknown_session_returns_not_found() {
        let s = svc();
        let r = s
            .free(Request::new(AttFreeRequest {
                session_id: "01HMMISSING0000000000000000".into(),
                layer: 0,
                all: true,
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn restore_with_bad_magic_returns_invalid_argument() {
        let s = svc();
        let session = AttentionSession::new(SessionId::new(), "m", 1, None);
        let id = session.id.clone();
        let _ = s.state.attention_sessions.insert(session).unwrap();
        let r = s
            .restore(Request::new(AttRestoreRequest {
                session_id: id.as_str().to_string(),
                snapshot: vec![0u8; 64],
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::InvalidArgument);
        assert!(r.message().contains("snapshot_magic_mismatch"));
    }

    #[tokio::test]
    async fn create_session_unknown_model_returns_not_found() {
        let s = svc();
        let r = s
            .create_session(Request::new(AttCreateSessionRequest {
                model_id: "no-such".into(),
                kv_format: String::new(),
                restore_from_snapshot: vec![],
            }))
            .await
            .unwrap_err();
        assert_eq!(r.code(), tonic::Code::NotFound);
    }
}
