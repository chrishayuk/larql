//! gRPC service implementation for VindexService.

use std::sync::Arc;

use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::band_utils::{
    HEALTH_STATUS_OK, INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK, PROBE_RELATION_SOURCE,
};
use crate::state::AppState;

pub mod proto {
    tonic::include_proto!("vindex");
}

use proto::vindex_service_server::VindexService;
use proto::*;

pub struct VindexGrpcService {
    pub state: Arc<AppState>,
}

#[tonic::async_trait]
impl VindexService for VindexGrpcService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.state.bump_requests();
        let uptime = self.state.started_at.elapsed().as_secs();
        let served = self
            .state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed);
        Ok(Response::new(HealthResponse {
            status: HEALTH_STATUS_OK.into(),
            uptime_seconds: uptime,
            requests_served: served,
        }))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        self.state.bump_requests();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;

        let config = &model.config;
        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        let fpl = config.layers.first().map(|l| l.num_features).unwrap_or(0);

        let has_inference = config.extract_level == larql_vindex::ExtractLevel::Inference
            || config.extract_level == larql_vindex::ExtractLevel::All
            || config.has_model_weights;

        let bands = config.layer_bands.as_ref().map(|b| LayerBands {
            syntax: vec![b.syntax.0 as u32, b.syntax.1 as u32],
            knowledge: vec![b.knowledge.0 as u32, b.knowledge.1 as u32],
            output: vec![b.output.0 as u32, b.output.1 as u32],
        });

        Ok(Response::new(StatsResponse {
            model: config.model.clone(),
            family: config.family.clone(),
            layers: config.num_layers as u32,
            features: total_features as u32,
            features_per_layer: fpl as u32,
            hidden_size: config.hidden_size as u32,
            vocab_size: config.vocab_size as u32,
            extract_level: config.extract_level.to_string(),
            dtype: config.dtype.to_string(),
            layer_bands: bands,
            loaded: Some(LoadedStatus {
                browse: true,
                inference: has_inference && !model.infer_disabled,
            }),
        }))
    }

    async fn describe(
        &self,
        request: Request<DescribeRequest>,
    ) -> Result<Response<DescribeResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let result = tokio::task::spawn_blocking(move || grpc_describe(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn walk(&self, request: Request<WalkRequest>) -> Result<Response<WalkResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let result = tokio::task::spawn_blocking(move || grpc_walk(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn select(
        &self,
        request: Request<SelectRequest>,
    ) -> Result<Response<SelectResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let result = tokio::task::spawn_blocking(move || grpc_select(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;

        if model.infer_disabled {
            return Err(Status::unavailable("inference disabled (--no-infer)"));
        }

        let model = Arc::clone(model);
        let result = tokio::task::spawn_blocking(move || grpc_infer(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn get_relations(
        &self,
        _request: Request<RelationsRequest>,
    ) -> Result<Response<RelationsResponse>, Status> {
        self.state.bump_requests();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let result = tokio::task::spawn_blocking(move || grpc_relations(&model))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    async fn walk_ffn(
        &self,
        request: Request<WalkFfnRequest>,
    ) -> Result<Response<WalkFfnResponse>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let result = tokio::task::spawn_blocking(move || grpc_walk_ffn(&model, &req))
            .await
            .map_err(|e| Status::internal(e.to_string()))??;

        Ok(Response::new(result))
    }

    type StreamDescribeStream = ReceiverStream<Result<DescribeLayerEvent, Status>>;

    async fn stream_describe(
        &self,
        request: Request<DescribeRequest>,
    ) -> Result<Response<Self::StreamDescribeStream>, Status> {
        self.state.bump_requests();
        let req = request.into_inner();
        let model = self
            .state
            .model(None)
            .ok_or_else(|| Status::not_found("no model loaded"))?;
        let model = Arc::clone(model);

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::task::spawn_blocking(move || {
            grpc_stream_describe(&model, &req, &tx);
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// ── Blocking handler implementations ──

fn grpc_describe(
    model: &crate::state::LoadedModel,
    req: &DescribeRequest,
) -> Result<DescribeResponse, Status> {
    let start = std::time::Instant::now();

    let encoding = model
        .tokenizer
        .encode(req.entity.as_str(), false)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(DescribeResponse {
            entity: req.entity.clone(),
            model: model.config.model.clone(),
            edges: vec![],
            latency_ms: 0.0,
        });
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let limit = if req.limit > 0 {
        req.limit as usize
    } else {
        20
    };
    let min_score = if req.min_score > 0.0 {
        req.min_score
    } else {
        5.0
    };

    let trace = patched.walk(&query, &all_layers, limit);
    let entity_lower = req.entity.to_lowercase();

    let mut edges = Vec::new();
    for (layer, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < min_score {
                continue;
            }
            let tok = hit.meta.top_token.trim();
            if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                continue;
            }

            let (relation, source) = model
                .probe_labels
                .get(&(*layer, hit.feature))
                .map(|r| (r.clone(), PROBE_RELATION_SOURCE.to_string()))
                .unwrap_or_default();

            edges.push(DescribeEdge {
                target: tok.to_string(),
                gate_score: hit.gate_score,
                layer: *layer as u32,
                relation,
                source,
                also: vec![],
                layer_min: 0,
                layer_max: 0,
                count: 0,
            });
        }
    }

    edges.sort_by(|a, b| b.gate_score.partial_cmp(&a.gate_score).unwrap());
    edges.truncate(limit);

    Ok(DescribeResponse {
        entity: req.entity.clone(),
        model: model.config.model.clone(),
        edges,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_walk(model: &crate::state::LoadedModel, req: &WalkRequest) -> Result<WalkResponse, Status> {
    let start = std::time::Instant::now();
    let top_k = if req.top > 0 { req.top as usize } else { 5 };

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(Status::invalid_argument("empty prompt"));
    }

    let last_tok = *token_ids.last().unwrap();
    let query = model
        .embeddings
        .row(last_tok as usize)
        .mapv(|v| v * model.embed_scale);

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let trace = patched.walk(&query, &all_layers, top_k);

    let hits: Vec<WalkHit> = trace
        .layers
        .iter()
        .flat_map(|(layer, hits)| {
            hits.iter().map(move |hit| {
                let relation = model
                    .probe_labels
                    .get(&(*layer, hit.feature))
                    .cloned()
                    .unwrap_or_default();
                proto::WalkHit {
                    layer: *layer as u32,
                    feature: hit.feature as u32,
                    gate_score: hit.gate_score,
                    target: hit.meta.top_token.trim().to_string(),
                    relation,
                }
            })
        })
        .collect();

    Ok(WalkResponse {
        prompt: req.prompt.clone(),
        hits,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_select(
    model: &crate::state::LoadedModel,
    req: &SelectRequest,
) -> Result<SelectResponse, Status> {
    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let limit = if req.limit > 0 {
        req.limit as usize
    } else {
        20
    };

    let scan_layers: Vec<usize> = if req.layer > 0 {
        vec![req.layer as usize]
    } else {
        all_layers
    };

    let mut edges = Vec::new();
    for &layer in &scan_layers {
        if let Some(metas) = patched.down_meta_at(layer) {
            for (feat_idx, meta_opt) in metas.iter().enumerate() {
                if let Some(meta) = meta_opt {
                    if !req.entity.is_empty()
                        && !meta
                            .top_token
                            .to_lowercase()
                            .contains(&req.entity.to_lowercase())
                    {
                        continue;
                    }
                    if req.min_confidence > 0.0 && meta.c_score < req.min_confidence {
                        continue;
                    }
                    let relation = model
                        .probe_labels
                        .get(&(layer, feat_idx))
                        .cloned()
                        .unwrap_or_default();
                    if !req.relation.is_empty()
                        && !relation
                            .to_lowercase()
                            .contains(&req.relation.to_lowercase())
                    {
                        continue;
                    }
                    edges.push(SelectEdge {
                        layer: layer as u32,
                        feature: feat_idx as u32,
                        target: meta.top_token.trim().to_string(),
                        c_score: meta.c_score,
                        relation,
                    });
                }
            }
        }
    }

    edges.sort_by(|a, b| b.c_score.partial_cmp(&a.c_score).unwrap());
    let total = edges.len() as u32;
    edges.truncate(limit);

    Ok(SelectResponse {
        edges,
        total,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_infer(
    model: &crate::state::LoadedModel,
    req: &InferRequest,
) -> Result<InferResponse, Status> {
    let weights_guard = model.get_or_load_weights().map_err(Status::unavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| Status::internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(Status::invalid_argument("empty prompt"));
    }

    let top_k = if req.top > 0 { req.top as usize } else { 5 };
    let start = std::time::Instant::now();
    let mode = if req.mode.is_empty() {
        INFER_MODE_WALK
    } else {
        &req.mode
    };

    let to_preds = |preds: &[(String, f64)]| -> Vec<Prediction> {
        preds
            .iter()
            .map(|(t, p)| Prediction {
                token: t.clone(),
                probability: *p,
            })
            .collect()
    };

    match mode {
        INFER_MODE_COMPARE => {
            let patched = model.patched.blocking_read();
            let walk_pred = larql_inference::infer_patched(
                weights,
                &model.tokenizer,
                &*patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
            );
            let walk_ms = walk_pred.walk_ms as f32;

            let ds = std::time::Instant::now();
            let dense_pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, top_k);
            let dense_ms = ds.elapsed().as_secs_f64() as f32 * 1000.0;

            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: vec![],
                mode: INFER_MODE_COMPARE.into(),
                walk_predictions: to_preds(&walk_pred.predictions),
                dense_predictions: to_preds(&dense_pred.predictions),
                walk_ms,
                dense_ms,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
        INFER_MODE_DENSE => {
            let pred = larql_inference::predict(weights, &model.tokenizer, &token_ids, top_k);
            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: to_preds(&pred.predictions),
                mode: INFER_MODE_DENSE.into(),
                walk_predictions: vec![],
                dense_predictions: vec![],
                walk_ms: 0.0,
                dense_ms: 0.0,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
        _ => {
            let patched = model.patched.blocking_read();
            let pred = larql_inference::infer_patched(
                weights,
                &model.tokenizer,
                &*patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
            );
            Ok(InferResponse {
                prompt: req.prompt.clone(),
                predictions: to_preds(&pred.predictions),
                mode: INFER_MODE_WALK.into(),
                walk_predictions: vec![],
                dense_predictions: vec![],
                walk_ms: 0.0,
                dense_ms: 0.0,
                latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
            })
        }
    }
}

fn grpc_relations(model: &crate::state::LoadedModel) -> Result<RelationsResponse, Status> {
    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    let mut counts: std::collections::HashMap<String, (usize, String)> =
        std::collections::HashMap::new();
    for &layer in &all_layers {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta in metas.iter().flatten() {
                let tok = meta.top_token.trim();
                if tok.len() >= 2 && meta.c_score >= 0.2 {
                    let example = meta
                        .top_k
                        .first()
                        .map(|t| t.token.trim().to_string())
                        .unwrap_or_default();
                    let entry = counts.entry(tok.to_string()).or_insert((0, example));
                    entry.0 += 1;
                }
            }
        }
    }

    let mut relations: Vec<RelationInfo> = counts
        .into_iter()
        .map(|(name, (count, example))| RelationInfo {
            name,
            count: count as u32,
            example,
        })
        .collect();
    relations.sort_by_key(|x| std::cmp::Reverse(x.count));
    let total = relations.len() as u32;
    relations.truncate(50);

    Ok(RelationsResponse {
        relations,
        total,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_walk_ffn(
    model: &crate::state::LoadedModel,
    req: &WalkFfnRequest,
) -> Result<WalkFfnResponse, Status> {
    let start = std::time::Instant::now();
    let hidden = model.config.hidden_size;
    let seq_len = if req.seq_len == 0 {
        1
    } else {
        req.seq_len as usize
    };

    let expected_len = if req.full_output {
        seq_len
            .checked_mul(hidden)
            .ok_or_else(|| Status::invalid_argument("seq_len * hidden overflow"))?
    } else {
        hidden
    };
    if req.residual.len() != expected_len {
        return Err(Status::invalid_argument(format!(
            "residual has {} elements, expected {expected_len} (seq_len={} * hidden={hidden})",
            req.residual.len(),
            if req.full_output { seq_len } else { 1 },
        )));
    }

    let scan_layers: Vec<usize> = if !req.layers.is_empty() {
        req.layers.iter().map(|l| *l as usize).collect()
    } else {
        vec![req.layer as usize]
    };

    let results = if req.full_output {
        grpc_walk_ffn_full_output(model, &scan_layers, &req.residual, seq_len, hidden)?
    } else {
        grpc_walk_ffn_features_only(model, &scan_layers, &req.residual, req.top_k)
    };

    Ok(WalkFfnResponse {
        results,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    })
}

fn grpc_walk_ffn_features_only(
    model: &crate::state::LoadedModel,
    scan_layers: &[usize],
    residual: &[f32],
    top_k_req: u32,
) -> Vec<WalkFfnLayerResult> {
    let patched = model.patched.blocking_read();
    let top_k = if top_k_req > 0 {
        top_k_req as usize
    } else {
        8092
    };
    let query = larql_vindex::ndarray::Array1::from_vec(residual.to_vec());

    scan_layers
        .iter()
        .map(|&layer| {
            let hits = patched.gate_knn(layer, &query, top_k);
            WalkFfnLayerResult {
                layer: layer as u32,
                features: hits.iter().map(|(f, _)| *f as u32).collect(),
                scores: hits.iter().map(|(_, s)| *s).collect(),
                output: Vec::new(),
                seq_len: 0,
            }
        })
        .collect()
}

fn grpc_walk_ffn_full_output(
    model: &crate::state::LoadedModel,
    scan_layers: &[usize],
    residual: &[f32],
    seq_len: usize,
    hidden: usize,
) -> Result<Vec<WalkFfnLayerResult>, Status> {
    use larql_inference::ffn::FfnBackend;
    use larql_vindex::ndarray::Array2;

    let weights_guard = model
        .get_or_load_weights()
        .map_err(Status::failed_precondition)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited(weights, &*patched);

    let x = Array2::from_shape_vec((seq_len, hidden), residual.to_vec())
        .map_err(|e| Status::internal(format!("reshape residual: {e}")))?;

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(Status::invalid_argument(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }
        let out = walk_ffn.forward(layer, &x);
        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);
        results.push(WalkFfnLayerResult {
            layer: layer as u32,
            features: Vec::new(),
            scores: Vec::new(),
            output,
            seq_len: seq_len as u32,
        });
    }
    Ok(results)
}

fn grpc_stream_describe(
    model: &crate::state::LoadedModel,
    req: &DescribeRequest,
    tx: &tokio::sync::mpsc::Sender<Result<DescribeLayerEvent, Status>>,
) {
    let token_ids: Vec<u32> = match model.encode_cached_ids(req.entity.as_str(), false) {
        Ok(ids) => ids,
        Err(_) => return,
    };
    if token_ids.is_empty() {
        let _ = tx.blocking_send(Ok(DescribeLayerEvent {
            layer: 0,
            edges: vec![],
            done: true,
            total_edges: 0,
            latency_ms: 0.0,
        }));
        return;
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let start = std::time::Instant::now();
    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();
    let entity_lower = req.entity.to_lowercase();
    let mut total_edges = 0u32;

    for &layer in &all_layers {
        let hits = patched.gate_knn(layer, &query, 20);
        let mut edges = Vec::new();

        for (feature, gate_score) in &hits {
            if *gate_score < 5.0 {
                continue;
            }
            if let Some(meta) = patched.feature_meta(layer, *feature) {
                let tok = meta.top_token.trim();
                if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                    continue;
                }
                let (relation, source) = model
                    .probe_labels
                    .get(&(layer, *feature))
                    .map(|r| (r.clone(), PROBE_RELATION_SOURCE.to_string()))
                    .unwrap_or_default();
                edges.push(DescribeEdge {
                    target: tok.to_string(),
                    gate_score: *gate_score,
                    layer: layer as u32,
                    relation,
                    source,
                    also: vec![],
                    layer_min: 0,
                    layer_max: 0,
                    count: 0,
                });
            }
        }

        total_edges += edges.len() as u32;

        if tx
            .blocking_send(Ok(DescribeLayerEvent {
                layer: layer as u32,
                edges,
                done: false,
                total_edges: 0,
                latency_ms: 0.0,
            }))
            .is_err()
        {
            return;
        }
    }

    let _ = tx.blocking_send(Ok(DescribeLayerEvent {
        layer: 0,
        edges: vec![],
        done: true,
        total_edges,
        latency_ms: start.elapsed().as_secs_f64() as f32 * 1000.0,
    }));
}
