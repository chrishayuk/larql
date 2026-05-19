//! HTTP client for the LARQL remote FFN protocol.
//!
//! `RemoteWalkBackend` holds a blocking HTTP client and dispatches FFN calls
//! to a `larql-server` over HTTP, implementing the same [`FfnBackend`] trait
//! as [`WalkFfn`](crate::vindex::WalkFfn).

use portable_atomic::AtomicU64;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use ndarray::Array2;

use super::codec::{
    decode_binary_batch, decode_binary_batch_f16, decode_binary_batch_i8, decode_binary_single,
    decode_binary_single_f16, decode_binary_single_i8, encode_binary_request,
    extract_response_latency_ms, RemoteLatencyStats, WalkFfnSingleResponse, BINARY_CT, F16_CT,
    I8_CT,
};
use super::q8k_wire::{decode_q8k_batch_response, encode_q8k_batch_request, Q8K_BATCH_CT};
use crate::ffn::FfnBackend;
use larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation;

const STATS_PATH: &str = "/v1/stats";
const WALK_FFN_PATH: &str = "/v1/walk-ffn";
const WALK_FFN_Q8K_PATH: &str = "/v1/walk-ffn-q8k";
const HIDDEN_SIZE_KEY: &str = "hidden_size";

// ── Config ───────────────────────────────────────────────────────────────────

/// Wire format preference for a `RemoteWalkBackend` connection.
///
/// Controls the `Accept` header sent on every request. The server honours
/// the best format it supports; if the requested format is unavailable it
/// falls back to f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WirePreference {
    /// Offer all formats; server picks the best (default — f16 on GT1+ servers).
    #[default]
    BestAvailable,
    /// Force f32 (`application/x-larql-ffn`).
    F32,
    /// Force f16 (`application/x-larql-ffn-f16`).
    F16,
    /// Force i8 symmetric (`application/x-larql-ffn-i8`).
    I8,
}

impl WirePreference {
    /// Construct an `Accept` header value for this preference.
    pub fn accept_header(self) -> String {
        match self {
            Self::BestAvailable => format!("{I8_CT}, {F16_CT}, {BINARY_CT}"),
            Self::F32 => BINARY_CT.to_owned(),
            Self::F16 => format!("{F16_CT}, {BINARY_CT}"),
            Self::I8 => format!("{I8_CT}, {F16_CT}, {BINARY_CT}"),
        }
    }

    /// Parse from a short string (`"f32"`, `"f16"`, `"i8"`).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "i8" => Some(Self::I8),
            _ => None,
        }
    }

    /// Short label for display.
    pub fn label(self) -> &'static str {
        match self {
            Self::BestAvailable => "best",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
        }
    }
}

/// Client config for talking to a remote FFN server.
#[derive(Clone, Debug)]
pub struct RemoteFfnConfig {
    /// Base URL, e.g. `"https://ffn.example.com:8080"`. Trailing slash
    /// stripped automatically.
    pub base_url: String,
    /// Per-request timeout. Applied to both connect and read.
    pub timeout: Duration,
    /// Wire format preference. Controls the `Accept` header.
    pub wire: WirePreference,
}

impl RemoteFfnConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(60),
            wire: WirePreference::BestAvailable,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_wire(mut self, wire: WirePreference) -> Self {
        self.wire = wire;
        self
    }
}

// ── Client ───────────────────────────────────────────────────────────────────

/// Remote FFN backend. Holds a blocking HTTP client plus the server URL.
///
/// Cloning is cheap — the underlying `reqwest::blocking::Client` is
/// connection-pooled and `Arc`-shared.
pub struct RemoteWalkBackend {
    config: RemoteFfnConfig,
    client: reqwest::blocking::Client,
    hidden_size: usize,
    /// Cumulative bytes sent (request bodies) across all calls.
    wire_bytes_sent: Arc<AtomicU64>,
    /// Cumulative bytes received (response bodies) across all calls.
    wire_bytes_recv: Arc<AtomicU64>,
}

impl RemoteWalkBackend {
    /// Build a backend. Performs a one-shot health check against
    /// `/v1/stats` so we fail fast if the server is unreachable at
    /// construction time rather than mid-forward-pass.
    pub fn connect(config: RemoteFfnConfig) -> Result<Self, RemoteFfnError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| RemoteFfnError::Client(e.to_string()))?;

        let stats_url = format!("{}{STATS_PATH}", config.base_url);
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| RemoteFfnError::Unreachable {
                url: stats_url.clone(),
                cause: e.to_string(),
            })?;
        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }
        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
        let hidden_size = stats[HIDDEN_SIZE_KEY].as_u64().ok_or_else(|| {
            RemoteFfnError::BadResponse(format!("stats missing {HIDDEN_SIZE_KEY}"))
        })? as usize;

        Ok(Self {
            config,
            client,
            hidden_size,
            wire_bytes_sent: Arc::new(AtomicU64::new(0)),
            wire_bytes_recv: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Reset wire byte counters. Call between bench runs to isolate measurements.
    pub fn reset_wire_counters(&self) {
        self.wire_bytes_sent.store(0, Ordering::Relaxed);
        self.wire_bytes_recv.store(0, Ordering::Relaxed);
    }

    /// Total bytes sent since construction or last `reset_wire_counters`.
    pub fn wire_bytes_sent(&self) -> u64 {
        self.wire_bytes_sent.load(Ordering::Relaxed)
    }

    /// Total bytes received since construction or last `reset_wire_counters`.
    pub fn wire_bytes_recv(&self) -> u64 {
        self.wire_bytes_recv.load(Ordering::Relaxed)
    }

    /// Hidden size advertised by the remote server.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Single-layer FFN call using the binary wire format.
    /// Returns a `Vec<f32>` of length `seq_len * hidden_size`, row-major.
    fn call_single(
        &self,
        layer: usize,
        residual_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, RemoteFfnError> {
        let url = format!("{}{WALK_FFN_PATH}", self.config.base_url);
        let body = encode_binary_request(Some(layer), None, residual_flat, seq_len, true, 8092);
        self.wire_bytes_sent
            .fetch_add(body.len() as u64, Ordering::Relaxed);

        let accept = self.config.wire.accept_header();
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
            .header(reqwest::header::ACCEPT, accept)
            .body(body)
            .send()
            .map_err(|e| RemoteFfnError::Http {
                layer,
                cause: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let resp_bytes = resp
            .bytes()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
        self.wire_bytes_recv
            .fetch_add(resp_bytes.len() as u64, Ordering::Relaxed);

        let output = if ct.starts_with(I8_CT) {
            let (_, floats) = decode_binary_single_i8(&resp_bytes, self.hidden_size)
                .map_err(RemoteFfnError::BadResponse)?;
            floats
        } else if ct.starts_with(F16_CT) {
            let (_, floats) =
                decode_binary_single_f16(&resp_bytes).map_err(RemoteFfnError::BadResponse)?;
            floats
        } else if ct.starts_with(BINARY_CT) {
            let (_, floats) =
                decode_binary_single(&resp_bytes).map_err(RemoteFfnError::BadResponse)?;
            floats
        } else {
            // Fallback: server returned JSON.
            let parsed: WalkFfnSingleResponse = serde_json::from_slice(&resp_bytes)
                .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            parsed.output
        };

        let expected = seq_len * self.hidden_size;
        if output.len() != expected {
            return Err(RemoteFfnError::BadResponse(format!(
                "layer {layer}: expected {expected} output floats, got {}",
                output.len()
            )));
        }
        Ok(output)
    }

    /// Batch FFN call — sends all `layers` in one round trip using the binary
    /// wire format. Returns a map from layer index to output floats.
    ///
    /// The server must serve all requested layers (i.e. they must all be in
    /// the same shard). For cross-shard batches, route through `larql-router`
    /// using JSON.
    pub fn call_batch(
        &self,
        layers: &[usize],
        residual_flat: &[f32],
        seq_len: usize,
    ) -> Result<HashMap<usize, Vec<f32>>, RemoteFfnError> {
        let url = format!("{}{WALK_FFN_PATH}", self.config.base_url);
        let body = encode_binary_request(None, Some(layers), residual_flat, seq_len, true, 8092);
        self.wire_bytes_sent
            .fetch_add(body.len() as u64, Ordering::Relaxed);

        let accept = self.config.wire.accept_header();
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
            .header(reqwest::header::ACCEPT, accept)
            .body(body)
            .send()
            .map_err(|e| RemoteFfnError::Http {
                layer: layers.first().copied().unwrap_or(0),
                cause: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let resp_bytes = resp
            .bytes()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
        self.wire_bytes_recv
            .fetch_add(resp_bytes.len() as u64, Ordering::Relaxed);

        if ct.starts_with(I8_CT) {
            decode_binary_batch_i8(&resp_bytes, self.hidden_size)
                .map_err(RemoteFfnError::BadResponse)
        } else if ct.starts_with(F16_CT) {
            decode_binary_batch_f16(&resp_bytes).map_err(RemoteFfnError::BadResponse)
        } else if ct.starts_with(BINARY_CT) {
            decode_binary_batch(&resp_bytes).map_err(RemoteFfnError::BadResponse)
        } else {
            // Fallback: JSON batch response.
            let v: serde_json::Value = serde_json::from_slice(&resp_bytes)
                .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            let mut out = HashMap::new();
            // Single-layer JSON response.
            if let Some(layer) = v.get("layer").and_then(|l| l.as_u64()) {
                let floats = json_output_floats(&v)?;
                out.insert(layer as usize, floats);
                return Ok(out);
            }
            // Multi-layer JSON response.
            if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
                for entry in results {
                    let layer = entry["layer"].as_u64().ok_or_else(|| {
                        RemoteFfnError::BadResponse("batch JSON: missing layer".into())
                    })? as usize;
                    let floats = json_output_floats(entry)?;
                    out.insert(layer, floats);
                }
                return Ok(out);
            }
            Err(RemoteFfnError::BadResponse(
                "batch response has neither 'layer' nor 'results'".into(),
            ))
        }
    }

    /// Q8K batch FFN call — sends pre-normed Q8K activations for one or more
    /// layers in a single HTTP round trip to `/v1/walk-ffn-q8k`.
    ///
    /// Returns a map from layer index to output floats, same as `call_batch`.
    ///
    /// Falls back to `Err` with a "not supported" message when the server
    /// returns 404 (older server without the Q8K endpoint), so callers can
    /// gracefully fall back to the f32 path.
    pub fn call_q8k_layers(
        &self,
        layers: &[(usize, &Q8KActivation)],
    ) -> Result<HashMap<usize, Vec<f32>>, RemoteFfnError> {
        let url = format!("{}{WALK_FFN_Q8K_PATH}", self.config.base_url);
        let body = encode_q8k_batch_request(layers);

        let first_layer = layers.first().map(|(l, _)| *l).unwrap_or(0);
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, Q8K_BATCH_CT)
            .body(body)
            .send()
            .map_err(|e| RemoteFfnError::Http {
                layer: first_layer,
                cause: e.to_string(),
            })?;

        // 404 means the server doesn't support the Q8K endpoint yet.
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(RemoteFfnError::BadResponse(
                "server does not support /v1/walk-ffn-q8k (404)".into(),
            ));
        }
        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let resp_bytes = resp
            .bytes()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;

        decode_q8k_batch_response(&resp_bytes).map_err(RemoteFfnError::BadResponse)
    }

    /// Measure round-trip latency breakdown over `n` calls.
    ///
    /// Sends a zero residual batch covering `layers` each time and reports:
    /// - `total_ms`: wall-clock time measured by the client
    /// - `server_ms`: compute time reported by the server in the response header
    /// - `overhead_ms`: `total_ms - server_ms` (HTTP + TCP + framing)
    ///
    /// First call is a warmup (excluded from stats). Results are averaged over
    /// the remaining `n - 1` calls.
    pub fn probe_latency(
        &self,
        layers: &[usize],
        n: usize,
    ) -> Result<RemoteLatencyStats, RemoteFfnError> {
        assert!(
            n >= 2,
            "probe_latency: need at least 2 calls (1 warmup + 1 measured)"
        );
        let residual = vec![0.0f32; self.hidden_size];
        let url = format!("{}{WALK_FFN_PATH}", self.config.base_url);
        let body = encode_binary_request(None, Some(layers), &residual, 1, true, 8092);

        let mut totals = Vec::with_capacity(n - 1);
        let mut servers = Vec::with_capacity(n - 1);

        for i in 0..n {
            let t0 = std::time::Instant::now();
            let resp = self
                .client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
                .body(body.clone())
                .send()
                .map_err(|e| RemoteFfnError::Http {
                    layer: layers[0],
                    cause: e.to_string(),
                })?;
            if !resp.status().is_success() {
                return Err(RemoteFfnError::ServerError {
                    status: resp.status().as_u16(),
                    body: resp.text().unwrap_or_default(),
                });
            }
            let resp_bytes = resp
                .bytes()
                .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Extract server-reported latency from bytes 8-11 of response.
            let server_ms = extract_response_latency_ms(&resp_bytes);

            if i > 0 {
                // Skip warmup call.
                totals.push(total_ms);
                servers.push(server_ms);
            }
        }

        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let total_ms = avg(&totals);
        let server_ms = avg(&servers);
        Ok(RemoteLatencyStats {
            total_ms,
            server_ms,
            overhead_ms: total_ms - server_ms,
            hidden_size: self.hidden_size,
            num_layers: layers.len(),
            samples: n - 1,
        })
    }

    /// Run the full FFN forward pass for every layer in `layers`, returning
    /// a map from layer → `Array2<f32>` shaped `[seq_len, hidden]`.
    ///
    /// All layers are sent in a single HTTP round trip (binary batch format).
    pub fn forward_all_layers(
        &self,
        layers: &[usize],
        x: &Array2<f32>,
    ) -> Result<HashMap<usize, Array2<f32>>, RemoteFfnError> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        assert_eq!(
            hidden, self.hidden_size,
            "RemoteWalkBackend: input hidden {hidden} != server hidden {}",
            self.hidden_size
        );
        let residual_flat: Vec<f32> = x.iter().copied().collect();
        let flat_map = self.call_batch(layers, &residual_flat, seq_len)?;
        let mut result = HashMap::with_capacity(flat_map.len());
        for (layer, floats) in flat_map {
            if floats.len() != seq_len * hidden {
                return Err(RemoteFfnError::BadResponse(format!(
                    "layer {layer}: expected {} output floats, got {}",
                    seq_len * hidden,
                    floats.len()
                )));
            }
            let arr =
                Array2::from_shape_vec((seq_len, hidden), floats).expect("shape validated above");
            result.insert(layer, arr);
        }
        Ok(result)
    }
}

impl FfnBackend for RemoteWalkBackend {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        assert_eq!(
            hidden, self.hidden_size,
            "RemoteWalkBackend: input hidden {hidden} != server hidden {}",
            self.hidden_size
        );

        let residual_flat: Vec<f32> = x.iter().copied().collect();
        let output = self
            .call_single(layer, &residual_flat, seq_len)
            .unwrap_or_else(|e| panic!("RemoteWalkBackend layer {layer}: {e}"));

        Array2::from_shape_vec((seq_len, hidden), output)
            .expect("RemoteWalkBackend: server output shape mismatch (validated above)")
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let out = self.forward(layer, x);
        let seq_len = x.shape()[0];
        let zeros = Array2::<f32>::zeros((seq_len, 1));
        (out, zeros)
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        let seq_len = h_post_attn.nrows();
        let hidden = h_post_attn.ncols();
        let residual: Vec<f32> = h_post_attn.iter().copied().collect();
        let body = serde_json::json!({
            "layer": layer,
            "residual": residual,
            "seq_len": seq_len,
            "full_output": true,
            "moe_layer": true,
        });
        let url = format!("{}{WALK_FFN_PATH}", self.config.base_url);
        let resp = self.client.post(&url).json(&body).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: serde_json::Value = resp.json().ok()?;
        let floats = v["output"]
            .as_array()?
            .iter()
            .filter_map(|x| x.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>();
        if floats.len() != seq_len * hidden {
            return None;
        }
        Array2::from_shape_vec((seq_len, hidden), floats).ok()
    }

    fn name(&self) -> &str {
        "remote-walk"
    }
}

// ── JSON fallback helper ──────────────────────────────────────────────────────

fn json_output_floats(v: &serde_json::Value) -> Result<Vec<f32>, RemoteFfnError> {
    v.get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| RemoteFfnError::BadResponse("missing 'output' array".into()))
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect()
        })
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum RemoteFfnError {
    #[error("remote FFN client setup failed: {0}")]
    Client(String),

    #[error("remote FFN server unreachable at {url}: {cause}")]
    Unreachable { url: String, cause: String },

    #[error("remote FFN HTTP call for layer {layer} failed: {cause}")]
    Http { layer: usize, cause: String },

    #[error("remote FFN server returned {status}: {body}")]
    ServerError { status: u16, body: String },

    #[error("remote FFN bad response: {0}")]
    BadResponse(String),
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ── RemoteFfnConfig ───────────────────────────────────────────────────────

    #[test]
    fn config_strips_trailing_slash() {
        let c = RemoteFfnConfig::new("https://example.com:8080/");
        assert_eq!(c.base_url, "https://example.com:8080");
    }

    #[test]
    fn config_strips_multiple_trailing_slashes() {
        let c = RemoteFfnConfig::new("https://example.com:8080///");
        assert_eq!(c.base_url, "https://example.com:8080");
    }

    #[test]
    fn config_preserves_url_without_trailing_slash() {
        let c = RemoteFfnConfig::new("http://127.0.0.1:8080");
        assert_eq!(c.base_url, "http://127.0.0.1:8080");
    }

    #[test]
    fn config_default_timeout_is_nontrivial() {
        let c = RemoteFfnConfig::new("http://x");
        assert!(c.timeout.as_secs() >= 10);
    }

    #[test]
    fn config_with_timeout_overrides_default() {
        let c = RemoteFfnConfig::new("http://x").with_timeout(Duration::from_secs(5));
        assert_eq!(c.timeout.as_secs(), 5);
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn error_display_messages_are_actionable() {
        let e = RemoteFfnError::Unreachable {
            url: "http://nope:1234".into(),
            cause: "connection refused".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("http://nope:1234"));
        assert!(s.contains("connection refused"));

        let e = RemoteFfnError::Http {
            layer: 7,
            cause: "timed out".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("layer 7"));
        assert!(s.contains("timed out"));

        let e = RemoteFfnError::ServerError {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn connect_fails_fast_on_unreachable_url() {
        let cfg =
            RemoteFfnConfig::new("http://127.0.0.1:1").with_timeout(Duration::from_millis(500));
        match RemoteWalkBackend::connect(cfg) {
            Ok(_) => panic!("expected connect to fail against 127.0.0.1:1"),
            Err(RemoteFfnError::Unreachable { url, .. }) => {
                assert!(url.contains("127.0.0.1:1"));
            }
            Err(other) => panic!("expected Unreachable, got {other:?}"),
        }
    }
}
