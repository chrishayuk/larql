//! Active-probe RTT collector for the route() tie-breaker.
//!
//! Optional sidecar to the rebalancer. When enabled via
//! `--rtt-probe-interval-secs`, this task periodically issues
//! `GET {listen_url}/v1/health` against every currently-serving
//! server and records the wall-clock round-trip as `rtt_ms` on the
//! corresponding [`crate::grid::ServerEntry`].
//!
//! Why a separate probe rather than piggybacking on heartbeats:
//! heartbeats are server→router only — the router never gets an
//! application-layer ack, so it can't compute a round-trip from the
//! existing stream. An explicit HTTP probe also measures the same
//! transport the production traffic uses (`POST /v1/walk-ffn`), so
//! the recorded RTT reflects realistic queueing on the server's
//! HTTP listener — not just the TCP layer.
//!
//! The probe is opt-in for two reasons:
//!   1. It adds a small constant rate of HTTP traffic per server,
//!      which is wasteful on single-host deployments where RTT
//!      variance is below the GT3 noise floor.
//!   2. In a steady production grid the GT3 per-layer latency
//!      already subsumes the RTT (it includes both compute + wire),
//!      so the probe is mainly useful for cold-start and
//!      cross-region tie-breaking.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::debug;

use crate::grid::GridState;
use crate::metrics::RouterMetrics;

/// Per-probe HTTP timeout. Independent of `--timeout-secs` (which
/// gates fan-out traffic, much heavier than a HEAD probe).
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// `GET /v1/health` path the probe issues. Kept as a constant so a
/// future server-side rename gets caught in one place.
const HEALTH_PATH: &str = "/v1/health";

/// Configuration for the probe task.
#[derive(Clone, Debug)]
pub struct RttProbeConfig {
    /// Cadence between probe rounds. Each round contacts every
    /// currently-serving server in parallel.
    pub interval: Duration,
}

impl RttProbeConfig {
    /// Build from a CLI `--rtt-probe-interval-secs` value. Returns
    /// `None` when the interval is 0 (the feature is disabled), so
    /// callers can `match` on the result and skip the spawn.
    pub fn from_cli(interval_secs: u64) -> Option<Self> {
        if interval_secs == 0 {
            None
        } else {
            Some(Self {
                interval: Duration::from_secs(interval_secs),
            })
        }
    }
}

/// Spawn the probe task. Returns immediately; the task runs for the
/// process lifetime.
pub fn spawn(
    state: Arc<RwLock<GridState>>,
    cfg: RttProbeConfig,
    metrics: Option<Arc<RouterMetrics>>,
) {
    let client = match reqwest::Client::builder().timeout(PROBE_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("RTT probe: cannot build HTTP client ({e}); probes disabled");
            return;
        }
    };
    tokio::spawn(probe_task(state, cfg, client, metrics));
}

async fn probe_task(
    state: Arc<RwLock<GridState>>,
    cfg: RttProbeConfig,
    client: reqwest::Client,
    metrics: Option<Arc<RouterMetrics>>,
) {
    let mut interval = tokio::time::interval(cfg.interval);
    // Skip the immediate first tick — give the grid a moment to fill
    // before the first probe round.
    interval.tick().await;
    loop {
        interval.tick().await;
        probe_round(&state, &client, metrics.as_deref()).await;
    }
}

/// One probe round: snapshot the current server list, hit each one
/// in parallel, write results back under the write lock.
async fn probe_round(
    state: &Arc<RwLock<GridState>>,
    client: &reqwest::Client,
    metrics: Option<&RouterMetrics>,
) {
    // Snapshot under a read lock so the probes run lock-free.
    let targets: Vec<(String, String)> = {
        let g = state.read();
        g.servers()
            .map(|(id, e)| (id.clone(), e.listen_url.clone()))
            .collect()
    };
    if targets.is_empty() {
        return;
    }
    let probes = targets
        .into_iter()
        .map(|(id, url)| probe_one(client.clone(), id, url, metrics));
    let results: Vec<(String, Option<f32>)> = futures::future::join_all(probes).await;

    // Write phase: single write lock, batch updates.
    let mut g = state.write();
    for (server_id, rtt) in results {
        g.update_rtt_ms(&server_id, rtt);
    }
}

/// Probe one server. Returns `Some(rtt_ms)` on a successful 2xx
/// response, `None` otherwise (the server gets its `rtt_ms` cleared
/// — better than reporting stale data).
async fn probe_one(
    client: reqwest::Client,
    server_id: String,
    listen_url: String,
    metrics: Option<&RouterMetrics>,
) -> (String, Option<f32>) {
    let url = format!("{}{}", listen_url.trim_end_matches('/'), HEALTH_PATH);
    let t0 = Instant::now();
    let outcome: (String, Option<f32>, &'static str) = match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => {
            let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
            (server_id, Some(elapsed_ms), "success")
        }
        Ok(r) => {
            debug!(server_id, status = %r.status(), "RTT probe: non-2xx response");
            (server_id, None, "non_2xx")
        }
        Err(e) => {
            debug!(server_id, error = %e, "RTT probe: request failed");
            (server_id, None, "error")
        }
    };
    if let Some(m) = metrics {
        m.rtt_probes_total.with_label_values(&[outcome.2]).inc();
    }
    (outcome.0, outcome.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_cli_zero_disables_probe() {
        assert!(RttProbeConfig::from_cli(0).is_none());
    }

    #[test]
    fn from_cli_positive_carries_interval() {
        let cfg = RttProbeConfig::from_cli(60).unwrap();
        assert_eq!(cfg.interval, Duration::from_secs(60));
    }

    /// A non-routable URL fails fast. The probe must surface `None`
    /// rather than panic or block past the configured timeout.
    #[tokio::test]
    async fn probe_one_returns_none_on_unreachable_host() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        let (id, rtt) = probe_one(
            client,
            "stub".into(),
            "http://127.0.0.1:1".into(), // never listened on
            None,
        )
        .await;
        assert_eq!(id, "stub");
        assert!(rtt.is_none(), "got {rtt:?}");
    }

    /// Empty server list: `probe_round` short-circuits without
    /// acquiring the write lock. Verifies the read-lock-then-maybe-
    /// write-lock contract holds.
    #[tokio::test]
    async fn probe_round_is_noop_on_empty_grid() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let client = reqwest::Client::builder().build().unwrap();
        probe_round(&state, &client, None).await;
        // No assertions on side effects — the test passes if it
        // returns within the timeout, proving no lock deadlock and
        // no panic on the empty path.
    }

    /// Spawn a tiny axum server that returns 200 on `/v1/health` so
    /// we can exercise `probe_one`'s success branch and
    /// `probe_round`'s write-back end-to-end.
    async fn spawn_health_server(status: axum::http::StatusCode) -> std::net::SocketAddr {
        use axum::{routing::get, Router};
        let app = Router::new().route("/v1/health", get(move || async move { (status, "ok") }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // Give axum a tick to start accepting.
        tokio::time::sleep(Duration::from_millis(50)).await;
        addr
    }

    fn make_entry(server_id: &str, listen_url: &str) -> crate::grid::ServerEntry {
        crate::grid::ServerEntry {
            server_id: server_id.into(),
            listen_url: listen_url.into(),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 4,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 0,
        }
    }

    #[tokio::test]
    async fn probe_one_success_returns_positive_rtt() {
        let addr = spawn_health_server(axum::http::StatusCode::OK).await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let (id, rtt) = probe_one(client, "srv".into(), format!("http://{addr}"), None).await;
        assert_eq!(id, "srv");
        let rtt = rtt.expect("2xx response must produce a rtt_ms");
        assert!((0.0..1000.0).contains(&rtt), "got {rtt} ms");
    }

    #[tokio::test]
    async fn probe_one_non_2xx_returns_none() {
        let addr = spawn_health_server(axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let (id, rtt) = probe_one(client, "srv".into(), format!("http://{addr}"), None).await;
        assert_eq!(id, "srv");
        assert!(rtt.is_none(), "non-2xx must surface None, got {rtt:?}");
    }

    #[tokio::test]
    async fn probe_round_writes_rtt_to_grid_state() {
        let addr = spawn_health_server(axum::http::StatusCode::OK).await;
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write();
            g.register(make_entry("srv", &format!("http://{addr}")));
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        probe_round(&state, &client, None).await;
        let g = state.read();
        let (_, entry) = g.servers().find(|(id, _)| **id == "srv").unwrap();
        let rtt = entry.rtt_ms.expect("probe_round must write rtt_ms");
        assert!((0.0..1000.0).contains(&rtt), "got {rtt} ms");
    }

    /// `spawn` returns without panicking when the HTTP client builds
    /// cleanly (default config always does). Covers the success
    /// branch of the builder match.
    #[test]
    fn spawn_succeeds_with_default_client() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let state = Arc::new(RwLock::new(GridState::default()));
            // Use the longest allowed interval so the spawned task
            // never gets a chance to fire before the runtime drops it.
            let cfg = RttProbeConfig {
                interval: Duration::from_secs(3600),
            };
            spawn(state, cfg, None);
        });
    }
}
