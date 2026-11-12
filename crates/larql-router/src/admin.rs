//! Helpers + gRPC-client wrappers for the `larql-router` admin
//! subcommands (ADR-0004 Phase 5).
//!
//! Pure helpers (`parse_layers`, `format_status`, `format_gaps`) and the
//! RPC client wrappers (`admin_status`, `admin_gaps`, `admin_drain`,
//! `admin_assign`) both live here. Each RPC wrapper opens one connection
//! to the router, sends the right request, and returns either pre-
//! formatted output lines or the raw `AdminAck`. main.rs prints what it
//! gets back and maps the exit code.

use larql_router_protocol::{
    AdminAck, AssignRangeRequest, DrainRequest, GridServiceClient, StatusRequest, StatusResponse,
};

/// Error type returned by the RPC wrappers. `Box<dyn Error + Send + Sync>`
/// matches the binary's `main()` Result alias.
pub type AdminError = Box<dyn std::error::Error + Send + Sync>;

/// Parse the admin `--layers START-END` flag into an inclusive range.
pub fn parse_layers(s: &str) -> Result<(u32, u32), String> {
    let (a, b) = s
        .split_once('-')
        .ok_or_else(|| format!("--layers: expected START-END, got {s:?}"))?;
    let start: u32 = a
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid start {a:?}"))?;
    let end: u32 = b
        .trim()
        .parse()
        .map_err(|_| format!("--layers: invalid end {b:?}"))?;
    if start > end {
        return Err(format!("--layers: start {start} must be <= end {end}"));
    }
    Ok((start, end))
}

/// Pure renderer for `larql-router status`. Returns the lines the CLI
/// prints; the caller does the actual `println!` (keeps stdout side-
/// effects out of unit tests).
pub fn format_status(resp: &StatusResponse, only_model: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    if resp.models.is_empty() && resp.servers.is_empty() {
        out.push("Grid: empty (no servers registered).".into());
        return out;
    }
    for model in &resp.models {
        if only_model.is_some_and(|m| m != model.model_id) {
            continue;
        }
        let total_layers = model
            .shards
            .iter()
            .map(|s| s.layer_end - s.layer_start + 1)
            .sum::<u32>();
        let coverage_pct = if model.num_layers > 0 {
            (total_layers as f64 / model.num_layers as f64) * 100.0
        } else {
            100.0
        };
        out.push(format!(
            "Model: {} ({} shard{}, coverage ~{:.0}%)",
            model.model_id,
            model.shards.len(),
            if model.shards.len() == 1 { "" } else { "s" },
            coverage_pct,
        ));
        for shard in &model.shards {
            out.push(format!(
                "  layers {:>3}-{:<3}  servers={}  replicas={}",
                shard.layer_start,
                shard.layer_end,
                shard.server_ids.join(","),
                shard.replica_count,
            ));
        }
        for gap in &model.gaps {
            out.push(format!(
                "  GAP layers {}-{}",
                gap.layer_start, gap.layer_end
            ));
        }
    }

    if !resp.servers.is_empty() {
        out.push(String::new());
        out.push("Servers:".into());
        for s in &resp.servers {
            out.push(format!(
                "  {:<24} {:<28} state={} model={}[{}-{}] cpu={:.0}% ram={:.0}MB inflight={}",
                s.server_id,
                s.listen_url,
                s.state,
                s.model_id,
                s.layer_start,
                s.layer_end,
                s.cpu_pct,
                s.ram_used as f64 / 1_048_576.0,
                s.requests_in_flight,
            ));
        }
    }
    out
}

/// Pure renderer for `larql-router gaps`. Returns the lines to print.
pub fn format_gaps(resp: &StatusResponse, only_model: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let mut any = false;
    for model in &resp.models {
        if only_model.is_some_and(|m| m != model.model_id) {
            continue;
        }
        if model.gaps.is_empty() {
            continue;
        }
        any = true;
        out.push(format!("Model: {}", model.model_id));
        for gap in &model.gaps {
            out.push(format!("  layers {}-{}", gap.layer_start, gap.layer_end));
        }
    }
    if !any {
        out.push("No gaps.".into());
    }
    out
}

// ── RPC client wrappers ──────────────────────────────────────────────────────

/// A client for the router's admin RPCs.
///
/// A router started with `--grid-key` refuses every RPC that does not
/// carry the key, so the admin commands send it too.
#[derive(Debug, Clone, Default)]
pub struct AdminClient {
    router_url: String,
    grid_key: Option<String>,
}

impl AdminClient {
    pub fn new(router_url: &str, grid_key: Option<&str>) -> Self {
        Self {
            router_url: router_url.to_string(),
            grid_key: grid_key.map(str::to_string),
        }
    }

    async fn connect(&self) -> Result<GridServiceClient<tonic::transport::Channel>, AdminError> {
        Ok(GridServiceClient::connect(self.router_url.clone()).await?)
    }

    /// `message` as a request carrying the grid key, when there is one.
    fn request<T>(&self, message: T) -> Result<tonic::Request<T>, AdminError> {
        let mut request = tonic::Request::new(message);
        if let Some(key) = &self.grid_key {
            let value = format!("Bearer {key}")
                .parse()
                .map_err(|_| AdminError::from("the grid key is not a valid header value"))?;
            request.metadata_mut().insert("authorization", value);
        }
        Ok(request)
    }

    /// Fetch the live grid status and render it as lines for the CLI.
    pub async fn status(&self) -> Result<Vec<String>, AdminError> {
        let resp = self
            .connect()
            .await?
            .status(self.request(StatusRequest {})?)
            .await?
            .into_inner();
        Ok(format_status(&resp, None))
    }

    /// Fetch live status and render only the gap report.
    pub async fn gaps(&self, model: Option<&str>) -> Result<Vec<String>, AdminError> {
        let resp = self
            .connect()
            .await?
            .status(self.request(StatusRequest {})?)
            .await?
            .into_inner();
        Ok(format_gaps(&resp, model))
    }

    /// Send the drain RPC and return the router's ack verbatim.
    pub async fn drain(&self, server_id: &str, reason: &str) -> Result<AdminAck, AdminError> {
        let request = self.request(DrainRequest {
            server_id: server_id.to_string(),
            reason: reason.to_string(),
        })?;
        Ok(self
            .connect()
            .await?
            .drain_server(request)
            .await?
            .into_inner())
    }

    /// Send the AssignRange RPC after splitting the `START-END` layers.
    pub async fn assign(
        &self,
        model_id: &str,
        layers: &str,
        target_server_id: Option<&str>,
        origin_url: Option<&str>,
        origin_sha256: &str,
    ) -> Result<AdminAck, AdminError> {
        let (start, end) = parse_layers(layers)?;
        let request = self.request(AssignRangeRequest {
            model_id: model_id.to_string(),
            layer_start: start,
            layer_end: end,
            target_server_id: target_server_id.unwrap_or_default().to_string(),
            explicit_origin_url: origin_url.unwrap_or_default().to_string(),
            explicit_origin_sha256: origin_sha256.to_string(),
        })?;
        Ok(self
            .connect()
            .await?
            .assign_range(request)
            .await?
            .into_inner())
    }
}

/// `larql-router status` — fetch the live grid status and render it as a
/// list of lines for the CLI to print. Sends no grid key; see
/// [`AdminClient`].
pub async fn admin_status(router_url: &str) -> Result<Vec<String>, AdminError> {
    AdminClient::new(router_url, None).status().await
}

/// `larql-router gaps` — fetch live status and render only the gap report.
pub async fn admin_gaps(router_url: &str, model: Option<&str>) -> Result<Vec<String>, AdminError> {
    AdminClient::new(router_url, None).gaps(model).await
}

/// `larql-router drain` — send the drain RPC and return the router's ack
/// verbatim. The CLI converts that to a success-line / error+exit-code.
pub async fn admin_drain(
    router_url: &str,
    server_id: &str,
    reason: &str,
) -> Result<AdminAck, AdminError> {
    AdminClient::new(router_url, None)
        .drain(server_id, reason)
        .await
}

/// `larql-router assign` — send the AssignRange RPC after splitting the
/// `--layers START-END` flag.
pub async fn admin_assign(
    router_url: &str,
    model_id: &str,
    layers: &str,
    target_server_id: Option<&str>,
    origin_url: Option<&str>,
    origin_sha256: &str,
) -> Result<AdminAck, AdminError> {
    AdminClient::new(router_url, None)
        .assign(
            model_id,
            layers,
            target_server_id,
            origin_url,
            origin_sha256,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_router_protocol::{Gap, ModelCoverage, ServerInfo, ShardInfo};

    // ── parse_layers ─────────────────────────────────────────────────────

    #[test]
    fn parse_layers_accepts_inclusive_pairs() {
        assert_eq!(parse_layers("0-14"), Ok((0, 14)));
        assert_eq!(parse_layers("3-3"), Ok((3, 3)));
        // Whitespace tolerated.
        assert_eq!(parse_layers("  0 - 4 "), Ok((0, 4)));
    }

    #[test]
    fn parse_layers_rejects_missing_dash() {
        let err = parse_layers("0").unwrap_err();
        assert!(err.contains("expected START-END"));
    }

    #[test]
    fn parse_layers_rejects_non_numeric() {
        assert!(parse_layers("a-b").is_err());
        assert!(parse_layers("0-z").is_err());
    }

    #[test]
    fn parse_layers_rejects_inverted_range() {
        let err = parse_layers("10-3").unwrap_err();
        assert!(err.contains("start 10 must be <= end 3"));
    }

    // ── format_status ────────────────────────────────────────────────────

    fn empty_status() -> StatusResponse {
        StatusResponse {
            models: vec![],
            servers: vec![],
        }
    }

    #[test]
    fn format_status_empty_grid_says_so() {
        let out = format_status(&empty_status(), None);
        assert_eq!(out.len(), 1);
        assert!(out[0].contains("empty"));
    }

    fn sample_status() -> StatusResponse {
        StatusResponse {
            models: vec![ModelCoverage {
                model_id: "gemma3-4b".into(),
                num_layers: 34,
                shards: vec![
                    ShardInfo {
                        layer_start: 0,
                        layer_end: 16,
                        server_ids: vec!["srv-a".into()],
                        replica_count: 1,
                    },
                    ShardInfo {
                        layer_start: 17,
                        layer_end: 33,
                        server_ids: vec!["srv-b".into(), "srv-c".into()],
                        replica_count: 2,
                    },
                ],
                gaps: vec![Gap {
                    layer_start: 34,
                    layer_end: 40,
                }],
            }],
            servers: vec![ServerInfo {
                server_id: "srv-a".into(),
                listen_url: "http://host-a:8080".into(),
                state: "serving".into(),
                model_id: "gemma3-4b".into(),
                layer_start: 0,
                layer_end: 16,
                cpu_pct: 42.0,
                ram_used: 2 * 1024 * 1024 * 1024,
                requests_in_flight: 3,
                rtt_ms: 0,
                layer_stats: vec![],
            }],
        }
    }

    #[test]
    fn format_status_renders_model_shards_servers_and_gaps() {
        let out = format_status(&sample_status(), None);
        let joined = out.join("\n");
        assert!(joined.contains("Model: gemma3-4b (2 shards"));
        assert!(joined.contains("0-16"));
        assert!(joined.contains("17-33"));
        assert!(joined.contains("srv-b,srv-c"));
        assert!(joined.contains("replicas=2"));
        assert!(joined.contains("GAP layers 34-40"));
        assert!(joined.contains("Servers:"));
        assert!(joined.contains("srv-a"));
        assert!(joined.contains("ram=2048MB"));
        assert!(joined.contains("inflight=3"));
    }

    #[test]
    fn format_status_filters_to_named_model() {
        let mut status = sample_status();
        // Add a second model the filter should hide.
        status.models.push(ModelCoverage {
            model_id: "other".into(),
            num_layers: 0,
            shards: vec![],
            gaps: vec![],
        });
        let out = format_status(&status, Some("gemma3-4b")).join("\n");
        assert!(out.contains("Model: gemma3-4b"));
        assert!(!out.contains("Model: other"));
    }

    #[test]
    fn format_status_handles_unknown_total_layer_count_as_full_coverage() {
        let mut status = sample_status();
        // num_layers = 0 → coverage_pct should still render at 100%, not
        // panic on divide-by-zero.
        status.models[0].num_layers = 0;
        let out = format_status(&status, None).join("\n");
        assert!(out.contains("coverage ~100%"));
    }

    // ── format_gaps ──────────────────────────────────────────────────────

    #[test]
    fn format_gaps_empty_status_reports_no_gaps() {
        let out = format_gaps(&empty_status(), None);
        assert_eq!(out, vec!["No gaps.".to_string()]);
    }

    #[test]
    fn format_gaps_lists_per_model_gaps() {
        let out = format_gaps(&sample_status(), None);
        let joined = out.join("\n");
        assert!(joined.contains("Model: gemma3-4b"));
        assert!(joined.contains("layers 34-40"));
    }

    #[test]
    fn format_gaps_filter_matches_only_named_model() {
        let mut status = sample_status();
        status.models.push(ModelCoverage {
            model_id: "other".into(),
            num_layers: 0,
            shards: vec![],
            gaps: vec![Gap {
                layer_start: 0,
                layer_end: 4,
            }],
        });
        let out = format_gaps(&status, Some("other")).join("\n");
        assert!(out.contains("Model: other"));
        assert!(!out.contains("gemma3-4b"));
    }

    #[test]
    fn format_gaps_models_without_gaps_are_skipped() {
        let mut status = sample_status();
        status.models[0].gaps.clear();
        let out = format_gaps(&status, None);
        assert_eq!(out, vec!["No gaps.".to_string()]);
    }
}
