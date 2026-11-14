use std::time::Duration;

use super::error::RemoteMoeError;

/// Default per-request HTTP timeout for remote-MoE shard calls: 30 s,
/// overridable via [`ShardConfig::with_timeout`].
const DEFAULT_SHARD_TIMEOUT: Duration = Duration::from_secs(30);

// ── Shard configuration ───────────────────────────────────────────────────────

/// One entry in the shard map: an expert-ID range + its URL.
///
/// Two ownership modes (mutually exclusive — `unit_set` takes precedence):
///
///   1. **Layer-uniform range** (`start..=end`) — same expert range applies
///      to every layer. Set via [`ShardConfig::new`] or `--moe-shards
///      "0-63=URL,..."`.
///   2. **Per-(layer, expert) set** (`unit_set`) — explicit ownership for
///      fine-grained shards. Set via [`ShardConfig::with_unit_set`] or
///      `--moe-units-manifest PATH`.
///
/// `start`/`end` are still populated in unit-set mode (carrying the
/// min/max expert id across all units) so RTT probes and existing
/// diagnostics keep working without special-casing.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// First expert ID this shard touches (inclusive).  When `unit_set` is
    /// `Some`, this is the min of the unit set, kept for diagnostics.
    pub start: usize,
    /// Last expert ID this shard touches (inclusive).  When `unit_set` is
    /// `Some`, this is the max of the unit set.
    pub end: usize,
    /// Base URL, e.g. `"http://shard-a.local:8081"`. Trailing slashes stripped.
    pub url: String,
    /// HTTP request timeout (default: 30 s).
    pub timeout: Duration,
    /// Fine-grained ownership: every `(layer, expert_id)` in this set is
    /// owned by this shard.  When `Some`, takes precedence over the
    /// `start..=end` range.  See `crate::ffn::moe_remote::UnitManifest`
    /// for the JSON shape that produces this set.
    pub unit_set: Option<std::sync::Arc<std::collections::HashSet<(usize, usize)>>>,
    /// The key sent to the shard as `Authorization: Bearer <key>`. Taken
    /// from `LARQL_SHARD_API_KEY` when the config is built; `None` sends
    /// no key.
    pub api_key: Option<String>,
}

impl ShardConfig {
    pub fn new(start: usize, end: usize, url: impl Into<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        Self {
            start,
            end,
            url,
            timeout: DEFAULT_SHARD_TIMEOUT,
            unit_set: None,
            api_key: crate::ffn::shard_auth::shard_api_key_from_env(),
        }
    }

    /// Send `api_key` to the shard, or no key for `None`, whatever the
    /// environment said when the config was built.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// Build a shard config that owns an explicit set of `(layer, expert_id)`
    /// pairs.  `start`/`end` are derived from the set's min/max for
    /// diagnostic compatibility; ownership checks use the set itself.
    pub fn with_units(
        url: impl Into<String>,
        units: std::collections::HashSet<(usize, usize)>,
    ) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        let (start, end) = if units.is_empty() {
            (0, 0)
        } else {
            let min = units.iter().map(|(_, e)| *e).min().unwrap();
            let max = units.iter().map(|(_, e)| *e).max().unwrap();
            (min, max)
        };
        Self {
            start,
            end,
            url,
            timeout: DEFAULT_SHARD_TIMEOUT,
            unit_set: Some(std::sync::Arc::new(units)),
            api_key: crate::ffn::shard_auth::shard_api_key_from_env(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Parse `"0-31"` → `(0, 31)`. Returns `None` on bad input.
    pub fn parse_range(s: &str) -> Option<(usize, usize)> {
        let mut parts = s.splitn(2, '-');
        let start: usize = parts.next()?.parse().ok()?;
        let end: usize = parts.next()?.parse().ok()?;
        if start <= end {
            Some((start, end))
        } else {
            None
        }
    }
}

// ── Unit manifest (fine-grained shard map) ───────────────────────────────────
//
// Mirrors the server's `--units PATH` JSON shape but augmented with `url`:
//
//   {
//     "shards": [
//       { "url": "grpc://hostA:9081",
//         "layer_experts": {"0": [[0,31]], "1": [[0,15]], "2": [[0,31]]} },
//       { "url": "grpc://hostB:9082",
//         "layer_experts": {"0": [[32,63]], "1": [[16,31],[64,79]]} }
//     ]
//   }
//
// One JSON object → many `ShardConfig`s.  Each shard has its own explicit
// `(layer, expert_id)` ownership set; the client routes per-(layer, expert)
// rather than per-expert.

/// Top-level JSON shape: a list of shards, each with its URL + per-layer
/// expert-range ownership.  Matches the server-side `--units` format
/// extended with `url` so a single manifest can describe the whole grid.
#[derive(serde::Deserialize)]
pub struct UnitManifest {
    pub shards: Vec<UnitShard>,
}

/// One shard's slice of the grid.
#[derive(serde::Deserialize)]
pub struct UnitShard {
    pub url: String,
    /// Per-layer list of inclusive `[start, end]` expert-id ranges.  Layers
    /// absent from the map are not owned by this shard.
    pub layer_experts: std::collections::BTreeMap<String, Vec<[usize; 2]>>,
}

impl UnitShard {
    /// Expand the per-layer ranges into a flat `(layer, expert_id)` set.
    pub fn into_unit_set(
        self,
    ) -> Result<std::collections::HashSet<(usize, usize)>, RemoteMoeError> {
        let mut units = std::collections::HashSet::new();
        for (layer_str, ranges) in self.layer_experts {
            let layer: usize = layer_str.parse().map_err(|_| {
                RemoteMoeError::Client(format!(
                    "unit-manifest: layer key '{layer_str}' is not a valid usize"
                ))
            })?;
            for [start, end] in ranges {
                if end < start {
                    return Err(RemoteMoeError::Client(format!(
                        "unit-manifest: layer {layer}: end ({end}) must be >= start ({start})"
                    )));
                }
                for eid in start..=end {
                    units.insert((layer, eid));
                }
            }
        }
        Ok(units)
    }
}

impl UnitManifest {
    /// Convert the parsed manifest into one `ShardConfig` per shard, each
    /// carrying its explicit `(layer, expert_id)` ownership set.
    pub fn into_shard_configs(self) -> Result<Vec<ShardConfig>, RemoteMoeError> {
        let mut out = Vec::with_capacity(self.shards.len());
        for shard in self.shards {
            let url = shard.url.clone();
            let units = shard.into_unit_set()?;
            out.push(ShardConfig::with_units(url, units));
        }
        Ok(out)
    }
}

/// Parse a unit-manifest JSON file from `path` into ready-to-connect
/// `ShardConfig`s.  Returns `RemoteMoeError::Client` on read or parse
/// failure with the path included so the operator can fix it without
/// grepping logs.
pub fn parse_unit_manifest(path: &std::path::Path) -> Result<Vec<ShardConfig>, RemoteMoeError> {
    let bytes = std::fs::read(path).map_err(|e| {
        RemoteMoeError::Client(format!("unit-manifest: read {}: {e}", path.display()))
    })?;
    let manifest: UnitManifest = serde_json::from_slice(&bytes).map_err(|e| {
        RemoteMoeError::Client(format!("unit-manifest: parse {}: {e}", path.display()))
    })?;
    manifest.into_shard_configs()
}
