//! Static shard map + binary-protocol header parsing.
//!
//! These are pure functions that live in the lib so they can be unit
//! tested. The actual dispatch (`AppState::resolve_all`, HTTP handlers)
//! stays in `main.rs` because it needs the live grid + reqwest client.

/// First word of a binary request body. When equal to `BATCH_MARKER`, the
/// header is `marker(4) + n(4) + n × layer_id(4)`. Otherwise it's a
/// single-layer request and the first word *is* the layer id.
pub const BATCH_MARKER: u32 = 0xFFFF_FFFF;

/// Static shard descriptor parsed from the `--shards` CLI flag. Each
/// shard owns the half-open layer range `[layer_start, layer_end)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    pub layer_start: usize, // inclusive
    pub layer_end: usize,   // exclusive
    pub url: String,
}

impl Shard {
    pub fn owns(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }
}

/// Parse a comma-separated shard map: `"START-END=URL,START-END=URL,..."`.
///
/// Ranges are *inclusive* in the flag (the historical user-facing
/// convention), so `0-15=...` becomes `layer_end = 16`. Whitespace is
/// tolerated; blank segments are skipped.
pub fn parse_shards(spec: &str) -> Result<Vec<Shard>, String> {
    let mut shards = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (range, url) = entry
            .split_once('=')
            .ok_or_else(|| format!("expected 'START-END=URL', got '{entry}'"))?;
        let (start_s, end_s) = range
            .split_once('-')
            .ok_or_else(|| format!("expected 'START-END', got '{range}'"))?;
        let start: usize = start_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid start '{start_s}'"))?;
        let end: usize = end_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid end '{end_s}'"))?;
        if end < start {
            return Err(format!("end ({end}) must be >= start ({start})"));
        }
        shards.push(Shard {
            layer_start: start,
            layer_end: end + 1,
            url: url.trim().to_string(),
        });
    }
    if shards.is_empty() {
        return Err("no shards specified".into());
    }
    Ok(shards)
}

/// Find the shard owning `layer` in a static shard list. Linear scan
/// because the list is small (typically 1-8 entries); avoids the
/// HashMap allocation that a route_table would need for one lookup.
pub fn find_shard_for_layer(shards: &[Shard], layer: usize) -> Option<&Shard> {
    shards.iter().find(|s| s.owns(layer))
}

/// Extract layer indices from a binary request body without parsing the
/// residual itself. Returns `None` if the header is malformed or
/// truncated; the caller falls back to a 400 response in that case.
pub fn peek_binary(body: &[u8]) -> Option<Vec<usize>> {
    if body.len() < 4 {
        return None;
    }
    let first = u32::from_le_bytes(body[0..4].try_into().ok()?);
    if first == BATCH_MARKER {
        if body.len() < 8 {
            return None;
        }
        let n = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
        let needed = 8 + n * 4;
        if body.len() < needed {
            return None;
        }
        let layers = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        Some(layers)
    } else {
        Some(vec![first as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shard / parse_shards ─────────────────────────────────────────────

    #[test]
    fn parse_shards_accepts_inclusive_range_flag() {
        let out = parse_shards("0-15=http://a:8080,16-31=http://b:8080").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].layer_start, 0);
        // Range is inclusive in the flag → exclusive layer_end is 16.
        assert_eq!(out[0].layer_end, 16);
        assert_eq!(out[0].url, "http://a:8080");
        assert_eq!(out[1].layer_start, 16);
        assert_eq!(out[1].layer_end, 32);
    }

    #[test]
    fn parse_shards_skips_blank_and_trims_whitespace() {
        let out = parse_shards("  0-3 = http://a:8080 , , 4-7 = http://b:8080 ").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].url, "http://a:8080");
    }

    #[test]
    fn parse_shards_rejects_missing_url() {
        let err = parse_shards("0-3").unwrap_err();
        assert!(err.contains("expected 'START-END=URL'"));
    }

    #[test]
    fn parse_shards_rejects_missing_dash() {
        let err = parse_shards("0=http://a").unwrap_err();
        assert!(err.contains("expected 'START-END'"));
    }

    #[test]
    fn parse_shards_rejects_non_numeric_endpoints() {
        assert!(parse_shards("a-3=http://a")
            .unwrap_err()
            .contains("invalid start"));
        assert!(parse_shards("0-z=http://a")
            .unwrap_err()
            .contains("invalid end"));
    }

    #[test]
    fn parse_shards_rejects_inverted_range() {
        let err = parse_shards("10-3=http://a").unwrap_err();
        assert!(err.contains("end (3) must be >= start (10)"));
    }

    #[test]
    fn parse_shards_rejects_empty_spec() {
        assert!(parse_shards("")
            .unwrap_err()
            .contains("no shards specified"));
        assert!(parse_shards(", ,")
            .unwrap_err()
            .contains("no shards specified"));
    }

    // ── Shard::owns ──────────────────────────────────────────────────────

    #[test]
    fn shard_owns_matches_half_open_range() {
        let s = Shard {
            layer_start: 5,
            layer_end: 10, // exclusive
            url: "x".into(),
        };
        assert!(!s.owns(4));
        assert!(s.owns(5));
        assert!(s.owns(9));
        assert!(!s.owns(10)); // exclusive
    }

    // ── find_shard_for_layer ─────────────────────────────────────────────

    #[test]
    fn find_shard_picks_owning_entry() {
        let shards = parse_shards("0-4=http://a,5-9=http://b").unwrap();
        assert_eq!(find_shard_for_layer(&shards, 0).unwrap().url, "http://a");
        assert_eq!(find_shard_for_layer(&shards, 4).unwrap().url, "http://a");
        assert_eq!(find_shard_for_layer(&shards, 5).unwrap().url, "http://b");
        assert_eq!(find_shard_for_layer(&shards, 9).unwrap().url, "http://b");
        assert!(find_shard_for_layer(&shards, 10).is_none());
    }

    #[test]
    fn find_shard_empty_list_returns_none() {
        assert!(find_shard_for_layer(&[], 0).is_none());
    }

    // ── peek_binary ──────────────────────────────────────────────────────

    fn le_u32(x: u32) -> [u8; 4] {
        x.to_le_bytes()
    }

    #[test]
    fn peek_binary_too_short_returns_none() {
        assert!(peek_binary(&[]).is_none());
        assert!(peek_binary(&[0, 0, 0]).is_none()); // < 4 bytes
    }

    #[test]
    fn peek_binary_single_layer_request() {
        // First word = layer id (not the batch marker).
        let body = le_u32(42).to_vec();
        assert_eq!(peek_binary(&body), Some(vec![42]));
    }

    #[test]
    fn peek_binary_batch_header_parses_layer_list() {
        let mut body = Vec::new();
        body.extend_from_slice(&le_u32(BATCH_MARKER));
        body.extend_from_slice(&le_u32(3));
        body.extend_from_slice(&le_u32(0));
        body.extend_from_slice(&le_u32(1));
        body.extend_from_slice(&le_u32(2));
        assert_eq!(peek_binary(&body), Some(vec![0, 1, 2]));
    }

    #[test]
    fn peek_binary_batch_header_truncated_count_returns_none() {
        // Marker present, but no n field.
        let body = le_u32(BATCH_MARKER).to_vec();
        assert!(peek_binary(&body).is_none());
    }

    #[test]
    fn peek_binary_batch_header_truncated_layers_returns_none() {
        let mut body = Vec::new();
        body.extend_from_slice(&le_u32(BATCH_MARKER));
        body.extend_from_slice(&le_u32(3)); // claims 3 layers
        body.extend_from_slice(&le_u32(0));
        // Missing layers 1 and 2.
        assert!(peek_binary(&body).is_none());
    }

    #[test]
    fn peek_binary_batch_zero_layers_returns_empty_vec() {
        let mut body = Vec::new();
        body.extend_from_slice(&le_u32(BATCH_MARKER));
        body.extend_from_slice(&le_u32(0));
        assert_eq!(peek_binary(&body), Some(vec![]));
    }
}
