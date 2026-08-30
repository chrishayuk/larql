//! Sync gate: the consumed-key registry must cover every string-literal key
//! the config parser reads.
//!
//! The registry in `config_keys.rs` is a hand-maintained mirror of
//! `detect/parser.rs`. Hand-maintained mirrors drift; this test scans the
//! parser source for the two access shapes it uses for JSON keys —
//! `["key"]` and `.get("key")` — and fails when an accessed key is missing
//! from the registry. A parser change that starts consuming a new key then
//! breaks this test until the registry (and so the inventory report) knows
//! about it.
//!
//! The scan is deliberately narrow: string literals in other positions
//! (comparisons like `== "gpt2"`, error text) do not match either shape, so
//! the test cannot demand registry entries for non-keys.

use crate::inventory::config_keys::{
    CONSUMED_CONTAINER_KEYS, CONSUMED_LEAF_KEYS, PATH_READ_CONTAINER_KEYS, PATH_READ_LEAF_KEYS,
    PRESENCE_ONLY_CONTAINER_KEYS,
};

/// The parser source this registry mirrors.
const PARSER_SOURCE: &str = include_str!("../../detect/parser.rs");

/// The alias constants live here.
const CONFIG_IO_SOURCE: &str = include_str!("../../detect/config_io.rs");

/// Extract string literals appearing as `["lit"]` or `.get("lit")`.
fn extract_key_literals(source: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for (i, _) in source.match_indices('"') {
        // Opening quote must directly follow `[` or `.get(`.
        let preceded = source[..i].trim_end_matches(char::is_whitespace);
        let is_key_position = preceded.ends_with('[') || preceded.ends_with(".get(");
        if !is_key_position {
            continue;
        }
        // Find the closing quote (no escapes appear in key names).
        let rest = &source[i + 1..];
        let Some(end) = rest.find('"') else { continue };
        let literal = &rest[..end];
        // Key names are identifier-like; skip anything else defensively.
        if !literal.is_empty()
            && literal
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            keys.push(literal.to_string());
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Keys the scan finds that are *not* config-leaf reads and need no registry
/// entry. Empty today; keep any future entry justified with a comment.
const SCAN_ALLOWLIST: &[&str] = &[];

#[test]
fn every_parser_key_access_is_in_the_registry() {
    let mut missing = Vec::new();
    for source in [PARSER_SOURCE, CONFIG_IO_SOURCE] {
        for key in extract_key_literals(source) {
            let known = CONSUMED_LEAF_KEYS.contains(&key.as_str())
                || CONSUMED_CONTAINER_KEYS.contains(&key.as_str())
                || PRESENCE_ONLY_CONTAINER_KEYS.contains(&key.as_str())
                || PATH_READ_CONTAINER_KEYS.contains(&key.as_str())
                || PATH_READ_LEAF_KEYS.contains(&key.as_str())
                || SCAN_ALLOWLIST.contains(&key.as_str());
            if !known {
                missing.push(key);
            }
        }
    }
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "detect/parser.rs (or config_io.rs) reads config keys the inventory \
         registry does not know about — add them to CONSUMED_LEAF_KEYS (or a \
         container list) in inventory/config_keys.rs so inspect-hf reports \
         stay honest: {missing:?}"
    );
}

/// The extractor actually finds the access shapes it claims to — guard the
/// guard against a refactor of the parser's access style silently emptying
/// the scan.
#[test]
fn extractor_finds_known_parser_accesses() {
    let keys = extract_key_literals(PARSER_SOURCE);
    for expected in ["head_dim", "num_key_value_heads", "rope_scaling", "factor"] {
        assert!(
            keys.contains(&expected.to_string()),
            "extractor lost sight of parser access `{expected}` — \
             its scan patterns no longer match the parser's access style"
        );
    }
    // And it must not pick up comparison literals.
    assert!(
        !keys.contains(&"gpt2".to_string()),
        "extractor started matching non-key string literals"
    );
}

/// Registry lists stay duplicate-free — a duplicate entry usually means a
/// mis-resolved merge.
#[test]
fn registry_lists_have_no_duplicates() {
    for list in [
        CONSUMED_LEAF_KEYS,
        CONSUMED_CONTAINER_KEYS,
        PRESENCE_ONLY_CONTAINER_KEYS,
    ] {
        let mut seen = std::collections::HashSet::new();
        for key in list {
            assert!(seen.insert(key), "duplicate registry entry: {key}");
        }
    }
}
