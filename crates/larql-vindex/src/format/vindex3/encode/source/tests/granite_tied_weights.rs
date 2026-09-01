//! **The shard index is not an authority for physical payload bytes.**
//!
//! Real metadata from `hf://ibm-granite/granite-4.2-3b`, header-only —
//! see `fixtures/granite_4_2_3b/PROVENANCE.md`. It is here because it
//! disagrees with itself in a way that will recur, and because the
//! disagreement is invisible to any test that only checks arithmetic:
//!
//! ```text
//! metadata.total_size                    6,805,672,960   logical, deduplicated
//! sum of header data_offsets               7,319,475,200   physical, serialised
//! ```
//!
//! `lm_head.weight` and `model.embed_tokens.weight` were tied in the source
//! model, so HF counts them once and the file writes both. For a range
//! source the physical spans are the only addressable thing, so the
//! headers are the authority — and the sibling test with a synthetic short
//! index can prove the reconciliation logic but never that the phenomenon
//! is upstream rather than imagined.
//!
//! The invariant this replaced — `inventory.total_bytes ==
//! index.metadata.total_size` — passed on every model anyone had tried
//! until this one, which is what made it dangerous.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::format::huggingface::metadata_checkpoint::StagedCheckpoint;
use crate::format::vindex3::encode::source::{index_staged_shards, staged_payload_bytes};

/// What the checked-in index declares.
const DECLARED_TOTAL: u64 = 6_805_672_960;

/// What the checked-in headers actually sum to.
const PHYSICAL_TOTAL: u64 = 7_319_475_200;

/// One `[100352, 2560]` BF16 embedding — the tied member counted once and
/// written twice.
const TIED_MEMBER_BYTES: u64 = 513_802_240;

/// Tensors the index's `weight_map` names.
const TENSOR_COUNT: usize = 363;

const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/format/vindex3/encode/source/tests/fixtures/granite_4_2_3b")
}

/// The fixture, shaped as staging would have left it.
fn staged() -> StagedCheckpoint {
    let dir = fixture_dir();
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    StagedCheckpoint {
        dir,
        commit: Some("b7e947307dd2efb3ad3b853b0e8a7e75f8ad4ac2".to_string()),
        shards: SHARDS.iter().map(|s| s.to_string()).collect(),
        metadata: vec!["config.json".to_string()],
        stub_bytes: 0,
        metadata_bytes: 0,
        declared_total_size: index["metadata"]["total_size"].as_u64(),
    }
}

#[test]
fn the_fixture_carries_headers_and_no_payload() {
    // If a future edit ever replaced these stubs with real shards, every
    // assertion below would still pass while the repo grew by 7 GB.
    for shard in SHARDS {
        let path = fixture_dir().join(shard);
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(
            len < 100_000,
            "{shard} is {len} B — a header stub should be tens of KB, \
             this looks like a real shard"
        );
    }
}

#[test]
fn the_index_under_declares_its_own_payload() {
    let staged = staged();
    assert_eq!(staged.declared_total_size, Some(DECLARED_TOTAL));
    assert_eq!(staged_payload_bytes(&staged).unwrap(), PHYSICAL_TOTAL);
    assert_eq!(
        PHYSICAL_TOTAL - DECLARED_TOTAL,
        TIED_MEMBER_BYTES,
        "the gap should be exactly one tied member, not an arbitrary drift"
    );
}

#[test]
fn the_tied_pair_occupies_two_distinct_regions() {
    // The mechanism, not just the number. If these two names aliased one
    // region, the index would be right and the header sum would be the
    // one double-counting — the opposite conclusion. They do not alias:
    // they are adjacent, non-overlapping, and equal in size.
    let locations = index_staged_shards(&staged()).unwrap();
    assert_eq!(locations.len(), TENSOR_COUNT);

    let head = &locations["lm_head.weight"];
    let embed = &locations["model.embed_tokens.weight"];
    assert_eq!(head.shard, embed.shard, "the tied pair shares a shard");
    assert_eq!(head.len, TIED_MEMBER_BYTES);
    assert_eq!(embed.len, TIED_MEMBER_BYTES);
    assert_ne!(
        head.offset, embed.offset,
        "distinct offsets are what make this two serialised regions \
         rather than one aliased tensor"
    );
    let (first, second) = if head.offset < embed.offset {
        (head, embed)
    } else {
        (embed, head)
    };
    assert_eq!(
        first.offset + first.len,
        second.offset,
        "the pair should be adjacent and non-overlapping"
    );
}

#[test]
fn no_two_tensors_overlap() {
    // The general form of the check above, and the property that makes a
    // header sum meaningful at all: if spans overlapped, summing them
    // would over-count and the index might be the better authority.
    let locations = index_staged_shards(&staged()).unwrap();
    let mut by_shard: BTreeMap<&Path, Vec<(u64, u64)>> = BTreeMap::new();
    for location in locations.values() {
        by_shard
            .entry(location.shard.as_path())
            .or_default()
            .push((location.offset, location.len));
    }
    for (shard, mut spans) in by_shard {
        spans.sort_unstable();
        for pair in spans.windows(2) {
            let (start, len) = pair[0];
            let (next, _) = pair[1];
            assert!(
                start + len <= next,
                "{}: span at {start}..+{len} overlaps the one at {next}",
                shard.display()
            );
        }
    }
}
