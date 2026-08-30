//! `tests` for [`super`], including one against the REAL container.

use super::*;

const PER: u64 = 4_718_592;
const CONTAINER_ENV: &str = "LARQL_KIMI_VINDEX3";

fn synthetic_store(bytes: usize) -> Arc<PhysicalStore> {
    Arc::new(PhysicalStore::owned(
        "source",
        vec![7u8; bytes],
        BTreeMap::new(),
    ))
}

/// Deliberately NON-MONOTONIC placement, so a table that secretly
/// assumed expert order would be wrong.
fn shuffled_offsets(experts: u32, per: u64) -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    for e in 0..experts {
        // A base that has nothing to do with `e * stride`.
        let base = u64::from((e * 97 + 13) % experts) * 3 * per;
        for (i, proj) in ["w1", "w2", "w3"].iter().enumerate() {
            m.insert(
                format!("1.block_sparse_moe.experts.{e}.{proj}.weight"),
                base + i as u64 * per,
            );
        }
    }
    m
}

/// The table carries the segment's own offsets, never anything derived
/// from an expert id.
#[test]
fn the_base_table_follows_the_segment_not_the_expert_order() {
    let experts = 16u32;
    let offsets = shuffled_offsets(experts, PER);
    let store = synthetic_store((u64::from(experts) * 3 * PER) as usize);
    let bank = source_expert_bank(&store, &offsets, 1, experts, PER).expect("addresses");

    for e in 0..experts {
        let want = offsets[&format!("1.block_sparse_moe.experts.{e}.w1.weight")];
        assert_eq!(
            u64::from(bank.bases[e as usize]),
            want,
            "expert {e}'s base must come from the segment header"
        );
    }
    // And it really is out of order, or the test proves nothing.
    assert!(
        bank.bases.windows(2).any(|w| w[0] > w[1]),
        "the fixture must be non-monotonic for this to mean anything"
    );
}

/// A source whose projections are NOT contiguous within an expert
/// cannot be addressed by one base table, and must say so rather than
/// silently reading a neighbouring projection.
#[test]
fn a_non_contiguous_source_is_refused() {
    let mut offsets = shuffled_offsets(4, PER);
    offsets.insert("1.block_sparse_moe.experts.2.w3.weight".into(), 999);
    let store = synthetic_store((4 * 3 * PER) as usize);
    let err = match source_expert_bank(&store, &offsets, 1, 4, PER) {
        Err(e) => e,
        Ok(_) => panic!("a non-contiguous source must be refused, not addressed"),
    };
    assert!(format!("{err}").contains("not contiguous"), "{err}");
}

/// **Against the real container.** The measured layout property — the
/// one that made a 94 GB rewrite unnecessary — becomes a regression
/// rather than a one-off observation.
///
/// Checked at deliberately non-monotonic ids, including the ones that
/// exposed the arbitrary ordering in the first place.
#[test]
fn the_real_source_container_is_addressable_at_non_monotonic_experts() {
    let Some(container) = std::env::var_os(CONTAINER_ENV).map(std::path::PathBuf::from) else {
        eprintln!("skipped: set {CONTAINER_ENV} to the source .vindex3");
        return;
    };
    let path = container.join("segments").join("target.expert_bank.bin");
    let (header, _) =
        crate::format::vindex3::encode::segment::read_segment_header(&path).expect("header");
    let offsets: BTreeMap<String, u64> = header
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.offset))
        .collect();
    let store = Arc::new(PhysicalStore::map_segment("source", &path).expect("mmap"));

    let bank = source_expert_bank(&store, &offsets, 1, 256, PER).expect("layer 1 addresses");
    for e in [7u32, 137, 255] {
        let want = offsets[&format!("1.block_sparse_moe.experts.{e}.w1.weight")];
        assert_eq!(
            u64::from(bank.bases[e as usize]),
            want,
            "expert {e} must be addressed where the segment actually put it"
        );
        // The property being defended: identity-derived addressing WOULD
        // be wrong here.
        assert_ne!(
            want,
            u64::from(e) * 3 * PER,
            "expert {e} happens to sit where identity would predict, so this id no longer \
             exercises the arbitrary-ordering case — pick another"
        );
    }
    eprintln!(
        "[source] layer 1: 256 experts addressed from the segment's own table; \
         expert 7 at {}, identity would have said {}",
        bank.bases[7],
        7 * 3 * PER
    );
    assert!(
        bank.bases.windows(2).any(|w| w[0] > w[1]),
        "the real source is out of expert order — that is the whole point"
    );

    // A DEEP layer: its block starts tens of gigabytes into the
    // segment, past what a 32-bit absolute offset can address, so the
    // table must be relative to the layer's own base.
    let deep = source_expert_bank(&store, &offsets, 26, 256, PER).expect("layer 26 addresses");
    assert!(
        deep.layer_base > u64::from(u32::MAX),
        "layer 26 must start past the 32-bit horizon for this arm to mean anything \
         (it starts at {})",
        deep.layer_base
    );
    for e in [7u32, 137, 255] {
        let want = offsets[&format!("26.block_sparse_moe.experts.{e}.w1.weight")];
        assert_eq!(
            u64::from(deep.bases[e as usize]) + deep.layer_base,
            want,
            "expert {e}'s rebased offset plus the layer base must be the segment's own"
        );
    }
    eprintln!(
        "[source] layer 26: base {} ({}x the 32-bit horizon), table rebased and 32-bit clean",
        deep.layer_base,
        deep.layer_base / u64::from(u32::MAX)
    );
}

/// A layer placed past the 32-bit horizon of a much larger segment is
/// rebased to its own block: the table stays 32-bit while the segment
/// does not.
///
/// The backing is a SPARSE file — a real segment header over a payload
/// the filesystem never materialises — because an owned fixture would
/// need a >4 GiB allocation to place a byte at a >4 GiB offset.
#[test]
fn a_layer_past_the_32_bit_horizon_is_rebased_to_its_own_block() {
    use crate::format::vindex3::encode::segment::SegmentHeader;
    use std::io::Write;

    let per: u64 = 64;
    let experts = 4u32;
    let deep = u64::from(u32::MAX) + 4096;

    let mut offsets = BTreeMap::new();
    for e in 0..experts {
        // Non-monotonic within the block, same as the real source.
        let base = deep + u64::from((e * 3 + 1) % experts) * 3 * per;
        for (i, proj) in ["w1", "w2", "w3"].iter().enumerate() {
            offsets.insert(
                format!("1.block_sparse_moe.experts.{e}.{proj}.weight"),
                base + i as u64 * per,
            );
        }
    }

    let header = SegmentHeader {
        schema: crate::format::vindex3::encode::segment::SEGMENT_HEADER_SCHEMA,
        representation: "test@BF16".to_string(),
        tensors: Vec::new(),
    };
    let header_json = serde_json::to_vec(&header).expect("header serialises");
    let dir = std::env::temp_dir().join(format!("larql-sparse-bank-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("deep.bin");
    {
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("len");
        f.write_all(&header_json).expect("header");
        let payload_end = header_json.len() as u64 + 8 + deep + u64::from(experts) * 3 * per;
        f.set_len(payload_end).expect("sparse payload");
    }
    let store = Arc::new(PhysicalStore::map_segment("deep", &path).expect("mmap sparse"));

    let bank = source_expert_bank(&store, &offsets, 1, experts, per)
        .expect("a deep layer must rebase, not overflow");
    assert_eq!(bank.layer_base, deep, "the base is the block's own start");
    for e in 0..experts {
        let want = offsets[&format!("1.block_sparse_moe.experts.{e}.w1.weight")];
        assert_eq!(u64::from(bank.bases[e as usize]) + bank.layer_base, want);
        assert!(u64::from(bank.bases[e as usize]) < u64::from(experts) * 3 * per);
    }
    assert_eq!(bank.layer_len, u64::from(experts) * 3 * per);
    // The views really do open at the deep offset.
    assert_eq!(
        bank.binding.gate.region.region.len(),
        bank.layer_len,
        "the gate view covers exactly the layer's block"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A bank over zero experts addresses nothing and says so, rather than
/// building an empty table whose `min`/`max` would panic.
#[test]
fn a_zero_expert_bank_is_refused() {
    let store = synthetic_store(1024);
    let err = match source_expert_bank(&store, &BTreeMap::new(), 1, 0, PER) {
        Err(e) => e,
        Ok(_) => panic!("zero experts must be refused"),
    };
    assert!(format!("{err}").contains("addresses nothing"), "{err}");
}

/// A tensor the segment does not hold is named, for each projection —
/// `w1` decides the base, and `w2`/`w3` are what the contiguity check
/// reads.
#[test]
fn a_missing_projection_is_named() {
    let store = synthetic_store((4 * 3 * PER) as usize);
    for missing in ["w1", "w2", "w3"] {
        let mut offsets = shuffled_offsets(4, PER);
        offsets.remove(&format!("1.block_sparse_moe.experts.2.{missing}.weight"));
        let err = match source_expert_bank(&store, &offsets, 1, 4, PER) {
            Err(e) => e,
            Ok(_) => panic!("a missing projection must be refused"),
        };
        let text = format!("{err}");
        assert!(text.contains(missing) && text.contains("has no"), "{text}");
    }
}

/// A segment too short for the layer's own block is refused by name
/// rather than handing out a view past the mapping.
#[test]
fn a_segment_too_short_for_the_layer_is_refused() {
    // Offsets describe four experts, the store holds one expert's worth.
    let offsets = shuffled_offsets(4, PER);
    let store = synthetic_store(PER as usize);
    let err = match source_expert_bank(&store, &offsets, 1, 4, PER) {
        Err(e) => e,
        Ok(_) => panic!("a short segment must be refused"),
    };
    assert!(format!("{err}").contains("too short"), "{err}");
}

/// The layer's extent is reported, and it is the block the table is
/// relative to — the span a caller registers with a backend.
#[test]
fn the_layer_extent_covers_exactly_the_experts_block() {
    let experts = 8u32;
    let offsets = shuffled_offsets(experts, PER);
    let store = synthetic_store((u64::from(experts) * 3 * PER) as usize);
    let bank = source_expert_bank(&store, &offsets, 1, experts, PER).expect("addresses");
    assert_eq!(bank.layer_base, 0, "this fixture places layer 1 at zero");
    assert_eq!(bank.layer_len, u64::from(experts) * 3 * PER);
    assert_eq!(bank.binding.gate.region.region.len(), bank.layer_len);
    assert_eq!(bank.binding.down.region.region.len(), bank.layer_len - PER);
    assert_eq!(
        bank.binding.up.region.region.len(),
        bank.layer_len - 2 * PER
    );
    assert!(bank.binding.shared.is_none(), "routed-only by construction");
    assert_eq!(bank.binding.gate.extent, ExtentPolicy::ContainingView);
    assert_eq!(
        bank.binding.gate.addressing.experts() as usize,
        experts as usize
    );
}

/// **A layer whose experts span more than 32 bits is refused**, not
/// silently truncated.
///
/// Rebasing to the layer's own block is what keeps the table 32-bit on
/// a 94 GB segment, but it only works while ONE layer's experts fit in
/// that range. A source that spread a single layer wider would wrap the
/// table and serve a different expert's bytes, so it is named instead.
#[test]
fn a_layer_wider_than_a_32_bit_table_is_refused() {
    use crate::format::vindex3::encode::segment::SegmentHeader;
    use std::io::Write;

    let per: u64 = 64;
    // Two experts, the second placed past the 32-bit horizon FROM THE
    // FIRST — so the block itself, not the segment, is too wide.
    let far = u64::from(u32::MAX) + 8192;
    let mut offsets = BTreeMap::new();
    for (e, base) in [(0u32, 0u64), (1, far)] {
        for (i, proj) in ["w1", "w2", "w3"].iter().enumerate() {
            offsets.insert(
                format!("1.block_sparse_moe.experts.{e}.{proj}.weight"),
                base + i as u64 * per,
            );
        }
    }
    let header = SegmentHeader {
        schema: crate::format::vindex3::encode::segment::SEGMENT_HEADER_SCHEMA,
        representation: "test@BF16".to_string(),
        tensors: Vec::new(),
    };
    let header_json = serde_json::to_vec(&header).expect("header");
    let dir = std::env::temp_dir().join(format!("larql-wide-bank-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("wide.bin");
    {
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .expect("len");
        f.write_all(&header_json).expect("header");
        f.set_len(header_json.len() as u64 + 8 + far + 3 * per)
            .expect("sparse payload");
    }
    let store = Arc::new(PhysicalStore::map_segment("wide", &path).expect("mmap"));
    let err = match source_expert_bank(&store, &offsets, 1, 2, per) {
        Err(e) => e,
        Ok(_) => panic!("a layer wider than a 32-bit table must be refused"),
    };
    std::fs::remove_dir_all(&dir).ok();
    let text = format!("{err}");
    assert!(
        text.contains("32-bit") && text.contains("past the layer's own base"),
        "{text}"
    );
}
