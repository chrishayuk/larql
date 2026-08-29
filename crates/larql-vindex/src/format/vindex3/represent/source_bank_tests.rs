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
}
