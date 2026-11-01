//! A residency observation keeps every read's outcome. These pin, on
//! observations built by hand, what no fixture can produce on demand: a
//! region whose `mincore` failed. A failure is counted and never summed
//! as zero, an observation with one is never a whole-image total, and a
//! delta compares only regions read on both sides.

use std::path::PathBuf;
use std::time::SystemTime;

use super::super::prepared::{
    MappedResidency, ObservedStore, RegionResidency, ResidencyDelta, ResidencyObservation,
};
use crate::format::vindex3::represent::physical::ResidencyReadError;

const LEN: u64 = 1 << 20;
const MINCORE_FAILED: ResidencyReadError = ResidencyReadError::Mincore { errno: 12 };

fn region(
    scan_index: usize,
    offset: u64,
    resident: Result<u64, ResidencyReadError>,
) -> RegionResidency {
    RegionResidency {
        scan_index,
        store: 0,
        file_offset: offset,
        len: LEN,
        resident,
    }
}

fn observation(regions: Vec<RegionResidency>) -> ResidencyObservation {
    ResidencyObservation {
        started_wall: SystemTime::UNIX_EPOCH,
        started_mono_ns: 0,
        ended_wall: SystemTime::UNIX_EPOCH,
        ended_mono_ns: 1,
        page_size: Some(16_384),
        stores: vec![ObservedStore {
            id: "target.expert_bank".to_string(),
            source_path: Some(PathBuf::from("segments/target.expert_bank.bin")),
        }],
        regions,
    }
}

#[test]
fn a_failed_region_is_counted_and_never_summed_as_zero() {
    let obs = observation(vec![
        region(0, 0, Ok(100)),
        region(1, LEN, Err(MINCORE_FAILED)),
        region(2, 2 * LEN, Ok(250)),
    ]);
    let s = obs.summary();
    assert_eq!(s.regions, 3);
    assert_eq!(s.failed_regions, 1);
    assert_eq!(s.read_regions(), 2);
    assert_eq!(s.resident_bytes, 350, "only the read regions are summed");
    assert_eq!(s.mapped_bytes, 3 * LEN, "address space covers every region");
    assert_eq!(
        s.mapped_bytes_read,
        2 * LEN,
        "read address space excludes the failure"
    );
    let (first, error) = obs.first_failure().expect("one region failed");
    assert_eq!(first.scan_index, 1);
    assert_eq!(error, MINCORE_FAILED);
}

#[test]
fn an_incomplete_or_empty_observation_is_never_a_whole_image_total() {
    let failed = observation(vec![
        region(0, 0, Ok(0)),
        region(1, LEN, Err(MINCORE_FAILED)),
    ]);
    assert_eq!(failed.summary().complete_resident_bytes(), None);

    let nothing = observation(Vec::new());
    assert_eq!(nothing.summary().complete_resident_bytes(), None);
    assert_eq!(MappedResidency::default().complete_resident_bytes(), None);

    let all_read = observation(vec![region(0, 0, Ok(7)), region(1, LEN, Ok(0))]);
    assert_eq!(all_read.summary().complete_resident_bytes(), Some(7));
}

#[test]
fn every_failure_kind_names_itself_and_only_mincore_carries_an_errno() {
    for (error, kind, errno) in [
        (MINCORE_FAILED, "mincore", Some(12)),
        (
            ResidencyReadError::PageSizeUnavailable,
            "page_size_unavailable",
            None,
        ),
        (ResidencyReadError::Unsupported, "unsupported", None),
    ] {
        assert_eq!(error.kind(), kind);
        assert_eq!(error.errno(), errno);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn a_delta_is_signed_per_region_over_regions_read_on_both_sides() {
    let before = observation(vec![
        region(0, 0, Ok(100)),
        region(1, LEN, Ok(500)),
        region(2, 2 * LEN, Err(MINCORE_FAILED)),
        region(3, 3 * LEN, Ok(40)),
    ]);
    let after = observation(vec![
        region(0, 0, Ok(300)),
        region(1, LEN, Ok(200)),
        region(2, 2 * LEN, Ok(900)),
        region(3, 3 * LEN, Err(MINCORE_FAILED)),
    ]);
    let d = ResidencyDelta::between(&before, &after);
    assert_eq!(d.comparable, 2, "regions 0 and 1 were read on both sides");
    assert_eq!(d.increase, 200);
    assert_eq!(d.decrease, 300);
    assert_eq!(
        d.excluded, 2,
        "a failure on either side excludes the region"
    );
    assert_eq!(d.signed_total(), Some(-100));
}

#[test]
fn regions_present_on_one_side_only_are_excluded_not_compared() {
    let before = observation(vec![region(0, 0, Ok(10)), region(1, LEN, Ok(10))]);
    let after = observation(vec![region(0, 0, Ok(30)), region(1, 9 * LEN, Ok(10))]);
    let d = ResidencyDelta::between(&before, &after);
    assert_eq!(d.comparable, 1);
    assert_eq!(d.increase, 20);
    assert_eq!(d.excluded, 2, "one region only before, one only after");
}

#[test]
fn a_delta_with_no_comparable_region_is_unmeasured_not_zero() {
    let before = observation(vec![region(0, 0, Err(MINCORE_FAILED))]);
    let after = observation(vec![region(0, 0, Ok(64))]);
    let d = ResidencyDelta::between(&before, &after);
    assert_eq!(d.comparable, 0);
    assert_eq!(d.signed_total(), None);

    let empty = ResidencyDelta::between(&observation(Vec::new()), &observation(Vec::new()));
    assert_eq!(empty.signed_total(), None);
}

#[test]
fn a_region_is_matched_by_its_file_offset_and_length_not_its_scan_position() {
    let before = observation(vec![region(0, 0, Ok(1)), region(1, LEN, Ok(2))]);
    // The same two regions, scanned in the other order.
    let after = observation(vec![region(0, LEN, Ok(5)), region(1, 0, Ok(1))]);
    let d = ResidencyDelta::between(&before, &after);
    assert_eq!(d.comparable, 2);
    assert_eq!(d.increase, 3);
    assert_eq!(d.decrease, 0);
    assert_eq!(
        before.key_of(&before.regions[1]),
        ("segments/target.expert_bank.bin".to_string(), LEN, LEN)
    );
}

#[test]
fn an_owned_store_is_keyed_by_its_id() {
    let mut obs = observation(vec![region(0, 8, Ok(0))]);
    obs.stores[0].source_path = None;
    assert_eq!(
        obs.key_of(&obs.regions[0]),
        ("target.expert_bank".to_string(), 8, LEN)
    );
}
