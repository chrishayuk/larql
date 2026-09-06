//! The access realization on its own terms: Demand does nothing and says
//! so; Advise and Touch cover exactly the selected ranges, page-rounded,
//! and leave every byte as it was; a touched range is resident; a gate
//! asks only for what the OS does not already hold; the policy is the
//! budget's and lands on every mapped pin; and every policy executes the
//! same token to the same bits.

use super::super::accounting::ResidencyBudget;
use super::super::decode::DecodeSession;
use super::super::kv::RowKvState;
use super::super::prefetch::{
    prefetch, probe_indices, residency, runs_of, PrefetchReport, Range, GATE_PROBE_PAGES,
};
use super::super::prepared::{select_realizations_within, ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::realization::{MappedAccess, RealizationForm};
use super::kimi_per_expert_prepared::Subject;
use crate::format::vindex3::fixtures_kimi::kimi_per_expert_moe_f32_model;

const PROMPT: [u32; 4] = [3, 17, 28, 11];

/// A file of `len` bytes with a recognisable pattern, mapped whole. The
/// prefetch works on address ranges of a mapping, so its witness maps a
/// plain file and asks the OS about residency the way a region does.
fn mapped_file(len: usize) -> (tempfile::TempDir, memmap2::Mmap) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bank.bin");
    let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    // SAFETY: the file is private to this test and never written again
    // while the mapping lives.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    (dir, mmap)
}

/// An anonymous mapping of `len` bytes: every page is zero-fill-on-demand
/// and NOT resident until something touches it, which is how a test holds
/// a cold page beside a warm one without evicting the machine. (A file
/// this process just wrote is in the page cache and therefore warm.)
fn anonymous(len: usize) -> memmap2::MmapMut {
    memmap2::MmapOptions::new().len(len).map_anon().unwrap()
}

/// Bytes of `range` the OS reports resident, by page.
#[cfg(unix)]
fn resident_bytes(range: Range) -> usize {
    // SAFETY: sysconf reads a constant; mincore is given a page-aligned
    // range inside a live mapping and a vector of one byte per page.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let start = range.address / page * page;
    let end = (range.address + range.bytes).div_ceil(page) * page;
    let pages = (end - start) / page;
    let mut vec = vec![0u8; pages];
    let rc = unsafe {
        libc::mincore(
            start as *mut libc::c_void,
            end - start,
            // The vector's element type is the platform's: `c_char` on
            // macOS, `c_uchar` on Linux. `cast` lets each say which.
            vec.as_mut_ptr().cast(),
        )
    };
    assert_eq!(rc, 0, "mincore failed");
    vec.iter().filter(|b| **b & 1 == 1).count() * page
}

#[test]
fn demand_brings_nothing_in_and_reports_nothing() {
    let (_dir, mapped) = mapped_file(64 * 1024);
    let range = Range::of(mapped[..].as_ref());
    assert_eq!(
        prefetch(MappedAccess::Demand, &[range], 4),
        PrefetchReport::default()
    );
    assert_eq!(
        prefetch(MappedAccess::Touch, &[], 4),
        PrefetchReport::default()
    );
}

#[test]
fn advise_and_touch_cover_the_ranges_page_rounded_and_change_no_byte() {
    let len = 200 * 1024 + 123;
    let (_dir, mapped) = mapped_file(len);
    let before: Vec<u8> = mapped[..].as_ref().to_vec();
    // Two ranges inside the mapping, deliberately unaligned and out of order.
    let a = Range::of(&mapped[..].as_ref()[70_000..150_000]);
    let b = Range::of(&mapped[..].as_ref()[1_000..20_000]);
    for access in [MappedAccess::Advise, MappedAccess::Touch] {
        let report = prefetch(access, &[a, b], 3);
        assert_eq!(report.ranges, 2, "{access:?}");
        assert!(
            report.bytes >= (150_000 - 70_000) + (20_000 - 1_000),
            "{access:?}: page rounding never shrinks a range ({})",
            report.bytes
        );
        assert!(
            report.bytes <= (150_000 - 70_000) + (20_000 - 1_000) + 4 * 64 * 1024,
            "{access:?}: rounding is bounded by a page at each end ({})",
            report.bytes
        );
        assert_eq!(mapped[..].as_ref(), &before[..], "{access:?} changed bytes");
    }
    // Touched pages are resident, whatever the parallelism.
    for parallelism in [1, 4, 64] {
        let whole = Range::of(mapped[..].as_ref());
        let report = prefetch(MappedAccess::Touch, &[whole], parallelism);
        assert_eq!(report.ranges, 1);
        #[cfg(unix)]
        assert!(
            resident_bytes(whole) >= len,
            "parallelism {parallelism}: {} of {len} resident",
            resident_bytes(whole)
        );
    }
}

#[test]
fn the_budget_stamps_its_access_on_every_mapped_pin_and_on_nothing_else() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let budget = ResidencyBudget::UNBOUNDED.with_expert_access(MappedAccess::Touch);
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &budget,
    )
    .unwrap();
    let mut mapped = 0;
    for record in &records {
        match record.selection.realization.form {
            RealizationForm::MappedStored { access, .. } => {
                assert_eq!(access, MappedAccess::Touch, "{record:?}");
                mapped += 1;
            }
            _ => assert_eq!(record.selection.realization.access(), MappedAccess::Demand),
        }
    }
    assert!(mapped > 0, "the per-expert bank pins as mapped");
    let named = records
        .iter()
        .find(|r| {
            matches!(
                r.selection.realization.form,
                RealizationForm::MappedStored { .. }
            )
        })
        .unwrap();
    assert!(
        named.selection.realization.name().contains("touch"),
        "the realization names its access: {}",
        named.selection.realization.name()
    );
}

/// The same token under every access policy, to the bit: the policy
/// changes when pages arrive, never what they hold.
#[test]
fn every_access_policy_executes_the_same_token_to_the_same_bits() {
    let subject = Subject::build(kimi_per_expert_moe_f32_model);
    let (plan, store) = subject.open();
    let backend = ProductionBackend::new();
    let mut logits: Vec<Vec<f32>> = Vec::new();
    for access in MappedAccess::ALL {
        let budget = ResidencyBudget::UNBOUNDED.with_expert_access(access);
        let ops =
            PreparedOperands::load_within(&plan, &store, &backend, ExecutionSlice::Full, &budget)
                .unwrap();
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
        let mut last = None;
        for &token in &PROMPT {
            last = session.step(token).unwrap().logits;
        }
        logits.push(last.expect("head"));
    }
    for (i, other) in logits.iter().enumerate().skip(1) {
        assert_eq!(
            &logits[0],
            other,
            "{:?} differs from demand",
            MappedAccess::ALL[i]
        );
    }
}

#[test]
fn an_access_policy_is_named_or_refused_by_name() {
    for access in MappedAccess::ALL {
        assert_eq!(MappedAccess::parse(access.name()).unwrap(), access);
    }
    let err = MappedAccess::parse("prescient").unwrap_err();
    assert!(
        err.contains("prescient") && err.contains("demand, advise, touch, coalesced, gated"),
        "{err}"
    );
}

/// Adjacent tensors become one run and one request; a gap keeps them
/// apart. The runs are what every arm's residency is read over.
#[test]
fn coalescing_merges_adjacent_ranges_into_fewer_requests() {
    let (_dir, mapped) = mapped_file(1024 * 1024);
    let bytes: &[u8] = mapped[..].as_ref();
    let first = Range::of(&bytes[0..300_000]);
    let second = Range::of(&bytes[300_000..600_000]);
    let far = Range::of(&bytes[900_000..1_000_000]);
    let runs = runs_of(&[far, second, first]);
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert!(
        runs[0].address < runs[1].address,
        "runs are in address order"
    );
    assert!(runs[0].bytes >= 600_000 && runs[0].bytes < 600_000 + 2 * 64 * 1024);
    let coalesced = prefetch(MappedAccess::Coalesced, &[far, second, first], 2);
    assert_eq!((coalesced.ranges, coalesced.requests), (3, 2));
    let advised = prefetch(MappedAccess::Advise, &[far, second, first], 2);
    assert_eq!((advised.ranges, advised.requests), (3, 3));
    assert_eq!(coalesced.bytes, advised.bytes, "the same pages either way");
    // Neither policy withholds a request, so each asked for everything it
    // covers — the reading a gate is about to change. Coalescing asks for
    // the UNION of the page-aligned ranges, which is smaller than their
    // sum exactly when two of them round onto a shared boundary page.
    assert_eq!(advised.requested_bytes, advised.bytes);
    assert_eq!(
        coalesced.requested_bytes,
        runs.iter().map(|r| r.bytes).sum::<usize>()
    );
    assert!(
        coalesced.requested_bytes < coalesced.bytes,
        "first and second share the page they round onto: {} vs {}",
        coalesced.requested_bytes,
        coalesced.bytes
    );
}

/// The gate's sample: the ends of a run and evenly spaced pages between
/// them, bounded however long the run is, and every page of a run shorter
/// than the sample. A hole anywhere else is what the sample can miss, and
/// naming the rule here is what makes that limit legible.
#[test]
fn the_gate_probes_the_ends_and_the_span_between() {
    assert!(probe_indices(0).is_empty());
    assert_eq!(probe_indices(1), vec![0]);
    assert_eq!(
        probe_indices(GATE_PROBE_PAGES),
        (0..GATE_PROBE_PAGES).collect::<Vec<_>>(),
        "shorter than the sample: every page"
    );
    let wide = probe_indices(1_000);
    assert_eq!(wide.len(), GATE_PROBE_PAGES);
    assert_eq!(wide.first(), Some(&0), "the first page");
    assert_eq!(wide.last(), Some(&999), "the last page");
    assert!(
        wide.windows(2).all(|w| w[0] < w[1]),
        "distinct and in order: {wide:?}"
    );
}

/// The point of the gate: a warm run costs no request, a cold one is
/// asked for, and the coverage is the same either way.
#[test]
fn a_gate_asks_only_for_the_runs_the_os_does_not_already_hold() {
    // Wider than any page size this runs on, so each region is many pages.
    const REGION: usize = 512 * 1024;
    let map = anonymous(8 * REGION);
    let bytes: &[u8] = &map[..];
    let warm = Range::of(&bytes[0..REGION]);
    // Four regions of untouched space away, so the two never coalesce.
    let cold = Range::of(&bytes[5 * REGION..6 * REGION]);
    prefetch(MappedAccess::Touch, &[warm], 1);

    let gated = prefetch(MappedAccess::Gated, &[warm, cold], 1);
    assert_eq!(gated.ranges, 2, "the gate covers what it was given");
    assert_eq!(gated.bytes, gated.ranges * REGION, "coverage is unchanged");
    #[cfg(unix)]
    {
        assert_eq!(gated.requests, 1, "the warm run is not asked for");
        assert_eq!(
            gated.requested_bytes, REGION,
            "it asked for the cold run only"
        );
        // Everything resident now: a gate on a warm selection is silent.
        prefetch(MappedAccess::Touch, &[cold], 1);
        let silent = prefetch(MappedAccess::Gated, &[warm, cold], 1);
        assert_eq!(
            (silent.ranges, silent.requests, silent.requested_bytes),
            (2, 0, 0),
            "a warm token issues no request"
        );
    }
    // Where the OS will not say what is resident, the gate asks for
    // everything — a reading it could not take is never evidence.
    #[cfg(not(unix))]
    assert_eq!(gated.requests, 2);

    // The unconditional arm asks for both runs whatever their residency.
    let coalesced = prefetch(MappedAccess::Coalesced, &[warm, cold], 1);
    assert_eq!(coalesced.requests, 2);
    assert_eq!(coalesced.requested_bytes, coalesced.bytes);
}

/// A run whose ends are resident but whose middle is not is still asked
/// for: the sample reaches inside, and residency at the edges is not
/// residency.
#[cfg(unix)]
#[test]
fn a_run_resident_only_at_its_ends_is_still_asked_for() {
    const REGION: usize = 512 * 1024;
    // SAFETY: sysconf reads a process-independent constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let map = anonymous(REGION);
    let bytes: &[u8] = &map[..];
    let run = Range::of(bytes);
    prefetch(MappedAccess::Touch, &[Range::of(&bytes[0..page])], 1);
    prefetch(
        MappedAccess::Touch,
        &[Range::of(&bytes[REGION - page..REGION])],
        1,
    );
    let gated = prefetch(MappedAccess::Gated, &[run], 1);
    assert_eq!(
        (gated.requests, gated.requested_bytes),
        (1, gated.bytes),
        "an interior probe found the hole"
    );
}

/// After a touch, the residency reader finds every page of the runs.
#[test]
fn residency_reads_the_runs_after_a_touch() {
    let len = 512 * 1024;
    let (_dir, mapped) = mapped_file(len);
    let bytes: &[u8] = mapped[..].as_ref();
    let ranges = [
        Range::of(&bytes[10..200_000]),
        Range::of(&bytes[300_000..len]),
    ];
    prefetch(MappedAccess::Touch, &ranges, 3);
    #[cfg(unix)]
    {
        let seen = residency(&ranges).expect("unix can read residency");
        assert!(seen.span_bytes >= 200_000 - 10 + len - 300_000);
        assert_eq!(seen.resident_bytes, seen.span_bytes, "{seen:?}");
    }
    #[cfg(not(unix))]
    assert!(residency(&ranges).is_none());
}
