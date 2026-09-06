//! The access realization on its own terms: Demand does nothing and says
//! so; Advise and Touch cover exactly the selected ranges, page-rounded,
//! and leave every byte as it was; a touched range is resident; the
//! policy is the budget's and lands on every mapped pin; and the three
//! policies execute the same token to the same bits.

use super::super::accounting::ResidencyBudget;
use super::super::decode::DecodeSession;
use super::super::kv::RowKvState;
use super::super::prefetch::{prefetch, PrefetchReport, Range};
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
        err.contains("prescient") && err.contains("demand, advise, touch"),
        "{err}"
    );
}
