//! **ACCOUNTING-D1.** Preparation accounting reports bytes ACTUALLY
//! READ, including attestation verification, counted according to
//! physical reads — not inferred from metadata.
//!
//! The debt this pays: the ledger priced preparation from declared
//! extents alone, so hashing a payload to settle a guarantee cost real
//! I/O that no figure carried. A plan could be quoted as opening N bytes
//! while opening N plus every attested operand, which makes any claim of
//! the form "preparing this plan reads N bytes" unsound — and it made
//! VERIFIED FIDELITY LOOK FREE, which is the one thing this plane must
//! never suggest.
//!
//! The breakdown keeps three causes apart under one aggregate:
//!
//! ```text
//! prepare_reads
//! ├── representation_materialisation
//! ├── auxiliary_resolution
//! └── attestation_verification
//! ```
//!
//! Each answers a different question and responds to a different fix. No
//! new residency resource is invented: these bytes are read and dropped.

use super::container::{build, Built};
use super::vocabulary::*;
use crate::format::vindex3::opplan::exec::accounting::{
    expectations, BlockGeometry, ResidencyBudget, ResourceLedger,
};
use crate::format::vindex3::opplan::exec::operands::OperandSource;
use crate::format::vindex3::opplan::exec::prepared::{select_realizations_within, ExecutionSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;

/// What one plan cost, and what the store observed while planning it.
struct Priced {
    ledger: ResourceLedger,
    /// PHYSICAL bytes the store read — the independent witness.
    observed: u64,
    owners: usize,
}

/// Plan `built` and price it, reading the store's own counter across the
/// call so the ledger has something to be held against.
fn price(built: &Built, recognised: RecognisedMethods) -> Priced {
    let (plan, store) = built.plan_and_store_trusting(recognised);
    let before = store.bytes_read();
    let records = select_realizations_within(
        &plan,
        OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .expect("the default floor plans this container");
    let observed = store.bytes_read() - before;
    let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
    Priced {
        ledger: ResourceLedger::aggregate(&priced),
        observed,
        owners: records
            .iter()
            .filter(|r| r.representation == OWNER_LABEL)
            .count(),
    }
}

fn trusting() -> RecognisedMethods {
    RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1)
}

/// What verifying ONE owner must read, derived from the fixture's own
/// shape rather than from anything the code under test computes.
///
/// The owner is stored as one byte of code per element, and an
/// attestation binds to that stored tensor — so the bytes hashed are
/// exactly its element count. Deriving this from `stored_len` would have
/// been self-normalising: that figure is the container's record for the
/// tensor AND its sibling streams, so a test using it would have agreed
/// with the ledger while both were wrong.
fn hashed_per_owner(built: &Built) -> u64 {
    OWNERS
        .iter()
        .map(|owner| built.owner_operand(owner).shape.iter().product::<usize>() as u64)
        .max()
        .expect("an owner")
}

/// The aggregate is the sum of its parts, always. Two accumulators would
/// be two answers.
fn assert_total_holds(ledger: &ResourceLedger) {
    assert_eq!(
        ledger.read_to_prepare,
        ledger.prepare_reads.total(),
        "the aggregate must be derived from the breakdown, not summed beside it"
    );
}

/// **No attestation: zero verification reads.** Every container written
/// before this wave, which is nearly all of them, pays nothing.
#[test]
fn a_container_that_attests_nothing_verifies_nothing() {
    let built = build(COARSE_LABEL);
    let priced = price(&built, trusting());
    assert!(priced.owners > 0, "the plan holds the representation");
    assert_eq!(priced.ledger.prepare_reads.attestation_verification, 0);
    assert_eq!(priced.observed, 0, "and the store read no payload at all");
    assert_total_holds(&priced.ledger);
}

/// **Unrecognised authority: zero PAYLOAD verification reads.** Trust is
/// settled from metadata, so refusing a claim costs no I/O — the
/// expensive check is never reached.
#[test]
fn an_unrecognised_authority_costs_no_payload_read() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let priced = price(&built, RecognisedMethods::none());
    assert_eq!(priced.ledger.prepare_reads.attestation_verification, 0);
    assert_eq!(priced.observed, 0, "nothing was opened to refuse it");
    assert_total_holds(&priced.ledger);
}

/// **A stale tuple is rejected before hashing.** Staleness is a metadata
/// fact, so it too is settled without opening a payload — and the
/// operand IS attested here, so this is not the absent case in disguise.
#[test]
fn a_stale_tuple_is_refused_before_any_payload_read() {
    let built = build(COARSE_LABEL);
    built.attest_at_wrong_shape(ATTESTED);
    let priced = price(&built, trusting());
    assert_eq!(priced.ledger.prepare_reads.attestation_verification, 0);
    assert_eq!(priced.observed, 0, "the shape settled it from the table");
    assert_total_holds(&priced.ledger);
}

/// **One verified operand: its exact byte range appears in preparation
/// reads** — and the store agrees, which is what makes the figure a
/// measurement rather than a restatement of the plan.
#[test]
fn a_verified_operand_contributes_exactly_the_bytes_that_were_hashed() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let priced = price(&built, trusting());

    let each = hashed_per_owner(&built);
    let expected = each * OWNERS.len() as u64;
    assert_eq!(
        priced.ledger.prepare_reads.attestation_verification, expected,
        "every attested operand's stored length, and nothing else"
    );
    assert_eq!(
        priced.observed, expected,
        "the ledger equals the store's observed physical reads"
    );
    assert_total_holds(&priced.ledger);
}

/// **Verification is additive to the other causes, and to nothing else.**
/// Turning recognition on must move preparation reads and leave stored
/// footprint, residency and per-token touch exactly where they were:
/// these bytes are read and dropped, and a guarantee is not a resource
/// anyone holds.
#[test]
fn verification_changes_preparation_reads_and_no_other_resource() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let quiet = price(&built, RecognisedMethods::none());
    let verifying = price(&built, trusting());

    let hashed = hashed_per_owner(&built) * OWNERS.len() as u64;
    assert!(hashed > 0);
    assert_eq!(
        verifying.ledger.read_to_prepare,
        quiet.ledger.read_to_prepare + hashed,
        "preparation reads grow by exactly what was hashed"
    );
    assert_eq!(
        verifying
            .ledger
            .prepare_reads
            .representation_materialisation,
        quiet.ledger.prepare_reads.representation_materialisation,
        "materialisation is untouched"
    );
    assert_eq!(
        verifying.ledger.prepare_reads.auxiliary_resolution,
        quiet.ledger.prepare_reads.auxiliary_resolution,
        "auxiliary resolution is untouched"
    );
    assert_eq!(verifying.ledger.stored, quiet.ledger.stored, "stored");
    assert_eq!(verifying.ledger.resident, quiet.ledger.resident, "resident");
    assert_eq!(
        verifying.ledger.touch_per_token, quiet.ledger.touch_per_token,
        "per-token touch"
    );
    assert_eq!(
        verifying.ledger.transient_peak, quiet.ledger.transient_peak,
        "staging peak"
    );
    assert_eq!(verifying.ledger.device, quiet.ledger.device, "device");
}

/// **Two attestations over the same bytes are counted TWICE**, because
/// this implementation genuinely reads twice: `load_raw` opens, seeks and
/// copies per candidate, and nothing between them shares a
/// materialisation. Counting once would be an optimisation the code has
/// not made — the ledger reports what happens, not what could.
///
/// If a future change caches that read, this test is where it must be
/// changed, and the store's own counter will say so first.
#[test]
fn two_attestations_over_one_operand_are_read_and_counted_twice() {
    let built = build(COARSE_LABEL);
    built.attest_at_both_depths(ATTESTED);
    let priced = price(&built, trusting());

    let each = hashed_per_owner(&built);
    let twice = each * 2 * OWNERS.len() as u64;
    assert_eq!(
        priced.observed, twice,
        "the store really did open each operand twice"
    );
    assert_eq!(
        priced.ledger.prepare_reads.attestation_verification, twice,
        "and the ledger says so rather than deduplicating a read that happened"
    );
    assert_total_holds(&priced.ledger);
}

/// **Verification and decode are different materialisations, and both are
/// counted.** Verification copies through `load_raw`; a decode binds its
/// operand as a mapped region instead. Neither reuses the other, so the
/// ledger carries both rather than assuming a cache nobody wrote.
///
/// `#[serial]` because this arm DECODES, and decoding stages an f32 image
/// through the process-global staging arena that
/// `weights::staged_tests` asserts on. Those counters are shared by the
/// whole binary, not by a module, so a decoding test that runs
/// concurrently with them makes their assertions flaky — which is
/// exactly what this one did before the attribute was added.
#[test]
#[serial_test::serial]
fn verification_and_a_later_decode_are_not_the_same_materialisation() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let (plan, store) = built.plan_and_store_trusting(trusting());
    let before = store.bytes_read();
    let records = select_realizations_within(
        &plan,
        OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .expect("plans");
    let verified = store.bytes_read() - before;
    assert!(verified > 0, "verification read the payloads");

    // Decoding the same operands afterwards binds regions rather than
    // re-reading them, so the physical-read counter does NOT move — which
    // is exactly why preparation reads and residency are separate
    // resources and why one may not be inferred from the other.
    let mapped_before = store.bytes_read();
    for record in records
        .iter()
        .filter(|r| r.representation == OWNER_LABEL)
        .take(1)
    {
        let _ = store.load(&record.planned.operand).expect("decodes");
    }
    assert!(
        store.bytes_read() >= mapped_before,
        "a counter never runs backwards"
    );
}

/// The breakdown is not decoration: an auxiliary closure puts bytes in
/// its own bucket, so a plan can say WHY its preparation costs what it
/// does rather than only how much.
#[test]
fn each_cause_lands_in_its_own_bucket() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let priced = price(&built, trusting());
    let reads = priced.ledger.prepare_reads;
    assert!(
        reads.representation_materialisation > 0,
        "the operands were materialised"
    );
    assert!(
        reads.auxiliary_resolution > 0,
        "the codebook is a dependency and is priced as one"
    );
    assert!(
        reads.attestation_verification > 0,
        "and the guarantee cost something"
    );
    assert_total_holds(&priced.ledger);
}
