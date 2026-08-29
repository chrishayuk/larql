//! `tests` for [`super`].
//!
//! Split out of `regions.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::super::BufferCache;
use metal::foreign_types::ForeignType;
use metal::Device;

fn dev() -> Device {
    Device::system_default().expect("Metal device available on test host")
}

/// Page-aligned by construction — the production contract's shape.
fn anon_region(len: usize) -> memmap2::Mmap {
    let mut m = memmap2::MmapMut::map_anon(len).expect("anon mmap");
    for (i, b) in m.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    m.make_read_only().expect("read-only")
}

/// Sub-slices anywhere inside a registered region resolve to the
/// region's buffer at the right byte offset; slices outside miss.
#[test]
fn resolve_returns_offset_within_registered_region() {
    let cache = BufferCache::new(&dev());
    let region = anon_region(3 * super::PAGE_SIZE / 2); // non-page-multiple len
    assert!(cache.register_region(&region[..]));

    let sub = &region[4096..4096 + 512];
    let (buf, off) = cache.resolve_region(sub).expect("inside must resolve");
    assert_eq!(off, 4096);
    // The buffer aliases the region: bytes at the offset match.
    let p = buf.contents() as *const u8;
    let via_buf = unsafe { std::slice::from_raw_parts(p.add(off as usize), sub.len()) };
    assert_eq!(via_buf, sub);

    let other = anon_region(super::PAGE_SIZE);
    assert!(
        cache.resolve_region(&other[..64]).is_none(),
        "unregistered allocation must miss"
    );
}

/// A sub-slice extending past the region's logical end must miss —
/// the rounded-up buffer tail is allocation padding, never data.
#[test]
fn resolve_rejects_slices_past_the_logical_end() {
    let cache = BufferCache::new(&dev());
    let len = super::PAGE_SIZE + 100; // logical end mid-page
    let region = anon_region(len);
    assert!(cache.register_region(&region[..]));
    assert!(cache.resolve_region(&region[len - 50..len]).is_some());
    // Reconstruct a slice crossing the logical end via raw parts —
    // the mmap maps the whole final page, so this is readable memory
    // that is nonetheless OUTSIDE the registered data.
    let past = unsafe { std::slice::from_raw_parts(region.as_ptr().add(len - 10), 20) };
    assert!(cache.resolve_region(past).is_none());
}

/// Re-registering the same base is a no-op; misaligned bases and
/// empty regions refuse.
#[test]
fn register_dedupes_and_rejects_unusable_regions() {
    let cache = BufferCache::new(&dev());
    let region = anon_region(super::PAGE_SIZE);
    assert!(cache.register_region(&region[..]));
    assert!(cache.register_region(&region[..]), "re-register is a no-op");
    assert_eq!(cache.region_count(), 1);

    // Interior pointer — not page-aligned.
    assert!(!cache.register_region(&region[8..]));
    assert!(!cache.register_region(&region[..0]));
    assert_eq!(cache.region_count(), 1);
}

// ── seal_residency: the arm the A/B/C ladder drives ──────────────────────
//
// `seal_residency` is the only part of this module the region tests above
// do not reach, and it is the one whose *silence* is dangerous: an arm that
// returns early still produces timings, so a null result is only
// interpretable if the arm demonstrably ran. Its own perf verdict (refuted
// — explicit residency buys nothing) lives in `buffers::residency`; nothing
// here re-litigates it.

use super::super::residency::tests::with_residency_env;

/// Arm A is the shipped default: `seal_residency` must return before it
/// touches the queue, so arm A is byte-identical to the pre-residency code.
#[test]
fn sealing_is_a_no_op_under_the_implicit_arm() {
    let cache = BufferCache::new(&dev());
    let queue = dev().new_command_queue();
    let region = anon_region(super::PAGE_SIZE);
    assert!(cache.register_region(&region[..]));

    with_residency_env(None, || cache.seal_residency(&queue));

    // The queue is untouched and still executes.
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.end_encoding();
    cmd.commit();
    let _ = crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/buffers/regions/tests.rs:104",
    );
}

/// Both explicit arms build, commit and attach a set over the registered
/// regions. Arm C additionally requests residency up front; from the
/// caller's side the observable contract is the same — the queue keeps
/// working — which is what makes the measured null result meaningful
/// rather than an artefact of a broken attach.
#[test]
fn sealing_runs_both_explicit_arms_and_leaves_the_queue_usable() {
    let cache = BufferCache::new(&dev());
    let queue = dev().new_command_queue();
    let region = anon_region(2 * super::PAGE_SIZE);
    assert!(cache.register_region(&region[..]));

    for arm in ["1", "2"] {
        with_residency_env(Some(arm), || cache.seal_residency(&queue));
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/buffers/regions/tests.rs:125",
        );
    }
}

/// Sealing is documented as idempotent and safe after each registration
/// batch, so a second call over a grown region list must also be fine.
#[test]
fn sealing_twice_over_a_growing_region_list_is_safe() {
    let cache = BufferCache::new(&dev());
    let queue = dev().new_command_queue();
    let first = anon_region(super::PAGE_SIZE);
    assert!(cache.register_region(&first[..]));
    with_residency_env(Some("1"), || cache.seal_residency(&queue));

    let second = anon_region(super::PAGE_SIZE);
    assert!(cache.register_region(&second[..]));
    assert_eq!(cache.region_count(), 2);
    with_residency_env(Some("1"), || cache.seal_residency(&queue));
}

/// With nothing registered there is nothing to declare: the explicit arms
/// must return before building a set rather than attaching an empty one.
#[test]
fn sealing_with_no_regions_returns_early() {
    let cache = BufferCache::new(&dev());
    let queue = dev().new_command_queue();
    with_residency_env(Some("2"), || cache.seal_residency(&queue));
    assert_eq!(cache.region_count(), 0);
}

/// `weights` must hand back the REGISTERED REGION's buffer, not a
/// private one over the same pages.
///
/// This is the whole point of the method, and the assertion that would
/// have caught the original defect is the object-identity one: a
/// `get_bytes` buffer aliases the same bytes and passes every value
/// check, while being a different `MTLBuffer` that the residency set
/// does not cover. On the 27 GB trajectory that distinction was worth
/// 29.9 ms a token.
#[test]
fn weights_binds_the_region_buffer_itself_not_an_alias() {
    let cache = BufferCache::new(&dev());
    let region = anon_region(4 * super::PAGE_SIZE);
    assert!(cache.register_region(&region[..]));

    let sub = &region[super::PAGE_SIZE..super::PAGE_SIZE + 1024];
    let (w_buf, w_off) = cache.weights(sub);
    let (r_buf, r_off) = cache.resolve_region(sub).expect("registered");
    assert_eq!(w_off, super::PAGE_SIZE as u64);
    assert_eq!(w_off, r_off);
    assert_eq!(
        w_buf.as_ptr(),
        r_buf.as_ptr(),
        "weights must return the region's own buffer object"
    );
    // The defect that motivated this: an aliasing buffer is a DIFFERENT
    // object, so the residency set never covers what the encoder binds.
    assert_ne!(
        w_buf.as_ptr(),
        cache.get_bytes(sub).as_ptr(),
        "get_bytes must be a distinct object — that is the hazard"
    );
}

/// Nothing registered → `weights` falls back to a whole-buffer bind at
/// offset zero, which is the pre-existing behaviour, not an error.
#[test]
fn weights_falls_back_to_a_zero_offset_buffer_when_unregistered() {
    let cache = BufferCache::new(&dev());
    let region = anon_region(2 * super::PAGE_SIZE);
    let sub = &region[64..64 + 256];

    let (buf, off) = cache.weights(sub);
    assert_eq!(off, 0, "the fallback buffer starts at the slice itself");
    let seen = unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, sub.len()) };
    assert_eq!(seen, sub);
}
