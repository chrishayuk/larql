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
    // Both slices start ALIGNED, so the only thing that can decide
    // either answer is whether it lies inside the logical extent — the
    // alignment rule must not be what makes the negative case pass.
    let inside = len / super::WEIGHT_BINDING_ALIGN * super::WEIGHT_BINDING_ALIGN
        - super::WEIGHT_BINDING_ALIGN;
    assert!(cache.resolve_region(&region[inside..len]).is_some());
    // Reconstruct a slice crossing the logical end via raw parts —
    // the mmap maps the whole final page, so this is readable memory
    // that is nonetheless OUTSIDE the registered data.
    let straddle = len.div_ceil(super::WEIGHT_BINDING_ALIGN) * super::WEIGHT_BINDING_ALIGN
        - super::WEIGHT_BINDING_ALIGN;
    assert!(straddle.is_multiple_of(super::WEIGHT_BINDING_ALIGN));
    let past = unsafe { std::slice::from_raw_parts(region.as_ptr().add(straddle), 64) };
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
    crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/buffers/regions/tests.rs:104",
    )
    .expect("command buffer completed");
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
        crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/buffers/regions/tests.rs:125",
        )
        .expect("command buffer completed");
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

/// **A GPU dispatch through a registered FILE-BACKED read-only mmap
/// must read the file's bytes.**
///
/// The CPU-side aliasing checks above cannot catch a mapping the GPU
/// sees differently: `contents()` is the host pointer. This runs a real
/// grouped matvec whose weights resolve into a registered region of a
/// read-only file mapping, at a non-zero offset, and compares against
/// the same bytes staged through the copy path.
#[test]
fn a_gpu_dispatch_reads_a_registered_file_mmap_at_an_offset() {
    use crate::trait_impl::grouped_experts::{ExpertOffset, InputLayout};
    use crate::MetalBackend;
    use std::io::Write;

    let Some(metal) = MetalBackend::new() else {
        panic!("Metal device available on test host");
    };
    let (n, k) = (8usize, 64usize);
    // A header-like prefix so the weights sit at a NON-ZERO, non-page
    // offset inside the mapping — the real container's shape.
    let prefix = 4096 + 12;
    let codes: Vec<u8> = (0..n * k)
        .flat_map(|i| {
            let v = ((i as f32) * 0.37).sin() * 0.5;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("larql-region-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let path = dir.join("weights.bin");
    {
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&vec![0xAAu8; prefix]).expect("prefix");
        f.write_all(&codes).expect("codes");
    }
    let file = std::fs::File::open(&path).expect("open");
    // SAFETY: read-only mapping of a file this test owns.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
    metal.bufs().register_region(&mmap[..]);

    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.11).cos()).collect();
    let weights = &mmap[prefix..prefix + codes.len()];
    let via_region = metal
        .bf16_grouped_experts(weights, &[ExpertOffset(0)], &x, n, k, InputLayout::Shared)
        .expect("region-resolved dispatch");

    // The control arm: identical bytes, staged through the copy path.
    let staged: Vec<u8> = codes.clone();
    let via_copy = metal
        .bf16_grouped_experts(&staged, &[ExpertOffset(0)], &x, n, k, InputLayout::Shared)
        .expect("staged dispatch");
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        via_region, via_copy,
        "the GPU must see the file's bytes through the registered mapping"
    );
}

/// **What alignment does a weight binding actually require?**
///
/// Measured rather than taken from a feature table: place identical
/// bf16 weights at offsets of each residue class inside a registered
/// file mapping, dispatch the real grouped kernel through the resolved
/// region, and compare against the staged-copy arm. The lowest residue
/// that agrees everywhere is the requirement, and `WEIGHT_BINDING_ALIGN`
/// must be at least that.
#[test]
fn the_binding_alignment_requirement_is_measured_not_assumed() {
    use crate::trait_impl::grouped_experts::{ExpertOffset, InputLayout};
    use crate::MetalBackend;
    use std::io::Write;

    let Some(metal) = MetalBackend::new() else {
        panic!("Metal device available on test host");
    };
    let (n, k) = (8usize, 64usize);
    let codes: Vec<u8> = (0..n * k)
        .flat_map(|i| {
            let v = ((i as f32) * 0.37).sin() * 0.5;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.11).cos()).collect();
    let want = metal
        .bf16_grouped_experts(&codes, &[ExpertOffset(0)], &x, n, k, InputLayout::Shared)
        .expect("staged reference");

    let dir = std::env::temp_dir().join(format!("larql-align-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let mut verdicts = Vec::new();
    for skew in [0usize, 1, 2, 4, 8, 12] {
        let prefix = super::PAGE_SIZE + skew;
        let path = dir.join(format!("w_{skew}.bin"));
        {
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(&vec![0xAAu8; prefix]).expect("prefix");
            f.write_all(&codes).expect("codes");
        }
        let file = std::fs::File::open(&path).expect("open");
        // SAFETY: read-only mapping of a file this test owns.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
        let cache = BufferCache::new(&dev());
        cache.register_region(&mmap[..]);
        let weights = &mmap[prefix..prefix + codes.len()];
        // Bind through the region DIRECTLY, bypassing the alignment
        // filter, because the filter is what this test is calibrating.
        let (buf, off) = {
            let p = weights.as_ptr() as usize;
            let base = mmap.as_ptr() as usize;
            let raw = cache
                .resolve_region(&mmap[..64])
                .expect("the region resolves at zero")
                .0;
            (raw, (p - base) as u64)
        };
        let out = metal
            .bf16_grouped_experts_at(&buf, off, &[ExpertOffset(0)], &x, n, k, InputLayout::Shared)
            .expect("dispatch");
        verdicts.push((skew, out == want));
    }
    std::fs::remove_dir_all(&dir).ok();
    eprintln!("[align] offset skew -> agrees with staged: {verdicts:?}");
    let worst_bad = verdicts
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(0);
    assert!(
        super::WEIGHT_BINDING_ALIGN > worst_bad,
        "WEIGHT_BINDING_ALIGN ({}) must exceed every skew that misreads ({worst_bad})",
        super::WEIGHT_BINDING_ALIGN
    );
    for (skew, ok) in &verdicts {
        if skew.is_multiple_of(super::WEIGHT_BINDING_ALIGN) {
            assert!(ok, "an ALIGNED binding must read correctly (skew {skew})");
        }
    }
}

/// **A misaligned sub-slice must MISS.**
///
/// The real Kimi container's `decoder_stack` payload started at byte
/// 56,925 — odd — so every bf16 tensor in it sat at an odd address.
/// Bound zero-copy, the grouped kernel read a `ushort*` at an odd
/// pointer and returned NaN with a command buffer that reported
/// success. Resolution must decline instead, leaving the caller on the
/// staged-copy path, which is slow and correct.
#[test]
fn a_misaligned_offset_misses_rather_than_binding() {
    let cache = BufferCache::new(&dev());
    let region = anon_region(2 * super::PAGE_SIZE);
    assert!(cache.register_region(&region[..]));

    let aligned = &region[super::WEIGHT_BINDING_ALIGN * 3..][..64];
    assert!(
        cache.resolve_region(aligned).is_some(),
        "an aligned slice must still resolve"
    );
    for skew in (1..super::WEIGHT_BINDING_ALIGN).chain([super::WEIGHT_BINDING_ALIGN + 1]) {
        let sub = &region[super::WEIGHT_BINDING_ALIGN * 3 + skew..][..64];
        assert!(
            cache.resolve_region(sub).is_none(),
            "an offset {skew} bytes off alignment must miss, not bind misaligned"
        );
    }
}
