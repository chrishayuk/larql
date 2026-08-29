//! **The shared branch is an independent physical region.**
//!
//! `Shared` vs `Routed` is semantic identity; it must not imply
//! co-location. The base fixture already binds the shared branch from
//! its own allocations — three separate `Vec`s, so three device buffers
//! distinct from the routed banks. These controls pin the rest of the
//! claim:
//!
//! * a shared region carved out of the SAME allocation as the routed
//!   bank — the old fixture's co-located layout — is still
//!   representable and still correct, without being an invariant;
//! * projections disagreeing about the branch's existence are refused;
//! * a shared region too small for its projection is refused by name.

use super::*;

/// **Co-location is a representable layout, never an invariant.**
///
/// Rebuild the old fixture shape — shared bytes appended to each routed
/// bank — and bind the shared region as a subrange of that same
/// allocation. The answer must be bit-identical to the base fixture's,
/// where the shared branch lives in its own allocations: the same
/// values are read through the same kernels in the same slot order, so
/// any difference at all is a binding fault.
#[test]
fn a_colocated_shared_region_equals_a_standalone_one() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();

    let state_a = KdaDeviceState::zeros(&b, shape());
    let (standalone, _) = b
        .kimi_decoder_layer(f.layer(&state_a), &f.x)
        .expect("standalone shared regions run");

    // The old layout: routed bank with the shared payload appended.
    let cat = |routed: &[u8], shared: &[u8]| {
        let mut v = routed.to_vec();
        v.extend_from_slice(shared);
        v
    };
    let (cat_gate, cat_up, cat_down) = (
        cat(&f.bank_gate, &f.shared_gate),
        cat(&f.bank_up, &f.shared_up),
        cat(&f.bank_down, &f.shared_down),
    );

    let state_b = KdaDeviceState::zeros(&b, shape());
    let mut w = f.layer(&state_b);
    {
        let m = moe_mut(&mut w);
        for (bank, cat, routed_len) in [
            (&mut m.gate, &cat_gate, f.bank_gate.len()),
            (&mut m.up, &cat_up, f.bank_up.len()),
            (&mut m.down, &cat_down, f.bank_down.len()),
        ] {
            bank.routed.bytes = cat.as_slice();
            bank.shared = Some(EncodedRegion {
                bytes: &cat[routed_len..],
                encoding: ExpertEncoding::Bf16,
            });
        }
    }
    let (colocated, _) = b
        .kimi_decoder_layer(w, &f.x)
        .expect("co-located shared regions run");

    assert_eq!(
        colocated, standalone,
        "the shared branch's answer must not depend on which allocation its bytes share"
    );
}

/// **The branch's existence is one semantic fact.** `Some` for gate but
/// `None` for down would silently drop one projection's shared
/// contribution — refused before anything is encoded.
#[test]
fn projections_disagreeing_about_the_shared_branch_are_refused() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let mut w = f.layer(&state);
    moe_mut(&mut w).down.shared = None;
    assert_eq!(
        b.kimi_decoder_layer(w, &f.x).map(|(o, _)| o),
        Err(GroupedError::SharedBranchInconsistent)
    );
}

/// A layer whose three projections consistently declare NO shared
/// branch runs routed-only — and its answer must differ from the
/// with-shared layer's, or the branch was contributing nothing and
/// every shared assertion in this module is vacuous.
#[test]
fn a_layer_without_a_shared_branch_runs_routed_only() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let mut w = f.layer(&state);
    {
        let m = moe_mut(&mut w);
        m.gate.shared = None;
        m.up.shared = None;
        m.down.shared = None;
    }
    let (routed_only, _) = b
        .kimi_decoder_layer(w, &f.x)
        .expect("a consistently shared-less layer runs");

    let state_full = KdaDeviceState::zeros(&b, shape());
    let (full, _) = b
        .kimi_decoder_layer(f.layer(&state_full), &f.x)
        .expect("the with-shared layer runs");
    assert_ne!(
        routed_only, full,
        "removing the shared branch must move the answer"
    );
}

/// A shared region too small for its projection is refused by name,
/// before any kernel runs — the same discipline the routed banks get.
#[test]
fn a_truncated_shared_region_is_refused_by_projection() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    for which in 0..3usize {
        let state = KdaDeviceState::zeros(&b, shape());
        let mut w = f.layer(&state);
        {
            let m = moe_mut(&mut w);
            let bank = match which {
                0 => &mut m.gate,
                1 => &mut m.up,
                _ => &mut m.down,
            };
            let half = bank.shared.expect("fixture has a shared branch");
            bank.shared = Some(EncodedRegion {
                bytes: &half.bytes[..half.bytes.len() / 2],
                encoding: half.encoding,
            });
        }
        assert!(
            matches!(
                b.kimi_decoder_layer(w, &f.x),
                Err(GroupedError::OffsetOutOfRange { slot, .. }) if slot == which
            ),
            "a half-length shared projection {which} must be refused by name"
        );
    }
}
