//! **Projection-specific physical addressing.**
//!
//! The invariant under test: a logical expert's identity does not imply
//! a shared physical coordinate. Each projection resolves its own.
//!
//! Only the ROUTED bank has coordinates to permute now — the shared
//! branch is its own per-projection region and never resolves through
//! an address table at all.

use super::*;

/// Three permutations of the five routed bank blocks that differ at
/// EVERY index.
///
/// Not value-shifts of one another: a constant offset would still let a
/// single hidden coordinate plus a constant reproduce all three, and the
/// whole point is that no such coordinate exists. Checked, not asserted
/// by eye — `they_differ_everywhere` below.
const PERM_GATE: [usize; RESIDENT] = [3, 0, 4, 1, 2];
const PERM_UP: [usize; RESIDENT] = [1, 4, 2, 0, 3];
const PERM_DOWN: [usize; RESIDENT] = [2, 3, 0, 4, 1];

#[test]
fn the_three_permutations_differ_everywhere() {
    for i in 0..RESIDENT {
        let (g, u, d) = (PERM_GATE[i], PERM_UP[i], PERM_DOWN[i]);
        assert!(
            g != u && u != d && g != d,
            "block {i} maps to {g}/{u}/{d} — a shared coordinate would survive here"
        );
    }
    for p in [PERM_GATE, PERM_UP, PERM_DOWN] {
        let mut seen = p.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..RESIDENT).collect::<Vec<_>>(), "not a permutation");
    }
}

/// Rewrite a bank so block `i` lands at physical position `perm[i]`.
fn permute(src: &[u8], per: usize, perm: &[usize]) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p * per..(p + 1) * per].copy_from_slice(&src[i * per..(i + 1) * per]);
    }
    out
}

/// The table that finds each logical expert after permutation.
fn permuted_table(residency: &[u32], per: usize, perm: &[usize]) -> Vec<u32> {
    residency
        .iter()
        .map(|off| {
            if *off == layer_shader::NOT_RESIDENT {
                layer_shader::NOT_RESIDENT
            } else {
                (perm[*off as usize / per] * per) as u32
            }
        })
        .collect()
}

/// **The decisive control.** One logical expert sits at three different
/// physical slots — one per projection — and the layer must produce the
/// same answer, bit for bit.
///
/// Bit-for-bit rather than within a tolerance, deliberately: the same
/// values are read in the same order, only from different addresses, so
/// any difference at all is an addressing fault rather than numerics.
#[test]
fn one_logical_expert_at_three_different_physical_slots_gives_the_same_answer() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let per = INTER * HIDDEN * 2;

    // The reference: every projection packed in the same order.
    let state_ref = KdaDeviceState::zeros(&b, shape());
    let mut trace_ref = ExecutionTrace::default();
    let (want, _) = b
        .kimi_decoder_layers(
            &[KimiLayerCall {
                weights: f.layer(&state_ref),
            }],
            &f.x,
            Some(&mut trace_ref),
        )
        .expect("reference layer runs");

    let (gate, up, down) = (
        permute(&f.bank_gate, per, &PERM_GATE),
        permute(&f.bank_up, per, &PERM_UP),
        permute(&f.bank_down, per, &PERM_DOWN),
    );
    let (tg, tu, td) = (
        permuted_table(&f.residency, per, &PERM_GATE),
        permuted_table(&f.residency, per, &PERM_UP),
        permuted_table(&f.residency, per, &PERM_DOWN),
    );

    let state = KdaDeviceState::zeros(&b, shape());
    let mut w = f.layer(&state);
    let moe = match &mut w.ffn {
        FfnSpec::Moe(m) => m,
        FfnSpec::Dense(_) => panic!("the fixture layer is routed"),
    };
    moe.gate = projection(&gate, &tg, &f.shared_gate);
    moe.up = projection(&up, &tu, &f.shared_up);
    moe.down = projection(&down, &td, &f.shared_down);

    let mut trace = ExecutionTrace::default();
    let (got, _) = b
        .kimi_decoder_layers(&[KimiLayerCall { weights: w }], &f.x, Some(&mut trace))
        .expect("permuted layer runs");

    // The router chose the same experts — permutation is physical only.
    assert_eq!(
        trace.routes, trace_ref.routes,
        "routing must not depend on where the bytes are stored"
    );
    // And every selected expert really does sit at three different
    // places, or the permutation proved nothing for THIS route.
    for &id in &trace.routes[0] {
        let (g, u, d) = (tg[id as usize], tu[id as usize], td[id as usize]);
        assert!(
            g != u && u != d && g != d,
            "selected expert {id} resolves to {g}/{u}/{d} — not three distinct slots"
        );
    }
    assert_eq!(
        got, want,
        "the same logical experts at different physical slots must give the same answer"
    );

    // The control that the permutation is load-bearing: reading the
    // permuted banks with the UNPERMUTED table must be wrong.
    let state_bad = KdaDeviceState::zeros(&b, shape());
    let mut bad = f.layer(&state_bad);
    if let FfnSpec::Moe(m) = &mut bad.ffn {
        m.gate = projection(&gate, &f.residency, &f.shared_gate);
        m.up = projection(&up, &tu, &f.shared_up);
        m.down = projection(&down, &td, &f.shared_down);
    }
    let (wrong, _) = b
        .kimi_decoder_layers(&[KimiLayerCall { weights: bad }], &f.x, None)
        .expect("runs");
    assert_ne!(
        wrong, want,
        "a wrong gate table must change the answer, or this test cannot fail"
    );
}

/// **A projection that cannot address a selected expert refuses, and
/// the other two succeeding does not rescue it.**
///
/// The failure has to be projection-local. Reusing gate's coordinate for
/// down, or inferring one from a stride, would read a real expert's
/// bytes and produce a plausible wrong answer. Once Q2 composes a
/// candidate overlay over a source container, "the Q6 down is missing,
/// use the BF16 one" is the same defect wearing a different hat.
#[test]
fn one_projection_missing_an_address_refuses_the_whole_layer() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();

    // Which expert the router actually picks, so the hole is put where
    // it will be hit rather than somewhere convenient.
    let probe = KdaDeviceState::zeros(&b, shape());
    let mut trace = ExecutionTrace::default();
    b.kimi_decoder_layers(
        &[KimiLayerCall {
            weights: f.layer(&probe),
        }],
        &f.x,
        Some(&mut trace),
    )
    .expect("probe runs");
    let selected = trace.routes[0][0] as usize;
    assert_ne!(
        f.residency[selected],
        layer_shader::NOT_RESIDENT,
        "the probe must select a resident expert for the hole to be meaningful"
    );

    // gate and up keep their addresses; down loses exactly this one.
    let mut down_table = f.residency.clone();
    down_table[selected] = layer_shader::NOT_RESIDENT;

    let state = KdaDeviceState::zeros(&b, shape());
    let mut w = f.layer(&state);
    if let FfnSpec::Moe(m) = &mut w.ffn {
        m.down.addressing = ExpertAddressing::Table(&down_table);
    }
    let err = b
        .kimi_decoder_layers(&[KimiLayerCall { weights: w }], &f.x, None)
        .expect_err("a projection with no address must refuse");
    assert!(
        matches!(err, GroupedError::LayerRouteNotResident { layer: 0, refusals } if refusals > 0),
        "expected a layer-0 route refusal, got {err:?}"
    );

    // And the control: with the hole removed the very same layer runs,
    // so the refusal is the missing address and not the fixture.
    let state_ok = KdaDeviceState::zeros(&b, shape());
    b.kimi_decoder_layers(
        &[KimiLayerCall {
            weights: f.layer(&state_ok),
        }],
        &f.x,
        None,
    )
    .expect("the same layer runs once every projection can address the route");
}
