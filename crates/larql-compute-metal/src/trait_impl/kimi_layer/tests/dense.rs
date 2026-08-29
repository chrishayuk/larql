//! The dense FFN half.
//!
//! Gated against the routed path rather than a second hand-written
//! reference: `branch_scale = 0` leaves only the shared branch, which
//! IS a dense MLP, so the two must agree element for element.

use super::*;

impl Fixture {
    /// The shared expert's own slice of each bank — one gated MLP.
    ///
    /// `residency` uses ONE offset for all three projections, which is
    /// only sound because `INTER * HIDDEN == HIDDEN * INTER`; the same
    /// fact the routed path already relies on.
    fn dense_banks(&self) -> (&[u8], &[u8], &[u8]) {
        let (o, per) = (self.shared_offset as usize, INTER * HIDDEN * 2);
        (
            &self.bank_gate[o..o + per],
            &self.bank_up[o..o + per],
            &self.bank_down[o..o + per],
        )
    }

    fn dense_layer<'a>(&'a self, state: &'a KdaDeviceState) -> KimiLayerWeights<'a> {
        let (gate, up, down) = self.dense_banks();
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: AttentionSpec::Kda {
                weights: self.kda(),
                shape: shape(),
                state,
            },
            ffn: FfnSpec::Dense(KimiDenseFfn {
                gate,
                up,
                down,
                inter: INTER,
            }),
            norm_eps: EPS,
        }
    }
}

/// **The dense FFN against the routed one, with the routed branch
/// switched off.**
///
/// `branch_scale = 0` zeroes every routed weight, leaving the shared
/// branch summed unscaled — which is exactly what a dense MLP is. So a
/// dense layer built on the shared expert's own weights must reproduce
/// that routed layer element for element, through the same attention,
/// the same norms and the same residuals.
///
/// This gates the new path against machinery that is already proven,
/// rather than against a second hand-written reference that could be
/// wrong in the same direction.
#[test]
fn a_dense_layer_equals_the_routed_layer_with_the_routed_branch_switched_off() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();

    let state_a = KdaDeviceState::zeros(&b, shape());
    let mut routed = f.layer(&state_a);
    moe_mut(&mut routed).branch_scale = 0.0;
    let (want, _) = b
        .kimi_decoder_layer(routed, &f.x)
        .expect("routed layer runs");

    let state_b = KdaDeviceState::zeros(&b, shape());
    let (got, _) = b
        .kimi_decoder_layer(f.dense_layer(&state_b), &f.x)
        .expect("dense layer runs");

    assert_eq!(got.len(), HIDDEN);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= TOLERANCE,
            "element {i}: dense {g} vs routed-with-shared-only {w}"
        );
    }

    // The control: the equality above must not be something any pair of
    // layers satisfies. Left as the routed layer WITH its routed branch,
    // the two must disagree.
    let state_c = KdaDeviceState::zeros(&b, shape());
    let (full, _) = b
        .kimi_decoder_layer(f.layer(&state_c), &f.x)
        .expect("routed layer runs");
    let moved = got
        .iter()
        .zip(&full)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > TOLERANCE,
        "the routed branch must change the answer, or the comparison above \
         is passing on a layer whose experts contribute nothing"
    );
}

/// A dense bank whose length disagrees with its declared `inter` is
/// refused by name, not read.
#[test]
fn a_truncated_dense_bank_is_refused() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let mut w = f.dense_layer(&state);
    let (gate, up, down) = f.dense_banks();
    for which in 0..3 {
        let mut d = KimiDenseFfn {
            gate,
            up,
            down,
            inter: INTER,
        };
        match which {
            0 => d.gate = &gate[..gate.len() / 2],
            1 => d.up = &up[..up.len() / 2],
            _ => d.down = &down[..down.len() / 2],
        }
        w.ffn = FfnSpec::Dense(d);
        assert!(
            matches!(
                b.kimi_decoder_layer(w, &f.x),
                Err(GroupedError::OffsetOutOfRange { slot, .. }) if slot == which
            ),
            "a half-length projection {which} must be refused by name"
        );
    }

    // Zero width is refused too, rather than dispatching an empty grid.
    w.ffn = FfnSpec::Dense(KimiDenseFfn {
        gate,
        up,
        down,
        inter: 0,
    });
    assert!(matches!(
        b.kimi_decoder_layer(w, &f.x),
        Err(GroupedError::NoExpertsSelected)
    ));
}

/// A dense layer routes nothing, and the trace must say so rather than
/// reporting the scratch placeholder as a selected expert.
#[test]
fn a_dense_layer_contributes_an_empty_route() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let state = KdaDeviceState::zeros(&b, shape());
    let mut trace = ExecutionTrace::default();
    b.kimi_decoder_layers(
        &[KimiLayerCall {
            weights: f.dense_layer(&state),
        }],
        &f.x,
        Some(&mut trace),
    )
    .expect("dense layer runs");
    assert_eq!(trace.routes, vec![Vec::<u32>::new()]);
}
