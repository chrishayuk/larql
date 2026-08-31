//! The model head — final norm plus the vocabulary projection.

use super::*;
use crate::trait_impl::kimi_layer::ExpertEncoding;

const VOCAB: usize = 5;

fn bf16_decode(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// The head against a host reference: RMS norm, then a plain
/// `vocab x hidden` projection over the SAME decoded values.
#[test]
fn the_head_matches_a_host_reference() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let x = synth(HIDDEN, 5.5);
    let norm_w: Vec<f32> = synth(HIDDEN, 6.6).iter().map(|v| v + 1.0).collect();
    let w = bf16_bytes(VOCAB, HIDDEN, 7.7);

    let (got, _) = b
        .kimi_head(
            &KimiHead {
                norm_weight: &norm_w,
                norm_eps: EPS,
                weight: &w,
                vocab: VOCAB,
                encoding: ExpertEncoding::Bf16,
            },
            &x,
        )
        .expect("the head runs at these shapes");

    let normed = rms_norm(&x, &norm_w, EPS);
    let rows = bf16_decode(&w);
    for v in 0..VOCAB {
        let want: f32 = rows[v * HIDDEN..(v + 1) * HIDDEN]
            .iter()
            .zip(&normed)
            .map(|(a, c)| a * c)
            .sum();
        assert!(
            (got[v] - want).abs() <= TOLERANCE,
            "vocab row {v}: {} vs {want}",
            got[v]
        );
    }
}

/// **The head encoded INSIDE the layer's command buffer must equal the
/// head run separately on that layer's output.**
///
/// This is the only assertion that can catch the head reading the wrong
/// buffer: it consumes the last layer's device-resident output rather
/// than anything the host handed it, so a binding that pointed at, say,
/// the post-attention plane would still produce plausible logits.
#[test]
fn the_head_inside_the_chain_equals_the_head_run_after_it() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let f = fixture();
    let norm_w: Vec<f32> = synth(HIDDEN, 8.8).iter().map(|v| v + 1.0).collect();
    let w = bf16_bytes(VOCAB, HIDDEN, 9.9);
    let head = KimiHead {
        norm_weight: &norm_w,
        norm_eps: EPS,
        weight: &w,
        vocab: VOCAB,
        encoding: ExpertEncoding::Bf16,
    };

    let state_a = KdaDeviceState::zeros(&b, shape());
    let (fused, _) = b
        .kimi_decoder_layers_with_head(
            &[KimiLayerCall {
                weights: f.layer(&state_a),
            }],
            &head,
            &f.x,
            None,
        )
        .expect("chain with head runs");

    let state_b = KdaDeviceState::zeros(&b, shape());
    let (hidden_out, _) = b
        .kimi_decoder_layer(f.layer(&state_b), &f.x)
        .expect("layer runs");
    let (separate, _) = b.kimi_head(&head, &hidden_out).expect("head runs");

    assert_eq!(fused.len(), VOCAB);
    for (v, (a, c)) in fused.iter().zip(&separate).enumerate() {
        assert!(
            (a - c).abs() <= TOLERANCE,
            "vocab row {v}: fused {a} vs separate {c}"
        );
    }
    // The control: the head must actually depend on the layer's output,
    // not merely on the head weights.
    let (other, _) = b.kimi_head(&head, &[0.25f32; HIDDEN]).expect("head runs");
    assert!(
        fused
            .iter()
            .zip(&other)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max)
            > TOLERANCE,
        "a different hidden state must give different logits"
    );
}

/// A head whose declared shape disagrees with its buffers is refused
/// before anything is encoded.
#[test]
fn head_shape_faults_are_refused() {
    let Some(b) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let x = synth(HIDDEN, 1.1);
    let norm_w: Vec<f32> = synth(HIDDEN, 2.2).iter().map(|v| v + 1.0).collect();
    let w = bf16_bytes(VOCAB, HIDDEN, 3.3);
    let ok = KimiHead {
        norm_weight: &norm_w,
        norm_eps: EPS,
        weight: &w,
        vocab: VOCAB,
        encoding: ExpertEncoding::Bf16,
    };

    let mut truncated = ok;
    truncated.weight = &w[..w.len() / 2];
    let mut no_vocab = ok;
    no_vocab.vocab = 0;
    let short_norm = synth(HIDDEN - 1, 2.2);
    let mut bad_norm = ok;
    bad_norm.norm_weight = &short_norm;

    for (name, bad) in [
        ("truncated projection", truncated),
        ("zero vocab", no_vocab),
        ("short norm weight", bad_norm),
    ] {
        assert!(
            matches!(
                b.kimi_head(&bad, &x),
                Err(GroupedError::HeadShapeMismatch { .. })
            ),
            "{name} must be refused"
        );
    }
    // And the Display arm names the shapes, since a caller sees only this.
    let msg = GroupedError::HeadShapeMismatch {
        vocab: VOCAB,
        hidden: HIDDEN,
        have_bytes: 3,
    }
    .to_string();
    assert!(msg.contains("model head") && msg.contains(&format!("{}", VOCAB * HIDDEN * 2)));
}
