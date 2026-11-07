//! VERIFY-N: the routed stack as a multi-position block.

use super::*;

/// VERIFY-N block rows for the routed stack: three positions, the block's
/// first at position 1 so every row's K/V slot and span fit the cache.
const BLOCK_ROWS: usize = 3;
const BLOCK_BASE: usize = 1;

/// The routed stack as a VERIFY-N block of `BLOCK_ROWS` positions; with
/// `block`, routed layers run as one grouped expert pass
/// (`StackScratch::routed_block`), without it row by row. Returns the
/// final `[rows, hidden]` hidden state. Layers and caches are uploaded
/// fresh per call, so two calls never share device state.
fn gpu_stack_rows(gpu: &MetalBackend, h0: &[f32], fx: &[LayerFixture], block: bool) -> Vec<f32> {
    let dev: Vec<LayerDevice> = fx
        .iter()
        .enumerate()
        .map(|(i, w)| upload_layer(gpu, i, w))
        .collect();
    let r = BLOCK_ROWS;
    let h_a = gpu.lowering_upload(h0).unwrap();
    let inv_freq = gpu.lowering_upload(&[0.0f32; HEAD_DIM / 2]).unwrap();
    let sc: Vec<metal::Buffer> = (0..13)
        .map(|i| match i {
            2..=4 => gpu.lowering_scratch(r * Q_ROWS),
            8..=10 => gpu.lowering_scratch(r * INTER),
            _ => gpu.lowering_scratch(r * HIDDEN),
        })
        .collect();
    let routed_block =
        block.then(|| larql_compute_metal::MoeBlockScratch::for_layer(gpu, &dev[0].scratch, r));
    let scratch = StackScratch {
        h_a: &h_a,
        h_b: &sc[0],
        attn_normed: &sc[1],
        q: &sc[2],
        gate: &sc[3],
        concat: &sc[4],
        gated: &sc[11],
        attn_out: &sc[5],
        attn_post: &sc[6],
        ffn_normed: &sc[7],
        ffn_gate: &sc[8],
        ffn_up: &sc[9],
        ffn_act: &sc[10],
        ffn_down: &sc[12],
        ffn_post: &sc[1],
        hybrid: None,
        splitk: None,
        routed_block: routed_block.as_ref(),
    };
    let layers: Vec<LayerLowering> = dev
        .iter()
        .zip(fx)
        .map(|(d, w)| LayerLowering {
            attn: AttnWeights {
                q: LoweredMatrix::F16 {
                    bytes: &d.weights[0],
                },
                k: LoweredMatrix::F16 {
                    bytes: &d.weights[1],
                },
                v: LoweredMatrix::F16 {
                    bytes: &d.weights[2],
                },
                o: LoweredMatrix::F16 {
                    bytes: &d.weights[3],
                },
                gate: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                o_bias: None,
                sinks: None,
                qk_norm: None,
                norm_weight: &d.attn_norm,
                post_norm: None,
            },
            attn_shape: AttnShape {
                position_index: BLOCK_BASE,
                kv_len: BLOCK_BASE + 1,
                ..attn_shape()
            },
            ffn: LayerFfnLowering::Routed(Box::new(RoutedFfnLowering {
                moe: w.moe(),
                scratch: &d.scratch,
                table: &d.table,
                eps: EPS,
            })),
            k_cache: &d.k_cache,
            v_cache: &d.v_cache,
            inv_freq: &inv_freq,
        })
        .collect();
    let mut out = None;
    run_once(gpu, |enc| {
        out = Some(
            gpu.encode_stack_rows(&mut SingleEncoder(enc), &h_a, &layers, &scratch, r)
                .expect("the routed stack encodes as a block"),
        );
    });
    gpu.lowering_readback(out.unwrap(), r * HIDDEN).unwrap()
}

/// **VERIFY-N rungs 5a/5b at stack level.** A routed MXFP4 stack run as a
/// block with the grouped expert pass must equal the same block run row by
/// row (rung 2, itself gated against single positions), bit for bit — the
/// grouped pass, its slot layout, the expert-sharing arm and the per-row
/// combine change no arithmetic.
#[test]
fn routed_stack_block_with_the_grouped_expert_pass_matches_the_per_row_block() {
    let gpu = device();
    let h0 = det(BLOCK_ROWS * HIDDEN, 4242, HIDDEN_AMPLITUDE);
    let fx: Vec<LayerFixture> = (0..LAYERS as u32).map(|l| build_layer(l, None)).collect();
    let per_row = gpu_stack_rows(&gpu, &h0, &fx, false);
    let grouped = gpu_stack_rows(&gpu, &h0, &fx, true);
    assert!(per_row.iter().all(|v| v.is_finite()));
    // Distinct rows must end distinct, or a block that computed row 0 for
    // every row could pass.
    assert_ne!(per_row[..HIDDEN], per_row[HIDDEN..2 * HIDDEN]);
    let first = per_row
        .iter()
        .zip(&grouped)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(first, None, "grouped block differs from the per-row block");
}
