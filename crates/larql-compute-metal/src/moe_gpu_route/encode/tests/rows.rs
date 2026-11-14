//! VERIFY-N: the routed layer and its route over a `[rows, hidden]` block.

use super::*;

// ── VERIFY-N: the routed layer over a [rows, hidden] block ──────────────

/// Rows in the block: enough that the per-row routes differ.
const BLOCK_ROWS: usize = 3;

/// A routed layer run as a VERIFY-N block — each row encoded at its byte
/// offset of one `[rows, hidden]` buffer, sharing ONE scratch on one serial
/// encoder — must be BIT-identical to the same rows as independent
/// single-position layers. Same dispatches, same arithmetic; only the
/// residual/output bindings move.
#[test]
fn routed_rows_at_offsets_are_bit_identical_to_single_positions() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let f = build_fixture(&metal);
    let mut moe = f.moe();
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    moe.pre_experts_norm = &f.norm_w;
    metal
        .routed_rows_supported(&moe)
        .expect("a pre-experts-norm layer runs as a block");

    let rows: Vec<Vec<f32>> = (0..BLOCK_ROWS as u32)
        .map(|i| residual(31 + 7 * i))
        .collect();
    let each: Vec<Vec<f32>> = rows
        .iter()
        .map(|x| run_gpu_route(&metal, &moe, x))
        .collect();
    // Distinct inputs must give distinct outputs, or the offsets could all
    // be reading row 0 and still "match".
    assert!(each[0] != each[1] && each[1] != each[2]);

    let s = scratch_for(&metal);
    let table = metal
        .descriptor_table_for_layer(0, &moe, INTER, HIDDEN)
        .expect("table");
    let flat: Vec<f32> = rows.concat();
    let h_block = metal.bufs.transient_from_f32(&flat);
    let out_block = metal.bufs.output((BLOCK_ROWS * HIDDEN * 4) as u64);
    let row_bytes = (HIDDEN * 4) as u64;
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for i in 0..BLOCK_ROWS as u64 {
        metal.encode_moe_layer_gpu_route_at(
            enc,
            &moe,
            &s,
            &table,
            (&h_block, i * row_bytes),
            (&out_block, i * row_bytes),
            1e-6,
        );
    }
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "routed rows block").expect("command buffer completed");
    let block = crate::buffers::read_buffer_f32(&out_block, BLOCK_ROWS * HIDDEN);

    for (i, want) in each.iter().enumerate() {
        let got = &block[i * HIDDEN..(i + 1) * HIDDEN];
        let first = got
            .iter()
            .zip(want)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(
            first, None,
            "block row {i} differs from its single position"
        );
    }
}

/// Under the identity router input the residual IS the expert input,
/// bound at offset 0 inside the router and expert kernels, so a block row
/// at an offset would read row 0. Refused by name before anything encodes.
#[test]
fn routed_rows_refuse_the_identity_router_input() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let f = build_fixture(&metal);
    let why = metal
        .routed_rows_supported(&f.moe())
        .expect_err("identity router input is refused");
    assert!(why.contains("pre-experts-norm"), "{why}");
}

/// Rows in the routing gate: not a power of two.
const ROUTE_ROWS: usize = 5;

/// Genuinely different rows (the fixture's `residual(seed)` moves the
/// input by ~1e-9 per seed, so every row would route alike): row `i` is
/// its own frequency pattern.
fn distinct_row(i: usize) -> Vec<f32> {
    (0..HIDDEN)
        .map(|j| ((j * (2 * i + 3)) as f32 * 0.37 + i as f32).sin())
        .collect()
}

fn read_bits(buf: &metal::Buffer, len: usize) -> Vec<u32> {
    crate::buffers::read_buffer_f32(buf, len)
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// One row routed alone through the single-position kernels, exactly as
/// `encode_moe_layer_gpu_route_at` routes it: norm (offset 0.0) at
/// `rms_norm`'s width, router logits, fused select. Returns the bits of
/// (x, logits, ids, weights).
fn route_one(metal: &MetalBackend, moe: &MoeLayerWeights<'_>, x: &[f32]) -> [Vec<u32>; 4] {
    let h = metal.bufs.transient_from_f32(x);
    let normed = metal.bufs.output((HIDDEN * 4) as u64);
    let norm_w = metal.bufs.get_f32(moe.pre_experts_norm);
    let w_buf = metal.bufs.get_f32(moe.router_proj);
    let bias = (!moe.router_bias.is_empty()).then(|| metal.bufs.get_f32(moe.router_bias));
    let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
        && !moe.router_per_expert_scale.is_empty())
    .then(|| metal.bufs.get_f32(moe.router_per_expert_scale));
    let renorm = moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    crate::stages::input_norm::encode_f32(
        enc,
        &metal.norms.rms_norm_pipeline,
        &h,
        0,
        &norm_w,
        &normed,
        0,
        HIDDEN,
        1e-6,
        0.0,
    );
    let logits =
        metal.encode_moe_router_logits(enc, &w_buf, &normed, bias.as_ref(), NUM_EXPERTS, HIDDEN);
    let (ids, weights) =
        metal.encode_moe_router_select(enc, &logits, pe_scale.as_ref(), NUM_EXPERTS, TOP_K, renorm);
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "route_one").expect("command buffer completed");
    [
        read_bits(&normed, HIDDEN),
        read_bits(&logits, NUM_EXPERTS),
        read_bits(&ids, TOP_K),
        read_bits(&weights, TOP_K),
    ]
}

/// **Rung 4.** A block routed in one pass — rows norm, rows logits, rows
/// select — must choose, for every row, exactly the experts and weights
/// that row gets routed alone, bit for bit, under each weight policy.
#[test]
fn routing_a_block_matches_each_row_routed_alone_bit_for_bit() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let f = build_fixture(&metal);
    let per_expert_scales: Vec<f32> = (0..NUM_EXPERTS).map(|e| 1.0 + e as f32 * 0.01).collect();
    let inputs: Vec<Vec<f32>> = (0..ROUTE_ROWS).map(distinct_row).collect();
    let flat = inputs.concat();

    for policy in ["raw", "renormalized", "per_expert_scale"] {
        let mut moe = f.moe();
        moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
        moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
        moe.pre_experts_norm = &f.norm_w;
        match policy {
            "renormalized" => {
                moe.routing_policy.selected_weight = MoeTopKWeightPolicy::RenormalizedSoftmax
            }
            "per_expert_scale" => {
                moe.routing_policy.expert_scale = MoeExpertScalePolicy::PerExpert;
                moe.router_per_expert_scale = &per_expert_scales;
            }
            _ => {}
        }

        let h_block = metal.bufs.transient_from_f32(&flat);
        let cmd = metal.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let route = metal
            .encode_moe_route_rows(enc, &moe, &h_block, ROUTE_ROWS, HIDDEN, 1e-6)
            .expect("a pre-experts-norm layer routes as a block");
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "route rows").expect("command buffer completed");
        let block = [
            read_bits(&route.x, ROUTE_ROWS * HIDDEN),
            read_bits(&route.logits, ROUTE_ROWS * NUM_EXPERTS),
            read_bits(&route.ids, ROUTE_ROWS * TOP_K),
            read_bits(&route.weights, ROUTE_ROWS * TOP_K),
        ];

        let mut routes = Vec::new();
        for (i, x) in inputs.iter().enumerate() {
            let one = route_one(&metal, &moe, x);
            for (part, (name, width)) in [
                ("x", HIDDEN),
                ("logits", NUM_EXPERTS),
                ("ids", TOP_K),
                ("weights", TOP_K),
            ]
            .iter()
            .enumerate()
            {
                assert_eq!(
                    &block[part][i * width..(i + 1) * width],
                    &one[part][..],
                    "{policy}: row {i} {name} differs from the row routed alone"
                );
            }
            routes.push(one[2].clone());
        }
        // Distinct rows must route differently somewhere, or a block that
        // routed row 0 for every row could pass.
        assert!(
            routes.windows(2).any(|w| w[0] != w[1]),
            "{policy}: every row chose the same experts — the gate cannot see row offsets"
        );
    }
}

/// **Rung 5a.** A routed layer over a block as ONE grouped expert pass —
/// `rows × top_k` slots, each reading its row's input — must equal the
/// same rows run through rung 2's per-row layer, bit for bit, whether the
/// block scratch is sized exactly or for more rows than the block uses.
#[test]
fn a_block_as_one_grouped_expert_pass_matches_per_row_layers_bit_for_bit() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let f = build_fixture(&metal);
    let mut moe = f.moe();
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    moe.pre_experts_norm = &f.norm_w;
    let s = scratch_for(&metal);
    let table = metal
        .descriptor_table_for_layer(0, &moe, INTER, HIDDEN)
        .expect("table");
    let flat: Vec<f32> = (0..ROUTE_ROWS).flat_map(distinct_row).collect();
    let bytes = (ROUTE_ROWS * HIDDEN * 4) as u64;
    let row_bytes = (HIDDEN * 4) as u64;

    let h = metal.bufs.transient_from_f32(&flat);
    let per_row = metal.bufs.output(bytes);
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for i in 0..ROUTE_ROWS as u64 {
        metal.encode_moe_layer_gpu_route_at(
            enc,
            &moe,
            &s,
            &table,
            (&h, i * row_bytes),
            (&per_row, i * row_bytes),
            1e-6,
        );
    }
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "per-row layers").expect("command buffer completed");
    let want = read_bits(&per_row, ROUTE_ROWS * HIDDEN);

    for capacity in [ROUTE_ROWS, 8] {
        let block = crate::MoeBlockScratch::for_layer(&metal, &s, capacity);
        let out = metal.bufs.output(bytes);
        let cmd = metal.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        metal
            .encode_moe_layer_rows(enc, &moe, &s, &block, &table, (&h, &out), ROUTE_ROWS, 1e-6)
            .expect("the block encodes");
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(cmd, "grouped block").expect("command buffer completed");
        let got = read_bits(&out, ROUTE_ROWS * HIDDEN);
        for r in 0..ROUTE_ROWS {
            let span = r * HIDDEN..(r + 1) * HIDDEN;
            assert_eq!(
                got[span.clone()],
                want[span],
                "capacity {capacity}: block row {r} differs from its per-row layer"
            );
        }
    }
}

/// A block scratch built for fewer rows, or for a differently shaped
/// layer, is refused by name rather than overrun.
#[test]
fn a_block_scratch_that_does_not_fit_is_refused() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let f = build_fixture(&metal);
    let mut moe = f.moe();
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    moe.pre_experts_norm = &f.norm_w;
    let s = scratch_for(&metal);
    let table = metal
        .descriptor_table_for_layer(0, &moe, INTER, HIDDEN)
        .expect("table");
    let block = crate::MoeBlockScratch::for_layer(&metal, &s, 2);
    assert!(block.fits(&s, 2) && !block.fits(&s, 3));
    let h = metal.bufs.output((3 * HIDDEN * 4) as u64);
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let why = metal
        .encode_moe_layer_rows(enc, &moe, &s, &block, &table, (&h, &h), 3, 1e-6)
        .expect_err("3 rows do not fit a 2-row block scratch");
    enc.end_encoding();
    assert!(why.contains("does not fit"), "{why}");
}
