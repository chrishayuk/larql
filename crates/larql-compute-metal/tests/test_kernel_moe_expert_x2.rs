//! The x2 expert arm (`mxfp4g_split_lut16_vec_x2`) against the
//! production vec arm: bit-identical per row, on both fused-row walks,
//! odd row counts, and multi-slot grids — the A-12 expert-pass kernel
//! (262 → 313 GB/s at the gpt-oss expert shape, `examples/moe_expert_alpha_b.rs`).

#![cfg(target_os = "macos")]

use larql_compute_metal::MetalBackend;
use metal::MTLSize;

const K: usize = 2816; // 88 groups
const GROUP_BYTES: usize = 16;

struct Fixture {
    /// Leaked, deliberately: `lowering_weight` caches device buffers by
    /// host ADDRESS, so bytes freed and reallocated at the same address
    /// would silently serve a stale buffer. Leaking for the test process
    /// lifetime makes every fixture's address unique.
    packed: &'static [u8],
    scales: &'static [u8],
    offs: Vec<u32>,
    soffs: Vec<u32>,
    x: Vec<f32>,
}

/// `slots` experts of `fused_rows` rows each, deterministic bytes.
fn fixture(fused_rows: usize, slots: usize, seed: u32) -> Fixture {
    let groups = K / 32;
    let per_p = fused_rows * groups * GROUP_BYTES;
    let per_s = fused_rows * groups;
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(99);
    let mut byte = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 24) as u8
    };
    Fixture {
        packed: Box::leak(
            (0..slots * per_p)
                .map(|_| byte())
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        ),
        scales: Box::leak(
            (0..slots * per_s)
                .map(|_| 125 + (byte() % 5))
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        ),
        offs: (0..slots).map(|s| (s * per_p) as u32).collect(),
        soffs: (0..slots).map(|s| (s * per_s) as u32).collect(),
        x: (0..K).map(|i| (i % 19) as f32 * 0.03 - 0.27).collect(),
    }
}

/// Run one arm over the fixture; `n` output rows per slot with the given
/// row walk into the fused bank.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &MetalBackend,
    x2: bool,
    f: &Fixture,
    n: usize,
    slots: usize,
    row_base: u32,
    row_stride: u32,
) -> Vec<f32> {
    let kh = if x2 {
        &gpu.quant.mxfp4_grouped_x2_pipeline
    } else {
        &gpu.quant.mxfp4_grouped_pipeline
    };
    let packed = gpu.lowering_weight(f.packed);
    let scales = gpu.lowering_weight(f.scales);
    // LEAKED, like `packed` and `scales` above, and for the same reason:
    // `lowering_weight` caches on `(ptr, len)`, which is only sound for
    // allocations that live for the process. Built into a temporary
    // `Vec` these two tables have the same length on every call, so the
    // allocator hands back the same address and a LATER iteration with a
    // different slot count silently reuses the FIRST one's buffer. The
    // debug assertion in `BufferCache::get_bytes` catches it; in a
    // release run it would simply compare two arms that both read the
    // wrong table.
    let offs = gpu.lowering_weight(Box::leak(
        f.offs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    ));
    let soffs = gpu.lowering_weight(Box::leak(
        f.soffs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    ));
    let xb = gpu.lowering_upload(&f.x).expect("x");
    let out = gpu.lowering_scratch(slots * n);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&kh.state);
    enc.set_buffer(0, Some(&packed), 0);
    enc.set_buffer(1, Some(&offs), 0);
    enc.set_buffer(2, Some(&scales), 0);
    enc.set_buffer(3, Some(&soffs), 0);
    enc.set_buffer(4, Some(&xb), 0);
    enc.set_buffer(5, Some(&out), 0);
    let set = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
    set(6, n as u32);
    set(7, K as u32);
    set(8, 0); // shared X
    set(9, row_base);
    set(10, row_stride);
    enc.dispatch_thread_groups(
        MTLSize::new((n as u64).div_ceil(kh.rows_per_tg), slots as u64, 1),
        MTLSize::new(kh.threads_per_tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got = gpu.lowering_readback(&out, slots * n).expect("readback");
    gpu.recycle_lowering_scratch(out);
    gpu.recycle_lowering_scratch(xb);
    got
}

#[test]
fn gu_fused_matches_two_x2_dispatches_bit_for_bit() {
    let gpu = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    // Interleaved walk (gpt-oss): gate (0,2), up (1,2) over 2*inter fused
    // rows; odd inter so the last logical row pair straddles the halves.
    let inter = 353;
    let slots = 4;
    let f = fixture(2 * inter, slots, 7);
    let want_g = run(&gpu, true, &f, inter, slots, 0, 2);
    let want_u = run(&gpu, true, &f, inter, slots, 1, 2);
    let khgu = &gpu.quant.mxfp4_grouped_x2_gu_pipeline;
    let packed = gpu.lowering_weight(f.packed);
    let scales = gpu.lowering_weight(f.scales);
    let offs_b: Vec<u8> = f.offs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let soffs_b: Vec<u8> = f.soffs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offs = gpu.lowering_weight(&offs_b);
    let soffs = gpu.lowering_weight(&soffs_b);
    let xb = gpu.lowering_upload(&f.x).expect("x");
    let out_g = gpu.lowering_scratch(slots * inter);
    let out_u = gpu.lowering_scratch(slots * inter);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&khgu.state);
    enc.set_buffer(0, Some(&packed), 0);
    enc.set_buffer(1, Some(&offs), 0);
    enc.set_buffer(2, Some(&scales), 0);
    enc.set_buffer(3, Some(&soffs), 0);
    enc.set_buffer(4, Some(&xb), 0);
    enc.set_buffer(5, Some(&out_g), 0);
    let set = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
    set(6, inter as u32);
    set(7, K as u32);
    set(8, 0);
    set(9, 0);
    set(10, 2);
    enc.set_buffer(11, Some(&out_u), 0);
    set(12, 1);
    set(13, 2);
    enc.dispatch_thread_groups(
        MTLSize::new(
            (2 * inter as u64).div_ceil(khgu.rows_per_tg),
            slots as u64,
            1,
        ),
        MTLSize::new(khgu.threads_per_tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got_g = gpu.lowering_readback(&out_g, slots * inter).expect("g");
    let got_u = gpu.lowering_readback(&out_u, slots * inter).expect("u");
    assert_eq!(want_g, got_g, "gate half diverged");
    assert_eq!(want_u, got_u, "up half diverged");
    gpu.recycle_lowering_scratch(out_g);
    gpu.recycle_lowering_scratch(out_u);
    gpu.recycle_lowering_scratch(xb);
}

#[test]
fn down_combine4_matches_down_then_combine_bit_for_bit() {
    let gpu = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    // Down geometry: N output rows (hidden), K = inter, 4 slots with
    // per-slot activations; odd N so the last threadgroup is ragged.
    let n = 353;
    let slots = 4;
    let f = fixture(n, slots, 11);
    // Per-slot activations, XSTRIDE = K.
    let acts: Vec<f32> = (0..slots * K)
        .map(|i| ((i % 23) as f32) * 0.04 - 0.4)
        .collect();
    let h: Vec<f32> = (0..n).map(|i| (i % 9) as f32 - 4.0).collect();
    let wroute = [0.4f32, 0.3, 0.2, 0.1];
    let bias: Vec<f32> = (0..slots * n).map(|i| (i % 5) as f32 * 0.01).collect();

    let packed = gpu.lowering_weight(f.packed);
    let scales = gpu.lowering_weight(f.scales);
    let offs_b: Vec<u8> = f.offs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let soffs_b: Vec<u8> = f.soffs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offs = gpu.lowering_weight(&offs_b);
    let soffs = gpu.lowering_weight(&soffs_b);
    let act_b = gpu.lowering_upload(&acts).expect("acts");
    let h_b = gpu.lowering_upload(&h).expect("h");
    let w_b = gpu.lowering_upload(&wroute).expect("w");
    let bias_b = gpu.lowering_upload(&bias).expect("bias");

    for has_bias in [0u32, 1] {
        // Reference: grouped x2 down per slot, then the combine on the CPU
        // in the combine kernel's exact order.
        let expert_outs = {
            let kh = &gpu.quant.mxfp4_grouped_x2_pipeline;
            let out = gpu.lowering_scratch(slots * n);
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&packed), 0);
            enc.set_buffer(1, Some(&offs), 0);
            enc.set_buffer(2, Some(&scales), 0);
            enc.set_buffer(3, Some(&soffs), 0);
            enc.set_buffer(4, Some(&act_b), 0);
            enc.set_buffer(5, Some(&out), 0);
            let set =
                |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
            set(6, n as u32);
            set(7, K as u32);
            set(8, K as u32); // per-slot activations
            set(9, 0);
            set(10, 1);
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(kh.rows_per_tg), slots as u64, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let got = gpu.lowering_readback(&out, slots * n).expect("outs");
            gpu.recycle_lowering_scratch(out);
            got
        };
        // Reference combine on the GPU — the production kernel, so FMA
        // contraction rounds identically (a CPU emulation differs at the
        // last ulp).
        let want = {
            let outs_b = gpu.lowering_upload(&expert_outs).expect("outs");
            let new_h = gpu.lowering_scratch(n);
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&gpu.ffn.moe_weighted_combine_pipeline);
            enc.set_buffer(0, Some(&outs_b), 0);
            enc.set_buffer(1, Some(&h_b), 0);
            enc.set_buffer(2, Some(&new_h), 0);
            let set =
                |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
            set(3, n as u32);
            set(4, slots as u32);
            enc.set_buffer(5, Some(&w_b), 0);
            enc.set_buffer(6, Some(&bias_b), 0);
            set(7, has_bias);
            enc.dispatch_threads(
                MTLSize::new(n as u64, 1, 1),
                MTLSize::new(256.min(n as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let w = gpu.lowering_readback(&new_h, n).expect("want");
            gpu.recycle_lowering_scratch(new_h);
            gpu.recycle_lowering_scratch(outs_b);
            w
        };

        let khdc = &gpu.quant.mxfp4_down_combine4_pipeline;
        let new_h = gpu.lowering_scratch(n);
        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&khdc.state);
        enc.set_buffer(0, Some(&packed), 0);
        enc.set_buffer(1, Some(&offs), 0);
        enc.set_buffer(2, Some(&scales), 0);
        enc.set_buffer(3, Some(&soffs), 0);
        enc.set_buffer(4, Some(&act_b), 0);
        enc.set_buffer(5, Some(&new_h), 0);
        let set = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
        set(6, n as u32);
        set(7, K as u32);
        set(8, K as u32);
        enc.set_buffer(9, Some(&h_b), 0);
        enc.set_buffer(10, Some(&w_b), 0);
        enc.set_buffer(11, Some(&bias_b), 0);
        set(12, has_bias);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(khdc.rows_per_tg), 1, 1),
            MTLSize::new(khdc.threads_per_tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let got = gpu.lowering_readback(&new_h, n).expect("new_h");
        assert_eq!(want, got, "down+combine diverged (has_bias={has_bias})");
        gpu.recycle_lowering_scratch(new_h);
    }
    gpu.recycle_lowering_scratch(act_b);
    gpu.recycle_lowering_scratch(h_b);
    gpu.recycle_lowering_scratch(w_b);
    gpu.recycle_lowering_scratch(bias_b);
}

#[test]
fn x2_matches_the_vec_arm_bit_for_bit_across_walks_and_slots() {
    let gpu = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    // (output rows, slots, row_base, row_stride): identity walk on an odd
    // row count, both interleaved halves, contiguous-halves up half.
    let inter = 353; // odd, not a multiple of 8
    for (n, slots, base, stride, fused_rows, seed) in [
        (inter, 4, 0u32, 1u32, inter, 1u32), // identity, odd N
        (256, 3, 0, 2, 512, 2),              // interleaved gate
        (256, 3, 1, 2, 512, 3),              // interleaved up
        (128, 1, 128, 1, 256, 4),            // contiguous up half
        (8, 2, 0, 1, 8, 5),                  // tiny N == rows_per_tg
    ] {
        let f = fixture(fused_rows, slots, seed);
        let want = run(&gpu, false, &f, n, slots, base, stride);
        let got = run(&gpu, true, &f, n, slots, base, stride);
        assert_eq!(
            want, got,
            "x2 diverged: n={n} slots={slots} base={base} stride={stride}"
        );
        assert!(want.iter().all(|v| v.is_finite()));
    }
}

// ── VERIFY-N rung 5b: the multi arm over a block's distinct experts ───────

/// A 3-row × top-4 block over 6 experts: every row picks 4 distinct
/// experts, and experts repeat across rows (2 and 4 three times each).
const BLOCK_IDS: [u32; 12] = [0, 2, 4, 5, 2, 1, 4, 0, 3, 2, 5, 4];
const BLOCK_TOP_K: usize = 4;
const BLOCK_EXPERTS: usize = 6;

/// The host's reading of `moe_slots_by_expert`: distinct experts in id
/// order, each with its slots in slot order.
fn host_slots_by_expert(ids: &[u32], experts: usize, maxm: usize) -> (Vec<u32>, Vec<u32>) {
    let s = ids.len();
    let (mut counts, mut lists) = (vec![0u32; s], vec![0u32; s * maxm]);
    let mut u = 0;
    for e in 0..experts as u32 {
        let slots: Vec<u32> = (0..s as u32).filter(|&i| ids[i as usize] == e).collect();
        if slots.is_empty() {
            continue;
        }
        counts[u] = slots.len() as u32;
        lists[u * maxm..u * maxm + slots.len()].copy_from_slice(&slots);
        u += 1;
    }
    (counts, lists)
}

fn leak_u32(gpu: &MetalBackend, v: &[u32]) -> metal::Buffer {
    gpu.lowering_weight(Box::leak(
        v.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    ))
}

#[test]
fn slots_by_expert_groups_a_block_as_the_host_does() {
    let gpu = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let maxm = BLOCK_IDS.len() / BLOCK_TOP_K;
    let (want_c, want_l) = host_slots_by_expert(&BLOCK_IDS, BLOCK_EXPERTS, maxm);
    let ids = leak_u32(&gpu, &BLOCK_IDS);
    let counts = gpu.lowering_scratch(BLOCK_IDS.len());
    let lists = gpu.lowering_scratch(BLOCK_IDS.len() * maxm);
    let order = gpu.lowering_scratch(BLOCK_IDS.len());
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gpu.ffn.moe_slots_by_expert_pipeline);
    enc.set_buffer(0, Some(&ids), 0);
    enc.set_buffer(1, Some(&counts), 0);
    enc.set_buffer(2, Some(&lists), 0);
    let set = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
    set(3, BLOCK_IDS.len() as u32);
    set(4, BLOCK_EXPERTS as u32);
    set(5, maxm as u32);
    enc.set_buffer(6, Some(&order), 0);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let bits = |b: &metal::Buffer, n| -> Vec<u32> {
        gpu.lowering_readback(b, n)
            .expect("readback")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(bits(&counts, BLOCK_IDS.len()), want_c, "counts");
    let got_l = bits(&lists, BLOCK_IDS.len() * maxm);
    for (u, &c) in want_c.iter().enumerate() {
        let span = u * maxm..u * maxm + c as usize;
        assert_eq!(
            got_l[span.clone()],
            want_l[span],
            "distinct expert {u}'s slots"
        );
    }
    // `order`: every slot, sorted by (expert, slot) — the lists concatenated.
    let want_order: Vec<u32> = want_c
        .iter()
        .enumerate()
        .flat_map(|(u, &c)| want_l[u * maxm..u * maxm + c as usize].to_vec())
        .collect();
    assert_eq!(
        bits(&order, BLOCK_IDS.len()),
        want_order,
        "expert-sorted slot order"
    );
}

#[test]
fn the_sharing_arms_match_x2_per_slot_bit_for_bit() {
    let gpu = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let slots = BLOCK_IDS.len();
    let maxm = slots / BLOCK_TOP_K;
    let (counts_h, lists_h) = host_slots_by_expert(&BLOCK_IDS, BLOCK_EXPERTS, maxm);
    // (output rows, row_base, row_stride, fused rows): odd N on the
    // identity walk, and both interleaved halves.
    for (n, base, stride, fused_rows, seed) in [
        (353usize, 0u32, 1u32, 353usize, 11u32),
        (256, 0, 2, 512, 12),
        (256, 1, 2, 512, 13),
    ] {
        let f = fixture(fused_rows, BLOCK_EXPERTS, seed);
        let packed = gpu.lowering_weight(f.packed);
        let scales = gpu.lowering_weight(f.scales);
        // Each slot carries its expert's offsets, as the descriptor gather
        // leaves them.
        let offs: Vec<u32> = BLOCK_IDS.iter().map(|&e| f.offs[e as usize]).collect();
        let soffs: Vec<u32> = BLOCK_IDS.iter().map(|&e| f.soffs[e as usize]).collect();
        let (offs, soffs) = (leak_u32(&gpu, &offs), leak_u32(&gpu, &soffs));
        // A different input per slot, `K` apart.
        let x: Vec<f32> = (0..slots * K)
            .map(|i| ((i % 23) as f32 * 0.021 - 0.2) * (1.0 + (i / K) as f32 * 0.1))
            .collect();
        let xb = gpu.lowering_upload(&x).expect("x");
        let (counts, lists) = (leak_u32(&gpu, &counts_h), leak_u32(&gpu, &lists_h));

        let order_h: Vec<u32> = counts_h
            .iter()
            .enumerate()
            .flat_map(|(u, &c)| lists_h[u * maxm..u * maxm + c as usize].to_vec())
            .collect();
        let order = leak_u32(&gpu, &order_h);
        let mut outs = Vec::new();
        for arm in ["x2", "multi", "sorted"] {
            let kh = match arm {
                "multi" => &gpu.quant.mxfp4_grouped_x2_multi_pipeline,
                "sorted" => &gpu.quant.mxfp4_grouped_x2_sorted_pipeline,
                _ => &gpu.quant.mxfp4_grouped_x2_pipeline,
            };
            let out = gpu.lowering_scratch(slots * n);
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&packed), 0);
            enc.set_buffer(1, Some(&offs), 0);
            enc.set_buffer(2, Some(&scales), 0);
            enc.set_buffer(3, Some(&soffs), 0);
            enc.set_buffer(4, Some(&xb), 0);
            enc.set_buffer(5, Some(&out), 0);
            let set =
                |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
            set(6, n as u32);
            set(7, K as u32);
            set(8, K as u32); // one input per slot
            set(9, base);
            set(10, stride);
            let tiles = (n as u64).div_ceil(kh.rows_per_tg);
            let grid = match arm {
                "multi" => {
                    enc.set_buffer(11, Some(&counts), 0);
                    enc.set_buffer(12, Some(&lists), 0);
                    set(13, maxm as u32);
                    MTLSize::new(tiles, slots as u64, 1)
                }
                "sorted" => {
                    enc.set_buffer(11, Some(&order), 0);
                    MTLSize::new(slots as u64, tiles, 1)
                }
                _ => MTLSize::new(tiles, slots as u64, 1),
            };
            enc.dispatch_thread_groups(grid, MTLSize::new(kh.threads_per_tg, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            outs.push(gpu.lowering_readback(&out, slots * n).expect("readback"));
        }
        for (arm, got) in ["multi", "sorted"].iter().zip(&outs[1..]) {
            let first = outs[0]
                .iter()
                .zip(got)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                first, None,
                "{arm} diverged from x2: n={n} base={base} stride={stride}"
            );
        }
        assert!(outs[0].iter().all(|v| v.is_finite()));
    }
}
