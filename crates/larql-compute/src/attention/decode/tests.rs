use super::q4k_direct::{q4k_direct_proj, q8k_direct_proj};
use super::*;
use crate::attention::SharedKV;
use larql_models::test_fixtures::make_test_weights;
use ndarray::Array2;

/// The rayon row-chunking in `q8k_direct_proj` must be bit-exact with one
/// whole-matrix kernel call (chunk boundaries change nothing), for both
/// Q4_K and Q6_K formats.
#[test]
fn q8k_direct_proj_chunking_is_bit_exact() {
    use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    use crate::cpu::ops::q4k_q8k_dot::{
        q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k,
    };

    let in_dim = 512usize;
    let num_rows = 70usize; // not a multiple of CHUNK_ROWS=32 — exercises the tail
    let w_f32: Vec<f32> = (0..num_rows * in_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.3)
        .collect();
    let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.017).cos()).collect();
    let q8 = quantize_x_to_q8k(&x);

    for fmt in [crate::QuantFormat::Q4_K, crate::QuantFormat::Q6_K] {
        let bytes = match fmt {
            crate::QuantFormat::Q4_K => quantize_q4_k(&w_f32),
            _ => quantize_q6_k(&w_f32),
        };
        let qw = crate::QuantWeight::new(fmt, &bytes, crate::QuantAux::None);
        let chunked = q8k_direct_proj(&qw, &q8, num_rows, in_dim).expect("q8k proj must run");

        let mut whole = vec![0.0f32; num_rows];
        match fmt {
            crate::QuantFormat::Q4_K => {
                q4k_q8k_matvec_into(&mut whole, &q8, &bytes, num_rows, in_dim)
            }
            _ => q6k_q8k_matvec_into(&mut whole, &q8, &bytes, num_rows, in_dim),
        }
        .expect("whole-matrix matvec on a valid shape");
        for (r, (&c, &w)) in chunked.iter().zip(whole.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                w.to_bits(),
                "{fmt:?} row {r}: chunked={c} whole={w}"
            );
        }
    }
}

/// Int8 projections vs the f32-activation projection: same weights, same
/// input — outputs agree within activation-quantisation tolerance (the
/// int8 route adds ONLY the Q8_K activation quant the production dense
/// attention path already carries; weight quant is identical bytes).
#[test]
fn q8k_direct_proj_matches_f32_activation_within_quant_tolerance() {
    use crate::cpu::ops::q4_common::quantize_q4_k;
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;

    let in_dim = 512usize;
    let num_rows = 48usize;
    let w_f32: Vec<f32> = (0..num_rows * in_dim)
        .map(|i| ((i as f32) * 0.011).sin() * 0.3)
        .collect();
    let bytes = quantize_q4_k(&w_f32);
    let qw = crate::QuantWeight::new(crate::QuantFormat::Q4_K, &bytes, crate::QuantAux::None);
    let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.017).cos()).collect();

    let q8 = quantize_x_to_q8k(&x);
    let int8_out = q8k_direct_proj(&qw, &q8, num_rows, in_dim).expect("int8 proj");

    let backend = crate::CpuBackend;
    let x_arr = Array2::from_shape_vec((1, in_dim), x).unwrap();
    let f32_out = q4k_direct_proj(&backend, &qw, &x_arr, num_rows, in_dim).expect("f32-act proj");

    // Scale-relative bound: Q8_K activation quant is ~1/255 per block
    // value; accumulated over 512 terms the practical error is well
    // under 1% of the output magnitude.
    let denom = f32_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for (r, (&a, &b)) in int8_out.iter().zip(f32_out.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 0.02 * denom.max(1e-3),
            "row {r}: int8={a} f32act={b} denom={denom}"
        );
    }
}

#[test]
fn decode_step_output_shape() {
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
    let (h_out, (k, v)) =
        run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
    assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
    assert_eq!(k.shape()[0], 1, "K should have 1 new row");
    assert_eq!(v.shape()[0], 1, "V should have 1 new row");
}

#[test]
fn decode_step_output_finite() {
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
    let (h_out, _) =
        run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
    assert!(h_out.iter().all(|v| v.is_finite()));
}

#[test]
fn decode_step_kv_grows_with_prior() {
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
    let (_, kv1) = run_attention_block_decode_step(&weights, &h, 0, None, 0).unwrap();
    assert_eq!(kv1.0.shape()[0], 1);
    let (_, kv2) = run_attention_block_decode_step(&weights, &h, 0, Some(&kv1), 1).unwrap();
    assert_eq!(kv2.0.shape()[0], 2, "K should grow by 1 per step");
}

#[test]
fn decode_step_all_layers_succeed() {
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.3f32);
    for layer in 0..weights.num_layers {
        let result = run_attention_block_decode_step(&weights, &h, layer, None, 0);
        assert!(result.is_some(), "layer {layer} decode step failed");
    }
}

#[test]
fn gqa_decode_step_applies_softcap() {
    // Drive the `Some(cap)` softcap branch in `gqa_attention_decode_step`
    // (otherwise dead under the default test models, none of which set
    // attention logit softcapping). With a single cached position the
    // softmax is degenerate (one weight = 1), so the output equals the
    // V row regardless of the (capped) score — the assertion just pins
    // a finite result while the cap path executes.
    let q = Array2::from_elem((1, 4), 0.5f32); // num_q=1, head_dim=4
    let k = Array2::from_elem((1, 4), 0.25f32);
    let v = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = gqa_attention_decode_step(&q, &k, &v, 1, 4, 1, 1.0, Some(30.0), None);
    assert_eq!(out.shape(), &[1, 4]);
    // Single key → softmax weight 1.0 → output is exactly the V row.
    for d in 0..4 {
        assert!((out[[0, d]] - v[[0, d]]).abs() < 1e-5);
    }
    assert!(out.iter().all(|x| x.is_finite()));
}

// ── Spin-pool vs rayon scheduling parity ───────────────────────────

/// Build a decode-step input set big enough that the two schedulers
/// genuinely partition the heads differently (the spin pool hands each
/// participant a contiguous block; rayon work-steals per chunk).
/// Deterministic, so the two runs are comparable bit-for-bit.
fn parity_inputs() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    const NUM_Q: usize = 16;
    const HEAD_DIM: usize = 8;
    const TOTAL_LEN: usize = 24;
    let q = Array2::from_shape_fn((1, NUM_Q * HEAD_DIM), |(_, i)| {
        ((i % 13) as f32 - 6.0) * 0.125
    });
    let k = Array2::from_shape_fn((TOTAL_LEN, NUM_Q * HEAD_DIM), |(t, i)| {
        ((t * 7 + i) % 11) as f32 * 0.0625 - 0.25
    });
    let v = Array2::from_shape_fn((TOTAL_LEN, NUM_Q * HEAD_DIM), |(t, i)| {
        ((t * 3 + i * 5) % 17) as f32 * 0.03125
    });
    (q, k, v)
}

/// The spin-pool and rayon paths must produce bit-identical output.
///
/// `gqa_attention_decode_step` branches on `spin_pool::enabled()` and
/// the two arms share one `head_body`, so the claim in that comment —
/// "the result is identical to the previous serial loop" — is only
/// actually checked here. It also matters for coverage: CI runs with a
/// single ambient setting, so whichever arm isn't the default is dead
/// unless a test drives both.
///
/// Uses the thread-local override rather than `std::env::set_var`,
/// which segfaults under parallel `getenv` (see `options::ENV_OVERRIDES`).
#[test]
fn spin_pool_and_rayon_paths_agree_bitwise() {
    use crate::options::{clear_fast_path_overrides, set_fast_path_override, ENV_SPIN_POOL};

    let (q, k, v) = parity_inputs();
    let run = || gqa_attention_decode_step(&q, &k, &v, 16, 8, 1, 0.35, None, None);

    set_fast_path_override(ENV_SPIN_POOL, true);
    let spin = run();
    set_fast_path_override(ENV_SPIN_POOL, false);
    let rayon = run();
    clear_fast_path_overrides();

    assert_eq!(spin.shape(), rayon.shape());
    assert!(spin.iter().all(|x| x.is_finite()), "spin path produced NaN");
    for (i, (s, r)) in spin.iter().zip(rayon.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            r.to_bits(),
            "element {i}: spin {s} != rayon {r} — the two schedulers must not \
             change the arithmetic, only which thread runs which head"
        );
    }
}

/// Both arms must also agree when sinks are in play — the sink is
/// indexed per-head (`sinks[h]`), so a scheduler that mis-assigns `h`
/// would silently apply the wrong head's sink.
#[test]
fn spin_pool_and_rayon_paths_agree_with_attention_sinks() {
    use crate::options::{clear_fast_path_overrides, set_fast_path_override, ENV_SPIN_POOL};

    let (q, k, v) = parity_inputs();
    // Distinct per-head values so a swapped head index changes the result.
    let sinks: Vec<f32> = (0..16).map(|h| h as f32 * 0.5 - 3.0).collect();
    let run = || gqa_attention_decode_step(&q, &k, &v, 16, 8, 1, 0.35, None, Some(&sinks));

    set_fast_path_override(ENV_SPIN_POOL, true);
    let spin = run();
    set_fast_path_override(ENV_SPIN_POOL, false);
    let rayon = run();
    clear_fast_path_overrides();

    for (i, (s, r)) in spin.iter().zip(rayon.iter()).enumerate() {
        assert_eq!(s.to_bits(), r.to_bits(), "element {i} diverged under sinks");
    }
}

// ── Q4K-direct decode-step path ────────────────────────────────────
//
// These exercise `run_attention_block_decode_step_q4k_direct` (and its
// `q4k_direct_proj` helper) directly — no env var needed, the function
// is purely argument-driven. The `LARQL_Q4K_DIRECT_ATTN` gate lives in
// `kv_dispatch::cpu::attention_step`, not here.

use crate::test_fixtures::make_q4k_fixture_index;
use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_silu};

#[test]
fn q4k_direct_decode_step_first_token_shape_and_finite() {
    // First decode step (kv_entry None → the `None` concat arm).
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
    let (h_out, (k, v)) =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
            .expect("q4k-direct decode step returns Some on the Q4K fixture");
    assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
    assert_eq!(k.shape()[0], 1, "K grows by 1 (first token)");
    assert_eq!(v.shape()[0], 1, "V grows by 1 (first token)");
    assert!(h_out.iter().all(|x| x.is_finite()));
}

#[test]
fn q4k_direct_decode_step_grows_kv_with_prior() {
    // Second decode step (kv_entry Some → the cache-concat arm that
    // copies prior K/V then appends the new row).
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
    let (_, kv1) =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
            .unwrap();
    assert_eq!(kv1.0.shape()[0], 1);
    let (h2, kv2) =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, Some(&kv1), 1, &backend, &idx)
            .expect("second q4k-direct step succeeds");
    assert_eq!(kv2.0.shape()[0], 2, "K grows by 1 per step");
    assert_eq!(kv2.1.shape()[0], 2, "V grows by 1 per step");
    assert!(h2.iter().all(|x| x.is_finite()));
}

/// The in-place form must be **bit-identical** to the owned-concat form
/// across a multi-step decode: same h_post_attn every step, and the
/// doubling-capacity buffer's `[..len]` view must equal the concat's owned
/// K/V. This is the parity gate that lets the walk engines drop the O(ctx)
/// concat. Runs a real multi-layer Q4K fixture for several steps so the
/// buffer crosses a capacity doubling.
#[test]
fn q4k_direct_inplace_is_bit_identical_to_owned_concat() {
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let num_layers = weights.num_layers;
    let kv_dim = {
        let arch = &*weights.arch;
        arch.num_kv_heads_for_layer(0) * arch.head_dim_for_layer(0)
    };

    // Concat-path cache: one owned SharedKV per layer (grows by concat).
    let mut concat_kv: Vec<Option<SharedKV>> = vec![None; num_layers];
    // In-place cache: doubling-capacity buffers per layer + a logical length.
    let mut inplace_k: Vec<Array2<f32>> = (0..num_layers)
        .map(|_| Array2::zeros((0, kv_dim)))
        .collect();
    let mut inplace_v: Vec<Array2<f32>> = (0..num_layers)
        .map(|_| Array2::zeros((0, kv_dim)))
        .collect();

    for step in 0..6 {
        // The buffer's logical length at the start of this step == `step`.
        let len = step;
        let h = Array2::from_elem((1, weights.hidden_size), 0.05 * (step as f32 + 1.0));
        for layer in 0..num_layers {
            let (h_concat, new_kv) = run_attention_block_decode_step_q4k_direct(
                &weights,
                &h,
                layer,
                concat_kv[layer].as_ref(),
                step,
                &backend,
                &idx,
            )
            .expect("concat step");

            let h_inplace = run_attention_block_decode_step_q4k_direct_inplace(
                &weights,
                &h,
                layer,
                &mut inplace_k[layer],
                &mut inplace_v[layer],
                len,
                step,
                &backend,
                &idx,
            )
            .expect("inplace step");

            // h_post_attn must match bit-for-bit.
            for (a, b) in h_concat.iter().zip(h_inplace.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "h_post_attn diverged step {step} layer {layer}"
                );
            }
            // The in-place buffer's logical view must equal the concat K/V.
            let total = len + 1;
            let k_view = inplace_k[layer].slice(ndarray::s![..total, ..]);
            let v_view = inplace_v[layer].slice(ndarray::s![..total, ..]);
            assert_eq!(
                new_kv.0.shape(),
                k_view.shape(),
                "K shape step {step} layer {layer}"
            );
            for (a, b) in new_kv.0.iter().zip(k_view.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "K diverged step {step} layer {layer}"
                );
            }
            for (a, b) in new_kv.1.iter().zip(v_view.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "V diverged step {step} layer {layer}"
                );
            }
            concat_kv[layer] = Some(new_kv);
        }
    }
    // Buffer must have grown past its first allocation (crossed a doubling).
    assert!(
        inplace_k[0].shape()[0] >= 6,
        "buffer should have grown to hold 6 rows"
    );
}

#[test]
fn q4k_direct_decode_step_all_layers_succeed() {
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.2f32);
    for layer in 0..weights.num_layers {
        let result = run_attention_block_decode_step_q4k_direct(
            &weights, &h, layer, None, 0, &backend, &idx,
        );
        assert!(result.is_some(), "q4k-direct layer {layer} step failed");
    }
}

#[test]
fn q4k_direct_decode_step_runs_on_non_post_norm_arch() {
    // The SiLU (TinyModel) fixture has `has_post_norms() == false`, so
    // the post-attention residual takes the non-post-norm branch
    // (`h_new + &attn_projected`) rather than the Gemma post-norm arm.
    let weights = make_test_q4k_weights_silu();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    assert!(
        !weights.arch.has_post_norms(),
        "TinyModel fixture should be a non-post-norm arch for this branch"
    );
    let h = Array2::from_elem((1, weights.hidden_size), 0.15f32);
    let (h_out, (k, _v)) =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
            .expect("q4k-direct decode step on SiLU fixture");
    assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
    assert_eq!(k.shape()[0], 1);
    assert!(h_out.iter().all(|x| x.is_finite()));
}

#[test]
fn q4k_direct_decode_step_falls_back_to_none_without_q4k_attn_bytes() {
    // An index with no Q4K attention bytes → `resolve_attn_weights`
    // returns None → the function short-circuits to None (the caller's
    // f32-fallback trigger). Covers the `?` on `resolve_attn_weights`.
    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}
    let weights = make_test_q4k_weights();
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
    let result =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &EmptyIdx);
    assert!(result.is_none(), "no Q4K attn bytes → None");
}

#[test]
fn q4k_direct_decode_step_matches_dequant_path_within_tolerance() {
    // This pins the strict <1e-3 WEIGHT parity of the *f32-activation*
    // Q4K-direct path. The int8 activation route is now on by default and
    // carries a looser (~2% scale-relative) bound by design, so disable it
    // here. Thread-local override (NOT `set_var`, which races concurrent
    // `getenv` on the decode path → SIGSEGV); cleared on drop.
    let _guard = crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_ATTN_INT8, false)]);

    // Parity contract (roadmap #16, "<1e-3"): the Q4K-direct decode
    // step should track the f32-BLAS path that runs on the SAME bytes
    // dequantised into `weights.tensors`. We dequantise the fixture's
    // Q4K attn slices into the weights (what `insert_q4k_layer_tensors`
    // does) and compare against `run_attention_block_decode_step_backend`.
    let weights = make_test_q4k_weights();
    let mut scratch = larql_models::DequantScratch::new();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    // Q4K-direct output (reads packed bytes from the index).
    let (h_direct, _) =
        run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
            .expect("q4k-direct step");

    // Dequant the index's layer-0 attn/ffn bytes into weights.tensors,
    // then run the f32 path against those same (dequantised) weights.
    let inserted = crate::kquant_forward::insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
        .expect("dequant layer 0 tensors");
    // The dequantised attn tensors live in `scratch` (not `weights.tensors`);
    // resolve them via `with_scratch` so the f32 path reads the SAME bytes
    // the Q4K-direct path read from the index.
    let (h_dequant, _) = run_attention_block_decode_step_backend(
        larql_models::WeightsView::with_scratch(&weights, &scratch),
        &h,
        0,
        None,
        0,
        Some(&backend),
    )
    .expect("f32 dequant step");
    let _ = inserted;

    assert_eq!(h_direct.shape(), h_dequant.shape());
    let mut max_abs = 0.0f32;
    for (a, b) in h_direct.iter().zip(h_dequant.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-3,
        "q4k-direct vs dequant decode drift {max_abs} exceeds 1e-3 parity bound"
    );
}

// ── auto-dispatcher gating (`run_attention_block_decode_step_auto[_inplace]`)
//
// These exercise the flag-driven choice between the Q4K-direct path and the
// f32 fallback. The gate (`q4k_direct_attn_enabled`) reads through the
// thread-local override, so `FastPathGuard` flips it deterministically
// without `set_var` (which races `getenv` on the decode path → SIGSEGV).

#[test]
fn auto_decode_step_takes_q4k_direct_when_flag_enabled() {
    let _guard = crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, true)]);
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    let (out, _kv) = run_attention_block_decode_step_auto(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        None,
        0,
        Some(&backend),
        Some(&idx),
    )
    .expect("flag on + Q4K index → Q4K-direct path returns Some");
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn auto_decode_step_falls_back_to_f32_when_flag_disabled() {
    let _guard =
        crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, false)]);
    let weights = make_test_q4k_weights();
    let mut scratch = larql_models::DequantScratch::new();
    let idx = make_q4k_fixture_index(&weights);
    // The f32 backend path reads its `WeightsView`; dequant the fixture's
    // Q4K layer-0 bytes into `scratch` so the fallback produces a real
    // result (resolved via `with_scratch` below).
    crate::kquant_forward::insert_q4k_layer_tensors(&mut scratch, &weights, &idx, 0)
        .expect("dequant layer 0");
    let backend = crate::CpuBackend;
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    // Flag off → Q4K branch skipped → `run_attention_block_decode_step_backend`.
    let (out, _kv) = run_attention_block_decode_step_auto(
        larql_models::WeightsView::with_scratch(&weights, &scratch),
        &h,
        0,
        None,
        0,
        Some(&backend),
        Some(&idx),
    )
    .expect("flag off → f32 fallback returns Some");
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn auto_inplace_decode_step_takes_q4k_direct_when_flag_enabled() {
    let _guard = crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, true)]);
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let kv_dim = {
        let arch = &*weights.arch;
        arch.num_kv_heads_for_layer(0) * arch.head_dim_for_layer(0)
    };
    let mut k_cache = Array2::<f32>::zeros((0, kv_dim));
    let mut v_cache = Array2::<f32>::zeros((0, kv_dim));
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    let out = run_attention_block_decode_step_auto_inplace(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &mut k_cache,
        &mut v_cache,
        0,
        0,
        Some(&backend),
        Some(&idx),
    )
    .expect("flag on + Q4K index → in-place Q4K-direct returns Some");
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    // The in-place buffer grew (doubling-capacity) to hold the appended row.
    assert!(
        k_cache.shape()[0] >= 1,
        "in-place K buffer must hold the new row"
    );
}

#[test]
fn auto_inplace_decode_step_returns_none_when_flag_disabled() {
    let _guard =
        crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, false)]);
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;
    let kv_dim = {
        let arch = &*weights.arch;
        arch.num_kv_heads_for_layer(0) * arch.head_dim_for_layer(0)
    };
    let mut k_cache = Array2::<f32>::zeros((0, kv_dim));
    let mut v_cache = Array2::<f32>::zeros((0, kv_dim));
    let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    // Flag off → returns None so the caller uses the owned-concat path.
    let out = run_attention_block_decode_step_auto_inplace(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &mut k_cache,
        &mut v_cache,
        0,
        0,
        Some(&backend),
        Some(&idx),
    );
    assert!(out.is_none(), "flag off → None");
}

/// The padded-row-store case the width derivation exists for: rows are
/// quantised at the next super-block boundary (the writer's
/// `pad_rows_to_block` shape — GPT-OSS's hidden 2880 → 3072 at model
/// scale), the activation is zero-padded to match, and the projection at
/// the STORED width must equal the dequant reference over the LOGICAL
/// width exactly-ish (f32 route) — the padded weight columns dequantise
/// to zero, so they contribute nothing.
#[test]
fn direct_proj_on_padded_row_store_matches_dequant_reference() {
    use super::q4k_direct::zero_pad_row;
    use crate::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};
    use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;

    let logical = 320usize; // % 256 != 0 — the awkward shape is the instrument
    let stored = logical.div_ceil(256) * 256; // 512
    let num_rows = 24usize;

    // Rows laid out at the stored width, zeros in the pad columns.
    let mut w = vec![0.0f32; num_rows * stored];
    for r in 0..num_rows {
        for c in 0..logical {
            w[r * stored + c] = (((r * 7 + c) % 23) as f32) * 0.05 - 0.5;
        }
    }
    let bytes = quantize_q4_k(&w);
    let qw = crate::QuantWeight::new(crate::QuantFormat::Q4_K, &bytes, crate::QuantAux::None);
    assert_eq!(qw.stored_cols(num_rows, logical), stored);

    let x: Vec<f32> = (0..logical).map(|i| ((i as f32) * 0.013).sin()).collect();
    let x_arr = Array2::from_shape_vec((1, logical), x.clone()).unwrap();
    let x_padded = zero_pad_row(&x_arr, stored);
    assert_eq!(x_padded.shape(), &[1, stored]);
    assert_eq!(x_padded[[0, logical]], 0.0);

    // Reference: dequantised stored rows · padded x over the stored width
    // (equals the logical-width product because the pad columns are 0).
    let deq = dequantize_q4_k(&bytes, num_rows * stored);
    let reference: Vec<f32> = (0..num_rows)
        .map(|r| {
            (0..stored)
                .map(|c| deq[r * stored + c] * x_padded[[0, c]])
                .sum()
        })
        .collect();

    // f32-activation route at the stored width.
    let backend = crate::CpuBackend;
    let out = q4k_direct_proj(&backend, &qw, &x_padded, num_rows, stored)
        .expect("padded-store proj must run");
    for (r, (&got, &want)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "f32 route row {r}: got {got}, want {want}"
        );
    }

    // Int8 route (Q8_K activation): the stored width is a 256-multiple,
    // so the fast path engages; agreement within Q8_K activation noise.
    let q8 = quantize_x_to_q8k(x_padded.as_slice().unwrap());
    let out8 = q8k_direct_proj(&qw, &q8, num_rows, stored).expect("int8 padded proj must run");
    for (r, (&got, &want)) in out8.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 2e-2 * want.abs().max(1.0),
            "int8 route row {r}: got {got}, want {want}"
        );
    }
}
