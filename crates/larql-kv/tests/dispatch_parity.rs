//! Bit-parity gate: dispatch helpers vs legacy generation loops.
//!
//! The `KvDispatch` helpers in `larql-inference` (`kv_prefill_via_dispatch`,
//! `kv_decode_step_via_dispatch`) must produce bit-identical output to the
//! legacy `kv_prefill_run` / `kv_decode_step_run` reference paths in
//! `larql_kv::generation` when both are driven against `CpuBackend` (whose
//! `KvDispatch` impl delegates to the same underlying functions).
//!
//! Lifted out of `larql-inference/src/kv_dispatch/helpers.rs::tests` in
//! 2026-05-16: the dual import (legacy from `larql-kv`, helpers from
//! `larql-inference`) forced cargo to compile `larql-inference` twice
//! when the parity tests lived inside the lib crate's `#[cfg(test)]`.
//!
//! BLAS on Windows has non-deterministic reduction order across
//! successive matmul calls (parallel accumulation), so bit-for-bit
//! parity between two dispatch paths doesn't hold there. Linux/macOS
//! BLAS is deterministic and the property holds; gate the whole file
//! on `not(windows)` rather than weaken to a fuzzy tolerance that
//! wouldn't catch real bugs.

#![cfg(not(windows))]

use larql_compute::CpuBackend;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::NoopHook;
use larql_inference::kv_dispatch::helpers::{kv_decode_step_via_dispatch, kv_prefill_via_dispatch};
use larql_inference::test_utils::make_test_weights;
use larql_kv::generation::{kv_decode_step_run, kv_prefill_run};

#[test]
fn prefill_via_dispatch_matches_legacy_kv_prefill_run() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3];

    let (h_trait, _handles) =
        kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).expect("prefill");
    let (h_legacy, _cache) =
        kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook)
            .expect("legacy prefill");

    assert_eq!(
        h_trait, h_legacy,
        "prefill_via_dispatch hidden must match legacy bit-for-bit"
    );
}

#[test]
fn prefill_via_dispatch_windowed_matches_legacy() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4];
    let window = Some(2);

    let (h_trait, _handles) =
        kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, window, None).expect("prefill");
    let (h_legacy, _cache) = kv_prefill_run(
        &weights,
        &ffn,
        &prompt,
        window,
        Some(&backend),
        &mut NoopHook,
    )
    .expect("legacy prefill");

    assert_eq!(
        h_trait, h_legacy,
        "windowed prefill_via_dispatch must match legacy bit-for-bit"
    );
}

#[test]
fn decode_step_via_dispatch_matches_legacy_kv_decode_step_run() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2];

    let (_, mut handles) =
        kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();
    let (_, mut cache) =
        kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook).unwrap();

    let next_token = 3u32;
    let abs_position = prompt.len();

    let h_trait = kv_decode_step_via_dispatch(
        &backend,
        &weights,
        &ffn,
        &mut handles,
        next_token,
        abs_position,
        None,
        None,
    )
    .expect("decode step trait");

    let h_legacy = kv_decode_step_run(
        &weights,
        &ffn,
        &mut cache,
        next_token,
        Some(&backend),
        &mut NoopHook,
    )
    .expect("legacy decode step");

    assert_eq!(
        h_trait, h_legacy,
        "decode_step_via_dispatch must match legacy bit-for-bit"
    );
}

#[test]
fn multi_step_decode_via_dispatch_matches_legacy() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1];

    let (_, mut handles) =
        kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();
    let (_, mut cache) =
        kv_prefill_run(&weights, &ffn, &prompt, None, Some(&backend), &mut NoopHook).unwrap();

    for step in 0..3 {
        let token = (2 + step) as u32;
        let abs_position = prompt.len() + step;
        let h_trait = kv_decode_step_via_dispatch(
            &backend,
            &weights,
            &ffn,
            &mut handles,
            token,
            abs_position,
            None,
            None,
        )
        .expect("decode trait");
        let h_legacy = kv_decode_step_run(
            &weights,
            &ffn,
            &mut cache,
            token,
            Some(&backend),
            &mut NoopHook,
        )
        .expect("decode legacy");
        assert_eq!(
            h_trait, h_legacy,
            "step {step} hidden must match legacy bit-for-bit"
        );
    }
}
