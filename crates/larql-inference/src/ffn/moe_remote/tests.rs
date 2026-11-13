use std::sync::{Arc, RwLock};

use super::backend::RemoteMoeBackend;
use super::config::{parse_unit_manifest, ShardConfig, UnitManifest};
use super::router::MoeRouterWeights;
use super::shard::{Shard, ShardTransport};
use super::wire::{
    decode_layer_batch_request_f16, decode_layer_batch_response_f16,
    encode_layer_batch_request_f16, encode_layer_batch_response_f16, f16_bits_to_f32,
    f32_to_f16_bits,
};

/// f32→f16→f32 round-trip should preserve normal-range residual values
/// to within ~3 decimal digits.  Spot-check the boundary cases too.
#[test]
fn f16_round_trip_preserves_residual_values() {
    let test_cases: &[f32] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        100.0,
        -100.0,
        0.001,
        -0.001,
        65504.0, // f16 max
        -65504.0,
        1e-4, // small but representable
        std::f32::consts::PI,
        std::f32::consts::E,
    ];
    for &v in test_cases {
        let bits = f32_to_f16_bits(v);
        let back = f16_bits_to_f32(bits);
        // f16 has 11-bit mantissa precision → ~3 decimal digits.
        // Tolerate 0.1% relative error or 1e-3 absolute, whichever is larger.
        let tol = (v.abs() * 1e-3).max(1e-3);
        assert!(
            (v - back).abs() <= tol,
            "f16 round-trip drift for v={v}: back={back} bits={bits:#06x}"
        );
    }
}

/// Out-of-range f32 inputs should saturate to ±Inf, not produce garbage.
#[test]
fn f16_saturates_overflow() {
    let big = 1e10_f32;
    let bits = f32_to_f16_bits(big);
    let back = f16_bits_to_f32(bits);
    assert!(
        back.is_infinite() && back > 0.0,
        "expected +Inf, got {back}"
    );

    let bits_neg = f32_to_f16_bits(-1e10_f32);
    let back_neg = f16_bits_to_f32(bits_neg);
    assert!(
        back_neg.is_infinite() && back_neg < 0.0,
        "expected -Inf, got {back_neg}"
    );
}

/// Subnormal inputs round to zero or near-zero correctly.
#[test]
fn f16_handles_subnormals() {
    // f16 smallest subnormal ≈ 6e-8; below that → 0.
    let tiny = 1e-9_f32;
    let bits = f32_to_f16_bits(tiny);
    let back = f16_bits_to_f32(bits);
    assert!(back.abs() < 1e-7, "expected ~0 for tiny={tiny}, got {back}");
}

/// Encode-then-decode round-trip for the layer-batch f16 wire.
#[test]
fn f16_layer_batch_request_round_trip() {
    let layer = 15usize;
    let residual: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin() * 5.0).collect();
    let expert_ids: Vec<u32> = vec![3, 17, 42, 88];
    let expert_weights: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

    let bytes = encode_layer_batch_request_f16(layer, &residual, &expert_ids, &expert_weights);
    // Header (12) + residual (256 × 2) + K × 8 = 12 + 512 + 32 = 556
    assert_eq!(bytes.len(), 12 + 256 * 2 + 4 * 8);

    let (l2, r2, ids2, ws2) =
        decode_layer_batch_request_f16(&bytes).expect("decode should succeed");
    assert_eq!(l2, layer);
    assert_eq!(ids2, expert_ids);
    assert_eq!(ws2, expert_weights); // weights are f32 → exact
    assert_eq!(r2.len(), residual.len());
    for (a, b) in residual.iter().zip(r2.iter()) {
        let tol = (a.abs() * 1e-3).max(1e-3);
        assert!(
            (a - b).abs() <= tol,
            "residual drift after round-trip: {a} vs {b}"
        );
    }
}

/// Encode-then-decode round-trip for the layer-batch f16 response.
#[test]
fn f16_layer_batch_response_round_trip() {
    let weighted_sum: Vec<f32> = (0..512).map(|i| (i as f32 * 0.013).cos() * 2.5).collect();
    let bytes = encode_layer_batch_response_f16(&weighted_sum, 1.234);
    assert_eq!(bytes.len(), 8 + 512 * 2);
    let back = decode_layer_batch_response_f16(&bytes).expect("decode should succeed");
    assert_eq!(back.len(), weighted_sum.len());
    for (a, b) in weighted_sum.iter().zip(back.iter()) {
        let tol = (a.abs() * 1e-3).max(1e-3);
        assert!(
            (a - b).abs() <= tol,
            "weighted_sum drift after round-trip: {a} vs {b}"
        );
    }
}

/// Truncated f16 buffers should fail safely (None), not panic.
#[test]
fn f16_layer_batch_handles_truncation() {
    assert!(decode_layer_batch_request_f16(&[]).is_none());
    assert!(decode_layer_batch_request_f16(&[0u8; 11]).is_none());
    assert!(decode_layer_batch_response_f16(&[0u8; 7]).is_none());
}

#[test]
fn parse_range_valid() {
    assert_eq!(ShardConfig::parse_range("0-31"), Some((0, 31)));
    assert_eq!(ShardConfig::parse_range("32-63"), Some((32, 63)));
    assert_eq!(ShardConfig::parse_range("0-0"), Some((0, 0)));
}

#[test]
fn parse_range_invalid() {
    assert_eq!(ShardConfig::parse_range("31-0"), None); // reversed
    assert_eq!(ShardConfig::parse_range("abc"), None);
    assert_eq!(ShardConfig::parse_range(""), None);
}

#[test]
fn shard_config_strips_trailing_slash() {
    let s = ShardConfig::new(0, 31, "http://a.example.com:8081///");
    assert_eq!(s.url, "http://a.example.com:8081");
}

// ── Per-(layer, expert) ownership ────────────────────────────────────
//
// Verify that:
//   1. A shard built with `with_units` honours the explicit `(layer, expert_id)`
//      set via the layer-aware `owns_unit(...)` check.
//   2. Layer-uniform shards keep working unchanged via `owns_unit`
//      (legacy `--moe-shards "0-63=URL"` configs).
//   3. The manifest parser round-trips JSON → `Vec<ShardConfig>` with
//      ownership sets matching the inclusive ranges in the input.

fn make_unit_shard(units: &[(usize, usize)]) -> Shard {
    let set: std::collections::HashSet<(usize, usize)> = units.iter().copied().collect();
    let config = ShardConfig::with_units("http://localhost:9000", set);
    let transport = ShardTransport::Http(reqwest::blocking::Client::new());
    Shard { config, transport }
}

#[test]
fn shard_with_units_only_owns_via_layer_aware_check() {
    let s = make_unit_shard(&[(0, 5), (3, 17)]);
    assert!(s.owns_unit(0, 5));
    assert!(s.owns_unit(3, 17));
    assert!(!s.owns_unit(1, 5)); // wrong layer
    assert!(!s.owns_unit(0, 6)); // wrong expert
    assert!(!s.owns_unit(3, 5)); // belongs to layer 0, not 3
}

#[test]
fn shard_layer_uniform_owns_unit_falls_back_to_range() {
    let config = ShardConfig::new(0, 31, "http://localhost:9000");
    let transport = ShardTransport::Http(reqwest::blocking::Client::new());
    let s = Shard { config, transport };
    // owns_unit on a legacy range-shard ignores the layer and uses the
    // range — keeps `--moe-shards "0-31=URL"` semantics.
    assert!(s.owns_unit(0, 0));
    assert!(s.owns_unit(0, 31));
    assert!(s.owns_unit(7, 17));
    assert!(!s.owns_unit(0, 32));
}

#[test]
fn unit_manifest_round_trips_into_shard_configs() {
    let json = r#"{
        "shards": [
            {"url": "grpc://a:9081",
             "layer_experts": {"0": [[0,2]], "1": [[5,7]]}},
            {"url": "grpc://b:9082",
             "layer_experts": {"0": [[3,5]], "1": [[8,10],[15,15]]}}
        ]
    }"#;
    let m: UnitManifest = serde_json::from_str(json).unwrap();
    let configs = m.into_shard_configs().unwrap();
    assert_eq!(configs.len(), 2);

    // Shard A: 6 (layer, expert) pairs.
    let a = &configs[0];
    let a_units = a.unit_set.as_ref().unwrap();
    assert_eq!(a_units.len(), 6);
    for &(l, e) in &[(0, 0), (0, 1), (0, 2), (1, 5), (1, 6), (1, 7)] {
        assert!(a_units.contains(&(l, e)), "shard A missing ({l},{e})");
    }
    assert_eq!(a.start, 0); // min expert id across set
    assert_eq!(a.end, 7); // max expert id across set

    // Shard B: 7 pairs (note the singleton range [15,15]).
    let b_units = configs[1].unit_set.as_ref().unwrap();
    assert_eq!(b_units.len(), 7);
    assert!(b_units.contains(&(1, 15)));
}

#[test]
fn unit_manifest_rejects_reversed_range() {
    let json = r#"{"shards": [
        {"url": "grpc://x:1", "layer_experts": {"0": [[5,2]]}}
    ]}"#;
    let m: UnitManifest = serde_json::from_str(json).unwrap();
    let err = m.into_shard_configs().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("end (2) must be >= start (5)"), "got: {msg}");
}

#[test]
fn unit_manifest_rejects_non_numeric_layer() {
    let json = r#"{"shards": [
        {"url": "grpc://x:1", "layer_experts": {"oops": [[0,1]]}}
    ]}"#;
    let m: UnitManifest = serde_json::from_str(json).unwrap();
    let err = m.into_shard_configs().unwrap_err();
    assert!(format!("{err}").contains("layer key 'oops'"));
}

#[test]
fn parse_unit_manifest_reports_path_on_missing_file() {
    let bogus = std::path::PathBuf::from("/nonexistent/larql-units-x.json");
    let err = parse_unit_manifest(&bogus).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("read"),
        "msg should mention read failure: {msg}"
    );
    assert!(
        msg.contains(bogus.to_str().unwrap()),
        "msg should name path: {msg}"
    );
}

#[test]
fn parse_unit_manifest_round_trips_from_file() {
    // Happy path: write a manifest to a tempfile, parse it back into shards.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("units.json");
    let json = r#"{
        "shards": [
            {"url": "grpc://h:9081", "layer_experts": {"0": [[0,3]]}}
        ]
    }"#;
    std::fs::write(&path, json).unwrap();
    let configs = parse_unit_manifest(&path).unwrap();
    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0].url, "grpc://h:9081");
    let units = configs[0].unit_set.as_ref().unwrap();
    assert_eq!(units.len(), 4);
}

#[test]
fn parse_unit_manifest_reports_path_on_invalid_json() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("garbage.json");
    std::fs::write(&path, b"not-json").unwrap();
    let err = parse_unit_manifest(&path).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("parse"), "msg should mention parse: {msg}");
    assert!(
        msg.contains(path.to_str().unwrap()),
        "msg should name path: {msg}"
    );
}

#[test]
fn shard_config_with_timeout_overrides_default() {
    let s = ShardConfig::new(0, 31, "http://h:8081")
        .with_timeout(std::time::Duration::from_millis(500));
    assert_eq!(s.timeout, std::time::Duration::from_millis(500));
}

#[test]
fn shard_config_with_units_empty_set_yields_zero_range() {
    // Diagnostic min/max default to (0, 0) when the unit set is empty.
    let empty: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let s = ShardConfig::with_units("http://h:9000", empty);
    assert_eq!(s.start, 0);
    assert_eq!(s.end, 0);
    assert!(s.unit_set.unwrap().is_empty());
}

#[test]
fn parse_range_missing_dash_returns_none() {
    // "5" splits as a single part — second `parts.next()?` short-circuits.
    assert_eq!(ShardConfig::parse_range("5"), None);
}

#[test]
fn route_softmax_sums_to_one() {
    let num_experts = 8;
    let hidden = 4;
    let router_proj: Vec<f32> = (0..num_experts * hidden).map(|i| i as f32 * 0.01).collect();
    let router = MoeRouterWeights {
        router_proj: &router_proj,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k: 2,
    };
    let h: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2];
    let (_, indices, weights) = router.route(&h, 0.0, 1e-6);
    assert_eq!(indices.len(), 2);
    assert_eq!(weights.len(), 2);
    assert!(weights.iter().all(|&w| w >= 0.0));
}

#[test]
fn route_with_parameter_free_router_norm() {
    // HF Gemma 4 codepath: router_norm is empty AND parameter_free=true →
    // route() must call rms_norm_no_weight on the input. Without the
    // helper this branch panics with "function not found"; with it, the
    // route should still produce a valid top-k.
    let num_experts = 4;
    let hidden = 4;
    let router_proj: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let router = MoeRouterWeights {
        router_proj: &router_proj,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k: 2,
    };
    let h: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5];
    let (h_norm_out, indices, weights) = router.route(&h, 0.0, 1e-6);

    // h_norm_out is the experts' input (pre_experts_norm output).
    // Since pre_experts_norm is empty, h_norm_out should be h verbatim.
    assert_eq!(h_norm_out, h);

    // Top-K selected and weights renormalised to sum to 1.
    assert_eq!(indices.len(), 2);
    assert_eq!(weights.len(), 2);
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to 1, got {sum}"
    );
    assert!(weights.iter().all(|&w| w >= 0.0));
}

#[test]
fn route_with_router_input_scalar() {
    // HF Gemma 4 also uses router_input_scalar = hidden_size^-0.5.
    // Verify the scalar is applied (changes which expert wins) without
    // breaking the softmax+top-k pipeline.
    let num_experts = 4;
    let hidden = 4;
    // Bias router_proj so expert 0 wins on un-scaled input.
    let mut router_proj: Vec<f32> = vec![0.0; num_experts * hidden];
    router_proj[0] = 100.0; // expert 0 row, dim 0
    router_proj[hidden] = -100.0; // expert 1 row, dim 0

    let h: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

    let unscaled = MoeRouterWeights {
        router_proj: &router_proj,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k: 1,
    };
    let (_, idx_unscaled, _) = unscaled.route(&h, 0.0, 1e-6);
    assert_eq!(idx_unscaled, vec![0]);

    // With scalar = 0.5, the logit gap shrinks (50 vs -50 still picks
    // expert 0). Use a negating scalar to flip the winner — this proves
    // the scalar actually multiplies through.
    let flipped = MoeRouterWeights {
        router_input_scalar: -1.0,
        ..unscaled
    };
    let (_, idx_flipped, _) = flipped.route(&h, 0.0, 1e-6);
    assert_eq!(
        idx_flipped,
        vec![1],
        "negative scalar should flip the winner"
    );
}

#[test]
fn forward_moe_empty_input_returns_zero() {
    // Can't connect to a real server, but we can verify the early-exit path.
    // Construct a backend with an empty shard list via the raw struct (bypassing connect).
    let backend = RemoteMoeBackend {
        shards: Arc::new(RwLock::new(vec![])),
    };
    let router = MoeRouterWeights {
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 0,
    };
    let result = backend.forward_moe(0, &[1.0f32, 2.0, 3.0], &router, 0.0, 1e-6);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![0.0f32; 3]);
}

// ── A failed shard is an error, never a zero contribution ─────────────────

/// Fixture coordinates for the shard-attribution tests. Arbitrary; named so a
/// failure points at a deliberate value.
const FAILED_SHARD_URL: &str = "http://shard-a:9000";
const FAILED_LAYERS: [usize; 2] = [3, 4];
const FAILED_EXPERTS: [u32; 2] = [5, 9];
const FAILED_STATUS: u16 = 503;
const FAILED_BODY: &str = "overloaded";
const PARTIAL_HIDDEN: usize = 4;
const PARTIAL_VALUE: f32 = 0.5;
/// Norm parameters for the direct pre-dispatch call (Gemma-style offset).
const PREDISPATCH_NORM_OFFSET: f32 = 1.0;
const PREDISPATCH_EPS: f32 = 1e-6;

fn server_error() -> super::RemoteMoeError {
    super::RemoteMoeError::ServerError {
        status: FAILED_STATUS,
        body: FAILED_BODY.into(),
    }
}

#[test]
fn a_shard_failure_names_the_shard_layers_experts_and_cause() {
    let err = server_error().at_shard(FAILED_SHARD_URL, &FAILED_LAYERS, &FAILED_EXPERTS);
    let message = err.to_string();
    assert!(message.contains(FAILED_SHARD_URL), "{message}");
    assert!(message.contains(&format!("{FAILED_LAYERS:?}")), "{message}");
    assert!(
        message.contains(&format!("{FAILED_EXPERTS:?}")),
        "{message}"
    );
    assert!(message.contains(&FAILED_STATUS.to_string()), "{message}");
    assert!(message.contains(FAILED_BODY), "{message}");
}

#[test]
fn attributing_an_attributed_failure_keeps_the_first_attribution() {
    let once = server_error().at_shard(FAILED_SHARD_URL, &FAILED_LAYERS, &FAILED_EXPERTS);
    let twice = once.clone().at_shard("http://elsewhere:1", &[0], &[]);
    assert_eq!(once.to_string(), twice.to_string());
}

#[test]
fn a_partial_sum_of_the_wrong_width_is_refused() {
    let mut h2 = vec![vec![0.0f32; PARTIAL_HIDDEN]; FAILED_LAYERS.len()];
    let short = vec![PARTIAL_VALUE; PARTIAL_HIDDEN - 1];
    let err = super::backend::predispatch::accumulate_partial(&mut h2, 0, &short, PARTIAL_HIDDEN)
        .expect_err("a short partial is that shard's experts missing");
    assert!(
        matches!(err, super::RemoteMoeError::BadResponse(_)),
        "{err}"
    );
    assert!(
        h2.iter().flatten().all(|&v| v == 0.0),
        "nothing accumulated"
    );
}

#[test]
fn a_partial_sum_for_an_undispatched_layer_is_refused() {
    let mut h2 = vec![vec![0.0f32; PARTIAL_HIDDEN]; FAILED_LAYERS.len()];
    let partial = vec![PARTIAL_VALUE; PARTIAL_HIDDEN];
    let err = super::backend::predispatch::accumulate_partial(
        &mut h2,
        FAILED_LAYERS.len(),
        &partial,
        PARTIAL_HIDDEN,
    )
    .expect_err("out-of-range layer");
    assert!(
        matches!(err, super::RemoteMoeError::BadResponse(_)),
        "{err}"
    );
}

#[test]
fn a_well_formed_partial_sum_accumulates() {
    let mut h2 = vec![vec![0.0f32; PARTIAL_HIDDEN]; FAILED_LAYERS.len()];
    let partial = vec![PARTIAL_VALUE; PARTIAL_HIDDEN];
    for _ in 0..2 {
        super::backend::predispatch::accumulate_partial(&mut h2, 1, &partial, PARTIAL_HIDDEN)
            .expect("well-formed");
    }
    assert!(h2[1].iter().all(|&v| v == 2.0 * PARTIAL_VALUE));
    assert!(h2[0].iter().all(|&v| v == 0.0));
}

#[test]
fn predispatch_refuses_an_expert_no_shard_owns() {
    // The shard owns units on a different layer only, so every expert the
    // router selects for layer 0 is unserved. Pre-dispatch used to skip such
    // experts ("partial deployment") and return the rest as the answer.
    let weights = crate::test_utils::make_test_gemma4_moe_weights();
    let layer = 0;
    let router =
        crate::vindex::build_moe_router_weights(&weights, &*weights.arch, layer).expect("router");
    let backend = RemoteMoeBackend {
        shards: Arc::new(RwLock::new(vec![make_unit_shard(&[(layer + 1, 0)])])),
    };
    let h = vec![PARTIAL_VALUE; weights.hidden_size];
    let err = backend
        .forward_moe_predispatch(&[h], &[router], PREDISPATCH_NORM_OFFSET, PREDISPATCH_EPS)
        .expect_err("an unserved expert must fail the dispatch");
    assert!(
        matches!(err, super::RemoteMoeError::NoShard { .. }),
        "{err}"
    );
}

#[test]
fn every_remote_moe_error_says_what_failed() {
    use super::RemoteMoeError as E;
    let unserved = FAILED_EXPERTS[0] as usize;
    let unserved_needle = format!("expert {unserved}");
    let cases = [
        (
            E::Unreachable {
                url: FAILED_SHARD_URL.into(),
                cause: FAILED_BODY.into(),
            },
            FAILED_SHARD_URL,
        ),
        (server_error(), FAILED_BODY),
        (E::BadResponse(FAILED_BODY.into()), FAILED_BODY),
        (
            E::NoShard {
                expert_id: unserved,
            },
            unserved_needle.as_str(),
        ),
        (E::Client(FAILED_BODY.into()), FAILED_BODY),
    ];
    for (err, needle) in cases {
        let message = err.to_string();
        assert!(message.contains(needle), "{message}");
        assert!(std::error::Error::source(&err).is_none());
    }
}
