//! Coverage push for `routes/expert/warmup.rs` (was 0%, target ≥ 90%).
//!
//! `warmup_hnsw_unit_cache` is a boot-time helper called from
//! bootstrap (not an HTTP route). The early-return branches —
//! `LARQL_NO_WARMUP=1` env-gated and `is_hybrid_moe() == false` —
//! cover the bulk of the function on non-MoE models, which is what
//! our synthetic fixture provides.

mod common;

#[tokio::test]
async fn warmup_hnsw_unit_cache_non_moe_model_returns_zero_built() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    // Synthetic uses "llama" arch — `is_hybrid_moe()` is false, so
    // the function takes the early-return branch.
    let result = larql_server::routes::expert::warmup::warmup_hnsw_unit_cache(&model);
    let (built, _layers, _experts) = result.expect("warmup must succeed on non-MoE");
    assert_eq!(built, 0, "non-MoE arch must trigger early return");
}

#[tokio::test]
async fn warmup_hnsw_unit_cache_with_no_warmup_env_short_circuits() {
    // LARQL_NO_WARMUP=1 must short-circuit before any model
    // inspection. This branch is the cheapest to cover and exists
    // specifically so low-RAM dev setups can skip warmup.
    //
    // SAFETY: setting an environment variable from a test is racy
    // against any other test that reads the same var. `warmup` is
    // the only LARQL_NO_WARMUP reader; we set + restore inline.
    unsafe { std::env::set_var("LARQL_NO_WARMUP", "1") };
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let result = larql_server::routes::expert::warmup::warmup_hnsw_unit_cache(&model);
    unsafe { std::env::remove_var("LARQL_NO_WARMUP") };
    assert_eq!(result.unwrap(), (0, 0, 0));
}
