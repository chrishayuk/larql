//! Smoke test for the Q4K-quantised synthetic vindex fixture.
//! Confirms (a) the on-disk layout matches what `VectorIndex::load_vindex`
//! expects for a Q4K vindex, and (b) the generation path that previously
//! panicked with "attn Q4K slices missing for layer 0" now succeeds —
//! i.e. `insert_q4k_layer_tensors` actually finds the K-quant data.

mod common;

#[test]
fn q4k_fixture_lists_actual_files() {
    let (_model, fixture) = common::model_with_q4k_weights("q4k-synthetic");
    let mut files: Vec<String> = std::fs::read_dir(&fixture.dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .collect();
    files.sort();
    println!("Q4K vindex files: {files:?}");
    // index.json is the one constant; the rest of the layout is
    // what we want to discover, not what we want to assert.
    assert!(fixture.dir.join("index.json").exists());
}

#[test]
fn q4k_fixture_satisfies_q4k_weight_loader() {
    let (model, _fixture) = common::model_with_q4k_weights("q4k-synthetic");
    // `get_or_load_weights` routes through `load_model_weights_kquant_shard`
    // when `config.quant == Q4K`. If the on-disk files are correctly
    // shaped this returns Ok; otherwise the loader bubbles up a parse
    // error.
    let weights = model.get_or_load_weights().expect("Q4K weights must load");
    assert_eq!(weights.num_layers, 2);
    assert_eq!(weights.hidden_size, 8);
    assert_eq!(weights.intermediate_size, 4);
    assert_eq!(weights.vocab_size, 16);
}

#[test]
fn q4k_fixture_unblocks_insert_q4k_layer_tensors() {
    // Direct call into the generation precondition that was failing
    // on the f32 fixture (`vindex/kquant_forward/cached.rs:106`).
    // If this returns Ok, the chat / completions / stream generation
    // paths are unblocked end-to-end.
    let (model, _fixture) = common::model_with_q4k_weights("q4k-synthetic");

    let mut weights_guard = model.lock_weights_for_gen().expect("lock weights for gen");
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;
    let patched = model.patched.blocking_read();
    let index = patched.base();

    let inserted = larql_inference::vindex::insert_q4k_layer_tensors(weights, index, 0);
    assert!(
        inserted.is_ok(),
        "insert_q4k_layer_tensors must succeed on the Q4K fixture; got {inserted:?}"
    );
}
