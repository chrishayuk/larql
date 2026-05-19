//! Mechanistic-interp surface demo — capture, lens, neighbors, ablate, steer, patch.
//!
//! Self-contained: builds synthetic weights via [`make_test_weights`] so it
//! runs without a vindex on any platform. Walks through the six primitives
//! that lazarus-style MCP tools build on:
//!
//! 1. **Capture** — `RecordHook` over `trace_forward_full_hooked` snapshots
//!    the residual at chosen layers.
//! 2. **Logit lens** — `logit_lens_topk` reads vocab off a mid-stack residual.
//! 3. **Embedding neighbors** — `embedding_neighbors` returns the closest
//!    vocab tokens to a vector under cosine similarity against `W_E`.
//! 4. **Ablation** — `ZeroAblateHook` zeros the post-layer residual at a
//!    chosen layer and measures the downstream effect.
//! 5. **Steering** — `SteerHook` adds `α·v` to the last-token row at a
//!    chosen layer and measures the downstream effect.
//! 6. **Activation patching** — `capture_donor_state` + `patch_and_trace`
//!    transplant residuals from one prompt's pass into another's.
//! 7. **Generate with hooks** — `generate_cached_hooked` runs multi-token
//!    generation with the hook firing on every layer of every step. Used
//!    here to show steered output diverging from the baseline.
//!
//! Usage: `cargo run --release -p larql-inference --example mech_interp_demo`
//!
//! All numbers are illustrative — the synthetic weights aren't a real
//! language model. The point is to exercise every primitive end-to-end so
//! you can see the API shapes and copy them into real workflows.
//!
//! [`make_test_weights`]: larql_inference::test_utils::make_test_weights

use ndarray::Array1;

use larql_inference::ffn::WeightFfn;
use larql_inference::forward::{
    capture_donor_state, embedding_neighbors, embedding_row, logit_lens_topk, patch_and_trace,
    project_through_unembed, trace_forward, trace_forward_full_hooked, RecordHook, SteerHook,
    ZeroAblateHook,
};
use larql_inference::test_utils::{make_test_tokenizer, make_test_weights};
use larql_kv::generation::{generate_cached, generate_cached_hooked};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn print_topk(label: &str, hits: &[(u32, f32)]) {
    print!("  {label:<20}");
    for (id, score) in hits.iter().take(5) {
        print!(" [id={id}, {score:.4}]");
    }
    println!();
}

fn main() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };

    println!("=== mech-interp surface demo ===");
    println!(
        "synthetic model: {} layers, hidden={}, vocab={}\n",
        weights.num_layers, weights.hidden_size, weights.vocab_size
    );

    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let last_layer = weights.num_layers - 1;
    // Mid-stack layer (or the only intermediate one on the 2-layer test model).
    let target_layer = weights.num_layers / 2;
    // Distinct layers to inspect — dedup so a 2-layer synthetic model
    // doesn't print the same row twice.
    let inspect_layers: Vec<usize> = {
        let mut v = vec![0usize, target_layer, last_layer];
        v.sort();
        v.dedup();
        v
    };

    // ── 1. Capture ──────────────────────────────────────────────────────────
    println!("[1] capture residuals via RecordHook");
    let mut record = RecordHook::for_layers(inspect_layers.iter().copied());
    let _ = trace_forward_full_hooked(
        &weights,
        &prompt,
        &inspect_layers,
        false,
        0,
        false,
        &ffn,
        &mut record,
    );
    for layer in &inspect_layers {
        let mat = record.post_layer.get(layer).unwrap();
        println!(
            "  layer {layer:>2}: post_layer shape = ({}, {})",
            mat.nrows(),
            mat.ncols()
        );
    }

    // ── 2. Logit lens ───────────────────────────────────────────────────────
    println!("\n[2] logit_lens_topk on the captured residuals");
    for layer in &inspect_layers {
        let res = record.post_layer.get(layer).unwrap();
        let last_row = res.row(res.nrows() - 1).to_vec();
        let top = logit_lens_topk(&weights, &last_row, 5);
        print_topk(&format!("layer {layer:>2}"), &top);
    }

    // ── 3. Embedding neighbors + raw unembed projection ─────────────────────
    println!("\n[3] embedding_neighbors + project_through_unembed");
    let token0 = embedding_row(&weights, 1).expect("token 1 embed");
    let neighbors = embedding_neighbors(&weights, &token0, 5);
    print_topk("embed neighbors", &neighbors);
    let dla = project_through_unembed(&weights, &token0, 5);
    print_topk("DLA top-5", &dla);

    // ── 4. Ablation ─────────────────────────────────────────────────────────
    println!("\n[4] zero-ablate post-layer residual at the middle layer");
    let baseline = trace_forward(&weights, &prompt, &[last_layer], false, 0).residuals[0]
        .1
        .clone();

    let mut ablate = ZeroAblateHook::for_layers([target_layer]);
    let ablated = trace_forward_full_hooked(
        &weights,
        &prompt,
        &[last_layer],
        false,
        0,
        false,
        &ffn,
        &mut ablate,
    )
    .residuals[0]
        .1
        .clone();
    println!(
        "  cos(baseline_last, ablated_last) = {:.4}",
        cosine(&baseline, &ablated)
    );

    // ── 5. Steering ─────────────────────────────────────────────────────────
    println!("\n[5] add α·v at the middle layer");
    let v = Array1::from_vec(
        (0..weights.hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect(),
    );
    let mut steer = SteerHook::new().add(target_layer, v, 0.5);
    let steered = trace_forward_full_hooked(
        &weights,
        &prompt,
        &[last_layer],
        false,
        0,
        false,
        &ffn,
        &mut steer,
    )
    .residuals[0]
        .1
        .clone();
    println!(
        "  cos(baseline_last, steered_last) = {:.4}",
        cosine(&baseline, &steered)
    );

    // ── 6. Activation patching ──────────────────────────────────────────────
    //
    // Patch the donor's residual at an *earlier* layer than the one we
    // capture, so attention in the layers after the patch can mix the
    // donor's value into the recipient's last-token row. Patching at the
    // capture layer would be a no-op for the last-token readout.
    let patch_layer = 0;
    println!("\n[6] activation patching donor → recipient");
    let recipient: Vec<u32> = vec![5, 6, 7, 8];
    let recipient_baseline = trace_forward(&weights, &recipient, &[last_layer], false, 0).residuals
        [0]
    .1
    .clone();
    let donor = capture_donor_state(&weights, &prompt, &[(patch_layer, recipient.len() - 1)]);
    println!(
        "  donor recorded {} coord(s) at (layer={patch_layer}, pos={})",
        donor.records.len(),
        recipient.len() - 1
    );
    let patched_trace = patch_and_trace(&weights, &recipient, &donor, &[last_layer]);
    let patched_last = &patched_trace.residuals[0].1;
    println!(
        "  cos(recipient_baseline, recipient_after_patch) = {:.4}",
        cosine(&recipient_baseline, patched_last)
    );

    // ── 7. Multi-token generation with a steering hook ─────────────────────
    //
    // `generate_cached_hooked` is the multi-token analogue of
    // `trace_forward_full_hooked` — same hook trait, fires on every layer
    // of every prefill + decode step. The Metal-fast `generate` path is
    // hook-free by design (kernels are fused); use this CPU path when
    // hooks have to be active during multi-token generation.
    println!("\n[7] generate with a steering hook (multi-token)");
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let max_new = 4usize;

    let baseline_ids = generate_cached(&weights, &tokenizer, &ffn, &prompt, max_new, |_, _| {});
    let v2 = Array1::from_vec(
        (0..weights.hidden_size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect(),
    );
    let mut steer = SteerHook::new().add(0, v2, 5.0);
    let steered_ids = generate_cached_hooked(
        &weights,
        &tokenizer,
        &ffn,
        &prompt,
        max_new,
        None,
        None,
        &mut steer,
        |_, _| {},
    );
    println!("  baseline ids = {baseline_ids:?}");
    println!("  steered  ids = {steered_ids:?}");
    println!(
        "  diverged at step = {}",
        baseline_ids
            .iter()
            .zip(steered_ids.iter())
            .position(|(a, b)| a != b)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "(no divergence)".into())
    );

    println!("\n=== done ===");
    println!(
        "next: register your own LayerHook impl, or wire these primitives \
         into a chuk-mcp-lazarus tool"
    );
}
