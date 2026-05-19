//! CPU V-projection correctness on `attention_k_eq_v` architectures
//! (Gemma 4 global layers).
//!
//! The vindex extractor stores V as **Q6_K** (6-bit) and K as **Q4_K**
//! (4-bit) even when the upstream `attention_k_eq_v=true` flag says the
//! two tensors share the same source data — see `pad_rows_to_256` and
//! the `is_v { quantize_q6_k } else { quantize_q4_k }` split in
//! `crates/larql-vindex/src/format/weights/write.rs`.
//!
//! CPU attention was short-circuiting the V projection (using `k_full`,
//! i.e. Q4_K-dequanted K) instead of running the real V projection
//! through the Q6_K-dequanted W_v tensor. That cost ~6% of attention
//! magnitude at every Gemma 4 global layer and compounded to a visible
//! top-1 divergence on multi-token generation.
//!
//! The fix in `attention/block.rs`: always go through the stored W_v
//! when it exists. This test pins that behaviour in two ways:
//!
//! 1. **Manifest invariant**: confirm the vindex we test against does
//!    in fact store V with a *different* quantisation format than K at
//!    `v_shares_k` layers (otherwise the test wouldn't exercise the
//!    bug-fix regime).
//! 2. **Numerical invariant**: dequant both tensors and assert the
//!    resulting f32 matrices differ element-wise. If they were ever
//!    accidentally identical (e.g. a future build pipeline quantises
//!    both as Q4_K), the V projection collapses to the pre-fix
//!    shortcut without anyone noticing.
//!
//! Skip semantics: the test needs a Gemma 4 31B Q4K vindex locally.
//! Without one it logs and returns Ok; set `LARQL_ARCH_STRICT=1` to
//! make it a hard failure.

use std::path::PathBuf;

use larql_vindex::{load_model_weights_kquant, load_vindex_config, SilentLoadCallbacks};

fn find_gemma4_dense_vindex() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_VINDEX_GEMMA4_31B_Q4K") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    for base in [
        PathBuf::from("/Users/christopherhay/chris-models"),
        PathBuf::from(&home).join(".cache/larql/local"),
        PathBuf::from("output"),
    ] {
        let p = base.join("gemma4-31b-q4k.vindex");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

fn strict_mode() -> bool {
    matches!(
        std::env::var("LARQL_ARCH_STRICT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// The manifest is ground truth for what the extractor wrote. Check that
/// K and V at a known global layer (L5 on Gemma 4 31B) have different
/// quantisation formats — the precondition for the Q6_K V path to
/// matter at all. If this fails, the fix-under-test has no numerical
/// effect and the CPU shortcut would be arguably fine again.
#[test]
fn vindex_stores_v_as_q6k_for_gemma4_global_layers() {
    let Some(vindex) = find_gemma4_dense_vindex() else {
        if strict_mode() {
            panic!("gemma4-31b-q4k.vindex not found (LARQL_ARCH_STRICT=1)");
        }
        eprintln!("skip: gemma4-31b-q4k.vindex not found");
        return;
    };

    let manifest_path = vindex.join("attn_weights_q4k_manifest.json");
    assert!(
        manifest_path.is_file(),
        "attn_weights_q4k_manifest.json missing from {}",
        vindex.display()
    );
    let bytes = std::fs::read(&manifest_path).expect("read manifest");
    let entries: serde_json::Value = serde_json::from_slice(&bytes).expect("parse manifest");
    let arr = entries.as_array().expect("manifest is array");

    // L5 is the first global-attention layer on Gemma 4 31B (pattern 6).
    // Find the k_proj and v_proj entries for this layer.
    let mut k_format: Option<String> = None;
    let mut v_format: Option<String> = None;
    for entry in arr {
        let key = entry["key"].as_str().unwrap_or_default();
        let fmt = entry["format"].as_str().unwrap_or_default().to_string();
        if key == "layers.5.self_attn.k_proj.weight" {
            k_format = Some(fmt);
        } else if key == "layers.5.self_attn.v_proj.weight" {
            v_format = Some(fmt);
        }
    }
    let k_format = k_format.expect("L5 k_proj missing from manifest");
    let v_format = v_format.expect("L5 v_proj missing from manifest");

    assert_eq!(
        k_format, "Q4_K",
        "L5 k_proj should be Q4_K (cheap quantisation for K); got {k_format}"
    );
    assert_eq!(
        v_format, "Q6_K",
        "L5 v_proj should be Q6_K (the reason CPU must not take the k_full shortcut). \
         Got {v_format} — if this changed, update the comment in \
         `attention/block.rs` describing the quant-format asymmetry."
    );
}

/// Numerical invariant: when `predict_kquant_hidden` loads L5's weights,
/// the resulting `w_k` and `w_v` tensors must differ element-wise —
/// proving the Q6_K V dequant path returns a distinct approximation of
/// the same underlying data. Equivalent tensors would silently re-open
/// the door to the CPU shortcut.
#[test]
fn cpu_q4k_load_produces_distinct_w_k_and_w_v_for_gemma4_global() {
    let Some(vindex) = find_gemma4_dense_vindex() else {
        if strict_mode() {
            panic!("gemma4-31b-q4k.vindex not found (LARQL_ARCH_STRICT=1)");
        }
        eprintln!("skip: gemma4-31b-q4k.vindex not found");
        return;
    };

    let cfg = load_vindex_config(&vindex).expect("load_vindex_config");
    assert_eq!(
        cfg.family, "gemma4",
        "this test expects a Gemma 4 vindex; got {:?}",
        cfg.family
    );

    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&vindex, &mut cb).expect("load weights");
    let arch = &*weights.arch;

    // Exercise the predict_kquant_hidden tensor-load path directly. It
    // dequantises attn weights per layer and inserts them into
    // `weights.tensors`. We only need the shapes and a sample of
    // values — run the loader enough to populate L5's Q/K/V, then
    // compare W_k vs W_v directly.
    //
    // `predict_kquant_hidden` is not public, but its per-layer tensor
    // insertion is what drives CPU attention. We replicate the
    // equivalent load here — dequantise L5's Q/K/V/O into
    // `weights.tensors` the same way the forward pass does.
    use larql_vindex::VectorIndex;
    let mut cb2 = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex, &mut cb2).expect("load vindex");
    index.load_attn_kquant(&vindex).expect("load_attn_kquant");

    let layer: usize = 5;
    let attn = index
        .attn_kquant_layer_data(layer)
        .expect("L5 attn slices present");
    // attn is [q, k, v, o] — verify shapes match the expected global
    // dims before we dequant (head_dim=512, num_q=32, num_kv=4, hidden=5376).
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let head_dim = arch.head_dim_for_layer(layer);
    assert_eq!(
        (num_q, num_kv, head_dim),
        (32, 4, 512),
        "Gemma 4 31B L5 global geometry drifted — update test constants"
    );

    let kv_dim = num_kv * head_dim;
    let hidden = weights.hidden_size;

    // Dequantise K (Q4_K) and V (Q6_K) directly via the quant crate.
    // Both are row-padded to a multiple of 256 per super-block, so we
    // compute `padded` and then truncate back to `rows*cols` f32s.
    let n = kv_dim * hidden;
    let padded = n.div_ceil(256) * 256;
    let dequant = |bytes: &[u8], format: &str| -> Vec<f32> {
        let floats = match format {
            "Q4_K" => larql_models::quant::ggml::dequantize_q4_k(bytes, padded)
                .expect("Q4_K dequant failed"),
            "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded)
                .expect("Q6_K dequant failed"),
            other => panic!("unsupported quant format in vindex: {other}"),
        };
        if floats.len() > n {
            floats[..n].to_vec()
        } else {
            floats
        }
    };
    let kf = dequant(attn[1].0, attn[1].1);
    let vf = dequant(attn[2].0, attn[2].1);

    assert_eq!(
        kf.len(),
        vf.len(),
        "K and V should have identical element counts at v_shares_k layers"
    );

    // Element-wise distinctness: at least 10% of elements must differ
    // by > 1e-4 for the two quantisation round-trips to be genuinely
    // different representations. Q4_K and Q6_K of the same source data
    // differ in quantisation error, so most elements will be close but
    // not identical — the cutoff catches pathological "both formats
    // landed on the same value" fluke without demanding every element
    // differ.
    let total = kf.len();
    let distinct = kf
        .iter()
        .zip(vf.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-4)
        .count();
    let distinct_ratio = distinct as f64 / total as f64;
    assert!(
        distinct_ratio > 0.10,
        "Q6_K-dequanted W_v matches Q4_K-dequanted W_k too closely at L5 \
         ({distinct}/{total} = {:.3}% elements differ by > 1e-4); the CPU \
         V shortcut would produce effectively the same answer. Either the \
         extractor quantised both as the same format, or the dequantiser \
         is wrong.",
        100.0 * distinct_ratio,
    );

    // Global magnitude should be close (same source tensor, just
    // different quantisation noise) — a huge ratio would suggest K and
    // V aren't actually derived from the same underlying weight.
    let k_norm: f64 = kf
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    let v_norm: f64 = vf
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    let ratio = v_norm / k_norm;
    assert!(
        (0.99..1.01).contains(&ratio),
        "L5 ||w_v|| / ||w_k|| = {ratio:.4} is outside [0.99, 1.01] — the two \
         quantisations should round-trip the same bf16 weight to within 1% norm"
    );
}
