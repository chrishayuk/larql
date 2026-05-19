//! `larql parity` — cross-backend numerical diff for inference components.
//!
//! Diffs the same input through multiple backends (slow naive reference,
//! production CPU, Metal, HF — backends added incrementally) and reports
//! the first checkpoint where they diverge beyond `--tolerance`.
//!
//! v1 (this file) ships:
//!   - `--component moe-expert` — single expert forward (gate / up / act / down)
//!   - `--component moe-block`  — full MoE block (router → top-K → experts → sum → norm)
//!   - backends: `reference` (slow naive), `cpu` (production)
//!
//! v2 (planned) — Metal as a third backend, attention/dense-ffn/layer/forward
//! components. v3 — HF Python sidecar for ground-truth reference.
//!
//! See `crates/larql-cli/ROADMAP.md` P0 → "`larql parity`" for the full design.

use clap::Args;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::commands::primary::cache;
use larql_compute::cpu::ops::moe::{cpu_moe_forward, run_single_expert_with_norm};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::{Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat};
use larql_models::weights::{per_layer_ffn_key, PER_LAYER_FFN_DOWN, PER_LAYER_FFN_GATE_UP};
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_vindex::{load_model_weights_kquant, load_vindex_config, SilentLoadCallbacks};

// ── Component / backend taxonomies ────────────────────────────────────────────

/// Inference checkpoints that can be diffed independently.
const COMPONENTS: &[&str] = &[
    "moe-expert", // single expert forward (gate/up/act/down)
    "moe-block",  // full MoE block (router → top-K → experts → sum → norm)
    "lm-head",    // final projection parity (Q4_K vs f32 reference)
    "layer",      // full hybrid-MoE layer: CPU vs Metal, per-layer residual diff
];

/// Backends available as comparison targets.
///
/// `reference` is the slow naive triple-loop CPU baseline. `cpu` is the
/// production path under test. `metal` is the GPU backend (v2 — used by
/// `--component layer`).
const BACKENDS: &[&str] = &[
    "reference", // slow naive baseline (moe-expert, moe-block)
    "cpu",       // production CPU path
    "metal",     // Metal GPU backend (layer component)
];

#[derive(Args)]
pub struct ParityArgs {
    /// Vindex directory, `hf://` URL, or cache shorthand. Same resolution
    /// as `larql run`.
    pub model: String,

    /// Inference checkpoint to diff. v1: `moe-expert`, `moe-block`.
    #[arg(long, default_value = "moe-block")]
    pub component: String,

    /// Layer index. Default 0.
    #[arg(long, default_value = "0")]
    pub layer: usize,

    /// Expert index (used when `--component moe-expert`).
    #[arg(long, default_value = "0")]
    pub expert: usize,

    /// Comma-separated list of backends to run. v1: `reference,cpu`.
    /// First backend in the list is the reference; subsequent backends
    /// are diffed against it.
    #[arg(long, default_value = "reference,cpu")]
    pub backends: String,

    /// Prompt for `--component layer` (drives the actual forward pass).
    /// For `moe-expert`/`moe-block`, the prompt seeds a synthetic residual
    /// if provided; otherwise a deterministic sin-pattern is used.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Random-ish seed for the synthetic residual. Ignored when `--prompt`
    /// is set. Default 0 produces the canonical sin pattern.
    #[arg(long, default_value = "0")]
    pub seed: u32,

    /// Max element-wise abs diff allowed before declaring divergence. The
    /// right value depends on component depth — per-expert ≈ 1e-3, full
    /// forward needs more headroom for accumulated f32 noise.
    #[arg(long, default_value = "1e-3")]
    pub tolerance: f64,

    /// Print intermediate values at each checkpoint, not just diffs.
    #[arg(long, short)]
    pub verbose: bool,
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn run(_args: ParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "`larql parity` requires the `metal` feature on macOS — Metal is the reference \
         backend this command compares CPU output against."
            .into(),
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn run(args: ParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !COMPONENTS.contains(&args.component.as_str()) {
        return Err(format!(
            "unknown --component '{}'. Available: {}",
            args.component,
            COMPONENTS.join(", ")
        )
        .into());
    }

    // `layer` component always uses metal+cpu internally; other components
    // need the backends list validated and require ≥2.
    if args.component != "layer" {
        let backends: Vec<&str> = args.backends.split(',').map(|s| s.trim()).collect();
        for b in &backends {
            if !BACKENDS.contains(b) {
                return Err(format!(
                    "unknown backend '{}'. Available: {}",
                    b,
                    BACKENDS.join(", ")
                )
                .into());
            }
        }
        if backends.len() < 2 {
            return Err("need at least 2 backends to diff (default is `reference,cpu`)".into());
        }
    }

    // ── Resolve + load vindex ────────────────────────────────────────────────
    let path = cache::resolve_model(&args.model)?;
    let config = load_vindex_config(&path)?;
    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&path, &mut cb)?;
    let arch = &*weights.arch;

    println!("Vindex:    {}", path.display());
    println!("Model:     {}", config.model);
    println!("Component: {}", args.component);
    println!("Layer:     {}", args.layer);
    println!();

    if args.component == "layer" {
        return run_layer_diff(&path, &config, &args);
    }

    // lm-head parity is backend-agnostic (Q4_K matvec vs f32 reference) —
    // works on any vindex that has an lm_head, MoE or dense.
    if !arch.is_hybrid_moe() && args.component != "lm-head" {
        return Err(format!(
            "vindex {} is not hybrid-MoE — moe-* components are MoE-only",
            args.model
        )
        .into());
    }

    let backends: Vec<&str> = args.backends.split(',').map(|s| s.trim()).collect();
    println!("Backends:  {}", backends.join(" → "));
    println!();

    match args.component.as_str() {
        "moe-expert" => run_moe_expert(&config, &weights, &args, &backends),
        "moe-block" => run_moe_block(&config, &weights, &args, &backends),
        "lm-head" => run_lm_head(&path, &config, &weights, &args, &backends),
        _ => unreachable!("validated above"),
    }
}

// ── lm-head: Q4_K-vs-reference logits for the final projection ───────────────
//
// Diagnostic motivation: a 2026-04-27 silent-corruption bug had the writer
// emit Q4_K (`format/weights/write_kquant`) while `lm_head_knn_backend` dispatched
// `q4_matvec` (Q4_0). Same byte-rate per element (0.5625 B/elem) → identical
// file size → no validation caught the format collision → multilingual
// gibberish under `--metal`. This component diffs the actual on-disk Q4_K
// lm_head against an f32 reference computed from `weights.lm_head` (the model's
// HF-loaded tied embedding for Gemma 3/4 / Llama-tied / etc.). Any future
// format swap (Q4_K → Q4_KF, transposition, scale offset, ...) makes the
// top-1 token mismatch loud.

fn run_lm_head(
    path: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_compute::CpuBackend;
    use larql_vindex::SilentLoadCallbacks;

    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    println!("hidden={hidden}, vocab={vocab}");

    // Build the same residual the moe-block / moe-expert variants use so a
    // cross-component diff at the same prompt seed is straightforward.
    let h = make_residual(hidden, args.seed);

    // Reference: f32 dot product against `weights.lm_head` (tied embedding
    // for Gemma 3 / Gemma 4 / Llama; explicit lm_head row for untied).
    let lm = &weights.lm_head;
    if lm.is_empty() {
        return Err("model has no lm_head loaded — re-run extract with weights enabled".into());
    }
    let ref_scores: Vec<f32> = lm
        .rows()
        .into_iter()
        .map(|row| row.iter().zip(h.iter()).map(|(a, b)| a * b).sum())
        .collect();

    // Vindex side: load the index *here* (separately from the f32 weights
    // load that load_model_weights_kquant did) so we exercise the production
    // `open_inference_vindex` path including `load_lm_head_kquant`.
    let mut cb = SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb)?;
    let _ = index.load_lm_head(path);
    let _ = index.load_lm_head_kquant(path);
    let has_q4 = index.has_lm_head_kquant();
    let has_full = index.has_lm_head();
    println!(
        "lm_head sources: q4_mmap={has_q4}  f32_mmap={has_full}  tied_embed={}",
        weights.lm_head.shape()[0] == config.vocab_size
    );

    // The cpu backend's lm_head_knn_backend does Q4_K matvec when the
    // q4 mmap is present, falls back to f16 mmap, then f32 BLAS. We
    // diff each available source against the reference so a regression
    // in any one path stands out.
    let cpu = CpuBackend;
    let h1d = ndarray::Array1::from_vec(h.clone());

    let mut traces: Vec<(&str, Vec<f32>)> = vec![("reference (f32 dot)", ref_scores.clone())];

    if backends.contains(&"cpu") {
        let hits = index.lm_head_knn_backend(&h1d, vocab.min(8), &cpu);
        if !hits.is_empty() {
            // hits is (token, score) sorted descending. Reconstruct a
            // sparse score vector for the diff helper.
            let mut sparse = vec![f32::NEG_INFINITY; vocab];
            for (tok, score) in &hits {
                sparse[*tok as usize] = *score;
            }
            traces.push(("cpu (lm_head_knn_backend)", sparse));
        } else {
            println!(
                "  WARN: lm_head_knn_backend returned empty — vindex has no lm_head sources \
                 (no lm_head_q4.bin, no lm_head.bin, no f16 mmap), and tied-embed fallback \
                 lives in larql-inference. Re-run via `larql run` for the production path."
            );
        }
    }

    println!();
    println!("=== lm-head top-1 token comparison ===");
    let (ref_name, ref_v) = &traces[0];
    let ref_top1 = argmax(ref_v);
    println!("  {ref_name:<28}  top-1 token = {ref_top1}");
    for (name, v) in traces.iter().skip(1) {
        let top1 = argmax(v);
        let verdict = if top1 == ref_top1 {
            "✓ matches reference"
        } else {
            "✗ DIFFERENT TOP-1 — likely format mismatch (Q4_K vs Q4_0, transposition, ...)"
        };
        println!("  {name:<28}  top-1 token = {top1}   {verdict}");
    }
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── moe-expert: one expert's forward pass (proven correct in v0) ─────────────

fn run_moe_expert(
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let arch = &*weights.arch;
    let hidden = config.hidden_size;
    let inter = arch.moe_intermediate_size();
    let inter_padded = inter.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
        * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let num_experts = arch.num_experts();
    if args.expert >= num_experts {
        return Err(format!(
            "expert {} out of range (model has {num_experts})",
            args.expert
        )
        .into());
    }

    let (gu_bytes, dn_bytes) = expert_bytes(weights, args.layer, args.expert)?;
    let pre_norm = pre_experts_norm_for(weights, args.layer);
    let activation = activation_for(arch);
    let h = make_residual(hidden, args.seed);

    println!("Expert: {}", args.expert);
    println!(
        "Per-expert bytes: gate_up={} ({:.2} MB), down={} ({:.2} MB)",
        gu_bytes.len(),
        gu_bytes.len() as f64 / 1e6,
        dn_bytes.len(),
        dn_bytes.len() as f64 / 1e6,
    );
    println!();

    let mut traces: Vec<(&str, Vec<f32>)> = Vec::new();
    for backend in backends {
        let out = match *backend {
            "reference" => reference_one_expert(
                &h,
                gu_bytes,
                dn_bytes,
                hidden,
                inter,
                inter_padded,
                pre_norm,
                arch.norm_weight_offset(),
                arch.norm_eps(),
                activation,
                args.verbose,
            ),
            "cpu" => run_single_expert_with_norm(
                &h,
                gu_bytes,
                dn_bytes,
                inter,
                pre_norm,
                arch.norm_weight_offset(),
                arch.norm_eps(),
                QuantFormat::Q4_K,
                activation,
            ),
            _ => return Err(format!("backend '{backend}' not yet wired for moe-expert").into()),
        };
        traces.push((backend, out));
    }

    println!("=== expert_output diff ===");
    diff_against_first(&traces, args.tolerance);
    Ok(())
}

// ── moe-block: full block — router + top-K + K experts + sum + post-norm ─────
//
// This is the v1 component that should localise the current Gemma 4 26B-A4B
// CPU MoE bug — per-expert compute is already proven correct (see v0
// prototype), so divergence here means routing or combination is off.

fn run_moe_block(
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let arch = &*weights.arch;
    let hidden = config.hidden_size;
    let inter = arch.moe_intermediate_size();
    let inter_padded = inter.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
        * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let num_experts = arch.num_experts();
    let top_k = arch.num_experts_per_token();

    let h = make_residual(hidden, args.seed);
    let pre_norm = pre_experts_norm_for(weights, args.layer);
    let post_norm = post_experts_norm_for(weights, args.layer);
    let router_proj = router_proj_for(weights, arch, args.layer)?;
    let router_per_expert_scale = router_per_expert_scale_for(weights, arch, args.layer);
    let router_norm = router_norm_for(weights, arch, args.layer);
    let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
    let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);
    let activation = activation_for(arch);
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    println!(
        "Block: layer {} of {}, hidden={hidden}, inter={inter} (padded {inter_padded}), \
         experts={num_experts} top_k={top_k}",
        args.layer, config.num_layers
    );
    println!();

    // Build per-expert byte tables once — both backends consume the same.
    let mut experts_gate_up: Vec<&[u8]> = Vec::with_capacity(num_experts);
    let mut experts_down: Vec<&[u8]> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let (gu, dn) = expert_bytes(weights, args.layer, e)?;
        experts_gate_up.push(gu);
        experts_down.push(dn);
    }

    let moe = MoeLayerWeights {
        experts_gate_up: experts_gate_up.clone(),
        experts_down: experts_down.clone(),
        routing_policy: match arch.moe_router_type() {
            "gemma4_top_k_softmax" => MoeRoutingPolicy::gemma4_hybrid(),
            _ => MoeRoutingPolicy::top_k_softmax(),
        },
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &router_proj,
        router_scale: &[],
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm: pre_norm,
        post_ffn1_norm: &[],
        post_experts_norm: post_norm,
        num_experts,
        top_k,
        intermediate_size: inter,
        activation,
    };

    let mut traces: Vec<(&str, Vec<f32>)> = Vec::new();
    for backend in backends {
        let out = match *backend {
            "reference" => reference_moe_block(
                &h,
                &experts_gate_up,
                &experts_down,
                &router_proj,
                &router_per_expert_scale,
                &router_norm,
                router_norm_parameter_free,
                router_input_scalar,
                pre_norm,
                post_norm,
                hidden,
                inter,
                inter_padded,
                num_experts,
                top_k,
                activation,
                norm_offset,
                eps,
                args.verbose,
            ),
            "cpu" => cpu_moe_forward(&h, &moe, norm_offset, eps),
            _ => return Err(format!("backend '{backend}' not yet wired for moe-block").into()),
        };
        traces.push((backend, out));
    }

    println!("=== moe_block_output diff ===");
    diff_against_first(&traces, args.tolerance);

    // Side-by-side routing-convention check: which top-K does each
    // convention select? Per HF Gemma4TextDecoderLayer.forward, the router
    // consumes the raw post-attention residual; experts consume
    // pre_experts_norm(residual). If h_norm and raw_h pick different
    // experts, mis-routing the input is what produces "fluent but wrong"
    // generation.
    println!();
    println!("=== Routing-convention comparison ===");
    let h_norm = naive_rms_norm(&h, pre_norm, eps, norm_offset);
    let (idx_raw, w_raw) = compute_top_k(
        &h,
        &router_proj,
        &router_per_expert_scale,
        &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        num_experts,
        top_k,
        hidden,
        eps,
        norm_offset,
    );
    let (idx_norm, w_norm) = compute_top_k(
        &h_norm,
        &router_proj,
        &router_per_expert_scale,
        &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        num_experts,
        top_k,
        hidden,
        eps,
        norm_offset,
    );
    println!("  router_in=raw_h    top_k: {idx_raw:?}");
    println!(
        "    weights:                 {}",
        w_raw
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("  router_in=h_norm   top_k: {idx_norm:?}  ← Metal/GPU convention");
    println!(
        "    weights:                 {}",
        w_norm
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let same: Vec<usize> = idx_raw
        .iter()
        .filter(|&&e| idx_norm.contains(&e))
        .copied()
        .collect();
    if same.len() == top_k {
        println!("  ✓ SAME top-{top_k} experts selected — routing input choice is not the bug");
    } else {
        println!(
            "  ✗ DIFFERENT top-{top_k}: {} overlap, {} differ — expert-selection convention IS the bug surface",
            same.len(),
            top_k - same.len()
        );
    }
    Ok(())
}

// ── layer: full hybrid-MoE layer CPU vs Metal residual diff ──────────────────
//
// Runs CPU `predict_kquant_hidden` and Metal `generate` on the same prompt with
// their respective dump hooks enabled, then compares per-layer residuals.
//
// CPU dumps:   LARQL_CPU_DUMP_LAYERS → cpu_layer_{LL}.f32 (last-position row)
//              LARQL_CPU_STAGE_DUMP  → cpu_L0_<stage>.f32
// Metal dump:  LARQL_DUMP_RESIDUALS  → binary (LARQL_RES_V2 header, then per-
//              layer records: u32 layer_idx, u32 hidden, f32[hidden] layer_in,
//              f32[hidden] h_post_attn, f32[hidden] layer_out)
//
// The comparison is decode-step vs prefill-last-token, so the two are in
// slightly different compute contexts (Metal uses KV cache; CPU re-processes
// the full sequence). This is sufficient to locate the first diverging layer
// but not to compute precise numeric agreement.

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_layer_diff(
    path: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    args: &ParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::{generate::generate, CachedLayerGraph};
    use larql_inference::vindex::predict_kquant_hidden;

    let num_layers = config.num_layers;
    let hidden = config.hidden_size;

    let prompt = args.prompt.as_deref().unwrap_or("The capital of France is");

    println!("Prompt:    {prompt:?}");
    println!("Backends:  metal (reference) → cpu");
    println!();

    // ── Set up temp dirs for dump files ─────────────────────────────────────
    let base = std::env::temp_dir().join(format!("larql_parity_{}", std::process::id()));
    let cpu_path_buf = base.join("cpu");
    let metal_path_buf = base.join("metal_residuals.bin");
    let metal_dense_dir = base.join("metal_dense");
    std::fs::create_dir_all(&cpu_path_buf)?;
    let cpu_path = cpu_path_buf.as_path();
    let metal_path = metal_path_buf.as_path();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(base);

    // ── Load vindex (shared mmap; two weight copies for the two runs) ────────
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb)?;
    index.load_attn_kquant(path)?;
    index.load_interleaved_kquant(path)?;
    let _ = index.load_lm_head_kquant(path);
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;
    let mut w_metal = larql_vindex::load_model_weights_kquant(path, &mut cb)?;
    let mut w_cpu = larql_vindex::load_model_weights_kquant(path, &mut cb)?;

    let wrapped = larql_inference::wrap_chat_prompt(path, Some(config.model.as_str()), prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrapped.prompt)?;
    println!("  seq_len: {} tokens post-template", token_ids.len());
    println!();

    // The MoE decode path writes a single LARQL_DUMP_RESIDUALS binary
    // covering every layer; the dense Metal decode path doesn't fire that
    // hook (it only runs in the MoE branch of decode_token_with_moe_split_fn).
    // For dense models we use LARQL_METAL_DUMP_LAYERS, which fires inside
    // prefill_kquant and writes one file per layer (metal_layer_NN_h_out.f32 +
    // metal_layer_NN_h_post_attn.f32). This aligns with the CPU dumps,
    // which are also captured during prefill.
    let is_moe = w_metal.arch.is_hybrid_moe();
    if !is_moe {
        std::fs::create_dir_all(&metal_dense_dir)?;
    }

    // ── Metal run (reference — produces correct output) ──────────────────────
    if is_moe {
        std::env::set_var("LARQL_DUMP_RESIDUALS", metal_path);
    } else {
        std::env::set_var("LARQL_METAL_DUMP_LAYERS", &metal_dense_dir);
    }
    println!("Running Metal…");
    let metal_result = {
        let backend = larql_compute_metal::MetalBackend::new()
            .ok_or("Metal backend unavailable — build with `--features metal` on M-series Mac")?;
        let cache = CachedLayerGraph::from_residuals(Vec::new());
        generate(
            &mut w_metal,
            &tokenizer,
            &token_ids,
            1,
            &index,
            &backend,
            &cache,
            0..num_layers,
        )
    };
    std::env::remove_var("LARQL_DUMP_RESIDUALS");
    std::env::remove_var("LARQL_METAL_DUMP_LAYERS");
    println!("  Metal output: {:?}", metal_result.text().trim());

    // ── CPU run ──────────────────────────────────────────────────────────────
    std::env::set_var("LARQL_CPU_DUMP_LAYERS", cpu_path);
    std::env::set_var("LARQL_CPU_STAGE_DUMP", cpu_path);
    println!("Running CPU…");
    predict_kquant_hidden(&mut w_cpu, &token_ids, &index, None);
    std::env::remove_var("LARQL_CPU_DUMP_LAYERS");
    std::env::remove_var("LARQL_CPU_STAGE_DUMP");

    // ── Load per-layer Metal output ──────────────────────────────────────────
    // MoE: parse binary residual dump (richer — includes h_post_attn).
    // Dense: read decode_layer_NN.f32 written by LARQL_DECODE_DUMP_LAYERS.
    let metal_layers: std::collections::BTreeMap<usize, ResidualRecord> = if is_moe {
        let metal_bytes = std::fs::read(metal_path)?;
        let parsed = parse_residual_dump(&metal_bytes);
        if parsed.is_empty() {
            return Err(
                "Metal residual dump is empty — LARQL_DUMP_RESIDUALS may not have fired".into(),
            );
        }
        parsed.into_iter().collect()
    } else {
        // Prefill dumps: metal_layer_NN_h_out.f32 (post-FFN residual) and
        // metal_layer_NN_h_post_attn.f32 (post-attention residual).
        // Both have shape [seq_len * hidden]; we take the last position.
        let last_pos_slice = |v: Vec<f32>| -> Vec<f32> {
            let n = v.len() / hidden;
            if n == 0 {
                v
            } else {
                v[(n - 1) * hidden..].to_vec()
            }
        };
        let mut out = std::collections::BTreeMap::new();
        for l in 0..num_layers {
            let h_out_path = metal_dense_dir.join(format!("metal_layer_{l:02}_h_out.f32"));
            let h_pa_path = metal_dense_dir.join(format!("metal_layer_{l:02}_h_post_attn.f32"));
            let layer_out = match read_parity_f32(&h_out_path) {
                Some(v) => last_pos_slice(v),
                None => continue,
            };
            let h_post_attn = read_parity_f32(&h_pa_path)
                .map(last_pos_slice)
                .unwrap_or_default();
            out.insert(
                l,
                ResidualRecord {
                    h_post_attn,
                    layer_out,
                },
            );
        }
        if out.is_empty() {
            return Err(
                "Metal dense dump is empty — LARQL_METAL_DUMP_LAYERS may not have fired".into(),
            );
        }
        out
    };

    // ── Compare per layer ────────────────────────────────────────────────────
    println!();
    println!("━━━ Layer-by-layer residual diff (Metal = reference) ━━━━━━━━━━");
    println!(
        "  {:>3}  {:>10}  {:>10}  {:>10}  {:>12}  note",
        "L", "cos(h_pa)", "cos(h_out)", "‖cpu‖", "‖metal‖"
    );
    println!("  {}", "─".repeat(72));

    const DRIFT: f32 = 0.9999;
    let mut first_bad: Option<usize> = None;

    for l in 0..num_layers {
        let cpu_out_path = cpu_path.join(format!("cpu_layer_{l:02}.f32"));
        let cpu_pa_path = cpu_path.join(format!("cpu_layer_{l:02}_h_post_attn.f32"));

        let cpu_out = match read_parity_f32(&cpu_out_path) {
            Some(v) => v,
            None => {
                println!("  L{l:02}  <cpu dump missing>");
                continue;
            }
        };
        let metal_rec = match metal_layers.get(&l) {
            Some(r) => r,
            None => {
                println!("  L{l:02}  <metal dump missing>");
                continue;
            }
        };

        // CPU dump has (seq_len × hidden) elements; take the last position.
        let seq_positions = cpu_out.len() / hidden;
        let cpu_last = if seq_positions > 0 {
            cpu_out[(seq_positions - 1) * hidden..].to_vec()
        } else {
            cpu_out.clone()
        };

        let cos_out = naive_cos_sim(&cpu_last, &metal_rec.layer_out);
        let norm_cpu = naive_rms_mag(&cpu_last);
        let norm_mtl = naive_rms_mag(&metal_rec.layer_out);

        // Dense path doesn't capture h_post_attn separately, so cos(h_pa)
        // is only computed when we have it (MoE).
        let cos_pa = if metal_rec.h_post_attn.is_empty() {
            None
        } else {
            read_parity_f32(&cpu_pa_path).map(|v| {
                let n = v.len() / hidden;
                let last = if n > 0 {
                    v[(n - 1) * hidden..].to_vec()
                } else {
                    v
                };
                naive_cos_sim(&last, &metal_rec.h_post_attn)
            })
        };

        if cos_out < DRIFT && first_bad.is_none() {
            first_bad = Some(l);
        }
        let flag = if cos_out < DRIFT { " ←" } else { "" };
        let note = match cos_pa {
            Some(ca) if ca < DRIFT && cos_out < DRIFT => "attn+ffn",
            Some(ca) if ca < DRIFT => "attn",
            Some(_) if cos_out < DRIFT => "ffn/moe",
            Some(_) => "clean",
            None => "?",
        };
        let hpa_s = cos_pa
            .map(|c| format!("{c:>10.6}"))
            .unwrap_or_else(|| "         -".into());
        println!(
            "  L{l:02}  {hpa_s}  {cos_out:>10.6}  {norm_cpu:>10.4}  {norm_mtl:>12.4}  {note}{flag}"
        );
    }

    println!();
    match first_bad {
        Some(l) => {
            println!("First divergence at L{l} (cos < {DRIFT}).");
            let note = if l == 0 {
                "L0 drift — culprit is embedding, pre-norm, attention, or MoE combine."
            } else {
                "Earlier layers match; drift introduced at this layer."
            };
            println!("{note}");
        }
        None => {
            println!("All layers match within cos ≥ {DRIFT}.");
            println!("Note: Metal decode vs CPU prefill — slight positional mismatch expected.");
        }
    }

    Ok(())
}

/// Per-layer record from `LARQL_DUMP_RESIDUALS` binary.
struct ResidualRecord {
    h_post_attn: Vec<f32>,
    layer_out: Vec<f32>,
}

/// Parse `LARQL_DUMP_RESIDUALS` binary (written by `moe_combine.rs / diag.rs`).
/// Returns a map from layer_idx → record. Skips the 16-byte magic header.
fn parse_residual_dump(bytes: &[u8]) -> std::collections::HashMap<usize, ResidualRecord> {
    let mut map = std::collections::HashMap::new();
    if bytes.len() < 16 {
        return map;
    }
    let mut pos = 16usize; // skip magic
    while pos + 8 <= bytes.len() {
        let layer_idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let n_bytes = hidden * 4;
        if pos + n_bytes * 3 > bytes.len() {
            break;
        }
        let layer_in: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let h_post_attn: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let layer_out: Vec<f32> = bytes[pos..pos + n_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += n_bytes;
        let _ = layer_in; // used for format validation only
        map.insert(
            layer_idx,
            ResidualRecord {
                h_post_attn,
                layer_out,
            },
        );
    }
    map
}

fn read_parity_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

fn naive_cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

fn naive_rms_mag(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

// ── Reference impls (slow + naive) ────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn reference_one_expert(
    h: &[f32],
    gu_bytes: &[u8],
    dn_bytes: &[u8],
    hidden: usize,
    inter: usize,
    inter_padded: usize,
    pre_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    activation: Activation,
    verbose: bool,
) -> Vec<f32> {
    let h_norm = naive_rms_norm(h, pre_norm, eps, norm_offset);
    if verbose {
        dump3("ref h_norm", &h_norm);
    }
    let gate_up_w = dequantize_q4_k(gu_bytes, 2 * inter * hidden);
    let down_w = dequantize_q4_k(dn_bytes, hidden * inter_padded);

    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = naive_matvec(&h_norm, gate_w, inter, hidden);
    let up_out = naive_matvec(&h_norm, up_w, inter, hidden);
    if verbose {
        dump3("ref gate_out", &gate_out);
        dump3("ref up_out  ", &up_out);
    }

    let mut hidden_state = vec![0.0f32; inter_padded];
    for j in 0..inter {
        hidden_state[j] = match activation {
            Activation::GeluTanh => naive_gelu_tanh(gate_out[j]) * up_out[j],
            _ => naive_silu(gate_out[j]) * up_out[j],
        };
    }
    naive_matvec(&hidden_state, &down_w, hidden, inter_padded)
}

#[allow(clippy::too_many_arguments)]
fn reference_moe_block(
    h: &[f32],
    experts_gate_up: &[&[u8]],
    experts_down: &[&[u8]],
    router_proj: &[f32],
    router_per_expert_scale: &[f32],
    router_norm: &[f32],
    router_norm_parameter_free: bool,
    router_input_scalar: f32,
    pre_norm: &[f32],
    post_norm: &[f32],
    hidden: usize,
    inter: usize,
    inter_padded: usize,
    num_experts: usize,
    top_k: usize,
    activation: Activation,
    norm_offset: f32,
    eps: f32,
    verbose: bool,
) -> Vec<f32> {
    // 1. Pre-experts norm — for the expert matmuls.
    let h_norm = naive_rms_norm(h, pre_norm, eps, norm_offset);
    if verbose {
        dump3("ref h_norm        ", &h_norm);
    }

    // 2. Router input norm — applied to h_norm (matching Metal's
    //    `cpu_moe_route(&h_norm, ...)` and the routing-convention fix
    //    in `cpu_moe_forward`). Empirically the trained 26B-A4B weights
    //    expect this even though HF's modeling_gemma4.py uses raw h.
    let router_in_normed = if !router_norm.is_empty() {
        naive_rms_norm(&h_norm, router_norm, eps, norm_offset)
    } else if router_norm_parameter_free {
        naive_rms_norm(&h_norm, &[], eps, 0.0)
    } else {
        h_norm.clone()
    };
    let mut router_in = router_in_normed;
    if router_input_scalar != 1.0 && router_input_scalar != 0.0 {
        for v in router_in.iter_mut() {
            *v *= router_input_scalar;
        }
    }
    if verbose {
        dump3("ref router_in     ", &router_in);
    }

    // 3. Router projection [hidden → num_experts].
    let mut logits = naive_matvec(&router_in, router_proj, num_experts, hidden);
    naive_softmax(&mut logits);

    // 4. Top-K + renormalisation.
    let (indices, mut weights) = naive_top_k(&logits, top_k);
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    if !router_per_expert_scale.is_empty() {
        for (i, &ei) in indices.iter().enumerate() {
            if ei < router_per_expert_scale.len() {
                weights[i] *= router_per_expert_scale[ei];
            }
        }
    }
    if verbose {
        println!(
            "  ref top_k indices: {:?}  weights: {:?}",
            indices,
            weights
                .iter()
                .map(|w| format!("{w:.4}"))
                .collect::<Vec<_>>()
        );
    }

    // 5. Sum K weighted expert outputs.
    let mut moe_out = vec![0.0f32; hidden];
    for (k, &ei) in indices.iter().enumerate() {
        let w = weights[k];
        if w == 0.0 {
            continue;
        }
        let contrib = reference_one_expert(
            h,
            experts_gate_up[ei],
            experts_down[ei],
            hidden,
            inter,
            inter_padded,
            pre_norm,
            norm_offset,
            eps,
            activation,
            false,
        );
        for (acc, &v) in moe_out.iter_mut().zip(contrib.iter()) {
            *acc += w * v;
        }
    }
    if verbose {
        dump3("ref pre-post-norm ", &moe_out);
    }

    // 6. Post-experts norm.
    if !post_norm.is_empty() {
        moe_out = naive_rms_norm(&moe_out, post_norm, eps, norm_offset);
    }
    moe_out
}

/// Run only the routing portion of the MoE block — return top-K indices +
/// renormalised weights. Used by the routing-convention diff to expose
/// whether two router-input variants pick different experts.
#[allow(clippy::too_many_arguments)]
fn compute_top_k(
    router_in_pre: &[f32],
    router_proj: &[f32],
    router_per_expert_scale: &[f32],
    router_norm: &[f32],
    router_norm_parameter_free: bool,
    router_input_scalar: f32,
    num_experts: usize,
    top_k: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) -> (Vec<usize>, Vec<f32>) {
    let router_in_normed = if !router_norm.is_empty() {
        naive_rms_norm(router_in_pre, router_norm, eps, norm_offset)
    } else if router_norm_parameter_free {
        naive_rms_norm(router_in_pre, &[], eps, 0.0)
    } else {
        router_in_pre.to_vec()
    };
    let mut router_in = router_in_normed;
    if router_input_scalar != 1.0 && router_input_scalar != 0.0 {
        for v in router_in.iter_mut() {
            *v *= router_input_scalar;
        }
    }
    let mut logits = naive_matvec(&router_in, router_proj, num_experts, hidden);
    naive_softmax(&mut logits);
    let (indices, mut weights) = naive_top_k(&logits, top_k);
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    if !router_per_expert_scale.is_empty() {
        for (i, &ei) in indices.iter().enumerate() {
            if ei < router_per_expert_scale.len() {
                weights[i] *= router_per_expert_scale[ei];
            }
        }
    }
    (indices, weights)
}

// ── Naive primitives (f64 accumulators, no BLAS) ──────────────────────────────

fn naive_matvec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_rows];
    for r in 0..out_rows {
        let mut s = 0.0f64;
        for c in 0..in_cols {
            s += (w[r * in_cols + c] as f64) * (x[c] as f64);
        }
        out[r] = s as f32;
    }
    out
}

fn naive_rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64 + eps as f64)
        .sqrt() as f32;
    if w.is_empty() {
        return x.iter().map(|v| v / rms).collect();
    }
    x.iter()
        .zip(w.iter())
        .map(|(v, ww)| (v / rms) * (ww + offset))
        .collect()
}

fn naive_softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    if sum > 0.0 {
        let inv = (1.0 / sum) as f32;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

fn naive_top_k(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    let weights: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
    (idx, weights)
}

fn naive_gelu_tanh(x: f32) -> f32 {
    // sqrt(2 / π); precision capped at f32 range — the tanh approx
    // already saturates well before more digits would matter.
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn naive_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Vindex helpers ────────────────────────────────────────────────────────────

fn expert_bytes(
    weights: &larql_models::ModelWeights,
    layer: usize,
    expert: usize,
) -> Result<(&[u8], &[u8]), Box<dyn std::error::Error>> {
    let gu_key = per_layer_ffn_key(layer, expert, PER_LAYER_FFN_GATE_UP);
    let dn_key = per_layer_ffn_key(layer, expert, PER_LAYER_FFN_DOWN);
    let gu = weights
        .get_packed_bytes(&gu_key)
        .ok_or_else(|| format!("missing per-layer entry: {gu_key}"))?;
    let dn = weights
        .get_packed_bytes(&dn_key)
        .ok_or_else(|| format!("missing per-layer entry: {dn_key}"))?;
    Ok((gu, dn))
}

fn pre_experts_norm_for(weights: &larql_models::ModelWeights, layer: usize) -> &[f32] {
    weights
        .arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

fn post_experts_norm_for(weights: &larql_models::ModelWeights, layer: usize) -> &[f32] {
    weights
        .arch
        .moe_post_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

fn router_proj_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let key = arch
        .moe_router_key(layer)
        .ok_or("arch has no router_proj key for this layer")?;
    weights
        .vectors
        .get(&key)
        .cloned()
        .ok_or_else(|| format!("router_proj not found in weights: {key}").into())
}

fn router_per_expert_scale_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Vec<f32> {
    arch.moe_router_per_expert_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .cloned()
        .unwrap_or_default()
}

fn router_norm_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Vec<f32> {
    arch.moe_router_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .cloned()
        .unwrap_or_default()
}

fn activation_for(arch: &dyn larql_models::ModelArchitecture) -> Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => Activation::GeluTanh,
        _ => Activation::Silu,
    }
}

fn make_residual(hidden: usize, seed: u32) -> Vec<f32> {
    // Deterministic per-(hidden, seed) sin pattern. seed=0 reproduces the
    // canonical pattern used by the bench / parity tests.
    let phase = (seed as f32) * 0.001;
    (0..hidden)
        .map(|i| ((i as f32 + 1.0) * 0.0007 + phase).sin())
        .collect()
}

// ── Diff reporter ─────────────────────────────────────────────────────────────

fn diff_against_first(traces: &[(&str, Vec<f32>)], tolerance: f64) {
    let (ref_name, ref_v) = &traces[0];
    println!(
        "Reference backend: {ref_name}  (first {} elems used as the truth)",
        ref_v.len()
    );
    let n = ref_v.len();
    print!("  {ref_name:<10} [0..3] = [");
    for (i, x) in ref_v.iter().take(3).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:+.4e}", x);
    }
    println!("]");

    for (name, v) in traces.iter().skip(1) {
        if v.len() != n {
            println!(
                "  {name:<10} LENGTH MISMATCH: ref.len={n}, {name}.len={}",
                v.len()
            );
            continue;
        }
        let mut max_abs = 0.0f64;
        let mut max_idx = 0;
        let mut max_a = 0.0f32;
        let mut max_b = 0.0f32;
        let mut nan = 0;
        for (i, (a, b)) in ref_v.iter().zip(v.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                nan += 1;
                continue;
            }
            let d = ((a - b) as f64).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
                max_a = *a;
                max_b = *b;
            }
        }
        let verdict = if max_abs < tolerance {
            "✓ within tolerance"
        } else if max_abs < tolerance * 100.0 {
            "⚠ small drift"
        } else {
            "✗ DIVERGENCE"
        };
        print!("  {name:<10} [0..3] = [");
        for (i, x) in v.iter().take(3).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.4e}", x);
        }
        println!("]");
        println!(
            "             max |Δ|={:.3e}  at idx {}  (ref={:+.4e}, {name}={:+.4e})  {verdict}",
            max_abs, max_idx, max_a, max_b
        );
        if nan > 0 {
            println!("             NaN count: {nan}");
        }
    }
}

fn dump3(label: &str, v: &[f32]) {
    let n = v.len().min(3);
    print!("  {label}: [");
    for (i, x) in v.iter().take(n).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:+.6e}", x);
    }
    if v.len() > n {
        print!(", …]  ({} elems)", v.len());
    } else {
        print!("]");
    }
    println!();
}
