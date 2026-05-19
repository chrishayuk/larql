//! `larql diag <vindex>` — engine diagnostic.
//!
//! Prints which kernel paths the inference layer will pick for this vindex.
//! Designed to catch silent slowdowns (vocab_size=0 forcing the f32 BLAS
//! lm_head fallback, stale 148-byte Q4_K stride forcing all-NaN, missing
//! attention weights forcing predict_honest CPU fallback) at a glance.
//!
//! Two passes:
//!   1. Static — manifest stride validation, file presence, declared
//!      config. Doesn't load the vindex; safe for huge models.
//!   2. Loaded — open via `open_inference_vindex`, report which paths the
//!      production inference loop would actually hit.
//!
//! Optional `--probe`: run a 5-token greedy decode and print the
//! `larql bench`-style per-stage timing breakdown. Catches "everything
//! looks fine on paper but the GPU phase is 2× slower than expected."

use clap::Args;
use larql_vindex::format::filenames::{
    ATTN_WEIGHTS_KQUANT_BIN, ATTN_WEIGHTS_KQUANT_MANIFEST_JSON, ATTN_WEIGHTS_Q4_BIN,
    ATTN_WEIGHTS_Q8_BIN, EMBEDDINGS_BIN, GENERATION_CONFIG_JSON, INDEX_JSON,
    INTERLEAVED_KQUANT_BIN, INTERLEAVED_KQUANT_MANIFEST_JSON, INTERLEAVED_Q4_BIN,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN, LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON, LEGACY_INTERLEAVED_Q4K_BIN,
    LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON, LEGACY_LM_HEAD_Q4_BIN, LM_HEAD_BIN, LM_HEAD_KQUANT_BIN,
    NORMS_BIN, TOKENIZER_CONFIG_JSON, TOKENIZER_JSON, WEIGHT_MANIFEST_JSON,
};

use crate::commands::primary::cache;

#[derive(Args)]
pub struct DiagArgs {
    /// Vindex directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub model: String,

    /// Run a real forward pass and print per-stage timings (5 tokens by default).
    #[arg(long)]
    pub probe: bool,

    /// Token count for `--probe`. Caps at 100 to keep the diagnostic snappy.
    #[arg(long, default_value = "5")]
    pub probe_tokens: usize,
}

/// One row in the lm_head-path resolution table.
struct PathDecision {
    label: &'static str,
    will_fire: bool,
    note: String,
}

pub fn run(args: DiagArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = cache::resolve_model(&args.model)?;
    println!("Engine diagnostic — {}", path.display());
    println!("{}", "=".repeat(70));

    // ── Pass 1: static (config + files + manifests) ──
    let cfg = larql_vindex::load_vindex_config(&path)?;
    println!("\nConfig (index.json):");
    println!("  family            : {}", cfg.family);
    println!("  num_layers        : {}", cfg.num_layers);
    println!("  hidden_size       : {}", cfg.hidden_size);
    println!("  vocab_size        : {}", cfg.vocab_size);
    println!("  intermediate_size : {}", cfg.intermediate_size);
    println!("  dtype             : {:?}", cfg.dtype);
    println!("  quant             : {:?}", cfg.quant);

    println!("\nFiles (inference-relevant):");
    let inference_files = [
        INDEX_JSON,
        TOKENIZER_JSON,
        TOKENIZER_CONFIG_JSON,
        EMBEDDINGS_BIN,
        ATTN_WEIGHTS_KQUANT_BIN,
        ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
        LEGACY_ATTN_WEIGHTS_Q4K_BIN,
        LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
        ATTN_WEIGHTS_Q4_BIN,
        ATTN_WEIGHTS_Q8_BIN,
        INTERLEAVED_KQUANT_BIN,
        INTERLEAVED_KQUANT_MANIFEST_JSON,
        LEGACY_INTERLEAVED_Q4K_BIN,
        LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
        INTERLEAVED_Q4_BIN,
        LM_HEAD_BIN,
        LM_HEAD_KQUANT_BIN,
        LEGACY_LM_HEAD_Q4_BIN,
        NORMS_BIN,
        WEIGHT_MANIFEST_JSON,
        GENERATION_CONFIG_JSON,
    ];
    for fname in inference_files {
        let fpath = path.join(fname);
        if let Ok(meta) = std::fs::metadata(&fpath) {
            if meta.is_file() {
                println!("  ✓ {:<38} {:>10}", fname, human_size(meta.len()));
            }
        } else {
            println!("  - {:<38} {:>10}", fname, "absent");
        }
    }

    // ── Stride validation (the 148-byte block_q4_K class of bugs) ──
    println!("\nStride validation:");
    let stride_status = validate_strides(&path)?;
    println!("  {}", stride_status);

    // ── Pass 2: loaded vindex (which kernels would actually fire) ──
    println!("\nLoading vindex…");
    let index = match larql_inference::open_inference_vindex(&path) {
        Ok(idx) => {
            println!("  ✓ open_inference_vindex succeeded");
            idx
        }
        Err(e) => {
            println!("  ✗ open_inference_vindex FAILED: {e}");
            println!("\nNo further diagnostics — vindex won't load for inference.");
            std::process::exit(2);
        }
    };

    println!("  vocab_size (loaded): {}", index.vocab_size);
    println!("  hidden_size (loaded): {}", index.hidden_size);

    if index.vocab_size == 0 {
        println!("  ⚠  vocab_size = 0 after load — Q4 lm_head fast path will silently bail!");
        println!("     This forces a 4× slower f32 BLAS gemv fallback. See");
        println!("     `load_lm_head_q4_sets_vocab_size_from_file_size` regression test.");
    }

    // ── LM head path resolution ──
    let backend = larql_compute::default_backend();
    println!("\nBackend: {}", backend.name());
    println!(
        "  supports_quant(Q4_K)  : {}",
        backend.supports_quant(::larql_compute::QuantFormat::Q4_K)
    );

    println!("\nLM-head path resolution (which kernel fires per next-token):");
    let path_table = resolve_lm_head_path(&index, backend.as_ref());
    let chosen = path_table.iter().find(|p| p.will_fire);
    for p in &path_table {
        let marker = if p.will_fire { "→" } else { "  " };
        println!("  {marker} {:<24} {}", p.label, p.note);
    }
    if let Some(c) = chosen {
        if c.label.contains("f32 BLAS") {
            println!("\n  ⚠  f32 BLAS fallback is the slowest path (~8 ms/tok on Gemma 3 4B vs");
            println!("     1.9 ms for the Q4 fast path). Check vocab_size and lm_head_q4.bin.");
        }
    } else {
        println!("\n  ⚠  No lm_head path will fire — generation will return empty.");
    }

    // ── Optional probe (real forward pass timing) ──
    if args.probe {
        println!("\nProbe — running {} greedy tokens…", args.probe_tokens);
        match probe_run(&path, &index, args.probe_tokens.min(100)) {
            Ok(report) => println!("{report}"),
            Err(e) => println!("  probe failed: {e}"),
        }
    }

    Ok(())
}

/// Walk every k-quant manifest in the vindex, compare each entry's
/// recorded `length` to `format.expected_bytes(&shape)`. Returns a single
/// line summary; on mismatch, the kernel reads off-stride and produces NaN.
/// Checks both the new kquant-named manifests and their legacy q4k
/// counterparts so existing vindexes still get validated.
fn validate_strides(dir: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let manifests = [
        ATTN_WEIGHTS_KQUANT_MANIFEST_JSON,
        INTERLEAVED_KQUANT_MANIFEST_JSON,
        LEGACY_ATTN_WEIGHTS_Q4K_MANIFEST_JSON,
        LEGACY_INTERLEAVED_Q4K_MANIFEST_JSON,
    ];
    let mut total_clean = 0usize;
    let mut total_bad = 0usize;
    let mut bad_examples: Vec<String> = Vec::new();

    for mname in manifests {
        let mpath = dir.join(mname);
        if !mpath.is_file() {
            continue;
        }
        let json: serde_json::Value = match std::fs::read_to_string(&mpath)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            Some(v) => v,
            None => continue,
        };
        let entries = match json.as_array() {
            Some(arr) => arr,
            None => continue,
        };
        for entry in entries {
            let key = entry["key"].as_str().unwrap_or("?");
            let fmt = match entry["format"].as_str() {
                Some(f) => f,
                None => continue,
            };
            let length = entry["length"].as_u64().unwrap_or(0) as usize;
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let qfmt = match larql_vindex::quant::registry::lookup(fmt) {
                Some(q) => q,
                None => continue,
            };
            if let Some(expected) = qfmt.expected_bytes(&shape) {
                if expected == length {
                    total_clean += 1;
                } else {
                    total_bad += 1;
                    if bad_examples.len() < 3 {
                        bad_examples.push(format!(
                            "{key} ({fmt}, shape {shape:?}): length {length} vs expected {expected}"
                        ));
                    }
                }
            }
        }
    }

    if total_bad == 0 {
        Ok(format!("✓ {total_clean} entries match canonical stride"))
    } else {
        let mut msg =
            format!("✗ {total_bad} entries mismatched, {total_clean} clean — vindex is STALE");
        for ex in &bad_examples {
            msg.push_str(&format!("\n      {ex}"));
        }
        msg.push_str(
            "\n      Likely cause: legacy 148-byte block_q4_K layout. Rebuild the vindex.",
        );
        Ok(msg)
    }
}

/// Simulate the lm_head_topk dispatch to figure out which path will fire.
/// Mirrors `lm_head_knn_backend` in `larql-vindex` so the table reflects
/// real production behaviour without running a real forward.
fn resolve_lm_head_path(
    index: &larql_vindex::VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
) -> Vec<PathDecision> {
    let has_q4_data = index.has_lm_head_kquant();
    let q4_ready = backend.supports_quant(::larql_compute::QuantFormat::Q4_K)
        && has_q4_data
        && index.vocab_size > 0;
    let f16_ready = index.has_lm_head_f16() && index.vocab_size > 0;
    let is_non_cpu_backend =
        backend.as_any().type_id() != std::any::TypeId::of::<larql_compute::CpuBackend>();
    let skip_q4k_env = std::env::var("LARQL_LM_HEAD_SKIP_Q4K").unwrap_or_default();
    let skip_q4k =
        is_non_cpu_backend && matches!(skip_q4k_env.as_str(), "1" | "true" | "on" | "yes");
    let stride32_env = std::env::var("LARQL_LM_HEAD_STRIDE32").unwrap_or_default();
    let stride32_disabled = matches!(stride32_env.as_str(), "0" | "false" | "off" | "no");

    // Default order (since the 2026-05-02 dispatch-geometry fix):
    //   1. Q4_K matvec (q4k_matvec_pipeline) — production default.
    //   2. f16 GEMV — fallback when Q4_K bytes aren't available.
    //   3. f32 KNN (lm_head.bin mmap).
    //   4. f32 BLAS gemv on weights.lm_head.
    //
    // `LARQL_LM_HEAD_SKIP_Q4K=1` skips path 1 and starts at:
    //   1. stride-32 Q4_K (`q4k_matvec_stride32`) when the Q4_K bytes exist
    //      (further suppressed by `LARQL_LM_HEAD_STRIDE32=0`).
    //   2. f16 GEMV, then the same f32 fallbacks.
    let q4_will_fire = q4_ready && !skip_q4k;
    let stride32_first_will_fire = skip_q4k && q4_ready && !stride32_disabled;
    let f16_will_fire = if skip_q4k {
        !stride32_first_will_fire && f16_ready
    } else {
        !q4_will_fire && f16_ready
    };
    let knn_ready =
        !q4_will_fire && !stride32_first_will_fire && !f16_will_fire && index.has_lm_head();
    let bls_fallback = !q4_will_fire && !stride32_first_will_fire && !f16_will_fire && !knn_ready;

    vec![
        PathDecision {
            label: "Q4 matvec (fast, default)",
            will_fire: q4_will_fire,
            note: format!(
                "lm_head_q4 mmap/synth = {}, backend.supports_quant(Q4_K) = {}, skip_q4k override = {}  → default Metal lm_head path post 2026-05-02 dispatch fix",
                has_q4_data,
                backend.supports_quant(::larql_compute::QuantFormat::Q4_K),
                skip_q4k,
            ),
        },
        PathDecision {
            label: "Q4 stride32 stable (skip_q4k)",
            will_fire: stride32_first_will_fire,
            note: format!(
                "available = {}  → diagnostic A/B path, fires only with LARQL_LM_HEAD_SKIP_Q4K=1",
                q4_ready,
            ),
        },
        PathDecision {
            label: "f16 gemv (tied embed)",
            will_fire: f16_will_fire,
            note: format!(
                "lm_head_f16 mmap = {}  → fallback when Q4_K unavailable",
                index.has_lm_head_f16(),
            ),
        },
        PathDecision {
            label: "f32 KNN (lm_head.bin)",
            will_fire: knn_ready,
            note: format!("lm_head.bin mmap = {}  → ~2 ms", index.has_lm_head()),
        },
        PathDecision {
            label: "f32 BLAS gemv (slow)",
            will_fire: bls_fallback,
            note: "no vindex KNN — falls back to weights.lm_head full gemv  → ~8 ms".to_string(),
        },
    ]
}

/// Run the model and return the same per-stage breakdown that `larql bench`
/// prints. Equivalent code path to the `bench` subcommand but trimmed —
/// fewer backends, shorter run, no parity table.
fn probe_run(
    vindex_path: &std::path::Path,
    _index: &larql_vindex::VectorIndex,
    tokens: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    use larql_inference::{default_backend, generate, CachedLayerGraph};

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
    index.load_attn_kquant(vindex_path)?;
    index.load_interleaved_kquant(vindex_path)?;
    let _ = index.load_lm_head_kquant(vindex_path);
    let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;

    let prompt = "The capital of France is";
    let token_ids: Vec<u32> = larql_inference::encode_prompt(&tokenizer, &*weights.arch, prompt)
        .map_err(|e| format!("{e}"))?;

    let backend = default_backend();
    let num_layers = weights.num_layers;
    let cache = CachedLayerGraph::from_residuals(Vec::new());

    // Warmup: allocate KV cache and warm Metal buffer caches.
    let _ = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        3,
        &index,
        &*backend,
        &cache,
        0..num_layers,
    );
    let r = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        tokens,
        &index,
        &*backend,
        &cache,
        0..num_layers,
    );

    let n = r.decode_ms.len() as f64;
    if n == 0.0 {
        return Ok("  (no decode steps recorded)".to_string());
    }
    let avg = r.stage_timings.avg_per_step(r.decode_ms.len());
    let total_per = r.avg_decode_ms();
    let tok_s = r.decode_tok_s();
    Ok(format!(
        "  prefill        {:>7.0} ms\n  per-step embed {:>7.2} ms\n  per-step gpu   {:>7.2} ms\n  per-step norm  {:>7.2} ms\n  per-step lmhd  {:>7.2} ms\n  per-step detok {:>7.2} ms\n  per-step total {:>7.2} ms = {:.1} tok/s",
        r.prefill_ms,
        avg.embed_ms_total,
        avg.gpu_ms_total,
        avg.norm_ms_total,
        avg.lm_head_ms_total,
        avg.detok_ms_total,
        total_per,
        tok_s,
    ))
}

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static stride validation must pass on a clean canonical-stride
    /// manifest and fail on a 148-byte legacy stride.
    #[test]
    fn validate_strides_accepts_canonical_144_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": 2048 * 10 * 144,
            }
        ]);
        std::fs::write(
            tmp.path().join("attn_weights_q4k_manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let result = validate_strides(tmp.path()).unwrap();
        assert!(
            result.starts_with("✓"),
            "clean stride should pass — got: {result}"
        );
    }

    #[test]
    fn validate_strides_rejects_legacy_148_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": 2048 * 10 * 148, // legacy block_q4_K stride
            }
        ]);
        std::fs::write(
            tmp.path().join("attn_weights_q4k_manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let result = validate_strides(tmp.path()).unwrap();
        assert!(
            result.starts_with("✗"),
            "stale stride must fail validation — got: {result}"
        );
        let lower = result.to_lowercase();
        assert!(
            lower.contains("stale") && lower.contains("rebuild"),
            "error must mention STALE + rebuild — got: {result}"
        );
    }

    /// Mixed Q4_K + Q6_K (Gemma-style attn V) — both formats must
    /// validate against their respective `expected_bytes`.
    #[test]
    fn validate_strides_handles_mixed_q4k_q6k() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!([
            {
                "key": "k", "shape": [1024, 2560], "format": "Q4_K",
                "offset": 0, "length": 1024 * 10 * 144,
            },
            {
                "key": "v", "shape": [1024, 2560], "format": "Q6_K",
                "offset": 0, "length": 1024 * 10 * 210,
            }
        ]);
        std::fs::write(
            tmp.path().join("attn_weights_q4k_manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let result = validate_strides(tmp.path()).unwrap();
        assert!(result.starts_with("✓"));
    }

    #[test]
    fn validate_strides_handles_missing_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        // Empty dir — neither manifest exists. Validation reports clean
        // (zero entries) rather than crashing.
        let result = validate_strides(tmp.path()).unwrap();
        assert!(result.starts_with("✓"), "missing manifest is not an error");
    }

    #[test]
    fn human_size_units() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1500), "1.5 KB");
        assert_eq!(human_size(1024 * 1024 * 5), "5.0 MB");
        assert_eq!(human_size(1024 * 1024 * 1024 * 2), "2.00 GB");
    }
}
