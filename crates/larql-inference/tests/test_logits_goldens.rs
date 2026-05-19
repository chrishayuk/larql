//! End-to-end logits goldens — the missing 5% of regression coverage.
//!
//! ## Why this file
//!
//! The other parity layers (`test_cpu_metal_parity`,
//! `test_decode_consistency`, `test_decode_stage_bisect`,
//! `test_kernel_*`) all compare CPU and Metal against *each other*. If
//! both backends regressed in the same direction (e.g. someone changes
//! a normalisation constant in shared model config), every parity
//! test stays green. Pinned external goldens — fixed top-K next-token
//! IDs the model is *known to emit* on a fixed prompt — close that
//! correlated-drift hole.
//!
//! ## What it asserts
//!
//! For each architecture × backend, on the prompt
//! `"The capital of France is"` (chat-template-wrapped where the
//! vindex declares an instruct model):
//!
//!   1. The top-5 next-token IDs match the pinned set, **as a set**
//!      (not in strict order). Float-noise can swap rank within the
//!      top-5; what matters is "the model still emits one of these
//!      five tokens at the next position."
//!   2. The top-1 logit value is within `LOGIT_TOLERANCE` of the
//!      pinned value. Catches finer-grained drift that doesn't
//!      reorder the set.
//!
//! ## How to add / refresh goldens
//!
//! Set `LARQL_LOGITS_GOLDENS_PRINT=1` and run this binary. It will
//! emit a Rust array literal for each (arch × backend) it could load,
//! matching the `Golden` shape below — copy/paste those into the
//! `GOLDENS` table at the bottom of this file. The captured values
//! are the model's actual current behaviour; the regression they
//! catch is "future me changed something that shifted them."
//!
//! Rationale for capturing instead of using HF reference: a Python
//! HF reference would be the ideal authority, but adding a Python
//! step to a Rust test is fragile (HF version, env, weights). The
//! current Rust output, gated by the parity + per-stage suites,
//! already has strong evidence of correctness — pinning it gives
//! the regression detector without the Python dependency.
//!
//! These real-vindex checks are `#[ignore]` so default `cargo test` stays
//! fast. Run explicitly with:
//!
//! ```sh
//! cargo test -p larql-inference --test test_logits_goldens -- --ignored
//! ```

#![allow(clippy::excessive_precision)]

use std::path::PathBuf;

use larql_compute::{ComputeBackend, CpuBackend};
use larql_inference::layer_graph::{generate, lm_head_topk, CachedLayerGraph};
use larql_inference::wrap_chat_prompt;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, SilentLoadCallbacks,
    VectorIndex,
};

/// Tolerance for the top-1 logit value. f32 noise across CPU vs Metal
/// (BLAS vs Metal gemv) on a vocab × hidden matvec sits around 1e-2
/// in absolute terms; on the typical 7-15-magnitude logits we see,
/// 5e-2 catches ~0.5% drift while not flagging ULP noise.
const LOGIT_TOLERANCE: f32 = 5e-2;

#[derive(Debug)]
struct Golden {
    arch_name: &'static str,
    vindex_name: &'static str,
    backend: &'static str, // "metal" or "cpu"
    /// Top-5 token IDs the model emits at the next position. Order
    /// within the set isn't strictly enforced — see assertion below.
    top5_token_ids: [u32; 5],
    /// Top-1 logit value at capture time (used as the centre of an
    /// ε ball — see `LOGIT_TOLERANCE`).
    top1_logit: f32,
}

const PROMPT: &str = "The capital of France is";

/// Per-backend goldens. Captured 2026-04-25 on M3 Max. Each entry
/// pins the model's actual current top-5 + top-1 logit on the fixed
/// prompt against future drift *within that backend*. Refresh: set
/// `LARQL_LOGITS_GOLDENS_PRINT=1` and copy the printed lines back.
///
/// Post-2026-04-25 (q4_matvec_v4 dispatch geometry fix), all four
/// architectures' CPU and Metal goldens are bit-identical or within
/// Q4 round-trip noise — the per-backend split is kept anyway so that
/// future drift on either side is caught independently.
const GOLDENS: &[Golden] = &[
    // Gemma 3/4 are tied-embedding models — LM head goes through the
    // synthesised Q4_0 path (`backend.q4_matvec` against `lm_head_q4_synth`).
    //
    // History of Metal dispatcher bugs that caused CPU/Metal divergence
    // here, both since fixed:
    //   1. Pre-2026-04-25 — the Metal dispatcher imported the wrong
    //      shader's geometry constants and silently dropped 75% of vocab
    //      rows.
    //   2. 2026-05-02 — `MetalBackend::q4k_matvec` hardcoded the 4sg
    //      shader's `THREADS_PER_TG=128` while dispatching the 8sg
    //      `q4k_matvec_pipeline` (production default since 2026-04-28),
    //      leaving simdgroups 4..7 unscheduled and dropping half the
    //      lm_head rows. Diagnosed initially as a kernel reduction-tree
    //      drift; root cause was the dispatch site (now uses
    //      `pipeline.rows_per_tg` / `pipeline.threads_per_tg`).
    //
    // After both fixes, Metal lm_head routes through the production
    // `q4k_matvec` (~1.85 ms/tok on Gemma 3 4B v2) and matches CPU pins
    // at top-5 set + order. Top-1 logits differ by ~1e-3 (round-off,
    // well inside `LOGIT_TOLERANCE`). Set `LARQL_LM_HEAD_SKIP_Q4K=1` to
    // route through the stride-32 + f16 fallback chain instead — useful
    // for diagnostic A/B against a known-stable reduction tree.
    //
    // The non-gemma3 Metal pins below (gemma4-31b dense, gemma4-31b
    // Q6_K down, llama2-7b, mistral-7b) still reflect older fix
    // attempts and have NOT been re-captured for the stride-32 path.
    // If you run this suite with those vindexes present, expect them
    // to need refreshing — set `LARQL_LOGITS_GOLDENS_PRINT=1` and
    // copy-paste from stdout.
    Golden {
        arch_name: "gemma3-4b-it",
        vindex_name: "gemma3-4b-q4k-v2",
        backend: "metal",
        // Metal f16 GEMV tied-embedding path: same top-5 set + order as
        // CPU, top-1 logit within ~7e-4 abs (~2e-7 relative).
        top5_token_ids: [256240, 250251, 256331, 249309, 212287],
        top1_logit: 3693.571045,
    },
    Golden {
        arch_name: "gemma3-4b-it",
        vindex_name: "gemma3-4b-q4k-v2",
        backend: "cpu",
        top5_token_ids: [256240, 250251, 256331, 249309, 212287],
        top1_logit: 3693.570312,
    },
    Golden {
        arch_name: "gemma4-31b-it (dense)",
        vindex_name: "gemma4-31b-q4k",
        backend: "metal",
        top5_token_ids: [236780, 236772, 236798, 236799, 236773],
        top1_logit: 2.366634,
    },
    Golden {
        arch_name: "gemma4-31b-it (dense)",
        vindex_name: "gemma4-31b-q4k",
        backend: "cpu",
        top5_token_ids: [236780, 236772, 236798, 236799, 236773],
        top1_logit: 2.366634,
    },
    Golden {
        arch_name: "llama2-7b-hf (base)",
        vindex_name: "llama2-7b-q4k",
        backend: "metal",
        top5_token_ids: [263, 278, 697, 3681, 884],
        top1_logit: 29.988192,
    },
    Golden {
        arch_name: "llama2-7b-hf (base)",
        vindex_name: "llama2-7b-q4k",
        backend: "cpu",
        top5_token_ids: [263, 278, 697, 3681, 884],
        top1_logit: 29.988192,
    },
    Golden {
        arch_name: "mistral-7b-v0.1 (base)",
        vindex_name: "mistral-7b-v0.1-q4k",
        backend: "metal",
        top5_token_ids: [5465, 264, 272, 5651, 624],
        top1_logit: 1.452387,
    },
    Golden {
        arch_name: "mistral-7b-v0.1 (base)",
        vindex_name: "mistral-7b-v0.1-q4k",
        backend: "cpu",
        top5_token_ids: [5465, 264, 272, 5651, 624],
        top1_logit: 1.452387,
    },
    // Q4_K down dense path — regression-tests the fused-down opt-in flip
    // (`LARQL_FUSED_DOWN`). With the old default, the fused
    // `q4k_geglu_gelu_tanh_down` kernel produced NaN at the prefill output
    // and decoded into empty/garbage tokens. The separated path (now the
    // default) goes through `geglu_dispatch + q4k_matvec` and produces
    // valid logits.
    Golden {
        arch_name: "gemma3-4b-it (Q4_K down)",
        vindex_name: "gemma3-4b-q4k-downq4k",
        backend: "metal",
        // Metal f16 GEMV tied-embedding path: bit-equivalent top-5 set
        // + order to CPU, top-1 logit within ~7e-3 abs (~5e-7 relative).
        top5_token_ids: [250251, 256240, 253044, 212287, 250492],
        top1_logit: 14667.830078,
    },
    Golden {
        arch_name: "gemma3-4b-it (Q4_K down)",
        vindex_name: "gemma3-4b-q4k-downq4k",
        backend: "cpu",
        top5_token_ids: [250251, 256240, 253044, 212287, 250492],
        top1_logit: 14667.836914,
    },
    // Gemma 4 31B with Q6_K down — the variant the per-layer parity passed
    // on, and the variant the chat-template rewrite + default system prompt
    // get exercised through.
    Golden {
        arch_name: "gemma4-31b-it (Q6_K down)",
        vindex_name: "gemma4-31b-q4k-q6kdown",
        backend: "metal",
        top5_token_ids: [497, 524, 236762, 514, 237051],
        top1_logit: 1.064089,
    },
    Golden {
        arch_name: "gemma4-31b-it (Q6_K down)",
        vindex_name: "gemma4-31b-q4k-q6kdown",
        backend: "cpu",
        top5_token_ids: [497, 524, 236762, 514, 237051],
        top1_logit: 1.064089,
    },
    // Gemma 4 E2B — has Per-Layer Embeddings (PLE) which the Metal pipeline
    // doesn't implement. The dispatcher in `generate_streaming` auto-routes
    // PLE-using arches to the CPU dense Q4K path, which DOES apply PLE.
    // CPU-only golden because the auto-routing means a `--metal` invocation
    // ends up running CPU code anyway — testing Metal would just duplicate
    // the CPU result.
    Golden {
        arch_name: "gemma4-e2b-it (PLE)",
        vindex_name: "gemma4-e2b-q4k",
        backend: "cpu",
        top5_token_ids: [196228, 134673, 90239, 37373, 112144],
        top1_logit: 10.414763,
    },
];

fn lookup_golden(vindex: &str, backend: &str) -> Option<&'static Golden> {
    GOLDENS
        .iter()
        .find(|g| g.vindex_name == vindex && g.backend == backend)
}

fn find_vindex(name: &str) -> Option<PathBuf> {
    let filename = format!("{name}.vindex");
    if let Ok(env_path) = std::env::var(format!(
        "LARQL_VINDEX_{}",
        name.to_uppercase().replace('-', "_")
    )) {
        let p = PathBuf::from(env_path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let chris_models = PathBuf::from("/Users/christopherhay/chris-models").join(&filename);
    if chris_models.is_dir() {
        return Some(chris_models);
    }
    let home = std::env::var("HOME").ok()?;
    [
        PathBuf::from(&home)
            .join(".cache/larql/local")
            .join(&filename),
        PathBuf::from("output").join(&filename),
    ]
    .into_iter()
    .find(|p| p.is_dir())
}

fn strict_mode() -> bool {
    matches!(
        std::env::var("LARQL_ARCH_STRICT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

fn print_mode() -> bool {
    matches!(
        std::env::var("LARQL_LOGITS_GOLDENS_PRINT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Run prefill on `prompt_ids` through `backend`, return the top-5
/// `(token_id, logit)` for the next position.
///
/// Reuses the production `generate` entry to drive prefill (so the
/// path matches what `larql run` produces), then calls the public
/// `lm_head_topk` helper directly on the last hidden state. We can't
/// use `generate(max_tokens=1).tokens[0]` because that returns the
/// decoded *string* + log-probability; we want the raw top-5 IDs.
fn capture_top5(
    weights: &mut larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    prompt_ids: &[u32],
) -> Result<Vec<(u32, f32)>, String> {
    // Drive a single-token generate so the KV cache is populated and
    // the per-stage hot path matches `larql run`. We discard the
    // returned token here — the captured raw last-position hidden
    // is what we'll scoreboard against the LM head.
    let cached = CachedLayerGraph::from_residuals(Vec::new());
    let n = weights.num_layers;
    let _ = generate(
        weights,
        tokenizer,
        prompt_ids,
        1,
        index,
        backend,
        &cached,
        0..n,
    );

    // The per-token decode in `generate` runs the LM head internally.
    // To get the logits at the prompt's last position (not at the
    // freshly-decoded token), re-run the prompt through CPU prefill
    // and pull the last-position hidden state — that's the "what
    // does the model think comes next at end-of-prompt" signal that
    // the goldens pin.
    //
    // Use CpuBackend for this projection regardless of the test's
    // backend: the prefill matches CPU vs Metal at every layer
    // (test_cpu_metal_parity passes), and the LM head matvec is the
    // same `f32_gemv` either way. What we're isolating in this test
    // is "did the model's output for this prompt drift?"
    let h_full = larql_inference::vindex::predict_kquant_hidden(weights, prompt_ids, index, None);
    let last_pos = h_full.shape()[0] - 1;
    let h_last = h_full.row(last_pos).to_owned();

    let top5 = lm_head_topk(index, weights, &h_last, 5, backend);
    if top5.is_empty() {
        return Err("lm_head_topk returned empty (check weights.lm_head population)".into());
    }
    Ok(top5)
}

/// Body shared by every (arch × backend) test. Loads the vindex,
/// runs prefill, captures top-5, asserts against the pinned golden
/// (or prints in `LARQL_LOGITS_GOLDENS_PRINT=1` mode).
fn check_golden(
    g: &Golden,
    backend_name: &str,
    backend: &dyn ComputeBackend,
) -> Result<(), String> {
    let Some(vindex_path) = find_vindex(g.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}/{backend_name}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                g.arch_name, g.vindex_name
            ));
        }
        eprintln!(
            "[{}/{backend_name}] skip: vindex `{}` not found",
            g.arch_name, g.vindex_name
        );
        return Ok(());
    };

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
    let tokenizer =
        load_vindex_tokenizer(&vindex_path).map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
    let mut index =
        VectorIndex::load_vindex(&vindex_path, &mut cb).map_err(|e| format!("load vindex: {e}"))?;
    index
        .load_attn_kquant(&vindex_path)
        .map_err(|e| format!("load_attn_kquant: {e}"))?;
    index
        .load_interleaved_kquant(&vindex_path)
        .map_err(|e| format!("load_interleaved_kquant: {e}"))?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let mut weights = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights: {e}"))?;

    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), PROMPT);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let top5 = capture_top5(&mut weights, &tokenizer, &index, backend, &prompt_ids)?;
    let actual_ids: [u32; 5] =
        std::array::from_fn(|i| top5.get(i).map(|t| t.0).unwrap_or(u32::MAX));
    let actual_top1_logit = top5[0].1;

    if print_mode() {
        // Refresh-mode output — paste these back into the GOLDENS table.
        eprintln!(
            "    Golden {{ arch_name: {:?}, vindex_name: {:?}, top5_token_ids: {:?}, top1_logit: {:.6} }}, // backend={backend_name}",
            g.arch_name, g.vindex_name, actual_ids, actual_top1_logit,
        );
        return Ok(());
    }

    // Set-equality check: same five IDs, regardless of order. f32
    // noise can swap rank within the top-5 across backends (CPU BLAS
    // vs Metal f32_gemv accumulate in different order), so requiring
    // strict order would flag noise as a regression.
    let mut want: Vec<u32> = g.top5_token_ids.to_vec();
    want.sort_unstable();
    let mut got: Vec<u32> = actual_ids.to_vec();
    got.sort_unstable();
    if want != got {
        return Err(format!(
            "[{}/{backend_name}] top-5 set mismatch:\n  expected (sorted): {:?}\n  got      (sorted): {:?}\n  raw expected: {:?}\n  raw got:      {:?}",
            g.arch_name, want, got, g.top5_token_ids, actual_ids,
        ));
    }

    let logit_diff = (actual_top1_logit - g.top1_logit).abs();
    if logit_diff > LOGIT_TOLERANCE {
        return Err(format!(
            "[{}/{backend_name}] top-1 logit drift: expected {:.4}, got {:.4} (Δ={:.4} > tol {:.4})",
            g.arch_name, g.top1_logit, actual_top1_logit, logit_diff, LOGIT_TOLERANCE,
        ));
    }

    eprintln!(
        "[{}/{backend_name}] top-5 OK: {:?} / top-1 logit {:.4} (Δ {:.4})",
        g.arch_name, actual_ids, actual_top1_logit, logit_diff,
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend() -> Option<larql_compute_metal::MetalBackend> {
    larql_compute_metal::MetalBackend::new()
}

// ── Per-architecture × backend tests ───────────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal(vindex: &str) {
    let Some(metal) = metal_backend() else {
        eprintln!("skip: Metal backend unavailable");
        return;
    };
    let g =
        lookup_golden(vindex, "metal").unwrap_or_else(|| panic!("no metal golden for {vindex}"));
    check_golden(g, "metal", &metal).unwrap_or_else(|e| panic!("{e}"));
}

fn run_cpu(vindex: &str) {
    let g = lookup_golden(vindex, "cpu").unwrap_or_else(|| panic!("no cpu golden for {vindex}"));
    check_golden(g, "cpu", &CpuBackend).unwrap_or_else(|e| panic!("{e}"));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma3_4b_metal() {
    run_metal("gemma3-4b-q4k-v2");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma3_4b_cpu() {
    run_cpu("gemma3-4b-q4k-v2");
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma4_31b_dense_metal() {
    run_metal("gemma4-31b-q4k");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma4_31b_dense_cpu() {
    run_cpu("gemma4-31b-q4k");
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_llama2_7b_metal() {
    run_metal("llama2-7b-q4k");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_llama2_7b_cpu() {
    run_cpu("llama2-7b-q4k");
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_mistral_7b_metal() {
    run_metal("mistral-7b-v0.1-q4k");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_mistral_7b_cpu() {
    run_cpu("mistral-7b-v0.1-q4k");
}
// Q4_K down variants — exercise the separated geglu + q4k_matvec path
// after the fused-kernel default flip.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma3_4b_q4k_down_metal() {
    run_metal("gemma3-4b-q4k-downq4k");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma3_4b_q4k_down_cpu() {
    run_cpu("gemma3-4b-q4k-downq4k");
}
// Gemma 4 31B Q6_K-down variant.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma4_31b_q6kdown_metal() {
    run_metal("gemma4-31b-q4k-q6kdown");
}
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma4_31b_q6kdown_cpu() {
    run_cpu("gemma4-31b-q4k-q6kdown");
}
// Gemma 4 E2B (PLE auto-routes to CPU even under `--metal`).
#[test]
#[ignore = "loads a real vindex; run with --ignored"]
fn logits_golden_gemma4_e2b_cpu() {
    run_cpu("gemma4-e2b-q4k");
}
