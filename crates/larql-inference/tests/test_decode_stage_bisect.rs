//! Per-stage divergence bisector: locates the *first* sub-stage of L0
//! where Metal decode disagrees with CPU prefill.
//!
//! ## Why
//!
//! End-of-layer parity (`test_decode_consistency`) tells us whether L0
//! drifts between Metal-prefill+decode and a fresh CPU prefill. It
//! doesn't tell us which **sub-stage of L0** introduced the drift —
//! input norm? Q projection? QK-norm? RoPE? V-norm? attention? O proj?
//! FFN gate+up? GEGLU? down? When every kernel-level test passes (as
//! it does after the kv_cache_append / rope_at_pos / qk_norm work
//! that cleared roadmap suspects 1 and 2), the only way to localise
//! the open Gemma 4 31B parity gap is to dump every intermediate at
//! L0 from both backends and diff stage-by-stage.
//!
//! [`StageCapture`] does the dumping (env-var plumbing + tempfile
//! lifecycle); [`compare_stages`] walks a stage-pair list and reports
//! the first divergence per the threshold.
//!
//! ## What it asserts
//!
//! For each available test vindex:
//!   - Run a single Metal `prefill(prompt) + decode(t1)` capture at L0.
//!   - Run a CPU prefill of `prompt + t1` and capture L0 from that.
//!   - Compare the canonical pre-attention chain stage-by-stage:
//!     `norm_out`, post-everything Q (= CPU `q_out_after_rope` ↔
//!     Metal `q_out`), K, V, attention output, O projection,
//!     post-attention residual, FFN-norm, FFN down output.
//!
//! Skip semantics mirror the other test_kernel_* / test_decode_*
//! suites: missing vindexes return early with a skip note unless
//! `LARQL_ARCH_STRICT=1`.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use larql_compute::DecodeBackend;
use larql_inference::residual_diff::{compare_stages, ParityThreshold, StageCapture};
use larql_inference::wrap_chat_prompt;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

struct StageCase {
    name: &'static str,
    vindex_name: &'static str,
}

const CASES: &[StageCase] = &[
    StageCase {
        name: "gemma3-4b-it",
        vindex_name: "gemma3-4b-q4k-v2",
    },
    StageCase {
        name: "gemma4-31b-it (dense)",
        vindex_name: "gemma4-31b-q4k",
    },
    StageCase {
        name: "llama2-7b-hf (base)",
        vindex_name: "llama2-7b-q4k",
    },
    StageCase {
        name: "mistral-7b-v0.1 (base)",
        vindex_name: "mistral-7b-v0.1-q4k",
    },
];

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

/// Stage-pair list mapping the CPU dump's per-stage names to the
/// Metal-decode dump's per-stage names.
///
/// The asymmetry is deliberate: CPU prefill captures Q at three points
/// (raw, post-QK-norm, post-RoPE) because each is a separate
/// `Array2<f32>` allocation; Metal decode does the same operations
/// in-place on a single buffer and only sees the post-everything
/// `q_out`. So pairing CPU's `q_out_after_rope` against Metal's
/// `q_out` is the right comparison for the post-attention input.
///
/// Order matters: this is the order [`compare_stages`] walks, and the
/// **first** divergence (per [`ParityThreshold`]) is the localised
/// stage. Coarser stages (norm) are checked before finer ones
/// (per-projection) so a divergence at a coarse stage doesn't get
/// shadowed by downstream amplification.
const STAGE_PAIRS: &[(&str, &str)] = &[
    // Pre-attention
    ("norm_out", "norm_out"),
    ("q_out_after_rope", "q_out"),
    ("k_out_after_rope", "k_out"),
    ("v_out", "v_out"),
    // Attention block
    ("attn_out", "attn_out"),
    ("o_out", "o_out"),
    ("h_post_attn", "h_post_attn"),
    // FFN block
    ("ffn_norm_out", "ffn_norm_out"),
    ("ffn_out_raw", "down_out"),
];

fn check_stage_bisect(case: &StageCase) -> Result<(), String> {
    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                case.name, case.vindex_name
            ));
        }
        eprintln!(
            "[{}] skip: vindex `{}` not found",
            case.name, case.vindex_name
        );
        return Ok(());
    };

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4K {
        return Err(format!("expected Q4K vindex, got {:?}", cfg.quant));
    }
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

    let mut w_metal = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (metal): {e}"))?;
    let mut w_cpu = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (cpu): {e}"))?;

    let prompt = "The capital of France is";
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;

    // Pick a deterministic next token by running one greedy step
    // through Metal, exactly as `test_decode_consistency` does. Keeps
    // the two suites referenced against the same (prompt, t1) pair.
    let cached = larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
    let metal_num_layers = w_metal.num_layers;
    let r0 = larql_inference::layer_graph::generate(
        &mut w_metal,
        &tokenizer,
        &prompt_ids,
        1,
        &index,
        &metal_backend,
        &cached,
        0..metal_num_layers,
    );
    let token_0_text = r0
        .tokens
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();
    if token_0_text.is_empty() {
        return Err(format!("[{}] generate produced no first token", case.name));
    }
    let appended_prompt = format!("{}{}", wrap.prompt, token_0_text);
    let appended_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &appended_prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;
    if appended_ids.len() != prompt_ids.len() + 1 {
        eprintln!(
            "[{}] note: tokeniser merged step-0 token at the prompt boundary; \
             skipping stage-bisect for this combination",
            case.name
        );
        return Ok(());
    }
    let token_0_id = *appended_ids.last().unwrap();

    // Capture L0 stages from both paths. Reset the Metal KV cache
    // before the decode capture so its prefill reproduces
    // `prompt_ids` cleanly.
    metal_backend.reset_kv_cache();
    let metal_stages = StageCapture::metal_decode(
        &mut w_metal,
        &prompt_ids,
        token_0_id,
        &index,
        &metal_backend,
        /*layer*/ 0,
    )?;
    // CPU prefill captures every stage as `[seq_len, stride]`. The
    // Metal-decode capture is single-position. Slice CPU's last
    // position out of every stage so 1:1 comparison works.
    let cpu_stages =
        StageCapture::cpu_prefill(&mut w_cpu, &appended_ids, &index, /*layer*/ 0)?
            .project_to_last_position();

    if cpu_stages.is_empty() {
        return Err(format!(
            "[{}] CPU stage capture empty — env var or path bug",
            case.name
        ));
    }
    if metal_stages.is_empty() {
        return Err(format!(
            "[{}] Metal stage capture empty — env var or path bug",
            case.name
        ));
    }

    // Loose threshold here, not tight. Metal decode and CPU prefill go
    // through different kernel families at every stage (Q4K matvec vs
    // BLAS, fused vs scalar). The kernel-level tests already pin the
    // tight bound; what we want from this bisect is to identify which
    // stage *jumps* (cos drops well below kernel-noise) when something
    // structural diverges.
    let report = compare_stages(
        &cpu_stages,
        &metal_stages,
        STAGE_PAIRS,
        ParityThreshold::loose(),
    );
    eprintln!("[{}] {}", case.name, report.summary());
    report
        .assert_clean()
        .map_err(|e| format!("[{}] L0 stage divergence:\n{e}", case.name))?;
    Ok(())
}

#[test]
fn stage_bisect_gemma3_4b() {
    check_stage_bisect(&CASES[0]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn stage_bisect_gemma4_31b_dense() {
    check_stage_bisect(&CASES[1]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn stage_bisect_llama2_7b() {
    check_stage_bisect(&CASES[2]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn stage_bisect_mistral_7b() {
    check_stage_bisect(&CASES[3]).unwrap_or_else(|e| panic!("{e}"));
}
