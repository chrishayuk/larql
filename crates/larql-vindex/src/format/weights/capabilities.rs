use crate::config::ExtractLevel;
use crate::error::VindexError;

pub(super) const SURFACE_F32_WEIGHT_WRITER: &str = "f32 weight writer";
pub(super) const SURFACE_Q4K_WEIGHT_WRITER: &str = "q4k weight writer";
pub(crate) const SURFACE_EXTRACT_PIPELINE: &str = "extract pipeline";

const FEATURE_MLA: &str = "multi-head latent attention (MLA)";

/// Ensure the current vindex weight layout can represent this architecture's
/// attention tensors.
///
/// The existing f32 and Q4K manifests store standard decoder attention as
/// Q/K/V/O tensors. Architectures such as DeepSeek MLA expose a different
/// tensor contract (`mla_*`) and must be implemented explicitly before the
/// writer accepts them.
///
/// As of #96, the f32 writer absorbs MLA Q_a/Q_b/KV_a/KV_b into standard
/// dense Q/K/V tensors at write time when full MLA geometry is known
/// (`qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` all present).
/// In that case MLA is accepted because the absorbed output is a standard
/// Q/K/V/O manifest. MLA architectures without complete geometry still
/// fail here — there's no defensible default split for `qk_head_dim`.
pub(super) fn ensure_standard_attention_supported(
    arch: &dyn larql_models::ModelArchitecture,
    surface: &'static str,
) -> Result<(), VindexError> {
    if arch.uses_mla() {
        // MLA absorption (#96) needs all three head-dim fields to recover
        // the pre-absorption split. When any is missing we cannot run
        // absorption safely, so the standard writer still has no way to
        // represent the attention block — reject up front.
        let has_geom = arch.mla_qk_nope_head_dim().is_some()
            && arch.mla_qk_rope_head_dim().is_some()
            && arch.mla_v_head_dim().is_some();
        if !has_geom {
            return Err(VindexError::UnsupportedArchitecture {
                family: arch.family().to_string(),
                feature: FEATURE_MLA.into(),
                surface: surface.into(),
            });
        }
    }

    Ok(())
}

/// Entry-point gate for the extract pipeline: reject unsupported attention
/// layouts before any partial vindex output is written.
///
/// Browse-level extracts only emit gate / embed / down_meta / tokenizer —
/// none of which depend on the attention layout — so this is a no-op there.
/// Any tier that writes attention (Attention / Inference / All) must reject
/// MLA-style architectures up front; failing inside the writer leaves a
/// half-populated vindex on disk that the caller would have to clean up.
pub(crate) fn ensure_extract_level_supported(
    arch: &dyn larql_models::ModelArchitecture,
    level: ExtractLevel,
) -> Result<(), VindexError> {
    if level.writes_attn() {
        ensure_standard_attention_supported(arch, SURFACE_EXTRACT_PIPELINE)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SURFACE: &str = "test";
    const TEST_Q4K_SURFACE: &str = SURFACE_Q4K_WEIGHT_WRITER;
    const MODEL_TYPE_LLAMA: &str = "llama";
    const MODEL_TYPE_DEEPSEEK_V2: &str = "deepseek_v2";
    const HIDDEN_SIZE_LLAMA_7B: usize = 4096;
    const HIDDEN_SIZE_TEST: usize = 4096;
    const INTERMEDIATE_SIZE_TEST: usize = 12288;
    const NUM_LAYERS_LLAMA_7B: usize = 32;
    const NUM_LAYERS_TEST: usize = 4;
    const NUM_ATTENTION_HEADS_LLAMA_7B: usize = 32;
    const NUM_ATTENTION_HEADS_TEST: usize = 32;
    const NUM_KV_HEADS_TEST: usize = 32;
    const HEAD_DIM_TEST: usize = 128;
    const KV_LORA_RANK_TEST: usize = 512;
    const Q_LORA_RANK_TEST: usize = 1536;
    const QK_NOPE_HEAD_DIM_TEST: usize = 128;
    const QK_ROPE_HEAD_DIM_TEST: usize = 64;
    const V_HEAD_DIM_TEST: usize = 128;

    #[test]
    fn standard_attention_accepts_llama() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_LLAMA,
            "hidden_size": HIDDEN_SIZE_LLAMA_7B,
            "num_hidden_layers": NUM_LAYERS_LLAMA_7B,
            "num_attention_heads": NUM_ATTENTION_HEADS_LLAMA_7B
        }));

        assert!(ensure_standard_attention_supported(&*arch, TEST_SURFACE).is_ok());
    }

    #[test]
    fn mla_architecture_is_rejected() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_DEEPSEEK_V2,
            "hidden_size": HIDDEN_SIZE_TEST,
            "intermediate_size": INTERMEDIATE_SIZE_TEST,
            "num_hidden_layers": NUM_LAYERS_TEST,
            "num_attention_heads": NUM_ATTENTION_HEADS_TEST,
            "num_key_value_heads": NUM_KV_HEADS_TEST,
            "head_dim": HEAD_DIM_TEST,
            "kv_lora_rank": KV_LORA_RANK_TEST,
            "q_lora_rank": Q_LORA_RANK_TEST
        }));

        let err = ensure_standard_attention_supported(&*arch, TEST_Q4K_SURFACE)
            .expect_err("MLA must not be accepted by standard Q/K/V/O writers");
        let msg = err.to_string();
        assert!(msg.contains(arch.family()), "{msg}");
        assert!(msg.contains(FEATURE_MLA), "{msg}");
        assert!(msg.contains(TEST_Q4K_SURFACE), "{msg}");
    }

    /// MLA arch without the qk_nope/qk_rope/v_head_dim fields — absorption
    /// cannot run, so the standard writer must still reject it.
    fn mla_arch() -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_DEEPSEEK_V2,
            "hidden_size": HIDDEN_SIZE_TEST,
            "intermediate_size": INTERMEDIATE_SIZE_TEST,
            "num_hidden_layers": NUM_LAYERS_TEST,
            "num_attention_heads": NUM_ATTENTION_HEADS_TEST,
            "num_key_value_heads": NUM_KV_HEADS_TEST,
            "head_dim": HEAD_DIM_TEST,
            "kv_lora_rank": KV_LORA_RANK_TEST,
            "q_lora_rank": Q_LORA_RANK_TEST
        }))
    }

    /// MLA arch with the full pre-absorption geometry exposed — `write_f32`
    /// can absorb into a standard Q/K/V/O manifest, so the gate must
    /// accept it.
    fn mla_arch_with_geometry() -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_DEEPSEEK_V2,
            "hidden_size": HIDDEN_SIZE_TEST,
            "intermediate_size": INTERMEDIATE_SIZE_TEST,
            "num_hidden_layers": NUM_LAYERS_TEST,
            "num_attention_heads": NUM_ATTENTION_HEADS_TEST,
            "num_key_value_heads": NUM_KV_HEADS_TEST,
            "head_dim": HEAD_DIM_TEST,
            "kv_lora_rank": KV_LORA_RANK_TEST,
            "q_lora_rank": Q_LORA_RANK_TEST,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM_TEST,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM_TEST,
            "v_head_dim": V_HEAD_DIM_TEST
        }))
    }

    fn llama_arch() -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": MODEL_TYPE_LLAMA,
            "hidden_size": HIDDEN_SIZE_LLAMA_7B,
            "num_hidden_layers": NUM_LAYERS_LLAMA_7B,
            "num_attention_heads": NUM_ATTENTION_HEADS_LLAMA_7B
        }))
    }

    #[test]
    fn extract_level_browse_passes_for_mla() {
        // Browse only emits gate / embed / down_meta / tokenizer — none
        // of which need the attention layout. MLA must succeed here.
        assert!(
            ensure_extract_level_supported(&*mla_arch(), ExtractLevel::Browse).is_ok(),
            "Browse-level extract should accept MLA architectures"
        );
    }

    #[test]
    fn extract_level_attention_rejects_mla() {
        let err = ensure_extract_level_supported(&*mla_arch(), ExtractLevel::Attention)
            .expect_err("Attention-level extract must reject MLA");
        let msg = err.to_string();
        assert!(msg.contains(FEATURE_MLA), "{msg}");
        assert!(msg.contains(SURFACE_EXTRACT_PIPELINE), "{msg}");
    }

    #[test]
    fn extract_level_inference_rejects_mla() {
        assert!(
            ensure_extract_level_supported(&*mla_arch(), ExtractLevel::Inference).is_err(),
            "Inference-level extract must reject MLA"
        );
    }

    #[test]
    fn extract_level_all_rejects_mla() {
        assert!(
            ensure_extract_level_supported(&*mla_arch(), ExtractLevel::All).is_err(),
            "All-level extract must reject MLA"
        );
    }

    #[test]
    fn extract_level_all_passes_for_llama() {
        assert!(
            ensure_extract_level_supported(&*llama_arch(), ExtractLevel::All).is_ok(),
            "Llama models with standard Q/K/V/O attention must pass at every level"
        );
    }

    #[test]
    fn mla_with_full_geometry_is_accepted_so_absorption_can_run() {
        // PR #96 wires MLA absorption into the f32 writer. The gate must
        // not block MLA archs whose qk_nope/qk_rope/v_head dims are exposed;
        // the writer will fuse Q_a/Q_b/KV_a/KV_b into standard Q/K/V/O.
        assert!(
            ensure_standard_attention_supported(&*mla_arch_with_geometry(), TEST_SURFACE).is_ok(),
            "MLA with full geometry must be accepted (post-#96 absorption path)"
        );
    }

    #[test]
    fn extract_level_inference_accepts_mla_with_full_geometry() {
        // End-to-end: DS-V2/V3/Kimi K2 GGUFs that expose qk_nope/qk_rope/v_head
        // (PR #135 wired this through from `attention.key_length[_mla]` etc.)
        // must extract to inference level via the absorption path.
        assert!(
            ensure_extract_level_supported(&*mla_arch_with_geometry(), ExtractLevel::Inference)
                .is_ok(),
            "Inference extract should accept MLA when geometry is complete"
        );
        assert!(
            ensure_extract_level_supported(&*mla_arch_with_geometry(), ExtractLevel::All).is_ok(),
            "All-level extract should also accept MLA when geometry is complete"
        );
    }
}
