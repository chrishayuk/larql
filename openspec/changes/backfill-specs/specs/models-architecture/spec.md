## ADDED Requirements

### Requirement: ModelArchitecture trait surface

The `larql_models::ModelArchitecture` trait SHALL be the single abstraction
through which all downstream crates (`larql-vindex`, `larql-inference`,
`larql-compute`, `larql-cli`) interact with model weights. The trait MUST
NOT depend on any compute or BLAS library, MUST provide default
implementations for every method so that adding a new architecture only
requires overriding what differs, and MUST take a `layer: usize`
parameter on every method whose behavior can vary by layer.

#### Scenario: Generic architecture works with zero overrides
- **WHEN** a `GenericArch` is constructed from a minimal `ModelConfig` with no overrides
- **THEN** every trait method SHALL return a default that is consistent with HuggingFace conventions (RMSNorm, SiLU activation, Gated FFN, no MoE, no MLA, scale = 1/sqrt(head_dim))
<!-- test: larql_models::test_architectures::generic_architecture_exercises_default_trait_contract -->
<!-- test: larql_models::test_architectures::generic_fallback -->

#### Scenario: Per-layer methods accept and respect a layer index
- **WHEN** a Gemma 4 architecture is queried for `head_dim_for_layer(layer)`, `rotary_fraction_for_layer(layer)`, `rope_base_for_layer(layer)`, or `is_sliding_window_layer(layer)` on a sliding vs global layer
- **THEN** the values SHALL differ between sliding and global layers per the published Gemma 4 spec
<!-- test: larql_models::test_architectures::gemma4_per_layer_head_dim -->
<!-- test: larql_models::test_architectures::gemma4_partial_rotary -->
<!-- test: larql_models::test_architectures::gemma4_rope_bases -->

#### Scenario: All registered architectures expose attention key patterns
- **WHEN** the test suite iterates every registered architecture and queries `attn_q_key`, `attn_k_key`, `attn_v_key`, `attn_o_key` for layer 0
- **THEN** each architecture SHALL return non-empty key strings that match the safetensors layout the architecture was loaded from
<!-- test: larql_models::test_architectures::all_architectures_have_attn_keys -->

### Requirement: Architecture detection from config JSON

`detect_from_json` SHALL construct an architecture from a HuggingFace
`config.json` even when fields are missing, ambiguous, or
non-canonical, returning the constructed architecture so callers can
inspect what was parsed. Strict construction SHALL be available via
`detect_from_json_validated`, which MUST return
`Err(Vec<ConfigValidationError>)` rather than panic when invariants
fail.

#### Scenario: Gemma 3 config detected and family-tagged
- **WHEN** a Gemma 3 `config.json` is passed to `detect_from_json`
- **THEN** the resulting architecture SHALL report `family() == "gemma3"`, return Gemma 3 key patterns, and pass `gemma_family_traits` introspection
<!-- test: larql_models::detect::test_detect_gemma3 -->
<!-- test: larql_models::test_architectures::gemma3_detection -->
<!-- test: larql_models::test_architectures::gemma3_gemma_family_traits -->

#### Scenario: Gemma 4 dual rope bases are surfaced
- **WHEN** a Gemma 4 (E2B) config is parsed
- **THEN** `rope_base_for_layer(layer)` SHALL return 10_000 for sliding layers and 1_000_000 for global layers
<!-- test: larql_models::test_architectures::gemma3_dual_rope_bases -->
<!-- test: larql_models::test_architectures::gemma4_rope_bases -->

#### Scenario: Other supported families detected
- **WHEN** a config for Llama, Mistral, Qwen2, or DeepSeek is parsed
- **THEN** detection SHALL return the family-specific architecture and key patterns
<!-- test: larql_models::detect::test_detect_llama -->
<!-- test: larql_models::detect::test_detect_mistral -->
<!-- test: larql_models::detect::test_detect_qwen2 -->
<!-- test: larql_models::test_architectures::deepseek_detection -->
<!-- test: larql_models::test_architectures::deepseek_moe -->

#### Scenario: Tiny / non-standard configs do not panic
- **WHEN** a "tiny" model config without all standard fields is parsed
- **THEN** detection SHALL still succeed and the resulting architecture SHALL provide every advertised key
<!-- test: larql_models::detect::test_detect_tinymodel -->
<!-- test: larql_models::detect::test_tinymodel_full_key_coverage -->

#### Scenario: Validated detection rejects invalid configs
- **WHEN** `detect_from_json_validated` is called on a config whose `head_dim` does not divide `hidden_size`, or whose KV heads exceed Q heads
- **THEN** detection SHALL return `Err(Vec<ConfigValidationError>)` listing the violated invariant(s)
<!-- test: larql_models::test_loading::detect_from_json_validated_returns_validation_error -->

### Requirement: Config validation invariants

`ModelConfig::validate()` SHALL return `Result<(), Vec<ConfigValidationError>>`
covering: positive core dimensions, divisibility (`hidden_size % head_dim == 0`,
`num_q_heads % num_kv_heads == 0`), KV heads not exceeding Q heads, finite
RoPE bases / scaling factors / partial-rotary fractions, explicit
`layer_types.len() == num_layers`, KV-sharing leaving at least one source,
MoE configs declaring expert count and top-k where top-k ≤ N, and hybrid
MoE configs supplying `moe_intermediate_size`.

#### Scenario: Generic-arch attention scale falls back to 1/sqrt(head_dim)
- **WHEN** `attention_scale_for_layer(layer)` is read on a Gemma 2 model with no QK-norm override
- **THEN** the scale SHALL be `1.0 / sqrt(head_dim)`
<!-- test: larql_models::test_architectures::gemma2_attention_scale -->

#### Scenario: Gemma 4 attention scale is one (QK-norm path)
- **WHEN** `attention_scale_for_layer(layer)` is read on a Gemma 4 model
- **THEN** the scale SHALL be `1.0` (the QK-norm absorbs the scale factor)
<!-- test: larql_models::test_architectures::gemma4_attention_scale_is_one -->

#### Scenario: Per-layer head_dim varies between sliding and global
- **WHEN** `head_dim_for_layer(layer)` is read on a Gemma 4 model
- **THEN** sliding layers SHALL return 256 and global layers SHALL return 512
<!-- test: larql_models::test_architectures::gemma4_per_layer_head_dim -->

### Requirement: Normalization and weight-offset semantics

The trait SHALL distinguish RMSNorm vs LayerNorm via `norm_type()` and
SHALL expose `norm_weight_offset()` (0.0 for Llama / Gemma 4; 1.0 for
Gemma 2/3, where the on-disk weight encodes `weight - 1`) and
`qk_norm_weight_offset()` for QK norms (1.0 for Gemma 2/3, 0.0 for
Gemma 4). Architectures with post-attention/post-FFN norms SHALL set
`has_post_norms() == true`.

#### Scenario: Gemma 2/3 norm offsets are 1.0
- **WHEN** `norm_weight_offset()` is read on a Gemma 2 or Gemma 3 architecture
- **THEN** it SHALL return `1.0`
<!-- test: larql_models::test_architectures::gemma2_norm_offsets -->
<!-- test: larql_models::test_architectures::gemma3_norm_offsets -->

#### Scenario: Gemma 4 norm offset is 0.0
- **WHEN** `norm_weight_offset()` is read on a Gemma 4 architecture
- **THEN** it SHALL return `0.0`
<!-- test: larql_models::test_architectures::gemma4_norm_offset_zero -->

#### Scenario: QK norm keys are present where expected
- **WHEN** `attn_q_norm_key(layer)` and `attn_k_norm_key(layer)` are queried on Gemma 2 / Gemma 4
- **THEN** they SHALL return `Some` keys for those families and `None` for families without QK norm
<!-- test: larql_models::test_architectures::gemma2_qk_norm_keys -->
<!-- test: larql_models::test_architectures::gemma4_v_norm -->

### Requirement: Sliding-window and softcapping semantics

Architectures with sliding-window attention SHALL signal it via
`is_sliding_window_layer(layer)`. Architectures with attention or logit
softcapping SHALL expose those caps via the corresponding trait
methods so inference can apply them without re-detecting the family.

#### Scenario: Gemma 3 sliding-window pattern alternates as documented
- **WHEN** `is_sliding_window_layer(layer)` is read across all 0..num_layers
- **THEN** the resulting boolean vector SHALL match the published Gemma 3 attention pattern
<!-- test: larql_models::test_architectures::gemma3_sliding_window_pattern -->

#### Scenario: Gemma 2 softcapping values are surfaced
- **WHEN** softcap values are read on a Gemma 2 architecture
- **THEN** the trait SHALL return finite, positive values for both attention and final-logit softcaps
<!-- test: larql_models::test_architectures::gemma2_softcapping -->

### Requirement: Mixture-of-Experts (MoE) tensor-key model

For MoE architectures the trait SHALL expose `is_moe()`,
`num_experts()`, `num_experts_per_token()`, `num_shared_experts()`,
`expert_format()` (`PerExpert` for Mixtral; `PackedMxfp4` for GPT-OSS),
plus per-expert key methods (`expert_ffn_gate_key`, `_up_key`,
`_down_key`) and packed-MXFP4 key methods (`packed_gate_up_blocks_key`,
`packed_gate_up_scales_key`, `packed_down_blocks_key`,
`packed_down_scales_key`). Shared experts (DeepSeek) SHALL be exposed
via `shared_expert_*_key`.

#### Scenario: DeepSeek MoE keys are present for routed and shared experts
- **WHEN** `expert_ffn_gate_key(layer, expert_id)` and `shared_expert_gate_key(layer)` are queried on a DeepSeek architecture
- **THEN** both return non-empty keys following the DeepSeek tensor naming convention
<!-- test: larql_models::test_architectures::deepseek_moe -->
<!-- test: larql_models::test_architectures::deepseek_expert_keys -->
<!-- test: larql_models::test_architectures::deepseek_shared_expert_keys -->

### Requirement: Multi-head Latent Attention (MLA) tensor-key model

DeepSeek architectures SHALL expose `uses_mla() == true`,
`kv_lora_rank()`, `q_lora_rank()`, and the four MLA tensor keys
(`mla_kv_a_key`, `mla_kv_b_key`, `mla_q_a_key`, `mla_q_b_key`).
Architectures without MLA SHALL keep `uses_mla() == false` and SHALL
NOT advertise MLA keys.

#### Scenario: DeepSeek MLA keys and ranks are exposed
- **WHEN** MLA properties are queried on a DeepSeek architecture
- **THEN** `uses_mla()` returns true, the LoRA ranks are positive integers, and all four MLA keys resolve to non-empty strings
<!-- test: larql_models::test_architectures::deepseek_mla -->

### Requirement: RoPE scaling and base values per family

Architectures SHALL expose `rope_base_for_layer(layer)` returning the
unscaled RoPE base, plus a scaling description retrievable from the
config (factor, low/high frequency bands, scaling type) where
applicable. RoPE base values MUST match the published model spec for
each family.

#### Scenario: DeepSeek RoPE scaling is parsed
- **WHEN** a DeepSeek config that declares RoPE scaling parameters is loaded
- **THEN** the architecture SHALL surface scaling type, factor, and band frequencies via the relevant trait methods
<!-- test: larql_models::test_architectures::deepseek_rope_scaling -->

### Requirement: Per-Layer Embeddings (PLE) for Gemma 4 E2B

Gemma 4 E2B SHALL surface `has_ple() == true`, expose the PLE tensor
key, and permit per-layer embedding lookup keyed by layer index.

#### Scenario: Gemma 4 E2B exposes PLE
- **WHEN** PLE properties are queried on a Gemma 4 E2B architecture
- **THEN** `has_ple()` returns true and the PLE tensor key resolves to a non-empty string
<!-- test: larql_models::test_architectures::gemma4_ple -->

### Requirement: Cross-layer KV sharing for Gemma 4

Gemma 4 SHALL support KV sharing: `v_shares_k(layer)` reports whether
V is computed from K's projection (not its own), and
`kv_shared_source_layer(layer)` returns the donor layer index when this
layer reuses KV from another layer. Validation MUST guarantee at least
one non-shared source layer.

#### Scenario: Gemma 4 KV sharing surface is correct
- **WHEN** KV-sharing methods are queried across all Gemma 4 layers
- **THEN** the source-layer mapping SHALL be acyclic and reach a non-shared root
<!-- test: larql_models::test_architectures::gemma4_kv_sharing -->

### Requirement: Tensor-key prefix stripping

`key_prefixes_to_strip()` SHALL list architecture-specific prefixes
(e.g. `"model."`) that the loader strips before resolving tensor names.
Stripping SHALL be applied left-to-right to the longest matching prefix.

#### Scenario: Gemma 4 prefix stripping is honored
- **WHEN** a Gemma 4 architecture's key prefixes are queried
- **THEN** the prefix list SHALL contain `"model."` and stripping a key produces the architecture-internal form
<!-- test: larql_models::test_architectures::gemma4_prefix_strip -->

#### Scenario: Safetensors prefix normalisation matches first prefix only
- **WHEN** `normalize_key` is invoked on a key matching multiple prefixes
- **THEN** it SHALL strip only the longest matching prefix once
<!-- test: larql_models::loading::safetensors::normalize_key_strips_first_matching_prefix -->
<!-- test: larql_models::loading::safetensors::normalize_key_falls_through_to_shorter_prefix -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_models::test_architectures::**::* -->
<!-- test: larql_models::detect::**::* -->
<!-- test: larql_models::detect::tests::**::* -->
