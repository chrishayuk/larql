# Roadmap — larql-models

## Current: 12 architectures, 337 tests (227 src + 82 + 28 integration), safetensors + GGUF loading, config-driven `rope_scaling` / `norm_eps` / GPT-2 legacy aliases

## Config-loading correctness pass 2026-05-16

Cross-engine Shannon verify (`larql shannon verify`) on Linux + macOS
revealed four config-loading defects in `larql-models` that drove the
LARQL Rust forward path off by 5.4 % (Gemma 3 4B) to 8.2 % (Mistral 7B)
bits/char relative to HF transformers. All four are now fixed in the
loader itself — env-var diagnostics stay in tree but production runs
need zero overrides:

| # | Bug | Models affected | Fix site |
|--:|---|---|---|
| 1 | `rms_norm_eps` from config.json was never read; trait default 1e-6 used everywhere | Mistral 7B, Llama 3.2, Gemma 3 4B | `parser.rs` parses `rms_norm_eps` / `layer_norm_eps` / `layer_norm_epsilon` / `norm_epsilon` (StarCoder2) into `ModelConfig.norm_eps`; default `norm_eps()` reads it |
| 2 | Per-layer-type `rope_scaling` (Gemma 3 structured `{full_attention, sliding_attention}` form) was not honoured | Gemma 3 4B | `RopeScaling.gemma3_global_only`; `Gemma3Arch::rope_position_divisor_for_layer` returns `factor` on full-attention layers only |
| 3 | `rope_scaling = llama3` (wavelength-dependent per-channel factors) was not implemented | Llama 3.2 1B | New `Llama3RopeScaling` type in `config.rs`; `LlamaArch::llama3_rope_scaling` returns parsed params; `attention/rope.rs::Llama3Scaling::apply` mirrors HF's `_compute_llama3_parameters` |
| 4 | `norm_epsilon` alias not recognised (StarCoder2's name for `rms_norm_eps`) | StarCoder2 3B | Added to the alias list in `parser.rs` |

Plus GPT-2 config aliases (`n_embd` / `n_layer` / `n_head` / `n_inner`)
parsed via the new alias-list machinery in
`detect/config_io.rs::CONFIG_KEY_*_ALIASES`. Loader path now resolves
`openai-community/gpt2`; raw-safetensors tensor-key renaming
(`wte`/`wpe`/`c_attn` → canonical) is a separate scope kept for the
GPT-2 safetensors loader item below.

Shared numerical defaults moved to a new `defaults` module
(`DEFAULT_NORM_EPS`, `ROPE_BASE_GEMMA`, `ROPE_BASE_DEFAULT`) so the
parser fallback, the trait default, and the per-arch fallback all
reference the same value — the drift between these three sites was
the mechanism of bug 1.

Verification:
`scripts/diagnose_models.py` (multi-arch sweep across SmolLM2-135M,
Llama-3.2-1B, Qwen3-0.6B, Gemma-2-2B, StarCoder2-3B, Mistral-7B-v0.1,
Gemma-3-4B-it) reports 7/9 PASS at <0.5 % threshold with **zero env
vars set**. The two ERR rows are pre-existing issues unrelated to this
work (Granite-4.0-micro MoE validator strictness; Gemma-4 not yet
supported by HF transformers in `.venv`).

CI gate at `.github/workflows/shannon-verify.yml` runs
`larql shannon verify HuggingFaceTB/SmolLM2-135M --engines hf` on
every PR + push to main.

Diagnostic doc:
[`docs/diagnoses/shannon-cross-engine-divergence.md`](../../docs/diagnoses/shannon-cross-engine-divergence.md).

## Roadmap Review 2026-04-26

The 2026-04-26 quality pass closed the known P0 items for `larql-models`: walk-only filtering, silent dtype reporting, quant test gaps, loader string constants, MXFP4 consolidation, config validation adoption, clippy, examples, benchmark coverage, and coverage refresh are complete. The 2026-04-30 follow-up fixed packed BF16 expert ownership, GGUF matrix layout/config-default handling, and refreshed coverage to the current baseline.

The 2026-05-07 follow-up fixed small-vocab GGUF handling, explicit embedding
orientation, missing GGUF attention metadata defaults, and checked mmap packed
byte ranges. It also added targeted regression coverage and refreshed CI to
run rustfmt plus a crate-scoped coverage summary.

Recommended next sequence:
- **GPT-2 raw-safetensors tensor-key renaming.** Config parses cleanly
  now; tensor loading needs the `wte` / `wpe` / `h.N.attn.c_attn` /
  `h.N.mlp.c_fc` → canonical mapping in `loading/safetensors.rs` (the
  existing `gpt2.rs` arch assumes GGUF→HF normalisation has already run).
- **Granite-4 MoE validator relaxation** so `granite-4.0-micro` loads —
  the dense Granite-4 model carries hybrid MoE *flags* without expert
  tensors, which the current validator rejects.
- Add Phi-3 / Phi-4 architecture support. Low effort, exercises the
  validation path, expands coverage without changing the trait.
- Use validated loading/detection APIs at downstream inference/extraction boundaries.
- Defer large loading changes until after architecture coverage. ADR-008 defines the additive lazy/quantized weight API shape.

## P0: Code Quality

### Downstream validation rollout
**Effort**: Medium
**Status**: Not started

`larql-models` now exposes validated APIs. Update downstream inference, vindex extraction, CLI, and server entry points to use `detect_*_validated` or `load_*_validated` where invalid configs should fail fast.

### Deterministic HuggingFace cache resolution
**Effort**: Low
**Status**: Not started

`loading/safetensors.rs::resolve_model_path` scans cached snapshot
directories and returns the first snapshot with safetensors. `read_dir` order
is not stable and the resolver ignores `refs/main`, so the same model ID can
resolve to an old or arbitrary cached revision. Prefer the commit recorded in
`models--.../refs/main` when no explicit revision is provided, then fall back
to a deterministic snapshot ordering.

### Architecture capability contracts
**Effort**: Medium  
**Status**: Not started

Detection currently says which family a config belongs to, but it does not
state which downstream surfaces are actually implemented for that family.
Add an explicit capability contract so extraction, vindex weight writing,
inference, trace, and prompt rendering can fail loudly instead of accepting an
architecture whose tensors are not consumed by the active path.

Immediate driver: DeepSeek is correctly detected as MoE + MLA and exposes
`mla_*` tensor keys, but vindex writers and inference paths currently consume
standard Q/K/V/O attention tensors only. Either implement the MLA extraction
and forward contract, or report it as unsupported at the boundary.

### Note on quant/dequant crate split
**Decision**: `larql-models/quant/` is **format deserialization** (GGUF/safetensors → f32). `larql-compute` has **compute operations** (quantized matvec, Metal shaders). The split is correct. The `f16_to_f32` copies in `larql-compute/cpu/ops/q4k_matvec.rs` and `q6k_matvec.rs` are intentional — CPU reference impls for Metal shader testing, isolated by design. `larql-compute` is dev-only dep; don't flip that direction.

## P1: Architecture Coverage

### Phi-3 / Phi-4
**Effort**: Low  
**Status**: Not started

Similar to Llama with some attention differences (partial RoPE, SuRoPE). Most trait defaults apply.

### Command R / Cohere
**Effort**: Medium  
**Status**: Not started

Different attention key pattern, different norm placement.

### Mamba / state-space models
**Effort**: Large  
**Status**: Research

Would require extending the trait beyond transformer assumptions (no attention keys, no KV cache). May warrant a separate trait hierarchy.

## P2: Loading Improvements

### Streaming safetensors loading
**Effort**: Medium  
**Status**: Not started

Current loader mmaps shards but eagerly converts retained dense tensors into f32 `ModelWeights`; packed BF16 expert tensors are already retained as mmap byte ranges. For 70B+ models, per-layer/lazy loading would reduce peak memory further. Already have mmap infrastructure — extend to lazy loading with `Arc<Mmap>` references and explicit tensor lifetimes.

Design direction: ADR-008 proposes additive `LazyModelWeights` / `load_model_dir_lazy(_validated)` APIs rather than overloading eager `ModelWeights`.

### GGUF quantized inference (skip dequant)
**Effort**: Large  
**Status**: Not started

Currently GGUF tensors are dequantized to f32 during loading. For Q4_K/Q6_K formats, keep data in quantized form and pass directly to `larql-compute` quantized kernels. Requires a `QuantizedWeights` variant alongside `ModelWeights`.

Design direction: ADR-008 proposes additive `QuantizedModelWeights` / `load_gguf_quantized(_validated)` APIs that preserve GGML type ids and byte ranges.

### MLX npz/safetensors hybrid
**Effort**: Low  
**Status**: Partial (MLX safetensors work, npz not yet)

Apple MLX models sometimes use `.npz` format. Add npz parsing alongside safetensors.

## P3: Trait Evolution

### Per-layer FFN type
**Effort**: Low  
**Status**: Not started

Some models (e.g., future MoE variants) may have different FFN types per layer (dense for early layers, MoE for later). Add `ffn_type_for_layer(layer)` method.

### Attention pattern abstraction
**Effort**: Medium  
**Status**: Research

Current sliding window is boolean per layer. Future models may have more complex patterns (local + global hybrid, dilated attention, prefix caching hints). Consider a richer `AttentionPattern` enum.

## Completed

| Item | Date | Impact |
|------|------|--------|
| ModelArchitecture trait | 2026-03 | Foundation — 83 methods with defaults |
| Gemma 2/3 support | 2026-03 | QK-norm, softcapping, sliding window |
| Llama/Mistral/Qwen/DeepSeek | 2026-03 | Core architecture coverage |
| Mixtral MoE (PerExpert) | 2026-03 | Expert key patterns |
| GPT-OSS (PackedMxfp4) | 2026-03 | MXFP4 dequantization, packed expert keys |
| Granite (scaling multipliers) | 2026-03 | Embedding/residual/attention/logits scaling |
| StarCoder2 | 2026-03 | LayerNorm, bias, GELU |
| GGUF loading | 2026-03 | Q4_0/Q4_1/Q8_0/F16/BF16 dequantization |
| Safetensors mmap + HF cache | 2026-03 | Zero-copy loading, cache resolution |
| drop_ffn_weights | 2026-04 | Walk-only mode saves ~13GB |
| Gemma 4 architecture | 2026-04 | Per-layer geometry, PLE, KV sharing, V-norm, layer scalars |
| Gemma 4 31B + E2B configs | 2026-04 | Both variants tested with real config.json |
| Gemma4Arch re-export | 2026-04-07 | Public API complete |
| v_shares_k from config | 2026-04-07 | Uses attention_k_eq_v flag instead of hardcoded false |
| Gemma 3 qk_norm_weight_offset | 2026-04-07 | Was missing (Gemma 2 had it, Gemma 3 didn't) |
| Architecture coverage milestone | 2026-04-07 | All 12 architectures tested: Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, DeepSeek, GPT-OSS, Granite, StarCoder2, Generic |
| GGML quant test gaps closed (51 tests) | 2026-04-26 | q4k_row_dot NEON≡scalar, q4k/q6k scaled_add correctness, Q4_K known nonzero values |
| Silent dtype skip fixed | 2026-04-26 | `skipped_tensors` field on ModelWeights; UnsupportedDtype collected, other errors bubbled |
| normalize_key_pub removed | 2026-04-26 | Dead wrapper gone; `normalize_key` is `pub(crate)` |
| Config alias constants | 2026-04-26 | `NUM_EXPERTS_KEYS`, `NUM_EXPERTS_PER_TOK_KEYS`, `field_u64` helper in `detect.rs` |
| MXFP4 consolidation | 2026-04-26 | `split_gate_up_experts` in `quant/mxfp4.rs`; loader thinned + renamed |
| Walk-only loader fixes | 2026-04-26 | GGUF filtering, GPT-OSS MXFP4 predicate-aware expansion, StarCoder2 c_fc/c_proj classification |
| Loader magic-string cleanup | 2026-04-26 | Centralized GGUF metadata/key rewrites, MXFP4 suffixes, HF cache path fragments, packed expert keys |
| Config validation | 2026-04-26 | `ModelArchitecture::validate()` with centralized diagnostic fields; catches dimensions, head geometry, RoPE values, per-layer metadata, KV sharing, and MoE inconsistencies |
| Validation adoption in larql-models APIs | 2026-04-26 | Added `detect_*_validated`, `load_model_dir*_validated`, and `load_gguf_validated` while preserving permissive inspection APIs |
| Detection hardening for invalid configs | 2026-04-26 | Malformed zero-head configs and short Gemma 4 `layer_types` no longer panic before validation |
| Lazy/quantized weight API design | 2026-04-26 | ADR-008 defines additive `LazyModelWeights` and `QuantizedModelWeights` direction for larger loading work |
| Coverage baseline refresh | 2026-04-26 | 274 tests; 88.02% line / 86.29% function coverage |
| Clippy clean (zero warnings) | 2026-04-26 | lib + examples + tests all pass `-D warnings` |
| Criterion benchmark suite | 2026-04-26 | `cargo bench -p larql-models --bench models` covers detection, validation, key mapping, FFN classification, synthetic loading, and GGML dequant |
| Documentation refresh | 2026-04-26 | README, roadmap, performance notes, loading/quant docs, and ADRs updated for validation and current metrics |
| Example suite (3 demos) | 2026-04-07 | architecture_demo (all 12), demo_tensor_keys (all 12), demo_loading |
| Packed BF16 mmap retention | 2026-04-30 | Gemma 4 A4B packed BF16 expert tensors are retained as mmap byte ranges instead of heap-cloned raw bytes |
| GGUF loader correctness fixes | 2026-04-30 | 2D tensors load as standard `[rows, cols]`; absent optional RoPE/vocab metadata falls back through architecture/tokenizer defaults |
| Coverage baseline refresh | 2026-04-30 | 282 tests; 81.41% line / 82.06% function coverage |
| GGUF loader regression fixes | 2026-05-07 | Small vocab metadata, shape-derived vocab fallback, missing KV/head-dim defaults, checked packed mmap ranges |
| Coverage baseline refresh | 2026-05-07 | 286 tests; 77.86% line / 78.30% function coverage |
