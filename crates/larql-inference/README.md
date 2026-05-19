# larql-inference

Inference engine for transformer models. Forward pass, BLAS-fused attention, hardware-accelerated matmul backends, and pluggable FFN routing.

## Overview

This crate runs transformer forward passes with Apple Accelerate (AMX) and optional Metal GPU acceleration. It uses `larql-vindex` for gate KNN (sparse feature selection) and `larql-models` for weight loading and architecture definitions.

```rust
use larql_inference::InferenceModel;

// Load a model
let model = InferenceModel::load("google/gemma-3-4b-it")?;

// Run inference
let result = larql_inference::predict(
    model.weights(), model.tokenizer(), &token_ids, 5,
);
println!("Top prediction: {} ({:.1}%)", result.predictions[0].0, result.predictions[0].1 * 100.0);
```

## Generation stack

```rust
use larql_inference::{
    open_inference_vindex, generate_streaming, ChatSession,
    SamplingConfig, EosConfig, Detokenizer,
};

let index = open_inference_vindex(&vindex_path)?;            // strict loader
let result = generate_streaming(
    weights, &tokenizer, &token_ids, max_tokens,
    &index, &*backend, &cache, 13..num_layers,
    SamplingConfig::temperature(0.8).with_top_p(0.9).with_seed(42),
    &EosConfig::from_vindex_dir(&vindex_path),
    |_id, text, _prob| { print!("{text}"); std::io::stdout().flush().ok(); },
);
```

| Type | Role |
|------|------|
| [`SamplingConfig`] / [`Sampler`] | Greedy / temperature / top-k / top-p / seeded. Sparse hot path is <2µs/call (<0.02% of decode budget) — see [PERFORMANCE.md](PERFORMANCE.md#sampling-overhead). |
| [`EosConfig`] | Stop-token detection. Reads `generation_config.json::eos_token_id` + `stop_strings`, layered on a built-in list (Gemma `<end_of_turn>`, ChatML `<|im_end|>`, Llama-3 `<|eot_id|>`). Falls back to `skip_special=false` decode when the streaming detok strips a special EOS marker. |
| [`Detokenizer`] | Cumulative-decode delta for streaming output. Preserves HF `▁` leading-space across SP and BPE tokenizers. Equivalent to llama.cpp `llama_token_to_piece`. |
| [`ChatSession`] | Multi-turn token buffer with whole-turn eviction at `max_context`. Pluggable [`TurnRenderer`] (Gemma / ChatML / Llama-3 built in). |
| [`generate`] / [`generate_with_sampling`] / [`generate_streaming`] | Three public entry points — greedy → sampled → streamed. Each thinly wraps the next so adding sampling or a callback is opt-in without breaking existing callers. |
| [`open_inference_vindex`] | Strict vindex loader. Propagates stride / manifest errors loudly (rebuild guidance) instead of silently degrading to a slower path. Use this in any tool that loads a vindex for inference. |

[`SamplingConfig`]: layer_graph::SamplingConfig
[`Sampler`]: layer_graph::Sampler
[`EosConfig`]: layer_graph::EosConfig
[`Detokenizer`]: layer_graph::Detokenizer
[`ChatSession`]: layer_graph::ChatSession
[`TurnRenderer`]: layer_graph::TurnRenderer
[`generate`]: layer_graph::generate
[`generate_with_sampling`]: layer_graph::generate_with_sampling
[`generate_streaming`]: layer_graph::generate_streaming
[`open_inference_vindex`]: vindex::open_inference_vindex

## Engine diagnostic

`larql diag <vindex>` (CLI) reports which kernel paths the loader will pick, validates Q4_K/Q6_K manifest strides, and (with `--probe`) runs a real forward to print the per-stage timing breakdown. Catches the silent-slowdown classes (stale 148-byte Q4_K stride → all-NaN; `vocab_size=0` → 4× slower lm_head fallback) at a glance:

```
$ larql diag output/gemma3-4b-v2.vindex
Stride validation:
  ✓ 238 entries match canonical stride
LM-head path resolution (which kernel fires per next-token):
  → Q4 matvec (Metal fast)   lm_head_q4 mmap = true, vocab_size > 0 = true  → ~1.9 ms
     f16 gemv (tied embed)    ...
     f32 KNN (lm_head.bin)    ...
     f32 BLAS gemv (slow)     ...
```

## Key Components

| Module | Purpose |
|--------|---------|
| `attention/` | BLAS-fused GQA attention: block, GQA, GPU dispatch, RoPE |
| `forward/` | Forward pass: embed, layer, predict, PLE (per-layer embeddings), trace |
| `ffn/` | FFN evaluation: `WeightFfn`, `SparseFfn`, plus `remote/` (HTTP single-shard) and `moe_remote/` (gRPC expert grid) |
| `layer_graph/` | Layer graphs + generation: `pipeline_layer`, `predict`, `prefill`, plus `generate/` (eos, detok, sampling, chat_session, gpu/cpu loops, lm_head, types) and `grid/` (remote MoE/FFN orchestration) |
| `residual.rs` | RMS norm, layer norm |
| `trace/` | Residual stream decomposition and tiered storage |
| `vindex/` | `open_inference_vindex` (strict loader) + `WalkFfn` (mmap'd FFN) + `q4k_forward/` |
| `kv_engine.rs` | `KvEngine` trait + `EngineInfo` + `DecodeStageSummary` — abstract dispatch surface shared with `larql-kv` (engine impls live there) |
| `kv_dispatch/` (`mod.rs`, `cpu.rs`, `metal.rs`, `helpers.rs`; future: `vulkan.rs`, `cuda.rs`) | `KvDispatch` per-layer-intent trait (sync) + `EngineBackend: ComputeBackend + KvDispatch` umbrella; `CpuBackend` and `MetalBackend` impls; `helpers::kv_prefill_via_dispatch` / `kv_decode_step_via_dispatch` (sync) + `_async` variants drive the per-layer prefill/decode loop. Spec: [`compute-backend-redesign.md`](docs/specs/compute-backend-redesign.md). |
| `async_compute_backend/` (`mod.rs`, `cpu.rs`, `metal.rs`; future: `vulkan.rs`, `cuda.rs`) | `AsyncComputeBackend: ComputeBackend + KvDispatch + Send` sibling trait — deferred-dispatch intent surface with `AttentionHandle` / `ResidualUploadHandle` for one-command-buffer-per-decode-step batching on GPU backends. CPU is a degenerate `Ready*` wrapper (parity reference). Backends live as submodules so adding Vulkan/CUDA is a single new file. Spec: [`async-compute-backend.md`](docs/specs/async-compute-backend.md). |
| `experts/` | WASM expert dispatcher and registry |
| `chat/` | Jinja-driven chat templates loaded from vindex |
| `capture.rs` | Residual-stream vector capture for probing |
| `model.rs` | Model loading (re-exports from larql-models) |

> Weight-level graph walkers live in the `larql-vindex` crate (`walker/`), not here.

## Compute Backend

All GPU pipeline operations use `larql_compute::ComputeBackend`:

```rust
use larql_compute::{default_backend, ComputeBackend};

let backend = default_backend();  // Auto-selects CPU or Metal, calibrates
println!("Using: {} ({})", backend.name(), backend.device_info());
```

The inference crate builds `FullPipelineLayer` structs (per-layer architecture params + quantized weights) and passes them to `backend.decode_token()` or `backend.prefill_q4()`. All model-specific behavior (norm type, activation, head_dim, RoPE base) is parameterized per-layer — no model-type branching in the compute path.

**CPU path**: BLAS matmul via Apple Accelerate (AMX). Used for attention in `predict_honest`.
**GPU path** (`--features metal`): Q4_K/Q8 Metal shaders with KV cache. Used for decode and prefill.

```bash
# Build with Metal GPU support
cargo build --release -p larql-inference --features metal
```

## BLAS-Fused Attention

The attention kernel uses BLAS `gemv` inside an online-softmax loop. For each query position:

1. `scores = K[0..=qi] @ Q[qi]` (BLAS gemv, AMX-accelerated)
2. Scale + optional softcap + two-pass softmax (f64 accumulation)
3. `output = V[0..=qi]^T @ softmax_scores` (BLAS gemv)

Never allocates the `[seq, seq]` attention matrix. At Gemma-3's head_dim=256, **1.6x faster** than the materialized path. Supports GQA, softcap (Gemma2), attention weight capture.

## WalkFfn

The WalkFfn replaces the dense down projection with a zero-copy mmap read from the vindex:

1. Gate + up projections from model weights (exact, same as dense)
2. GEGLU activation (exact, same as dense)
3. Down projection from mmap'd `down_features.bin` (zero-copy, feature-major)
4. Result is identical to dense FFN — **and faster** (517ms vs 535ms)

The mmap'd feature-major layout has better page cache behavior than the safetensors weight layout.

Build the required vindex files:
```bash
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex
```

### Walk-only mode

Drop FFN weights — 16.6GB → 5.5GB:

```rust
let model = InferenceModel::load_walk_only("google/gemma-3-4b-it")?;
// Frees 10.7 GB of FFN tensors. Requires down_features.bin + up_features.bin.
```

### Server

```bash
cargo run --release -p larql-server -- path/to/vindex --port 8080

curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "top": 5, "mode": "walk"}'
```

## Examples

### Generation stack

```bash
# Token spacing — standalone, no model. Shows the bug
# ("thecapitaloffranceisparis") and the fix.
cargo run --release -p larql-inference --example detok_demo

# Sampling overhead — standalone benchmark across vocab sizes
# (32K/128K/256K) and configs (greedy/temp/top-p/top-k).
cargo run --release -p larql-inference --example bench_sampling

# Sampling, EOS, streaming, chat — model-backed.
cargo run --release --features metal -p larql-inference \
  --example sampling_demo  -- --vindex output/gemma3-4b-v2.vindex
cargo run --release --features metal -p larql-inference \
  --example streaming_demo -- --vindex output/gemma3-4b-v2.vindex --max-tokens 24
cargo run --release --features metal -p larql-inference \
  --example eos_demo       -- --vindex output/gemma3-4b-v2.vindex --max-tokens 80
cargo run --release --features metal -p larql-inference \
  --example chat_demo      -- --vindex output/gemma3-4b-v2.vindex --max-context 256
```

### Other

```bash
# Walk inference benchmark (dense vs walk vs HNSW, needs model + vindex)
cargo run --release -p larql-inference --example bench_walk_inference -- \
  --model google/gemma-3-4b-it --vindex path/to/vindex

# Walk boundary sweep (correctness proof across all 34 layers)
cargo run --release -p larql-inference --example walk_boundary_sweep -- \
  --model google/gemma-3-4b-it --vindex path/to/vindex

# Fused attention demo and benchmark
cargo run --release -p larql-inference --example attention_demo
cargo run --release -p larql-inference --example bench_attention

# Backend demo and benchmark (CPU vs Metal)
cargo run --release -p larql-inference --example backend_demo --features metal
cargo run --release -p larql-inference --example bench_backend --features metal

# Full inference benchmark (needs model weights)
cargo run --release -p larql-inference --example bench_inference

# End-to-end inference demo (needs model weights)
cargo run --release -p larql-inference --example inference_demo

# Clustering and pair matching demos
cargo run -p larql-inference --example clustering_demo
cargo run -p larql-inference --example pair_matching_demo

# Per-layer residual diff: CPU prefill vs Metal prefill (end of every layer)
cargo run --release --features metal -p larql-inference \
    --example residual_diff -- <vindex> "The capital of France is"

# Per-stage L0 bisect: CPU prefill vs Metal KV-cached decode. Locates
# which sub-stage (norm / Q / K / V / attn / O / FFN) first diverges.
# Closed the open Gemma 4 31B parity gap (2026-04-25 ship log) by
# pointing at the FFN block when every attention stage matched at cos=1.0.
cargo run --release --features metal -p larql-inference \
    --example stage_bisect -- <vindex> "The capital of France is" 0
```

### Vindex tools

```bash
# Convert gate vectors from f16 to f32 (zero-copy mmap)
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex

# Build feature-major down vectors (contiguous per-feature layout)
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex
```

## Tests

```bash
# Inference lib tests (904 tests, 65.67% line coverage)
cargo test -p larql-inference --lib

# Gemma 3 4B regression smoke test (set the env var):
LARQL_VINDEX_PATH=$(pwd)/output/gemma3-4b-v2.vindex CI_INTEGRATION=1 \
  cargo test --release -p larql-inference --test test_gemma3_smoke -- --ignored

# All tests including ignored (per-component, kept for reference)
cargo test -p larql-inference

# HNSW tests
cargo test -p larql-vindex --test test_hnsw --release

# Individual test suites
cargo test -p larql-inference --test test_fused_attention             # 23 tests
cargo test -p larql-inference --test test_backend                     # 19 tests
cargo test -p larql-inference --test test_modules                     # 15 tests
cargo test -p larql-inference --test test_trace                       # 14 tests
cargo test -p larql-inference --test test_logits_goldens              # 13 tests
cargo test -p larql-inference --test test_arch_golden                 # 10 tests
cargo test -p larql-inference --test test_layer_graph_integration     #  7 tests
cargo test -p larql-inference --test test_decode_consistency          #  5 tests
```

Walker tests live in `larql-vindex`; weight-level walkers moved out of this crate.

| Area | Tests | Coverage |
|------|-------|----------|
| Generation: EOS / detok / sampling / chat session | 38 | Builtin stops, special-token EOS via tokenizer fallback, leading-space, seed reproducibility, top-k/top-p truncation, whole-turn eviction |
| Vindex strict loader | 2 | open_inference_vindex error paths |
| Backend (ComputeBackend) | 19 | Shape, correctness, batch, Metal vs CPU |
| Fused attention | 23 | GQA, softcap, capture, reference agreement, edge cases |
| FFN + modules | 15 | SiLU, GELU, dense, highway, multi-position |
| Trace stores | 14 | Write/read, tiers, boundaries, additive property |
| Logits goldens | 13 | LM-head dispatch correctness across Q4_K / f16 / f32 paths |
| Arch goldens | 10 | Per-arch fingerprint (Gemma 2/3/4, Llama, Mistral, Qwen) |
| Layer-graph integration | 7 | DenseLayerGraph + WalkLayerGraph + cached graph round-trip |
| Decode consistency | 5 | CPU-decode equality across batch sizes / KV-cache states |
| Unit (lib) | total 904 | Core module tests + everything above. **65.67% line / 67.69% region / 72.96% function coverage; 64 of 127 files at ≥90% line cov.** See [CHANGELOG.md](CHANGELOG.md) for the per-file lift table. |
| Gemma 3 4B smoke (`#[ignore]`) | 1 | First-token regression — gated on `LARQL_VINDEX_PATH` + `CI_INTEGRATION=1` |

## Crate Dependencies

```
larql-models      ModelWeights, architecture traits, quant
larql-compute     ComputeBackend, Q4 matvec, Metal GPU (used by vindex + walk_ffn Q4 paths)
larql-vindex      VectorIndex, gate KNN, adaptive residency, Q4 gates
larql-core        Graph, Edge, algorithms (knowledge graph engine)
    ↓
larql-inference   Forward pass, attention, backends, WalkFfn
```

> **Note:** The GPU pipeline paths (`predict_honest`, `predict_pipeline`) use `larql_compute::ComputeBackend`
> with `FullPipelineLayer` structs that carry per-layer architecture params from `larql_models::ModelArchitecture`.
> The CPU attention and FFN paths use direct BLAS calls via ndarray. Both paths produce identical results.

### Vindex module structure

| Module | Responsibility |
|--------|---------------|
| `types` | FeatureMeta, GateIndex trait, WalkHit, callbacks |
| `core` | VectorIndex struct, constructors, loading, accessors |
| `gate` | Gate KNN: search, batch, scores, HNSW, warmup |
| `walk` | Walk FFN data: mmap'd down/up feature-major vectors |
| `hnsw` | HNSW graph index |
| `mutate` | INSERT/DELETE mutations |
| `router` | MoE expert routing |

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Component breakdown, cross-crate comparison, Ollama reference |
| [ROADMAP.md](ROADMAP.md) | Forward-looking work + open frontiers |
| [CHANGELOG.md](CHANGELOG.md) | Dated ship log (coverage push, bug fixes, H12 splits, MockArch fixtures, magic-strings cleanup) |
| [docs/adr/001](docs/adr/001-fused-attention.md) | BLAS-fused online softmax attention |
| [docs/adr/002](docs/adr/002-walk-ffn.md) | WalkFfn — zero-copy mmap'd down projection |
| [docs/adr/003](docs/adr/003-cached-layer-graph.md) | Cached layer graph for template-fixed layers |
| [docs/adr/004](docs/adr/004-predict-honest.md) | predict_honest — production pipeline with per-layer params |
| [docs/adr/005](docs/adr/005-per-layer-graph.md) | PerLayerGraph — adaptive per-layer strategy |

## License

Apache-2.0
