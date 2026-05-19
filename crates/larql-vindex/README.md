# larql-vindex

The queryable model format. Decompile, browse, edit, and recompile neural networks.

## What is a Vindex?

A vindex (vector index) is a directory containing a transformer model's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format.

```rust
use larql_vindex::*;

// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut SilentLoadCallbacks)?;
let mut patched = PatchedVindex::new(index);

// Query — which features fire for "France"?
let hits = patched.gate_knn(layer, &query, 10);  // 2.7ms/layer at full dim

// Walk — multi-layer feature scan
let trace = patched.walk(&query, &layers, 10);

// Mutate via patch overlay (base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);
patched.set_down_vector(layer, feature, down_vec);

// Apply a saved patch
let patch = VindexPatch::load("medical.vlp")?;
patched.apply_patch(patch);

// Bake patches into a new clean VectorIndex (in-memory)
let baked = patched.bake_down();
baked.save_vindex(&output_path, &mut config)?;

// Or bake the constellation into the canonical down_weights.bin
// via COMPILE INTO VINDEX (see larql-lql) — produces a real
// standalone vindex with no overlay needed at load time, and the
// inserted facts survive a future COMPILE INTO MODEL safetensors
// export because the bytes are sitting in the standard
// down_proj tensors that the manifest references.
```

### Layering note: gate vs down overrides

`PatchedVindex` stores the two kinds of override in different places:

- **Gate vectors** (`insert_feature`, `update_feature_meta`) live in
  `overrides_gate` / `overrides_meta` on the patch overlay. The
  `gate_vectors.bin` on disk is never touched.
- **Down vectors** (`set_down_vector`) are forwarded to the underlying
  base index's `down_overrides` HashMap. The `down_weights.bin` on
  disk is never touched at runtime.

This asymmetry is intentional and load-bearing for `COMPILE INTO
VINDEX`. The dense FFN inference path reads gate scores from
`gate_vectors.bin`; baking norm-matched override gates there would
produce moderate dense activations that combined with the override
down vectors would blow up the residual stream. Keeping the source's
weak free-slot gate at the inserted index keeps the dense activation
small, so `small_activation × poseidon_vector` per layer accumulates
into the validated multi-layer constellation effect. See
`patch/core.rs` for the full doc on `PatchedVindex`.

### Refine pass (`patch/refine.rs`)

`refine_gates(inputs, decoy_residuals) -> RefineResult` orthogonalises
each patched gate against the other patched gates at the same layer,
plus any decoy residuals supplied by the caller. Pure Gram-Schmidt over
`Array1<f32>` slices — no model dependency, no forward passes. The
result carries the refined gates plus per-fact `retained_norm`
statistics.

This is the load-bearing fix for cross-fact bleed and is called by
INSERT's batch refine pass at install time.
Refining is per-layer (facts at different layers can't interfere
through the FFN math). Decoy residuals are layer-scoped — the caller is
responsible for capturing them at the correct depth, which is exactly
what `larql_inference::capture_decoy_residuals` does. Validated against
synthetic constellations by the unit tests in `patch/refine.rs`; the
end-to-end Gemma 3 4B reproduction lives in
`larql-lql/examples/refine_demo.rs`.

## The Headline

A 1T model in 10.9 GB on a laptop.

```
Model          Full Inference RAM    Vindex Infer RAM    Ratio
Gemma 3 4B              7 GB              1.3 GB          5x
Llama 3 8B             15 GB              2.2 GB          7x
Llama 3 70B           130 GB              4.9 GB         27x
Llama 3 405B          754 GB              8.6 GB         88x
DeepSeek V3          1250 GB             10.9 GB        115x
Kimi-K2              1863 GB             10.9 GB        171x
```

Vindex inference uses mmap: only 1 layer of gate vectors + 1 layer of attention
weights are resident at a time. The rest stays on disk until touched.

## Features

- **Extract** from safetensors, GGUF, or MLX models (streaming — no full model load)
- **Gate KNN** via BLAS matmul, Q4 matvec (CPU/Metal/CUDA), or HNSW approximate search
- **Walk** across all layers with down-meta annotation
- **Readonly base** — base vindex files are never modified after extraction
- **Patch overlay** — all mutations go through PatchedVindex (INSERT/DELETE/UPDATE)
- **Patches** — stackable, reversible knowledge diffs (.vlp files)
- **Vindexfile** — declarative model builds (FROM + PATCH + INSERT, like Dockerfile)
- **HuggingFace Hub** — download and publish vindexes (`hf://user/repo` URI scheme)
- **Split weight files** — gate, up, down, attn, norms, lm_head (no duplication)
- **Zero-copy mmap** — gate vectors sliced directly from disk, no heap allocation
- **Binary down_meta** — compact binary format (no JSONL)
- **f16 storage** — halves file sizes with negligible accuracy loss
- **MoE support** — Mixtral, DeepSeek (experts as contiguous features)
- **Layer bands** — per-family boundaries (Gemma, Llama, Qwen, etc.)
- **Checksums** — SHA256 integrity verification for all binary files
- **Provenance** — model source, timestamp, version tracking
- **LM head KNN** — top-K token lookup via single BLAS gemv against output projection
- **Adaptive residency** — pin hot layers in memory, stream cold ones. More memory = faster. Smooth gradient vs llama.cpp's all-or-nothing cliff

## Crate Structure

> **Note**: tree below reflects the layout after the 2026-05-09 round-5
> cleanup (see `CHANGELOG.md`). All the directories under
> `index/`, `format/weights/`, and `extract/` follow the same pattern:
> a `mod.rs` declares siblings and owns the public surface; each
> sibling carries one `impl <Type>` block, one trait, or one emitted
> artefact. After round-5 no non-test source file in the crate exceeds
> the soft 600-LOC threshold.

```
larql-vindex/src/
├── lib.rs                      Crate root + re-exports
├── error.rs                    VindexError
├── describe.rs                 DescribeEdge, LabelSource
├── mmap_util.rs                madvise-optimized mmap helper
│
├── config/                     Configuration types
│   ├── index.rs                VindexConfig, VindexLayerInfo, ExtractLevel,
│   │                           LayerBands, VindexSource (v1 provenance:
│   │                           base_model_sha, extractor_sha,
│   │                           base_safetensors_sha256), checksums
│   ├── quantization.rs         QuantFormat, Fp4Config, Precision, Projections
│   ├── model.rs                VindexModelConfig, MoeConfig
│   ├── compliance.rs           ComplianceGate
│   └── dtype.rs                StorageDtype (f32/f16), encode/decode/write_floats
│
├── index/                      In-memory KNN engine (zero-copy mmap)
│   ├── types/                  Shared types + capability traits (round-5 split)
│   │   ├── mod.rs              FeatureMeta, DEFAULT_C_SCORE, WalkHit, WalkTrace,
│   │   │                       StorageBucket, GateLayerSlice, GateQ4Slice,
│   │   │                       DownMetaMmap (binary record decoder),
│   │   │                       IndexLoadCallbacks + SilentLoadCallbacks
│   │   ├── gate_lookup.rs      GateLookup trait (gate KNN + feature meta)
│   │   ├── patch_overrides.rs  PatchOverrides trait (overlay vector hooks)
│   │   ├── native_ffn.rs       NativeFfnAccess trait (f32/f16 row access)
│   │   ├── quantized_ffn.rs    QuantizedFfnAccess trait (Q4_0/Q4_K/Q6_K)
│   │   ├── fp4_ffn.rs          Fp4FfnAccess trait (FP4/FP8, exp 26)
│   │   └── ffn_row.rs          FfnRowAccess unified dispatch + GateIndex
│   │                           (composed blanket impls — fp4 → native → q4k)
│   ├── core/                   VectorIndex + capability impls (round-5 split)
│   │   ├── mod.rs              VectorIndex struct + Clone + constructors +
│   │   │                       cross-store regression tests
│   │   ├── gate_lookup.rs      impl GateLookup for VectorIndex
│   │   ├── patch_overrides.rs  impl PatchOverrides (real, against MetadataStore)
│   │   ├── native_ffn.rs       impl NativeFfnAccess (delegation to inherent)
│   │   ├── quantized_ffn.rs    impl QuantizedFfnAccess
│   │   └── fp4_ffn.rs          impl Fp4FfnAccess
│   ├── compute/                KNN dispatch + HNSW + GPU paths
│   │   ├── gate_knn/
│   │   │   ├── mod.rs          top_k_by_abs free fn + top_k_from_scores impl shim + tests
│   │   │   ├── dispatch.rs     gate_knn, gate_knn_expert, gate_knn_batch,
│   │   │   │                   gate_knn_adaptive, gate_knn_q4, walk, gate_walk
│   │   │   ├── scores_batch.rs gate_scores_batch + GPU/BLAS fast paths
│   │   │   └── hnsw_lifecycle.rs build/install/warmup + HNSW-backed knn variants
│   │   ├── hnsw.rs             HNSW graph index (random projection + exact rescoring)
│   │   ├── q4k_dispatch.rs     Compute-side Q4_K codec dispatch (matmul + row decode)
│   │   └── router.rs           MoE expert router
│   ├── mutate/                 set_down_vector, set_up_vector, save_*
│   └── storage/                Substores composed into VectorIndex
│       ├── gate_store.rs       GateStore (mmap + heap gate vectors + warmed cache)
│       ├── gate_accessors.rs   feature_meta, gate_vector, num_features, warmup
│       ├── ffn_store/
│       │   ├── mod.rs          FfnStore struct + Clone + ffn_layer_byte_offset
│       │   ├── down.rs         down_features.bin (feature-major f32)
│       │   ├── up.rs           up_features.bin (feature-major f32) + has_full_mmap_ffn
│       │   ├── interleaved.rs  interleaved.bin (f32 [gate|up|down])
│       │   ├── interleaved_q4.rs   interleaved_q4.bin (Q4_0)
│       │   ├── interleaved_q4k.rs  interleaved_q4k.bin + manifests +
│       │   │                       down_features_q4k.bin (Q4_K/Q6_K)
│       │   ├── gate_q4.rs      Q4_0 gate-vector mmap (KNN side-channel)
│       │   ├── fp4.rs          FP4 / FP8 FFN storage (exp 26)
│       │   └── q4k_cache.rs    Bounded LRU dequant cache (q4k_ffn_cache)
│       ├── lm_head/
│       │   ├── mod.rs          Q4 byte-rate constants + manifest helper + tests
│       │   ├── loaders.rs      load_lm_head_q4, synthesize_lm_head_q4,
│       │   │                   set_lm_head_f16_mmap, load_lm_head
│       │   └── knn.rs          lm_head_knn_backend (Q4/f16/f32) + skip_q4k variant +
│       │                       top_k_sorted reduce + lm_head_knn (f32 fallback)
│       ├── attn.rs             Attention weight loaders (Q8, Q4_K, Q4)
│       ├── vindex_storage/     VindexStorage trait + MmapStorage impl
│       │                       (single source of truth for every
│       │                       file-backed mmap; loaders mutate via
│       │                       Arc::make_mut + set_*; release_pages
│       │                       madvise covers all tracked handles)
│       ├── metadata_store.rs   MetadataStore (down_meta + overrides)
│       ├── fp4_store.rs        Fp4Storage runtime store (exp 26)
│       └── residency.rs        Adaptive layer pinning (memory → performance)
│
├── format/                     Vindex file I/O
│   ├── load.rs                 load_vindex, load_embeddings, load_tokenizer
│   ├── down_meta.rs            Binary down_meta read/write
│   ├── filenames.rs            Single source of truth for *.bin / *.json names —
│   │                           UP_WEIGHTS_BIN / DOWN_WEIGHTS_BIN added 2026-05-01
│   ├── weights/
│   │   ├── mod.rs              Re-exports
│   │   ├── write_f32.rs        write_model_weights (f32/f16), WeightEntry/Source
│   │   ├── write_q4k/          Q4_K / Q6_K streaming writer (round-5 split — one
│   │   │                       sibling per emitted artefact)
│   │   │   ├── mod.rs          Orchestrator + Q4kWriteOptions + QuantBlockFormat +
│   │   │   │                   pad_rows_to_block + resolve_v_tensor + helper tests
│   │   │   ├── attn.rs         attn_weights_q4k.bin + manifest
│   │   │   ├── ffn.rs          interleaved_q4k.bin + opt down_features_q4k.bin
│   │   │   ├── moe_layers.rs   layers/layer_{L:02}.weights (hybrid MoE)
│   │   │   ├── norms.rs        norms.bin (norms + MoE router/scales)
│   │   │   ├── ple.rs          ple_weights.bin (Gemma 4 E2B PLE, f16)
│   │   │   ├── lm_head.rs      lm_head_q4.bin
│   │   │   └── feature_major_down.rs  W2 down_features_q4k.bin state
│   │   ├── write_layers.rs     Per-layer FFN file writer (§5.12)
│   │   ├── manifest.rs         Q4kManifestEntry + format_tag
│   │   └── load/               load_model_weights, find_tokenizer_path
│   │       ├── mod.rs          Public API + LoadWeightsOptions + expert_in_shard
│   │       ├── f32.rs          load_model_weights_with_opts (f32/f16 path)
│   │       └── q4k.rs          load_model_weights_q4k_shard (Q4_K path)
│   ├── checksums.rs            SHA256 computation + verification
│   ├── fp4_codec.rs            FP4 / FP8 codec (extraction-side)
│   ├── huggingface/            HuggingFace Hub download/publish
│   ├── spec.rs                 Translation: VindexConfig → public v1
│   │                           manifest (`larql-vindex-spec::VindexManifest`).
│   │                           TryFrom impl, surfaces missing provenance
│   │                           (base_model_sha / extractor_sha /
│   │                           safetensors-digests map) as typed errors.
│   └── quant/mod.rs            Re-exports from larql_models::quant
│
├── extract/                    Build pipeline (model → vindex)
│   ├── build/
│   │   ├── mod.rs              BuildContext struct + small stages + build_vindex + tests
│   │   ├── down_meta.rs        Stage 3: per-feature top-k + cluster collection
│   │   └── index_json.rs       Stage 6: config + provenance + checksums
│   ├── build_helpers.rs        chrono_now, build_whole_word_vocab,
│   │                           compute_gate_top_tokens, compute_offset_direction,
│   │                           run_clustering_pipeline, ClusterData
│   ├── streaming/              Streaming extraction (mmap, no full model load)
│   │   ├── mod.rs              Orchestrator (StreamingContext lifecycle)
│   │   ├── context.rs          StreamingContext struct + new + finalize
│   │   ├── tensor_io.rs        MmapShard, GateSink, get_tensor_f32, normalize_key
│   │   └── stages/             One sibling per pipeline stage (round-5 split)
│   │       ├── gate_vectors.rs    Stage 1 — gate_vectors.bin + layer_infos
│   │       ├── router_weights.rs  Stage 1b — router_weights.bin (MoE only)
│   │       ├── embeddings.rs      Stage 2 — embeddings.bin
│   │       ├── down_meta.rs       Stage 3 — per-feature top-K → down_meta.bin
│   │       ├── tokenizer.rs       Stage 4 — tokenizer.json
│   │       ├── index_json.rs      Stage 5 — preliminary index.json
│   │       └── model_weights.rs   Stage 6 — write_model_weights / _q4k dispatch
│   ├── stage_labels.rs         15 labels for IndexBuildCallbacks (compile-time pinned)
│   ├── callbacks.rs            IndexBuildCallbacks trait
│   ├── checkpoint.rs           Phase-level resume checkpoint
│   └── build_from_vectors.rs   Build from pre-extracted NDJSON
│
├── patch/                      Patch system
│   ├── format.rs               VindexPatch, PatchOp (Insert/Update with optional
│   │                           gate/up/down vectors), PatchDownMeta + base64
│   ├── overlay.rs              PatchedVindex (queries, mutators, walk, bake_down)
│   ├── overlay_apply.rs        apply_patch, remove_patch, rebuild_overrides
│   ├── overlay_gate_trait.rs   impl GateIndex for PatchedVindex
│   ├── knn_store.rs            L0 KnnStore (arch-B residual-key KNN)
│   ├── knn_store_io.rs         KnnStore .lknn save / load (f16 keys)
│   └── refine.rs               Gate refine pass (Gram-Schmidt orthogonalisation
│                               of patched gates + optional decoy residuals)
│
├── engine/                     Storage engine + L2 MEMIT cycles
│   ├── engine.rs               StorageEngine (PatchedVindex + epoch + memit_store)
│   ├── epoch.rs                Monotonic mutation counter
│   ├── status.rs               CompactStatus snapshot
│   └── memit_store.rs          MemitStore + MemitFact + memit_solve
│
├── quant/                      Quant codec registry + format scanning
│   ├── registry.rs             QUANT_FORMATS table + lookup() — adding a K-quant
│   │                           is one entry. LEGACY_BLOCK_Q4_K_STRIDE = 148
│   │                           (round-4 M5)
│   ├── convert.rs              f32/f16 → Q4_K conversion (post-extract path)
│   ├── convert_q4k.rs          Whole-vindex f32 → Q4_K conversion + auxfile linking
│   └── scan.rs                 FP4 compliance scanner (exp 26 Q1 outcomes)
│
├── clustering/                 Relation discovery
│   ├── kmeans.rs               k-means clustering (BLAS via larql-compute)
│   ├── labeling.rs             Pattern detection, TF-IDF labels
│   ├── categories.rs           Entity category word lists
│   ├── pair_matching/          RelationDatabase + Wikidata/WordNet loaders
│   └── probe.rs                Probe label loading
│
└── vindexfile/                 Declarative model builds
    ├── mod.rs                  Build executor (FROM + PATCH + INSERT → bake_down)
    └── parser.rs               Vindexfile parser (FROM, PATCH, INSERT, DELETE, etc.)
```

All matrix operations go through `larql-compute` (BLAS on CPU, Metal GPU planned for gate KNN).

## MEMIT decomposition (`storage/memit_store.rs`)

`memit_solve` is the vanilla closed-form MEMIT decomposition that
populates `MemitStore` during `COMPACT MAJOR`. It wraps the generic
`larql_compute::cpu::ops::linalg::ridge_decomposition_solve` with the
MEMIT interpretation:

```rust
use larql_vindex::{memit_solve, MemitFact, MemitStore};

let solve = memit_solve(&keys, &targets, lambda)?;
//   solve.delta_w           — (d, d) weight update
//   solve.decomposed[i]     — ΔW @ k_i   (one row per fact)
//   solve.reconstruction_cos[i] — cos(ΔW k_i, t_i)
//   solve.max_off_diagonal  — cross-template interference
//   solve.frobenius_norm    — ‖ΔW‖_F

let facts: Vec<MemitFact> = /* package decomposed pairs */;
store.add_cycle(layer, facts, solve.frobenius_norm,
                min_cos, solve.max_off_diagonal);
```

This is **vanilla** MEMIT — no covariance whitening. Cross-template
bleed grows with N when keys share a dominant direction (the canonical-
form template case from exp 8). For production weight edits with C⁻¹
whitening + per-fact optimised target deltas (the validated v11 200/200
pipeline), use `larql-inference::forward::memit`.

| Run | Command |
|-----|---------|
| Demo | `cargo run --release -p larql-vindex --example demo_memit_solve` |
| Bench | `cargo bench -p larql-vindex --bench memit_solve` |

## Compute Integration

| Module | Operation | Backend |
|--------|-----------|---------|
| gate.rs | Gate KNN f32 (matmul_transb) | CPU BLAS |
| gate.rs | Gate KNN Q4 (q4_matvec) | Any ComputeBackend |
| gate.rs | Adaptive KNN (pinned → Q4 → f32) | Any ComputeBackend |
| gate.rs | Gate walk (gemv) | CPU BLAS |
| gate.rs | Batch gate scores (matmul_transb) | CPU BLAS |
| hnsw.rs | Random projection (matmul) | CPU BLAS |
| hnsw.rs | Dot product (graph traversal) | CPU BLAS |
| walk.rs | LM head KNN (matmul_transb) | CPU BLAS |
| kmeans.rs | Similarity matrix (matmul_transb) | CPU BLAS |
| router.rs | MoE routing (matmul) | CPU BLAS |

## Supported Architectures

| Family | Models | FFN Type | Notes |
|--------|--------|----------|-------|
| Gemma 4 | Gemma 4 31B/E2B | Gated (GeGLU) | Per-layer head_dim, K=V, V-norm, partial RoPE, PLE, KV sharing |
| Gemma 3 | Gemma 3 (4B-27B) | Gated (GeGLU) | QK-norm, sliding window, dual RoPE |
| Gemma 2 | Gemma 2 (2B-27B) | Gated (GeGLU) | Softcapping, QK-norm |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) | GQA, RoPE scaling |
| Mistral | Mistral 7B | Gated (SiLU) | Sliding window |
| Mixtral | Mixtral 8x7B/8x22B | MoE (8 experts) | PerExpert format |
| Qwen | Qwen 2/2.5/3 | Gated (SiLU) | Attention bias, QK-norm |
| Phi | Phi 2/3 | Gated | |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) | MLA, YaRN |
| Granite | Granite | Gated (SiLU) | Scaling multipliers |
| StarCoder2 | StarCoder2 | Standard (GELU) | LayerNorm, bias, non-gated FFN |
| GPT-OSS | GPT-OSS | MoE (PackedMxfp4) | MXFP4 packed experts |
| GPT-2 | GPT-2 | Dense (GELU) | |

## File Layout

```
model.vindex/
├── gate_vectors.bin        W_gate per layer (f32/f16 KNN index)
├── gate_vectors_q4.bin     W_gate Q4_0 (7x smaller, for Q4 KNN)
├── embeddings.bin          W_embed matrix
├── down_meta.bin           Per-feature output metadata (binary)
├── attn_weights.bin        Q, K, V, O per layer
├── up_weights.bin          W_up per layer
├── down_weights.bin        W_down per layer
├── norms.bin               LayerNorm parameters
├── lm_head.bin             Output projection
├── interleaved.bin         gate|up|down packed per layer (optional)
├── interleaved_q4.bin      Q4_0 quantized version (optional, 7x smaller)
├── interleaved_q4k.bin     Q4_K gate/up + Q6_K down (when quant=q4k)
├── interleaved_q4k_manifest.json  Per-tensor offsets for interleaved_q4k.bin
├── attn_weights_q4k.bin    Q4_K Q/K/O + Q6_K V (when quant=q4k)
├── attn_weights_q4k_manifest.json Per-tensor offsets for attn_weights_q4k.bin
├── ple_weights.bin         Per-Layer Embedding tensors at f16 (Gemma 4 E2B only)
├── index.json              Config, layer bands, provenance, checksums, quant format
├── tokenizer.json          Tokenizer
├── relation_clusters.json  Discovered relation types
├── feature_labels.json     Probe-confirmed labels
└── weight_manifest.json    Weight file → offset mapping
```

## Extract Levels

| Level | Size (f16) | Enables |
|-------|-----------|---------|
| Browse | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | ~6 GB | + INFER |
| All | ~8.5 GB | + COMPILE |

## Streaming Quantisation (`--quant q4k`)

`build_vindex_streaming` can quantise model weights inline as it reads
the safetensors shards, skipping the f32 intermediate entirely. Pass
`QuantFormat::Q4k` (or `--quant q4k` on the CLI) to emit Ollama-
compatible blocks:

- Q/K/O/gate/up → Q4_K (144 bytes per 256 values, GGUF-canonical)
- V/down → Q6_K (210 bytes per 256 values)

Output files: `attn_weights_q4k.bin` + `interleaved_q4k.bin` with
per-tensor manifests. `VindexConfig.quant = Q4k` in `index.json` so
loaders can dispatch on config.

### Stride validation (loud failure on stale vindexes)

`load_attn_q4k` walks every manifest entry and compares its `length`
to `QuantFormatInfo::expected_bytes(&shape)`. On mismatch it returns
`VindexError::Parse` with rebuild guidance:

```
attn_weights_q4k_manifest: tensor "layers.0.self_attn.q_proj.weight"
(Q4_K, shape [2048, 2560]) has length 3031040 but format expects 2949120
(144 bytes/block × 21048). Likely cause: vindex built with legacy
148-byte block_q4_K layout — rebuild the vindex with current code.
```

Pre-stride-validation, a vindex written before the GGUF-canonical
144-byte writer landed (the legacy `block_q4_K` MSL struct uses 148
bytes/block — 4 extra `mins[4]` padding) loaded silently. The kernel
read off-stride by 4 bytes per superblock, drift accumulated across
rows, and GPU prefill produced all-NaN. The validator catches this at
load time so callers see a clear "rebuild" error rather than garbage
decode output. See `index/storage/attn.rs::load_attn_q4k_rejects_legacy_148_byte_stride`.

### `vocab_size` propagation

`load_vindex` propagates `config.vocab_size` from `index.json` to the
loaded `VectorIndex` unconditionally. Previously this only happened in
the embeddings-as-tied-lm_head adoption block, so a vindex shipping
`lm_head_q4.bin` (current Q4_K writer's default) but no `lm_head.bin`
loaded with `vocab_size = 0`. The Q4 lm_head fast path then silently
bailed (`if vocab > 0`), forcing a 4× slower fallback through the f32
BLAS gemv — measured 8.4 ms vs 1.9 ms per token on Gemma 3 4B. Belt
and braces: `load_lm_head_q4` also derives `vocab_size` from the file
size when it's still 0 (Q4_K and Q4_0 both work out to 0.5625
bytes/element). Regression test:
`load_lm_head_q4_sets_vocab_size_from_file_size`.

When `quant != None`, `--level browse` is implicitly promoted to
`--level all` — the Q4_K writer emits all of attention, FFN, norms,
and `lm_head` in one pass, and a browse-only Q4k vindex would be
incoherent.

### Per-Layer Embeddings (Gemma 4 E2B)

E2B's Per-Layer Embedding tensors don't go through Q4_K because the
per-super-block (d, dmin) calibration destroys embedding-style tensors
— one outlier row per super-block pulls the scale, zeroing the other
255 cells. The noise then compounds across 35 layers' additive PLE
contributions. Instead they land in `ple_weights.bin` at **f16**:

- `per_layer_model_projection.weight`  (~27 MB at f16)
- `embed_tokens_per_layer.weight`      (~4.7 GB at f16 on E2B)
- `layers.N.per_layer_input_gate.weight` + `per_layer_projection.weight`

Load dequantises to f32 at mmap time and inserts into `weights.tensors`.
`larql_inference::forward::ple::precompute_per_layer_inputs` and
`apply_per_layer_embedding` then work unchanged.

### E2B caveats worth knowing

- **Cross-layer KV sharing** (`num_kv_shared_layers=20`): layers 15-34
  reuse K/V computed by the last unshared sliding / global layer. The
  Q4 forward path threads a `kv_cache` through the loop to honour this.
- **Double-wide MLP** (`use_double_wide_mlp=True`): half the layers
  ship with `intermediate=12288` while the model-wide config reports
  6144. `VectorIndex::num_features(layer)` is the authoritative
  per-layer FFN width; don't read `weights.intermediate_size` in any
  dequant / forward code.
- **Final-logit softcap** (`final_logit_softcapping=30.0`): preserved
  through `VindexModelConfig.final_logit_softcapping`. Missing it lets
  `logits_to_predictions` peak on the wrong token — there is no "fail
  loudly" mode for a dropped softcap, only a silent accuracy hit.

## Recommended setup for `larql-inference`

Production decode through `larql-inference` is **full-K Metal**:
`q4k_matmul_transb` streams Q4_K bytes from the mmap straight into a
GPU shader (no per-feature loops, no dequant cache). The vindex's job
on this path is to be a thin mmap shim — most knobs below shift weight
between disk, RSS, and startup latency rather than steady-state tok/s.

### Default — single-host Metal decode (Gemma / Llama / Qwen / ...)

```bash
larql extract-index <model> -o <vindex> --quant q4k
```

That's it. Metal decode bypasses the `q4k_ffn_layer` cache entirely
(`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB` — see
`PERFORMANCE.md`), so you don't need `--feature-major-down`. HNSW is
optional — leave it off unless you're going to interpret-walk.

### Multi-shard grid (`larql-router` + per-layer-range `larql-server`)

Two topology options:

**Option A — static grid (`--shards`)**: simpler ops, router needs
all shards' URLs at boot.

```bash
larql extract-index <model> -o <vindex> --quant q4k --feature-major-down
# (or, for an existing q4k vindex without W2:)
larql convert add-feature-major-down --input <vindex>

# Per shard — same vindex path, distinct port, distinct layer range.
larql-server <vindex> --port 9181 --layers 0-14 --no-infer \
    --max-q4k-cache-layers 1 --warmup-walk-ffn
larql-server <vindex> --port 9182 --layers 15-29 --no-infer \
    --max-q4k-cache-layers 1 --warmup-walk-ffn

# Router with static map.
larql-router --shards 0-14=http://127.0.0.1:9181,15-29=http://127.0.0.1:9182 \
             --port 9090
```

**Option B — self-assembling grid (`--grid-port` + `--join`)**:
shards register dynamically over gRPC; the router tracks coverage
live and reports `total_layers_covered` as shards join/leave.
Recommended for production where shards may be added or restarted
without bouncing the router.

```bash
# Router exposes HTTP on 9090 + grid gRPC on 50052.
larql-router --grid-port 50052 --grid-key <secret> --port 9090

# Shards register themselves via --join. They need --public-url so
# the router knows where to send clients.
larql-server <vindex> --port 9181 --layers 0-14 --no-infer \
    --max-q4k-cache-layers 1 --warmup-walk-ffn \
    --join http://127.0.0.1:50052 --grid-key <secret> \
    --public-url http://host-a:9181

larql-server <vindex> --port 9182 --layers 15-29 --no-infer \
    --max-q4k-cache-layers 1 --warmup-walk-ffn \
    --join http://127.0.0.1:50052 --grid-key <secret> \
    --public-url http://host-b:9182
```

Live-validated (2026-04-26): auto-join, coverage tracking, graceful
failure (router returns HTTP 400 `"layer N has no owning shard"`
when a covering shard is gone), auto-recovery on rejoin.

Either way, each shard `larql-server` mmaps its layer range. Adding
`--feature-major-down` at extract time (W2, see ADR-009) emits
`down_features_q4k.bin`, which lets each shard skip the ~840 MB
heap cache ceiling on its slice. Recommended when:

- shard count is high (per-shard RSS budget is tight),
- the model is large enough that 14 MB / layer of disk overhead is
  acceptable in exchange for bounded RSS (Gemma 4B → +500 MB),
- workloads include CPU walk fallback (the cache *would* otherwise fire).

If the shard host has spare cores at startup, eager-build HNSW across
its layer range:

```rust
index.enable_hnsw(200);
index.warmup_hnsw_all_layers();   // 3.6× speedup on 8L Gemma; ~700 ms for 34L
```

Live perf snapshot (Gemma 26B, 2-shard grid, M3 Max): full-30-layer
fan-out **5.9 ms warm** via either router topology; cold first
request **12.6 ms** with `--warmup-walk-ffn`, **1247 ms** without.
8-way concurrent × 15-layer fan-out: **112 ms wall, ~1070
layer-evals/sec**.

### MoE expert hosts (Kimi K-series, DeepSeek-V3+)

Same as the grid recipe. Each expert host touches its experts once or
twice per token, never amortising the `q4k_ffn_layer` cache. With
`--feature-major-down` the per-feature down decode is a single row
dequant (2440× faster on first access at K=100, 25× at full K — see
PERFORMANCE.md round-4). Cap the legacy cache at 1 layer or 0:

```bash
larql serve <vindex> --max-q4k-cache-layers 1
```

### Interpretability / walk-heavy CPU pipelines

Walks query gate KNN per layer rather than full-K matmul. Enable the
parallel batch path (automatic for `seq_len ≥ 16`) and HNSW warmup at
startup:

```rust
let index = VectorIndex::load_vindex(&path, ...)?;
index.enable_hnsw(200);
index.warmup_hnsw_all_layers();
let trace = index.walk(&query, &layers, 10);
```

For batch / prefill (multi-position walks), `gate_knn_batch` already
parallelises per-position top-K extraction when `seq_len ≥ 16` — no
caller change needed. Production prefill at seq_len=256 sees -24 % vs
the serial path.

## Recommended setup for `larql-server`

`larql-server` exposes a vindex over HTTP/gRPC for `larql-router`-driven
multi-shard grids. It's a long-running daemon — startup latency, RSS
ceilings, and per-request KNN tail latency all matter.

### Single-host serve (one shard, full model)

```bash
larql-server <vindex.path> --port 9180
```

Out of the box, `larql-server` mmaps the whole vindex, exposes
`/knn`, `/walk`, `/infer`, etc. Production decode auto-selects the
Metal backend on Apple Silicon — full-K matmul through
`q4k_matmul_transb` is 2.4–4× faster than CPU on Gemma 4B
10240×2560 (see the CPU-vs-GPU table in `PERFORMANCE.md`).

For interp-style endpoints (`/walk`, `/knn` per layer), opt in to
HNSW + parallel warmup — typical 34-layer Gemma 4B startup goes
from ~2.6 s lazy to ~700 ms eager:

```bash
larql-server <vindex.path> --port 9180 --hnsw --hnsw-ef-search 200 --warmup-hnsw
```

`--warmup-hnsw` triggers `warmup_hnsw_all_layers()` at boot (3.6×
speedup vs lazy build); requires `--hnsw`.

**For `walk-ffn` traffic** (any model that serves `/v1/walk-ffn`),
add `--warmup-walk-ffn` to pay the ~1.3 s lazy `get_or_load_weights`
cost at boot instead of on the first request. Measured on a Gemma
26B vindex: first walk-ffn drops from **1247 ms** (cold) to **12.6 ms**
(warm) — a **99× speedup**. The cost is +3.2 GB pre-allocated RSS
and ~1.3 s of additional boot time. Operators can also fire `POST
/v1/warmup` against a running server without a restart (request
body is `{layers?, skip_weights?, warmup_hnsw?}`, all optional).

### Multi-shard grid (`larql-router` + N × `larql-server`)

Each shard owns a layer range. Recommended extract + run:

```bash
# Build the vindex once with feature-major down so each shard avoids
# the ~840 MB heap cache ceiling on its slice.
larql extract-index <model> -o <vindex> --quant q4k --feature-major-down

# Per shard — same vindex path, distinct port, distinct layer range.
larql-server <vindex.path> --port 9181 --layers 0-16 --no-infer \
  --max-q4k-cache-layers 1
larql-server <vindex.path> --port 9182 --layers 17-33 --no-infer \
  --max-q4k-cache-layers 1

# Router on top.
larql-router --shards 0-16=http://127.0.0.1:9181,17-33=http://127.0.0.1:9182 \
             --port 9190
```

Why each flag matters:
- `--feature-major-down` (extract-time) — emits `down_features_q4k.bin`.
  Activates when the FFN walk dispatches through the *sparse* path
  (`walk_ffn_sparse` — INSERT-patched layers, explicit sparse-K, or
  FP4 storage). On those paths, per-feature down decode reads one row
  from the new file instead of dequantising the whole layer +
  transposing through the cache; deletes the binding RSS constraint
  on per-shard memory budget. The default dense Q4K HTTP walk
  (`walk_ffn_q4k_dequant`) does its own one-shot whole-layer dequant
  and uses neither the cache nor W2 — so for pure-dense grids
  W2's value is the *capability* (you can attach a patch / switch on
  sparse mode without the cache lighting up), not the ms saved on
  every request. See [docs/adr/009](docs/adr/009-feature-major-down.md)
  for the architectural decision and `/v1/stats.q4k_ffn` for live
  status (`feature_major_down: true` + `cache_slots: 0` is the
  healthy steady state).
- `--max-q4k-cache-layers 1` — caps the legacy `q4k_ffn_layer` cache
  at one layer. With feature-major down loaded the cache is barely
  used; this just bounds it. (Set to 0 to disable entirely once
  every vindex on the grid has feature-major down.)
- `--no-infer` — shards typically don't run the decode loop; the
  router orchestrates. Skipping inference setup saves a chunk of
  GPU buffer allocation per shard.
- `--layers <range>` — server reads + answers queries only for its
  range. The mmaps are demand-paged so unowned layers stay
  paged-out.

### Bench discipline on grid hosts

The `vindex_scaling` and `cpu_vs_gpu` benches refuse to run while
`larql-server` or `larql-router` is on the same host (3× run-to-run
swing observed in the 2026-04-25 audit). To bench against a live
grid intentionally, set `LARQL_BENCH_ALLOW_DAEMONS=1`.

## Testing

```bash
# Local CI gate, matching the crate-specific Makefile surface.
make larql-vindex-ci                                                            # fmt, clippy, tests, examples, bench tests, coverage policy
make larql-vindex-test                                                          # cargo test -p larql-vindex
make larql-vindex-fmt-check                                                     # cargo fmt -p larql-vindex -- --check
make larql-vindex-lint                                                          # cargo clippy -p larql-vindex --all-targets -- -D warnings
make larql-vindex-examples                                                      # cargo check -p larql-vindex --examples
make larql-vindex-bench-test                                                    # cargo test -p larql-vindex --benches
make larql-vindex-bench                                                         # cargo bench -p larql-vindex --bench vindex_ops
make larql-vindex-coverage-summary                                              # aggregate + per-file coverage policy
make larql-vindex-coverage-html                                                 # HTML report plus the same coverage policy

cargo test -p larql-vindex                                                      # 857 lib tests listed as of 2026-05-10

# Demos (synthetic fixtures, no model download needed)
cargo run -p larql-vindex --example demo_features                               # Feature showcase (build, KNN, patches, MoE, f16)
cargo run --release -p larql-vindex --example mmap_demo                         # mmap RAM behaviour + scaling table
cargo run --release -p larql-vindex --example q4k_demo                          # Streaming Q4_K showcase: size comparison, file layout, dequant round-trip
cargo run --release -p larql-vindex --example demo_memit_solve                  # MEMIT closed-form decomposition + MemitStore round-trip

# Criterion benches (run with --quick for a fast sweep, omit for full sample)
cargo bench  -p larql-vindex --bench vindex_ops                                 # KNN, walk, save/load, mutate, MoE, batch top-K
cargo bench  -p larql-vindex --bench vindex_scaling                             # Production dims (CPU only — Metal in cpu_vs_gpu below)
cargo bench  -p larql-vindex --bench cpu_vs_gpu                                 # CPU only (Accelerate)
cargo bench  -p larql-vindex --features metal --bench cpu_vs_gpu                # CPU + Metal side-by-side at production dims
cargo bench  -p larql-vindex --bench memit_solve                                # Ridge decomposition throughput
cargo bench  -p larql-vindex --bench extract_throughput                         # Streaming extract: f32 vs Q4K vs Q4K-resume
cargo bench  -p larql-vindex --bench q4k_vs_f32                                 # Per-layer attn retrieval: mmap memcpy vs mmap + dequant
cargo bench  -p larql-vindex --bench q4k_cache                                  # Q4_K dequant cache vs row + W2 down feature-major
cargo bench  -p larql-vindex --bench hnsw_decode                                # HNSW vs brute + parallel warmup_hnsw_all_layers

# Streaming build (one-shot, skips f32 intermediate)
larql extract-index <model> -o <vindex> --quant q4k                             # Q4_K/Q6_K attn + FFN + norms + lm_head in one pass

# Multi-tier build pipeline (post-hoc, uses larql-compute quantizers on an
# already-extracted f32 vindex — kept for backwards compatibility)
cargo run --release -p larql-vindex --example build_q4k_weights -- <vindex>     # Q4_K/Q6_K attn + FFN
cargo run --release -p larql-vindex --example build_attn_q8 -- <vindex>         # Q8 attention (fallback)
cargo run --release -p larql-vindex --example build_interleaved -- <vindex>     # Pack gate|up|down
cargo run --release -p larql-vindex --example build_down_features -- <vindex>   # Feature-major transpose
cargo run --release -p larql-vindex --example build_up_features -- <vindex>     # f16 → f32 decode
cargo run --release -p larql-vindex --example build_gate_q4 -- <vindex>         # Q4 gate vectors
cargo run --release -p larql-vindex --example build_lm_head_q4 -- <vindex>      # Q4 logits projection
```

### Quality gates

`larql-vindex` has a crate-specific Makefile block rather than relying
only on the workspace-wide `make ci`. The local gate is:

- format: `cargo fmt -p larql-vindex -- --check`
- lint: `cargo clippy -p larql-vindex --all-targets -- -D warnings`
- tests: `cargo test -p larql-vindex`
- examples: `cargo check -p larql-vindex --examples`
- benches: `cargo test -p larql-vindex --benches`
- coverage: `cargo llvm-cov --package larql-vindex` plus
  `scripts/check_coverage_policy.py`

The coverage policy lives in `coverage-policy.json`. The aggregate
line-coverage floor is currently 71% from the 2026-05-08 local
baseline of 71.56%; the 2026-05-10 round-6 push lifted measured
aggregate to **88.90% lines** (24,886 lines instrumented). Source
files default to **90% line coverage**; files below that have
explicit debt baselines that should only ratchet upward. **85 of
125 files at the 90% default** (was 41 on 2026-05-08, +44 across
rounds 5-6). Remaining 40 debt baselines cluster in the
integration-driven write/load paths
(`format/weights/load/{f32,q4k}.rs`, `format/weights/write_q4k/{moe_layers,norms}.rs`,
`format/load.rs`), the Q4_K codec dispatch family
(`index/compute/q4k_dispatch.rs`, `index/storage/ffn_store/{interleaved_q4k,q4k_cache}.rs`),
and a few HF HTTP happy-path corners that still need full mockito
infrastructure (`format/huggingface/{download,publish/mod}.rs`).
See `CHANGELOG.md` for round-by-round per-file deltas.

GitHub Actions runs the same model-agnostic surface on Linux, Windows,
and macOS. The examples step is compile-only because several tools
need an external vindex path; CI must stay synthetic and portable.

### Bench measurements (typical machine, synthetic Gemma-like fixture)

| Bench | Operation | Time |
|---|---|---|
| `extract_throughput` | streaming extract, f32 | ~49 ms |
| `extract_throughput` | streaming extract, **Q4K** | ~33 ms (1.5× faster; output is ~3× smaller so disk I/O dominates) |
| `extract_throughput` | streaming extract, **Q4K + resume after gate** | ~28 ms (gate-phase auto-skip; ~15% saved on single-layer fixture, scales with layer count) |
| `q4k_vs_f32` | f32 per-layer Q retrieval (mmap → Vec<f32>) | ~880 µs |
| `q4k_vs_f32` | **Q4K** per-layer Q retrieval (mmap → dequant → Vec<f32>) | ~3.3 ms (3.7× slower per-layer to save 6.26× on disk) |

Test coverage (857 lib tests as of 2026-05-10 round-6, **88.90% aggregate lines**):
- Construction, dimensions, layer counts, feature counts
- Gate KNN: brute-force, f32, Q4 via compute backend, top-K ordering
- Gate walk: BLAS gemv path matches brute-force KNN
- Walk: multi-layer tracing, metadata annotation
- LM head KNN: top-K token lookup via matmul_transb
- HNSW: enable/disable, integration with VectorIndex, valid results
- Q4 gate: load round-trip, data slice correctness, Q4 vs f32 top-1 match
- Mutation: set gate vectors, metadata, patch overlay
- Patching: apply, revert, bake down
- Binary serialization: checksums, dtype, config
- MoE: expert-scoped queries, multiple experts per layer
- Router weights: dense-model absence, incomplete files, top-k routing
- Down metadata: binary read/write, mmap lookup, malformed header rejection
- Layer weights: per-layer file headers, offsets, dense/MoE quant helpers
- Gate-score batch paths: heap, f32 mmap, f16 mmap decode cache, backend GEMV
- Streaming extraction: safetensors mmap, one layer at a time
- Adaptive residency: pin/evict, budget enforcement, auto_pin, pin_range, adaptive dispatch

## Benchmarks

Criterion benches live in `benches/`. Run with `cargo bench -p
larql-vindex` (full sample) or `-- --quick` (5-iter sweep). HTML
reports go to `target/criterion/`.

### Core operations (`benches/vindex_ops.rs`, M3 Max, synthetic dims)

| Operation | Time |
|---|---|
| `gate_knn_per_layer / 1024f×256h` | **22.7 µs** |
| `gate_knn_per_layer / 4096f×512h` | 365 µs |
| `gate_knn_per_layer / 10240f×2560h` (Gemma production) | **2.64 ms** |
| `walk_all_layers / 8L×1024f×256h` | 216 µs |
| `walk_all_layers / 14L×4096f×512h` | 2.19 ms |
| `walk_all_layers / 8L×10240f×2560h` (8L Gemma band) | 21.2 ms |
| `gate_knn_batch / seq1_10240f×2560h` (decode) | 2.63 ms |
| `gate_knn_batch / seq256_10240f×2560h` (prefill) | **8.44 ms** (-24 % via parallel per-position top-K) |
| `hnsw_warmup / dense-8L-10240×2560 / serial` | 395 ms |
| `hnsw_warmup / dense-8L-10240×2560 / parallel` | **109 ms** (3.6× via `warmup_hnsw_all_layers`) |
| `q4k_down / cache+transpose / K=100` (Gemma 4B Q4_K) | 77.6 ms |
| `q4k_down / feature_major / K=100` (Gemma 4B Q4_K) | **31.8 µs** (2440× via `down_features_q4k.bin`, opt-in at extract) |
| `feature_meta_lookup` (per call) | ~245 ns |
| `mutate / set_meta_plus_gate` | 301 ns |
| `save_load / save_gate_vectors` | 2.01 ms |
| `save_load / save_down_meta` | 462 µs |
| `save_load / load_vindex` | 261 µs |
| `moe_scaling / 8x_experts` (vs 1x baseline) | 17.6× for 8× features (sub-linear) |

### Production dimensions (M3 Max, synthetic data)

| Model | Features | Hidden | f32 BLAS | Q4 CPU | Q4 Metal | Speedup | Walk 14L |
|---|---|---|---|---|---|---|---|
| Gemma 3 4B | 10,240 | 2,560 | 2.7ms | 0.96ms | **0.50ms** | 5x | 7.0ms |
| Llama 3 8B | 14,336 | 4,096 | 15.7ms | 2.1ms | **0.95ms** | 17x | 15.2ms |
| Llama 3 70B | 28,672 | 8,192 | 98.3ms | 8.2ms | **1.31ms** | **75x** | 63.1ms |

Vindex provides Q4 gate data. Compute crate scores it. Same interface, any backend.

### HNSW vs brute-force (dim=2560)

| Features | Brute | HNSW | Winner |
|---|---|---|---|
| 1,024 | 0.18ms | 0.14ms | HNSW |
| 4,096 | 2.3ms | 1.9ms | HNSW |
| 10,240 | 2.6ms | 1.7ms | HNSW |
| 28,672 | 18.8ms | 15.2ms | HNSW |

### Memory (mmap, 34L × 4096 × 2560)

| Metric | Value |
|---|---|
| Cold KNN (first access) | 0.39ms |
| Warm KNN (paged) | 0.37ms |
| Page fault overhead | 0.02ms |
| Zero-copy mmap | true (0 bytes heap) |

### Adaptive residency (simulated 70B, M3 Max Metal)

```
Budget    Pinned   KNN/layer   Walk 48L    tok/s
stream     0/80     0.28ms      13.4ms      75      ← 0 MB pinned
200 MB    14/80     0.28ms      13.4ms      75
500 MB    35/80     0.28ms      13.3ms      75
all       80/80     0.29ms      13.8ms      72      ← all pinned

llama.cpp 70B:
40GB VRAM  all                              8-12    ← needs ALL weights
24GB VRAM  partial                          2-3     ← PCIe cliff
CPU only                                    1-2
```

On unified memory (Apple Silicon), mmap is effectively pinned — the gradient
is flat because there's no PCIe bottleneck. On discrete GPU systems,
pinned layers skip PCIe transfers and the gradient steepens.

## Design Principles

1. **Readonly base** — binary files on disk are never modified after extraction
2. **Patch overlay** — all mutations via in-memory PatchedVindex
3. **Zero-copy mmap** — gate vectors are sliced from the file, not loaded to heap
4. **One file per matrix type** — gate, attn, up, down stored separately
5. **Streaming extraction** — processes one layer at a time (~2 GB peak for 120B models)
6. **All compute through larql-compute** — BLAS dispatch, no raw ndarray .dot() calls
7. **Adaptive residency** — pin hot layers in memory budget, stream cold ones from mmap
8. **Format-agnostic storage** — vindex stores raw quantized bytes, compute dequants at inference

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmark data, scaling projections, compute integration |
| [ROADMAP.md](ROADMAP.md) | Active P0/P1/P2 work + parked / won't-fix |
| [CHANGELOG.md](CHANGELOG.md) | Reverse-chronological history of shipped work |
| [docs/vindex-format.md](docs/vindex-format.md) | File format specification, directory layout, manifest schemas |
| [docs/compute-integration.md](docs/compute-integration.md) | How vindex stores data and compute consumes it |
| [docs/adr/001](docs/adr/001-weights-as-database.md) | Transformer weights as queryable database |
| [docs/adr/002](docs/adr/002-quantization-strategy.md) | Ollama-compatible Q4_K/Q6_K quantization |
| [docs/adr/003](docs/adr/003-mmap-zero-copy.md) | Mmap zero-copy architecture |
| [docs/adr/004](docs/adr/004-three-storage-tiers.md) | Three-tier weight storage (f32, Q8, Q4_K) |
| [docs/adr/005](docs/adr/005-patch-overlay.md) | Patch overlay for editable knowledge |
| [docs/adr/006](docs/adr/006-hnsw-index.md) | HNSW graph index for sub-linear KNN |
| [docs/adr/007](docs/adr/007-interleaved-layout.md) | Interleaved weight layout (TLB optimization) |
| [docs/adr/008](docs/adr/008-quantizer-source-of-truth.md) | Single source of truth for quantizers |
| [docs/adr/009](docs/adr/009-feature-major-down.md) | Feature-major Q4_K down (W2 cache bypass) |

## Status

```
Tests:      457 passing (306 unit + 151 integration; clippy clean as of 2026-04-26)
Coverage:   61% lines / 57% functions (cargo-llvm-cov; W2 files 95–100%)
Warnings:   0 (build), 0 (clippy --all-targets)
Formats:    f32, Q8_0, Q4_K, Q6_K, Q4_0, FP4, FP8
Models:     Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, Phi, DeepSeek, Granite, StarCoder2, GPT-OSS, GPT-2
```

## License

Apache-2.0
