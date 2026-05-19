# Vindex Format Specification

**Version:** 0.4
**Date:** 2026-04-24
**Status:** Implemented (~98%); FP4/FP8 storage in progress (exp 26)
**Implementation:** `larql-vindex` crate (Rust)
**Companion specs:** [Operations](operations-spec.md), [Ecosystem](ecosystem-spec.md), [LQL](../../larql-lql/docs/spec.md)
**FP4 companion specs:** [FP4 format](fp4-format-spec.md), [FP4 precision policy](fp4-precision-policy.md), [Quantize CLI](../../larql-cli/docs/quantize-spec.md)

**Implementation coverage:** File layout, binary formats, extract levels, f16 storage, checksums, mmap loading, streaming extraction, `larql verify`, Q4_K quantisation — all implemented. **FP4/FP8 block storage** — codec layer landed (see §5.10), writer and walk-kernel dispatch in progress.

---

## 1. What is a Vindex?

A vindex (vector index) is a directory containing a neural network's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format for the operations it supports.

**Key principle:** `gate_vectors.bin` IS W_gate. `embeddings.bin` IS W_embed. They are the canonical storage, not copies. COMPILE reads `gate_vectors.bin` to reconstruct W_gate in safetensors format. No data is stored twice.

### Weights separated by function, not by file size

Existing model formats — safetensors, GGUF, ONNX — store all weights together. Sharded formats split by file size, not by purpose. Every format assumes you want all the weights or none of them.

Vindex separates weights by what you want to do with the model:

| Intent | Weights needed | Size (4B, f16) | What you skip |
|--------|---------------|-----------------|---------------|
| Browse knowledge | Gate + embeddings + down metadata | ~3 GB | Attention, up projections, norms, LM head |
| Run inference | + attention weights + norms | ~6 GB | Up projections, LM head |
| Edit and recompile | All weights | ~10 GB | Nothing |

This separation exists because these operations touch fundamentally different weight matrices. Querying what a model knows about France uses gate vectors and embeddings — a dot product against pre-extracted rows. It never touches attention weights. Running inference additionally needs attention for token routing. Recompiling needs everything to reconstruct the full safetensors.

---

## 2. Architecture Support

The vindex format is model-agnostic. It stores weights by function (gate, down, embed, attention) not by architecture name. Any transformer model with a gated FFN (gate + up + down projections) and standard attention (Q, K, V, O) can be extracted into a vindex.

### Supported model families

| Family | Models | FFN Type | Notes |
|--------|--------|----------|-------|
| Gemma | Gemma 2/3/4 (2B-31B) | Gated (gate + up + down) | GeGLU activation, GQA attention; Gemma 4 adds per-layer head_dim, QK-norm, partial RoPE, cross-layer KV sharing (E2B), Per-Layer Embeddings (E2B), double-wide MLP (E2B) |
| Llama | Llama 2/3 (7B-405B) | Gated (gate + up + down) | SiLU activation, GQA attention |
| Mistral | Mistral 7B | Gated (gate + up + down) | Sliding window attention |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts × gate + up + down) | Sparse MoE, top-2 routing |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (gate + up + down) | SiLU activation |
| Phi | Phi 2/3 (2.7B-14B) | Gated (gate + up + down) | Partial attention |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed experts) | Fine-grained MoE, shared expert |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (W_in + W_out) | Soft gating via GELU |

### MoE layout

Mixture-of-Experts models have multiple FFN "experts" per layer. In the vindex, each expert's features are separate entries in the same `gate_vectors.bin`. A dense model has `intermediate_size` features per layer. An MoE model with N experts has `N × intermediate_size` features per layer.

```
Dense (Gemma 4B):    10,240 features per layer → 348,160 total
MoE (Mixtral 8x7B):  8 × 14,336 = 114,688 features per layer → 3,670,016 total
```

Gate KNN naturally selects features across all experts — no router needed for browse operations.

### Dense FFN support (GPT-2)

Dense FFN models have no explicit gate matrix — the FFN is `W_out @ GELU(W_in @ x)`. `W_in` rows serve the same functional role as gate vectors: each row defines a direction in residual space that activates a feature. Extraction uses `W_in` rows as `gate_vectors.bin` and `W_out` columns as down projections. All query operations work unchanged.

### Architecture variations handled

- **GQA (Grouped Query Attention):** Fewer KV heads than Q heads. Stored as-is in attention weights.
- **Sliding window attention:** Window size stored in `model_config`. Browse operations unaffected.
- **Tied embeddings:** LM head shares embedding matrix. `lm_head.bin` omitted, `embeddings.bin` used for both.
- **RoPE variants:** Base frequency and scaling stored in `model_config`. Used by INFER only.
- **MoE shared expert (DeepSeek):** Stored as Expert 0 with a `shared: true` flag.
- **Fine-grained MoE (DeepSeek V3):** 256 experts with top-8 routing. Same structure, more features per layer.
- **Per-Layer Embeddings (Gemma 4 E2B):** Carried in `ple_weights.bin` at f16. `VectorIndex::num_features(layer)` is the authoritative per-layer FFN width for models with `use_double_wide_mlp=True`.
- **Logit softcap (Gemma 2/3/4):** `final_logit_softcapping` is stored in `index.json::model_config` and reapplied before softmax at inference time. Dropping it silently mis-peaks the top-1 token.
- **Cross-layer KV sharing (Gemma 4 E2B):** `num_kv_shared_layers` in `model_config` drives the forward-pass cache. Storage is unchanged — the source-layer Q4K weights still ship; the reader skips K/V compute at shared layers.
- **Granite-family scaling multipliers (Granite 3.x, 4.0, 4.1):** `attention_multiplier`, `residual_multiplier`, `logits_scaling`, and `norm_eps` live in `index.json::model_config`; `embed_scale` (== `embedding_multiplier`) is captured at the top level of `index.json` (legacy carve-out). The loader's `build_arch_json` re-emits all five through the safetensors-style field names so `detect_from_json` populates `ModelConfig`, and the forward path reads them through `arch.embed_scale()` / `arch.attention_multiplier()` / `arch.residual_multiplier()` / `arch.logits_scaling()` / `arch.norm_eps()`. Without these the reconstructed arch defaults each to 1.0 (or the arch-class `norm_eps` default), the Metal/CPU Q4K forward runs without Granite scaling, and the model emits gibberish.

---

## 3. File Layout

```
model.vindex/
│
│  # ═══ Query Index (browse-only core) ═══
│  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
│
├── gate_vectors.bin          # W_gate rows per layer (KNN index)
├── embeddings.bin            # W_embed matrix (token lookup)
├── down_meta.bin             # Per-feature output metadata (binary)
│
│  # ═══ Inference Weights (for INFER) ═══
│
├── attn_weights.bin          # Q, K, V, O projection matrices per layer
├── norms.bin                 # LayerNorm parameters per layer
│
│  # ═══ Compile Weights (for COMPILE) ═══
│
├── up_weights.bin            # W_up per layer
├── down_weights.bin          # W_down per layer (full vectors)
├── lm_head.bin               # Output projection (omitted if tied to embeddings)
│
│  # ═══ Quantised Weights (when quant = Q4k) ═══
│  # Written instead of attn_weights.bin / up_weights.bin / down_weights.bin.
│
├── attn_weights_q4k.bin      # Q/K/O = Q4_K, V = Q6_K per layer
├── attn_weights_q4k_manifest.json
├── interleaved_q4k.bin       # FFN gate/up = Q4_K, down = Q6_K (or Q4_K with --down-q4k) per layer
├── interleaved_q4k_manifest.json
│
│  # ═══ FP4/FP8 Storage (when index.json.fp4 is set — exp 26) ═══
│  # Per-projection precision controlled by the `fp4.projections` manifest.
│  # Written alongside or instead of the legacy gate/up/down files depending
│  # on the per-projection `precision` tag. Loaders dispatch on the tag, never
│  # sniff filenames.
│
├── gate_vectors_fp4.bin      # Gate at FP4 E2M1, 256-elem blocks (137 B/block)
├── up_features_fp4.bin       # Up at FP4 E2M1, same layout
├── down_features_fp8.bin     # Down at FP8 E4M3, 256-elem blocks (257 B/block)
├── fp4_compliance.json       # Extract-time Q1 compliance scan + per-projection actions
│
│  # ═══ Gemma 4 E2B Per-Layer Embeddings ═══
│  # Emitted only when has_per_layer_embeddings() == true.
│  # f16 deliberately — Q4_K super-block calibration destroys
│  # embedding-style tensors, and PLE contributions are additive
│  # into every layer's residual so per-cell noise compounds.
│
├── ple_weights.bin           # per_layer_model_projection + embed_tokens_per_layer
│                             # + per-layer input_gate & projection (all f16)
│
│  # ═══ Metadata & Labels ═══
│
├── index.json                # VindexConfig: layers, sizes, checksums, provenance
├── tokenizer.json            # HuggingFace tokenizer
├── relation_clusters.json    # Cluster centres, labels, counts
├── feature_labels.json       # Probe-confirmed labels
└── weight_manifest.json      # Weight file → offset mapping
```

Gate vectors are NOT duplicated — `gate_vectors.bin` IS the W_gate weight matrix. COMPILE reads it directly to reconstruct the safetensors gate tensor.

---

## 4. Extract Levels

A vindex can be built at three levels, each adding more weight components:

| Level | LQL Syntax | Components | Size (f16, 4B) | Enables |
|-------|-----------|------------|-----------------|---------|
| Browse | `EXTRACT MODEL ... INTO ...` | gate + embed + down_meta | ~3 GB | WALK, DESCRIBE, SELECT |
| Inference | `... WITH INFERENCE` | + attn_weights + norms | ~6 GB | + INFER, EXPLAIN INFER |
| All | `... WITH ALL` | + up, down, lm_head | ~10 GB | + COMPILE |

```rust
pub enum ExtractLevel {
    Browse,      // Default: gate + embed + down_meta
    Inference,   // + attention weights + norms
    All,         // + all remaining weights for COMPILE
}
```

The `index.json` stores the extract level. Operations that require a higher level than available return `VindexError::InsufficientExtractLevel`.

---

## 5. Binary Formats

### 5.1 gate_vectors.bin

Raw floats (f32 or f16 per `dtype` in config), contiguous, no headers. Layer-by-layer concatenation.

**Layout:**
```
[Layer 0: num_features × hidden_size × sizeof(float)]
[Layer 1: num_features × hidden_size × sizeof(float)]
...
[Layer N: num_features × hidden_size × sizeof(float)]
```

**Per-layer shape:** `(intermediate_size, hidden_size)` — one row per FFN feature.

**Byte order:** Little-endian.

**Index:** `VindexLayerInfo` in `index.json` stores byte offset and length for each layer, enabling random access without reading the entire file.

**MoE layout (superseded — see §5.12):** Experts are contiguous within each layer. The `layers/layer_{L}.weights` per-layer format described in §5.12 replaces this for both dense and MoE models.
```
[Layer 0, Expert 0: intermediate_size × hidden_size]
[Layer 0, Expert 1: intermediate_size × hidden_size]
...
[Layer 0, Expert N: intermediate_size × hidden_size]
[Layer 1, Expert 0: ...]
```

### 5.2 embeddings.bin

Raw floats (f32 or f16), no headers. Single contiguous matrix.

**Shape:** `(vocab_size, hidden_size)` in row-major order.

**Usage:** Token embedding lookup. Multiply by `embed_scale` (from config) to match gate vector magnitudes.

### 5.3 down_meta.bin (binary, primary)

Compact binary format. Preferred over JSONL — ~80x smaller.

**Header:**
```
magic:       [u8; 4]   "DMET"
version:     u32
num_layers:  u32
top_k:       u32       (entries per feature)
```

**Per feature (fixed size):**
```
top_token_id:  u32  (4 bytes)
c_score:       f32  (4 bytes)
num_top_k:     u8   (1 byte)
top_k entries:
  token_id:    u32  (4 bytes)
  score:       f32  (4 bytes)
```

Token strings resolved at read time via tokenizer. No string storage needed.

Only `down_meta.bin` is written during extraction. Loading falls back to `down_meta.jsonl` if binary is absent (v1 compatibility).

### 5.4 down_meta.jsonl (v1 legacy)

NDJSON format from v1 vindexes. No longer written during extraction. Supported for loading only (backward compatibility). ~5.5x larger than binary.

### 5.5 attn_weights.bin

Sequential float tensors (f32 or f16), no headers. Per-layer Q, K, V, O projection matrices. Layout described by `weight_manifest.json`.

Only present at Inference or All extract level.

### 5.6 up_weights.bin / down_weights.bin

Sequential float tensors (f32 or f16). W_up and W_down per layer respectively.

Only present at All extract level. Gate vectors are NOT stored here — they are in `gate_vectors.bin`.

### 5.7 norms.bin

LayerNorm parameters per layer (input_layernorm, post_attention_layernorm) plus final norm. Small file (~1 MB).

Only present at Inference or All extract level.

### 5.8 lm_head.bin

Output projection matrix. Omitted when `tie_word_embeddings` is true in `model_config` (embeddings.bin used instead).

Only present at All extract level.

### 5.9 weight_manifest.json

JSON array mapping tensor keys to byte offsets in the weight files.

```json
[
  {
    "key": "layers.0.self_attn.q_proj.weight",
    "kind": "tensor",
    "shape": [2048, 2560],
    "offset": 0,
    "length": 20971520
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Tensor key (architecture-specific naming) |
| `kind` | string | `"tensor"` (2D f32/f16), `"vector"` (1D f32/f16), `"tensor_q4k"` (2D Q4_K 144-byte blocks, used by `lm_head_q4.bin`), or `"tensor_f16"` (2D IEEE half, used by `ple_weights.bin`) |
| `shape` | [usize] | Dimensions |
| `offset` | u64 | Byte offset into the relevant weight file |
| `length` | u64 | Byte length |

`tensor_q4k` and `tensor_f16` entries are decoded to f32 at load time
and surface in `ModelWeights.tensors`, so the downstream forward code
can read them like any other dense matrix.

### 5.10 FP4/FP8 block storage (exp 26)

When `index.json.fp4` is present, the vindex stores one or more FFN
projections in a block-quantised format instead of (or alongside) the
f16/f32 gate_vectors.bin, up_features.bin, down_features.bin files. Per-
projection precision is controlled by `fp4.projections.{gate|up|down}.
precision` — legal values are `fp4`, `fp8`, `f16`, `f32`.

**Block geometry (v1).** All blocks cover 256 elements, chosen as the
largest block size that divides every model family LARQL currently ships
(hidden ∈ {512, 1536, 2560, 5376}). Each 256-element block holds 8
sub-blocks of 32 elements each, matching the OCP MXFP4 sub-block size.

**FP4 block layout — 137 bytes per 256 elements:**

| Offset  | Size  | Contents                                    |
| ------- | ----- | ------------------------------------------- |
| 0–127   | 128 B | 256 FP4 E2M1 values, nibble-packed (2/byte) |
| 128–135 | 8 B   | 8 FP8 E4M3 sub-block scales                 |
| 136     | 1 B   | 1 FP8 E4M3 block scale                      |

Dequantisation: `x = fp4_value × sub_block_scale × block_scale / 6`. Nibble
packing: lower nibble = even-indexed element of each pair.

**FP8 block layout — 257 bytes per 256 elements:**

| Offset | Size  | Contents                      |
| ------ | ----- | ----------------------------- |
| 0–255  | 256 B | 256 FP8 E4M3 values           |
| 256    | 1 B   | 1 FP8 E4M3 block scale        |

Dequantisation: `x = fp8_value × block_scale`. No sub-block scales — E4M3's
dynamic range (±448) absorbs typical FFN weight magnitude spread directly.

**Per-file byte layout.** Same layer/feature concatenation convention as
legacy projection files. Per-layer byte offsets come from the existing
`layers[i].num_features` field — no new layer-offset metadata needed;
the writer knows the block count per feature from `hidden / 256`.

**Mmap-friendliness.** Each feature vector's blocks are contiguous — one
cacheline-friendly prefetch walk per feature, same access pattern as the
legacy f16 layout.

**Compression vs F16 (4B, 3 projections):**

| Configuration                          | Per-feature | Compression |
| -------------------------------------- | -----------:| -----------:|
| F16 baseline (3 × 2560 × 2 bytes)      | 15,360 B    | 1.00×       |
| Uniform FP4 (all 3 projections)        | 4,110 B     | **3.74×**   |
| FP4 gate/up + FP8 down (default)       | 5,310 B     | **2.89×**   |
| FP4 gate/up + F16 down (conservative)  | 7,860 B     | 1.95×       |

**Policy default.** Option B (`{gate: fp4, up: fp4, down: fp8}`). The
`down` projection carries FFN's heaviest-tailed per-feature magnitude
distribution (exp 26 cross-model data); FP8 E4M3 absorbs that tail
without any distributional assumption, at an ~8% FFN-vindex cost vs
uniform FP4. See [precision policy](fp4-precision-policy.md) §5.

**Full byte-layout specification** including nibble-order, E2M1 table,
and E4M3 encoding detail is in the experiment format spec:
[fp4-format-spec.md](fp4-format-spec.md).

### 5.11 fp4_compliance.json

Extract-time sidecar emitted alongside any vindex written with FP4
storage. Contains the full output of the Q1 compliance scan plus
per-projection actions taken by the extractor:

```json
{
  "extracted_at": "2026-04-24T...",
  "extractor_version": "...",
  "scanner_version": "...",
  "block_elements_scanned": 256,
  "compliance_gate_threshold_ratio": 16.0,
  "compliance_gate_min_fraction": 0.99,
  "per_projection": [
    {"projection": "gate", "compliance_at_R16": 0.99999, "action": "wrote_fp4"},
    {"projection": "up",   "compliance_at_R16": 0.99999, "action": "wrote_fp4"},
    {"projection": "down", "compliance_at_R16": 0.99950, "action": "wrote_fp8_per_policy_default"}
  ],
  "full_scan": { /* fp4_q1_scan.rs JSON */ }
}
```

Advisory for humans; the authoritative precision per projection is always
`index.json.fp4.projections.{gate|up|down}.precision`. The sidecar records
*why* each projection landed at the precision it did (met the compliance
gate, was downgraded after failing it, or was set by policy regardless).

---

### 5.12 Per-layer FFN weight storage (`layers/`)

**Status:** Shipped 2026-04-26 for MoE — `experts_packed.bin` (BF16 monolith) is no longer written. Dense layers still use `interleaved_q4k.bin` for now; per-layer dense is a future migration. Activated when `index.json` carries `"ffn_layout": "per_layer"`.

**Reading code (current):** `format/weights/load/q4k.rs::load_model_weights_q4k_shard` mmaps each `layers/layer_{L}.weights`, parses the LYRW header + offset table, and exposes per-expert byte ranges via `ModelWeights::get_layer_entry_bytes(layer, entry)`. The CPU MoE path (`larql-compute::cpu::ops::moe`) and the remote-expert HTTP handler (`larql-server::routes::expert::run_expert`) both consume per-expert slices directly — no monolith arithmetic.

**Migrating an old MoE vindex:** run `cargo run --release -p larql-cli --example convert_moe_to_per_layer -- <vindex>` to write the `layers/*.weights` files and set `"ffn_layout": "per_layer"`, then strip the `packed_bf16` rows referencing `experts_packed.bin` from `weight_manifest.json` and delete the file. Validated end-to-end on Gemma 4 26B A4B: `forward_moe` warm latency 4.86 → 1.91 ms (2.5×), 30-layer sweep 866 → 56 ms (15×), RSS 16.6 → 9.7 GB, disk 58 → 16 GB.

**Design principles.**

1. **Structure is orthogonal to quantization.** The file format is `per_layer` — one file per transformer layer. The *quantization* is declared in the file header. All entries within a file use the same format; there is no mixing (no "Q4_K gate/up + Q6_K down" within one file). Re-quantizing a layer is replacing one file.

2. **Unified for dense and MoE.** A dense layer is `num_entries = 1`. A MoE layer is `num_entries = num_experts`. The binary format and GPU dispatch path are identical.

3. **Native OS addressability.** Each file is independently mmap'd. A server shard with `--layers 0-14` maps only its 15 files; a shard with `--experts 0-31` reads only those entries' byte ranges within each file. No offset arithmetic into a shared flat blob.

**Why the old formats fail.**

*`interleaved_q4k.bin` (dense):* One flat file for all 34 layers. Server `--layers` sharding works via byte-offset filtering but the OS faults in the full virtual range. Layer-level replacement or re-quantization requires rewriting the whole file.

*`experts_packed.bin` (MoE BF16):* historical 43 GB monolithic BF16 blob. CPU BF16→f32 dequant at ~2.9 GB/token on Gemma 4 26B A4B; near-zero LRU cache hit rate. 30 GPU commit/wait syncs per decode step. No per-expert addressability. **Removed from new MoE vindexes 2026-04-26.**

Measured on Gemma 4 26B A4B: 4.1 tok/s with BF16 blob vs 56.8 tok/s GPU-only baseline (decode dominated by CPU MoE). After per-layer migration the CPU MoE remote-expert path runs at 1.91 ms / call warm.

**File layout.**

```
layers/
  layer_00.weights   ← dense: 1 entry. MoE: 128 entries.
  layer_01.weights
  ...
  layer_{L-1}.weights
```

Each file is self-describing:

```
[header]
  magic:         u32   0x4C595257 ("LYRW")
  format_version: u32  = 1
  quant_format:  u32   0=f32, 1=f16, 2=bf16, 3=q4_0, 4=q4_k, 5=q6_k, 6=q8_0, 7=fp4, ...
  num_entries:   u32   1 (dense) or num_experts (MoE)
  intermediate:  u32   intermediate_size or moe_intermediate_size
  hidden:        u32   hidden_size

[offset table]   num_entries × 4 × u64:
                   gate_up_offset, gate_up_bytes,
                   down_offset,    down_bytes
                 (all offsets from start of file)

[entry 0 gate+up]   quant_format blocks, shape [2*inter, hidden]
[entry 0 down]      quant_format blocks, shape [hidden, inter]
[entry 1 gate+up]
[entry 1 down]
...
```

The `quant_format` field is the **single source of truth** for the encoding. Adding a new quantization (FP8, FP4, Q3_K, …) is a new enum value; the file structure is unchanged.

**Access pattern (decode).**

```
Startup:   mmap layers/layer_{L}.weights for owned layers
           read header + offset table into memory (~4 KB per file at 128 experts)

Dense (num_entries=1):
           read entry 0 gate+up + down slices → GPU dispatch via existing FFN shaders

MoE (num_entries=128):
           router projection → top-K indices {e0, ..., eK-1}
           copy gate_up slices for eK into contiguous staging buffer
           GPU dispatch: quant_matvec, N = K × inter, K = hidden
           copy down slices for eK into staging buffer
           GPU dispatch: quant_matvec, N = K × hidden, K = inter
           CPU weighted sum (K scalars × hidden — trivial)
```

One GPU command buffer per decode step for both dense and MoE paths.

**Server-side sharding.**

`--layers START-END`: map only those layer files — other layers never touch RAM.  
`--experts START-END` (MoE): mmap all layer files in range, read only the assigned entry byte ranges. Out-of-range entry requests return HTTP 404 before any byte is read. See §13.4.

**File sizes (Gemma 4 26B A4B, Q4_K).**

| Old format | Size | New format | Size |
|---|---|---|---|
| `experts_packed.bin` (BF16) | 43 GB | `layers/*.weights` (Q4_K) | ~24 GB |
| `interleaved_q4k.bin` (dense) | — | `layers/*.weights` (Q4_K) | same bytes, per-layer |

---

## 6. index.json (VindexConfig)

The central configuration file. Version 2 is the current format.

```json
{
  "version": 2,
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "dtype": "f16",

  "source": {
    "huggingface_repo": "google/gemma-3-4b-it",
    "huggingface_revision": "a1b2c3d4e5f6",
    "safetensors_sha256": "e3b0c44298fc1c149afb...",
    "extracted_at": "2026-04-01T14:22:00Z",
    "larql_version": "0.1.0"
  },

  "checksums": {
    "gate_vectors.bin": "a1b2c3d4...",
    "embeddings.bin": "e5f6a7b8...",
    "down_meta.bin": "c9d0e1f2...",
    "attn_weights.bin": "13141516..."
  },

  "num_layers": 34,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "vocab_size": 262144,
  "embed_scale": 50.596,
  "extract_level": "inference",
  "has_model_weights": true,          // deprecated — use extract_level instead
  "down_top_k": 10,

  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },

  "layers": [
    {"layer": 0, "num_features": 10240, "offset": 0, "length": 52428800},
    {"layer": 1, "num_features": 10240, "offset": 52428800, "length": 52428800}
  ],

  "model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 10000.0,
    "rope_scaling_factor": 1.0,
    "sliding_window": 1024,
    "attention_type": "gqa",
    "activation": "geglu",
    "tie_word_embeddings": true,
    // Granite-family scaling multipliers (see §2 "Granite-family
    // scaling multipliers"). All four optional; when absent the
    // reconstructed arch's trait getters return 1.0 (multipliers) or
    // the arch-class default (norm_eps). Required for any
    // GraniteForCausalLM variant — without them the model emits
    // gibberish from `larql run` even though the safetensors path is
    // correct.
    "attention_multiplier": 0.015625,
    "residual_multiplier": 0.22,
    "logits_scaling": 10.0,
    "norm_eps": 0.00001
  },

  // FFN weight layout. "per_layer" = layers/layer_{L}.weights, one file per layer,
  // format declared in file header (see §5.12). Works for both dense and MoE.
  // Absent = legacy flat-file layout (interleaved_q4k.bin / experts_packed.bin).
  "ffn_layout": "per_layer",

  "fp4": {
    "fp4_format_version": 1,
    "block_elements": 256,
    "sub_block_elements": 32,
    "sub_block_scale_dtype": "fp8_e4m3",
    "block_scale_dtype": "fp8_e4m3",
    "value_encoding": "fp4_e2m1_mxfp4_nibble_order",
    "projections": {
      "gate": { "precision": "fp4", "file": "gate_vectors_fp4.bin" },
      "up":   { "precision": "fp4", "file": "up_features_fp4.bin" },
      "down": { "precision": "fp8", "file": "down_features_fp8.bin" }
    },
    "compliance_gate": {
      "threshold_ratio": 16.0,
      "min_compliant_fraction": 0.99,
      "fallback_precision": "fp8"
    },
    "compliance_report": "fp4_compliance.json"
  }
}
```

The `fp4` field is optional. Absent or null → the vindex uses legacy
f16/f32 projection files as before. Present → per-projection precision
is authoritative from this field; loaders dispatch on the tag and never
sniff filenames. Adding this field does **not** bump the parent
`version` — FP4 is additive opt-in, not a breaking change.

### Key fields

**`version`** — Config format version. Current: 2.

**`dtype`** — Storage precision for all binary files. `"f32"` or `"f16"`. Cast to f32 at load time.

**`source`** — Provenance tracking. Records exactly which model checkpoint was extracted, when, and by which version of LARQL.

**`checksums`** — SHA256 hash of each binary file. Enables integrity verification after download.

**`layer_bands`** — Model-specific boundaries for DESCRIBE layer grouping. Per-family values auto-detected during EXTRACT:

| Family | Syntax | Knowledge | Output |
|--------|--------|-----------|--------|
| Gemma 3 (34L) | 0-13 | 14-27 | 28-33 |
| Llama 3 (32L) | 0-7 | 8-24 | 25-31 |
| Qwen 2.5 (28L) | 0-7 | 8-21 | 22-27 |
| Mixtral (32L) | 0-7 | 8-24 | 25-31 |
| GPT-2 (12L) | 0-3 | 4-9 | 10-11 |

**`layers`** — Per-layer byte offset and length into `gate_vectors.bin`. For MoE models, includes `num_experts` and `num_features_per_expert`.

**MoE config** (when applicable):
```json
"moe_config": {
  "num_experts": 8,
  "top_k_experts": 2,
  "has_shared_expert": false,
  "router_type": "top_k_softmax"
}
```

---

## 7. Label Files

### relation_clusters.json

Discovered relation clusters from offset-direction clustering:
```json
{
  "k": 512,
  "labels": ["capital", "language", "morphological"],
  "counts": [142, 89, 1203],
  "top_tokens": [["Paris", "Berlin", "Tokyo"], ["French", "English", "German"]]
}
```

### feature_labels.json

Probe-confirmed labels (from the larql-knowledge pipeline):
```json
{
  "26:9515": "capital",
  "24:4532": "language",
  "25:3877": "continent"
}
```

Key format: `"layer:feature"`. These override cluster labels at query time.

---

## 8. Storage Precision

Two surfaces control storage precision:

**`dtype`** (top-level): controls legacy gate_vectors.bin, up_features.bin,
down_features.bin, attn_weights.bin, embeddings.bin, lm_head.bin. `"f32"`
or `"f16"`. Cast to f32 at load time. Gate KNN accuracy at f16 is
effectively identical to f32 — top-K ranking is preserved.

| Dtype | Bytes/float | gate_vectors (4B) | embeddings (4B) | Total browse |
|-------|-------------|-------------------|-----------------|--------------|
| f32 | 4 | 3.32 GB | 2.50 GB | ~6 GB |
| f16 | 2 | 1.66 GB | 1.25 GB | ~3 GB |

**`fp4.projections.{gate|up|down}.precision`** (optional, per-projection):
overrides `dtype` for the FFN projections when the `fp4` field is set.
Legal values: `fp4`, `fp8`, `f16`, `f32`. The FP4 and FP8 formats are
block-quantised (see §5.10); the f16 and f32 values map to the legacy
files and the legacy codepath.

```rust
// Legacy global storage precision.
pub enum StorageDtype { F32, F16 }

// Per-projection precision tag (exp 26).
pub enum Precision { Fp4, Fp8, F16, F32 }

pub struct ProjectionFormat {
    pub precision: Precision,
    pub file: String,   // e.g. "gate_vectors_fp4.bin"
}
```

FP4/FP8 data is dequantised to f32 lazily at walk time — the block codec
(`larql-models::quant::{fp4,fp8,fp4_block}`) handles this per-feature.

---

## 9. Size Reference (Gemma 3 4B)

### f32

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 3.32 GB | 34 × 10,240 × 2,560 × 4 bytes |
| embeddings.bin | 2.50 GB | 262,144 × 2,560 × 4 bytes |
| down_meta.bin | ~2 MB | Binary token IDs + scores |
| attn_weights.bin | ~6 GB | Q, K, V, O per layer |
| up_weights.bin | ~3.4 GB | W_up per layer |
| down_weights.bin | ~3.4 GB | W_down per layer |
| norms.bin | ~1 MB | LayerNorm parameters |
| lm_head.bin | ~2.6 GB | Output projection |
| **Browse total** | **~6 GB** | |
| **Inference total** | **~12 GB** | |
| **All total** | **~18 GB** | |

### f16

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 1.66 GB | Half precision |
| embeddings.bin | 1.25 GB | Half precision |
| down_meta.bin | ~2 MB | Same (integer token IDs) |
| attn_weights.bin | ~3 GB | Half precision |
| up_weights.bin | ~1.7 GB | Half precision |
| down_weights.bin | ~1.7 GB | Half precision |
| norms.bin | ~1 MB | Small regardless |
| lm_head.bin | ~1.3 GB | Half precision |
| **Browse total** | **~3 GB** | |
| **Inference total** | **~6 GB** | |
| **All total** | **~10 GB** | |

### FP4 + FP8 (Option B default, exp 26)

Gate and up in FP4, down in FP8. Inference-level FFN storage only — rest
of the vindex (embeddings, attn, lm_head) stays at the `dtype` setting
(typically f16).

| File | Size | Description |
|------|------|-------------|
| gate_vectors_fp4.bin | ~0.48 GB | 34 × 10,240 × 1,370 B per feature |
| up_features_fp4.bin | ~0.48 GB | Same layout as gate |
| down_features_fp8.bin | ~0.89 GB | 34 × 10,240 × 2,570 B per feature |
| fp4_compliance.json | <100 KB | Extract-time Q1 scan |
| **FFN total (vs ~5.0 GB F16)** | **~1.85 GB (2.89× compression)** | |

At 31B scale (Gemma 4 31B, hidden=5376, intermediate=21504, 60 layers):

| Config | FFN storage | vs F16 FFN (41.6 GB) |
|--------|-------------|----------------------|
| F16 baseline | 41.6 GB | 1.00× |
| Uniform FP4 (Option A) | 11.1 GB | **3.74×** |
| FP4 gate/up + FP8 down (Option B, default) | 14.4 GB | **2.89×** |
| FP4 gate/up + F16 down (Option C) | 21.2 GB | 1.95× |

---

## 10. Version History

| Version | Changes |
|---------|---------|
| 1 | Original: gate + embed + down_meta JSONL + model_weights.bin |
| 2 | Added extract_level, layer_bands, model_config, source, checksums, dtype. Binary down_meta. Split weight files (attn, up, down, norms, lm_head). f16 storage. Q4_K/Q6_K quantisation (interleaved_q4k.bin + manifest). |

**FP4/FP8 storage is an additive extension, not a version bump.** Version
2 vindexes can optionally carry an `fp4` field in `index.json` with
per-projection precision and byte layout per §5.10 / §6. Readers that
don't understand the field ignore it and use the legacy f16/f32 files.
The `fp4.fp4_format_version` field is independent of the parent version
and bumps only on byte-layout changes to FP4 blocks themselves, not on
schema additions (new precision tags, new manifest fields).

**Compatibility:** v1 vindexes load with sensible defaults for missing fields:
- Missing `layer_bands` → auto-computed from layer count
- Missing `source` → empty (provenance unknown)
- Missing `checksums` → skip verification
- Missing `extract_level` → inferred from `has_model_weights`
- Missing `dtype` → assumed f32
- Missing `fp4` → legacy f16/f32 codepath (never FP4/FP8)

Legacy `model_weights.bin` is still supported for loading. The engine checks for split weight files first, falls back to `model_weights.bin` + `weight_manifest.json`.

---

## 11. Implemented Format Features

### 11.1 Memory-Mapped Loading — IMPLEMENTED

`gate_vectors.bin` and `embeddings.bin` are loaded via `mmap` (memmap2). For f32 files, gate KNN reinterprets the mmap'd bytes directly as `&[f32]` — zero heap allocation, zero copy. The OS pages data in on demand. Only queried layers consume physical RAM.

Saves use atomic write-to-temp + rename to avoid invalidating active mmaps. Multiple vindexes can share physical memory for overlapping pages.

### 11.2 Checksums and Verification — IMPLEMENTED

SHA256 hashes of all binary files, stored in `index.json` under `checksums`. Computed during EXTRACT. Verified via CLI:

```bash
larql verify gemma3-4b.vindex
# gate_vectors.bin ... OK (1.66 GB)
# embeddings.bin ... OK (1.25 GB)
# down_meta.bin ... OK (29 MB)
# All 3 files verified.
```

---

## 12. Future Format Changes

### 12.1 Quantised Browse — SUPERSEDED BY FP4 (exp 26, in progress)

The earlier int8 / int4 proposal is superseded by the FP4 block format
described in §5.10. The FP4 path is a richer version of the original
idea: per-block FP8 E4M3 block scales preserve ranking better than
integer quantisation, and the measurement-first approach (Q1 scan,
compliance floor, self-policing extractor) removes the "nearly identical
ranking" handwave that the int8/int4 proposal relied on.

Projected storage under Option B (FP4 gate/up + FP8 down) at 4B:
- FFN storage: **~1.85 GB (vs 5.0 GB F16, 2.89× compression)**
- Under uniform FP4 (Option A): 1.43 GB (3.74× compression)

### 12.2 MXFP4 Quantized Models

Models distributed with MXFP4 block quantization (e.g., GPT-OSS-120B) can be extracted to vindex format, but gate KNN produces noisy results due to 4-bit weight precision. The model works correctly at inference time because the full forward pass (SiLU gating × up projection, transformed residuals) compensates for quantization noise. Isolated gate dot products cannot.

**Note the distinction.** OCP/MXFP4 (the GPT-OSS format) uses single-level
e8m0 per-sub-block scales. The LARQL FP4 format (§5.10) reuses the same
FP4 E2M1 value encoding and nibble packing but adds a two-level scale
hierarchy (FP8 E4M3 sub-block scales + FP8 E4M3 block scale) to absorb
the per-feature magnitude distributions measured in exp 26. The value
encoding is compatible; the scale format is LARQL's own extension.

See [Operations Spec Section 6](operations-spec.md) for strategies.

### 12.3 Streaming Build — IMPLEMENTED

Extracts vindex from safetensors without loading the full model into memory. Mmaps safetensors shards, processes one layer at a time. Peak memory = embeddings + 1 layer's weights, not the full model. Enables 120B+ MoE extraction on machines with 16 GB RAM.

---

## License

Apache-2.0
