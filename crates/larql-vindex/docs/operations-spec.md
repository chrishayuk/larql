# Vindex Operations Specification

**Version:** 0.3  
**Date:** 2026-04-01  
**Status:** Implemented (~98%)  
**Implementation:** `larql-vindex` crate (Rust)  
**Companion specs:** [Format](format-spec.md), [Ecosystem](ecosystem-spec.md), [LQL](../../larql-lql/docs/spec.md)

**Implementation coverage:** All core operations (load, KNN, walk, describe, mutate, compile), full patch lifecycle, build pipeline (safetensors/GGUF/MLX), Vindexfile, HuggingFace publish/download — all implemented. Readonly base with auto-patch overlay. GateIndex trait for transparent patched/unpatched access. 600 tests.

---

## 1. Core Operations

### 1.1 Load

```rust
let config = load_vindex_config(&path)?;
let mut cb = SilentLoadCallbacks;
let index = VectorIndex::load_vindex(&path, &mut cb)?;
```

Loading reads:
1. `index.json` → `VindexConfig` (layer offsets, model metadata)
2. `gate_vectors.bin` → per-layer `Array2<f32>` matrices (via offset lookup, f16→f32 cast if needed)
3. `down_meta.bin` → per-feature `FeatureMeta` (tries binary first, falls back to JSONL)

Embeddings, tokenizer, and label files are loaded separately on demand:
```rust
let (embed, embed_scale) = load_vindex_embeddings(&path)?;
let tokenizer = load_vindex_tokenizer(&path)?;
let labels = load_feature_labels(&path)?;
```

Model weights for INFER are loaded lazily — only when an INFER statement is executed.

### 1.2 Gate KNN

```rust
let hits: Vec<(usize, f32)> = index.gate_knn(layer, &residual, top_k);
```

Computes `gate_matrix @ residual` via BLAS matmul, returns top-K feature indices sorted by absolute dot product. This is both the gate computation and the nearest-neighbor search.

Both `VectorIndex` and `PatchedVindex` implement the `GateIndex` trait, which provides `gate_knn`, `feature_meta`, and `num_features`. This allows consumers like `WalkFfn` to work transparently with patched or unpatched indexes — INSERT/DELETE/UPDATE to the vindex immediately affect KNN results and inference output.

**Performance:** exact brute-force BLAS gemv, M3 Max: 22.7 µs/layer at the reduced 1024×256 synthetic shape, **2.64 ms/layer at the Gemma 3 4B production shape (10240×2560)** — criterion `benches/vindex_ops.rs` (`gate_knn_per_layer`); full tables in `README.md`/`PERFORMANCE.md`. (An earlier revision quoted 0.008 ms/layer — that was the pre-2026-04-05 `vindex_bench` example at the reduced synthetic shape, not the production shape.)

### 1.3 Walk

```rust
let trace: WalkTrace = index.walk(&query, &layers, top_k);
```

Runs gate KNN at each layer, annotates hits with down_meta (what each feature outputs). Returns a `WalkTrace` with per-layer `WalkHit` entries:

```rust
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}
```

### 1.4 Describe

```rust
let edges: Vec<DescribeEdge> = describe_entity(
    &entity, &index, &embed, embed_scale, &tokenizer, &labels, &clusters, opts
);
```

Multi-layer gate KNN across the specified layer band, with:
- Probe label lookup (highest priority)
- Cluster label lookup (fallback)
- Edge merging across layers (same target token → combined entry)
- Noise filtering (non-Latin tokens, low gate scores)
- Source tagging (`(probe)`, `(cluster)`, or blank for TF-IDF)

### 1.5 Mutate

The base vindex is always readonly. All mutations go through a `PatchedVindex` overlay. Base files on disk are never modified.

```rust
let mut patched = PatchedVindex::new(index);

// Insert: gate vector + metadata into the patch overlay
patched.insert_feature(layer, feature, gate_vec, meta);

// Delete: mark feature as deleted in the overlay
patched.delete_feature(layer, feature);

// Update: replace metadata in the overlay
patched.update_feature_meta(layer, feature, new_meta);

// Find unused slot (returns weakest feature when no empty slots)
let slot = patched.find_free_feature(layer);

// Bake patches into a clean VectorIndex and save
let baked = patched.bake_down();
baked.save_vindex(&output_path, &mut config)?;
```

In the LQL REPL, INSERT/DELETE/UPDATE automatically create a patch session. Name the patch with `BEGIN PATCH "file.vlp"` and run `SAVE PATCH` to persist, or edits are discarded on exit.

### 1.6 Compile

Two compile targets:

#### `COMPILE INTO VINDEX` — flatten patches into a standalone vindex

Produces a fresh vindex directory whose **bytes** contain the inserted
features. No overlay file, no auto-applied sidecar — the result loads
like any other vindex.

**Algorithm:**
1. Hard-link every read-only weight file from the source (`attn_weights.bin`,
   `up_weights.bin`, `norms.bin`, `weight_manifest.json`, `embeddings.bin`,
   `tokenizer.json`, `up_features.bin`, `down_features.bin`, `down_meta.bin`).
   On APFS this is instant — same inode, same bytes.
2. Save `gate_vectors.bin` from the (unmodified) base — this is byte-identical
   to the source since INSERT does not write gate vectors into this file. The
   inserted gate vector lives only in the patch overlay's `overrides_gate`
   HashMap and is *not* baked into `gate_vectors.bin`. (Doing so would make
   the dense FFN read a moderate-magnitude gate at the inserted slot, which
   combined with the override down vector blows up the residual stream.)
3. Bake the inserted features' down vectors into a fresh copy of
   `down_weights.bin`. For each `(layer, feature)` override, copy the source
   layer slab into RAM, splice the override values into the column at index
   `feature` (which is `hidden_size` scattered cells across the
   `[hidden, intermediate]` row-major matrix), then write the slab back.
4. Recompute checksums and write `index.json`.

**Why down_weights.bin specifically:** the dense FFN inference path
(`walk_ffn_exact` / `load_model_weights`) reads the down projection from
`down_weights.bin` via `weight_manifest.json`. Replacing the column at the
inserted slot makes the inserted feature fire through the standard FFN
path with no runtime overlay. The source weak gate at that slot keeps the
dense activation small, so `small_activation × poseidon_vector` per layer
reproduces the constellation effect — exactly matching the patched session
within f32→f16 round-trip precision.

**End-to-end verification:** On Gemma 4B with `INSERT Atlantis → Poseidon`
followed by `COMPILE CURRENT INTO VINDEX`, a fresh `USE` of the compiled
vindex (no patch overlay loaded) produces:
- `INFER "The capital of Atlantis is"` → **Pose 56.91%** at #1
- `INFER "The capital of France is"` → **Paris 67.34%** (preserved)

This matches the live patched session within rounding (which gave Pose
56.16% / Paris 67.28% — the small delta is f32→f16 down vector
quantisation since `down_weights.bin` is stored as f16).

#### `COMPILE INTO MODEL` — export to HuggingFace / GGUF

Compiles the vindex (with patch overlay) into plain model weights. If the
patch overlay contains INSERT operations, MEMIT closed-form weight editing
is used to bake the inserted facts into `W_down` at the install layer(s).
The output is a standard safetensors directory with no vindex dependency.

**Algorithm:**
1. Load model weights from vindex split files
2. If patches contain INSERTs: run the MEMIT pipeline per install layer:
   a. Estimate FFN activation covariance C from diverse prompts
   b. Capture per-fact FFN activations k* at canonical prompts
   c. Compute target deltas from embedding directions
   d. Solve `ΔW = R^T S⁻¹ Q` where `S = K C⁻¹ K^T + λI`
   e. Apply `ΔW` to `W_down` at each layer
3. Write modified weights as safetensors with HuggingFace naming
4. Copy `tokenizer.json` from vindex metadata

**Requires:** Extract level `All` (model weights present).

**Round-trip test:** EXTRACT → COMPILE (no edits) should produce weights identical to the original within floating-point tolerance.

**Insert round-trip test:** EXTRACT → INSERT → COMPILE → load in
HuggingFace Transformers should produce the inserted facts via standard
generate(), with no special loader code.

### 1.7 Feature Lookup

```rust
let meta: Option<&FeatureMeta> = index.feature_meta(layer, feature);
let n: usize = index.num_features(layer);
let layers: Vec<usize> = index.loaded_layers();
let label: Option<&str> = labels.get(&(layer, feature));
```

---

## 2. Patches

Patches are lightweight, shareable diffs that modify a vindex without changing the base files. They capture INSERT, DELETE, and UPDATE operations as a portable JSON file (.vlp) that can be applied to any vindex built from the same base model.

### 2.1 Patch File Format (.vlp)

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_checksum": "a1b2c3d4...",
  "created_at": "2026-04-01T15:00:00Z",
  "larql_version": "0.1.0",
  "description": "Medical knowledge: drug interactions and side effects",
  "author": "medical-team",
  "tags": ["medical", "pharmacology"],
  "operations": [
    {
      "op": "insert",
      "layer": 26,
      "feature": 8821,
      "relation": "side_effect",
      "entity": "aspirin",
      "target": "bleeding",
      "confidence": 0.85,
      "gate_vector_b64": "<base64 encoded f32 × hidden_size>",
      "up_vector_b64":   "<base64 encoded f32 × hidden_size>",
      "down_vector_b64": "<base64 encoded f32 × hidden_size>",
      "down_meta": {"t": "bleeding", "i": 12847, "c": 4.2}
    },
    {
      "op": "update",
      "layer": 27,
      "feature": 9515,
      "gate_vector_b64": "<base64 encoded f32 × hidden_size>",
      "up_vector_b64":   "<base64 encoded f32 × hidden_size>",
      "down_vector_b64": "<base64 encoded f32 × hidden_size>",
      "down_meta": {"t": "Paris", "i": 8921, "c": 5.1}
    },
    {
      "op": "delete",
      "layer": 24,
      "feature": 1337,
      "reason": "hallucinated fact"
    }
  ]
}
```

**Size:** A single fact carries up to three vectors (gate + up + down, each `hidden_size × f32`) ≈ 30 KB + metadata. Compose-mode `INSERT` writes all three so the `.vlp` round-trips losslessly through `apply_patch` → `COMPILE INTO VINDEX`; a metadata-only `update` typically omits the vector fields. A 1,000-fact patch is ~30 MB at most. Compared to the full model at 8 GB, this is still 1/250th the size. The `up_vector_b64` and `down_vector_b64` fields are optional — `.vlp` files written before they were introduced still parse, with both fields defaulting to `None`.

### 2.2 LQL Patch Operations

```sql
-- ═══ Creating Patches ═══

-- Start a patch session (edits captured, base vindex unchanged)
BEGIN PATCH "medical-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

-- Save the patch (base vindex is NOT modified)
SAVE PATCH;

-- ═══ Applying Patches ═══

USE "gemma3-4b.vindex";
APPLY PATCH "medical-knowledge.vlp";

-- Stack multiple patches (applied in order)
APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
APPLY PATCH "company-facts.vlp";

-- See active patches
SHOW PATCHES;

-- Remove a patch
REMOVE PATCH "fix-hallucinations.vlp";

-- ═══ Baking Down ═══

-- Flatten patches into a new clean vindex
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

-- Or compile straight to safetensors for deployment
COMPILE CURRENT INTO MODEL "gemma3-4b-medical/" FORMAT safetensors;

-- ═══ Extracting Patches from Diffs ═══

DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex"
    INTO PATCH "medical-changes.vlp";
```

### 2.3 Patch Application

Patches modify the in-memory vindex without touching base files:

```rust
pub struct PatchedVindex {
    base: VectorIndex,                          // Immutable base
    patches: Vec<VindexPatch>,                  // Applied in order
    overrides: HashMap<(usize, usize), PatchOp>, // (layer, feature) → operation
}

impl PatchedVindex {
    /// Gate KNN checks overrides first, then falls through to base
    fn gate_knn(&self, layer: usize, residual: &[f32], top_k: usize) -> Vec<(usize, f32)>;
    
    /// Feature lookup checks overrides, then base
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<&FeatureMeta>;
    
    /// Flatten all patches into the base, producing a new clean VectorIndex
    fn bake_down(&self) -> VectorIndex;
}
```

The base vindex files on disk are never modified. Patches are an overlay:
- Multiple users can apply different patches to the same base
- Patches can be reverted cleanly (remove from the stack)
- The base vindex remains cacheable and immutable

### 2.4 Conflict Resolution

Later patches override earlier ones for the same feature:

```sql
-- Patch A inserts F8821@L26 with target "Colchester"
-- Patch B updates F8821@L26 with target "London"
-- Result: F8821@L26 → "London" (Patch B wins)
```

Explicit strategies for COMPILE INTO VINDEX:
```sql
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT LAST_WINS;          -- Default
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT HIGHEST_CONFIDENCE;
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT FAIL;
```

### 2.5 Comparison with LoRA

| Dimension | LoRA Adapter | Vindex Patch |
|-----------|-------------|-------------|
| **Size** | ~50-200 MB | ~10 KB per fact |
| **Creation** | Training (hours, GPU) | INSERT statement (seconds, CPU) |
| **Granularity** | Low-rank approximation | Exact: specific features, specific facts |
| **Human-readable** | No | Yes (JSON with entity, relation, target) |
| **Composable** | Limited (merging is lossy) | Yes (patches stack, conflicts resolved) |
| **Reversible** | Partially | Fully (base unchanged) |
| **Training required** | Yes | No |

LoRA is for broad behaviour adaptation (tone, style). Vindex patches are for specific fact insertion/correction. They're complementary.

---

## 3. Build Pipeline

### 3.1 Extract from Model

Supports safetensors (HuggingFace), GGUF (llama.cpp, dequantized to f32), and MLX (Apple, safetensors layout). Auto-detected from file extension and directory structure. **Streaming mode** — mmaps safetensors shards and processes one layer at a time. Peak memory = embeddings + 1 layer, not the full model. Vindexes can be published to and downloaded from HuggingFace Hub.

```bash
# From safetensors (HuggingFace)
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --f16

# From GGUF
larql convert gguf-to-vindex model-Q4_K_M.gguf -o model.vindex --f16

# Download pre-built vindex from HuggingFace
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Or use directly in the REPL (auto-downloads)
# USE "hf://chrishayuk/gemma-3-4b-it-vindex";

# With inference weights
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16

# With all weights
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

**Build steps:**
1. Mmap safetensors shards (streaming — no full model load)
2. Extract gate vectors → `gate_vectors.bin`
3. Extract embeddings → `embeddings.bin`
4. Compute down metadata → `down_meta.bin`
5. Compute relation clusters → `relation_clusters.json`
6. Copy tokenizer → `tokenizer.json`
7. (Inference level) Write attn_weights.bin, norms.bin
8. (All level) Write up_weights.bin, down_weights.bin, lm_head.bin
9. Compute checksums → stored in `index.json`

**Total build time:** ~12 minutes on M-series Mac (Gemma 3 4B).

### 3.2 Add Labels

```bash
larql label gemma3-4b.vindex --probes feature_labels.json
larql label gemma3-4b.vindex --triples wikidata_triples.json --wordnet wordnet_relations.json
```

Labels are additive — new probes add labels without removing existing ones.

### 3.3 Resume

```bash
larql extract-index ... --resume
```

Checks which files exist and skips completed steps. Enables incremental rebuilds.

---

## 4. Rust API

### 4.1 Core Types

```rust
pub struct VectorIndex { ... }

pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

pub struct DescribeEdge {
    pub relation: Option<String>,
    pub source: LabelSource,
    pub target: String,
    pub gate_score: f32,
    pub layer_min: usize,
    pub layer_max: usize,
    pub count: usize,
    pub also_tokens: Vec<String>,
}

pub enum LabelSource {
    Probe,      // Model inference confirmed
    Cluster,    // Cluster-based matching
    Pattern,    // Entity pattern detection
    None,       // TF-IDF fallback
}

pub struct VindexConfig {
    pub version: u32,
    pub model: String,
    pub family: String,
    pub dtype: StorageDtype,
    pub source: Option<VindexSource>,
    pub checksums: Option<HashMap<String, String>>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub embed_scale: f32,
    pub extract_level: ExtractLevel,
    pub has_model_weights: bool,
    pub layer_bands: LayerBands,
    pub layers: Vec<VindexLayerInfo>,
    pub down_top_k: usize,
    pub model_config: Option<VindexModelConfig>,
}

pub struct VindexSource {
    pub huggingface_repo: Option<String>,
    pub huggingface_revision: Option<String>,
    pub safetensors_sha256: Option<String>,
    pub extracted_at: String,
    pub larql_version: String,
}

pub struct LayerBands {
    pub syntax: (usize, usize),
    pub knowledge: (usize, usize),
    pub output: (usize, usize),
}

pub enum ExtractLevel { Browse, Inference, All }
pub enum StorageDtype { F32, F16 }

pub struct VindexPatch {
    pub version: u32,
    pub base_model: String,
    pub base_checksum: Option<String>,
    pub created_at: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub operations: Vec<PatchOp>,
}

pub enum PatchOp {
    Insert {
        layer, feature, relation, entity, target, confidence,
        // Per-component overrides — each carried as Option<base64-f32>.
        // Compose-mode INSERT writes all three; older patches that only
        // had `gate_vector_b64` still parse (up/down default to None).
        gate_vector_b64, up_vector_b64, down_vector_b64,
        down_meta,
    },
    Update {
        layer, feature,
        gate_vector_b64, up_vector_b64, down_vector_b64,
        down_meta,
    },
    Delete { layer, feature, reason },
    // Architecture B residual-key KNN ops:
    InsertKnn { layer, entity, relation, target, target_id, confidence, key_vector_b64 },
    DeleteKnn { entity },
}

pub struct PatchedVindex {
    pub base: VectorIndex,
    pub patches: Vec<VindexPatch>,
    pub overrides: HashMap<(usize, usize), PatchOp>,
}

pub enum VindexError {
    NotADirectory(PathBuf),
    NoSafetensors(PathBuf),
    MissingTensor(String),
    Parse(String),
    UnsupportedDtype(String),
    InsufficientExtractLevel { needed: ExtractLevel, have: ExtractLevel },
    Io(std::io::Error),
}
```

### 4.2 Load Functions

```rust
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError>;
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError>;
pub fn load_vindex_tokenizer(dir: &Path) -> Result<Tokenizer, VindexError>;
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError>;
```

### 4.3 Callbacks

```rust
pub trait IndexBuildCallbacks {
    fn on_layer_start(&mut self, layer: usize, total: usize) {}
    fn on_layer_done(&mut self, layer: usize, elapsed_ms: f64) {}
    fn on_phase(&mut self, phase: &str) {}
    fn on_complete(&mut self, elapsed_ms: f64) {}
}

pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {}
    fn on_progress(&mut self, records: usize) {}
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {}
}
```

---

## 5. Crate Structure

The full annotated source tree lives in [`crates/larql-vindex/README.md`](../README.md#crate-structure)
under the **Crate Structure** section. It's the single source of
truth — keeping two trees in two places is exactly the kind of
drift the round-1/2/4 audits found.

Highlights of the layout consumers usually need to know:

- **`index/`** — `VectorIndex` + the substores it composes. Sibling
  modules under `index/compute/gate_knn/`, `index/storage/ffn_store/`,
  and `index/storage/lm_head/` each carry one impl-block fragment for
  one concern (KNN dispatch, HNSW lifecycle, per-format FFN accessors,
  lm_head loaders/KNN). All public methods stay reachable through the
  same `VectorIndex` API.
- **`extract/build/`** — `BuildContext` 6-stage pipeline:
  `mod.rs` orchestrates + holds the small stages, with `down_meta.rs`,
  `index_json.rs`, and `resume.rs` as siblings.
- **`format/filenames.rs`** — single source of truth for every
  `.bin` / `.json` filename. A typo at any reader/writer site is now
  a compile error.
- **`format/weights/manifest.rs` + `quant/registry.rs`** — typed
  Q4_K manifest entries and the format registry (`QUANT_FORMATS` +
  `lookup`). Adding a K-quant is one entry plus codec functions.
- **`engine/`** (formerly `storage/`) — `StorageEngine` +
  epoch + MEMIT cycles.
- **`patch/`** — `VindexPatch` / `PatchOp` / `PatchedVindex`
  overlay. `Insert` / `Update` carry optional `gate_vector_b64` /
  `up_vector_b64` / `down_vector_b64` so a `.vlp` round-trips losslessly
  through `apply_patch` → `COMPILE INTO VINDEX`.

**Dependencies:** `larql-models` (ModelWeights, architectures, quant, loading), `ndarray` (BLAS), `serde`/`serde_json`, `tokenizers`, `thiserror`

Model loading (safetensors, GGUF, MLX) and quantization (f16, Q4_0, MXFP4) live in `larql-models`.

---

## 6. MoE and Quantized Models

### 6.1 Knowledge Extraction by Architecture Type

| Architecture | Weights | DESCRIBE | WALK | INFER | Notes |
|---|---|---|---|---|---|
| Dense (Gemma, Llama, Qwen) | f32/f16/bf16 | ✅ Works | ✅ Works | ✅ Works | Gate KNN with raw embeddings is accurate |
| MoE, full precision (Mixtral) | f16/bf16 per expert | ✅ Expected | ✅ Expected | ✅ Works | Per-expert gate vectors have enough precision |
| MoE, MXFP4 (GPT-OSS) | 4-bit block quantized | ❌ Noisy | ❌ Noisy | ✅ Works | 4-bit gate vectors lack precision for isolated KNN |

### 6.2 Why MXFP4 Models Work at Inference but Not for Browse

At inference time, GPT-OSS produces correct answers ("The capital of France is Paris") using MXFP4 weights. The knowledge IS encoded in the 4-bit weights. But DESCRIBE with raw embeddings fails because:

1. **Gate KNN is not how the model uses gate vectors.** The model computes `SiLU(x @ W_gate) * (x @ W_up)` — a multiplicative interaction between gate and up projections. The SiLU gating combined with the up projection selects very different features than the raw gate dot product alone.

2. **The model uses transformed residuals, not raw embeddings.** By layer 20, the input has been through 20 layers of attention and FFN. The raw token embedding for "France" at layer 20 is meaningless — the model sees a transformed representation.

3. **4-bit precision creates noisy dot products.** MXFP4 quantizes each weight to one of 16 values (±{0, 0.5, 1, 1.5, 2, 3, 4, 6} × shared scale). For dense models at f16 (65K distinct values per weight), gate KNN produces ~14 features above threshold 5.0. For GPT-OSS at MXFP4, **59,717 features** score above 5.0 — the signal is lost in noise.

4. **MoE expert specialization.** With 128 experts of 2,880 features each, individual features are highly context-specific. They're designed to activate in specific routing contexts, not for isolated entity lookup.

### 6.3 Strategies for MXFP4/MoE Knowledge Access

**Working now:**
- **INFER** (with model weights) — full forward pass produces correct answers
- **Serving** — the vindex loads in 2 seconds, gate KNN still provides fast approximate retrieval

**Future approaches:**
- **Residual-based DESCRIBE** — run a forward pass through attention layers to get the actual residual at each layer, then use that for gate KNN. Requires inference-level extraction.
- **Probe-based labeling** — run known entity prompts through the model, capture which features activate in each expert, build labels empirically. The probe pipeline from larql-knowledge.
- **Router-level knowledge** — use the 128 router directions per layer as "macro-features". The router weights are bf16 (full precision) and cleanly separate entity types into expert clusters.
- **Gated KNN** — compute the full `SiLU(gate) × up` activation instead of raw gate dot product. Requires both halves of the fused tensor and is 2× the computation, but produces accurate feature activations.
- **Unquantized extraction** — if full-precision weights become available, standard gate KNN will work.

### 6.4 Detection and User Guidance

When loading an MXFP4-quantized model, LARQL detects `ExpertFormat::PackedMxfp4` and should warn:

```
⚠ MXFP4 quantized experts detected (GPT-OSS family).
  DESCRIBE/WALK use approximate gate KNN — results may be noisy.
  For accurate knowledge queries, use INFER (requires --level inference).
  For browse-quality results, probe labels are recommended.
```

---

## 7. Benchmarks

Criterion `benches/vindex_ops.rs`, M3 Max, synthetic data (exact
brute-force BLAS gemv — see `README.md` for the full per-shape tables):

| Operation | Latency |
|---|---|
| Gate KNN, per layer (1024f × 256h synthetic) | 22.7 µs |
| Gate KNN, per layer (10240f × 2560h — Gemma 3 4B shape) | 2.64 ms |
| Walk (8L × 1024f × 256h synthetic) | 216 µs |
| Walk (8L × 10240f × 2560h — Gemma band) | 21.2 ms |
| Feature meta lookup | ~245 ns |
| Save gates (8 MB) | 2.0 ms |
| Load vindex (mmap) | 261 µs |
| Mutate (meta + gate) | 301 ns |
| Checksum (SHA256) | 19.9 ms (1024×256 synthetic — `PERFORMANCE.md` core-ops table) |
| MoE 8x scaling | 17.6× for 8× features (sub-linear) |

---

## License

Apache-2.0
