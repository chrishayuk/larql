# Vindex Ecosystem Specification

**Version:** 0.3  
**Date:** 2026-04-01  
**Status:** Implemented (~85%)  
**Companion specs:** [Format](format-spec.md), [Operations](operations-spec.md), [LQL](../../larql-lql/docs/spec.md)

**What's implemented:** Format comparisons (Section 1), distributed hosting with HTTP/gRPC server (Section 2/5), USE REMOTE in REPL (Section 2.3), HuggingFace publish/download (Section 4), remote serving API with all endpoints (Section 5), multi-tenant serving with per-session patches (Section 5.4), Vindexfile parsing and build (Section 6), browse-only and knowledge patching strategies (Section 3).  
**What's planned:** GGUF output format, CI/CD templates.

---

## 1. How Vindex Relates to Other Formats

Vindex serves a different purpose from existing model formats. Where safetensors and GGUF optimise for inference, and SAE features optimise for interpretability, vindex optimises for knowledge access — treating the model's weights as a queryable, editable database.

### Where each format excels

**safetensors** is the standard distribution format. It stores weight tensors efficiently with fast random access and memory mapping. Every inference framework reads it. Vindex doesn't replace safetensors for inference — it's what you produce *from* safetensors when you want to query the knowledge.

**GGUF** is optimised for efficient CPU inference with quantisation. Compresses models to 2-4× smaller than f16 while maintaining quality. Vindex doesn't do quantised inference — it separates knowledge browsing from inference entirely.

**ONNX** targets cross-framework deployment with computation graph optimisation. Vindex isn't a deployment format — it's an inspection and editing format.

**SAE features** (sparse autoencoders) are the closest relative to vindex. Both extract meaningful features from model weights. SAEs learn features post-hoc by training an autoencoder on activations. Vindex extracts features directly from existing weight matrices — no additional training. And vindex features are writable: INSERT adds a fact, COMPILE produces a new model.

### Comparative strengths

| Dimension | safetensors | GGUF | SAE features | Vindex |
|-----------|-------------|------|-------------|--------|
| **Primary purpose** | Model distribution | Efficient CPU inference | Interpretability research | Knowledge access + editing |
| **Query without inference** | No | No | Partially (need forward pass) | Yes (gate KNN, 33ms) |
| **Edit a fact** | Retrain (hours, GPU) | Retrain + requantise | Not supported | INSERT + COMPILE (seconds, CPU) |
| **Browse-only size (4B)** | 8 GB (full model) | 2-4 GB (full model) | 2-10 GB (on top of model) | 3 GB (gate + embed only) |
| **GPU required** | For inference | No | For training SAE | No (browse), optional (infer) |
| **Typed knowledge edges** | No | No | Manual labelling | Yes (probe + Wikidata + WordNet) |
| **Hostable on CDN** | No (needs compute) | No (needs compute) | No (needs compute) | Yes (static files, dot products) |
| **Round-trip to weights** | N/A (is the weights) | Lossy (quantisation) | No | Yes (EXTRACT → edit → COMPILE) |

### Performance comparison

To answer "What does the model know about France?":

| Format | Method | Time | Hardware |
|--------|--------|------|----------|
| safetensors | Full forward pass with probing prompts | ~800ms | GPU |
| GGUF Q4 | Quantised forward pass | ~200ms | CPU |
| SAE features | Forward pass + SAE decode + inspection | ~2s | GPU |
| Vindex browse | Gate KNN across 14 knowledge layers | ~33ms | CPU |
| Vindex infer | Attention + walk FFN (full prediction) | ~200ms | CPU |

### Scaling by model size

| Model | safetensors | GGUF Q4 | Vindex browse | Vindex full |
|-------|-------------|---------|---------------|-------------|
| 4B (Gemma 3) | 8 GB, 1 GPU | 2 GB, CPU | 3 GB, CPU | 10 GB, CPU |
| 8B (Llama 3) | 16 GB, 1 GPU | 4 GB, CPU | 5 GB, CPU | 18 GB, CPU |
| 70B (Llama 3) | 140 GB, multi-GPU | 35 GB, CPU | ~25 GB, CPU | ~80 GB, CPU |
| 405B (Llama 3) | 800 GB, GPU cluster | 200 GB, multi-node | ~120 GB, distributed | ~500 GB, distributed |

---

## 2. Distributed Hosting

The vindex format is designed for distributed access. Each file is independently loadable, serves a specific function, and has a known size.

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│  Phone / Laptop / Edge device                                    │
│  Runs: larql binary (no ML framework, no GPU)                   │
│                                                                  │
│  USE REMOTE "https://models.example.com/gemma3-4b.vindex";      │
│  DESCRIBE "France";                                              │
└─────────────┬───────────────────────────────────────────────────┘
              │ HTTPS / gRPC
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Server                              │
│  Hosts: gate_vectors.bin + embeddings.bin + down_meta.bin        │
│  Serves: DESCRIBE, WALK, SELECT, EXPLAIN WALK                   │
│  Size: ~3 GB (f16 browse-only)                                  │
│  Hardware: Any CPU, no GPU needed                               │
│  Latency: <50ms per query                                        │
└─────────────┬───────────────────────────────────────────────────┘
              │ (optional, only for INFER)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Server                               │
│  Hosts: attn_weights.bin + norms.bin                             │
│  Serves: INFER, EXPLAIN INFER                                   │
│  Size: ~3 GB additional (f16)                                   │
│  Latency: ~200ms per inference                                   │
└─────────────┬───────────────────────────────────────────────────┘
              │ (optional, rare)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Compile Server                                 │
│  Hosts: up_weights.bin + down_weights.bin + lm_head.bin         │
│  Serves: COMPILE (batch jobs, not real-time)                    │
│  Size: ~4 GB additional                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Access Patterns

| Operation | Files Needed | Server | Latency | Bandwidth |
|-----------|-------------|--------|---------|-----------|
| DESCRIBE | gate + embed + down_meta + labels | Knowledge | <50ms | ~1 KB response |
| WALK | gate + embed | Knowledge | <50ms | ~1 KB response |
| SELECT | gate + down_meta + labels | Knowledge | <10ms | ~1 KB response |
| SHOW RELATIONS | labels only | Knowledge | <1ms | ~10 KB response |
| STATS | index.json only | Any | <1ms | ~8 KB response |
| INFER | gate + embed + attn_weights | Inference | ~200ms | ~1 KB response |
| COMPILE | All files | Compile | minutes | ~10 GB output |

### 2.3 Remote Protocol

```sql
-- Connect to a remote vindex
USE REMOTE "https://models.example.com/gemma3-4b.vindex";

-- Knowledge queries go to the knowledge server
DESCRIBE "France";
-- → GET /api/v1/describe?entity=France
-- ← JSON: [{relation: "capital", target: "Paris", gate_score: 1436.9}]

WALK "Einstein" TOP 10;
-- → GET /api/v1/walk?prompt=Einstein&top=10

-- Inference goes to the inference server
INFER "The capital of France is" TOP 5;
-- → POST /api/v1/infer {prompt: "...", top: 5}
```

The REPL detects whether the backend is local or remote and routes accordingly. Same LQL statements, same output format.

### 2.4 Layer-Level Access

For constrained clients, individual layers can be fetched on demand:

```
GET /api/v1/layers/27/gate_vectors
← Binary: 10,240 × 2,560 × 2 bytes = ~50 MB (f16)
```

The `index.json` layer offsets enable byte-range requests against a static file server:

```
GET /files/gate_vectors.bin
Range: bytes=5242880000-5347737600
← Layer 27 data
```

### 2.5 Decoupled Models

The distributed architecture enables mixing components from different models:

```sql
-- Use a small model's attention with a large model's knowledge
USE ATTENTION MODEL "google/gemma-3-4b-it";
USE KNOWLEDGE REMOTE "https://models.example.com/llama-70b.vindex";

INFER "The capital of France is" TOP 5;
-- Small model runs attention locally
-- Knowledge query hits the 70B vindex remotely
-- Paris comes out (70B's knowledge, 4B's speed)
```

This works because the cross-model entity×relation coordinates align at 0.946 cosine (proven via Procrustes), attention routing is template-based (99% of heads are fixed), and the gate KNN accepts any residual vector.

### 2.6 Caching

- **gate_vectors.bin, embeddings.bin** — cache indefinitely (immutable after EXTRACT)
- **index.json** — cache indefinitely (versioned)
- **down_meta** — cache indefinitely (immutable unless INSERT)
- **feature_labels.json** — cache with TTL (updates as probes run)

Standard HTTP caching (ETag, Cache-Control: immutable) works perfectly. The checksums in index.json serve as ETags.

---

## 3. Large Models on Small Hardware

### 3.1 The Hardware Problem

| Model | Size (f16) | Minimum Hardware for Inference |
|-------|-----------|-------------------------------|
| Llama 3 8B | 16 GB | 1× GPU (24GB VRAM) or 32GB RAM |
| Llama 3 70B | 140 GB | 2-4× A100 GPUs |
| Llama 3 405B | 800 GB | 8× A100 or 4× H100 |

Most people cannot run a 70B model. The knowledge inside these models is locked behind a hardware wall.

### 3.2 What Vindex Changes

| Model | Full Inference | Vindex Browse | Hardware for Browse |
|-------|---------------|---------------|---------------------|
| Llama 3 8B | 16 GB, GPU | ~5 GB | Any laptop |
| Llama 3 70B | 140 GB, multi-GPU | ~25 GB | Workstation (32GB RAM) |
| Llama 3 405B | 800 GB, GPU cluster | ~120 GB | Server or streamed |

### 3.3 Five Strategies

**Strategy 1: Browse-only.** Load gate vectors and embeddings. DESCRIBE, WALK, SELECT work fully. No generation, but full knowledge access.

```sql
-- On a laptop with 8GB RAM, browse a 4B model's knowledge
larql> USE "gemma3-4b.vindex";  -- loads 3 GB
larql> DESCRIBE "France";       -- 33ms, full knowledge graph
```

**Strategy 2: Layer-on-demand.** Fetch individual layers as needed. Each layer is ~50-100MB (f16). A phone can query one layer at a time.

```sql
larql> USE "gemma3-4b.vindex" LAYERS ON DEMAND;
larql> DESCRIBE "France" AT LAYER 27;  -- fetches ~100 MB, scans 1 layer
```

**Strategy 3: Decoupled inference.** Run a small model's attention locally. Query a large model's knowledge remotely.

```sql
larql> USE ATTENTION MODEL "google/gemma-3-4b-it";
larql> USE KNOWLEDGE REMOTE "https://models.example.com/llama-70b.vindex";
larql> INFER "The mechanism of action of metformin is" TOP 5;
```

**Strategy 4: Knowledge patching.** Start with a small model. Apply patches from a large model to inject domain knowledge.

```sql
larql> USE "gemma3-4b.vindex";
larql> APPLY PATCH "llama70b-medical.vlp";  -- 50 MB
larql> DESCRIBE "metformin";
-- mechanism → biguanide (probe)
-- treats → diabetes (probe)
```

**Strategy 5: Quantised browse.** Store gate vectors at int8 or int4 precision.

```
Gate vectors at f32:  3.32 GB
Gate vectors at f16:  1.66 GB
Gate vectors at int8: 0.83 GB
Gate vectors at int4: 0.42 GB — a 4B model's knowledge in 400 MB
```

### 3.4 What Requires Full Hardware

| Operation | Small Hardware | Full Hardware |
|-----------|--------------|---------------|
| DESCRIBE (knowledge lookup) | Yes | Yes |
| WALK (feature scan) | Yes | Yes |
| SELECT (knowledge query) | Yes | Yes |
| INFER (text generation) | Decoupled only | Yes |
| COMPILE (model editing) | No (needs all weights) | Yes |

Knowledge access works on anything, generation needs compute. Most use cases are knowledge access.

---

## 4. Publishing and Registry

### 4.1 HuggingFace Publishing

Vindexes publish to HuggingFace as dataset repos. The directory maps 1:1 to the repo structure.

```bash
# Build locally
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all

# Run probes for labels (optional)
python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex gemma3-4b.vindex

# Publish
larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex

# Anyone can now use it
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
```

**Lazy loading.** `USE` downloads only `index.json` first (~5 KB). Gate vectors download on first DESCRIBE (~3 GB at f16). Attention weights download only if INFER is called. You pay for what you use.

### 4.2 Patch Publishing

Patches publish as HuggingFace repos or any URL:

```bash
larql publish-patch medical-knowledge.vlp \
  --repo medical-ai/drug-interactions \
  --base google/gemma-3-4b-it
```

```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
larql> APPLY PATCH "hf://medical-ai/drug-interactions@2.1.0";
larql> APPLY PATCH "hf://legal-team/uk-case-law-2026@1.0.0";
```

A patch ecosystem on HuggingFace:

```
Base models:
  chrishayuk/gemma-3-4b-it-vindex
  chrishayuk/llama-3-8b-vindex

Knowledge patches:
  medical-ai/drug-interactions           (5K facts, 50 MB)
  legal-team/uk-case-law-2026            (3K facts, 30 MB)
  sports-data/premier-league-2026        (10K facts, 100 MB)
  fix-hallucinations/gemma-3-4b-v1       (200 corrections, 2 MB)
```

Patches are versioned, dependency-tracked, and composable. Like npm packages for model knowledge.

---

## 5. Remote Serving

### 5.1 Vindex Server

A lightweight Rust binary that loads a vindex and serves queries over HTTP. No GPU, no ML framework, no Python.

```bash
# Serve a single model
larql serve gemma3-4b.vindex --port 8080

# Serve multiple models
larql serve --dir ./vindexes/ --port 8080

# Serve from HuggingFace directly
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080
```

### 5.2 API Endpoints

```
Knowledge (browse-only, no GPU):
  GET  /v1/describe?entity=France
  GET  /v1/walk?prompt=Einstein&top=10
  GET  /v1/select?relation=capital&limit=20
  GET  /v1/relations
  GET  /v1/stats

Inference (requires attention weights):
  POST /v1/infer    {"prompt": "The capital of France is", "top": 5}

Management:
  GET  /v1/health
  GET  /v1/models
  POST /v1/patches/apply  {"url": "hf://medical-ai/drug-interactions@2.1.0"}
  GET  /v1/patches
```

### 5.3 Client-Side Patches on Remote Base

The server hosts the base model. The client brings patches:

```sql
USE REMOTE "https://vindex.larql.dev/gemma-3-4b-it";
APPLY PATCH "medical-knowledge.vlp";      -- 50 MB local
DESCRIBE "aspirin";
-- side_effect → bleeding   (local patch)
-- occupation → drug        (remote base)
```

Patches never leave the client. Proprietary knowledge stays local.

### 5.4 Multi-Tenant Serving

Same server, different knowledge per client:

```
Server:   3 GB base vindex (f16), serves all clients
Client A: base + medical.vlp          (doctor)
Client B: base + legal.vlp            (lawyer)
Client C: base + medical + company.vlp (pharma company)
```

No fine-tuning per customer. One base model + patches:

```
Traditional: Fine-tune per customer      $50K each, $500K for 10 customers
Vindex:      One base + patches/customer  $240/year total for all customers
```

---

## 6. Vindexfile — Declarative Model Builds

### 6.1 Format

A `Vindexfile` is a declarative specification for building a custom model from a base vindex plus patches and edits. Like a Dockerfile for model knowledge.

```dockerfile
# Vindexfile

# Base model
FROM hf://chrishayuk/gemma-3-4b-it-vindex

# Community knowledge patches
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH hf://medical-ai/anatomy-basics@1.0.0
PATCH hf://legal-team/uk-case-law-2026@1.0.0

# Bug fixes
PATCH hf://fix-hallucinations/gemma-3-4b-v1@1.2.0

# Local patches
PATCH ./patches/company-facts.vlp

# Inline edits
INSERT ("Acme Corp", "headquarters", "London")
INSERT ("Acme Corp", "ceo", "Jane Smith")
# DELETE removes a fact an earlier INSERT or PATCH in this file installed,
# matched on all three fields exactly. Base features carry no entity or
# relation, so a DELETE that matches no installed fact fails the build.
DELETE entity = "Acme Corp" AND relation = "competitor" AND target = "WrongCo"

# Probe labels
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest

# Build configuration
EXPOSE browse inference
```

### 6.2 Build Commands

```bash
# Build a custom model from a Vindexfile
larql build .

# Build and compile to safetensors
larql build . --compile safetensors --output acme-model/

# Build and compile to GGUF for deployment
larql build . --compile gguf --quant Q4_K_M --output acme-model.gguf

# Serve directly without compiling (development mode)
larql serve .

# Publish the built model
larql build . --publish hf://acme-corp/acme-medical-model
```

### 6.3 Build Layers and History

Like Docker, builds are layered. Each PATCH and INSERT is a layer. Builds are reproducible.

```bash
larql history build/vindex/
# Layer 0: FROM gemma-3-4b-it-vindex          (348,160 features)
# Layer 1: PATCH drug-interactions@2.1.0      (+5,000 features modified)
# Layer 2: PATCH anatomy-basics@1.0.0         (+1,200 features modified)
# Layer 3: INSERT 3 edges                     (+3 features added)
# Total: 348,163 active features, 6,203 modified from base
```

### 6.4 Dependency Resolution

Patches can declare dependencies. The build resolves them automatically:

```json
{
  "name": "drug-interactions",
  "version": "2.1.0",
  "depends_on": ["hf://medical-ai/anatomy-basics@>=1.0.0"]
}
```

### 6.5 Environment Variants

```dockerfile
# Vindexfile

FROM hf://chrishayuk/gemma-3-4b-it-vindex

# Shared patches
PATCH hf://fix-hallucinations/gemma-3-4b-v1@1.2.0

# Development: all knowledge, full access
STAGE dev
  PATCH ./patches/experimental-knowledge.vlp
  EXPOSE browse inference compile

# Production: curated knowledge only
STAGE prod
  PATCH hf://medical-ai/drug-interactions@2.1.0
  PATCH ./patches/company-facts.vlp
  EXPOSE browse inference

# Edge: minimal knowledge, browse only
STAGE edge
  PATCH ./patches/core-facts.vlp
  EXPOSE browse
```

```bash
larql build . --stage prod
larql build . --stage edge --compile gguf --quant Q4_K_M
```

### 6.6 CI/CD Integration

Vindexfiles are version-controlled. CI validates builds automatically:

```yaml
# .github/workflows/build-model.yml
name: Build Model
on:
  push:
    paths: ['Vindexfile', 'patches/**']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: larql/setup-action@v1
      
      - name: Build
        run: larql build . --stage prod
      
      - name: Verify round-trip
        run: larql verify build/vindex/
      
      - name: Test knowledge
        run: |
          larql query build/vindex/ "DESCRIBE 'France'" | grep "capital.*Paris"
          larql query build/vindex/ "DESCRIBE 'Acme Corp'" | grep "headquarters.*London"
      
      - name: Publish
        if: github.ref == 'refs/heads/main'
        run: larql build . --publish hf://acme-corp/acme-model
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

Every push to the Vindexfile triggers a rebuild. Tests verify key facts exist. Main branch auto-publishes. Model knowledge gets the same CI/CD rigour as application code.

---

## 7. Hosting Economics

```
Traditional inference hosting:
  GPU cluster:     $2-10/hour per A100
  Serving 1 model: ~$50K/year
  Scaling:         More GPUs, linear cost

Vindex knowledge hosting:
  VPS with 8GB RAM: $20/month
  Serving 1 model:  $240/year
  Scaling:          CDN for static files, near-zero marginal cost

  The "inference" is a dot product. No GPU. No framework.
  The files are static. CDN-cacheable.
  10,000 users querying the same vindex = same cost as 1 user.
```

Knowledge hosting scales like a CDN. Inference (when needed) scales like compute. Most queries are knowledge queries.

| Format | Hosting requirement | Annual cost | Marginal cost per user |
|--------|--------------------|-----------------------|-----------------------|
| safetensors | GPU inference cluster | ~$50,000 | Linear (more GPUs) |
| GGUF | CPU compute nodes | ~$5,000 | Linear (more CPUs) |
| Vindex browse | Static file server / CDN | ~$240 | Near-zero (CDN-cacheable) |
| Vindex infer | CPU compute + static files | ~$2,500 | Linear for infer, zero for browse |

---

## License

Apache-2.0
