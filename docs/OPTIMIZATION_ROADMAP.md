# LARQL Optimization Roadmap & System Design Notes

**PR #55 - Optimization Overhaul**
**Date**: 2026-04-17
**Status**: Active Development Roadmap

This document captures the comprehensive optimization strategy for LARQL based on profiling, experimental results, and system design analysis. Use this as the authoritative reference for optimization priorities and architectural decisions.

---

## Executive Summary

LARQL has achieved **walk FFN faster than dense** (517ms vs 535ms) with 100% accuracy preservation. The system is ready for three major optimization phases that will reduce inference from 517ms to <100ms while maintaining the core architectural principles:

1. **Local knowledge stays local** (patch system, no base mutation)
2. **Zero-copy mmap-first** (OS-managed pages, no allocations)
3. **Extraction levels as gates** (browse/inference/all, not features)

---

## Current State (Baseline Metrics)

### Performance Profile (Gemma-3 4B, Apple Silicon)

```
Component              Time      % of 541ms    Status
──────────────────────────────────────────────────────────
Logits projection      221ms     41%           ← #1 BOTTLENECK
FFN × 34 layers        206ms     38%           ✓ SOLVED (walk: 517ms)
Attention × 34          84ms     16%           ← #2 TARGET
Softmax + top-k          2ms      0%           ✓ MINIMAL
Framework overhead       7ms      1%           ✓ CLEAN
──────────────────────────────────────────────────────────
Total                  520ms     100%
```

### Memory Profile

```
Component                        Size          Notes
────────────────────────────────────────────────────────────
Model safetensors (mmap)      16,613 MB       Full model weights
Vindex gate vectors (mmap)        84 MB       f32, demand-paged
Walk FFN pages                 3,404 MB       down_features.bin
Dense forward pass                48 MB       Temporary buffers
────────────────────────────────────────────────────────────
Current total                 ~20,000 MB
Walk-only potential            ~3,500 MB      13GB savings
```

### Accuracy Achievements

- **Walk boundary sweep**: 100% match at all 34 layer boundaries
- **Training-free INSERT**: 94.6% confidence (Atlantis→Poseidon)
- **Neighbor preservation**: Paris 60.5% (20pt degradation from 80.5%)
- **Cross-lingual discovery**: Automatic emergence in FFN features

---

## Phase 1: Low-Hanging Fruit (Weeks 1-4)

### 1.1 Logits from Vindex Metadata ⚡ HIGHEST IMPACT

**Problem**: Full 262K vocab gemv takes 221ms (41% of total time)

**Solution**: Pre-compute top-K vocabulary from vindex down_meta

**Implementation**:
```rust
// crates/larql-inference/src/forward.rs
pub fn logits_from_vindex_meta(
    residual: &Array1<f32>,
    index: &VectorIndex,
    top_k: usize,
) -> Vec<(usize, f32)> {
    // 1. Get candidate tokens from vindex metadata (active features only)
    // 2. Compute logits only for candidates (~500 tokens vs 262K)
    // 3. Return top-K from candidates
}
```

**Expected Gain**: 221ms → ~20ms (200ms saved, ~39% faster total)

**Files**:
- `crates/larql-inference/src/forward.rs` - Add logits_from_vindex_meta
- `crates/larql-vindex/src/index/core.rs` - Expose down_meta token API
- `crates/larql-lql/src/executor/query.rs` - Use in INFER executor

**Validation**: Benchmark on standard prompts, verify top-5 predictions match

---

### 1.2 Walk-Only Mode 💾 MEMORY OPTIMIZATION

**Problem**: Loading 16.6GB model when walk only needs 3.5GB

**Solution**: Skip FFN weights, load only attention + embeddings + norms

**Implementation**:
```rust
// crates/larql-inference/src/models.rs
pub struct InferenceModel {
    // ... existing fields
    pub load_mode: LoadMode,
}

pub enum LoadMode {
    Full,           // All weights (current default)
    WalkOnly,       // Skip W_gate, W_up, W_down (requires down_features.bin)
    AttentionOnly,  // Skip all FFN (future: template cache)
}

impl InferenceModel {
    pub fn load_walk_only(model_id: &str, vindex_path: &Path) -> Result<Self> {
        // Load only: embeddings, attn_*, norms, lm_head
        // Verify vindex has down_features.bin + up_features.bin
        // Return model with load_mode = WalkOnly
    }
}
```

**Expected Gain**: 13GB RAM savings, faster startup

**Files**:
- `crates/larql-inference/src/models.rs` - Add LoadMode, load_walk_only
- `crates/larql-cli/src/commands/query/infer.rs` - Add --walk-only flag
- `crates/larql-server/src/main.rs` - Add walk-only server mode

**Validation**: Run inference, verify identical predictions, check RSS

---

### 1.3 Document Multi-Layer INSERT Tuning 📝

**Problem**: INSERT alpha/layer tuning is undocumented

**Solution**: Create tuning guide with empirical data

**Content** (in `docs/insert-tuning-guide.md`):
```markdown
# Knowledge Insertion Tuning Guide

## Alpha/Layer Sweet Spots

| Config | New Fact | Neighbor Degradation | Use Case |
|--------|----------|---------------------|----------|
| 8L × 0.25 | 94.6% | -20.0 pts | Maximum confidence |
| 16L × 0.12 | 78.4% | -13.7 pts | Balanced (recommended) |
| 20L × 0.08 | 26.0% | -7.0 pts | Minimal disruption |

## Residual Capture

Always use `infer_trace()` not `embed()`:
- embed("Atlantis"): norm=51, cos=0.01 with L24 residual (orthogonal!)
- residual[L24]: norm=38,319, actual query vector for gate KNN

## Layer Selection

Knowledge band (L20-L27) for factual relations.
Syntax band (L0-13) for morphological/grammar.
```

**Files**:
- `docs/insert-tuning-guide.md` - New comprehensive guide
- `docs/training-free-insert.md` - Link to tuning guide

---

## Phase 2: Quantization (Months 2-3)

### 2.1 Q4_K_M Attention Weights 🎯

**Target**: Q/K/V/O projections only (FFN already solved by walk)

**Method**: 4-bit quantization with 6-bit scales per 32-value block

**Expected**:
- 4× bandwidth reduction
- Attention: 136ms → ~41ms
- Minimal accuracy loss (< 0.5% on benchmarks)

**Files**:
- `crates/larql-models/src/quantize/` - New module
- `crates/larql-inference/src/attention.rs` - Q4_K_M matmul path
- `crates/larql-vindex/src/extract/` - Quantized extraction option

---

### 2.2 Q4_K_M Embeddings & Logits

**Target**: 2.7GB lm_head matrix (262K vocab)

**Expected**:
- Logits: 20ms → ~7ms (if not using vindex meta optimization)
- Combined with Phase 1.1: Redundant (already fast)

**Decision**: Skip if Phase 1.1 succeeds, otherwise implement

---

## Phase 3: Template Engine (Months 4-6)

### 3.1 Attention Pattern Cache 🚀 GAME CHANGER

**Insight**: "99% of heads are fixed" (from findings.md)

**Method**:
```rust
pub struct TemplateCache {
    // Pattern: "The capital of {X} is"
    template_patterns: HashMap<String, CachedAttention>,
}

pub struct CachedAttention {
    // Per-layer, per-head attention weights for template tokens
    // Only entity token needs fresh attention
    fixed_patterns: Vec<Array2<f32>>,  // [layer][head] → weights
    entity_slot: usize,  // Where {X} appears
}
```

**Expected**:
- Attention: 84ms → ~5ms (only entity-specific heads)
- Template hit rate: >90% on factual queries

**Files**:
- `crates/larql-inference/src/template_cache.rs` - New module
- `crates/larql-lql/src/executor/query.rs` - Use cache in DESCRIBE/WALK

---

### 3.2 Precompiled Routing Graphs

**Method**: Static analysis of attention heads to build routing DAG

**Expected**: Sub-10ms factual queries (graph walk only, no matmuls)

---

## Phase 4: Ecosystem & Research (Ongoing)

### 4.1 HuggingFace Vindexfile Resolution

**Status**: TODO at `crates/larql-vindex/src/vindexfile/mod.rs:162`

**Blocker**: Remote patch downloading, version resolution

**Priority**: High (enables distributed ecosystem)

---

### 4.2 WASM Compute Engine (Experiment 07)

**Goal**: Deterministic solvers for arithmetic/algebra

**Status**: In progress (token-level Python solvers working)

**Next**: Residual-level dispatch, then WASM runtime in Rust

---

### 4.3 Manifold Compression (Experiment 02)

**Hypothesis**: Knowledge manifold has ~15 true dimensions

**Test**: SVD of L14-L27 gate vectors, check variance

**Potential**: 71GB → 416MB (170× compression)

**Status**: Needs validation

---

## Optimization Sequence Timeline

```
Week 1-2:   Logits from vindex (Phase 1.1)
Week 3:     Walk-only mode (Phase 1.2)
Week 4:     INSERT tuning docs (Phase 1.3)
─────────────────────────────────────────
Expected:   ~300ms inference, 3.5GB RAM

Month 2:    Q4_K_M attention (Phase 2.1)
Month 3:    Benchmark & validate
─────────────────────────────────────────
Expected:   ~200ms inference

Month 4-5:  Template cache (Phase 3.1)
Month 6:    Precompiled routes (Phase 3.2)
─────────────────────────────────────────
Expected:   ~40-50ms inference

Ongoing:    HF ecosystem, WASM, experiments
```

---

## Critical Architectural Invariants

These must **never** be violated:

### 1. Base Vindexes Are Immutable

All mutation flows through `PatchedVindex` overlay. Base files never modified.

### 2. Storage Is Mmap-First

Gate vectors, embeddings, down weights are zero-copy mmap'd. No full tensor loads.

### 3. Three Extraction Levels, Not Features

`browse` (~3 GB), `inference` (~6 GB), `all` (~10 GB) gated by `ExtractLevel` enum.

### 4. Walk FFN Sparse-by-Design

Gate KNN (K≈10) skips most features. Preserve this for speed.

### 5. Local Knowledge Stays Local

Client-side patches never transmitted. Proprietary knowledge privacy guaranteed.

---

## Key Files by Optimization Area

### Performance (Phase 1-3)
```
crates/larql-inference/src/forward.rs           ← Logits optimization
crates/larql-inference/src/vindex/walk_ffn.rs   ← Walk-only mode
crates/larql-inference/src/attention.rs         ← Template cache
crates/larql-inference/src/template_cache.rs    ← New: cache impl
crates/larql-models/src/quantize/               ← New: Q4_K_M
```

### Knowledge (Phase 1.3, 4.2)
```
crates/larql-lql/src/executor/mutation.rs       ← INSERT tuning
knowledge/scripts/probe_mlx.py                  ← Automated labeling
docs/insert-tuning-guide.md                     ← New: tuning docs
```

### Ecosystem (Phase 4.1)
```
crates/larql-vindex/src/vindexfile/mod.rs       ← HF resolution
crates/larql-server/src/main.rs                 ← Distributed serving
```

---

## Success Metrics

Track these after each phase:

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Target |
|--------|----------|---------|---------|---------|--------|
| **Inference latency** | 517ms | ~300ms | ~200ms | ~50ms | <100ms |
| **Memory footprint** | 16.6GB | 3.5GB | 3.5GB | 3.5GB | <5GB |
| **INSERT accuracy** | 94.6% | 94.6% | 94.6% | 94.6% | >90% |
| **Neighbor degradation** | -20pts | -14pts | -14pts | <-10pts | <-15pts |
| **Vindex size (f16)** | 3-10GB | 3-10GB | 2-6GB | 2-6GB | <5GB |

---

## References

- [Walk Boundary Sweep](walk-boundary-sweep.md) - Correctness proof
- [FFN Graph Layer](ffn-graph-layer.md) - Walk architecture
- [Training-Free Insert](training-free-insert.md) - Constellation method
- [Inference Engine](inference-engine.md) - BLAS-fused attention
- [Findings](findings.md) - Research discoveries

---

## License

Apache-2.0
