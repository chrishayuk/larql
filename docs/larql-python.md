# larql-python — Python Bindings for Vindex

**Class: ARCHIVE.** The draft API below is retained as a design record. Current exports and source-build instructions are in the [Python crate README](../crates/larql-python/README.md); [runtime interfaces](runtime-surfaces.md) explain its V2/V3 limits.

**Version:** 0.1  
**Date:** 2026-04-01  
**Status:** Draft  
**Depends on:** larql-vindex (Rust crate)

---

## 1. Purpose

Python bindings for larql-vindex, enabling vindex operations from Python, MLX, and chuk-lazarus without going through the CLI or REPL. A native Python library, not a subprocess wrapper.

```python
import larql

vindex = larql.load("gemma3-4b.vindex")
edges = vindex.describe("France")
# [Edge(relation="capital", target="Paris", confidence=0.97, source="probe"), ...]
```

---

## 2. Core API

### 2.1 Loading

```python
import larql

# Load from local path
vindex = larql.load("gemma3-4b.vindex")

# Load from HuggingFace (lazy download)
vindex = larql.load("hf://chrishayuk/gemma-3-4b-it-vindex")

# Load browse-only (gate + embed + down_meta only)
vindex = larql.load("gemma3-4b.vindex", level="browse")

# Load with inference weights
vindex = larql.load("gemma3-4b.vindex", level="inference")

# Connect to remote server (no download)
vindex = larql.remote("https://vindex.larql.dev/gemma-3-4b-it")
```

### 2.2 Knowledge Queries

```python
# Describe — all knowledge edges for an entity
edges = vindex.describe("France")
# Returns: List[Edge]
# Edge(relation="capital", target="Paris", gate_score=1436.9, 
#      layer=27, source="probe", also=["Berlin", "Tokyo"])

# Describe with options
edges = vindex.describe("France", band="knowledge")   # L14-27 only
edges = vindex.describe("def", band="syntax")          # L0-13 only
edges = vindex.describe("France", band="all")           # All layers
edges = vindex.describe("France", verbose=True)         # Include TF-IDF labels

# Walk — gate KNN for a prompt
hits = vindex.walk("The capital of France is", top=10, layers=range(24, 34))
# Returns: List[WalkHit]
# WalkHit(layer=27, feature=9515, gate_score=1436.9, target="Paris")

# Select — query edges
results = vindex.select(relation="capital", limit=10)
results = vindex.select(entity="France", min_confidence=0.5)
results = vindex.select(relation="occupation", target_like="composer")

# Relations — list all known relation types
relations = vindex.relations()
# Returns: List[Relation]
# Relation(name="capital", count=94, source="probe", layers=[24,25,26,27])

# Stats
stats = vindex.stats()
# Stats(model="google/gemma-3-4b-it", layers=34, features=348160, 
#       probe_confirmed=1967, cluster_labelled=103)
```

### 2.3 Feature Access

```python
# Direct feature lookup
meta = vindex.feature(layer=27, feature=9515)
# FeatureMeta(top_token="Paris", c_score=5.1, 
#             top_k=[("Paris", 5.1), ("Berlin", 3.2), ...])

# Gate vector access (raw numpy array)
gate_vec = vindex.gate_vector(layer=27, feature=9515)
# numpy.ndarray, shape (2560,), dtype float32

# Embedding lookup
embed = vindex.embed("France")
# numpy.ndarray, shape (2560,), dtype float32

# Multi-token embedding (averaged)
embed = vindex.embed("John Coyle")
# Averages token embeddings for "John" and "Coyle"

# Gate KNN — raw scores
scores = vindex.gate_knn(layer=27, query_vector=embed, top_k=20)
# List[(feature_id, score)]
```

### 2.4 Mutation

```python
# Insert a fact
vindex.insert("John Coyle", "located_in", "Colchester")
vindex.insert("John Coyle", "occupation", "engineer", layer=26, confidence=0.8)

# Delete
vindex.delete(entity="John Coyle", relation="located_in")

# Update
vindex.update(entity="John Coyle", relation="located_in", target="London")

# Check if edge exists
exists = vindex.has_edge("France", "capital")
target = vindex.get_target("France", "capital")  # "Paris"
```

### 2.5 Patches

```python
# Create a patch
patch = vindex.begin_patch("medical.vlp")
vindex.insert("aspirin", "side_effect", "bleeding")
vindex.insert("aspirin", "treats", "headache")
vindex.save_patch()

# Apply patches
vindex.apply_patch("medical.vlp")
vindex.apply_patch("hf://medical-ai/drug-interactions@2.1.0")

# List patches
patches = vindex.patches()
# [Patch(name="medical.vlp", operations=2), ...]

# Remove patch
vindex.remove_patch("medical.vlp")

# Bake down
vindex.compile_vindex("gemma3-4b-medical.vindex")
```

### 2.6 Compile

```python
# Compile to safetensors
vindex.compile("edited-model/", format="safetensors")

# Compile to GGUF
vindex.compile("model.gguf", format="gguf", quant="Q4_K_M")

# Compile to MLX
vindex.compile("mlx-model/", format="mlx")
```

---

## 2.5 Residual Stream Trace

Capture a decomposed forward pass — attn/FFN deltas at every layer, queryable.

```python
import larql

wm = larql.WalkModel("gemma3-4b.vindex")

# Capture trace (last token position by default)
t = wm.trace("The capital of France is")
# ResidualTrace('The capital of France is', 6 tokens, 34 layers, 35 nodes)

# Answer trajectory — track a token through all layers
traj = t.answer_trajectory("Paris")
for w in traj:
    print(f"L{w.layer}: rank={w.rank}, prob={w.prob:.3f}, attn={w.attn_logit:.1f}, ffn={w.ffn_logit:.1f}")

# Inspect any layer
t.top_k(24)              # top-5 predictions at L24
t.rank_of("Paris", 23)   # rank of Paris at L23
t.residual(24)            # raw residual vector at L24
t.attn_delta(24)          # what attention added at L24
t.ffn_delta(24)           # post-attention contribution at L24
t.summary()               # per-layer compact summary

# Multi-position trace (all token positions)
t = wm.trace("The capital of France is", positions="all")
t.residual(24, position=4)  # France's residual at L24

# Save / load (mmap'd, zero-copy)
t.save("trace.bin")
from larql._native import TraceStore
store = TraceStore("trace.bin")
store.residual(0, 25)  # token 0, layer 25

# Boundary store (production context, 10 KB/window)
from larql._native import BoundaryWriter, BoundaryStore
writer = BoundaryWriter("ctx.bndx", hidden_size=2560, window_size=200)
writer.append(token_offset=0, window_tokens=200, residual=vec)
writer.finish()
store = BoundaryStore("ctx.bndx")
store.residual(0)  # zero-copy from mmap
```

See [residual-trace.md](residual-trace.md) for architecture details and tiered context storage.

---

## 3. MLX Integration

### 3.1 Three Inference Modes

All use full weights. No feature dropping. Correct output.

```python
import mlx_lm

# 1. Dense — all weights in GPU memory. Fast. For models that fit.
import larql
model, tokenizer = larql.mlx.load("gemma3-4b.vindex")
response = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)

# 2. Streaming — mmap'd weights, Metal pages from SSD on demand.
#    For models that don't fit in GPU memory. 120B on 8GB MacBook.
from larql.streaming import load
model, tokenizer = load("gpt-oss-120b.vindex")
response = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)

# 3. Walk FFN — FFN in Rust, vindex as editable knowledge layer.
#    INSERT/DELETE/UPDATE mutations reflected in inference. Slower.
from larql.walk_ffn import load
model, tokenizer = load("gemma3-4b.vindex")
response = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)
```

| Mode | Load | Speed | Memory | Editable |
|------|------|-------|--------|----------|
| Dense (`larql.mlx`) | Slow (eval all) | Fast | Full model in GPU | No |
| Streaming (`larql.streaming`) | Fast (lazy) | Same* | ~1 layer at a time | No |
| Walk FFN (`larql.walk_ffn`) | Fast | Slow (CPU FFN) | Attention only | Yes |

\* Streaming matches dense speed when the OS page cache keeps weights hot.
For models that exceed physical memory, speed is SSD-bound (~1.5s/token at 3 GB/s NVMe).

### 3.2 Knowledge Queries (No Inference)

Skip inference entirely when a knowledge lookup suffices.

```python
import larql

vindex = larql.load("gemma3-4b.vindex")

# Knowledge query — no inference needed (0ms)
edges = vindex.describe("France")
capital = next(e.target for e in edges if e.relation == "capital")
# "Paris"
```

### 3.3 Walk FFN with Editable Knowledge

Walk FFN uses the vindex for feature selection. INSERT/DELETE/UPDATE mutations
are reflected in inference output.

```python
import larql
import mlx_lm
from larql.walk_ffn import load

model, tokenizer = load("gemma3-4b.vindex")

# Insert a fact into the vindex
vindex = larql.load("gemma3-4b.vindex")
vindex.insert("aspirin", "side_effect", "bleeding")

# Walk FFN inference uses the mutated vindex
response = mlx_lm.generate(model, tokenizer, prompt="The side effects of aspirin include", max_tokens=20)
```

### 3.4 Residual Capture for Probing

Use `WalkModel.capture_residuals` — no MLX required. Residuals come back
as numpy arrays directly from the Rust forward pass.

```python
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
vindex = larql.load_vindex("gemma3-4b.vindex")

# Capture last-token residual at every layer in one call.
all_layers = list(range(wm.num_layers))
residuals = wm.capture_residuals("The capital of France is", layers=all_layers)
# residuals: {0: np.ndarray(hidden,), 1: ..., ...}

# Feed each residual to vindex for analysis.
for layer, residual in residuals.items():
    hits = vindex.gate_knn(layer, residual, top_k=5)
    for feat, score in hits:
        meta = vindex.feature(layer, feat)
        label = vindex.feature_label(layer, feat)
        print(f"  L{layer} F{feat} gate={score:.1f} → {meta.top_token} ({label})")
```

For ablation, steering, activation patching, logit lens, embedding
neighbors, raw DLA, KV-cache surgery, and multi-token generation under
hooks, see [docs/mech-interp.md](mech-interp.md).

---

## 4. chuk-lazarus Integration

### 4.1 Replace Ad-Hoc Storage

chuk-lazarus currently stores knowledge data in separate JSON files, numpy arrays, and ad-hoc formats. Vindex consolidates all of these.

```python
# BEFORE (chuk-lazarus current):
import json
import numpy as np

# Load knowledge from scattered files
with open("knowledge/entities.json") as f:
    entities = json.load(f)
residuals = np.load("residuals/france_l26.npy")
with open("knowledge/relations.json") as f:
    relations = json.load(f)

# AFTER (chuk-lazarus with vindex):
import larql

vindex = larql.load("gemma3-4b.vindex")
edges = vindex.describe("France")
residual = vindex.walk("France", layers=[26])[0].gate_vector
relations = vindex.relations()
```

### 4.2 The Map (01_the_map.py)

The three-pass swap demo — normal → L0 swap → L26 swap — reads knowledge coordinates from the vindex.

```python
import larql

vindex = larql.load("gemma3-4b.vindex")

# Read the knowledge map for an entity
france_edges = vindex.describe("France")
germany_edges = vindex.describe("Germany")

# Get gate vectors for specific features
capital_feature_france = vindex.gate_vector(layer=27, feature=9515)
capital_feature_germany = vindex.gate_vector(layer=27, feature=9515)

# The map is the vindex. No separate data structure needed.
# Each edge IS a coordinate in the knowledge manifold.

# Swap test: replace France's gate vector with Germany's
# and verify the model now predicts Berlin instead of Paris
```

### 4.3 The Injection (02_the_injection.py)

The four-pass injection demo — full context → blank → residual → 12 bytes — maps to INSERT.

```python
import larql

vindex = larql.load("gemma3-4b.vindex")

# The injection that was 24 bytes is now:
vindex.insert("John Coyle", "lives-in", "Colchester")
# Internally: synthesises gate vector from entity embedding + relation cluster centre
# Writes to the vindex. Same operation, proper storage.

# Verify
edges = vindex.describe("John Coyle")
assert any(e.target == "Colchester" for e in edges)

# The "99.8% of the residual is scaffolding" insight
# is captured by the gate vector: 2560 floats that encode one fact.
# The down vector writes the target. Together: one edge in the graph.
```

### 4.4 Relation Steering

The steering work (8/8 correct) used cluster centres to shift predictions. Vindex provides these directly.

```python
import larql
import numpy as np

vindex = larql.load("gemma3-4b.vindex")

# Get the cluster centre for a relation
capital_direction = vindex.cluster_centre("capital")
language_direction = vindex.cluster_centre("language")

# Steer: shift residual toward capital direction
residual = capture_residual("France is known for its")
steered_capital = residual + 2.0 * capital_direction
steered_language = residual + 2.0 * language_direction

# Walk with steered residuals
capital_hits = vindex.gate_knn(layer=26, query_vector=steered_capital, top_k=5)
language_hits = vindex.gate_knn(layer=26, query_vector=steered_language, top_k=5)
# capital_hits → Paris, Berlin, Tokyo
# language_hits → French, German, Japanese
```

### 4.5 Cross-Model Comparison

The Procrustes alignment work (0.946 cosine across models) becomes a vindex operation.

```python
import larql

gemma = larql.load("gemma3-4b.vindex")
llama = larql.load("llama3-8b.vindex")

# What does Llama know that Gemma doesn't?
gemma_france = set(e.relation for e in gemma.describe("France"))
llama_france = set(e.relation for e in llama.describe("France"))
llama_only = llama_france - gemma_france
# {"population", "anthem", "motto", ...}

# Compare relation coverage
gemma_rels = {r.name: r.count for r in gemma.relations()}
llama_rels = {r.name: r.count for r in llama.relations()}
for rel in sorted(set(gemma_rels) | set(llama_rels)):
    g = gemma_rels.get(rel, 0)
    l = llama_rels.get(rel, 0)
    if g != l:
        print(f"  {rel}: gemma={g}, llama={l}")
```

### 4.6 Navigation Map Visualiser

The nav_map.html visualiser — real-time residual stream trajectory — reads from the vindex.

```python
import larql
import json

vindex = larql.load("gemma3-4b.vindex")

# Generate data for the visualiser
def export_nav_map(entity, output_path):
    hits_by_layer = {}
    for layer in range(34):
        walk_hits = vindex.walk(entity, layers=[layer], top=5)
        hits_by_layer[layer] = [
            {
                "feature": h.feature,
                "gate_score": h.gate_score,
                "target": h.target,
                "relation": vindex.feature_label(layer, h.feature),
            }
            for h in walk_hits
        ]
    
    with open(output_path, "w") as f:
        json.dump(hits_by_layer, f)

export_nav_map("France", "nav_data/france.json")
# nav_map.html reads this and renders the trajectory
```

---

## 5. Implementation

### 5.1 Rust → Python Bridge

Use PyO3 for Rust → Python bindings. The Python library calls directly into larql-vindex Rust code — no subprocess, no CLI parsing, no serialisation overhead for gate vectors.

```
larql-python/
  Cargo.toml          # PyO3 dependency
  src/
    lib.rs            # #[pymodule] larql
    vindex.rs         # PyVindex wrapper around VectorIndex
    edge.rs           # PyEdge, PyWalkHit, PyFeatureMeta
    patch.rs          # PyPatch wrapper
  python/
    larql/
      __init__.py     # Import from native module
      types.py        # Type hints
```

```rust
// src/lib.rs
use pyo3::prelude::*;

#[pymodule]
fn larql(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(remote, m)?)?;
    m.add_class::<PyVindex>()?;
    m.add_class::<PyEdge>()?;
    Ok(())
}

#[pyfunction]
fn load(path: &str, level: Option<&str>) -> PyResult<PyVindex> {
    // Calls larql_vindex::load()
}
```

### 5.2 Numpy Interop

Gate vectors and embeddings return as numpy arrays without copying — use PyO3's numpy integration to share memory between Rust and Python.

```rust
use numpy::{PyArray1, IntoPyArray};

#[pymethods]
impl PyVindex {
    fn gate_vector(&self, py: Python, layer: usize, feature: usize) -> PyResult<Py<PyArray1<f32>>> {
        let vec = self.inner.gate_vector(layer, feature)?;
        Ok(vec.into_pyarray(py).to_owned())
    }
    
    fn embed(&self, py: Python, entity: &str) -> PyResult<Py<PyArray1<f32>>> {
        let vec = self.inner.embed_entity(entity)?;
        Ok(vec.into_pyarray(py).to_owned())
    }
}
```

### 5.3 Installation

```bash
# From PyPI
pip install larql

# From source — managed via uv
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release

# Verify
uv run --no-sync python -c "import larql; print(larql.load('test.vindex').stats())"
```

---

## 6. Package Structure

```
larql-python/
  Cargo.toml
  pyproject.toml
  src/
    lib.rs
    vindex.rs
    edge.rs
    patch.rs
    compile.rs
  python/
    larql/
      __init__.py
      mlx.py          # Dense MLX loading from vindex
      streaming.py    # Streaming mmap'd inference (large models)
      walk_ffn.py     # Walk FFN: Rust FFN + MLX attention
      types.py        # Dataclass definitions for type hints
      _native.pyi     # Stub file for IDE support
  tests/
    test_load.py
    test_describe.py
    test_walk.py
    test_patch.py
    test_mlx.py
    test_lazarus.py
  examples/
    basic_describe.py
    mlx_hybrid.py
    lazarus_map.py
    lazarus_injection.py
    cross_model_diff.py
    remote_query.py
```
