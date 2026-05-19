# Mechanistic interpretability surface

LARQL exposes a programmatic forward-hook system plus the standard
mech-interp primitives — capture, ablation, steering, activation
patching, full logit lens, embedding-neighbor lookups, raw DLA, and
KV-cache surgery. All of it works on real models and on synthetic
weights, with **zero overhead when no hook is registered**.

This is the surface lazarus-style MCP servers (e.g. `chuk-mcp-lazarus`)
build on top of.

---

## The hook trait

Five callbacks fire inside `forward::trace_forward_full_hooked` and
`forward::generate_cached_hooked`. Two of them take `&mut Array2<f32>` so
the hook can mutate the residual in place:

```text
pre_layer
   │
   ▼ on_pre_layer(layer, &h)
attention
   │
   ▼ on_attention_weights(layer, &w)        // capture_attention=true
   │ on_post_attention(layer, &mut h)       // ← intervention point
FFN
   │
   ▼ on_ffn_activation(layer, &gate)        // capture_activations=true
PLE + scalar
   │
   ▼ on_post_layer(layer, &mut h)           // ← intervention point
```

Implement [`forward::LayerHook`] for any custom transform; defaults are
no-ops so impls override only what they need. The two `&mut`
callbacks unlock the entire intervention surface — ablation, steering,
patching, and subspace surgery are all just `LayerHook` impls over
those points.

### Built-in hooks

| Hook | Purpose |
|------|---------|
| `NoopHook` | Default, never fires. Zero-cost when no real hook is registered. |
| `RecordHook::for_layers([L,…])` | Capture pre-layer / post-attention / post-layer / attention-weights / FFN-activation at the listed layers. |
| `ZeroAblateHook::for_layers([L,…])` | Zero the post-layer residual at the listed layers (full row or specific positions). |
| `SteerHook::new().add(L, vec, α)` | Add `α·v` to the last-token row at layer `L` post-layer. |
| `CompositeHook::new(vec![&mut a, &mut b, …])` | Run multiple hooks in order. |

---

## Rust API

```rust
use larql_inference::forward::{
    RecordHook, SteerHook, ZeroAblateHook,
    trace_forward_full_hooked, generate_cached_hooked,
    capture_donor_state, patch_and_trace,
    logit_lens_topk, track_token, track_race,
    embedding_neighbors, project_through_unembed,
    embedding_row, embedding_row_scaled, unembedding_row,
};
use larql_inference::ffn::WeightFfn;

let ffn = WeightFfn { weights: &weights };

// 1. Capture residuals at chosen layers.
let mut record = RecordHook::for_layers([12, 18, 24]);
let _ = trace_forward_full_hooked(
    &weights, &tokens,
    /*capture_layers=*/ &[12, 18, 24],
    /*capture_activations=*/ false, /*activation_top_k=*/ 0,
    /*capture_attention=*/ false,
    &ffn, &mut record,
);
let residual_at_18 = record.post_layer.get(&18).unwrap();

// 2. Logit lens: top-k tokens at any layer (norm + lm_head + softmax).
let top_k     = logit_lens_topk(&weights, residual_at_18.row(0).as_slice().unwrap(), 5);
let p_paris   = track_token(&weights, residual_at_18.row(0).as_slice().unwrap(), /*paris_id=*/ 1234);

// 3. Embedding-space neighbors + raw DLA.
let neighbors = embedding_neighbors(&weights, &query_vec, 10);   // cosine vs W_E
let dla       = project_through_unembed(&weights, &head_out, 10);// raw lm_head @ vec, no norm

// 4. Ablate or steer mid-forward.
let mut ablate = ZeroAblateHook::for_layers([14usize]);
let mut steer  = SteerHook::new().add(20, steer_vec, 0.5);

// 5. Activation patching: donor → recipient at chosen (layer, position) coords.
let donor   = capture_donor_state(&weights, &donor_tokens, &[(10, 4)]);
let patched = patch_and_trace(&weights, &recipient_tokens, &donor, &[28]);

// 6. Multi-token generation with hooks active on every layer of every step.
let ids = generate_cached_hooked(
    &weights, &tokenizer, &ffn, &prompt_ids,
    /*max_new_tokens=*/ 32,
    /*window=*/ None, /*backend=*/ None,
    &mut steer,
    |id, text| print!("{text}"),
);
```

KV-cache surgery (lazarus's `prefill_inject` / `kv_inject_test`):

```rust
use larql_inference::attention::KvCache;

let mut recipient_cache = KvCache::with_layers(num_layers);
let donor_cache: KvCache = /* built elsewhere */;

// Lift one entire layer of K/V from donor into recipient.
recipient_cache.clone_layer_from(&donor_cache, /*layer=*/ 12);

// Or slice a position range.
recipient_cache.clone_layer_position_range(&donor_cache, 12, /*start=*/ 0, /*end=*/ 64);
```

---

## Python API (`larql._native.WalkModel`)

Returned tensors are numpy arrays. All the methods below take a
prompt string (tokenized internally with the model's tokenizer):

| Method | What it does |
|--------|--------------|
| `capture_residuals(prompt, layers) -> {layer: np.ndarray}` | Last-token residual at each layer |
| `forward_with_capture(prompt, layers) -> {layer: (seq, hidden)}` | Full per-position residual matrix |
| `forward_ablate(prompt, ablate_layers, capture_layers) -> dict` | Zero-ablate then capture last-token residuals |
| `forward_steer(prompt, [(layer, vec, α), …], capture_layers) -> dict` | Steer then capture |
| `patch_activations(donor, recipient, [(layer, pos), …], capture_layers)` | Cross-prompt residual patching |
| `logit_lens(residual, k=10) -> [(token_id, prob)]` | Top-k vocab through final norm + lm_head |
| `track_token_at(residual, token_id) -> float` | Probability of a specific token |
| `track_race({layer: residual}, k=5) -> {layer: [(id, prob)]}` | Top-k per layer for several layers |
| `embedding_neighbors(query, k=10) -> [(token_id, cosine)]` | Vocab tokens nearest a vector under cosine vs W_E |
| `project_through_unembed(vec, k=10) -> [(token_id, logit)]` | Raw `W_U @ vec` (no norm/softcap) — DLA |
| `embedding_for(token_id, scaled=True) -> np.ndarray` | Row of W_E (with or without `embed_scale`) |
| `unembedding_for(token_id) -> np.ndarray` | Row of W_U |
| `generate_with_hooks(prompt, max_new_tokens, ablate_layers=None, steers=None) -> (text, ids)` | Multi-token generation with hooks active every step |

```python
import larql

wm = larql.WalkModel("gemma3-4b.vindex")

# Capture residuals at three layers, get numpy arrays back.
residuals = wm.capture_residuals("The capital of France is", layers=[12, 18, 24])
# {12: ndarray(hidden,), 18: ndarray(hidden,), 24: ndarray(hidden,)}

# Logit lens at L24.
top5 = wm.logit_lens(residuals[24], k=5)
# [(token_id, prob), ...]

# Steer the answer toward a different concept (multi-token generation).
direction = ...  # numpy float32 array of shape (hidden,)
text, ids = wm.generate_with_hooks(
    "The capital of France is",
    max_new_tokens=10,
    steers=[(20, direction, 1.5)],
)
```

---

## Backend split: hooks-on-CPU, Metal-stays-fast

- **Hooks during single-forward** (`trace_forward_full_hooked` and the
  capture/ablate/steer/patch wrappers above) are zero-cost when no hook
  is registered and run on the existing CPU forward path.
- **Hooks during multi-token generation** (`generate_cached_hooked` /
  `WalkModel.generate_with_hooks`) use the **CPU KV-cache path**. The
  Metal-fast `predict` is hook-free **by design** — the kernel pipeline
  is fused; threading hooks through it would split the fast path even
  when no hook is registered. Mech-interp tools want correctness over
  throughput, so the CPU-when-hooks-active trade is the right one.

`on_attention_weights` and `on_ffn_activation` callbacks **do not fire**
on the multi-token generation path — the production decode kernels don't
capture those intermediates. Use `trace_forward_full_hooked` for a
single forward pass when you need them.

---

## End-to-end demo

```bash
cargo run --release -p larql-inference --example mech_interp_demo
```

Walks through all seven primitives on synthetic weights (no vindex
required). Source: `crates/larql-inference/examples/mech_interp_demo.rs`.

---

## Design + roadmap

The hook system landed across milestones M1–M8. Per-item file paths and
design rationale: `crates/larql-inference/ROADMAP.md` § "P0:
Mechanistic hooks (lazarus parity)".

The next roadmap item is k-quant / vindex-backed research intervention:
promote the reusable OV/RD plumbing into `larql-inference` so
experiments can share k-quant per-layer tensor insertion, hooked
k-quant forward passes, and stable trace/export contracts while
keeping PQ variants and address probes in the dev harness.

Current engine surface:
`larql_inference::vindex::insert_q4k_layer_tensors` for scoped
per-layer dense tensor materialization (the `q4k` in the name is
historical — the function handles the full k-quant family), and
`larql_inference::vindex::predict_kquant_hidden_hooked` for dense-FFN
k-quant hidden-state forward passes with `LayerHook` callbacks.
Pre-W_O experiments can use
`larql_inference::forward::run_layer_with_mapped_pre_o_head` at layer
scope or
`larql_inference::vindex::predict_kquant_hidden_with_mapped_pre_o_head`
for a full k-quant forward pass with one mapped head.
