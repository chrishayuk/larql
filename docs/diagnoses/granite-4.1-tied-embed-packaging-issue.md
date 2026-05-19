<!--
Draft GitHub issue text for an ibm-granite tracker (or a Hugging Face
"Community" discussion on the model repo). Body assumes a markdown-aware
renderer. The full diagnosis with measurement context lives in
docs/diagnoses/granite-4.1-tied-embed-packaging-bug.md — this file is the
copy-pasteable short form.

Suggested target:
- https://huggingface.co/ibm-granite/granite-4.1-8b → "Community" tab → New discussion
  (and the same on -30b / each -base variant if confirmed)
- or whichever public IBM Granite issue tracker is canonical
-->

# `granite-4.1-8b` ships `lm_head.weight` alongside `tie_word_embeddings: true` — breaks MLX, wastes ~822 MB

## Summary

`ibm-granite/granite-4.1-8b` declares `"tie_word_embeddings": true` in
`config.json` but **also** maps an explicit `lm_head.weight` tensor in
`model.safetensors.index.json`. The two declarations contradict each
other: tied embeddings, by definition, mean the output projection
shares weights with the input embedding — there should be no separate
`lm_head` tensor on disk.

The duplicated tensor is **bit-identical** to `model.embed_tokens.weight`
(verified with `torch.equal`), so it's ~822 MB (`100352 × 4096 × bf16`)
of redundant data shipped per model copy. The 3B sibling
(`ibm-granite/granite-4.1-3b`) is packaged correctly — it has no
`lm_head.weight` in the safetensors index. So this appears to be a
regression in the 8B export, not an intentional design.

## Consumer-side impact

| Consumer | Behaviour |
|---|---|
| `transformers.AutoModelForCausalLM` | Loads both, silently overwrites the tied head with the redundant `lm_head.weight`. Forward correct; ~10 s extra load time and ~822 MB extra peak host RAM. |
| `mlx_lm` (Apple MLX) | **Fails to load.** `mlx.nn.Module.load_weights(..., strict=True)` raises `ValueError: Received 1 parameters not in model: lm_head.weight`. The MLX Granite implementation correctly sets up a tied head from the config flag, so there's no slot for the redundant tensor. |
| Third-party loaders that key off `tie_word_embeddings` | Behaviour depends on whether they pre-emptively allocate the tied head or look for `lm_head.weight` first. Either way they encounter an inconsistency. |

## Reproduction

```python
import json, torch
from pathlib import Path
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM
import mlx_lm

REPO = "ibm-granite/granite-4.1-8b"
SNAP = Path("~/.cache/huggingface/hub/...").expanduser()  # snapshot dir

# 1. config says tied embeddings:
cfg = json.loads((SNAP / "config.json").read_text())
print(f"tie_word_embeddings = {cfg['tie_word_embeddings']}")  # True

# 2. but the safetensors index has an explicit lm_head.weight:
idx = json.loads((SNAP / "model.safetensors.index.json").read_text())
print(f"lm_head.weight shard: {idx['weight_map'].get('lm_head.weight')}")
# → "model-00001-of-00004.safetensors"

# 3. and that tensor is bit-identical to embed_tokens.weight:
with safe_open(SNAP / "model-00001-of-00004.safetensors", framework="pt") as f:
    lm = f.get_tensor("lm_head.weight")
    em = f.get_tensor("model.embed_tokens.weight")
print(f"bit-identical? {torch.equal(lm, em)}")  # True

# 4. mlx_lm refuses to load:
mlx_lm.load(REPO)
# → ValueError: Received 1 parameters not in model: lm_head.weight
```

Compare to the 3B, where `model.safetensors.index.json` has no
`lm_head.weight` mapping:

```bash
$ grep -c '"lm_head' .../granite-4.1-3b/.../model.safetensors.index.json
0
$ grep -c '"lm_head' .../granite-4.1-8b/.../model.safetensors.index.json
1
```

## Suggested fix

Drop `lm_head.weight` from the shipped safetensors so the layout
matches the 3B. The tensor carries no information beyond
`embed_tokens.weight`, and removing it unblocks `mlx_lm` while
shrinking the on-disk footprint.

```python
import json
from pathlib import Path
from safetensors.torch import save_file, safe_open

snap = Path(".../snapshots/<rev>")
idx_path = snap / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text())
src_shard = idx["weight_map"].pop("lm_head.weight")
idx_path.write_text(json.dumps(idx, indent=2))

shard_path = snap / src_shard
with safe_open(shard_path, framework="pt") as f:
    tensors = {k: f.get_tensor(k) for k in f.keys() if k != "lm_head.weight"}
save_file(tensors, shard_path)
```

Alternative: set `tie_word_embeddings: false` in `config.json`
instead. This is a behaviour change — the model now has an
independent (currently bit-identical) output projection that could
diverge under future fine-tunes. Only the right answer if there's an
intentional plan to keep `lm_head` separately updatable.

## Affected repos

Verified (`grep '"lm_head' model.safetensors.index.json` on the snapshot):
- [x] `ibm-granite/granite-4.1-8b` — **affected** (ships `lm_head.weight`)
- [ ] `ibm-granite/granite-4.1-8b-base` — likely affected (same export pipeline), please verify

Verified correct (no `lm_head.weight` on disk, matches 3B layout):
- [x] `ibm-granite/granite-4.1-3b` + `-base`
- [x] `ibm-granite/granite-4.1-30b`
- Entire 4.0 family

So this is **isolated to the 8B export**: the 30B (and 3B) correctly
omit the redundant tensor. The 8B's `tokenizer_config.json` /
`config.json` are identical in spirit to the 3B/30B's, so the
divergence is purely on the safetensors export side — likely a
flag that wasn't carried over in the 8B's export pipeline.
