# IBM Granite 4.1 8B — `lm_head.weight` shipped alongside `tie_word_embeddings: true`

Status: **diagnosed**, no workaround needed in LARQL (loader honours
the present tensor when one is shipped, falls back to the tied embed
otherwise — `crates/larql-models/src/loading/safetensors.rs:329`).
MLX's `mlx_lm` rejects the model and needs an upstream fix. Last
updated 2026-05-17.

Affected repos (verified by inspecting `model.safetensors.index.json`
for the `lm_head.weight` key on each snapshot, 2026-05-17):

- [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) — **affected** (ships `lm_head.weight`)
- [ibm-granite/granite-4.1-8b-base](https://huggingface.co/ibm-granite/granite-4.1-8b-base) — inferred (same export pipeline as -8b)
- [ibm-granite/granite-4.1-30b-base](https://huggingface.co/ibm-granite/granite-4.1-30b-base) — inferred (please verify with the snippet below)

Unaffected (correctly packaged, ship only `model.embed_tokens.weight`):

- `ibm-granite/granite-4.1-3b` — verified
- `ibm-granite/granite-4.1-30b` — **verified** (no `lm_head.weight` on disk, 12-shard layout, matches the 3B's correct pattern)
- The entire 4.0 family

So this is a **regression isolated to the 8B export** (and presumably
its `-base` sibling — please confirm). The 30B sized down from 64 GB
to 56 GB on disk *because* it doesn't carry the redundant tensor
that 8B does. Worth diffing the two export pipelines to find what
triggered the divergence.

## TL;DR

`config.json` declares `"tie_word_embeddings": true` — meaning the
output projection should share weights with the input embedding.
`model.safetensors.index.json` then **also** maps an explicit
`lm_head.weight` tensor into shard 1 of the safetensors split.
The two declarations contradict each other: tied embeddings, by
definition, don't have a separate output-projection tensor.

The `lm_head.weight` tensor is **bit-identical** to
`model.embed_tokens.weight` — `torch.equal(lm_head, embed_tokens) →
True`. So the duplication isn't communicating different weights; it's
~822 MB (100352 × 4096 × bf16) of redundant data shipped because the
tensor was emitted by mistake.

Three consumer behaviours follow:

| Consumer | Behaviour |
|---|---|
| `transformers.AutoModelForCausalLM` | Loads both, silently overwrites the tied head with the redundant `lm_head.weight`. Forward pass is correct (the tensors are identical, so the choice doesn't matter), but loading is ~10 s slower and uses an extra ~822 MB of host RAM during the load. |
| LARQL (this repo) | `crates/larql-models/src/loading/safetensors.rs:329` prefers an explicit `lm_head.weight` when present; falls back to `embed.clone()` otherwise. Behaves like HF. Forward correct, 0.000 % bits/char delta to HF on the Frankenstein gate. |
| `mlx_lm` (Apple MLX, ≥ v0.x) | `mlx.nn.Module.load_weights(..., strict=True)` rejects the load with `ValueError: Received 1 parameters not in model: lm_head.weight`. The MLX Granite implementation sets up tied embeddings from `tie_word_embeddings=true` and then has no slot for the redundant tensor. **Cannot load the 8B at all** without patching the safetensors or `mlx_lm`. |

## Reproduction

```python
from pathlib import Path
import json, torch
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM
import mlx_lm

REPO = "ibm-granite/granite-4.1-8b"
SNAP = Path("~/.cache/huggingface/hub/...").expanduser()  # snapshot dir

# 1. config.json says tied embeddings.
cfg = json.loads((SNAP / "config.json").read_text())
print(f"tie_word_embeddings = {cfg['tie_word_embeddings']}")  # True

# 2. ...but the safetensors index has an explicit lm_head.weight.
idx = json.loads((SNAP / "model.safetensors.index.json").read_text())
print(f"lm_head.weight shard: {idx['weight_map'].get('lm_head.weight')}")
# → "model-00001-of-00004.safetensors"

# 3. ...and that tensor is bit-identical to embed_tokens.weight.
with safe_open(SNAP / "model-00001-of-00004.safetensors", framework="pt") as f:
    lm = f.get_tensor("lm_head.weight")
    em = f.get_tensor("model.embed_tokens.weight")
print(f"bit-identical? {torch.equal(lm, em)}")  # True

# 4. HF transformers tolerates the redundancy.
m = AutoModelForCausalLM.from_pretrained(REPO)  # loads, runs

# 5. mlx_lm refuses.
mlx_lm.load(REPO)
# → ValueError: Received 1 parameters not in model: lm_head.weight
```

(`SNAP` is the resolved HF-cache snapshot dir, e.g.
`~/.cache/huggingface/hub/models--ibm-granite--granite-4.1-8b/
snapshots/<rev>/`.)

## Compared to 3B (correct packaging)

```bash
# 3B — no separate lm_head tensor:
$ grep -c lm_head .../granite-4.1-3b/.../model.safetensors.index.json
0

# 8B — has a separate lm_head tensor on top of the tied embed:
$ grep -c lm_head .../granite-4.1-8b/.../model.safetensors.index.json
1
```

Same `config.json` declaration (`tie_word_embeddings: true`) on both;
3B drops the tensor, 8B keeps it. The 3B is what
`transformers.PreTrainedModel.tie_weights()` expects to see on disk.

## Suggested upstream fix

Either of:

**A (preferred): drop the redundant tensor.** Re-export the 8B
safetensors without `lm_head.weight`, matching the 3B's layout. Frees
~822 MB of disk + bandwidth + HF Hub storage per copy and unblocks
`mlx_lm`. Single-line change to the export pipeline. The 30B (and
both `-base` variants) likely need the same treatment — please
verify with the repro snippet above.

```python
# Quick one-shot fix on a cached snapshot (illustrative, not for prod):
import json
from pathlib import Path
from safetensors.torch import save_file, safe_open

snap = Path(".../snapshots/<rev>")
idx_path = snap / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text())
src_shard = idx["weight_map"].pop("lm_head.weight")  # "model-00001-of-00004.safetensors"
idx_path.write_text(json.dumps(idx, indent=2))

shard_path = snap / src_shard
with safe_open(shard_path, framework="pt") as f:
    tensors = {k: f.get_tensor(k) for k in f.keys() if k != "lm_head.weight"}
save_file(tensors, shard_path)
```

**B: untie the embeddings.** Set `tie_word_embeddings: false` in
`config.json` and leave `lm_head.weight` on disk. This is a behaviour
change — the model now has an independent output projection — but
since the two tensors are currently bit-identical it produces the
same logits at t = 0. Future fine-tunes / patches that touch
`lm_head.weight` would then diverge from `embed_tokens.weight`. Only
the right answer if IBM *intends* the head to be untie-able going
forward.

## Related: how MLX could be more forgiving

`mlx_lm.utils.load_model` uses `model.load_weights(weights,
strict=True)` and raises on any unconsumed key. A more permissive
mode would skip a key when (a) `config.tie_word_embeddings=True` and
(b) the unconsumed key is `lm_head.weight` and (c) it equals the
shipped `embed_tokens.weight` — but the right place to fix this is
the model packaging, not the loader.

## LARQL impact

None — LARQL Rust loads the 8B cleanly because `safetensors.rs:329`
prefers the present `lm_head.weight` and `shannon verify` passes at
0.000 % delta vs HF on the Frankenstein 1 KB corpus (263 tokens,
3.9126 bits/token; PR #TBD). Q4K vindex generation through `larql
run` produces coherent text:

> "The capital of France is Paris. **Step-by-step reasoning** 1.
> **Identify the question** – It asks for the capital of France.
> 2. **Recall basic geographic knowledge** – France, a country in
> Western Europe, has Paris as its capital city. 3. **…"

The dev-machine cross-engine sweep in
`scripts/diagnose_models.py::Granite-4.1-8B` runs the LARQL × HF leg
only, with `--engines hf`, until the upstream packaging is fixed and
the MLX leg can be brought back.
