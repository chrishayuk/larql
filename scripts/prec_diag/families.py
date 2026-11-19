"""Model families the PREC-DIAG oracle supports, as facts that need no torch.

The experiment runner reads these to derive expected identities from a
freeze; the oracle (`hf_oracle.py`) reads them to choose its patches. One
table, so the two can never disagree.

A family is keyed by `config.json`'s `model_type`:

  gemma3_text  Gemma 3 text (270M, 1B). The family PREC-DIAG-1 was built on.
  qwen3        Qwen3 dense (0.6B). Logits only: E1 and E2, the runs the
               `hf-f64-accuracy` tests read. E0a's site captures are not
               mapped for it, so a capture run is refused.

`modeling_sha256` is transformers 5.5.0's modeling file for the family: the
file whose f32 forcings A64 patches. A different file is refused.
"""

from __future__ import annotations

FAMILIES = {
    "gemma3_text": {
        "modeling": "gemma3",
        "modeling_sha256": "221f147ccc0b16d7fa0aa007231ee0a6bd40fb37feb79eb9e537290498468aef",
        "captures": True,
    },
    "qwen3": {
        "modeling": "qwen3",
        "modeling_sha256": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
        "captures": False,
    },
}


class UnsupportedFamily(ValueError):
    pass


def family_of(config: dict) -> str:
    model_type = config.get("model_type")
    if model_type not in FAMILIES:
        raise UnsupportedFamily(f"model_type {model_type!r}: the oracle supports {sorted(FAMILIES)}")
    return model_type


def config_layer_types(config: dict) -> list[str]:
    """Per-layer attention types.

    - A config that lists `layer_types` is taken as written.
    - An older Gemma 3 config (1B) names only `sliding_window_pattern`;
      transformers expands it as every `pattern`-th layer full, the rest
      sliding, and so does this.
    - Qwen3 with `use_sliding_window` false is full attention throughout.
      With it true, or any layer not full attention, the config is
      refused: no window is patched.
    """
    if config.get("model_type") == "qwen3":
        if config.get("use_sliding_window"):
            raise UnsupportedFamily("qwen3 with use_sliding_window: not supported")
        types = config.get("layer_types") or ["full_attention"] * config["num_hidden_layers"]
        if set(types) != {"full_attention"}:
            raise UnsupportedFamily(f"qwen3 layer_types {sorted(set(types))}: only full attention")
        return list(types)
    if config.get("layer_types"):
        return list(config["layer_types"])
    pattern = config.get("sliding_window_pattern") or config.get("_sliding_window_pattern")
    if not pattern:
        return []
    return ["sliding_attention" if (i + 1) % pattern else "full_attention"
            for i in range(config["num_hidden_layers"])]
