"""Model families the PREC-DIAG oracle supports, as facts that need no torch.

The experiment runner reads these to derive expected identities from a
freeze; the oracle (`hf_oracle.py`) reads them to choose its patches. One
table, so the two can never disagree.

A family is keyed by `config.json`'s `model_type`:

  gemma3_text  Gemma 3 text (270M, 1B). The family PREC-DIAG-1 was built on.
  qwen3        Qwen3 dense (0.6B). Logits only: E1 and E2, the runs the
               `hf-f64-accuracy` tests read. E0a's site captures are not
               mapped for it, so a capture run is refused.
  qwen3_5      Qwen3.5 hybrid (0.8B): Gated DeltaNet (`linear_attention`)
               and gated full-attention layers. Logits only, like qwen3.
               The checkpoint is composite (a vision tower and an MTP head
               beside the language model); the oracle loads the text model
               only, from `text_config`, and the loading gate accounts for
               every checkpoint tensor: loaded under `text_prefix`, or
               excluded under one of `excluded_prefixes`, never silently
               skipped. Only full-attention layers consume rotary
               embeddings (`rope_layer_types`).

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
    "qwen3_5": {
        "modeling": "qwen3_5",
        "modeling_sha256": "aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048",
        "captures": False,
        # The family's own oracle module: hashed into every run's identity,
        # and bound by a freeze as `script.hf_qwen3_5` (`oracle_module_label`).
        "oracle_module": "hf_qwen3_5.py",
        # Composite checkpoint: the language model's tensors sit under this
        # prefix (transformers maps it to `model.`), the head is tied.
        "text_prefix": "model.language_model.",
        # Checkpoint components outside the text-only oracle. A checkpoint
        # tensor under neither `text_prefix` nor one of these is refused.
        "excluded_prefixes": {
            "model.visual": "vision tower: multimodal input, out of scope for the text-only oracle",
            "mtp": "multi-token-prediction head: not on the next-token logits path, out of scope",
        },
    },
}

#: Layer types per family that apply rotary position embeddings. A family
#: absent here applies them on every layer type it has. Qwen3.5's DeltaNet
#: layers take no positions at all, so a RoPE table has no meaning there.
ROPE_LAYER_TYPES = {"qwen3_5": ("full_attention",)}
#: The layer types a qwen3_5 config may list (anything else has no patch).
QWEN3_5_LAYER_TYPES = ("full_attention", "linear_attention")


class UnsupportedFamily(ValueError):
    pass


def family_of(config: dict) -> str:
    model_type = config.get("model_type")
    if model_type not in FAMILIES:
        raise UnsupportedFamily(f"model_type {model_type!r}: the oracle supports {sorted(FAMILIES)}")
    return model_type


def oracle_module_label(family: str) -> str | None:
    """The freeze label binding a family's own oracle module, if it has one."""
    module = FAMILIES[family].get("oracle_module")
    return f"script.{module.removesuffix('.py')}" if module else None


def text_config(config: dict) -> dict:
    """The language model's config: `text_config` for a composite family,
    the config itself otherwise."""
    if config.get("model_type") in FAMILIES and "text_prefix" in FAMILIES[config["model_type"]]:
        if not isinstance(config.get("text_config"), dict):
            raise UnsupportedFamily(f"{config['model_type']}: a composite config with no text_config")
        return config["text_config"]
    return config


def rope_layer_types(config: dict) -> list[str]:
    """The distinct layer types that apply rotary embeddings (and so the
    ones a RoPE table can be given for), sorted."""
    types = sorted(set(config_layer_types(config)))
    scoped = ROPE_LAYER_TYPES.get(config.get("model_type"))
    return types if scoped is None else [t for t in types if t in scoped]


def config_layer_types(config: dict) -> list[str]:
    """Per-layer attention types.

    - Qwen3.5 reads `text_config.layer_types`, which must be listed and use
      only `linear_attention` and `full_attention`.
    - A config that lists `layer_types` is taken as written.
    - An older Gemma 3 config (1B) names only `sliding_window_pattern`;
      transformers expands it as every `pattern`-th layer full, the rest
      sliding, and so does this.
    - Qwen3 with `use_sliding_window` false is full attention throughout.
      With it true, or any layer not full attention, the config is
      refused: no window is patched.
    """
    if config.get("model_type") == "qwen3_5":
        types = text_config(config).get("layer_types")
        if not types:
            raise UnsupportedFamily("qwen3_5 with no text_config.layer_types: the hybrid layout must be explicit")
        if not set(types) <= set(QWEN3_5_LAYER_TYPES):
            raise UnsupportedFamily(f"qwen3_5 layer_types {sorted(set(types))}: only {list(QWEN3_5_LAYER_TYPES)}")
        return list(types)
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
