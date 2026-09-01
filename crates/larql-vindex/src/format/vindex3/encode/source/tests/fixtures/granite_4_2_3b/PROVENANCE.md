# granite-4.2-3b — real checkpoint metadata, no weights

Verbatim from `hf://ibm-granite/granite-4.2-3b` at commit
`b7e947307dd2efb3ad3b853b0e8a7e75f8ad4ac2`, staged by
`larql vindex3 plan hf://ibm-granite/granite-4.2-3b`.

The two `*.safetensors` files are **header-only stubs**: the 8-byte
little-endian length prefix and the JSON header it announces, and nothing
after. No tensor payload is present — together they are 41 KB standing in
for 7.32 GB.

## Why this checkpoint is the fixture

Its shard index disagrees with its own headers, and does so for a reason
that will recur:

```text
model.safetensors.index.json  metadata.total_size = 6,805,672,960
sum of header data_offsets, 363 tensors           = 7,319,475,200
difference                                        =   513,802,240
```

`lm_head.weight` and `model.embed_tokens.weight` were **tied** in the
source model. HF computes `total_size` from deduplicated parameter
storage, so it counts them once — while the file serialises both, at
`[0, 513802240)` and `[513802240, 1027604480)` of shard one. Two distinct
regions, one declaration.

For range-backed work the physical spans are the only thing that matters,
so the headers are the authority and the index is not. A synthetic fixture
can express the arithmetic; only a real one proves the phenomenon is
upstream rather than imagined.

Licensed Apache-2.0 by IBM, same as the source repo. Metadata only.
