# VINDEX3 inputs, continuation and layer workers

**Class: CURRENT.** These interfaces execute the container's component plan.
They do not reconstruct language-model `ModelWeights` or select a V2 engine.
The [execution guide](execution.md) covers building the binaries and encoding.

## Exact continuation choices

```bash
larql run model.vindex3 "Hello" --engine row
larql run model.vindex3 "Hello" --engine standard
larql run model.vindex3 "Hello" --engine no-cache
```

`--engine` values are aliases for continuation identities: `row` (the
default) names `row/v1` and `standard` names `canonical/v1`. The CLI selects
the identity from a fresh registry against the plan before anything runs,
and a run reports the identity it resolved. Both implement the canonical
interpreter's continuation contract (CONTINUATION-PLUGIN-1; see
[the continuation plane](../continuation-plane-inventory.md)). An explicit
`--engine` wins over `LARQL_KV_ENGINE`; unsupported values refuse. Attention
windows come from the V3 plan, so a separate `--context-window` override refuses. `no-cache`
(also `--kv-cache none`) retains the input history and replays it through fresh
state for each step. It avoids retaining KV between steps, but still allocates
KV during replay and retains image rows. It is an exact execution control,
not a measured memory or speed improvement. Quantized/approximate engine names
and `markov-bounded` remain refused on V3. Conflicting selectors refuse.

The runtime's [input module](../../crates/larql-inference/src/vindex3/input.rs)
accepts token IDs and ready-to-use embedding rows. It validates the complete
new input before advancing state. External rows have sequential positions and
enter the same canonical layer loop after token lookup/scaling. Their producer
owns projection and scaling. Replay preserves those rows alongside token IDs.

## Image prefixes

```bash
larql run gemma3.vindex3 "Describe this image" \
  --image picture.png --mm-weights /path/to/gemma3/checkpoint --engine standard
```

This first image adapter supports Gemma 3's SigLIP tower and average-pool
projector on CPU. The external checkpoint supplies vision/projector weights
and protocol metadata; the V3 container supplies all language-model operations
and operands. Family and hidden-width checks reject incompatible sources;
those checks do not prove the two artifacts came from the same model revision.
Use matching model revisions.

Each image becomes a start token, 256 projected rows and an end token, followed
by the text prompt. Multiple images form consecutive prefixes. The adapter uses
the existing LARQL image protocol and the container's declared attention spans;
it does not add image-specific bidirectional masks, crop/tiling protocols,
interleaved images or Metal image execution. Those remain separate work.

Tests cover actual image decoding, checkpoint loading and vision/projector
forward, plus numerical parity at the external-input seam. They do not establish
full-model caption quality or Hugging Face multimodal parity.

## CPU layer workers

For a **two-layer** container, start two workers in separate terminals:

```bash
larql-server model.vindex3 --layers 0-0 --port 9181
larql-server model.vindex3 --layers 1-1 --port 9182
larql run model.vindex3 "Hello" \
  --v3-shards http://localhost:9181,http://localhost:9182
```

Adjust the inclusive server ranges to cover your model exactly once. The CLI
loads only embedding/final norm/output-head operands. Each worker prepares only
its layer range. All nodes open the same container; this change does not package or distribute
shard files.

`GET /v1/vindex3/layers` returns a versioned binding; `POST` accepts that binding
and a complete matrix of prefix rows starting at absolute position zero. The
wire range is half-open. The binding identifies the declared artifact/plan,
CPU lowering identity/revision, layer range, total layers and hidden width. The coordinator refuses
missing/overlapping/out-of-order ranges, mismatched artifacts and malformed
responses. A failed request leaves its logical input history unchanged.
The identity names declared payload hashes; run `vindex verify` to check bytes.

Workers recompute the entire prefix on every step and keep no remote KV session.
This makes retry state explicit, at considerable compute and network cost.
Requests allow at most 4096 positions and are also subject to HTTP body limits.
The first implementation supports CPU single-residual-stream softmax stacks;
Metal, recurrent/MLA stacks, hyper-connections, grid discovery, automatic retries
and remote continuation caches are not implemented here. Use the same build
on coordinator and workers. No distributed throughput claim has been measured.

The route is mounted on the single-model serving profile, not the public
explorer or multi-model profile. Whole-model generation requests against a
layer worker refuse. `/v1/runtime` reports its `layer_shard` binding. Existing
server authentication applies; `--v3-shard-token-env ENV_NAME` reads a bearer
token for the CLI transport. The grid/OpenAI router remains a separate path.

The local and loopback HTTP parity fixtures cross a sliding-window boundary.
They compare split execution with the complete program, including logits,
invalid bindings and refusal to generate from partial stacks.
