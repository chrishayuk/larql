# larql expert-server on fly.io

CPU-only MoE expert servers. No GPU, no VRAM. The laptop runs the hot path
(attention + routing); fly.io machines serve the expert bank from
memory-mapped vindex shards.

## Memory sizing

Each `performance-8x` (16 GB) machine serves one 64-expert shard cleanly:
- ~6.2 GB: expert pages (64 experts × 30 layers × 421 MB / 128)
- ~1.8 GB: embeddings + dense FFN + norms (shared overhead)
- ~8 GB headroom (no thrashing)

`--warmup-walk-ffn` pre-faults owned expert pages at startup. Pages for
other shards' experts are never accessed (rejected by `--experts` filter),
so they never consume physical RAM.

## Prerequisites

- `fly` CLI installed and authenticated
- HuggingFace account (to host the expert-server slice)
- Vindex extracted locally: `output/gemma4-26b-a4b-q4k.vindex`

## Step 1 — Publish the expert-server slice to HuggingFace

The `expert-server` preset includes everything the server needs: embeddings,
norms, dense FFN (`interleaved_kquant.bin`), per-layer expert weights (`layers/`),
and tokenizer. Total: ~14.1 GB.

```bash
larql slice output/gemma4-26b-a4b-q4k.vindex \
  -o /tmp/gemma4-26b-expert-server.vindex \
  --preset expert-server

larql publish /tmp/gemma4-26b-expert-server.vindex \
  --repo chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server \
  --slices none
```

The live slice is already published at
`hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server`.

## Step 2 — Deploy two shards (recommended)

These apps are reachable from the public internet (`fly.toml` publishes
both the HTTP and the gRPC port), so each one needs an API key. The start
script refuses to boot without `API_KEY`. Pick one key for the shards and
give it to the client as `LARQL_SHARD_API_KEY`:

```bash
export SHARD_KEY="$(openssl rand -hex 32)"
```

The client sends the key over `https://` shard URLs. It cannot send it
over `grpcs://` yet, and refuses a gRPC shard while the key is set.

Each shard serves half the expert bank. Pages for the owned half are
pre-faulted at startup; the other half is never touched.

**Shard A — experts 0–63:**
```bash
fly apps create larql-expert-server-a
fly volumes create expert_data --size 25 --app larql-expert-server-a --region lhr --yes
fly secrets set HF_TOKEN=hf_... API_KEY="$SHARD_KEY" EXPERTS="0-63" WARMUP="1" --app larql-expert-server-a
fly deploy --app larql-expert-server-a --config deploy/fly/fly.toml --remote-only
```

**Shard B — experts 64–127:**
```bash
fly apps create larql-expert-server-b
fly volumes create expert_data --size 25 --app larql-expert-server-b --region lhr --yes
fly secrets set HF_TOKEN=hf_... API_KEY="$SHARD_KEY" EXPERTS="64-127" WARMUP="1" --app larql-expert-server-b
fly deploy --app larql-expert-server-b --config deploy/fly/fly.toml --remote-only
```

Each machine downloads the full vindex on first boot (~2 min on fly's LHR
network). The `--experts` filter ensures only the owned half's pages are
ever faulted into RAM.

## Step 3 — Point the client at the two shards

```bash
LARQL_SHARD_API_KEY="$SHARD_KEY" \
larql run output/gemma4-26b-a4b-q4k.vindex --max-tokens 20 \
  --moe-shards "0-63=https://larql-expert-server-a.fly.dev,\
64-127=https://larql-expert-server-b.fly.dev" \
  "The capital of France is"
```

## Single-machine option (simpler, demo only)

One machine serves all 128 experts. Requires performance-8x (16 GB) and
tolerates some page pressure under sustained load.

```bash
fly apps create larql-expert-server
fly volumes create expert_data --size 25 --app larql-expert-server --region lhr --yes
fly secrets set HF_TOKEN=hf_... API_KEY="$SHARD_KEY" --app larql-expert-server
fly deploy --app larql-expert-server --config deploy/fly/fly.toml --remote-only
```

Test:
```bash
LARQL_SHARD_API_KEY="$SHARD_KEY" \
larql run output/gemma4-26b-a4b-q4k.vindex --max-tokens 1 \
  --moe-shards "0-127=https://larql-expert-server.fly.dev" \
  "The capital of France is"
```

## Env vars

| Variable | Default | Description |
|---|---|---|
| `EXPERTS` | `""` | Expert range for this shard, e.g. `"0-63"`. Empty = all experts. |
| `WARMUP` | `"0"` | Set to `"1"` to pre-fault owned expert pages at startup. |
| `LAYERS` | `""` | Layer range, e.g. `"0-14"`. Empty = all layers. |
| `HF_REPO` | `chrishayuk/...` | HuggingFace repo to download the vindex from. |
| `VINDEX_PATH` | `/data/vindex` | Local path for the vindex on the persistent volume. |
| `PORT` | `8080` | HTTP listen port. |

## Latency note

Public internet (UK ↔ fly LHR): ~0.7 tok/s (30 serial RTTs × 45 ms each).
LAN or same-datacenter: ~19 tok/s. For batch dispatch (1 RTT/token,
approximate but usable): `larql run ... --moe-dispatch batch`.
