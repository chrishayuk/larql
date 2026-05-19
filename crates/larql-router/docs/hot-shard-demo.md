# Hot-shard load-rate replication — operator walkthrough

Demonstrates `--hot-shard-rps`: when a shard's per-replica request rate
exceeds the threshold, the router treats it as effectively
under-replicated (`target + 1`) and pulls a spare from the available
pool. When the rate subsides, the over-replication tick prunes the
extra replica.

See the shipped entry in [`../ROADMAP.md`](../ROADMAP.md) for the
implementation reference; this doc is the operator-facing
"is it working?" walkthrough.

## Prerequisites

- `larql-router` and `larql-server` built (`cargo build --release`).
- A vindex on disk you can point at (`$VINDEX`).
- Three free TCP ports on localhost — defaults: `50052` (router gRPC),
  `9090` (router HTTP), `9181` / `9182` / `9183` (three shards).

## Topology

```
        clients              ┌────────────────────┐
            │                │   larql-router     │
            ▼                │  --grid-port 50052 │
       :9090 HTTP ◄──────────┤  --port 9090       │
            │                │  --target-replicas 1
            ▼                │  --hot-shard-rps 5 │
   route(layer) per layer    │  --rebalance-interval 5
            │                └─────────┬──────────┘
            │                          │  AssignMsg
            ▼                          ▼
  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ shard-a :9181   │  │ shard-b :9182    │  │ shard-c :9183    │
  │ (Mode A)        │  │ (Mode A)         │  │ (Mode B avail.)  │
  │ layers 0-9      │  │ layers 0-9       │  │ ram-only         │
  └─────────────────┘  └──────────────────┘  └──────────────────┘
       (serving)             (serving)              (spare)
```

Two Mode-A replicas of `0-9` plus one Mode-B spare. Under normal load,
`target_replicas=1` means the rebalancer wants to *drop* one of the
serving replicas — but with `--hot-shard-rps 5` set, sustained traffic
above 5 req/s lifts the effective target to 2 and the surplus replica
stays. Once the load falls, the elevated flag clears and over-rep
prunes back to 1.

## Run it

The reproducible path is the demo script — it starts everything,
drives load, and `tail`s the router log so you can see the elevation
fire:

```bash
VINDEX=path/to/your.vindex ./scripts/demo-hot-shard.sh
```

The script:

1. Starts the router with `--target-replicas 1 --hot-shard-rps 5
   --rebalance-interval 5` and short heartbeat timeouts so the demo
   fits in a couple minutes.
2. Starts two serving shards (Mode A) covering layers 0-9 and one
   spare (Mode B `--available-ram`).
3. Hammers the router with `larql bench --concurrent 16 --tokens 50
   --ffn http://localhost:9090` to push `req_per_sec` above 5.
4. `grep`s the router log for the rebalancer's elevation /
   demotion lines.

Expected log lines (router stderr):

```
Rebalancer: hot shard detected — effective_target raised by 1
    model_id=… layers=0-9 threshold=5
Grid: Mode B assignment sent
    server_id=srv-…-3 model_id=… layers=0-9 origin_url=http://localhost:9181
```

When the load stops:

```
Rebalancer: hot shard cooled — effective_target restored
    model_id=… layers=0-9
Rebalancer: dropping over-replicated replica
    server_id=srv-…-3 model_id=… layers=0-9
```

`larql-router status` (admin RPC) prints the live replica count after
each step. The spare's `server_id` and its add/drop sequence are the
load-bearing observable.

## Manual setup (if the script is too opinionated)

```bash
# Terminal 1 — router
larql-router \
    --grid-port 50052 --port 9090 \
    --target-replicas 1 \
    --hot-shard-rps 5 \
    --rebalance-interval 5 \
    --log-level info

# Terminal 2 — serving shard A
larql-server $VINDEX \
    --port 9181 --layers 0-9 \
    --join http://localhost:50052 \
    --public-url http://localhost:9181

# Terminal 3 — serving shard B
larql-server $VINDEX \
    --port 9182 --layers 0-9 \
    --join http://localhost:50052 \
    --public-url http://localhost:9182

# Terminal 4 — Mode B spare (no layers loaded yet; sized for the same shape)
larql-server \
    --port 9183 \
    --available-ram 8GB \
    --join http://localhost:50052 \
    --public-url http://localhost:9183

# Terminal 5 — drive load
larql bench $VINDEX \
    --backends "" --ffn http://localhost:9090 \
    --concurrent 16 --tokens 50

# Terminal 6 — watch the grid
watch -n 1 'larql-router --grid-port 50052 status'
```

When `--concurrent 16` lifts `req_per_sec` past 5, the router log
fires the "hot shard detected" line and `larql-router status` shows
three serving replicas for the `0-9` range. Stop the bench → CoV
drops → "hot shard cooled" line → `status` shows two replicas again.

## Knobs worth tweaking

| Flag | What it does in this demo |
|---|---|
| `--hot-shard-rps 5` | Threshold to elevate. Drop lower if your local load is gentle; raise for production-scale traffic. |
| `--rebalance-interval 5` | Tick cadence. Set short for the demo so elevations land in seconds; production default is 30 s. |
| `--target-replicas 1` | Base target. Hot-shard elevation is a `+1` on top — set `--target-replicas 2` to see it climb to 3. |
| `--concurrent N` on bench | Drives the rate. Each client opens one connection and serializes calls, so 16 ≈ 16× the single-client throughput. |

## Caveats

- The Mode B spare must have enough RAM (`--available-ram`) to hold a
  shard of the announced model. Setting `--available-ram 8GB` for a
  4 B-parameter model is comfortable; size up for bigger weights.
- The rebalancer's `cov_threshold` (Exp 41 retry rule) is unrelated
  to hot-shard elevation; both ride the same tick but track different
  metrics.
- This is single-host loopback. On a real LAN, RTT jitter affects
  `req_per_sec` indirectly — bench from the client side to see the
  rate the router actually observes.
