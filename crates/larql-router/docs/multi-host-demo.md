# Multi-host LAN deployment — operator walkthrough

A copyable runbook for bringing up a router and two `larql-server`
shards on **three separate Linux/macOS hosts** on a LAN. Mirrors the
single-host hot-shard demo but explicitly covers everything that
changes when boxes are physically separated:

- IP / DNS resolution for `--public-url` and `--join`
- Firewall rules (gRPC TCP + optional QUIC UDP)
- TLS: shared cert vs auto-generated self-signed + fingerprint pinning
- `--grid-key` shared-secret authentication
- Sanity checks (`curl /v1/health`, `larql-router status`)

See `hot-shard-demo.md` for the single-host load-pattern walkthrough.
This doc focuses on the multi-host **plumbing** — once it's up, the
hot-shard demo's traffic-generation script works against it
unchanged.

## Prerequisites

- Three hosts on the same LAN. Example IPs throughout:
  - `router-a` — `10.0.0.10`
  - `shard-1` — `10.0.0.11`
  - `shard-2` — `10.0.0.12`
- `larql-router` and `larql-server` built (`cargo build --release`)
  and on each host's `PATH`. Use the same git commit on all three.
- A vindex file you can copy to each shard host (`gemma3-4b.vindex`
  in the examples below).
- Free TCP ports on each box:
  - `router-a`: `9090` (HTTP), `50052` (grid gRPC)
  - `shard-1` / `shard-2`: `8080` (shard HTTP)

## Topology

```
                   ┌─────────────────────────────┐
                   │   router-a  (10.0.0.10)     │
       clients ───►│   :9090  /v1/walk-ffn       │
                   │   :50052 grid gRPC          │
                   └─────────┬──────────┬────────┘
                             │          │
                       Join+Heartbeat   │
                             │          │
              ┌──────────────▼──┐  ┌────▼────────────┐
              │ shard-1 :8080   │  │ shard-2 :8080   │
              │ (10.0.0.11)     │  │ (10.0.0.12)     │
              │ layers 0-16     │  │ layers 17-33    │
              └─────────────────┘  └─────────────────┘
                  (gemma3:4b)         (gemma3:4b)
```

The shards each own half of the model. A `POST /v1/walk-ffn` for
layer 5 lands on `shard-1`; for layer 25 on `shard-2`. Multi-layer
requests fan out in parallel.

## Step 1 — pick the wire trust model

Three choices, listed by typical use case:

| Mode | When | Setup |
|---|---|---|
| **Plain TCP, no auth** | Trusted LAN, dev | nothing extra |
| **TCP + `--grid-key`** | LAN with multiple tenants | shared 32-char secret |
| **QUIC + `--grid-key` + fingerprint pin** | Untrusted segment / production | `--features quic` build; cert + fingerprint |

The walkthrough below uses **TCP + `--grid-key`**. The "QUIC" §
below shows the diff for the third mode.

## Step 2 — start the router on `router-a`

```bash
# router-a (10.0.0.10)
GRID_KEY=$(openssl rand -hex 16)         # 32-char shared secret
echo "$GRID_KEY" > /etc/larql/grid-key

larql-router \
    --grid-port 50052 \
    --grid-key  "$GRID_KEY" \
    --port      9090 \
    --target-replicas       1 \
    --rebalance-interval   30 \
    --rebalance-threshold  2.0 \
    --rtt-probe-interval-secs 0
```

`larql-router` listens on **two** ports:
- `9090/tcp` for `POST /v1/walk-ffn` and `GET /v1/health`, `/metrics`
- `50052/tcp` for the gRPC `GridService.Join` stream the shards use

Open both in the firewall:

```bash
# Linux (firewalld example)
firewall-cmd --add-port=9090/tcp  --permanent
firewall-cmd --add-port=50052/tcp --permanent
firewall-cmd --reload
```

```bash
# macOS — /etc/pf.conf or `pfctl`. Adjust to local rules.
```

Verify the router is up:

```bash
curl -sf http://10.0.0.10:9090/v1/health
# → {"status":"ok"}
```

## Step 3 — start `shard-1`

Copy the vindex to `shard-1:/var/lib/larql/gemma3-4b.vindex` first.

```bash
# shard-1 (10.0.0.11)
GRID_KEY="$(cat /etc/larql/grid-key)"  # same secret as the router

larql-server \
    /var/lib/larql/gemma3-4b.vindex \
    --port        8080 \
    --layers      0-16 \
    --join        http://10.0.0.10:50052 \
    --public-url  http://10.0.0.11:8080 \
    --grid-key    "$GRID_KEY"
```

Critical fields:

- **`--public-url`** is the URL **clients** should use to reach this
  shard — the router will hand it to anyone resolving layer 0-16.
  Use the **host-reachable IP**, not `127.0.0.1`.
- **`--join`** is the router's gRPC port (`50052`), not its HTTP port.
- **`--grid-key`** must match the router's — the gRPC stream is
  rejected with `UNAUTHENTICATED` otherwise.

Open the shard's HTTP port:

```bash
firewall-cmd --add-port=8080/tcp --permanent
firewall-cmd --reload
```

## Step 4 — start `shard-2`

Symmetric to step 3 — same `--grid-key`, different layer range and
host IP:

```bash
# shard-2 (10.0.0.12)
larql-server \
    /var/lib/larql/gemma3-4b.vindex \
    --port        8080 \
    --layers      17-33 \
    --join        http://10.0.0.10:50052 \
    --public-url  http://10.0.0.12:8080 \
    --grid-key    "$(cat /etc/larql/grid-key)"
```

## Step 5 — verify the grid converged

On any host with network access to `router-a:50052`:

```bash
larql-router status --router http://10.0.0.10:50052
```

Expected output (server IDs vary):

```
Grid summary:
  models: 1
  servers: 2 serving, 0 available
  coverage gaps: 0

Model gemma3:4b:
  shard 0-16  : srv-1700000000-1   http://10.0.0.11:8080
  shard 17-33 : srv-1700000000-2   http://10.0.0.12:8080
  gaps: none
```

If one server doesn't show up:

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` from server | router gRPC port firewalled | open `50052/tcp` |
| Server logs `UNAUTHENTICATED` | `--grid-key` mismatch | re-share the secret |
| Server logs `connect timed out` | `--join` uses wrong IP | use the router's LAN IP, not `127.0.0.1` |
| `0 servers` after a clean start | server's `--public-url` is unreachable | ping the URL from the router |

## Step 6 — send a request

```bash
curl -sf -X POST http://10.0.0.10:9090/v1/walk-ffn \
     -H 'Content-Type: application/json' \
     -d '{"layers":[0,5,10,20,30],"residual":[0.0, 0.1, 0.2]}'
```

The router resolves each layer:

- `0, 5, 10` → `shard-1`
- `20, 30` → `shard-2`

…issues two parallel sub-requests, merges the responses, and returns
the unified envelope. Wall-clock latency = `max(shard1, shard2)`.

## Step 7 — scrape `/metrics`

```bash
curl -sf http://10.0.0.10:9090/metrics | grep -E "^larql_router_"
```

Key signals from ADR-0017:

- `larql_router_grid_servers{state="serving"}` = `2`
- `larql_router_grid_coverage_gaps` = `0`
- `larql_router_walk_ffn_duration_seconds_bucket{le="0.1"}` — p99 wire latency
- `larql_router_rebalancer_actions_total` — counts replicate / drop /
  elevate / demote / evict over the lifetime of the router

Plug into Prometheus + Grafana for live dashboards.

## QUIC variant (production / untrusted-segment)

Add `--quic-port 50053` to the router. The router auto-generates a
self-signed cert and prints the fingerprint at startup:

```
INFO QUIC: generated self-signed cert. Clients must pin this
     fingerprint via --quic-cert-fingerprint: 8f3a:b2:7e:...
```

Each shard then joins via:

```bash
larql-server ... \
    --join                  quic://10.0.0.10:50053 \
    --quic-cert-fingerprint "8f3ab27e..."
```

Open the QUIC port (UDP) on the router:

```bash
firewall-cmd --add-port=50053/udp --permanent
firewall-cmd --reload
```

The fingerprint pin is the trust anchor — no CA setup, no
LetsEncrypt. The cert lives only in the router's memory; rotation
means restarting the router and updating each shard's fingerprint.

Mixed-mode (some shards TCP, some QUIC) works — `--join` accepts
both `http://` and `quic://` URLs on the same router.

## MoE variant (ADR-0018, K2 / V3 / V4 scale)

For trillion-parameter MoE models, each host owns
`(single layer, expert subset)` rather than a contiguous layer
range. Example DeepSeek-V3 layout (60 layers × 256 experts × 4-way
expert split):

```
shard-001: layer_start=0, layer_end=0, expert_start=0,   expert_end=63
shard-002: layer_start=0, layer_end=0, expert_start=64,  expert_end=127
shard-003: layer_start=0, layer_end=0, expert_start=128, expert_end=191
shard-004: layer_start=0, layer_end=0, expert_start=192, expert_end=255
shard-005: layer_start=1, layer_end=1, expert_start=0,   expert_end=63
...        (60 layers × 4 hosts/layer = 240 shards)
```

Each `larql-server` invocation announces its slice via
`--layers L-L` plus the expert range (flag wiring is server-side;
see `crates/larql-server/docs/server-spec.md` for the per-shard
configuration).

Client requests use the new MoE shape:

```json
{
  "layer_experts": [
    {"layer": 0, "experts": [12, 47, 88, 200]},
    {"layer": 1, "experts": [3, 67, 130, 251]}
  ],
  "residual": [...]
}
```

The router resolves each `(layer, expert_id)` to its owning shard,
fans out per-token to all the relevant hosts, and merges. See
ADR-0018 for the design rationale and ADR-0019 for the planned
HTTP/3 transport that unblocks the worst-case fan-out latency.

## Tear-down

Stop the router last so the shards see a clean stream close (the
router emits `Grid: server left` log lines):

```bash
# shard-1, shard-2: Ctrl-C
# router-a:        Ctrl-C
```

Each shard's announce-task gracefully drops its `Join` stream; the
router deregisters the server within one heartbeat interval and
updates the `larql_router_grid_servers` gauge accordingly.

## Common gotchas

1. **`--public-url` uses `127.0.0.1`** — the router stores it and
   hands it to other hosts. They can't reach it. Always use the
   shard's LAN-routable IP or DNS name.
2. **Different `--grid-key` per host** — symptom is `UNAUTHENTICATED`
   in the shard logs on every reconnect. The shared secret is
   `--grid-key VALUE`, not a path; rotate by restarting all
   processes simultaneously.
3. **Firewall drops UDP for QUIC** — corp networks sometimes block
   UDP outright. Symptom is a hung shard with no logs after
   "Connecting to router grid...". Confirm UDP-50053 reachability
   with `nc -uvz 10.0.0.10 50053`.
4. **NTP skew across hosts** — the rebalancer's stale-heartbeat
   eviction uses each host's local wall clock. If two hosts'
   clocks drift more than `--stale-heartbeat-timeout` apart, you'll
   see spurious evictions. NTP everywhere.
5. **MTU mismatch on the QUIC path** — QUIC requires path-MTU
   discovery. Some VPNs / IPSec tunnels lower the effective MTU
   below QUIC's minimum (~1200 bytes). Symptom: TCP works,
   `quic://` doesn't. Either fix the MTU or fall back to TCP.

## See also

- [`hot-shard-demo.md`](./hot-shard-demo.md) — load-driven elevation +
  cool-down on the same topology.
- [`../../larql-server/docs/router-spec.md`](../../larql-server/docs/router-spec.md)
  — full CLI reference for both `larql-router` and `larql-server`.
- [`../ROADMAP.md`](../ROADMAP.md) — per-feature shipping notes.
- ADR-0017 (`/metrics`), ADR-0018 (MoE expert routing), ADR-0019
  (HTTP/3 transport).
