# Docker deployment — CPU + GPU split topology

This directory ships the two-container LARQL deployment:

- **`Dockerfile.ffn`** — CPU container running the FFN expert bank from RAM.
  Linux Ubuntu base, OpenBLAS, no GPU dependencies. Image size ~250–400 MB.
- **`Dockerfile.gpu`** — GPU container running attention + RotorQuant
  KV-cache. `nvidia/cuda:13.1.0-devel-ubuntu24.04` builder + matching
  runtime. Image size ~3 GB. Requires `--gpus all` (or `nvidia-container-toolkit`).
  Set `ENABLE_CUDA_OXIDE=1` only for the experimental cuda-oxide
  custom-kernel pilot; it adds LLVM 21, clang-21, a nightly Rust
  toolchain, and roughly 400 MB to the builder image.
- **`docker-compose.yml`** — orchestrates `ffn`, `attention`, and `router`
  on the dev box, sharing a `vindex_data` volume.
- **`docker-compose.cpu.yml`** — single-binary fallback for hosts without
  an NVIDIA GPU. Mirrors the existing single-server mode.
- **`start.sh`** — launcher consumed by both Dockerfiles. Vindex bootstrap,
  role-aware arg construction, GPU preflight.

## Quick start (RTX 4090 / CUDA 13)

```bash
cd deploy/docker

# build + run both containers + the router
docker compose up --build

# router on host:8082, attention on host:8081 (debug), ffn on host:8080 (debug)
curl http://localhost:8082/v1/health

# boot, wait for all three health checks, and run one prompt via the router
make -C ../.. demo
```

## Quick start (CPU laptop)

```bash
cd deploy/docker
docker compose -f docker-compose.cpu.yml up --build
curl http://localhost:8080/v1/health
```

## Topology

```
                     ┌──────────────────┐
client request   →   │   larql-router   │   (port 8082)
                     │  (CPU, plain)    │
                     └────┬─────────┬───┘
                attention RPC      FFN RPC
                          │         │
                ┌─────────▼──┐  ┌───▼─────────────┐
                │ attention  │  │  ffn            │
                │ container  │  │  container      │
                │ (GPU/CUDA) │  │  (CPU/RAM)      │
                │  port 8080 │  │  port 8080      │
                │            │  │                 │
                │ Q,K,V,O    │  │  expert weights │
                │ KV cache   │  │  (Q4_K)         │
                │ (iso3)     │  │  gate vectors   │
                │ embed      │  │  metadata       │
                └────────────┘  └─────────────────┘
                          ▲                ▲
                          └─── vindex_data volume ───┘
                              (shared, read-only)
```

The router runs a self-assembling grid on port 50052 and routes by
capability + layer range. The `attention` shard joins with
`capabilities: ["attention"]`; the `ffn` shard joins with
`capabilities: ["expert"]`. Backwards-compat: a single `--role all`
container declares both.

## VRAM and RAM budget on a 24 GB GPU

> _The numbers below are upper-bound estimates derived from each model's
> attention dim, KV size at the documented context length, and our
> RotorQuant compression ratios. Re-anchor with measurements as soon as
> the CUDA backend lands._

| Model | Context | KV format | VRAM idle | VRAM prefill peak | VRAM full-decode | CPU FFN RAM |
|---|---:|---|---:|---:|---:|---:|
| Gemma 3 4B | 8k | iso3 | 2.4 GB | 3.8 GB | 3.0 GB | 8.5 GB |
| Gemma 3 4B | 32k | iso3 | 2.4 GB | 5.0 GB | 4.6 GB | 8.5 GB |
| Llama 3 8B | 8k | iso3 | 5.0 GB | 7.4 GB | 6.5 GB | 14 GB |
| Llama 3 8B | 32k | iso3 | 5.0 GB | 9.8 GB | 9.0 GB | 14 GB |
| Llama 3 8B | 128k | iso3 | 5.0 GB | 12.0 GB | 11.0 GB | 14 GB |
| Qwen 2.5 14B | 8k | iso3 | 9.0 GB | 13.5 GB | 11.5 GB | 26 GB |
| Qwen 2.5 14B | 32k | iso3 | 9.0 GB | 18.0 GB | 16.0 GB | 26 GB |

For comparison, the same models with **fp16 KV** (no compression) need
roughly 10× more VRAM at the long-context configurations and exceed the
24 GB budget on Llama 3 8B at 32k.

Recommended Docker flags for the GPU container:

```
--gpus all
--shm-size=2g
--ulimit memlock=-1
```

## Environment variables

| Var | Container | Default | Purpose |
|---|---|---|---|
| `ROLE` | both | container default | `ffn` / `attention` / `all` |
| `VINDEX_PATH` | both | `/data/vindex` | local path for the shared vindex |
| `VINDEX_SOURCE` | compose | `vindex_data` | Docker named volume or host bind source mounted at `/data/vindex` |
| `HF_REPO` | both | gemma-4 expert server | repo to fetch if vindex absent |
| `HF_TOKEN` | both | unset | optional, for private repos |
| `KV_FORMAT` | gpu | `iso3` | `fp16` / `iso3` / `planar3` / `iso4` / `planar4` |
| `EXPERTS` | ffn | unset | range like `0-63` |
| `LAYERS` | gpu | unset | attention layer range like `0-15` |
| `WARMUP` | ffn | `1` | pre-fault expert pages at boot |
| `LARQL_BACKEND` | both | auto | force `cpu` / `metal` / `cuda` |
| `LARQL_CUDA_ARCH` | gpu | `89` | nvcc target arch; use `86` for RTX 30xx, `89` for RTX 40xx |
| `ENABLE_CUDA_OXIDE` | gpu build | `0` | install cuda-oxide pilot tooling in the builder image; only use while developing Rust-authored CUDA kernels |
| `FFN_HOST_PORT` | compose | `8080` (`make demo`: `18080`) | host port mapped to the FFN shard |
| `ATTENTION_HOST_PORT` | compose | `8081` (`make demo`: `18081`) | host port mapped to the attention shard |
| `ROUTER_HOST_PORT` | compose | `8082` (`make demo`: `18082`) | host port mapped to the router |
| `ATTN_SESSION_TTL` | gpu | `600` (s) | idle TTL for attention KV sessions |
| `MAX_ATTN_SESSIONS` | gpu | `256` | concurrent-session cap on the GPU shard |
| `JOIN` | both | compose-set | router grid URL, `http://router:50052` |
| `PUBLIC_URL` | both | compose-set | shard URL announced to the router |

`start.sh` translates `ROLE` to the server's `--role` flag:
`ffn → expert`, `attention → attention`, `all → both`. The
attention container also accepts `--attention-session-ttl-secs`
and `--max-attention-sessions` via the env vars above.

## Attention service endpoints

The GPU container exposes the `attention-service-routes` HTTP
surface (and gRPC equivalent) once the change ships:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/attention/session` | Create a session (FP32 or RotorQuant KV) |
| `GET` | `/v1/attention/session/{id}` | Read state |
| `DELETE` | `/v1/attention/session/{id}` | Drop |
| `POST` | `/v1/attention/prefill` | Run prefill, populate cache |
| `POST` | `/v1/attention/decode` | One decode step |
| `POST` | `/v1/kv-cache/snapshot` | Get the cache as a versioned binary blob |
| `POST` | `/v1/kv-cache/restore` | Replace the cache from a snapshot |
| `POST` | `/v1/kv-cache/free` | Free one or all layers |

The router uses heartbeat-bound `cached_prefixes` blooms to
prefer shards with the warm prefix for a session — see
`openspec/changes/router-prefix-aware-routing/`.

## Build commands

The Makefile has shortcuts (run from repo root):

```
make docker-ffn      # docker build -f deploy/docker/Dockerfile.ffn .
make docker-gpu      # docker build -f deploy/docker/Dockerfile.gpu .
make docker-up       # docker compose -f deploy/docker/docker-compose.yml up
make docker-up-cpu   # docker compose -f deploy/docker/docker-compose.cpu.yml up
make docker-down     # docker compose ... down -v
make docker-logs     # docker compose ... logs -f
make demo            # compose up -d, wait <=60s, one-shot router-backed prompt
```

## Demo Run

`make demo` uses `output/gemma-3-4b-it-vindex` when it exists on the
host, mounts it into both containers, waits up to 60 seconds for
`ffn`, `attention`, and `router` health checks, then runs:

```bash
larql dev walk \
  --index output/gemma-3-4b-it-vindex \
  --prompt "The capital of France is" \
  --predict \
  --predict-top-k 3 \
  --max-tokens 1 \
  --ffn-remote http://127.0.0.1:18082
```

This is the current one-shot Gemma 3 4B path: attention runs in the
local CLI driver and FFN calls go through `larql-router` to the `ffn`
container. The `attention` container is still booted and health-checked
by the same demo so the split topology is live before the prompt runs.

Expected warm local numbers on the RTX 4090 box:

| Path | Prompt | Expected |
|---|---|---:|
| Compose service boot, cached images | `make demo` | ≤ 60 s; observed 13 s on the RTX 4090 box |
| Router-backed one-shot | `The capital of France is` | first token in a few seconds on cold mmap; roughly 0.2-1 tok/s for the 1-token smoke |
| FFN hop deadline | router → `ffn` | 5 s default via `LARQL_ROUTER_HOP_DEADLINE_SECS` |
| GPU VRAM at decode | `attention` idle + iso3 KV | ~3.0 GB at 8k Gemma 3 4B context |

## Troubleshooting

- **GPU container exits with "no NVIDIA runtime detected"** — install
  `nvidia-container-toolkit` and ensure `docker run --gpus all
  nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi` works first.
- **GPU image is huge / slow to build** — `nvidia/cuda:13.1.0-devel`
  is ~3 GB. Use BuildKit cache: `DOCKER_BUILDKIT=1` is on by default in
  recent Docker. The default build arg is `LARQL_CUDA_ARCH=89` for the
  RTX 4090 dev box; override it for a different NVIDIA generation.
- **CUDA driver/toolkit mismatch** — error like `CUDA error: forward
  compatibility was attempted on non supported HW`. Update the
  driver to ≥ the toolkit's minimum (CUDA 13.x needs ≥ 545.x).
- **`docker compose up` hangs at "starting attention"** — first run
  pulls a ~3 GB CUDA image and downloads the vindex (~10 GB). Be
  patient. `docker compose logs -f attention` shows progress.
- **HF download fails on private repos** — set `HF_TOKEN` in your
  environment before `docker compose up`.

## Status

The CUDA backend kernel surface is **shipped** (see
`docs/cuda-rotorquant-status.md` for the parity-test ledger).
The attention-service-routes change is **partially shipped**:
session lifecycle (create / get / delete), KV-cache snapshot /
restore / free, role-aware capability announce, and Q4_K prefill
all work. Q4_K decode remains intentionally unsupported until a
one-step quantized decode helper lands.

The `ffn` container works today with the existing CPU FFN
service; the GPU container serves the session-lifecycle routes
and accepts heartbeat-driven prefix routing from the router.

See `openspec/changes/cuda-and-rotorquant-kv/` and
`openspec/changes/attention-service-routes/` for the full plan
and follow-up tasks.
