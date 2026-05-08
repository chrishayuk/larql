# deploy-cpu-gpu-split Specification

## Purpose
TBD - created by archiving change cuda-and-rotorquant-kv. Update Purpose after archive.
## Requirements
### Requirement: Two-Dockerfile deployment topology

The repository SHALL provide two production Dockerfiles under
`deploy/docker/`:

- `Dockerfile.ffn` — Linux Ubuntu base, builds `larql-server` without
  the `cuda` feature, no GPU runtime dependencies. Image size
  SHOULD be ≤ 400 MB.
- `Dockerfile.gpu` — `nvidia/cuda:13.1-devel-ubuntu24.04` base,
  builds `larql-server --features cuda`, ships the CUDA runtime.

Both images SHALL accept the same `start.sh` arguments and the
existing `--role` flag (`ffn`, `attention`, or `all`) SHALL be the
only thing that distinguishes their behaviour at runtime.

#### Scenario: FFN image builds without CUDA installed
- **WHEN** `docker build -f deploy/docker/Dockerfile.ffn .` runs on a host without CUDA
- **THEN** the build SHALL succeed and the resulting image SHALL run the FFN service
<!-- test: unbacked -->

#### Scenario: GPU image refuses to start without an NVIDIA runtime
- **WHEN** the GPU image is run with `docker run` (not `docker run --gpus all`)
- **THEN** the container SHALL exit non-zero with a clear "no NVIDIA runtime detected" message
<!-- test: unbacked -->

### Requirement: docker-compose orchestration for the dev box

`deploy/docker/docker-compose.yml` SHALL define three services:

- `ffn` — uses `Dockerfile.ffn`, no GPU access, exposes port 8080,
  mounts the shared `vindex_data` volume.
- `attention` — uses `Dockerfile.gpu`, requests `gpus: all`, exposes
  port 8081, mounts the shared `vindex_data` volume.
- `router` — runs `larql-router` against both shards.

Bringing the stack up SHALL be a single `docker compose up`. A
single `docker compose down -v` SHALL clean up.

#### Scenario: docker compose up brings all three services healthy
- **WHEN** a fresh user runs `docker compose up` from `deploy/docker/`
- **THEN** within 60 seconds all three services SHALL pass health checks at `/v1/health`
<!-- test: unbacked -->

### Requirement: Shared vindex storage between containers

The two containers SHALL read the same vindex from a shared volume
(or bind-mount). The compose file MUST not require copying weights
into both containers.

#### Scenario: Both containers see the same vindex bytes
- **WHEN** `docker compose exec ffn sha256sum /data/vindex/index.json` and `docker compose exec attention sha256sum /data/vindex/index.json` are run
- **THEN** the two checksums SHALL be identical
<!-- test: unbacked -->

### Requirement: Container memory budget documentation

`deploy/docker/README.md` SHALL document, for at least Gemma 3 4B,
Llama 3 8B, and Qwen 2.5 14B (each at FP16 attention + iso3 KV +
Q4_K FFN), the expected:

- VRAM footprint of the attention container at idle, prefill, and
  full-context decode.
- RAM footprint of the FFN container at idle and during expert dispatch.
- Recommended `--shm-size` and `ulimit -m` settings.

#### Scenario: README has a memory budget table
- **WHEN** `deploy/docker/README.md` is read
- **THEN** it SHALL contain a markdown table indexed by `(model, kv_format)` with columns for VRAM idle / prefill / decode and CPU RAM idle / dispatch
<!-- test: unbacked -->

### Requirement: Single-box mode for laptops without a GPU

`deploy/docker/docker-compose.cpu.yml` SHALL provide a CPU-only
compose alternative that runs everything in the FFN container without
attention being remote. This mirrors the existing single-binary mode
and gives users without a GPU a working stack.

#### Scenario: CPU compose works on a laptop
- **WHEN** `docker compose -f docker-compose.cpu.yml up` is run on a Mac without `--gpus`
- **THEN** the stack SHALL pass health checks and serve a `larql lql 'SHOW MODELS;'` query
<!-- test: unbacked -->

