## 1. Phase 1 — scaffolding (this change)

- [x] 1.1 Create `crates/larql-rotorquant/` skeleton (Cargo.toml, src/lib.rs with the public API surface as `unimplemented!()`, UPSTREAM.md placeholder, cuda/ dir with a README explaining the vendor strategy).
- [x] 1.2 Add `cuda` feature flag to `crates/larql-compute/Cargo.toml`. Add optional `cudarc` as a target-gated dep on Linux.
- [x] 1.3 Create `crates/larql-compute/src/cuda/` with `mod.rs`, `backend.rs`, `matmul.rs`, `quant_matvec.rs`, `decode.rs`, `error.rs`. Stub `CudaBackend` impls of the trait family that return `Err(CudaInitError::NotImplemented)` or `unimplemented!()`.
- [x] 1.4 Update `default_backend()` in `crates/larql-compute/src/lib.rs` to honour the CUDA → Metal → CPU precedence (with `LARQL_BACKEND` override).
- [x] 1.5 `cargo check --features cuda` runs cleanly on the dev box.
- [x] 1.6 `crates/larql-compute/src/lib.rs` doc table lists CUDA in the backends table (no longer "planned").
- [x] 1.7 Add `--role` flag to `larql-server` (parse only — no behaviour change in this phase).

## 2. Phase 1 — deploy/docker (this change)

- [x] 2.1 `deploy/docker/Dockerfile.ffn` — Linux Ubuntu base, mirrors `deploy/fly/Dockerfile`, builds `cargo build --release -p larql-server`.
- [x] 2.2 `deploy/docker/Dockerfile.gpu` — `nvidia/cuda:13.1-devel-ubuntu24.04` base, builds with `--features cuda`.
- [x] 2.3 `deploy/docker/start.sh` — launcher that respects `ROLE`, `VINDEX_PATH`, `EXPERTS`, `WARMUP`, `KV_FORMAT` env vars.
- [x] 2.4 `deploy/docker/docker-compose.yml` — `ffn`, `attention`, `router` services with shared `vindex_data` volume; `gpus: all` on the attention service.
- [x] 2.5 `deploy/docker/docker-compose.cpu.yml` — single-binary fallback for non-GPU laptops.
- [x] 2.6 `deploy/docker/README.md` — topology diagram, VRAM / RAM budget table, build commands, troubleshooting (NVIDIA runtime, driver mismatch, large-image cleanup).
- [x] 2.7 `Makefile` targets: `docker-ffn`, `docker-gpu`, `docker-up`, `docker-down`, `docker-logs`.

## 3. Phase 2 — minimal CUDA backend (follow-up change `cuda-f32-baseline`)

- [x] 3.1 cuBLAS f32 GEMM via cudarc; pass the existing `test_correctness` matmul tests with feature `cuda`.
- [x] 3.2 cuBLAS f32 GEMV; pass existing LM-head gemv tests.
- [x] 3.3 Backend init compiles and caches PTX for one custom kernel (no-op kernel) to validate the cache plumbing.
- [ ] 3.4 `larql-cli predict` runs end-to-end on RTX 4090 in the GPU container.
- [x] 3.5 Wire `make ci-cuda` to run the CUDA-feature subset of the test suite when `LARQL_CUDA_AVAILABLE=1`.

## 4. Phase 2 — Q4 / Q6 matvec (follow-up change `cuda-q4-matvec`)

- [x] 4.1 Q4_0 matvec kernel — match the existing CPU correctness tests.
- [x] 4.2 Q4_K matvec kernel — pass `test_q4k_parity` at production dimensions (Gemma 4B `hidden=2560`, `intermediate=10240`).
- [x] 4.3 Q4_KF matvec kernel — pass FFN routing parity tests.
- [x] 4.4 Q6_K matvec kernel.
- [x] 4.5 Update `larql-compute` `quant_matvec` dispatch table.

## 5. Phase 2 — fused attention (follow-up change `cuda-fused-attention`)

- [ ] 5.1 Fused QKV-projection + RMS norm kernel.
- [ ] 5.2 Fused QK-norm + RoPE + softcap kernel.
- [ ] 5.3 Fused KV-append + scaled-dot-product + softmax + V-aggregate kernel for FP16 KV.
- [ ] 5.4 Pass `test_fused_attention` in the `cuda` feature build.
- [ ] 5.5 Pass `test_cpu_metal_parity` extended to include CUDA.

## 6. Phase 3 — RotorQuant kernels (follow-up change `rotorquant-kernels`)

- [ ] 6.1 Vendor `planar3.cu`, `iso3.cu`, `planar4.cu`, `iso4.cu` from the llama.cpp fork's `feature/planarquant-kv-cache` branch into `crates/larql-rotorquant/cuda/`.
- [ ] 6.2 Record source URL + commit + date in `crates/larql-rotorquant/UPSTREAM.md`.
- [ ] 6.3 `build.rs` compiles the vendored kernels with `nvcc` for sm_70+ archs (override via `LARQL_CUDA_ARCH`).
- [ ] 6.4 Rust FFI wrappers in `crates/larql-rotorquant/src/ffi.rs`.
- [ ] 6.5 Safe Rust API in `crates/larql-rotorquant/src/lib.rs` (`KvFormat`, `quantize_k`, `quantize_v`, `dequantize_k`, `dequantize_v_with_inverse_rotation`, `KvScratch`).
- [ ] 6.6 Round-trip tests against the upstream Triton reference (`crates/larql-rotorquant/ref/` shim) for `iso3` and `planar3`.
- [ ] 6.7 `make rotorquant-sync` target diffs vendored vs upstream and exits non-zero on drift.

## 7. Phase 3 — RotorQuant attention integration (follow-up change `rotorquant-attention-integration`)

- [ ] 7.1 Add `KvFormat` parameter to `larql_inference::attention::KvCache`.
- [ ] 7.2 Quantize-on-write / dequantize-on-read paths in the attention forward.
- [ ] 7.3 Deferred-K behaviour during prefill (FP16 backing store, lazy quantize on decode insert).
- [ ] 7.4 KV-surgery operations (`get_layer`, `set_layer`, `clone_layer_position_range`) round-trip across formats.
- [ ] 7.5 `RotorQuantStrategy { variant: Iso3 | Planar3 | Iso4 | Planar4 }` joins `kv-cache-benchmark` strategy enum.
- [ ] 7.6 Accuracy harness reports PPL / decode tok/s alongside existing strategies; numbers within ±2% of upstream paper on Llama 3.1 8B.

## 8. Phase 4 — attention service routes (follow-up change `attention-service-routes`)

- [x] 8.1 HTTP routes — `POST /v1/attention/session`, `DELETE /v1/attention/session/{id}`, `GET /v1/attention/session/{id}`.
- [x] 8.2 HTTP routes — `POST /v1/attention/prefill`, `POST /v1/attention/decode`.
- [x] 8.3 HTTP routes — `POST /v1/kv-cache/snapshot`, `POST /v1/kv-cache/restore`, `POST /v1/kv-cache/free`.
- [x] 8.4 gRPC parity in `larql-router-protocol` proto definitions.
- [x] 8.5 Snapshot/restore round-trip integration test.
- [x] 8.6 Topology announce includes `capabilities: ["attention"]` when `--role attention`.

## 9. Phase 5 — heterogeneous router topology (follow-up change `router-heterogeneous-shards`)

- [ ] 9.1 `larql-router` shard registration accepts `capabilities` field.
- [ ] 9.2 Routing layer picks shard by capability + layer range.
- [ ] 9.3 Per-hop deadline (5 s default, env-overridable) prevents heterogeneous deadlocks.
- [ ] 9.4 Backwards-compat: pre-change shards (no capabilities field) get `["attention","expert"]` by default.
- [ ] 9.5 Status endpoint reports the capability map.

## 10. Phase 5 — end-to-end docker compose (follow-up change `deploy-compose-end-to-end`)

- [ ] 10.1 Single `docker compose up` boots all three services healthy in ≤ 60 s.
- [ ] 10.2 Gemma 3 4B end-to-end inference run (LM benchmark prompt) succeeds via the router.
- [ ] 10.3 README documents the run, expected tok/s, expected VRAM at decode.
- [ ] 10.4 `make demo` target boots the stack and runs a one-shot inference.

## 11. Validation (this change)

- [x] 11.1 `openspec validate cuda-and-rotorquant-kv --strict` passes.
- [x] 11.2 `cargo check --features cuda` passes on the dev box (Linux + RTX 4090 + CUDA 13.1).
- [x] 11.3 `cargo check` (default features) passes — no regression for existing builds.
- [x] 11.4 `make traceability-check` and `make openspec-validate` pass.
- [x] 11.5 Capability inventory in proposal.md matches `specs/<capability>/spec.md` directories one-to-one.
- [x] 11.6 Commit with `[#cuda-and-rotorquant-kv]` tag in the subject; archive after the first follow-up sub-change ships.
