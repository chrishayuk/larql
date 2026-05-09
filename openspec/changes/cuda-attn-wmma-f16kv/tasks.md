# cuda-attn-wmma-f16kv — tasks

## Phase 1A: storage type

- [x] 1.1 `CudaKvLayer::{k, v}: CudaSlice<half::f16>`.
- [x] 1.2 `CudaKvCache::new_device` allocates with
      `stream.alloc_zeros::<half::f16>(n)`.

## Phase 1B: kernel f16 read/write helpers

- [x] 1.3 `FUSED_DECODE_ATTN_SRC` — `unsigned short* k_cache,
      unsigned short* v_cache` + `ld_kvcache` / `st_kvcache`
      PTX inlines (`cvt.f32.f16` / `cvt.rn.f16.f32`).
- [x] 1.4 `KV_CACHE_WRITE_SEQ_SRC` — same change; `st_kv_seq`
      writer.
- [x] 1.5 `FUSED_PREFILL_ATTN_SRC` — same change; `ld_kvc_pf`
      reader. Cache args become `const unsigned short*`.

## Phase 1C: Rust wrapper signature plumbing

- [x] 1.6 `fused_decode_attention_device_kv` — `&mut
      CudaSlice<half::f16>` for cache args.
- [x] 1.7 `fused_decode_attention_device_kv_into` — same.
- [x] 1.8 `fused_prefill_attention_seq_device` — same.
- [x] 1.9 Host-wrapper paths
      (`fused_decode_attention`, `fused_decode_attention_device`)
      internally convert `&[f32]` → `Vec<half::f16>` → htod, run
      the kernel, dtoh f16 → `Vec<f32>` for return.

## Phase 1D: host-boundary helpers

- [x] 1.10 `CudaBackend::htod_f32_as_f16_into_slice`.
- [x] 1.11 `CudaBackend::dtoh_f16_as_f32`.
- [x] 1.12 `populate_kv_layer` uses
      `htod_f32_as_f16_into_slice`.
- [x] 1.13 Legacy `decode_token` host-fallback path uses both
      helpers for cache R/W.

## Phase 1E: tests + bench

- [x] 1.14 `fused_decode_attention_matches_cpu_reference` —
      bound bump from 1e-3 / 1e-6 to 5e-3 (both K and V
      caches), with comment explaining f16 quant noise.
- [x] 1.15 All 200+ unit + integration tests pass.
- [x] 1.16 Bench gate (5-run avg, RTX 4090, Gemma 3 4B Q4_K,
      with `LARQL_CUDA_PREFILL_TENSOR_CORES=1`):
      pre-change 8.22 ms / 121.7 tok/s →
      post-change **8.04 ms / 124.4 tok/s**. Run-to-run variance
      collapsed from 8.05–8.52 ms to 8.03–8.05 ms.
- [x] 1.17 `larql run` produces identical generated text vs
      pre-change.

## Phase 2 (separate change, deferred)

- [ ] 2.1 WMMA fragment-based attention compute.
- [ ] 2.2 Tensor Core dispatch for K^T @ Q and (softmax) @ V.
- [ ] 2.3 ~3-5 days of focused CUDA work; benefits long-context
      decode most.

## Documentation + archive

- [x] 3.1 `proposal.md` documents the rotorquant relationship
      and the Phase 2 deferral.
- [ ] 3.2 Archive Phase 1 when reviewed; Phase 2 lands as
      its own change.
