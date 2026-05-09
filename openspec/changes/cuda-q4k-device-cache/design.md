## Context

`cuda-resident-q4k-matvec` removed CPU dequant and f32 upload, but the direct
kernel still copies packed Q4_K bytes to the GPU for every call. The real bench
now spends about 3.6 s/token in CUDA forward and 1.5 s/token in LM-head. The
LM-head path repeatedly touches the same packed Q4_K matrix, making it the
clearest target for device residency.

## Goals / Non-Goals

**Goals:**

- Reuse device-resident packed Q4_K buffers across repeated matvec calls.
- Avoid one-CUDA-block-per-output-row launch geometry for large LM-head shapes.
- Keep the cache backend-local and transparent to existing callers.
- Improve real `larql bench --backends cuda` timing without changing vindex
  file formats or public compute traits.

**Non-Goals:**

- Evict cached weights or implement a VRAM budget in this slice.
- Cache f32 inputs or output buffers.
- Implement a Q4_K GPU top-k kernel; this change still reads scores back for
  top-k selection.

## Decisions

### D1 - Backend-local cache

`CudaBackend` owns the cache because it already owns the CUDA driver and has
the same lifetime as decode. The key is derived from the host byte slice
pointer, length, and a small first/last-byte fingerprint. Production vindex
weights are immutable mmaps, so pointer identity is stable and cheap.

### D2 - Hold the cache lock for launch

The first version keeps the cache lock while launching the kernel with a
borrowed `CudaSlice<u8>`. Decode is single-threaded for one backend today, so
this avoids unsafe raw pointer lifetimes. If multi-threaded CUDA decode lands,
the cache can move to `Arc<CudaSlice<u8>>` entries.

### D3 - Multiple rows per CUDA block

The Q4_K direct kernel computes four rows per CUDA block using a 2D block
layout. This reduces LM-head block count from roughly 128k blocks to roughly
32k blocks while keeping a simple per-row reduction tree.

## Risks / Trade-offs

- **Risk: VRAM growth.** Mitigation: packed Q4_K is much smaller than expanded
  f32 and fits the current 24 GB target for the local vindex.
- **Risk: stale cache if mutable caller reuses a pointer.** Mitigation: cache
  key includes a small fingerprint; public docs and specs limit this cache to
  immutable packed weights.
