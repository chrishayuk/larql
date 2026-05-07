## Context

The parent change `cuda-and-rotorquant-kv` declared the
`kv-cache-rotorquant` capability with public API surface
(`KvFormat`, `quantize_k`, `dequantize_v_with_inverse_rotation`, the
`QuantizedKv` struct, the deferred-K invariant, etc.) and marked
every scenario `<!-- test: unbacked -->`. This sub-change implements
those entrypoints on the CPU and lights up the test annotations.

We deliberately do **not** vendor the upstream
`feature/planarquant-kv-cache` `.cu` files. Two reasons:

1. The vendored kernel surface is ~2k lines of CUDA C and a `nvcc`
   build is non-trivial cross-compile — particularly painful when
   we already have NVRTC working for our softmax kernel.
2. Owning a from-scratch reference makes the CUDA implementation a
   transparent re-write rather than a black-box port. We retain the
   right to port specific upstream optimisations later, but the
   reference oracle is ours.

The trade-off is that the CPU reference's quality is currently below
the upstream paper's published numbers (we hit ~0.95 cosine on
synthetic data; upstream reports ~0.99 on real attention tensors).
A targeted "tune codebooks + rotation table" sub-change will close
this gap before the integration test in `deploy-compose-end-to-end`
needs production-quality numbers.

## Goals / Non-Goals

**Goals:**

- A new workspace crate with the public API exactly as the parent
  change spec declares.
- Round-trip tests pass for every (format × K|V) combo with cosine
  ≥ 0.95 on Gemma 4B-shaped blocks.
- The crate compiles standalone (no `larql-compute` dep).
- A `cuda` feature flag that's a no-op stub (all entrypoints route
  through the CPU path) so future work has a known integration
  point.

**Non-Goals:**

- GPU acceleration of any kind (next sub-change).
- Bit-exact parity with the upstream Triton implementation (next-
  next sub-change tunes codebooks + rotation tables).
- Disk format for `QuantizedKv` (the attention-service-routes
  change defines the wire format).
- 4-bit codebook quality on its own — Iso4/Planar4 work but
  improvements are tracked separately.

## Decisions

### D1 — New workspace member, MIT-licensed

`crates/larql-rotorquant` joins the workspace as a top-level member.
MIT (matches the upstream RotorQuant license, simpler to reason about
than dual-licensing). The crate has zero LARQL deps so it can be
extracted to a sibling repo later — same pattern as `model-compute`.

### D2 — Absmax row scaling, uniform codebook in [-1, 1]

Three quantisation strategies considered:

1. **Lloyd-Max codebook tuned for unit-Gaussian** (upstream's choice):
   high quality but requires per-dataset tuning + an offline training
   loop.
2. **L2 normalize + Lloyd-Max**: same trouble as #1 plus a
   normalisation factor that varies per row.
3. **Absmax + uniform codebook in [-1, 1]**: every row is scaled so
   the largest component is exactly 1 (or -1); the codebook is fixed
   uniform mid-points. Round-trip quality scales with the codebook
   size; rotation-decorrelation still helps because rotated blocks
   have lower per-coordinate variance.

Chose **option 3**. Cosine ≥ 0.95 is enough for the reference oracle;
production tuning lands later when we have real K/V attention tensors
to benchmark against.

### D3 — Pre-tabulated rotation tables

8 evenly-spaced Givens angles in [0, π/2) for Planar, 16
unit-quaternion samples around (1,1,1)/√3 for Iso. The "best
rotation" search is brute-force per block (8 or 16 candidates × 2 or
4 multiplies each = ~50 flops per block — negligible). A future
tuning change replaces these with empirically-derived sets.

### D4 — Inverse rotation on dequant for both K and V

Mathematically, recovering the unrotated values requires the
inverse of the forward rotation regardless of whether you call the
tensor K or V. The function name
`dequantize_v_with_inverse_rotation` is **a contract reminder**, not
a maths branch — its function body is the same as `dequantize_k`
internally. The upstream bug (commit `6e5a4aa`) was V dequant
applying the FORWARD rotation; preserving the explicit "with inverse
rotation" suffix forecloses that mistake at the API level.

### D5 — Public API stable across CPU / CUDA paths

The `cuda` feature flag changes the implementation but not the
signatures. Callers can flip the feature off (CPU only) or on
(currently routes through CPU; later: through PTX kernels) without
code changes. This matches the convention from `larql-compute`'s
`metal` and `cuda` features.

### D6 — Bit-packed codes layout

3-bit and 4-bit codes pack LSB-first into `Vec<u8>` row-major. A
helper pair `pack_code` / `unpack_code` does 16-bit-window slicing
to handle codes spanning byte boundaries cleanly. This keeps the
storage footprint exactly `(n_rows × head_dim × bits + 7) / 8`
bytes, matching the parent change's compression-ratio scenario.

## Risks / Trade-offs

- **Risk: cosine recovery below upstream paper's numbers.** Our
  ~0.95 threshold vs upstream's ~0.99. → Mitigation: that gap is
  inherent to our smaller rotation table + uniform codebook. Phase
  2 of `rotorquant-attention-integration` introduces an offline
  tuning step that lifts the rotation/codebook quality before we
  use this in production benchmarks.
- **Risk: brute-force per-block rotation search is O(rotation_count)**.
  At 16 quaternions × 4 components × small const, that's tens of ns
  per block. For a Gemma 4B head_dim=320 row, ~80 blocks × 16 trials
  ≈ 5 µs/row on CPU. Acceptable for the reference oracle.
- **Risk: codes layout drift.** Changing pack format silently
  invalidates on-disk snapshots. → Mitigation: bake the layout
  version into `QuantizedKv` once we ship the snapshot wire format
  in `attention-service-routes`.

## Migration Plan

Land. The crate is unused by other workspace members; the next
sub-change `rotorquant-attention-integration` adds it as a dep to
`larql-inference` and routes the KV cache through it.

Rollback: revert. Workspace builds cleanly on either side of this
commit.

## Open Questions

- **Q1: Should the codebook be runtime-tunable?** Currently it's a
  `const` array. A future profile tool could tune from real K/V
  histograms. Recommendation: leave the const for v1; add a
  `with_codebook` constructor in a future change.
- **Q2: f16 codes?** All current API is f32. A `dequantize_*_f16`
  variant would save a copy in the attention path. Recommendation:
  add when the integration sub-change actually wires this in.
