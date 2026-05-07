## Context

The `KvStrategy` trait in `kv-cache-benchmark` is the bench harness's
interface to every compression strategy. It has a tight contract:

```rust
pub trait KvStrategy {
    fn name(&self) -> &str;
    fn encode(&self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<u8>;
    fn decode(&self, encoded: &[u8], num_vectors: usize, dim: usize)
              -> (Vec<Vec<f32>>, Vec<Vec<f32>>);
    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize;
}
```

`larql-rotorquant`'s API is structured around `quantize_k(format, &[f32],
n_rows, head_dim)` returning a `QuantizedKv` struct. Bridging the two
requires:

1. flattening `Vec<Vec<f32>>` → row-major `Vec<f32>`,
2. running `quantize_k` and `quantize_v` separately,
3. serialising both `QuantizedKv` halves into a single `Vec<u8>` for
   the benchmark's wire,
4. doing the inverse on decode.

## Goals / Non-Goals

**Goals:**

- Wire `larql-rotorquant` to the existing benchmark harness so all
  four formats (Iso3 / Planar3 / Iso4 / Planar4) have a `KvStrategy`
  impl.
- A single binary wire format that round-trips the per-side
  `(codes, norms, rotation_indices)` triple.
- Inline tests that prove the strategy compiles + decodes cleanly
  through the existing `run_strategy_benchmark` entrypoint.

**Non-Goals:**

- Integration with `larql-inference`. The KV cache there is a
  separate type with separate concerns (KV-surgery, deferred-K,
  per-layer slicing).
- Production-quality compression numbers. Our reference CPU
  implementation has lower cosine recovery than the upstream
  paper; tuning is tracked in
  `rotorquant-attention-integration`.
- Zero-copy serialisation. The wire format does explicit copies
  for clarity; future work can introduce bytemuck-backed views.

## Decisions

### D1 — Four constructors, one struct

`RotorQuantStrategy::iso3() / planar3() / iso4() / planar4()`
all return `RotorQuantStrategy` with different `format` and
`name` fields. Avoids a generic that would force every consumer
to spell the format type. Consumers can also construct directly
via `RotorQuantStrategy { format, name }` if they want.

### D2 — Hand-rolled binary wire format, not bincode/serde

Three options considered:

1. **bincode** — requires `serde` derives on `QuantizedKv` plus a
   serde dep that the rotorquant crate doesn't currently have.
2. **postcard / msgpack** — same trade-off plus a less well-known
   format.
3. **Hand-rolled little-endian** — ~30 lines, no extra deps,
   transparent layout.

Chose option 3. The wire format is benchmark-internal; the on-disk
KV-snapshot format used by the future `attention-service-routes`
sub-change is its own concern and probably *will* use a serde-based
format.

### D3 — Pad-to-block bail-out on bad shapes

If the harness ever calls a strategy with `head_dim` that doesn't
divide cleanly by the format's block size, encode returns an empty
`Vec<u8>`. Decode sees the empty buffer and fills zeros. The
benchmark records the resulting `cosine_sim ≈ 0` in the metrics
table. This degrades gracefully rather than panicking — important
for harness loops that iterate many configs.

### D4 — `memory_bytes` ignores serialisation overhead

The wire format has ~24 bytes of metadata per side. For benchmarks
the relevant number is the on-device or on-disk KV footprint, not
the per-call wire cost. So `memory_bytes` calculates from the
format's bits + block size + rotation table only. This matches what
real consumers (the attention service) would store.

## Risks / Trade-offs

- **Risk: harness assumes all strategies preserve `dim` exactly.**
  Our rotation table padding may shrink/expand the recovered
  `head_dim`. → Mitigation: `reshape` pads / truncates to
  `num_vectors * dim` so the harness's downstream metrics see the
  shape it expected.
- **Risk: cosine drops to 0 silently when block size doesn't
  divide.** → Mitigation: the benchmark's standard accuracy table
  surfaces this directly. Documentation in the strategy doc-comment
  notes the constraint.
- **Risk: serialisation overhead distorts memory numbers at small
  seq_len.** → Mitigation: not relevant — `memory_bytes` is
  analytical (no encode call), and uses the format's compressed
  footprint per row × n_rows.

## Migration Plan

Land. The bench's accuracy suite picks up the new strategies on its
next run. The `rotorquant-attention-integration` sub-change can
reuse this strategy as a known-good oracle.

Rollback: revert. The benchmark works without RotorQuant rows.

## Open Questions

- **Q1: Should we expose `RotorQuantStrategy` from `lib.rs`'s
  `prelude`-style re-export?** Currently consumers `use
  kv_cache_benchmark::rotorquant::RotorQuantStrategy`. Adding to
  the crate's top-level re-exports would tidy call sites at the
  cost of a slightly more crowded `kv_cache_benchmark::*`.
- **Q2: f16 codes in the wire format.** Currently norms are f32.
  f16 would halve the per-row metadata. **Recommendation**: f16
  in a future change once we have measured production impact.
