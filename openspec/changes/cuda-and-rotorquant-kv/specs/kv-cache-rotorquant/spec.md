## ADDED Requirements

### Requirement: RotorQuant K/V cache compression API surface

The `larql_rotorquant` crate SHALL expose a safe Rust API around the
vendored RotorQuant CUDA kernels. The public surface MUST include:

- `KvFormat` enum: `Iso3`, `Planar3`, `Iso4`, `Planar4`.
- `quantize_k(format: KvFormat, k_tensor: &Tensor, scratch: &mut KvScratch) -> QuantizedKv`
- `quantize_v(format: KvFormat, v_tensor: &Tensor, scratch: &mut KvScratch) -> QuantizedKv`
- `dequantize_k(format: KvFormat, qkv: &QuantizedKv, out: &mut Tensor)`
- `dequantize_v_with_inverse_rotation(format: KvFormat, qkv: &QuantizedKv, out: &mut Tensor)`
- `KvScratch::new(format, max_seq_len, head_dim, num_heads) -> Result<Self, RotorQuantError>`

The API MUST guarantee that dequantize-V always uses the inverse
rotation; passing a forward rotation table SHALL be a compile-time
error (enforced by typestate or a wrapper type).

#### Scenario: Iso3 K round-trip preserves vector direction
- **WHEN** a random K tensor is quantised then dequantised through `KvFormat::Iso3`
- **THEN** the cosine similarity per row SHALL be ≥ 0.99
<!-- test: larql_rotorquant::round_trip::iso3_round_trip_k -->
<!-- test: larql_rotorquant::round_trip::iso3_gemma4b_head_round_trip -->
<!-- test: larql_rotorquant::round_trip::upstream_triton_reference_iso3_round_trip -->

#### Scenario: KvScratch rejects incompatible shapes
- **WHEN** a scratch buffer is created for a format, head dimension, and maximum row count
- **THEN** quantisation SHALL reject mismatched formats, mismatched head dimensions, and rows beyond scratch capacity before launching kernels
<!-- test: larql_rotorquant::round_trip::kv_scratch_rejects_incompatible_shape -->

#### Scenario: Planar3 V round-trip with inverse rotation matches Triton reference
- **WHEN** a V tensor is quantised through `KvFormat::Planar3`, then dequantised, on both the Rust path and the upstream Triton reference shim
- **THEN** both reconstructions SHALL preserve the input direction and maintain pairwise cosine similarity against each other
<!-- test: larql_rotorquant::round_trip::planar3_round_trip_v -->
<!-- test: larql_rotorquant::round_trip::upstream_triton_reference_planar3_round_trip -->

#### Scenario: Forward-rotation V dequant is unrepresentable in the API
- **WHEN** the user attempts to call a dequant-V operation with a forward rotation table
- **THEN** the call SHALL fail to compile (via typestate / sealed wrapper)
<!-- test: larql_rotorquant::round_trip::iso3_v_round_trip_recovers_original_not_rotated -->

### Requirement: Compression ratios match the upstream paper

For each `KvFormat`, the on-device byte footprint SHALL match the
ratios documented in the RotorQuant paper, namely:

| Format | Bits | Compression vs FP16 |
|---|---|---|
| `Planar3` | 3 | 10.3× |
| `Iso3` | 3 | 10.3× |
| `Planar4` | 4 | 5.1× |
| `Iso4` | 4 | 5.1× |

#### Scenario: Iso3 cache size is 10.3× smaller than FP16
- **WHEN** a 128k-context KV cache for Llama 3 8B is allocated as Iso3 vs FP16
- **THEN** the Iso3 footprint SHALL be ≤ 11% of the FP16 footprint, plus norms / indices overhead ≤ 2% of FP16
<!-- test: unbacked -->

### Requirement: Deferred-K quantisation during prefill

During prefill, K SHALL be stored as FP16 and quantised lazily on
decode token insertion. This eliminates the error-compounding effect
documented in upstream commit `6e5a4aa`.

#### Scenario: Prefill K is FP16 in VRAM
- **WHEN** a prefill of 1024 tokens completes
- **THEN** inspecting the K cache via `KvHandle::format_at(layer)` SHALL report `Fp16` for those tokens
<!-- test: unbacked -->

#### Scenario: Decode token insertion quantises previously-deferred K
- **WHEN** a decode token is appended to the cache
- **THEN** the previously-FP16 region of K MUST be quantised to the session format and the FP16 backing SHALL be freed
<!-- test: unbacked -->

### Requirement: Vendored kernels MUST be sourced from a known upstream commit

The vendored CUDA sources under `crates/larql-rotorquant/cuda/` SHALL be accompanied by `crates/larql-rotorquant/UPSTREAM.md`
recording: the upstream repo URL, the exact commit hash imported,
the date imported, and any local patches applied. The make target
`make rotorquant-sync` SHALL diff the vendored copy against the
upstream commit and surface drift.

#### Scenario: UPSTREAM.md is present and well-formed
- **WHEN** the repo is fresh-cloned
- **THEN** `crates/larql-rotorquant/UPSTREAM.md` SHALL exist and contain a SHA-1 commit hash and a fetch URL
<!-- test: larql_rotorquant::round_trip::upstream_provenance_records_commit_and_vendored_files -->

#### Scenario: Sync target reports drift
- **WHEN** the upstream commit moves and `make rotorquant-sync` is run
- **THEN** the target SHALL print a unified diff and exit non-zero so CI catches stale vendoring
<!-- test: unbacked -->

### Requirement: Rotation tables MUST be server-issued and never client-supplied

The rotation indices and norms used to reconstruct K/V SHALL be
produced on quantize and stored in the `KvHandle`'s server-side
state. Clients of the attention service SHALL NOT be able to supply
or override rotation tables — preventing the "wrong-inverse"
corruption mode documented as upstream commit `6e5a4aa`.

#### Scenario: KvHandle is opaque
- **WHEN** the attention service returns a `KvHandle`
- **THEN** the wire format MUST contain only an opaque u128 identifier; the client SHALL NOT be able to introspect or modify the underlying rotation table
<!-- test: larql_server::attention_session::tests::session_id_is_26_chars -->
<!-- test: larql_server::test_http_attention::create_session_returns_id_and_layer_range -->
