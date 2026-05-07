## ADDED Requirements

### Requirement: CUDA backend MUST ship PTX kernels for Iso and Planar formats

The `larql-rotorquant` crate (with `--features cuda` on Linux + working CUDA driver) SHALL ship four PTX kernels — one quantize + one dequantize per family (Iso, Planar) — compiled via cudarc NVRTC and cached under `$XDG_CACHE_HOME/larql/cudarc/<arch>/rotorquant-{kernel}.cubin`.

#### Scenario: kernel compile cached on disk
- **WHEN** `quantize_k(KvFormat::Iso3, ...)` is called for the first time after a fresh cache wipe
- **THEN** a `rotorquant-iso_quantize.cubin` SHALL appear in the cache directory
<!-- test: unbacked -->

### Requirement: CUDA round-trip MUST cosine-match the CPU reference within 1e-3

For every (format, kind) combo a CUDA-side round-trip `quantize_k → dequantize_k` (or `quantize_v → dequantize_v_with_inverse_rotation`) SHALL produce the same result as the CPU reference within 1e-3 absolute element difference on synthetic Gemma 4B-shaped inputs (head_dim = 320, n_rows up to 64).

#### Scenario: Iso3 K round-trip CPU ↔ CUDA parity
- **WHEN** the same synthetic K tensor is round-tripped through the CPU reference and the CUDA path
- **THEN** the maximum absolute element difference SHALL be ≤ 1e-3 and cosine ≥ 0.99
<!-- test: unbacked -->

#### Scenario: Iso3 V round-trip uses inverse rotation correctly
- **WHEN** a synthetic V tensor is quantised then dequantised via `dequantize_v_with_inverse_rotation` on the CUDA path
- **THEN** the recovered V SHALL match the CPU equivalent within 1e-3
<!-- test: unbacked -->

#### Scenario: Planar3 round-trip parity
- **WHEN** a synthetic K tensor is round-tripped through both paths via `KvFormat::Planar3`
- **THEN** the result SHALL match within 1e-3
<!-- test: unbacked -->

### Requirement: CUDA path MUST be ≥ 10× faster than CPU reference

The CUDA quantize + dequantize round-trip SHALL complete in ≤ 1/10 of the CPU reference's wall-clock time on RTX 4090 at Gemma 4B head_dim = 320 with n_rows = 32.

#### Scenario: throughput beats 10× on Gemma 4B head shape
- **WHEN** the round-trip is timed on both paths at n_rows = 32, head_dim = 320, format = Iso3
- **THEN** the CUDA wall-clock SHALL be ≤ 10% of the CPU wall-clock
<!-- test: unbacked -->

### Requirement: Capability::KvCompressionRotorQuant MUST flip on CudaBackend

After this change, `CudaBackend::supports(Capability::KvCompressionRotorQuant)` SHALL return `true` if and only if the PTX kernels have successfully compiled or been loaded from cache. A failed compile SHALL leave the bit `false` so dispatch falls back to CPU transparently.

#### Scenario: capability bit reflects kernel availability
- **WHEN** the CUDA backend boots and `supports(KvCompressionRotorQuant)` is read
- **THEN** the result SHALL be `true` on a healthy host and `false` if NVRTC compilation fails
<!-- test: unbacked -->
