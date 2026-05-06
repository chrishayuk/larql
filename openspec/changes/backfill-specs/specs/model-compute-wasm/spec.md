## ADDED Requirements

### Requirement: SolverRuntime engine and module compilation

The `model_compute::wasm::SolverRuntime` SHALL own a long-lived
`wasmtime::Engine` configured with `consume_fuel(true)`, expose
`SolverLimits` (per-call `fuel` budget and `memory_pages` cap) with
defaults of 100M fuel units and 256 pages (16 MiB), and provide
`compile(&[u8]) -> Result<Module, SolverError>` so callers can compile
WASM bytes once and reuse the resulting `Module` across many
`Session`s. Construction SHALL surface engine-creation failures as
`SolverError::Engine(_)` and module-compilation failures as
`SolverError::InvalidModule(_)`.

#### Scenario: Default runtime constructs an engine with default limits
- **WHEN** `SolverRuntime::new()` is called
- **THEN** the runtime SHALL be constructed successfully and `limits()` SHALL return `SolverLimits { fuel: 100_000_000, memory_pages: 256 }`
<!-- test: model_compute::wasm_roundtrip::echo_roundtrip -->

#### Scenario: Custom limits are honored
- **WHEN** `SolverRuntime::with_limits(SolverLimits { fuel: 10_000, memory_pages: 16 })` is called
- **THEN** the runtime SHALL be constructed and the configured fuel cap SHALL bound subsequent solve calls
<!-- test: model_compute::wasm_roundtrip::fuel_cap_stops_infinite_loop -->

#### Scenario: WAT fixture compiles into a reusable module
- **WHEN** an echo WAT fixture is compiled via `runtime.compile(&bytes)`
- **THEN** the call SHALL return `Ok(Module)` and that module SHALL be reusable across multiple sessions
<!-- test: model_compute::wasm_roundtrip::echo_two_sessions_isolated -->

### Requirement: Per-call session isolation

`SolverRuntime::session(&Module) -> Result<Session, SolverError>` MUST
construct a fresh `wasmtime::Store` parameterized by the runtime's
`SolverLimits`, install a `StoreLimitsBuilder` with
`memory_size = pages × 64 KiB`, set the per-call fuel budget via
`Store::set_fuel`, and instantiate the module with no shared host
imports. No state SHALL leak between two sessions backed by the same
compiled module.

#### Scenario: Echo round-trip succeeds end-to-end
- **WHEN** an echo session is given a 20-byte input via `session.solve(input)`
- **THEN** the returned `Vec<u8>` SHALL equal the input
<!-- test: model_compute::wasm_roundtrip::echo_roundtrip -->

#### Scenario: Two sessions over the same module are isolated
- **WHEN** two sessions are opened on the same compiled echo module and each is given a different input
- **THEN** each session SHALL return only its own input — no state SHALL bleed between them
<!-- test: model_compute::wasm_roundtrip::echo_two_sessions_isolated -->

#### Scenario: Fuel remaining strictly decreases across a solve call
- **WHEN** `session.fuel_remaining()` is sampled before and after `session.solve(b"hello")`
- **THEN** the post-call value SHALL be strictly less than the pre-call value
<!-- test: model_compute::wasm_roundtrip::fuel_remaining_decreases_after_call -->

### Requirement: alloc-write-solve-read ABI

`Session::solve(&[u8]) -> Result<Vec<u8>, SolverError>` MUST implement
the canonical alloc-write-solve-read ABI: (1) call the guest export
`alloc(len: u32) -> i32` to obtain an input pointer, (2) write the
caller's bytes to guest memory at that pointer, (3) call
`solve(ptr: i32, len: u32) -> u32` and treat any non-zero status as a
solve failure, (4) call `solution_ptr()` and `solution_len()` to
locate the output, and (5) copy the output back to host memory. Any
missing export SHALL surface as `SolverError::MissingExport(name)`,
and a non-zero solve status SHALL surface as
`SolverError::SolveFailed(status)`.

#### Scenario: Missing alloc export is reported clearly
- **WHEN** a session is opened on a module that has only a `memory` export and `session.solve(b"")` is called
- **THEN** the call SHALL return `Err(SolverError::MissingExport("alloc"))`
<!-- test: model_compute::wasm_roundtrip::missing_export_errors_clearly -->

#### Scenario: Non-zero solve status surfaces as SolveFailed
- **WHEN** the guest's `solve` returns status `42`
- **THEN** the host SHALL return `Err(SolverError::SolveFailed(42))`
<!-- test: model_compute::wasm_roundtrip::nonzero_solve_status_reported -->

#### Scenario: Multi-page input round-trips through the ABI
- **WHEN** a 48 000-byte input is fed into the echo session
- **THEN** the returned bytes SHALL equal the input across the multi-page region
<!-- test: model_compute::wasm_roundtrip::large_input_crosses_multiple_pages -->

### Requirement: Fuel and memory caps stop runaway guests

The host SHALL enforce both a fuel budget and a linear-memory page
cap. A guest that exhausts the fuel budget SHALL surface as
`SolverError::FuelExhausted` or `SolverError::Trap`. A guest that
attempts to grow memory past the configured cap SHALL surface as
`SolverError::Trap` (the configured `StoreLimits` cause `memory.grow`
to return `-1`, and the canonical guest pattern is to treat that as
`unreachable`). Neither condition SHALL wedge or crash the host
process.

#### Scenario: Infinite-loop guest is stopped by fuel cap
- **WHEN** a fuel budget of 10 000 is configured and a guest with `(loop $forever (br $forever))` is invoked
- **THEN** the call SHALL return `Err(SolverError::FuelExhausted { .. })` or `Err(SolverError::Trap { .. })`
<!-- test: model_compute::wasm_roundtrip::fuel_cap_stops_infinite_loop -->

#### Scenario: memory.grow past the cap is rejected as a Trap
- **WHEN** a memory cap of 16 pages is configured and a guest tries to `memory.grow` by 1024 pages then traps via `unreachable` on `-1`
- **THEN** the call SHALL return `Err(SolverError::Trap { .. })`
<!-- test: model_compute::wasm_roundtrip::memory_cap_rejects_grow -->

### Requirement: SolverError variants

`SolverError` SHALL distinguish at minimum the variants `Engine`,
`InvalidModule`, `Instantiate`, `MissingExport`, `ExportSignature`,
`FuelExhausted { budget }`, `MemoryExceeded { pages }`,
`Trap { call, trap }`, `OutOfMemory`, `InvalidGuestPointer`, and
`SolveFailed(u32)`. Conversion from `wasmtime::Error` SHALL be
implemented and SHALL default to `SolverError::Engine` for unclassified
upstream errors.

#### Scenario: Bad guest pointer surfaces as InvalidGuestPointer
- **WHEN** the guest's `alloc` returns a pointer + length that exceeds the linear memory size
- **THEN** the host SHALL return `Err(SolverError::InvalidGuestPointer(_))` rather than reading past the end
<!-- test: model_compute::wasm_roundtrip::echo_roundtrip -->

#### Scenario: Missing export name is preserved in the error
- **WHEN** a guest is missing the `alloc` export and `session.solve(b"")` is called
- **THEN** the error SHALL be `SolverError::MissingExport(name)` with `name == "alloc"`
<!-- test: model_compute::wasm_roundtrip::missing_export_errors_clearly -->

#### Scenario: Non-zero status preserves the guest-supplied code
- **WHEN** the guest returns `42` from `solve`
- **THEN** the error SHALL be `SolverError::SolveFailed(42)` with the original `u32` preserved
<!-- test: model_compute::wasm_roundtrip::nonzero_solve_status_reported -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: model_compute::wasm_roundtrip::**::* -->
