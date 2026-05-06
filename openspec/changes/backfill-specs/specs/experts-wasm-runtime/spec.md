## ADDED Requirements

### Requirement: Expert WASM ABI

Every expert module SHALL be a `wasm32-wasip1` `cdylib` that exposes
exactly the following C-ABI exports: `larql_call(op_ptr: u32,
op_len: u32, args_ptr: u32, args_len: u32) -> u32` returning a pointer
to a null-terminated JSON `ExpertResult` (or `0` if the expert does
not handle the requested op), `larql_metadata() -> u32` returning a
pointer to a null-terminated JSON `ExpertMetadata`, and the host-side
memory helpers `larql_alloc(len: u32) -> u32` and
`larql_dealloc(ptr: u32, len: u32)`. All payloads on the WASM
boundary MUST be UTF-8 JSON; op names MUST be language-neutral
identifiers; argument keys and result values MUST be typed JSON.
Hosts MUST pair every `larql_alloc` with a `larql_dealloc` of the
exact same length.

#### Scenario: Registry dispatches a structured op call across the ABI
- **WHEN** `registry.call("gcd", &json!({"a": 144, "b": 60}))` is invoked on a registry loaded from `crates/larql-experts/target/wasm32-wasip1/release`
- **THEN** the call SHALL return `Some(ExpertResult { value: 12, expert_id: "arithmetic", op: "gcd", .. })`
<!-- test: larql_inference::test_experts::registry_dispatches_by_op -->

#### Scenario: Unknown op returns None without consulting any expert
- **WHEN** `registry.call("definitely_not_an_op", &json!({}))` is invoked
- **THEN** the registry SHALL return `None` and SHALL NOT invoke any expert's `larql_call`
<!-- test: larql_inference::test_experts::registry_unknown_op_returns_none -->

#### Scenario: Every expert exposes well-formed metadata
- **WHEN** `registry.list()` is iterated for a freshly loaded registry
- **THEN** each `ExpertMetadata` SHALL have a non-empty `id`, a non-empty `version`, and at least one `OpSpec` in `ops`
<!-- test: larql_inference::test_experts::registry_all_experts_have_metadata -->

### Requirement: Compile-once, instantiate-on-demand loading

`ExpertRegistry::load_dir(dir)` SHALL compile every `.wasm` file in
`dir` (or load a precompiled `.cwasm` cache from a sibling path), but
MUST NOT keep a live `Store` + `Instance` resident per expert at load
time. A live instance SHALL be created on the first `call()` per
expert and reused for subsequent calls. `evict_all()` MUST drop every
live instance without unloading the compiled modules so re-materialising
on the next call is microsecond-scale, not millisecond-scale.

#### Scenario: Newly loaded experts report zero memory pages until called
- **WHEN** a registry is loaded and `wasm_infos()` is read before any `call()`
- **THEN** every entry SHALL report `instantiated == false` and `memory_pages == 0`
<!-- test: larql_inference::test_experts::registry_experts_are_lazy_instantiated -->

#### Scenario: First call materialises the linear memory
- **WHEN** a single op is dispatched against an otherwise idle registry
- **THEN** the targeted expert's `wasm_info` SHALL flip to `instantiated == true` with `memory_pages > 0`, while untouched experts SHALL remain at zero
<!-- test: larql_inference::test_experts::registry_experts_are_lazy_instantiated -->

#### Scenario: Compiled-module cache is written and reused
- **WHEN** an expert is loaded for the first time, then loaded again from the same directory
- **THEN** the loader SHALL write a sibling `.cwasm` file on the first load and SHALL deserialize that artifact (rather than recompile) on the second load when its mtime is at least as new as the source `.wasm`
<!-- test: larql_inference::test_experts::module_cache_file_is_written_and_reused -->

### Requirement: Op→expert dispatch index

`ExpertRegistry` SHALL maintain an `op name → expert index` map built
at load time from each expert's advertised `ops`. Dispatch by op name
MUST be O(1). When two experts advertise the same op name, the
expert with the lower `tier` (sorted earlier) SHALL win and the
higher-tier expert SHALL be shadowed for that op only. `ops()` MUST
return every dispatchable op name in sorted order.

#### Scenario: Tier order determines which expert wins shared op names
- **WHEN** a registry is loaded from a directory containing both tier-1 and tier-2 experts that share an op name
- **THEN** `list()` SHALL return experts ordered by ascending tier and the lower-tier expert SHALL be the dispatch target for the shared op
<!-- test: larql_inference::test_experts::registry_load_dir_tier_order -->

#### Scenario: Ops are discoverable in sorted order
- **WHEN** `registry.ops()` is read on a freshly loaded registry
- **THEN** the result SHALL list every advertised op name and SHALL be sorted
<!-- test: larql_inference::test_experts::registry_ops_are_discoverable -->

#### Scenario: Dispatch routes by op name with no English parsing
- **WHEN** `registry.call("days_between", &json!({"start": "2025-01-01", "end": "2025-12-31"}))` is invoked
- **THEN** the registry SHALL route directly to the `date` expert via the op→expert index without scanning other experts
<!-- test: larql_inference::test_experts::registry_dispatches_by_op -->

### Requirement: Per-call memory stability

Every `registry.call()` SHALL allocate three buffers in the target
module's linear memory (the op name, the args JSON, and the result
JSON) and SHALL free all three via `larql_dealloc` before returning.
Linear memory SHALL stay flat across millions of calls; no permanent
growth SHALL accumulate from successful invocations.

#### Scenario: Linear memory is stable across many sequential calls
- **WHEN** a single op is dispatched many thousands of times against a single expert
- **THEN** the expert's linear-memory page count SHALL remain bounded, with no per-call leak
<!-- test: larql_inference::test_experts::registry_memory_stable_across_many_calls -->

#### Scenario: Eviction drops live instances without losing compiled modules
- **WHEN** `evict_all()` is invoked after the registry has serviced calls
- **THEN** every expert SHALL report `instantiated == false` and a subsequent `call()` SHALL succeed without recompilation
<!-- test: larql_inference::test_experts::registry_experts_are_lazy_instantiated -->

### Requirement: Result and metadata JSON shapes

`ExpertResult` SHALL serialise as a JSON object with the keys
`value`, `confidence`, `latency_ns`, `expert_id`, and `op`.
`ExpertMetadata` SHALL serialise with the keys `id`, `tier`,
`description`, `version`, and `ops`. Each entry of `ops` SHALL be an
`OpSpec` carrying `name` and `args` (a list of JSON object keys the
op reads). The host SHALL parse these shapes on every call and on
load, and SHALL surface a typed Rust struct to callers — not raw
JSON.

#### Scenario: ExpertResult round-trips through host call
- **WHEN** `registry.call("add", &json!({"a": 2, "b": 3}))` returns a result
- **THEN** the result SHALL have `value == 5`, `expert_id == "arithmetic"`, and `op == "add"`
<!-- test: larql_inference::test_experts::arithmetic_add -->

#### Scenario: Op argument schemas are advertised in metadata
- **WHEN** `registry.list()` is iterated and the `gcd` op is located on the arithmetic expert
- **THEN** the op's `args` field SHALL list the JSON object keys the op reads (e.g. `["a", "b"]`)
<!-- test: larql_inference::test_experts::registry_all_experts_have_metadata -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::experts::session::tests::**::* -->
<!-- test: larql_inference::experts::mask::tests::**::* -->
<!-- test: larql_inference::experts::parser::tests::**::* -->
