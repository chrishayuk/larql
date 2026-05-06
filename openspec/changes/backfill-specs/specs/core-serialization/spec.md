## ADDED Requirements

### Requirement: Format enumeration and detection

`larql_core::io::format::Format` SHALL enumerate every supported
serialization format (JSON, CSV, packed, msgpack) and SHALL provide
extension-based detection so callers can pick a format from a path
without explicit selection. Unknown extensions SHALL be reported
rather than silently defaulted.

#### Scenario: Format from path resolves known extensions
- **WHEN** a path with a `.json`, `.csv`, `.bin` (packed), or `.msgpack` extension is passed to the format detector
- **THEN** the corresponding `Format` variant SHALL be returned
<!-- test: larql_core::test_roundtrip::test_format_from_path -->

#### Scenario: CSV format is selectable as a first-class format
- **WHEN** the format enum is queried for CSV
- **THEN** the CSV variant SHALL be exposed and SHALL drive CSV serialization without re-implementation
<!-- test: larql_core::test_new_algos::test_csv_format -->

### Requirement: JSON round-trip of graphs

`larql_core::io::json` SHALL serialize a `Graph` to and from a
serde-compatible JSON value, an in-memory byte buffer, and a file on
disk, preserving every edge's confidence, source, and metadata in all
three modes. The on-disk JSON document SHALL follow the documented
schema (top-level fields for edges, schema, and metadata).

#### Scenario: Round-trip via value, bytes, and file
- **WHEN** a graph is serialized via `to_value`, `to_bytes`, or `to_file` and then deserialized again
- **THEN** the resulting graph SHALL equal the original under structural comparison
<!-- test: larql_core::test_roundtrip::test_json_value_roundtrip -->
<!-- test: larql_core::test_roundtrip::test_json_bytes_roundtrip -->
<!-- test: larql_core::test_roundtrip::test_json_file_roundtrip -->

#### Scenario: Confidence, source, and metadata survive a JSON round-trip
- **WHEN** an edge with non-default confidence, source, or metadata is round-tripped through JSON
- **THEN** all three fields SHALL be preserved on the deserialized edge
<!-- test: larql_core::test_roundtrip::test_json_preserves_confidence -->
<!-- test: larql_core::test_roundtrip::test_json_preserves_source -->
<!-- test: larql_core::test_roundtrip::test_json_preserves_metadata -->

#### Scenario: Empty graph and document structure
- **WHEN** an empty graph is serialized and when any graph is serialized to JSON
- **THEN** the round-trip SHALL produce an empty graph in the empty case and the resulting JSON SHALL contain the documented top-level structure
<!-- test: larql_core::test_roundtrip::test_empty_graph_roundtrip -->
<!-- test: larql_core::test_roundtrip::test_json_format_structure -->

### Requirement: CSV round-trip of graphs

`larql_core::io::csv` SHALL serialize and deserialize a graph as CSV
preserving the triple, confidence, and metadata that survive flat
encoding. Quoted fields containing commas, quotes, or whitespace
SHALL round-trip without corruption, and the CSV format SHALL
preserve numeric confidence values within the documented precision.

#### Scenario: CSV round-trip preserves edges and confidence
- **WHEN** a graph is written to CSV and read back
- **THEN** every edge SHALL be reconstructed and confidence SHALL match the original within the documented precision
<!-- test: larql_core::test_new_algos::test_csv_roundtrip -->
<!-- test: larql_core::test_new_algos::test_csv_preserves_confidence -->

#### Scenario: Quoted CSV fields are correctly escaped and parsed
- **WHEN** a graph contains entity or relation strings that include commas, quotes, or whitespace
- **THEN** CSV serialization SHALL escape them and deserialization SHALL recover the original strings
<!-- test: larql_core::test_new_algos::test_csv_roundtrip_quoted_fields -->

### Requirement: Packed binary format

`larql_core::io::packed` SHALL provide a compact binary format that
interns repeated strings, encodes per-edge metadata and injection
flags, and round-trips on disk. The reader SHALL reject malformed
input (bad magic number, invalid string-table offset, truncated
edge section, out-of-range string index, invalid metadata range, or
unsupported flags) without panicking, and SHALL preserve the full
range of supported source types and confidence precision.

#### Scenario: Round-trip basic, metadata, injection, and combined edges
- **WHEN** a graph is serialized and then deserialized through the packed format with no metadata, with metadata, with injection flags, or with both
- **THEN** the deserialized graph SHALL equal the original
<!-- test: larql_core::io::packed::test_roundtrip_basic -->
<!-- test: larql_core::io::packed::test_roundtrip_with_metadata -->
<!-- test: larql_core::io::packed::test_roundtrip_with_injection -->
<!-- test: larql_core::io::packed::test_roundtrip_with_metadata_and_injection -->

#### Scenario: Empty graph, source-type coverage, and string interning
- **WHEN** an empty graph is round-tripped, and **WHEN** a graph that exercises every supported `SourceType` is round-tripped, and **WHEN** the string table interns repeated values
- **THEN** the deserialized graph SHALL equal the original, every source type SHALL be preserved, and repeated strings SHALL share a single table entry
<!-- test: larql_core::io::packed::test_roundtrip_empty_graph -->
<!-- test: larql_core::io::packed::test_roundtrip_source_types -->
<!-- test: larql_core::io::packed::test_string_interning -->

#### Scenario: Confidence precision and on-disk file round-trip
- **WHEN** confidence values near the format's quantization boundary are written, and **WHEN** the format is round-tripped through a real file on disk
- **THEN** confidence SHALL be preserved within the documented precision and the on-disk graph SHALL reload to an equal graph
<!-- test: larql_core::io::packed::test_confidence_precision -->
<!-- test: larql_core::io::packed::test_file_roundtrip -->

#### Scenario: Malformed inputs return errors instead of panicking
- **WHEN** a packed payload has a wrong magic number, an invalid string-table offset, a truncated edge section, an out-of-range string index, an invalid metadata range, or unsupported flags
- **THEN** the reader SHALL return a `GraphError` and SHALL NOT panic
<!-- test: larql_core::io::packed::test_invalid_magic -->
<!-- test: larql_core::io::packed::test_invalid_string_table_offset_returns_error -->
<!-- test: larql_core::io::packed::test_truncated_edge_section_returns_error -->
<!-- test: larql_core::io::packed::test_out_of_range_string_index_returns_error -->
<!-- test: larql_core::io::packed::test_invalid_metadata_range_returns_error -->
<!-- test: larql_core::io::packed::test_unsupported_flags_return_error -->

### Requirement: MessagePack round-trip and size

`larql_core::io::msgpack` SHALL serialize and deserialize a graph
through both an in-memory byte buffer and a file on disk, preserving
confidence and metadata. The msgpack representation SHALL be
strictly smaller than the equivalent JSON representation, and the
JSON and msgpack formats SHALL be interoperable through a documented
JSON-to-msgpack conversion.

#### Scenario: msgpack round-trip via bytes and file
- **WHEN** a graph is serialized to msgpack bytes or to a file and read back
- **THEN** the resulting graph SHALL equal the original
<!-- test: larql_core::test_roundtrip::test_msgpack_bytes_roundtrip -->
<!-- test: larql_core::test_roundtrip::test_msgpack_file_roundtrip -->

#### Scenario: Confidence preservation and size advantage over JSON
- **WHEN** an edge with a non-default confidence is round-tripped through msgpack, and **WHEN** the same graph is encoded as both JSON and msgpack
- **THEN** confidence SHALL be preserved and the msgpack output SHALL be smaller than the JSON output
<!-- test: larql_core::test_roundtrip::test_msgpack_preserves_confidence -->
<!-- test: larql_core::test_roundtrip::test_msgpack_smaller_than_json -->

#### Scenario: JSON-to-msgpack conversion preserves the graph
- **WHEN** a graph is converted from JSON to msgpack and back to a graph
- **THEN** the resulting graph SHALL equal the original
<!-- test: larql_core::test_roundtrip::test_json_to_msgpack_roundtrip -->

### Requirement: Sequential checkpoint log

`larql_core::io::checkpoint` SHALL provide a sequential append-only
checkpoint log that lets a long-running engine flush incremental
edges to disk and replay them on start-up. Replay SHALL reconstruct
the graph in original insertion order, MUST tolerate a fresh empty
file, MUST append correctly across sessions without overwriting
prior records, and MUST preserve every edge's metadata.

#### Scenario: Write and replay reconstructs the graph
- **WHEN** edges are appended to a checkpoint and the file is replayed
- **THEN** the reconstructed graph SHALL contain every appended edge
<!-- test: larql_core::test_checkpoint::test_checkpoint_write_and_replay -->

#### Scenario: Append across sessions preserves prior records
- **WHEN** a checkpoint is appended to in a second session after being closed
- **THEN** the new edges SHALL be added without overwriting the prior records, and replay SHALL surface every edge from both sessions
<!-- test: larql_core::test_checkpoint::test_checkpoint_append_across_sessions -->

#### Scenario: Empty file replays as an empty graph
- **WHEN** replay is called on a fresh, empty checkpoint file
- **THEN** the operation SHALL succeed and yield an empty graph
<!-- test: larql_core::test_checkpoint::test_checkpoint_empty_file -->

#### Scenario: Metadata survives checkpoint replay
- **WHEN** edges with metadata are appended to a checkpoint and replayed
- **THEN** every metadata field SHALL be present on the reconstructed edges
<!-- test: larql_core::test_checkpoint::test_checkpoint_preserves_metadata -->

### Requirement: Python-compatibility I/O layer

`larql_core` SHALL load graphs produced by the Python reference
implementation without conversion, preserving confidence, source,
schema, type-inference rules, and stats. JSON and msgpack files
written by the Python implementation MUST round-trip through the
Rust loader without information loss.

#### Scenario: Load a Python-produced graph
- **WHEN** a graph file produced by the Python implementation is loaded
- **THEN** the resulting `Graph` SHALL contain the expected edges
<!-- test: larql_core::test_python_compat::test_load_python_produced_graph -->

#### Scenario: Confidence, source, schema, and rules are preserved
- **WHEN** the Python-produced graph is inspected for confidence, source, schema, and type-inference rules
- **THEN** all four fields SHALL match the values asserted by the Python reference
<!-- test: larql_core::test_python_compat::test_python_graph_confidence -->
<!-- test: larql_core::test_python_compat::test_python_graph_source -->
<!-- test: larql_core::test_python_compat::test_python_graph_schema -->
<!-- test: larql_core::test_python_compat::test_python_graph_type_rules -->

#### Scenario: Aggregate stats and JSON / msgpack round-trip
- **WHEN** stats are computed on the Python-produced graph and **WHEN** the graph is round-tripped through JSON and msgpack
- **THEN** the stats SHALL match the Python reference and the round-tripped graph SHALL equal the original
<!-- test: larql_core::test_python_compat::test_python_graph_stats -->
<!-- test: larql_core::test_python_compat::test_python_graph_json_roundtrip -->
<!-- test: larql_core::test_python_compat::test_python_graph_msgpack_roundtrip -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_core::test_roundtrip::**::* -->
<!-- test: larql_core::test_checkpoint::**::* -->
<!-- test: larql_core::test_python_compat::**::* -->
<!-- test: larql_core::io::packed::tests::**::* -->
