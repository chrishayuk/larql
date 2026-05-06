## ADDED Requirements

### Requirement: Token classes and lexical structure

The `larql_lql::lexer::Lexer` SHALL convert an LQL input string into a
`Vec<Token>` covering every token class the parser expects: keywords
(case-insensitive), string literals (double- and single-quoted with
backslash escapes), integer literals, floating-point literals,
identifiers (column names and unquoted entity names), the pipe
operator (`|>`), comparison operators (`=`, `!=`, `>`, `<`, `>=`,
`<=`), structural punctuation (`*`, `,`, `;`, `(`, `)`, `.`), the
`Dash` token (`-`) used for ranges and negative literals, and a
trailing `Eof` token. Whitespace and `--` line comments MUST be
skipped without producing tokens. An unterminated string literal,
unknown character, or stand-alone `|`/`!` MUST return a
`LexError` rather than panic.

#### Scenario: Keywords tokenise case-insensitively
- **WHEN** the input `walk WALK Walk wAlK` is lexed
- **THEN** every token SHALL be `Token::Keyword(Keyword::Walk)`
<!-- test: larql_lql::lexer::tests::case_insensitive_keywords -->

#### Scenario: Lifecycle, query, mutation, component, mode, conflict, and format keywords are recognised
- **WHEN** representative keyword groups are tokenised
- **THEN** each word MUST become its `Keyword` variant rather than an `Ident`
<!-- test: larql_lql::lexer::tests::all_lifecycle_keywords -->
<!-- test: larql_lql::lexer::tests::all_query_keywords -->
<!-- test: larql_lql::lexer::tests::all_mutation_keywords -->
<!-- test: larql_lql::lexer::tests::component_keywords -->
<!-- test: larql_lql::lexer::tests::mode_keywords -->
<!-- test: larql_lql::lexer::tests::conflict_strategy_keywords -->
<!-- test: larql_lql::lexer::tests::format_keywords -->

#### Scenario: Numeric literals separate ints, floats, and ranges
- **WHEN** integer, float, negative, and range inputs are lexed
- **THEN** integers SHALL emit `IntegerLit`, floats SHALL emit `NumberLit`, `-5` SHALL be `Dash` followed by `IntegerLit(5)`, and `0-33` SHALL be `IntegerLit(0)`, `Dash`, `IntegerLit(33)`
<!-- test: larql_lql::lexer::tests::integer_literal -->
<!-- test: larql_lql::lexer::tests::float_literal -->
<!-- test: larql_lql::lexer::tests::negative_number_is_dash_plus_int -->
<!-- test: larql_lql::lexer::tests::range_with_dash -->

#### Scenario: Strings parse with both quote styles and reject unterminated input
- **WHEN** double-quoted, single-quoted, escaped, empty, and unterminated string inputs are lexed
- **THEN** valid strings SHALL produce `StringLit`, and unterminated input MUST return `LexError`
<!-- test: larql_lql::lexer::tests::double_quoted_string -->
<!-- test: larql_lql::lexer::tests::single_quoted_string -->
<!-- test: larql_lql::lexer::tests::string_with_escape -->
<!-- test: larql_lql::lexer::tests::empty_string -->
<!-- test: larql_lql::lexer::tests::unterminated_string_error -->

#### Scenario: Operators and punctuation tokenise distinctly
- **WHEN** the comparison operators `= != > < >= <=`, the pipe `|>`, and punctuation `* , ; ( ) .` are lexed
- **THEN** each MUST emit its dedicated `Token` variant
<!-- test: larql_lql::lexer::tests::comparison_operators -->
<!-- test: larql_lql::lexer::tests::pipe_operator -->
<!-- test: larql_lql::lexer::tests::all_punctuation -->

#### Scenario: Comments and whitespace are skipped
- **WHEN** input contains leading, trailing, multi-line, or inline `--` comments and whitespace
- **THEN** the resulting tokens SHALL contain only the underlying statement tokens plus `Eof`
<!-- test: larql_lql::lexer::tests::comment_skipping -->
<!-- test: larql_lql::lexer::tests::multiple_comments -->
<!-- test: larql_lql::lexer::tests::inline_comment_after_statement -->
<!-- test: larql_lql::lexer::tests::empty_input -->
<!-- test: larql_lql::lexer::tests::whitespace_only -->

#### Scenario: Unknown characters and incomplete operators error
- **WHEN** an `@`, lone `|`, or lone `!` is lexed
- **THEN** the lexer MUST return `LexError` rather than emit a token
<!-- test: larql_lql::lexer::tests::unexpected_character_error -->
<!-- test: larql_lql::lexer::tests::incomplete_pipe_error -->
<!-- test: larql_lql::lexer::tests::incomplete_bang_error -->

#### Scenario: Identifiers fall through when no keyword matches
- **WHEN** an alphabetic word with underscores does not match any keyword
- **THEN** it MUST tokenise as `Token::Ident`
<!-- test: larql_lql::lexer::tests::unknown_word_is_ident -->

#### Scenario: Full statements tokenise to expected token counts
- **WHEN** complete `EXTRACT`, `INSERT`, and multi-line `SELECT` statements are lexed
- **THEN** the token sequence MUST match the spec's expected count and ordering
<!-- test: larql_lql::lexer::tests::extract_statement_tokens -->
<!-- test: larql_lql::lexer::tests::insert_statement_tokens -->
<!-- test: larql_lql::lexer::tests::multiline_statement_tokens -->
<!-- test: larql_lql::lexer::tests::walk_simple -->
<!-- test: larql_lql::lexer::tests::use_vindex -->
<!-- test: larql_lql::lexer::tests::select_with_conditions -->

### Requirement: AST statement coverage

The `Statement` enum in `larql_lql::ast` SHALL provide a variant for
every supported LQL statement, grouped into lifecycle (`Use`,
`Stats`, `Extract`, `Compile`, `Diff`), query (`Walk`, `Infer`,
`Select`, `Describe`, `Explain`, `Trace`), mutation (`Insert`,
`Delete`, `Update`, `Merge`, `Rebalance`), introspection
(`ShowRelations`, `ShowLayers`, `ShowFeatures`, `ShowEntities`,
`ShowModels`, `ShowCompactStatus`, `CompactMinor`, `CompactMajor`),
and patch lifecycle (`BeginPatch`, `SavePatch`, `ApplyPatch`,
`ShowPatches`, `RemovePatch`). Two statements SHALL be composable
via the `Statement::Pipe { left, right }` variant. Every variant
MUST round-trip through the parser test suite without losing
information.

#### Scenario: Lifecycle statements parse to their AST variants
- **WHEN** `EXTRACT`, `COMPILE`, `DIFF`, and `USE` statements are parsed via the demo Act 1/Act 5 scripts
- **THEN** the parser SHALL produce `Statement::Extract`, `Statement::Compile`, `Statement::Diff`, and `Statement::Use` with all optional fields populated correctly
<!-- test: larql_lql::parser::tests::parse_demo_script_act1 -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act5 -->
<!-- test: larql_lql::parser::tests::parse_extract_minimal -->
<!-- test: larql_lql::parser::tests::parse_compile_current_safetensors -->
<!-- test: larql_lql::parser::tests::parse_use_vindex -->

#### Scenario: Query statements (WALK / SELECT / DESCRIBE / EXPLAIN / INFER / TRACE) parse
- **WHEN** the demo scripts and per-statement parser tests cover WALK, SELECT, DESCRIBE, EXPLAIN, INFER, and TRACE inputs
- **THEN** each MUST yield its corresponding `Statement` variant carrying every parsed clause
<!-- test: larql_lql::parser::tests::parse_demo_script_act3 -->
<!-- test: larql_lql::parser::tests::parse_walk_minimal -->
<!-- test: larql_lql::parser::tests::parse_select_star -->
<!-- test: larql_lql::parser::tests::parse_describe_minimal -->
<!-- test: larql_lql::parser::tests::parse_explain_walk_minimal -->
<!-- test: larql_lql::parser::tests::parse_infer_minimal -->
<!-- test: larql_lql::parser::tests::parse_trace_minimal -->

#### Scenario: Mutation statements parse to their AST variants
- **WHEN** INSERT/DELETE/UPDATE/MERGE/REBALANCE inputs are parsed
- **THEN** the parser SHALL produce the matching `Statement::Insert/Delete/Update/Merge/Rebalance` variant
<!-- test: larql_lql::parser::tests::parse_insert_minimal -->
<!-- test: larql_lql::parser::tests::parse_delete_single_condition -->
<!-- test: larql_lql::parser::tests::parse_update_single_set -->
<!-- test: larql_lql::parser::tests::parse_merge_minimal -->
<!-- test: larql_lql::parser::tests::parse_rebalance_minimal -->

#### Scenario: Introspection statements parse to SHOW / STATS / COMPACT variants
- **WHEN** SHOW RELATIONS / LAYERS / FEATURES / ENTITIES / MODELS / PATCHES, STATS, and COMPACT inputs are parsed
- **THEN** each MUST yield the appropriate AST variant and option payload
<!-- test: larql_lql::parser::tests::parse_show_relations_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_layers_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_features_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_entities_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_models -->
<!-- test: larql_lql::parser::tests::parse_show_patches -->
<!-- test: larql_lql::parser::tests::parse_stats_no_path -->
<!-- test: larql_lql::parser::tests::parse_show_compact_status -->
<!-- test: larql_lql::parser::tests::parse_compact_minor -->

### Requirement: Clause and expression types

The AST SHALL expose strongly-typed clause and expression enums:
`VindexRef` (`Path` / `Current`), `UseTarget` (`Vindex` / `Model` /
`Remote`), `ExtractLevel` (`Browse` / `Inference` / `All`),
`CompileTarget` (`Model` / `Vindex`), `OutputFormat`
(`Safetensors` / `Gguf`), `WalkMode` (`Hybrid` / `Pure` / `Dense`),
`InsertMode` (`Knn` / `Compose`), `ConflictStrategy`
(`KeepSource` / `KeepTarget` / `HighestConfidence`),
`CompileConflict` (`LastWins` / `HighestConfidence` / `Fail`),
`Component` (FFN / embeddings / attention components),
`SelectSource` (`Edges` / `Features` / `Entities`), `Field`
(`Star` / `Named`), `Condition` with `CompareOp` (`Eq`, `Neq`,
`Gt`, `Lt`, `Gte`, `Lte`, `Like`, `In`), `Value` (`String`,
`Number`, `Integer`, `List`), `NearestClause`, `OrderBy`,
`LayerBand` (`Syntax` / `Knowledge` / `Output` / `All`),
`DescribeMode` (`Verbose` / `Brief` / `Raw`), `ExplainMode`
(`Walk` / `Infer`), `TracePositionMode` (`Last` / `All`), `Range`,
and `Assignment`. Every variant MUST be reachable from the parser
when the corresponding LQL syntax is used.

#### Scenario: Component, mode, format, and conflict variants are populated by the parser
- **WHEN** `EXTRACT … COMPONENTS …`, `WALK … MODE …`, `COMPILE … FORMAT …`, and `MERGE … ON CONFLICT …` are parsed
- **THEN** the AST MUST hold the matching `Component`, `WalkMode`, `OutputFormat`, and `ConflictStrategy` variant
<!-- test: larql_lql::parser::tests::parse_extract_with_components_and_layers -->
<!-- test: larql_lql::parser::tests::parse_extract_attn_components -->
<!-- test: larql_lql::parser::tests::parse_walk_mode_pure -->
<!-- test: larql_lql::parser::tests::parse_walk_mode_dense -->
<!-- test: larql_lql::parser::tests::parse_compile_current_safetensors -->
<!-- test: larql_lql::parser::tests::parse_compile_path_gguf -->
<!-- test: larql_lql::parser::tests::parse_merge_keep_source -->
<!-- test: larql_lql::parser::tests::parse_merge_keep_target -->
<!-- test: larql_lql::parser::tests::parse_merge_into_with_conflict -->

#### Scenario: SELECT compare operators and value types match the AST
- **WHEN** `SELECT … WHERE` is parsed with `=`, `!=`, `>`, `<`, `>=`, `<=`, `LIKE`, and `IN (…)` and with string / integer / list values
- **THEN** the produced `Condition.op` and `Condition.value` MUST be the corresponding `CompareOp` and `Value` variants
<!-- test: larql_lql::parser::tests::parse_select_neq -->
<!-- test: larql_lql::parser::tests::parse_select_gte_lte -->
<!-- test: larql_lql::parser::tests::parse_select_like -->
<!-- test: larql_lql::parser::tests::parse_select_in -->
<!-- test: larql_lql::parser::tests::parse_select_multiple_conditions -->

#### Scenario: Layer bands and describe modes round-trip
- **WHEN** `DESCRIBE "x" SYNTAX|KNOWLEDGE|OUTPUT|ALL LAYERS` and `VERBOSE|BRIEF|RAW` are parsed
- **THEN** the resulting `band` and `mode` fields MUST match the requested `LayerBand` and `DescribeMode` values, with `Brief` as the default
<!-- test: larql_lql::parser::tests::parse_describe_syntax -->
<!-- test: larql_lql::parser::tests::parse_describe_knowledge -->
<!-- test: larql_lql::parser::tests::parse_describe_output -->
<!-- test: larql_lql::parser::tests::parse_describe_all_layers -->
<!-- test: larql_lql::parser::tests::parse_describe_verbose -->
<!-- test: larql_lql::parser::tests::parse_describe_brief -->
<!-- test: larql_lql::parser::tests::parse_describe_raw -->

#### Scenario: VindexRef, UseTarget, and ExtractLevel cover Current/Path/Model/Remote/Browse/Inference/All
- **WHEN** `COMPILE CURRENT`, `DIFF "a" CURRENT`, `USE MODEL "id" AUTO_EXTRACT`, and `EXTRACT … WITH ALL|INFERENCE|WEIGHTS` are parsed
- **THEN** each variant SHALL map to its expected enum case in the AST
<!-- test: larql_lql::parser::tests::parse_compile_current_safetensors -->
<!-- test: larql_lql::parser::tests::parse_diff_with_current -->
<!-- test: larql_lql::parser::tests::parse_use_model -->
<!-- test: larql_lql::parser::tests::parse_use_model_auto_extract -->
<!-- test: larql_lql::parser::tests::parse_extract_with_inference -->
<!-- test: larql_lql::parser::tests::parse_extract_with_all -->
<!-- test: larql_lql::parser::tests::parse_extract_with_weights_legacy -->

### Requirement: Range, OrderBy, and NearestClause grammar

`Range { start, end }` SHALL be produced by the lexer/parser pair
for any `start-end` clause (e.g. `LAYERS 0-33`, `RANGE 0-10`) and
MUST reject ranges where `start > end`. `OrderBy { field,
descending }` SHALL default to ascending when neither `ASC` nor
`DESC` is given. `NearestClause { entity, layer }` SHALL be parsed
from `NEAREST TO "entity" AT LAYER <n>` inside a `SELECT`.

#### Scenario: Ranges accept equal start/end and reject inverted ranges
- **WHEN** `SHOW LAYERS 5-5;` and `SHOW LAYERS 10-5;` are parsed
- **THEN** the equal-bounds form SHALL succeed with `start == end == 5` and the inverted form MUST return a parse error
<!-- test: larql_lql::parser::tests::range_valid_same_start_end -->
<!-- test: larql_lql::parser::tests::range_invalid_start_greater_than_end -->

#### Scenario: ORDER BY defaults to ascending and honours ASC / DESC
- **WHEN** `SELECT * … ORDER BY layer`, `… ASC`, and `… DESC` are parsed
- **THEN** the `OrderBy.descending` flag MUST reflect the explicit suffix and default to `false`
<!-- test: larql_lql::parser::tests::parse_select_order_default_asc -->
<!-- test: larql_lql::parser::tests::parse_select_order_asc -->
<!-- test: larql_lql::parser::tests::parse_select_named_fields -->

#### Scenario: NEAREST TO clauses parse with explicit layer
- **WHEN** `SELECT … NEAREST TO "Mozart" AT LAYER 26 LIMIT 20` is parsed
- **THEN** the resulting `NearestClause` MUST contain the entity and layer; LIMIT MUST coexist with NEAREST
<!-- test: larql_lql::parser::tests::parse_select_nearest -->

### Requirement: Keyword-as-field-name mapping

`Keyword::as_field_name` SHALL provide a stable lowercase string for
every keyword that may appear as a column name (`layer`,
`confidence`, `relation`, `ffn_gate`, `ffn_down`, `attn_ov`,
`auto_extract`, etc.) so the parser can use these tokens
interchangeably with bare identifiers in SELECT/UPDATE/DELETE/WHERE
clauses.

#### Scenario: Field-name keywords map to lowercase canonical names
- **WHEN** `Keyword::as_field_name` is called for `Layer`, `Confidence`, `Relation`, `FfnGate`, `FfnDown`, `AttnOv`, and `AutoExtract`
- **THEN** the returned strings MUST be `layer`, `confidence`, `relation`, `ffn_gate`, `ffn_down`, `attn_ov`, and `auto_extract`
<!-- test: larql_lql::parser::tests::keyword_field_names_consistent -->

### Requirement: Relation classifier schema

`larql_lql::relations::RelationClassifier` SHALL load discovered
clusters (`relation_clusters.json`), per-feature cluster
assignments (`feature_clusters.jsonl`), and probe-confirmed labels
(`feature_labels.json`) from a vindex directory and SHALL expose
`label_for_feature`, `cluster_for_feature`, `cluster_info`,
`num_clusters`, `has_clusters`, `is_probe_label`,
`num_probe_labels`, `classify_direction`, `cluster_for_relation`,
`cluster_centre_for_relation`, and `typical_layer_for_relation`.
Probe-confirmed labels MUST take priority over cluster-assigned
labels. The constructor MUST return `None` when the vindex contains
none of the three sources.

#### Scenario: Known features resolve to their cluster labels
- **WHEN** `label_for_feature(layer, feature)` is called on a layer/feature pair that exists in the cluster assignments
- **THEN** the classifier MUST return the cluster's label string
<!-- test: larql_lql::relations::tests::label_for_known_feature -->

#### Scenario: Unknown features and missing vindex paths return None
- **WHEN** `label_for_feature` is called for an unmapped pair, or `from_vindex` is called for a non-existent path
- **THEN** the result MUST be `None`
<!-- test: larql_lql::relations::tests::label_for_unknown_feature -->
<!-- test: larql_lql::relations::tests::from_nonexistent_vindex -->

#### Scenario: Cluster lookups expose cluster_for_feature, cluster_info, and num_clusters
- **WHEN** the classifier exposes per-feature cluster ids, cluster info tuples, and cluster counts
- **THEN** each accessor MUST return values consistent with the loaded `ClusterResult`
<!-- test: larql_lql::relations::tests::cluster_for_feature -->
<!-- test: larql_lql::relations::tests::cluster_info -->
<!-- test: larql_lql::relations::tests::num_clusters -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_lql::lexer::tests::**::* -->
<!-- test: larql_lql::lexer::**::* -->
<!-- test: larql_lql::relations::tests::**::* -->
