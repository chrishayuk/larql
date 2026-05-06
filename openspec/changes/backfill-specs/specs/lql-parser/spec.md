## ADDED Requirements

### Requirement: Recursive-descent dispatch and pipe composition

`larql_lql::parser::Parser::parse` SHALL implement a recursive-descent
top-level dispatcher that selects between `parse_extract`,
`parse_compile`, `parse_diff`, `parse_use`, `parse_walk`,
`parse_infer`, `parse_select`, `parse_describe`, `parse_explain`,
`parse_insert`, `parse_delete`, `parse_update`, `parse_merge`,
`parse_rebalance`, `parse_show`, `parse_stats`, `parse_begin`,
`parse_save`, `parse_apply`, `parse_remove`, `parse_trace`, and
`parse_compact` based on the leading keyword. After parsing one
statement the parser SHALL consume an optional pipe operator (`|>`)
followed by another statement and emit `Statement::Pipe { left,
right }`. The parser MUST reject any trailing tokens after a
single-statement parse, including a second statement after a
semicolon or a stray identifier with no terminator.

#### Scenario: Pipe operator composes WALK and EXPLAIN
- **WHEN** `WALK "x" TOP 5 |> EXPLAIN WALK "x";` is parsed
- **THEN** the result MUST be `Statement::Pipe` whose `left` is `Statement::Walk` and whose `right` is `Statement::Explain`
<!-- test: larql_lql::parser::tests::parse_pipe_walk_to_explain -->

#### Scenario: Single-statement parser rejects tails
- **WHEN** the input contains a trailing semicolon-separated statement (`STATS; SELECT * FROM EDGES;`) or a stray identifier (`STATS unexpected`)
- **THEN** `parse` MUST return a parse error
<!-- test: larql_lql::parser::tests::parser_rejects_trailing_tokens_after_semicolon -->
<!-- test: larql_lql::parser::tests::parser_rejects_trailing_identifier_without_semicolon -->

#### Scenario: Empty and comment-only inputs error
- **WHEN** the input is empty or contains only `--` comments
- **THEN** `parse` MUST return a parse error
<!-- test: larql_lql::parser::tests::parse_error_empty_input -->
<!-- test: larql_lql::parser::tests::parse_error_comment_only -->

#### Scenario: Unknown statement keywords error
- **WHEN** `FOOBAR;` or `SHOW FOOBAR;` is parsed
- **THEN** `parse` MUST return a parse error rather than dispatching
<!-- test: larql_lql::parser::tests::parse_error_unknown_statement -->
<!-- test: larql_lql::parser::tests::parse_error_show_invalid_noun -->

### Requirement: Lifecycle statement parsing

The lifecycle parsers SHALL accept all documented LQL lifecycle syntax. `parse_extract`, `parse_compile`, `parse_diff`, `parse_use`, and `parse_compact` MUST honour every optional clause (`COMPONENTS`, `LAYERS`, `WITH INFERENCE|ALL|WEIGHTS`, `INTO MODEL|VINDEX`, `FORMAT`, `ON CONFLICT`, `AT LAYER`, `RELATION[S]`, `LIMIT`, `INTO PATCH`, `CURRENT`, `MODEL`, `AUTO_EXTRACT`, `MAJOR FULL`, `WITH LAMBDA = …`) and MUST reject combinations the grammar disallows, such as `ON CONFLICT` on `COMPILE INTO MODEL`.

#### Scenario: EXTRACT accepts components, layers, and extract level
- **WHEN** `EXTRACT MODEL "id" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33;` is parsed
- **THEN** the AST MUST list all four components, the layer range `0..33`, and an extract level of `Browse`
<!-- test: larql_lql::parser::tests::parse_extract_with_components_and_layers -->
<!-- test: larql_lql::parser::tests::parse_extract_attn_components -->

#### Scenario: EXTRACT WITH ALL / INFERENCE / WEIGHTS / minimal map to ExtractLevel
- **WHEN** `WITH ALL`, `WITH INFERENCE`, `WITH WEIGHTS`, and minimal forms are parsed
- **THEN** the `extract_level` MUST be `All`, `Inference`, `Inference` (legacy), and `Browse` respectively
<!-- test: larql_lql::parser::tests::parse_extract_minimal -->
<!-- test: larql_lql::parser::tests::parse_extract_with_inference -->
<!-- test: larql_lql::parser::tests::parse_extract_with_all -->
<!-- test: larql_lql::parser::tests::parse_extract_with_weights_legacy -->
<!-- test: larql_lql::parser::tests::parse_extract_with_all_and_components -->

#### Scenario: COMPILE supports CURRENT/path, INTO MODEL/VINDEX, formats, and conflict policies
- **WHEN** `COMPILE CURRENT|"path" INTO MODEL "out/" FORMAT safetensors|gguf` and `COMPILE … INTO VINDEX … ON CONFLICT LAST_WINS|HIGHEST_CONFIDENCE|FAIL` are parsed
- **THEN** the AST `target`, `format`, and `on_conflict` MUST match the requested clause
<!-- test: larql_lql::parser::tests::parse_compile_current_safetensors -->
<!-- test: larql_lql::parser::tests::parse_compile_path_gguf -->
<!-- test: larql_lql::parser::tests::parse_compile_no_format -->
<!-- test: larql_lql::parser::tests::parse_compile_into_vindex -->
<!-- test: larql_lql::parser::tests::parse_compile_into_vindex_on_conflict_last_wins -->
<!-- test: larql_lql::parser::tests::parse_compile_into_vindex_on_conflict_highest_confidence -->
<!-- test: larql_lql::parser::tests::parse_compile_into_vindex_on_conflict_fail -->
<!-- test: larql_lql::parser::tests::parse_compile_into_model_explicit -->

#### Scenario: COMPILE INTO MODEL rejects ON CONFLICT
- **WHEN** `COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors ON CONFLICT FAIL;` is parsed
- **THEN** `parse` MUST return an error because `ON CONFLICT` is INTO VINDEX-only
<!-- test: larql_lql::parser::tests::parse_compile_into_model_with_on_conflict_errors -->

#### Scenario: DIFF supports two paths, CURRENT, layer/relation/limit/INTO PATCH
- **WHEN** various `DIFF` clauses are parsed
- **THEN** the AST MUST hold the requested `a`, `b`, `layer`, `relation`, `limit`, and `into_patch` fields
<!-- test: larql_lql::parser::tests::parse_diff_two_paths -->
<!-- test: larql_lql::parser::tests::parse_diff_with_current -->
<!-- test: larql_lql::parser::tests::parse_diff_with_limit -->
<!-- test: larql_lql::parser::tests::parse_diff_with_layer -->
<!-- test: larql_lql::parser::tests::parse_diff_with_relation_singular -->
<!-- test: larql_lql::parser::tests::parse_diff_with_relations_plural -->
<!-- test: larql_lql::parser::tests::parse_diff_with_relation_and_limit -->
<!-- test: larql_lql::parser::tests::parse_diff_into_patch -->
<!-- test: larql_lql::parser::tests::parse_diff_without_into_patch -->

#### Scenario: USE recognises vindex, model, and AUTO_EXTRACT variants
- **WHEN** `USE "x.vindex"`, `USE MODEL "id"`, and `USE MODEL "id" AUTO_EXTRACT` are parsed
- **THEN** the resulting `UseTarget` MUST be `Vindex`, `Model { auto_extract: false }`, and `Model { auto_extract: true }`
<!-- test: larql_lql::parser::tests::parse_use_vindex -->
<!-- test: larql_lql::parser::tests::parse_use_model -->
<!-- test: larql_lql::parser::tests::parse_use_model_auto_extract -->

#### Scenario: COMPACT MAJOR/MINOR forms and lambda parse
- **WHEN** `COMPACT MINOR`, `COMPACT MAJOR`, `COMPACT MAJOR FULL`, and `COMPACT MAJOR WITH LAMBDA = 0.001` are parsed
- **THEN** the AST `Statement::CompactMinor` / `CompactMajor { full, lambda }` MUST reflect the syntax
<!-- test: larql_lql::parser::tests::parse_compact_minor -->
<!-- test: larql_lql::parser::tests::parse_compact_major -->
<!-- test: larql_lql::parser::tests::parse_compact_major_full -->
<!-- test: larql_lql::parser::tests::parse_compact_major_with_lambda -->

### Requirement: Query statement parsing

The query parsers SHALL accept the full grammar for query statements. `parse_walk`, `parse_infer`, `parse_select`, `parse_describe`, and `parse_explain` MUST honour every optional clause (`TOP`, `LAYERS`, `MODE`, `COMPARE`, `WHERE`, `ORDER BY`, `LIMIT`, `NEAREST TO … AT LAYER`, `AT LAYER`, `SYNTAX`/`KNOWLEDGE`/`OUTPUT`/`ALL LAYERS`, `RELATIONS ONLY`, `VERBOSE`/`BRIEF`/`RAW`, `WITH ATTENTION`, `WALK`/`INFER` mode for EXPLAIN) and MUST reject statements missing required parts (e.g. `WALK` without a prompt, `SELECT` without `FROM`).

#### Scenario: WALK accepts TOP, LAYERS, MODE, COMPARE
- **WHEN** `WALK "prompt" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;` and other WALK variants are parsed
- **THEN** the resulting `Statement::Walk` MUST carry `top`, `layers`, `mode`, and `compare` matching the input
<!-- test: larql_lql::parser::tests::parse_walk_minimal -->
<!-- test: larql_lql::parser::tests::parse_walk_with_top -->
<!-- test: larql_lql::parser::tests::parse_walk_full_options -->
<!-- test: larql_lql::parser::tests::parse_walk_mode_pure -->
<!-- test: larql_lql::parser::tests::parse_walk_mode_dense -->
<!-- test: larql_lql::parser::tests::parse_walk_layers_all -->

#### Scenario: WALK without a prompt errors
- **WHEN** `WALK TOP 5;` is parsed
- **THEN** `parse` MUST return an error
<!-- test: larql_lql::parser::tests::parse_error_walk_missing_prompt -->

#### Scenario: SELECT supports star, named fields, conditions, ORDER BY, LIMIT, and NEAREST
- **WHEN** SELECT inputs span `*`, named fields, multiple WHERE conditions, ORDER BY (default/ASC/DESC), LIMIT, and NEAREST clauses
- **THEN** the AST `Select { fields, conditions, order, limit, nearest }` MUST match each input
<!-- test: larql_lql::parser::tests::parse_select_star -->
<!-- test: larql_lql::parser::tests::parse_select_named_fields -->
<!-- test: larql_lql::parser::tests::parse_select_multiple_conditions -->
<!-- test: larql_lql::parser::tests::parse_select_by_layer_and_feature -->
<!-- test: larql_lql::parser::tests::parse_select_nearest -->
<!-- test: larql_lql::parser::tests::parse_select_no_where -->
<!-- test: larql_lql::parser::tests::parse_select_order_asc -->
<!-- test: larql_lql::parser::tests::parse_select_order_default_asc -->

#### Scenario: SELECT without FROM errors
- **WHEN** `SELECT * WHERE entity = "x";` is parsed
- **THEN** `parse` MUST return an error because `FROM` is required
<!-- test: larql_lql::parser::tests::parse_error_select_missing_from -->

#### Scenario: DESCRIBE supports entity, layer, layer band, RELATIONS ONLY, and modes
- **WHEN** DESCRIBE inputs cover minimal/AT LAYER/RELATIONS ONLY/SYNTAX/KNOWLEDGE/OUTPUT/ALL LAYERS/VERBOSE/BRIEF/RAW
- **THEN** each variant MUST surface the corresponding `band`, `layer`, `relations_only`, and `mode` AST fields
<!-- test: larql_lql::parser::tests::parse_describe_minimal -->
<!-- test: larql_lql::parser::tests::parse_describe_at_layer -->
<!-- test: larql_lql::parser::tests::parse_describe_relations_only -->
<!-- test: larql_lql::parser::tests::parse_describe_layer_and_relations_only -->
<!-- test: larql_lql::parser::tests::parse_describe_syntax -->
<!-- test: larql_lql::parser::tests::parse_describe_knowledge -->
<!-- test: larql_lql::parser::tests::parse_describe_output -->
<!-- test: larql_lql::parser::tests::parse_describe_all_layers -->
<!-- test: larql_lql::parser::tests::parse_describe_band_with_relations_only -->
<!-- test: larql_lql::parser::tests::parse_describe_verbose -->
<!-- test: larql_lql::parser::tests::parse_describe_brief -->
<!-- test: larql_lql::parser::tests::parse_describe_raw -->
<!-- test: larql_lql::parser::tests::parse_describe_band_verbose -->

#### Scenario: EXPLAIN dispatches WALK vs INFER and full option set
- **WHEN** `EXPLAIN WALK "p" LAYERS 24-33 VERBOSE;`, `EXPLAIN INFER "p" KNOWLEDGE TOP 1 RELATIONS ONLY WITH ATTENTION;`, and other EXPLAIN inputs are parsed
- **THEN** the AST `mode`, `layers`, `band`, `top`, `relations_only`, `with_attention`, and `verbose` fields MUST match the requested clauses
<!-- test: larql_lql::parser::tests::parse_explain_walk_minimal -->
<!-- test: larql_lql::parser::tests::parse_explain_walk_with_layers_and_verbose -->
<!-- test: larql_lql::parser::tests::parse_explain_walk_with_top -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_minimal -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_with_options -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_with_band -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_relations_only -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_with_attention -->
<!-- test: larql_lql::parser::tests::parse_explain_infer_all_options -->

#### Scenario: INFER honours TOP and COMPARE
- **WHEN** `INFER "prompt" TOP 5;`, `INFER "p" TOP 3 COMPARE;`, and `INFER "p";` are parsed
- **THEN** the AST `top` and `compare` fields MUST reflect each input
<!-- test: larql_lql::parser::tests::parse_infer_minimal -->
<!-- test: larql_lql::parser::tests::parse_infer_with_compare -->
<!-- test: larql_lql::parser::tests::parse_infer_no_top -->

### Requirement: Mutation statement parsing

The mutation parsers SHALL accept the full mutation grammar. `parse_insert`, `parse_delete`, `parse_update`, `parse_merge`, and `parse_rebalance` MUST handle every documented form (`INSERT INTO EDGES (…) VALUES (…) [AT LAYER … CONFIDENCE … ALPHA …]`, `DELETE FROM EDGES WHERE …`, `UPDATE EDGES SET … WHERE …`, `MERGE "src" [INTO "tgt"] [ON CONFLICT …]`, `REBALANCE [UNTIL CONVERGED] [MAX n] [FLOOR p] [CEILING p]`) and MUST reject malformed INSERTs (e.g. missing VALUES) so the executor never sees an incomplete tuple.

#### Scenario: INSERT honours layer, confidence, alpha, and combinations
- **WHEN** `INSERT … VALUES (…) [AT LAYER 26 CONFIDENCE 0.8 ALPHA 0.3]` variants are parsed
- **THEN** the AST MUST capture each clause; missing clauses MUST be `None`
<!-- test: larql_lql::parser::tests::parse_insert_minimal -->
<!-- test: larql_lql::parser::tests::parse_insert_with_layer_and_confidence -->
<!-- test: larql_lql::parser::tests::parse_insert_with_alpha -->
<!-- test: larql_lql::parser::tests::parse_insert_with_layer_confidence_alpha -->

#### Scenario: INSERT without VALUES errors
- **WHEN** `INSERT INTO EDGES (entity, relation, target);` is parsed
- **THEN** `parse` MUST return an error
<!-- test: larql_lql::parser::tests::parse_error_insert_missing_values -->

#### Scenario: DELETE supports single, multi-condition, and layer filters
- **WHEN** various `DELETE FROM EDGES WHERE …` inputs are parsed
- **THEN** the resulting `Delete.conditions` MUST match the requested predicates
<!-- test: larql_lql::parser::tests::parse_delete_single_condition -->
<!-- test: larql_lql::parser::tests::parse_delete_multiple_conditions -->
<!-- test: larql_lql::parser::tests::parse_delete_by_layer -->

#### Scenario: UPDATE supports single and multi-assignment SET
- **WHEN** `UPDATE EDGES SET target = "London"` and `SET target = "London", confidence = 0.9` inputs are parsed
- **THEN** `Update.set` MUST contain the corresponding number of assignments
<!-- test: larql_lql::parser::tests::parse_update_single_set -->
<!-- test: larql_lql::parser::tests::parse_update_multiple_assignments -->

#### Scenario: MERGE handles minimal, INTO, and ON CONFLICT variants
- **WHEN** `MERGE "src.vindex"`, `… INTO "tgt"`, and `… ON CONFLICT KEEP_SOURCE|KEEP_TARGET|HIGHEST_CONFIDENCE` inputs are parsed
- **THEN** the AST `target` and `conflict` fields MUST reflect each variant
<!-- test: larql_lql::parser::tests::parse_merge_minimal -->
<!-- test: larql_lql::parser::tests::parse_merge_into_no_conflict -->
<!-- test: larql_lql::parser::tests::parse_merge_into_with_conflict -->
<!-- test: larql_lql::parser::tests::parse_merge_keep_source -->
<!-- test: larql_lql::parser::tests::parse_merge_keep_target -->

#### Scenario: REBALANCE honours UNTIL CONVERGED, MAX, FLOOR, CEILING
- **WHEN** `REBALANCE`, `REBALANCE UNTIL CONVERGED`, `REBALANCE MAX 32`, `REBALANCE FLOOR 0.3 CEILING 0.9`, and the all-clauses form are parsed
- **THEN** `max_iters`, `floor`, and `ceiling` MUST match the requested values; the bare form MUST leave them all `None`
<!-- test: larql_lql::parser::tests::parse_rebalance_minimal -->
<!-- test: larql_lql::parser::tests::parse_rebalance_until_converged -->
<!-- test: larql_lql::parser::tests::parse_rebalance_max_iters -->
<!-- test: larql_lql::parser::tests::parse_rebalance_floor_ceiling -->
<!-- test: larql_lql::parser::tests::parse_rebalance_all_clauses -->

### Requirement: Introspection and patch statement parsing

The introspection and patch parsers SHALL accept every SHOW noun form, `STATS [path]`, and the patch lifecycle. `parse_show`, `parse_stats`, `parse_begin`, `parse_save`, `parse_apply`, `parse_remove`, and the SHOW COMPACT STATUS path MUST recognise `RELATIONS [WITH EXAMPLES] [AT LAYER n] [VERBOSE|BRIEF|RAW]`, `LAYERS [RANGE n-m]` or bare range, `FEATURES n [WHERE …] [LIMIT n]`, `ENTITIES [n] [AT LAYER n] [LIMIT n]`, `MODELS`, `PATCHES`, `COMPACT STATUS`, `STATS [path]`, and `BEGIN PATCH`/`SAVE PATCH`/`APPLY PATCH`/`REMOVE PATCH`, and MUST tolerate optional trailing semicolons.

#### Scenario: SHOW RELATIONS supports examples, layer, and modes
- **WHEN** `SHOW RELATIONS`, `SHOW RELATIONS WITH EXAMPLES`, `SHOW RELATIONS AT LAYER 26`, `SHOW RELATIONS VERBOSE/RAW`, and combined forms are parsed
- **THEN** the AST `layer`, `with_examples`, and `mode` MUST match each input with `Brief` as the default mode
<!-- test: larql_lql::parser::tests::parse_show_relations_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_relations_with_examples -->
<!-- test: larql_lql::parser::tests::parse_show_relations_at_layer -->
<!-- test: larql_lql::parser::tests::parse_show_relations_verbose -->
<!-- test: larql_lql::parser::tests::parse_show_relations_raw -->
<!-- test: larql_lql::parser::tests::parse_show_relations_verbose_with_examples -->

#### Scenario: SHOW LAYERS, FEATURES, ENTITIES, MODELS parse with all forms
- **WHEN** `SHOW LAYERS`, `SHOW LAYERS RANGE 0-10`, bare-range `SHOW LAYERS 0-10`, `SHOW FEATURES n [WHERE …] [LIMIT n]`, `SHOW ENTITIES …`, and `SHOW MODELS;` are parsed
- **THEN** the corresponding AST variants MUST be produced with `range`, `layer`, `conditions`, and `limit` populated
<!-- test: larql_lql::parser::tests::parse_show_layers_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_layers_with_range -->
<!-- test: larql_lql::parser::tests::parse_show_layers_bare_range -->
<!-- test: larql_lql::parser::tests::parse_show_features_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_features_with_where_and_limit -->
<!-- test: larql_lql::parser::tests::parse_show_models -->
<!-- test: larql_lql::parser::tests::parse_show_entities_minimal -->
<!-- test: larql_lql::parser::tests::parse_show_entities_bare_layer -->
<!-- test: larql_lql::parser::tests::parse_show_entities_at_layer_with_limit -->
<!-- test: larql_lql::parser::tests::parse_show_entities_limit_only -->

#### Scenario: STATS, SHOW COMPACT STATUS tolerate optional semicolon
- **WHEN** `STATS;`, `STATS`, `STATS "path"`, `SHOW COMPACT STATUS;`, and `SHOW COMPACT STATUS` (no semicolon) are parsed
- **THEN** the AST MUST be `Statement::Stats` / `Statement::ShowCompactStatus` regardless of trailing semicolon
<!-- test: larql_lql::parser::tests::parse_stats_no_path -->
<!-- test: larql_lql::parser::tests::parse_stats_with_path -->
<!-- test: larql_lql::parser::tests::parse_stats_no_semicolon -->
<!-- test: larql_lql::parser::tests::parse_show_compact_status -->
<!-- test: larql_lql::parser::tests::parse_show_compact_status_no_semicolon -->

#### Scenario: Patch lifecycle and DIFF INTO PATCH parse
- **WHEN** `BEGIN PATCH "x"`, `SAVE PATCH;`, `APPLY PATCH "x"`, `SHOW PATCHES`, `REMOVE PATCH "x"`, and `DIFF "a" "b" INTO PATCH "p"` inputs are parsed
- **THEN** the AST MUST contain `BeginPatch`, `SavePatch`, `ApplyPatch`, `ShowPatches`, `RemovePatch`, and a `Diff { into_patch }` field as appropriate
<!-- test: larql_lql::parser::tests::parse_begin_patch -->
<!-- test: larql_lql::parser::tests::parse_save_patch -->
<!-- test: larql_lql::parser::tests::parse_apply_patch -->
<!-- test: larql_lql::parser::tests::parse_show_patches -->
<!-- test: larql_lql::parser::tests::parse_remove_patch -->
<!-- test: larql_lql::parser::tests::parse_patch_workflow -->
<!-- test: larql_lql::parser::tests::parse_diff_into_patch -->

### Requirement: Trace statement parsing

`parse_trace` SHALL parse `TRACE "<prompt>" [FOR "<answer>"]
[DECOMPOSE] [LAYERS start-end] [POSITIONS LAST|ALL] [SAVE
"<path>"]` clauses in any documented order and emit
`Statement::Trace`. Missing optional clauses MUST default to
`None`/`false`.

#### Scenario: TRACE captures FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE
- **WHEN** TRACE inputs from minimal through fully populated are parsed
- **THEN** the AST `prompt`, `answer`, `decompose`, `layers`, `positions`, and `save` fields MUST match the input clauses
<!-- test: larql_lql::parser::tests::parse_trace_minimal -->
<!-- test: larql_lql::parser::tests::parse_trace_with_for_token -->
<!-- test: larql_lql::parser::tests::parse_trace_decompose_with_layers -->
<!-- test: larql_lql::parser::tests::parse_trace_save -->
<!-- test: larql_lql::parser::tests::parse_trace_positions_all -->
<!-- test: larql_lql::parser::tests::parse_trace_positions_last -->
<!-- test: larql_lql::parser::tests::parse_trace_full -->

### Requirement: Whitespace, comments, and demo-script regression

The parser SHALL be robust to leading and trailing `--` line
comments, multi-line statements, and SHALL parse every statement in
the published demo script (Acts 1-5) without error so that the LQL
spec stays in lockstep with the parser implementation.

#### Scenario: Comments and multi-line statements parse
- **WHEN** an input begins with `-- comment\n`, has trailing comment, or spans multiple lines
- **THEN** `parse` MUST succeed and the AST MUST be unchanged from a single-line equivalent
<!-- test: larql_lql::parser::tests::parse_with_leading_comment -->
<!-- test: larql_lql::parser::tests::parse_with_trailing_comment -->
<!-- test: larql_lql::parser::tests::parse_multiline_statement -->

#### Scenario: Demo script Acts 1-5 parse end-to-end
- **WHEN** every statement from Act 1 (`EXTRACT`, `USE`, `STATS`), Act 2 (`SHOW RELATIONS`, `DESCRIBE`), Act 3 (`WALK`, `EXPLAIN WALK`, `INFER`), Act 4 (`DESCRIBE`, `INSERT`, `DESCRIBE`), and Act 5 (`DIFF CURRENT`, `COMPILE INTO MODEL FORMAT safetensors`) is parsed
- **THEN** every parse MUST succeed
<!-- test: larql_lql::parser::tests::parse_demo_script_act1 -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act2 -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act3 -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act4 -->
<!-- test: larql_lql::parser::tests::parse_demo_script_act5 -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_lql::parser::tests::**::* -->
