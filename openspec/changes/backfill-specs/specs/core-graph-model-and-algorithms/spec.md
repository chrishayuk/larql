## ADDED Requirements

### Requirement: Typed property-graph model

The `larql_core::core` module SHALL provide an in-memory property graph
keyed by `(subject, relation, object)` triples with adjacency indexes
that support O(1) outgoing and incoming neighborhood lookup. `Graph`
MUST de-duplicate identical triples on insert, MUST rebuild adjacency
indexes when an edge is removed, and SHALL NOT collapse distinct
objects that share the same `(subject, relation)` prefix.

#### Scenario: New graph is empty
- **WHEN** `Graph::new()` is constructed
- **THEN** `len()` SHALL be zero, `nodes()` SHALL be empty, and the graph SHALL accept any subsequent insertion
<!-- test: larql_core::test_graph::test_empty_graph -->
<!-- test: larql_core::test_graph::test_stats_empty -->

#### Scenario: Identical triples de-duplicate on insert
- **WHEN** the same `(subject, relation, object)` is added twice via `add_edge` or `add_edges` batch
- **THEN** the graph SHALL retain a single edge and `try_add_edge` SHALL report the duplicate to the caller
<!-- test: larql_core::test_graph::test_add_edge -->
<!-- test: larql_core::test_graph::test_add_edges_batch -->
<!-- test: larql_core::test_graph::test_duplicate_skipped -->
<!-- test: larql_core::test_graph::test_try_add_edge_reports_duplicate -->

#### Scenario: Distinct objects under same subject/relation are preserved
- **WHEN** two edges share `subject` and `relation` but differ in `object`
- **THEN** both edges SHALL be retained and exposed via the multi-edge lookup helpers
<!-- test: larql_core::test_graph::test_same_subject_relation_different_object -->
<!-- test: larql_core::test_graph::test_multiedge_lookup_helpers -->
<!-- test: larql_core::test_graph::test_get_edge_exact_triple -->

#### Scenario: Edge removal rebuilds adjacency indexes
- **WHEN** an existing edge is removed via `remove_edge`
- **THEN** outgoing and incoming indexes SHALL no longer reference the removed triple, and removing a non-existent edge SHALL be a no-op
<!-- test: larql_core::test_graph::test_remove_edge -->
<!-- test: larql_core::test_graph::test_remove_nonexistent_edge -->
<!-- test: larql_core::test_graph::test_remove_rebuilds_indexes -->
<!-- test: larql_core::test_components_walk::remove_edge_rebuilds_indexes -->
<!-- test: larql_core::test_components_walk::remove_edge_nonexistent -->

#### Scenario: Insert replaces a changed payload on an existing triple
- **WHEN** an edge with the same triple but a different confidence, source, or metadata payload is inserted
- **THEN** the prior payload SHALL be replaced and the resulting graph SHALL hold one edge with the new payload
<!-- test: larql_core::test_graph::test_insert_edge_replaces_changed_payload -->

### Requirement: Edge value semantics

`larql_core::core::edge::Edge` SHALL clamp confidence values into the
inclusive range `[0.0, 1.0]`, MUST treat two edges as equal when their
`(subject, relation, object)` triple matches regardless of confidence
or metadata, and MUST hash consistently with that equality. The
compact serialization form SHALL omit unknown source types and SHALL
round-trip through JSON without information loss for known sources.

#### Scenario: Default and builder construction
- **WHEN** an edge is constructed via `Edge::new` or via the builder methods (`with_confidence`, `with_source`, `with_metadata`)
- **THEN** the resulting edge SHALL carry the configured triple, confidence, source, and metadata, and the triple accessor SHALL return all three components
<!-- test: larql_core::test_edge::test_edge_new_defaults -->
<!-- test: larql_core::test_edge::test_edge_builder -->
<!-- test: larql_core::test_edge::test_triple -->

#### Scenario: Confidence is clamped into [0,1]
- **WHEN** a confidence outside `[0.0, 1.0]` is supplied
- **THEN** the stored confidence SHALL be clamped into `[0.0, 1.0]`
<!-- test: larql_core::test_edge::test_confidence_clamped -->

#### Scenario: Equality and hashing key on the triple only
- **WHEN** edges with identical triples but different confidence are compared, and **WHEN** edges with different triples are compared
- **THEN** equality SHALL ignore confidence, equality SHALL hold for matching triples, and hashing SHALL agree with equality
<!-- test: larql_core::test_edge::test_edge_equality_ignores_confidence -->
<!-- test: larql_core::test_edge::test_edge_equality_different_triple -->
<!-- test: larql_core::test_edge::test_edge_hash_consistency -->

#### Scenario: Compact form round-trips and serializes deterministically
- **WHEN** an edge is converted to its compact form and back, and **WHEN** the compact form is serialized to JSON
- **THEN** the round-trip SHALL preserve the triple/confidence/source/metadata payload, unknown source types SHALL be omitted, and the JSON SHALL contain the documented field set
<!-- test: larql_core::test_edge::test_compact_roundtrip -->
<!-- test: larql_core::test_edge::test_compact_unknown_source_omitted -->
<!-- test: larql_core::test_edge::test_compact_json_serialization -->
<!-- test: larql_core::test_edge::test_source_type_as_str -->

### Requirement: Schema and type inference

The `larql_core::core::schema::Schema` registry SHALL store named
relation rules, expose them by name, infer entity types from incoming
or outgoing relation patterns, and round-trip through JSON. When two
rules match the same entity, the first registered rule SHALL win to
keep inference deterministic.

#### Scenario: Empty schema returns no inferences
- **WHEN** type inference is performed on an empty schema
- **THEN** the inference SHALL return no type and no rule lookup SHALL succeed
<!-- test: larql_core::test_schema::test_empty_schema -->

#### Scenario: Add and retrieve rules
- **WHEN** rules are added to the schema and queried by name
- **THEN** the registered rules SHALL be returned with their relation patterns and target types
<!-- test: larql_core::test_schema::test_add_and_get -->

#### Scenario: Outgoing and incoming patterns match
- **WHEN** an entity has outgoing or incoming edges matching a registered relation pattern
- **THEN** the schema SHALL return the rule's target type via the corresponding inference path
<!-- test: larql_core::test_schema::test_type_inference_outgoing -->
<!-- test: larql_core::test_schema::test_type_inference_incoming -->

#### Scenario: No-match and tie-breaking
- **WHEN** no rule matches the entity, and **WHEN** more than one rule matches
- **THEN** inference SHALL return no type when there is no match, and SHALL return the first registered rule's type on a tie
<!-- test: larql_core::test_schema::test_type_inference_no_match -->
<!-- test: larql_core::test_schema::test_type_inference_first_match_wins -->

#### Scenario: Schema round-trips through JSON
- **WHEN** a schema is serialized to JSON and back
- **THEN** every rule and its target type SHALL be preserved
<!-- test: larql_core::test_schema::test_schema_json_roundtrip -->

### Requirement: Selection, search, and inspection

`Graph` SHALL provide selection by `(subject, relation)` with optional
reverse traversal, full-graph search by entity name, single-hop walk
by relation that prefers the highest-confidence neighbor, subgraph
extraction rooted at an entity, edge-existence checks, and node
inspection. Search results SHALL be ordered deterministically by
insertion order on ties and SHALL respect a caller-supplied result
limit.

#### Scenario: Selection in forward and reverse direction
- **WHEN** `select` is called with and without the reverse flag
- **THEN** outgoing edges SHALL be returned in forward mode and incoming edges SHALL be returned in reverse mode
<!-- test: larql_core::test_graph::test_select -->
<!-- test: larql_core::test_graph::test_select_reverse -->

#### Scenario: Walk picks the highest-confidence neighbor and fails on missing relations
- **WHEN** `walk` is invoked along a chain of relations
- **THEN** each hop SHALL choose the neighbor with the highest confidence, and the walk SHALL fail when a hop has no matching edge
<!-- test: larql_core::test_graph::test_walk -->
<!-- test: larql_core::test_graph::test_walk_picks_highest_confidence -->
<!-- test: larql_core::test_graph::test_walk_fails_on_missing_hop -->
<!-- test: larql_core::test_components_walk::walk_picks_highest_confidence -->
<!-- test: larql_core::test_components_walk::walk_returns_none_on_missing_relation -->
<!-- test: larql_core::test_components_walk::walk_multi_hop -->

#### Scenario: Search is bounded, deterministic, and case-insensitive
- **WHEN** `search` is called with a query, a `max_results` cap, an empty query, or mixed-case input
- **THEN** results SHALL be capped, ordered by insertion order on ties, return nothing for an empty query or no match, and match case-insensitively
<!-- test: larql_core::test_graph::test_search -->
<!-- test: larql_core::test_graph::test_search_max_results -->
<!-- test: larql_core::test_graph::test_search_tie_order_is_insertion_order -->
<!-- test: larql_core::test_components_walk::search_empty_query -->
<!-- test: larql_core::test_components_walk::search_no_match -->
<!-- test: larql_core::test_components_walk::search_case_insensitive -->

#### Scenario: Subgraph, describe, exists, and node accessors
- **WHEN** subgraph extraction, `describe`, `exists`, or `node` is called on a known or unknown entity
- **THEN** the operation SHALL return the connected subgraph for known entities, an empty result for unknown entities, and the documented edge/triple existence answer
<!-- test: larql_core::test_graph::test_subgraph -->
<!-- test: larql_core::test_graph::test_subgraph_unknown_entity -->
<!-- test: larql_core::test_graph::test_describe -->
<!-- test: larql_core::test_graph::test_exists -->
<!-- test: larql_core::test_graph::test_node -->

#### Scenario: Counts, stats, listings, and node ordering
- **WHEN** `count`, `stats`, `list_relations`, `list_entities`, or the nodes accessor is invoked
- **THEN** counts and stats SHALL reflect the current graph contents, listings SHALL include every relation/entity, and nodes SHALL be sorted by name
<!-- test: larql_core::test_graph::test_count -->
<!-- test: larql_core::test_graph::test_stats -->
<!-- test: larql_core::test_graph::test_list_relations -->
<!-- test: larql_core::test_graph::test_list_entities -->
<!-- test: larql_core::test_graph::test_nodes_are_sorted_by_name -->

### Requirement: Merge and diff over graphs

`larql_core::algo::merge` SHALL merge two graphs into a target graph
using a configurable `MergeStrategy` covering union (take-first),
take-recent, take-higher-confidence, and combine-by-source-priority,
returning the count of newly inserted edges. `larql_core::algo::diff`
SHALL compute edge-level changes (added, removed, changed) where a
change is detected on confidence, source, metadata, or
parametric/document injection differences.

#### Scenario: Union merge inserts every previously absent triple
- **WHEN** two graphs are merged using the default union strategy
- **THEN** edges absent from the target SHALL be inserted, edges already present SHALL be retained as-is, and the return value SHALL count the new insertions
<!-- test: larql_core::test_algo::test_merge_graphs -->
<!-- test: larql_core::test_algo::test_merge_empty_into_existing -->
<!-- test: larql_core::test_algo::test_merge_into_empty -->
<!-- test: larql_core::test_new_algos::test_merge_union -->

#### Scenario: Strategy-aware merge respects confidence and source priority
- **WHEN** a `take-higher-confidence` or source-priority strategy merges two graphs that disagree on a triple
- **THEN** the resulting edge SHALL keep the higher-confidence value or the higher-priority source, leaving lower-priority payloads untouched in the source graph
<!-- test: larql_core::test_new_algos::test_merge_max_confidence -->
<!-- test: larql_core::test_new_algos::test_merge_max_confidence_keeps_higher -->
<!-- test: larql_core::test_new_algos::test_merge_source_priority -->

#### Scenario: Merge composes with deduplication
- **WHEN** a merge introduces edges that duplicate existing triples and `deduplicate` is then applied
- **THEN** the final graph SHALL contain a single edge per triple with the maximum-confidence payload retained
<!-- test: larql_core::test_graph::test_deduplicate_max_confidence -->
<!-- test: larql_core::test_components_walk::deduplicate_after_merge -->

#### Scenario: Diff reports added, removed, and changed edges
- **WHEN** `diff` compares two graphs that differ by edge presence, confidence, metadata, source, or injection flags
- **THEN** the result SHALL classify each difference as added, removed, or changed, and SHALL be empty when the graphs are identical
<!-- test: larql_core::test_new_algos::test_diff_identical -->
<!-- test: larql_core::test_new_algos::test_diff_added -->
<!-- test: larql_core::test_new_algos::test_diff_removed -->
<!-- test: larql_core::test_new_algos::test_diff_changed_confidence -->
<!-- test: larql_core::test_new_algos::test_diff_changed_metadata_source_and_injection -->

### Requirement: Traversal, walk, and components

`larql_core::algo` SHALL provide breadth-first and depth-first
traversal bounded by `max_depth`, all-paths walk between two entities
with an optional path limit, and connected-component decomposition.
BFS and DFS MUST start from the supplied entity and SHALL NOT visit
beyond the configured depth. Connected components SHALL produce a
deterministic, name-sorted ordering when sizes tie.

#### Scenario: BFS and DFS respect depth limits and unknown entities
- **WHEN** `bfs` or `dfs` is invoked with various depths and on a non-existent source
- **THEN** the traversal SHALL include only nodes within `max_depth`, return an empty edge list at depth zero, and yield no results for unknown entities
<!-- test: larql_core::test_new_algos::test_bfs_traversal -->
<!-- test: larql_core::test_new_algos::test_bfs_depth_limit -->
<!-- test: larql_core::test_new_algos::test_bfs_unknown_entity -->
<!-- test: larql_core::test_new_algos::test_dfs_traversal -->
<!-- test: larql_core::test_new_algos::test_dfs_depth_zero_has_no_traversed_edges -->

#### Scenario: All-paths walk enumerates and bounds results
- **WHEN** `walk_all_paths` is invoked with and without a `max_paths` cap, with no matching path, and over a single hop
- **THEN** every distinct path SHALL be returned within the cap, an empty result SHALL be returned when no path exists, and single-hop cases SHALL be included
<!-- test: larql_core::test_components_walk::walk_all_paths_finds_multiple -->
<!-- test: larql_core::test_components_walk::walk_all_paths_max_limit -->
<!-- test: larql_core::test_components_walk::walk_all_paths_no_match -->
<!-- test: larql_core::test_components_walk::walk_all_paths_single_hop -->

#### Scenario: Connected components are correct and deterministic
- **WHEN** `connected_components` is run on multi-component, single-component, equal-size, and empty graphs
- **THEN** the partition SHALL place every node in exactly one component, return one component for fully connected graphs, return an empty list for empty graphs, and order ties deterministically
<!-- test: larql_core::test_graph::test_connected_components -->
<!-- test: larql_core::test_graph::test_single_component -->
<!-- test: larql_core::test_components_walk::components_finds_two_components -->
<!-- test: larql_core::test_components_walk::components_equal_size_order_is_deterministic -->
<!-- test: larql_core::test_components_walk::components_europe_and_asia_separate -->
<!-- test: larql_core::test_components_walk::components_single_component -->
<!-- test: larql_core::test_components_walk::components_empty_graph -->

#### Scenario: are_connected handles trivial cases
- **WHEN** `are_connected` is asked about an entity and itself, and **WHEN** asked about a non-existent entity
- **THEN** the result SHALL be true for an entity-to-self query and false when either side is unknown
<!-- test: larql_core::test_components_walk::are_connected_same_node -->
<!-- test: larql_core::test_components_walk::are_connected_nonexistent -->

### Requirement: PageRank and shortest-path search

`larql_core::algo::pagerank` SHALL compute PageRank scores over the
graph's adjacency, exposing a `top_k` accessor; the empty graph SHALL
return an empty score map. `larql_core::algo::shortest_path` SHALL
compute weighted shortest paths preferring high-confidence edges,
SHALL provide an A* variant that accepts a heuristic, SHALL return
`None` when no route exists, and SHALL surface the count of nodes
explored on the result.

#### Scenario: PageRank covers normal, single-edge, and empty graphs
- **WHEN** PageRank is run on a multi-node graph, a single-edge graph, and an empty graph
- **THEN** the scores SHALL be finite and ranked, single-edge graphs SHALL distribute mass between source and target, and the empty graph SHALL yield empty scores
<!-- test: larql_core::test_new_algos::test_pagerank_basic -->
<!-- test: larql_core::test_new_algos::test_pagerank_single_edge -->
<!-- test: larql_core::test_new_algos::test_pagerank_empty -->

#### Scenario: Shortest path is direct, multi-hop, and confidence-aware
- **WHEN** `shortest_path` is asked between adjacent nodes, between distant nodes, between equally short routes that differ in confidence, between disconnected nodes, or between an entity and itself
- **THEN** it SHALL return the single-edge path, the minimum-hop path, the higher-confidence path on a tie, `None` for disconnected nodes, and an empty path for the same node
<!-- test: larql_core::test_algo::test_shortest_path_direct -->
<!-- test: larql_core::test_algo::test_shortest_path_multi_hop -->
<!-- test: larql_core::test_algo::test_shortest_path_prefers_high_confidence -->
<!-- test: larql_core::test_algo::test_shortest_path_returns_selected_multiedge -->
<!-- test: larql_core::test_algo::test_shortest_path_no_route -->
<!-- test: larql_core::test_algo::test_shortest_path_same_node -->

#### Scenario: A* finds, fails, and accepts a heuristic
- **WHEN** `astar` is called with no heuristic, with a heuristic, and on a disconnected pair
- **THEN** it SHALL return the optimal path when one exists, return `None` when no path exists, accept a non-trivial heuristic, and report `nodes_explored` on the result
<!-- test: larql_core::test_new_algos::test_astar_finds_path -->
<!-- test: larql_core::test_new_algos::test_astar_no_path -->
<!-- test: larql_core::test_new_algos::test_astar_with_heuristic -->
<!-- test: larql_core::test_new_algos::test_path_result_nodes_explored -->

### Requirement: Filter predicates over edges

`larql_core::algo::filter` SHALL filter a `Graph` by combinations of
confidence (min/max), relation whitelist or blacklist, source type,
metadata layer (min/max), metadata selectivity, and substring matches
on subject or object. Filters MUST preserve metadata, MUST behave as
the identity when no predicate is supplied, MUST compose so that all
predicates are conjoined, and MUST NOT panic on an empty graph.

#### Scenario: No predicates pass every edge through
- **WHEN** the filter is applied with no predicates set
- **THEN** every edge in the source graph SHALL be present in the filtered graph
<!-- test: larql_core::algo::filter::test_no_filters_passes_all -->

#### Scenario: Confidence bounds restrict edges
- **WHEN** `min_confidence` or `max_confidence` is set
- **THEN** only edges within `[min, max]` SHALL pass the filter
<!-- test: larql_core::algo::filter::test_min_confidence -->
<!-- test: larql_core::algo::filter::test_max_confidence -->

#### Scenario: Relation whitelist and blacklist
- **WHEN** a relation whitelist or blacklist is supplied
- **THEN** only listed (or unlisted) relations SHALL pass the filter
<!-- test: larql_core::algo::filter::test_relation_whitelist -->
<!-- test: larql_core::algo::filter::test_relation_blacklist -->

#### Scenario: Source, layer, and selectivity predicates
- **WHEN** predicates on `SourceType`, `min_layer`/`max_layer`, or `min_selectivity` are configured
- **THEN** only edges whose source/metadata fields satisfy the predicate SHALL pass
<!-- test: larql_core::algo::filter::test_source_filter -->
<!-- test: larql_core::algo::filter::test_min_layer -->
<!-- test: larql_core::algo::filter::test_max_layer -->
<!-- test: larql_core::algo::filter::test_min_selectivity -->

#### Scenario: Substring predicates and combination behave as conjunction
- **WHEN** `subject_contains` or `object_contains` predicates are configured, alone and in combination with other predicates
- **THEN** only edges that satisfy every active predicate SHALL pass the filter
<!-- test: larql_core::algo::filter::test_subject_contains -->
<!-- test: larql_core::algo::filter::test_object_contains -->
<!-- test: larql_core::algo::filter::test_combined_filters -->

#### Scenario: Metadata is preserved and empty graphs are handled
- **WHEN** the filter is applied to a graph with metadata, and **WHEN** it is applied to an empty graph
- **THEN** all metadata fields SHALL survive the filter and an empty graph SHALL produce an empty result without error
<!-- test: larql_core::algo::filter::test_preserves_metadata -->
<!-- test: larql_core::algo::filter::test_empty_graph -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_core::test_graph::**::* -->
<!-- test: larql_core::test_edge::**::* -->
<!-- test: larql_core::test_schema::**::* -->
<!-- test: larql_core::test_algo::**::* -->
<!-- test: larql_core::test_new_algos::**::* -->
<!-- test: larql_core::test_components_walk::**::* -->
<!-- test: larql_core::algo::filter::tests::**::* -->
