## ADDED Requirements

### Requirement: ModelProvider trait surface

`larql_core::engine::provider::ModelProvider` SHALL be the single
abstraction through which the inference engine obtains
language-model completions. The trait MUST be implementable by both
the bundled `MockProvider` (used in unit tests) and the
`HttpProvider` (used against external OpenAI-compatible endpoints).
The BFS and chain consumers MUST NOT depend on any concrete provider
type.

#### Scenario: BFS executes against a mock provider
- **WHEN** BFS is configured with a `MockProvider` and runs over a seed entity
- **THEN** the engine SHALL route every prompt through the trait and SHALL produce a graph populated by the mock's responses
<!-- test: larql_core::test_bfs_mock::test_bfs_basic -->
<!-- test: larql_core::test_bfs_mock::test_bfs_empty_provider -->

#### Scenario: Chain accepts a mock provider for token-by-token streams
- **WHEN** the chain generator is invoked with a `MockProvider` configured to return tokens
- **THEN** the chain SHALL emit those tokens until a stop condition fires, exercising only the trait surface
<!-- test: larql_core::test_chain::test_chain_single_token -->
<!-- test: larql_core::test_chain::test_chain_stops_on_empty_response -->

### Requirement: BFS expansion with confidence and entity caps

`larql_core::engine::bfs` SHALL implement breadth-first knowledge
expansion that, for each visited entity, renders a relation prompt
from the `TemplateRegistry`, asks the provider for candidates, and
records edges whose confidence meets the configured `min_confidence`.
Expansion MUST follow only entity-typed objects when crossing depth
boundaries, MUST honor `max_entities`, MUST NOT visit the same
entity twice, MUST stamp every emitted edge with the configured
parametric source/metadata, and SHALL terminate when the queue is
empty or a cap is reached. Engine prompt validation SHALL reject
non-entity strings before they enter the queue.

#### Scenario: Depth-1 expansion follows entity-typed objects only
- **WHEN** BFS runs at `max_depth = 1`
- **THEN** only entity-shaped objects SHALL be enqueued for further expansion, while non-entity values SHALL be recorded as terminal edges
<!-- test: larql_core::test_bfs_mock::test_bfs_depth_1_follows_entities -->

#### Scenario: Multiple seeds and entity caps are respected
- **WHEN** BFS is seeded with multiple roots, with a small `max_entities` cap
- **THEN** each seed SHALL contribute, and the run SHALL stop once `max_entities` is reached
<!-- test: larql_core::test_bfs_mock::test_bfs_multiple_seeds -->
<!-- test: larql_core::test_bfs_mock::test_bfs_respects_max_entities -->

#### Scenario: Confidence floor filters out low-quality edges
- **WHEN** the provider returns answers whose confidence is below `min_confidence`
- **THEN** those edges SHALL NOT be added to the resulting graph
<!-- test: larql_core::test_bfs_mock::test_bfs_respects_min_confidence -->

#### Scenario: Visits and result counts are deduplicated and reported
- **WHEN** BFS encounters the same entity through more than one path, and **WHEN** it completes
- **THEN** the engine SHALL NOT re-issue prompts for that entity and SHALL report visited and edge counts in the result
<!-- test: larql_core::test_bfs_mock::test_bfs_no_duplicate_visits -->
<!-- test: larql_core::test_bfs_mock::test_bfs_result_counts -->

#### Scenario: Emitted edges carry parametric source and metadata
- **WHEN** BFS records edges from provider responses
- **THEN** each edge SHALL carry the `Parametric` source type and the engine-supplied metadata fields
<!-- test: larql_core::test_bfs_mock::test_bfs_edges_have_source_parametric -->
<!-- test: larql_core::test_bfs_mock::test_bfs_edges_have_metadata -->

#### Scenario: Entity validator rejects sentences and lowercase fragments
- **WHEN** the engine's `is_valid_entity` predicate is consulted on candidate text
- **THEN** capitalized proper nouns and short numeric tokens SHALL be accepted while lowercase phrases, empty strings, and long sentences SHALL be rejected
<!-- test: larql_core::engine::bfs::test_valid_entity -->

### Requirement: Token-by-token chain generation

`larql_core::engine::chain` SHALL produce a chain of tokens by
delegating to a `ModelProvider` once per step, terminating early
when the next-token confidence falls below the configured threshold,
when the provider returns an empty response, or when an explicit
maximum is reached. The chain result SHALL expose the minimum
observed probability and SHALL be representable as an empty result
when no tokens were generated.

#### Scenario: Single-token chains halt after one step
- **WHEN** the chain generator is configured for a single token
- **THEN** the result SHALL contain exactly one token and the chain SHALL stop without invoking the provider again
<!-- test: larql_core::test_chain::test_chain_single_token -->

#### Scenario: Low confidence and empty responses stop the chain
- **WHEN** the provider's reported confidence drops below the threshold, and **WHEN** the provider returns no tokens at all
- **THEN** the chain SHALL stop in both cases without raising an error
<!-- test: larql_core::test_chain::test_chain_stops_on_low_confidence -->
<!-- test: larql_core::test_chain::test_chain_stops_on_empty_response -->

#### Scenario: Chain result reports min probability and supports empty
- **WHEN** the chain has produced one or more tokens, and **WHEN** it has produced none
- **THEN** the result SHALL expose the minimum probability across produced tokens and SHALL be representable as an empty result
<!-- test: larql_core::test_chain::test_chain_result_min_probability -->
<!-- test: larql_core::test_chain::test_chain_result_empty -->

### Requirement: Template registry lifecycle

`larql_core::engine::templates::TemplateRegistry` SHALL store named
prompt templates keyed by relation, expose registration and lookup,
render templates in subject and reverse-subject orientations, and
round-trip through JSON so registries can be persisted alongside
graphs. An empty registry SHALL return no templates and lookup
SHALL fail safely when a relation is unknown.

#### Scenario: Empty registry returns no templates
- **WHEN** the registry is constructed and queried before any registration
- **THEN** lookups SHALL return no template and listing SHALL be empty
<!-- test: larql_core::test_templates::test_empty_registry -->

#### Scenario: Register and retrieve a template
- **WHEN** a template is registered for a relation and looked up by that relation
- **THEN** the registered template SHALL be returned with its forward and reverse text
<!-- test: larql_core::test_templates::test_register_and_get -->

#### Scenario: Subject and reverse rendering substitute the entity
- **WHEN** `format` is called with a subject and the template defines either a forward or a forward-only orientation
- **THEN** the rendered prompt SHALL substitute the entity in the documented position, and SHALL fall back when reverse rendering is unavailable
<!-- test: larql_core::test_templates::test_format_subject -->
<!-- test: larql_core::test_templates::test_format_no_reverse -->

#### Scenario: Registry persists through JSON and example files
- **WHEN** a registry is serialized to JSON and reloaded, and **WHEN** the bundled example file is loaded
- **THEN** every registered relation, prompt, and reverse prompt SHALL be preserved
<!-- test: larql_core::test_templates::test_json_roundtrip -->
<!-- test: larql_core::test_templates::test_load_from_example_file -->

### Requirement: Provider implementations

The engine module SHALL ship a `MockProvider` that returns
caller-configured fixtures (used as the test double in BFS and chain
tests) and an `HttpProvider` that targets OpenAI-compatible HTTP
endpoints. Both implementations SHALL satisfy the `ModelProvider`
trait so that callers can swap between offline tests and live
inference without code changes outside provider construction.

#### Scenario: Mock provider drives both BFS and chain test suites
- **WHEN** the BFS and chain test suites construct a `MockProvider` from fixture data and run the engine
- **THEN** every BFS and chain test SHALL execute end-to-end against the trait without contacting any HTTP endpoint
<!-- test: larql_core::test_bfs_mock::test_bfs_basic -->
<!-- test: larql_core::test_bfs_mock::test_bfs_depth_1_follows_entities -->
<!-- test: larql_core::test_chain::test_chain_single_token -->
<!-- test: larql_core::test_chain::test_chain_stops_on_low_confidence -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_core::test_bfs_mock::**::* -->
<!-- test: larql_core::test_chain::**::* -->
<!-- test: larql_core::test_templates::**::* -->
<!-- test: larql_core::engine::bfs::tests::**::* -->
