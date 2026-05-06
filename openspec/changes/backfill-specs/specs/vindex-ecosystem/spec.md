## ADDED Requirements

### Requirement: HuggingFace path scheme and discovery

`larql_vindex::format::huggingface` SHALL recognise the
`hf://owner/repo[@revision]` URI scheme via `is_hf_path`,
parse owner / repo / revision via the canonical splitter, and
SHALL only accept paths that begin with `hf://` (so absolute and
relative filesystem paths cannot be mistaken for remote
references). Discovery SHALL list collection items, ensure a
collection exists in a namespace (creating it if absent), and
add items idempotently so re-runs are safe.

#### Scenario: hf:// scheme is recognised and rejected for filesystem paths
- **WHEN** `is_hf_path` is called on `hf://...` paths and on filesystem paths (`./local.vindex`, `/absolute/path`)
- **THEN** only the `hf://` paths SHALL return true
<!-- test: larql_vindex::format::huggingface::discovery::tests::test_is_hf_path -->

#### Scenario: hf path with revision parses cleanly
- **WHEN** `hf://chrishayuk/gemma-3-4b-it-vindex@v2.0` is split via the canonical parser
- **THEN** the result SHALL be repo `chrishayuk/gemma-3-4b-it-vindex` and revision `v2.0`
<!-- test: larql_vindex::format::huggingface::discovery::tests::test_parse_hf_path -->

### Requirement: HuggingFace publish, download, and resolve with checksum verification

The HuggingFace integration SHALL publish a vindex directory to a
HuggingFace repo with progress reporting, resolve a remote vindex
by name (with optional revision pin), and download artefacts to a
local cache. Downloads SHALL verify per-file checksums against
the manifest before considering the cache valid, and re-publish
SHALL be idempotent so the same source produces an identical
remote tree.

#### Scenario: PublishOptions exposes a stable default
- **WHEN** `PublishOptions::default()` is constructed
- **THEN** the defaults SHALL match the published publish flow (private flag, progress callback, repo type)
<!-- test: unbacked -->

### Requirement: Vindexfile DSL parser

`larql_vindex::vindexfile::parser` SHALL parse a Vindexfile that
declares a base vindex (`FROM`), zero or more `PATCH`,
`INSERT`, `DELETE`, `LABELS`, and `EXPOSE` directives at the
top level, plus zero or more named `STAGE` blocks containing
their own directives. The parser MUST treat lines starting with
`#` and blank lines as no-ops, MUST require at least one `FROM`
directive at the top level, and MUST accept `DELETE` either as a
triple form or as the explicit tuple form documented in the
ecosystem spec.

#### Scenario: Minimal Vindexfile parses
- **WHEN** a Vindexfile containing only a `FROM` directive is parsed
- **THEN** parsing SHALL succeed and the directive list SHALL contain exactly one `From`
<!-- test: larql_vindex::vindexfile::parser::tests::parse_minimal_vindexfile -->

#### Scenario: Full Vindexfile with every top-level directive
- **WHEN** a Vindexfile containing `FROM`, `PATCH`, `INSERT`, `DELETE`, `LABELS`, and `EXPOSE` is parsed
- **THEN** parsing SHALL produce one directive of each kind in input order
<!-- test: larql_vindex::vindexfile::parser::tests::parse_full_vindexfile -->

#### Scenario: STAGE blocks group their directives
- **WHEN** a Vindexfile with multiple `STAGE` blocks is parsed
- **THEN** each stage SHALL carry its own directive list and the top-level directive list SHALL contain only directives that appeared before the first `STAGE`
<!-- test: larql_vindex::vindexfile::parser::tests::parse_stages -->

#### Scenario: DELETE accepts the tuple form
- **WHEN** a `DELETE` directive in tuple form is parsed
- **THEN** parsing SHALL succeed and produce a `Delete { entity, relation, target }` value
<!-- test: larql_vindex::vindexfile::parser::tests::parse_delete_tuple_form -->

#### Scenario: A Vindexfile without FROM is an error
- **WHEN** a Vindexfile that lacks a top-level `FROM` is parsed
- **THEN** parsing SHALL return a structured parse error
<!-- test: larql_vindex::vindexfile::parser::tests::missing_from_is_error -->

#### Scenario: Comments and blank lines are ignored
- **WHEN** a Vindexfile containing `#` comments and blank lines is parsed
- **THEN** those lines SHALL not appear in any directive list and parsing SHALL succeed
<!-- test: larql_vindex::vindexfile::parser::tests::comments_and_blank_lines_ignored -->

### Requirement: K-means clustering for relation discovery

`larql_vindex::clustering::kmeans` SHALL provide a deterministic
k-means implementation that classifies a direction vector
against a set of centres via `classify_direction` and returns
both the chosen cluster index and the similarity score. The
clusterer MUST handle the degenerate single-cluster case without
panic.

#### Scenario: Basic k-means clusters known data
- **WHEN** k-means is run on a small synthetic dataset with k > 1
- **THEN** every input SHALL be assigned to a cluster and the cluster centres SHALL converge to the published baseline
<!-- test: larql_vindex::clustering::kmeans::tests::kmeans_basic -->

#### Scenario: Single-cluster k-means is a no-op
- **WHEN** k-means is run with k = 1
- **THEN** every input SHALL be assigned to cluster 0 and the centre SHALL equal the input mean
<!-- test: larql_vindex::clustering::kmeans::tests::kmeans_single_cluster -->

### Requirement: Auto-label clusters from vocabulary and patterns

`larql_vindex::clustering::labeling` SHALL produce TF-IDF cluster
labels from member vocabularies, filtering English stop words and
honouring pipe-separated multi-token members. It SHALL also
detect well-known entity patterns (country, language, month,
number, morphological) and SHALL return no pattern when none of
the rules match or when the member set is empty.
`larql_vindex::clustering::categories` SHALL provide the seeded
category vocabulary used by the labeller, with a path-based
fallback when the wikidata category file is absent.

#### Scenario: TF-IDF labels are computed and filtered
- **WHEN** TF-IDF is run on cluster members containing a mix of stop words, content tokens, and pipe-separated multi-token entries
- **THEN** the labels SHALL include only content tokens, SHALL split pipe-separated members, and SHALL drop English stop words
<!-- test: larql_vindex::clustering::labeling::tests::tfidf_labels_basic -->
<!-- test: larql_vindex::clustering::labeling::tests::tfidf_with_pipe_separated -->
<!-- test: larql_vindex::clustering::labeling::tests::tfidf_filters_stop_words -->

#### Scenario: Pattern detector classifies known entity classes
- **WHEN** the pattern detector is given clusters whose members are countries, languages, months, numbers, or morphological forms
- **THEN** it SHALL return the matching pattern, and SHALL return no pattern for unrelated members or empty input
<!-- test: larql_vindex::clustering::labeling::tests::detect_country_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_language_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_month_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_number_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_morphological_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_no_pattern -->
<!-- test: larql_vindex::clustering::labeling::tests::detect_empty_members -->

#### Scenario: Category seeds expose content vs stop words
- **WHEN** the seeded category vocabulary is queried
- **THEN** it SHALL be non-empty, contain key terms, classify English stop words as stop, classify content words as content, and SHALL fall back to a path-derived category when the wikidata file is absent
<!-- test: larql_vindex::clustering::categories::tests::category_words_not_empty -->
<!-- test: larql_vindex::clustering::categories::tests::category_words_contains_key_terms -->
<!-- test: larql_vindex::clustering::categories::tests::stop_words_detected -->
<!-- test: larql_vindex::clustering::categories::tests::content_words_not_stop -->
<!-- test: larql_vindex::clustering::categories::tests::category_words_from_path_fallback -->

### Requirement: Pair matching against external knowledge bases

`larql_vindex::clustering::pair_matching` SHALL match cluster
member pairs against external knowledge bases (Wikidata, WordNet)
loaded into a `ReferenceDatabase`, with a case-insensitive
lookup, support for multiple databases queried in priority
order, support for multiple relations sharing the same pair,
graceful behaviour on empty databases or empty cluster pairs,
and a configurable similarity threshold so weak matches do not
become labels.

#### Scenario: Lookup returns matching relations and is case-insensitive
- **WHEN** a `(subject, object)` pair is looked up against a database that contains it (or its case-folded variant)
- **THEN** the matching relation SHALL be returned regardless of letter casing
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_lookup -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_case_insensitive_lookup -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_add_relation -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_multiple_relations_same_pair -->

#### Scenario: Cluster labelling honours threshold and database order
- **WHEN** clusters are labelled against multiple databases with a configurable threshold
- **THEN** only clusters whose pair-match fraction meets the threshold SHALL be labelled, and databases SHALL be queried in their declared priority order
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_label_clusters -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_threshold_met -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_multiple_databases -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_mixed_databases -->

#### Scenario: Empty inputs are handled gracefully
- **WHEN** an empty database or an empty cluster-pair list is passed to the labeller
- **THEN** the labeller SHALL return no labels rather than panic
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_empty_database -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_empty_cluster_pairs -->

#### Scenario: Realistic Wikidata and WordNet pairs match
- **WHEN** realistic Wikidata pairs (capital-of, located-in, etc.) and WordNet synonym pairs are looked up
- **THEN** the matching relation labels SHALL be recovered, including partial matches per the published rules
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_realistic_wikidata_pairs -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_wordnet_synonym_matching -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::test_partial_matches -->

### Requirement: Token-level edge labelling with multi-source ranking

`larql_vindex::describe` SHALL define `LabelSource` (Probe,
Cluster, Pattern, KnnStore, None) and `DescribeEdge` carrying
relation, source, target, gate score, layer min/max, count, and
also-tokens. Probe-derived labels SHALL outrank cluster-derived
labels, which SHALL outrank TF-IDF / fallback labels, so
DESCRIBE never demotes a confirmed label to a guess.
`larql_vindex::clustering::probe` SHALL extract probe entities
from a JSON probe trace so probe-driven labels feed back into
the labeller.

#### Scenario: LabelSource Display covers every variant
- **WHEN** every `LabelSource` variant is rendered via `Display`
- **THEN** the strings SHALL be the published spellings (`probe`, `cluster`, `pattern`, empty for `None`, `knn` for `KnnStore`)
<!-- test: larql_vindex::describe::tests::label_source_display_all_variants -->

#### Scenario: LabelSource equality is reflexive
- **WHEN** two equal `LabelSource` values and two distinct values are compared
- **THEN** equal values SHALL be `==` and distinct values SHALL be `!=`
<!-- test: larql_vindex::describe::tests::label_source_equality -->

#### Scenario: DescribeEdge fields are constructible and observable
- **WHEN** a `DescribeEdge` is built with relation, source, target, gate score, layer span, count, and also-tokens
- **THEN** every field SHALL be readable on the resulting value
<!-- test: larql_vindex::describe::tests::describe_edge_fields_accessible -->

#### Scenario: Unlabelled DescribeEdge omits the relation
- **WHEN** a `DescribeEdge` is constructed with `relation = None`
- **THEN** the field SHALL be observably None and the source SHALL still be set
<!-- test: larql_vindex::describe::tests::describe_edge_none_relation -->

#### Scenario: Probe trace yields entities for the labeller
- **WHEN** `extract_entities_from_json` is called on a probe trace
- **THEN** the entities SHALL be returned in trace order so the labeller can rank probe labels above cluster and TF-IDF labels
<!-- test: larql_vindex::clustering::probe::tests::extract_entities_from_json -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::format::huggingface::discovery::tests::**::* -->
<!-- test: larql_vindex::vindexfile::parser::tests::**::* -->
<!-- test: larql_vindex::clustering::kmeans::tests::**::* -->
<!-- test: larql_vindex::clustering::labeling::tests::**::* -->
<!-- test: larql_vindex::clustering::categories::tests::**::* -->
<!-- test: larql_vindex::clustering::pair_matching::labeling::tests::**::* -->
<!-- test: larql_vindex::clustering::probe::tests::**::* -->
<!-- test: larql_vindex::describe::tests::**::* -->
