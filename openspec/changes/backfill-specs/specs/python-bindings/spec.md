## ADDED Requirements

### Requirement: Vindex loading and metadata

The `larql-python` crate SHALL expose `larql.load(path)` and the
underlying `PyVindex.open(path)` constructor that opens a vindex
directory and returns a Python object whose `repr` starts with
`Vindex(`. The Vindex MUST surface the on-disk metadata as Python
attributes — `num_layers`, `hidden_size`, `vocab_size`, `model`,
`family`, `total_gate_vectors`, `loaded_layers`, `num_features(layer)`,
`stats()`, and `layer_bands()` — and these values SHALL match the
contents of `index.json` exactly.

#### Scenario: Vindex loads from a synthetic directory
- **WHEN** `larql.load(tmpdir)` is called against a synthetic vindex
- **THEN** the returned object SHALL be non-`None` and `repr()` SHALL begin with `Vindex(`
<!-- test: py:tests/test_bindings.py::TestLoading::test_load -->

#### Scenario: Vindex properties match the on-disk config
- **WHEN** `num_layers`, `hidden_size`, `vocab_size`, `model`, `family`, and `total_gate_vectors` are read on a loaded Vindex
- **THEN** every property SHALL equal the value declared in `index.json` (and `total_gate_vectors == num_layers * num_features_per_layer`)
<!-- test: py:tests/test_bindings.py::TestLoading::test_properties -->
<!-- test: py:tests/test_bindings.py::TestLoading::test_num_features -->
<!-- test: py:tests/test_bindings.py::TestLoading::test_loaded_layers -->

#### Scenario: stats and layer bands are exposed as plain Python objects
- **WHEN** `stats()` and `layer_bands()` are called on a Vindex
- **THEN** they SHALL return a dict and a band-name → `(start, end)` tuple mapping respectively, mirroring `index.json["layer_bands"]`
<!-- test: py:tests/test_bindings.py::TestLoading::test_stats -->
<!-- test: py:tests/test_bindings.py::TestLoading::test_layer_bands -->

### Requirement: Numpy array I/O for embeddings, gate vectors, and tokens

The Vindex SHALL expose embeddings, gate vectors, and tokenizer access
as zero-copy numpy arrays of dtype `float32` with shapes that match the
declared model dimensions. `embed(text)`, `embedding(token_id)`,
`gate_vector(layer, feature)`, and `gate_vectors(layer)` MUST raise
`ValueError` for out-of-range indices rather than crash, and bulk
accessors MUST agree with their per-element equivalents.

#### Scenario: embed returns a hidden-size float32 vector
- **WHEN** `vindex.embed("hello")` is called
- **THEN** the result SHALL be a numpy `(hidden_size,)` array of dtype `float32` with non-zero norm
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_embed -->

#### Scenario: Different inputs produce different embeddings
- **WHEN** `embed("hello")` and `embed("world")` are compared
- **THEN** the two arrays SHALL NOT be element-wise close
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_embed_different_tokens -->

#### Scenario: Tokenizer roundtrip is available from Python
- **WHEN** `tokenize(text)` and `decode(ids)` are called on a Vindex
- **THEN** `tokenize` SHALL return a non-empty list of `int` token ids and `decode` SHALL return a `str`
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_tokenize -->
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_decode -->

#### Scenario: Embedding lookup by id and out-of-range error
- **WHEN** `embedding(token_id=0)` is called on a Vindex with a `(hidden_size,)` shape, and `embedding(token_id=999999)` is called for an out-of-range id
- **THEN** the first SHALL return a numpy `(hidden_size,)` array and the second SHALL raise `ValueError`
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_embedding_by_id -->
<!-- test: py:tests/test_bindings.py::TestEmbeddings::test_embedding_out_of_range -->

#### Scenario: Gate vector accessors agree across single and bulk paths
- **WHEN** `gate_vector(layer, feature)` is compared to the corresponding row of `gate_vectors(layer)`
- **THEN** the two arrays SHALL be element-wise close, the bulk shape SHALL be `(num_features, hidden_size)` with dtype `float32`, and out-of-range layer/feature indices SHALL raise `ValueError`
<!-- test: py:tests/test_bindings.py::TestGateVectors::test_gate_vector_single -->
<!-- test: py:tests/test_bindings.py::TestGateVectors::test_gate_vectors_layer -->
<!-- test: py:tests/test_bindings.py::TestGateVectors::test_gate_vectors_match_singles -->
<!-- test: py:tests/test_bindings.py::TestGateVectors::test_gate_vector_invalid_layer -->
<!-- test: py:tests/test_bindings.py::TestGateVectors::test_gate_vector_invalid_feature -->

### Requirement: KNN search and walk APIs

The Vindex SHALL expose KNN entry points (`gate_knn`, `entity_knn`)
returning `(feature, score)` tuples sorted by absolute score, and
walk entry points (`walk`, `entity_walk`) returning a list of
`WalkHit` objects with at minimum `layer` and `gate_score` attributes.
`gate_knn` MUST accept a Python list as its query vector so callers
do not have to allocate numpy arrays.

#### Scenario: gate_knn returns at most top_k tuples sorted by abs score
- **WHEN** `gate_knn(layer=0, query_vector=embed.tolist(), top_k=5)` is called
- **THEN** the result SHALL be a list of `(int, float)` tuples of length ≤ 5, sorted by `abs(score)` descending
<!-- test: py:tests/test_bindings.py::TestKNN::test_gate_knn -->

#### Scenario: entity_knn embeds the entity and returns hits
- **WHEN** `entity_knn("hello", layer=0, top_k=5)` is called on a populated synthetic vindex
- **THEN** the result SHALL be a non-empty list of hits
<!-- test: py:tests/test_bindings.py::TestKNN::test_entity_knn -->

#### Scenario: walk and entity_walk return WalkHit lists
- **WHEN** `walk(embed.tolist(), top_k=3)` and `entity_walk("hello", layers=[0, 1], top_k=3)` are called
- **THEN** each SHALL return a list of `WalkHit` whose entries expose `layer` and `gate_score`
<!-- test: py:tests/test_bindings.py::TestKNN::test_walk -->
<!-- test: py:tests/test_bindings.py::TestKNN::test_entity_walk -->
<!-- test: py:tests/test_bindings.py::TestKNN::test_walk_hit_properties -->

### Requirement: DESCRIBE, relations, and feature metadata

The Vindex SHALL provide a knowledge-graph surface — `describe`,
`has_edge`, `get_target`, `relations`, `cluster_centre`,
`typical_layer`, `feature_meta`, and `feature` — that returns
plain-Python lists, dicts, and `DescribeEdge` / `FeatureMeta`
objects without requiring the LQL parser. `describe` MUST accept
`band` ∈ {`syntax`, `knowledge`, `output`, `all`} and a `verbose`
flag whose verbose form returns a superset of the non-verbose edges.

#### Scenario: describe returns a list of edges with the expected attributes
- **WHEN** `describe("hello")` and `describe("hello", verbose=True)` are called
- **THEN** both SHALL return a `list` and the verbose form's edges SHALL expose `target`, `gate_score`, `relation`, `layer`, and `source` attributes
<!-- test: py:tests/test_bindings.py::TestDescribe::test_describe_returns_list -->
<!-- test: py:tests/test_bindings.py::TestDescribe::test_describe_edge_properties -->

#### Scenario: describe accepts every supported band
- **WHEN** `describe("hello", band=b)` is called for `b` in {syntax, knowledge, output, all}
- **THEN** each call SHALL return a `list` (possibly empty) without raising
<!-- test: py:tests/test_bindings.py::TestDescribe::test_describe_bands -->

#### Scenario: verbose describe is a superset of the default
- **WHEN** the lengths of `describe(..., verbose=False)` and `describe(..., verbose=True)` are compared
- **THEN** the verbose result SHALL have at least as many edges as the default
<!-- test: py:tests/test_bindings.py::TestDescribe::test_describe_verbose_more_edges -->

#### Scenario: has_edge and get_target return primitive types
- **WHEN** `has_edge("hello")` and `get_target("hello", "capital")` are called
- **THEN** the first SHALL return a `bool` and the second SHALL return `None` or `str`
<!-- test: py:tests/test_bindings.py::TestDescribe::test_has_edge -->
<!-- test: py:tests/test_bindings.py::TestDescribe::test_get_target -->

#### Scenario: relations and clusters degrade gracefully on a synthetic vindex
- **WHEN** `relations()`, `cluster_centre("capital")`, and `typical_layer("capital")` are called on a vindex with no `relation_clusters.json`
- **THEN** `relations()` SHALL return a `list`, and `cluster_centre` and `typical_layer` SHALL return `None`
<!-- test: py:tests/test_bindings.py::TestRelations::test_relations_list -->
<!-- test: py:tests/test_bindings.py::TestRelations::test_cluster_centre_none -->
<!-- test: py:tests/test_bindings.py::TestRelations::test_typical_layer_none -->

#### Scenario: feature_meta and feature do not crash on out-of-range indices
- **WHEN** `feature_meta(0, 0)` is called on a populated layer and `feature_meta(999, 999)` is called on an out-of-range pair
- **THEN** the first SHALL return `None` or a `FeatureMeta` whose `top_token` is a `str`, and the second SHALL return `None`
<!-- test: py:tests/test_bindings.py::TestFeatures::test_feature_meta -->
<!-- test: py:tests/test_bindings.py::TestFeatures::test_feature_dict -->
<!-- test: py:tests/test_bindings.py::TestFeatures::test_feature_meta_out_of_range -->

### Requirement: Mutation surface (insert, delete)

The Vindex SHALL expose `insert(entity, relation, target, layer=None, ...)`
which finds a free feature slot (optionally restricted to a hinted
layer) and writes a knowledge edge backed by the on-disk
`down_meta.bin` records, and `delete(entity, relation=None, layer=None)`
which removes matching edges and returns an integer count. When no
free feature slots are available `insert` MUST raise `RuntimeError`
with the message `No free feature slot` rather than corrupt the
store.

#### Scenario: insert succeeds or raises a structured error
- **WHEN** `vindex.insert("TestEntity", "capital", "TestCity")` is called
- **THEN** the call SHALL either return an `(int, int)` `(layer, feature)` tuple whose `feature_meta.top_token == "TestCity"`, or raise `RuntimeError` containing `"No free feature slot"`
<!-- test: py:tests/test_bindings.py::TestMutation::test_insert_or_skip -->

#### Scenario: insert respects a layer hint
- **WHEN** `insert("Hint", "rel", "City", layer=0)` is called and a free slot exists at layer 0
- **THEN** the returned `layer` SHALL equal `0`; otherwise a `RuntimeError` SHALL be raised
<!-- test: py:tests/test_bindings.py::TestMutation::test_insert_layer_hint_or_skip -->

#### Scenario: delete returns an integer match count
- **WHEN** `delete("NonexistentEntity123")` is called
- **THEN** the result SHALL be an `int` (zero is a valid count)
<!-- test: py:tests/test_bindings.py::TestMutation::test_delete -->

### Requirement: Session API and LQL bridge

`larql.session(path)` SHALL return a `PySession` whose `repr` starts
with `Session(`, exposes `query(lql) -> list[str]` and
`query_text(lql) -> str` for executing LQL programs against the
underlying vindex, and a `vindex` property granting numpy-grade
access to the same vindex without re-loading it.

#### Scenario: session opens against the same path as load
- **WHEN** `larql.session(vindex_path)` is called
- **THEN** the returned session's `repr()` SHALL begin with `Session(`
<!-- test: py:tests/test_bindings.py::TestSession::test_session_create -->

#### Scenario: session executes an LQL STATS query
- **WHEN** `session.query("STATS")` is called
- **THEN** the result SHALL be a `list[str]` with at least one element
<!-- test: py:tests/test_bindings.py::TestSession::test_session_query_stats -->

#### Scenario: session.query_text returns a joined string
- **WHEN** `session.query_text("STATS")` is called
- **THEN** the result SHALL be a non-empty `str`
<!-- test: py:tests/test_bindings.py::TestSession::test_session_query_text -->

#### Scenario: session.vindex shares state with the LQL engine
- **WHEN** `session.vindex` is read and used to call `embed("hello")`
- **THEN** the underlying object SHALL report the same `num_layers` as `larql.load(path)` and produce a `(hidden_size,)` numpy array
<!-- test: py:tests/test_bindings.py::TestSession::test_session_vindex_access -->

### Requirement: WalkModel zero-copy mmap inference

`larql.WalkModel(path, top_k=...)` SHALL load a vindex with
mmap-backed weights — never materialising them on the heap — and
expose `predict(prompt, top_k_predictions=...)` for full forward
passes plus `ffn_layer(layer, x_bytes, seq_len)` for per-layer
sparse FFN evaluation. `vindex.infer(prompt)` SHALL route through
`larql_inference::infer_patched` so that its top-k predictions are
byte-identical to the LQL `SELECT ... INFER` path (ADR 0001), and
both surfaces SHALL keep the load-time RSS low enough that a 4B
model fits in well under 8 GB and a `WalkModel` load fits in well
under 2 GB on the integration host.

#### Scenario: vindex.infer predicts Paris on the canonical prompt
- **WHEN** `real_vindex.infer("The capital of France is", top_k_predictions=3)` is called against the integration vindex
- **THEN** the top-1 result's token SHALL be `"Paris"` with probability above `0.5`
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_infer -->

#### Scenario: vindex.infer reuses mmap'd weights across calls
- **WHEN** a second `infer` call follows a warm-up `infer` call
- **THEN** the second call SHALL return the expected token (`"Jupiter"` for the largest-planet prompt) without re-loading the weights
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_infer_reuses_weights -->

#### Scenario: WalkModel exposes dimensions and predicts with mmap'd weights
- **WHEN** `larql.WalkModel(REAL_VINDEX, top_k=4096)` is constructed and `predict("The capital of France is")` is called
- **THEN** `num_layers > 0`, `hidden_size > 0`, and the top-1 prediction SHALL be `"Paris"`
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_walk_model -->

#### Scenario: WalkModel.ffn_layer takes and returns raw bytes
- **WHEN** `wm.ffn_layer(layer=0, x_bytes=<hidden_size f32 bytes>, seq_len=1)` is called
- **THEN** the result SHALL be `bytes` of length `hidden_size * 4`
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_walk_model_ffn_layer -->

#### Scenario: WalkModel and vindex.infer keep load RSS bounded
- **WHEN** `larql.WalkModel(REAL_VINDEX, top_k=256)` is loaded and `larql.load(REAL_VINDEX)` is called
- **THEN** the `RUSAGE_SELF` delta SHALL stay below 2000 MB for the WalkModel load and below 8000 MB for the vindex load
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_walk_model_memory -->
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_infer_memory -->

#### Scenario: Python infer is byte-identical to LQL INFER (ADR 0001)
- **WHEN** `vindex.infer(prompt, top_k_predictions=5)` is compared against the parsed top-k tokens of `session.query("INFER '{prompt}' TOP 5")` for prompts in {`"The capital of France is"`, `"Water is"`, `"hello"`}
- **THEN** the two token lists SHALL be equal element-wise
<!-- test: py:tests/test_bindings.py::TestV11InferParity::test_parity -->

#### Scenario: walk_ffn.load drives MLX generation through Rust FFN
- **WHEN** `larql.walk_ffn.load(REAL_VINDEX, top_k=4096)` returns a model + tokenizer pair and `mlx_lm.generate` is invoked with prompt `"The capital of France is"`
- **THEN** the generated string SHALL contain `"Paris"`
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_walk_ffn_mlx -->

#### Scenario: larql.mlx.load returns an MLX model from the vindex
- **WHEN** `larql.mlx.load(REAL_VINDEX)` is called with `mlx` and `mlx_lm` available
- **THEN** the returned model SHALL be non-`None`
<!-- test: py:tests/test_bindings.py::TestRealVindex::test_mlx_load -->
