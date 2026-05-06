## ADDED Requirements

### Requirement: KvStrategy trait surface

The `kv_cache_benchmark::KvStrategy` trait SHALL be the single
abstraction every cache-rung implementation plugs into. Each
implementation MUST provide `name()`, `encode(keys, values) -> Vec<u8>`,
`decode(encoded, num_vectors, dim) -> (Vec<Vec<f32>>, Vec<Vec<f32>>)`,
and `memory_bytes(config, seq_len) -> usize`. The
`run_strategy_benchmark` driver SHALL produce a `StrategyResult`
populated with MSE, cosine similarity, inner-product error,
compression ratio, encode/decode microseconds, and original/encoded
byte counts so that every strategy is comparable on the same metric
slate.

#### Scenario: Standard KV benchmark fills out every metric field
- **WHEN** `run_strategy_benchmark(&StandardKv, &ModelConfig::gemma_4b(), 64, &mut rng)` is called
- **THEN** the result SHALL report `strategy_name == "Standard KV (FP16)"`, `seq_len == 64`, MSE below `0.001`, and cosine similarity above `0.999`
<!-- test: kv_cache_benchmark::test_standard::test_standard_kv_benchmark_runs -->

#### Scenario: TurboQuant benchmark reports lossy-but-aligned reconstruction
- **WHEN** `run_strategy_benchmark` is invoked on `TurboQuant::new(4)`
- **THEN** the result SHALL have a non-zero MSE, cosine similarity above `0.9`, and a compression ratio strictly greater than `1.0`
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_benchmark_runs -->

### Requirement: ModelConfig dimensions and analytical memory

`kv_cache_benchmark::model_config::ModelConfig` SHALL expose
real-model dimensions for at least Gemma 3-4B (34 layers, 2 KV
heads, 10 Q heads, 256 head dim, 2560 hidden, 10240 intermediate,
262144 vocab), Llama 3 8B, and Llama 3 70B. It MUST compute
analytical memory consistently — `kv_bytes_per_token() = layers ×
2 × kv_heads × head_dim × 2`, `kv_memory(seq_len) = seq_len ×
kv_bytes_per_token()`, and `kv_dim() = head_dim` — so any strategy
can derive a per-config memory bound without owning a sample.

#### Scenario: Standard KV memory matches the analytical formula
- **WHEN** `StandardKv.memory_bytes(&ModelConfig::gemma_4b(), 4096)` and `... 370_000` are evaluated
- **THEN** the first SHALL equal `4096 × 34 × 2 × 2 × 256 × 2` and the second SHALL fall in the `[20 GB, 30 GB]` range
<!-- test: kv_cache_benchmark::test_standard::test_standard_kv_memory_formula -->
<!-- test: kv_cache_benchmark::standard_kv::tests::test_standard_kv_memory_formula -->

#### Scenario: ModelConfig::all enumerates the benchmark fleet
- **WHEN** `ModelConfig::all()` is iterated
- **THEN** Standard KV SHALL be larger than TurboQuant 4-bit at 4K tokens for every config
<!-- test: kv_cache_benchmark::test_comparative::test_multi_model_memory -->

### Requirement: Standard KV (FP16) baseline

The `StandardKv` strategy SHALL roundtrip f32 vectors through an
IEEE-754 fp16 codec, returning reconstructed values whose
element-wise error is bounded by `< 0.01`. It MUST advertise
`memory_bytes` equal to `config.kv_memory(seq_len)` so the rest of
the rung ladder can be expressed as ratios against this single
source of truth.

#### Scenario: FP16 roundtrip stays within fp16 precision
- **WHEN** random vectors are encoded then decoded via `StandardKv`
- **THEN** every element-wise error SHALL be below `0.01`
<!-- test: kv_cache_benchmark::test_standard::test_standard_kv_exact_roundtrip -->
<!-- test: kv_cache_benchmark::standard_kv::tests::test_fp16_roundtrip -->

### Requirement: TurboQuant (3- and 4-bit) compression

The `TurboQuant` strategy SHALL combine a Walsh-Hadamard rotation
with Lloyd-Max scalar quantization at 3- or 4-bit precision. The
WHT MUST be self-inverse and norm-preserving (relative error
< `1e-4`) on the supported dimensions. Reconstruction MSE MUST stay
within paper bounds with margin (≤ `0.05` at 4-bit, ≤ `0.1` at
3-bit) and cosine similarity at 4-bit MUST stay above `0.95`.
Compression ratios for Gemma 3-4B MUST land in the published
`[2.5, 6.0]×` band at 4-bit and `[3.0, 7.0]×` band at 3-bit.

#### Scenario: WHT is self-inverse and norm-preserving
- **WHEN** the WHT is applied twice to a vector, and the L2 norm of the rotated vector is compared to the original
- **THEN** the reconstruction error SHALL be below `1e-4` and the relative norm change SHALL be below `1e-4`
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_wht_invertible -->
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_rotation_preserves_norm -->

#### Scenario: 4-bit and 3-bit MSE stay within paper-relative bounds
- **WHEN** 100 random vectors are encoded/decoded at 4-bit and at 3-bit
- **THEN** the average MSE SHALL be below `0.05` (4-bit) and `0.1` (3-bit), and 4-bit cosine similarity SHALL exceed `0.95`
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_4bit_mse_within_paper -->
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_3bit_mse_within_paper -->
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_cosine_above_threshold -->

#### Scenario: Compression ratios match the published band
- **WHEN** TurboQuant 3-bit and 4-bit memory are compared with Standard KV at 4K tokens on Gemma 3-4B
- **THEN** the 4-bit ratio SHALL fall in `(2.5, 6.0)` and the 3-bit ratio in `(3.0, 7.0)`
<!-- test: kv_cache_benchmark::test_turboquant::test_turboquant_compression_ratio -->

### Requirement: Markov Residual Stream cache elimination

`MarkovResidual::new(window_size)` SHALL eliminate the per-token KV
cache in favour of a bounded residual window plus a 4-byte-per-token
cold tier. Memory MUST remain dominated by the window at long
contexts (Standard KV / Markov RS > 100× at 370K tokens; Markov RS
< 10% of Standard KV for `seq_len ≥ 32 768`). Reconstruction inside
the active window MUST be bit-perfect (KL = 0.0). Checkpoint spacing
SHALL keep the maximum recompute distance at most 10 layers and
provide 5 named checkpoints for Gemma 3-4B.

#### Scenario: Cold tier dominates at very long contexts
- **WHEN** `MarkovResidual::new(512).memory_bytes` is compared to `StandardKv.memory_bytes` at 370K tokens
- **THEN** the ratio SHALL exceed `100×`
<!-- test: kv_cache_benchmark::test_markov::test_markov_cold_tier_size -->

#### Scenario: Memory growth past the window is exactly the cold tier
- **WHEN** `memory_bytes` is sampled at 4K, 32K, and 370K tokens
- **THEN** the growth from 32K to 370K SHALL equal `(370_000 − 32_768) × 4` bytes
<!-- test: kv_cache_benchmark::test_markov::test_markov_window_bounded -->

#### Scenario: Markov RS dominates Standard KV at scale
- **WHEN** memory is compared at 32K, 131K, and 370K tokens
- **THEN** Markov RS SHALL be smaller than `Standard KV / 10` at every length, and smaller than Standard KV at 4K
<!-- test: kv_cache_benchmark::test_markov::test_markov_much_smaller_than_standard -->

#### Scenario: Window reconstruction is bit-perfect
- **WHEN** 100 vectors fitting entirely inside the window are encoded then decoded
- **THEN** every element SHALL match the original within `1e-6` absolute error
<!-- test: kv_cache_benchmark::test_markov::test_markov_reconstruction_exact -->
<!-- test: kv_cache_benchmark::test_markov::test_markov_encode_decode -->

#### Scenario: Checkpoint config respects the recompute budget
- **WHEN** `CheckpointConfig::gemma_4b().max_recompute()` and `.layers.len()` are evaluated
- **THEN** the maximum recompute SHALL be at most `10` layers and the checkpoint count SHALL equal `5`
<!-- test: kv_cache_benchmark::test_markov::test_markov_checkpoint_spacing -->

### Requirement: UnlimitedContext (Tier 2) and Apollo (Tier 3) engines

`UnlimitedContextEngine` SHALL replay any closed window with the
exact same K/V tensors the live engine produced (cosine ≥
`0.99999`), SHALL be deterministic across replays of the same
window (cosine ≥ `0.999999`), and SHALL produce a stats summary
whose compression ratio is at least `5×` for a 256-token / 64-window
configuration. The engine SHALL preserve canonical output shapes
so callers can swap it into any inference pipeline. The Apollo
store loader SHALL accept the canonical Apollo 11 fixture (version
`12`, `3585` entries, `176` windows, `90 000` tokens, window size
`512`, crystal layer `30`, injection layer `30`, inject coefficient
`10.0`) and surface its boundaries (`176 × 2560` floats), window
tokens, and `VecInjectEntry` records without copy. The Apollo
routing + injection pipeline SHALL produce finite logits for both
the uncompressed and compressed forward paths, and the compressed
path SHALL forward at most 32 tokens (the compressed-context proof
of the ~20 000× claim).

#### Scenario: Window 0 replay is bit-exact against fresh extend
- **WHEN** a closed window is replayed and compared against a fresh `rs_extend_from_checkpoint` over the same tokens
- **THEN** per-layer K and V cosine SHALL exceed `0.99999`
<!-- test: kv_cache_benchmark::test_unlimited_context::test_window0_replay_bit_exact -->

#### Scenario: Replay is deterministic
- **WHEN** the same window is replayed twice
- **THEN** per-layer K and V cosine SHALL exceed `0.999999`
<!-- test: kv_cache_benchmark::test_unlimited_context::test_replay_is_deterministic -->

#### Scenario: Compression ratio holds for short prompts
- **WHEN** a 256-token prompt is processed with `window_size = 64`
- **THEN** at least 2 windows SHALL be archived and `stats.compression_ratio` SHALL exceed `5.0`
<!-- test: kv_cache_benchmark::test_unlimited_context::test_compression_ratio -->

#### Scenario: Extend output shapes match the model
- **WHEN** `rs_extend_from_checkpoint` runs against an empty prior
- **THEN** `last_hidden` SHALL be 1 row, `kv_cache.len()` SHALL equal `weights.num_layers`, and per-layer K/V SHALL have `tokens.len()` rows
<!-- test: kv_cache_benchmark::test_unlimited_context::test_extend_output_shapes -->

#### Scenario: Apollo manifest matches the apollo11 fixture
- **WHEN** `ApolloStore::load(apollo11_store)` is read
- **THEN** the manifest SHALL report `version=12`, `num_entries=3585`, `num_windows=176`, `num_tokens=90 000`, `window_size=512`, `crystal_layer=30`, `injection_layer=30`, and `inject_coefficient=10.0`
<!-- test: kv_cache_benchmark::test_apollo_store::test_load_apollo11_store_manifest -->

#### Scenario: Apollo boundaries and window tokens load with the right shapes
- **WHEN** boundaries, window tokens, and entries are inspected
- **THEN** there SHALL be `176` boundaries each of length `2560`, every window SHALL have at most `512` tokens with totals at least `num_tokens`, and every entry's `window_id`/`position_in_window` SHALL be in range
<!-- test: kv_cache_benchmark::test_apollo_store::test_load_apollo11_boundaries -->
<!-- test: kv_cache_benchmark::test_apollo_store::test_load_apollo11_window_tokens -->
<!-- test: kv_cache_benchmark::test_apollo_store::test_load_apollo11_entries -->
<!-- test: kv_cache_benchmark::test_apollo_store::test_apollo11_total_bytes_reasonable -->
<!-- test: kv_cache_benchmark::test_apollo_store::test_apollo11_entry_distribution -->
<!-- test: kv_cache_benchmark::test_apollo_store::test_entry_struct_roundtrips_cleanly -->

#### Scenario: Apollo routing resolves factual queries
- **WHEN** queries `"porridge eating contest"`, `"Corby England"`, and `"John Coyle"` are routed
- **THEN** the resolver SHALL return a non-empty list of windows for each
<!-- test: kv_cache_benchmark::test_apollo_query::test_routing_resolves_porridge_to_w170_region -->

#### Scenario: Apollo retrieve_entries respects the top_k bound
- **WHEN** entries are retrieved for a routed query
- **THEN** the returned entry count SHALL be ≤ `engine.config().top_k`
<!-- test: kv_cache_benchmark::test_apollo_query::test_retrieve_entries_for_query -->

#### Scenario: Apollo end-to-end query produces a finite top-1 logit
- **WHEN** `query_greedy` runs `"Who won the porridge eating contest?"` against the loaded store + model
- **THEN** the trace SHALL have at least one routed window, a positive context length, and a finite `top1_logit`
<!-- test: kv_cache_benchmark::test_apollo_query::test_end_to_end_query_produces_nonempty_answer -->

#### Scenario: Apollo compressed path forwards ≤ 32 tokens
- **WHEN** `query_greedy_compressed` is run against the same query
- **THEN** `trace.context_tokens` SHALL be ≤ `32` and the resulting `top1_logit` SHALL be finite
<!-- test: kv_cache_benchmark::test_apollo_query::test_end_to_end_query_compressed_path -->

#### Scenario: Apollo iterative compressed decode emits non-empty tokens
- **WHEN** `query_generate_compressed` runs for 25 max tokens
- **THEN** at least one token SHALL be generated and every per-step logit SHALL be finite
<!-- test: kv_cache_benchmark::test_apollo_query::test_apollo_generate_compressed -->

#### Scenario: Apollo accuracy sweep runs both forward paths
- **WHEN** `query_greedy` and `query_greedy_compressed` are invoked on each prompt in the canonical sweep
- **THEN** both calls SHALL succeed for every prompt and yield finite logits
<!-- test: kv_cache_benchmark::test_apollo_accuracy::test_apollo_accuracy_sweep -->

#### Scenario: Apollo side-by-side decoding works on both paths
- **WHEN** `query_generate_compressed` and `query_generate_uncompressed` run against the same query
- **THEN** both SHALL produce the same kind of trace (initial context tokens + generated token list)
<!-- test: kv_cache_benchmark::test_apollo_query::test_apollo_generate_side_by_side -->

### Requirement: Graph Walk projection (per-conversation only token IDs)

`GraphWalk::gemma_4b()` SHALL define a per-conversation cost of
`4 × seq_len` bytes (token IDs only) and a fixed shared infrastructure
cost in `[1 GB, 2 GB]`. Routing SHALL classify common factual
prompts (e.g. capital-of, currency-of, birthplace) as `WalkMode::Factual`,
extract the relation and entity, and resolve to the `CachedTemplate`
tier; free-form prompts SHALL fall back to `WalkTier::MarkovFallback`.
Pattern walks for cached templates MUST identify critical layers
(including layer `24`), keep mean cosine `> 0.99` across entities,
keep KNN lookups ≤ `10`, and estimate sub-millisecond latency.

#### Scenario: Graph Walk per-conversation memory is token-id only
- **WHEN** `GraphWalk::gemma_4b().memory_bytes(seq_len)` is evaluated
- **THEN** the result SHALL equal `seq_len × 4` and stay below `2 MB` even at 370K tokens
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_memory_tiny -->

#### Scenario: Shared infrastructure stays in the published band
- **WHEN** `shared_bytes()` is read
- **THEN** the value SHALL fall in `[1 GB, 2 GB]`
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_shared_infrastructure_size -->

#### Scenario: Routing detects France/Paris as factual capital-of
- **WHEN** `WalkState::from_tokens(["What", "is", "the", "capital", "of", "France"])` is constructed
- **THEN** mode SHALL be `Factual`, relation `capital-of`, entity `France`, tier `CachedTemplate`
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_france_paris_detection -->

#### Scenario: Routing covers capital, birthplace, and currency relations
- **WHEN** the canonical query suite is evaluated
- **THEN** every query SHALL detect as `Factual`, with the expected relation and entity, and `WalkState` SHALL classify all 50 capital/birthplace/currency prompts as factual
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_matches_forward_pass_detection -->
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_matches_forward_pass_50_queries -->

#### Scenario: Tier distribution stays realistic
- **WHEN** `TierDistribution::from_states(...)` is computed over a 10-prompt mix
- **THEN** Tier A count SHALL be > 0, Tier C count SHALL be > 0, and `(A+B)/total` SHALL fall in `(0.2, 0.9)`
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_routing_table_coverage -->

#### Scenario: Free-form prompts trigger Markov fallback
- **WHEN** prompts like `["tell", "me", "about", "your", "day"]` are routed
- **THEN** `WalkState.tier` SHALL be `MarkovFallback`
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_fallback_triggers -->

#### Scenario: Cached templates expose pattern-walk telemetry
- **WHEN** `PatternWalk::capital_of()` and `TemplateCache::with_defaults()` are inspected
- **THEN** `critical_layers` SHALL be non-empty and contain `24`, `mean_cosine > 0.99`, `knn_lookups() ≤ 10`, `estimated_latency_us() < 1000.0`, and `lookup("capital-of")` SHALL be `Some`, `lookup("nonexistent")` SHALL be `None`
<!-- test: kv_cache_benchmark::test_graph_walk::test_graph_walk_template_decomposition -->

### Requirement: Cross-strategy comparisons and shader micro-benchmarks

The `benchmark` module SHALL produce comparative tables and memory
sweeps that highlight the rung-by-rung win: Standard KV is always
the largest at non-trivial sequence lengths; Markov RS is bounded
by its window so it stays below `Standard / 10` for `seq_len ≥
32 768`; Graph Walk per-conversation is below Markov RS at every
length; and TurboQuant exceeds Markov RS at 370K tokens. The
`shader_bench` module SHALL run CPU-side micro-benchmarks for WHT
and TurboQuant encode/decode, returning timings + accuracy proxies
that downstream Metal/CUDA shaders are validated against.

#### Scenario: Memory ordering across strategies
- **WHEN** memory is sampled at 4K, 32K, and 370K tokens
- **THEN** Standard > TurboQuant, Standard > Markov RS, and Graph Walk < Markov RS at every length, and TurboQuant 4-bit > Markov RS at 370K
<!-- test: kv_cache_benchmark::test_comparative::test_all_strategies_memory_ordering -->

#### Scenario: 370K headline ratios match the rung ladder
- **WHEN** Standard / TurboQuant / Markov / Graph Walk are compared at 370K tokens
- **THEN** TurboQuant ratio SHALL fall in `(2.0, 8.0)`, Markov RS ratio SHALL exceed `100×`, and Graph Walk ratio SHALL exceed Markov RS
<!-- test: kv_cache_benchmark::test_comparative::test_370k_memory_ratios -->

#### Scenario: Memory sweep emits one point per (strategy, length)
- **WHEN** `benchmark::memory_sweep` runs over 3 strategies × 3 lengths
- **THEN** the sweep SHALL produce 9 non-zero points
<!-- test: kv_cache_benchmark::test_comparative::test_memory_sweep_produces_data -->

#### Scenario: Comparative table mentions every supported strategy
- **WHEN** `benchmark::format_comparative_table` is rendered
- **THEN** the output SHALL contain `Gemma 3-4B`, `Standard KV`, `TurboQuant`, and `Markov Residual Stream`
<!-- test: kv_cache_benchmark::test_comparative::test_comparative_table_format -->

#### Scenario: WHT and TurboQuant CPU shaders return positive throughput
- **WHEN** `bench_wht_cpu`, `bench_tq_encode_cpu`, and `bench_tq_decode_cpu` are run
- **THEN** every result SHALL have positive `time_us` and `throughput_ops_per_sec`
<!-- test: kv_cache_benchmark::test_shaders::test_wht_cpu_benchmark -->
<!-- test: kv_cache_benchmark::test_shaders::test_tq_encode_cpu_benchmark -->
<!-- test: kv_cache_benchmark::test_shaders::test_tq_decode_cpu_benchmark -->

#### Scenario: Shader roundtrip accuracy meets per-bit thresholds
- **WHEN** `bench_tq_roundtrip_cpu` is run at 4-bit and 3-bit
- **THEN** 4-bit MSE SHALL be `< 0.1` and cosine `> 0.9`; 3-bit MSE SHALL be `< 0.2` and cosine `> 0.85`
<!-- test: kv_cache_benchmark::test_shaders::test_tq_roundtrip_accuracy -->
<!-- test: kv_cache_benchmark::test_shaders::test_tq_3bit_roundtrip_accuracy -->

#### Scenario: Shader benchmark suite produces the canonical 10 results
- **WHEN** `run_cpu_benchmark_suite` is called
- **THEN** the result SHALL contain 10 rows (WHT × 2 dims + TQ encode/decode × 2 bits × 2 dims) all with positive timings
<!-- test: kv_cache_benchmark::test_shaders::test_full_cpu_benchmark_suite -->
<!-- test: kv_cache_benchmark::test_shaders::test_wht_d128_faster_than_d256 -->

### Requirement: Real-model integration top-1 and bit-perfect Markov

`run_all_strategies` SHALL drive every rung through the same forward pass against real Gemma 3-4B weights (under the `real-model` feature) and SHALL report a per-strategy `top1_token`. Standard KV and TurboQuant 4-bit MUST predict `Paris` for `"The capital of France is"`. Markov RS MUST be bit-perfect against Standard KV (top-1 match and hidden-state cosine > `0.9999` on every prompt). Multi-turn benchmarks MUST show Standard KV memory growing while Markov RS stays bounded. The engine performance benchmark MUST keep every engine's hidden-state cosine above `0.99` and its total bytes strictly below the Standard KV reference.

#### Scenario: All strategies route through the same forward pass for Paris
- **WHEN** `run_all_strategies` is run on `"The capital of France is"`
- **THEN** Standard KV and TurboQuant 4-bit SHALL contain `"Paris"` and Markov RS SHALL match the baseline top-1
<!-- test: kv_cache_benchmark::test_real_model::test_all_strategies_produce_paris -->

#### Scenario: Markov RS is bit-perfect across the default prompt set
- **WHEN** `run_all_strategies` is run on each `default_prompts()` entry
- **THEN** Markov RS top-1 SHALL match Standard KV's, and `hidden_cosine` SHALL exceed `0.9999`
<!-- test: kv_cache_benchmark::test_real_model::test_markov_rs_bit_perfect -->

#### Scenario: TurboQuant on real K/V meets cosine + compression bars
- **WHEN** TurboQuant 4-bit is applied to captured K/V on the canonical prompt
- **THEN** cosine SHALL exceed `0.98` and compression ratio SHALL exceed `3.0`
<!-- test: kv_cache_benchmark::test_real_model::test_turboquant_compression_on_real_vectors -->

#### Scenario: Multi-turn growth respects the rung ladder
- **WHEN** the prompt grows over 5 turns
- **THEN** Standard KV memory SHALL grow and Markov RS growth SHALL be measurably smaller
<!-- test: kv_cache_benchmark::test_real_model::test_multi_turn_memory_bounded -->

#### Scenario: Engine performance benchmark guards accuracy and memory
- **WHEN** `run_all_engines_bench` is run on the canonical prompts
- **THEN** every engine's `hidden_cosine` SHALL exceed `0.99` and its `total_bytes` SHALL be strictly less than `kv_ref_bytes`
<!-- test: kv_cache_benchmark::test_real_model::test_engine_performance -->

#### Scenario: Adversarial entity confusion stays grounded
- **WHEN** `run_all_strategies` is run on capital-of prompts for France/Germany/Japan
- **THEN** Markov RS top-1 SHALL equal the Standard KV top-1 for every prompt
<!-- test: kv_cache_benchmark::test_real_model::test_adversarial_entity_confusion -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: kv_cache_benchmark::test_standard::**::* -->
<!-- test: kv_cache_benchmark::test_turboquant::**::* -->
<!-- test: kv_cache_benchmark::test_markov::**::* -->
<!-- test: kv_cache_benchmark::test_unlimited_context::**::* -->
<!-- test: kv_cache_benchmark::test_apollo_accuracy::**::* -->
<!-- test: kv_cache_benchmark::test_apollo_store::**::* -->
<!-- test: kv_cache_benchmark::test_apollo_query::**::* -->
<!-- test: kv_cache_benchmark::test_graph_walk::**::* -->
<!-- test: kv_cache_benchmark::test_real_model::**::* -->
<!-- test: kv_cache_benchmark::test_comparative::**::* -->
<!-- test: kv_cache_benchmark::test_shaders::**::* -->
