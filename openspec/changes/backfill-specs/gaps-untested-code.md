# Untested Code Gap Report

**`target/llvm-cov/coverage.json` not found.** Falling back to a structural heuristic: every public symbol in a `.rs` file with no `#[test]` attribute is flagged as a candidate untested-code gap. Run `make coverage` to regenerate this report from real coverage measurements.

**Totals:** 393 file(s), 1704 pub symbol(s) with no inline `#[test]`.

## kv-cache-benchmark

- Files with no inline tests: **20**
- Public symbols in those files: **114**

| File | pub symbols |
|---|---:|
| `crates/kv-cache-benchmark/src/accuracy.rs` | 16 |
| `crates/kv-cache-benchmark/src/real_model/runner.rs` | 11 |
| `crates/kv-cache-benchmark/src/graph_walk/template.rs` | 10 |
| `crates/kv-cache-benchmark/src/benchmark.rs` | 9 |
| `crates/kv-cache-benchmark/src/real_model/decode_comparison.rs` | 9 |
| `crates/kv-cache-benchmark/src/accuracy_suite/runner.rs` | 9 |
| `crates/kv-cache-benchmark/src/model_config.rs` | 8 |
| `crates/kv-cache-benchmark/src/shader_bench.rs` | 7 |
| `crates/kv-cache-benchmark/src/graph_walk/routing_table.rs` | 6 |
| `crates/kv-cache-benchmark/src/accuracy_suite/needle.rs` | 6 |
| `crates/kv-cache-benchmark/src/metrics.rs` | 4 |
| `crates/kv-cache-benchmark/src/accuracy_suite/prompts.rs` | 4 |
| `crates/kv-cache-benchmark/src/lib.rs` | 3 |
| `crates/kv-cache-benchmark/src/real_model/graph_walk_layer.rs` | 3 |
| `crates/kv-cache-benchmark/src/real_model/kv_capture.rs` | 3 |
| `crates/kv-cache-benchmark/src/real_model/turboquant_layer.rs` | 2 |
| `crates/kv-cache-benchmark/examples/ffn_coverage.rs` | 1 |
| `crates/kv-cache-benchmark/examples/bit_budget_additivity_q4k.rs` | 1 |
| `crates/kv-cache-benchmark/examples/q4k_ffn_raw_bridge.rs` | 1 |
| `crates/kv-cache-benchmark/examples/patch_propagation_q4k.rs` | 1 |

## larql-cli

- Files with no inline tests: **74**
- Public symbols in those files: **287**

| File | pub symbols |
|---|---:|
| `crates/larql-cli/src/commands/dev/ov_rd/reports.rs` | 40 |
| `crates/larql-cli/src/commands/dev/ov_rd/address.rs` | 32 |
| `crates/larql-cli/src/commands/dev/ov_rd/basis.rs` | 16 |
| `crates/larql-cli/src/commands/dev/ov_rd/metrics.rs` | 13 |
| `crates/larql-cli/src/commands/dev/ov_rd/oracle_pq_address.rs` | 10 |
| `crates/larql-cli/src/commands/dev/ov_rd/stats.rs` | 9 |
| `crates/larql-cli/src/commands/dev/ov_rd/oracle_pq_forward.rs` | 9 |
| `crates/larql-cli/src/commands/dev/ov_rd/gamma_address.rs` | 8 |
| `crates/larql-cli/src/commands/dev/ov_rd/pq.rs` | 7 |
| `crates/larql-cli/src/commands/dev/ov_rd/input.rs` | 7 |
| `crates/larql-cli/src/commands/dev/ov_rd/oracle_pq_reports.rs` | 7 |
| `crates/larql-cli/src/commands/extraction/compile_cmd/save.rs` | 5 |
| `crates/larql-cli/src/commands/dev/ov_rd/oracle.rs` | 4 |
| `crates/larql-cli/src/commands/dev/ov_rd/zero_ablate.rs` | 3 |
| `crates/larql-cli/src/commands/dev/ov_rd/types.rs` | 3 |
| `crates/larql-cli/src/commands/dev/ov_rd/static_replace.rs` | 3 |
| `crates/larql-cli/src/commands/extraction/compile_cmd/detect.rs` | 3 |
| `crates/larql-cli/src/utils.rs` | 2 |
| `crates/larql-cli/src/commands/diagnostics/parity.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/hf_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/predict_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/circuit_discover_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/convert_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/residuals_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/attn_bottleneck_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/embedding_jump_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/weight_walk_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/bfs_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/verify_cmd.rs` | 2 |
| `crates/larql-cli/src/commands/extraction/kg_bench_cmd.rs` | 2 |

_(truncated; 44 more files have no inline tests)_

## larql-compute

- Files with no inline tests: **98**
- Public symbols in those files: **314**

| File | pub symbols |
|---|---:|
| `crates/larql-compute/src/metal/mod.rs` | 11 |
| `crates/larql-compute/src/metal/shaders/f32_gemv.rs` | 11 |
| `crates/larql-compute/src/metal/moe_dispatch.rs` | 9 |
| `crates/larql-compute/src/metal/decode/mod.rs` | 9 |
| `crates/larql-compute/src/metal/decode/diag.rs` | 9 |
| `crates/larql-compute/src/metal/decode/gpu_timing.rs` | 7 |
| `crates/larql-compute/src/metal/f32_ops.rs` | 6 |
| `crates/larql-compute/src/metal/shaders/q4k_q6k_qkv_proj.rs` | 6 |
| `crates/larql-compute/src/metal/direct_ops.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q4k_matmul.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/fused_ops.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q4kf_qkv_proj.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/rope.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q4k_qkv_proj.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q4k_geglu_down.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q6k_geglu_down.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q8_attn_proj.rs` | 5 |
| `crates/larql-compute/src/metal/decode/encode_ffn.rs` | 5 |
| `crates/larql-compute/src/metal/stages/qkv_proj.rs` | 5 |
| `crates/larql-compute/src/metal/shaders/q4k_ffn_gate_up_f16acc.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q4_matvec_v4.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/kv_attention.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q6k_geglu_gelu_tanh_down_cached.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q4k_ffn_gate_up.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q4k_ffn_gate_up_8sg.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/residual_inject.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q6k_matvec.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q4k_ffn_gate_up_coop.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/f16_gemv.rs` | 4 |
| `crates/larql-compute/src/metal/shaders/q4k_matvec_8sg.rs` | 4 |

_(truncated; 68 more files have no inline tests)_

## larql-core

- Files with no inline tests: **23**
- Public symbols in those files: **135**

| File | pub symbols |
|---|---:|
| `crates/larql-core/src/core/graph.rs` | 35 |
| `crates/larql-core/src/core/schema.rs` | 13 |
| `crates/larql-core/src/engine/templates.rs` | 12 |
| `crates/larql-core/src/core/edge.rs` | 8 |
| `crates/larql-core/src/engine/provider.rs` | 7 |
| `crates/larql-core/src/io/mod.rs` | 6 |
| `crates/larql-core/src/io/checkpoint.rs` | 5 |
| `crates/larql-core/src/algo/shortest_path.rs` | 5 |
| `crates/larql-core/src/engine/mock_provider.rs` | 4 |
| `crates/larql-core/src/engine/http_provider.rs` | 4 |
| `crates/larql-core/src/engine/chain.rs` | 4 |
| `crates/larql-core/src/io/msgpack.rs` | 4 |
| `crates/larql-core/src/io/json.rs` | 4 |
| `crates/larql-core/src/core/enums.rs` | 3 |
| `crates/larql-core/src/io/format.rs` | 3 |
| `crates/larql-core/src/algo/traversal.rs` | 3 |
| `crates/larql-core/src/algo/pagerank.rs` | 3 |
| `crates/larql-core/src/algo/diff.rs` | 3 |
| `crates/larql-core/src/io/csv.rs` | 2 |
| `crates/larql-core/src/algo/merge.rs` | 2 |
| `crates/larql-core/src/algo/walk.rs` | 2 |
| `crates/larql-core/src/algo/components.rs` | 2 |
| `crates/larql-core/src/core/node.rs` | 1 |

## larql-experts

- Files with no inline tests: **1**
- Public symbols in those files: **12**

| File | pub symbols |
|---|---:|
| `crates/larql-experts/expert-interface/src/lib.rs` | 12 |

## larql-inference

- Files with no inline tests: **46**
- Public symbols in those files: **263**

| File | pub symbols |
|---|---:|
| `crates/larql-inference/src/walker/vector_extractor.rs` | 25 |
| `crates/larql-inference/src/ffn/moe_remote/wire.rs` | 19 |
| `crates/larql-inference/src/experts/registry.rs` | 17 |
| `crates/larql-inference/src/capture.rs` | 14 |
| `crates/larql-inference/src/ffn/moe_remote/backend.rs` | 14 |
| `crates/larql-inference/src/ffn/moe_remote/shard.rs` | 12 |
| `crates/larql-inference/src/ffn/moe_remote/config.rs` | 10 |
| `crates/larql-inference/src/ffn/mod.rs` | 9 |
| `crates/larql-inference/src/forward/predict/dense.rs` | 9 |
| `crates/larql-inference/src/vindex/walk_config.rs` | 8 |
| `crates/larql-inference/src/engines/profiler.rs` | 8 |
| `crates/larql-inference/src/layer_graph/generate/gpu.rs` | 8 |
| `crates/larql-inference/src/experts/caller.rs` | 7 |
| `crates/larql-inference/src/forward/inference_weights.rs` | 7 |
| `crates/larql-inference/src/vindex/q4k_forward/interventions.rs` | 7 |
| `crates/larql-inference/src/ffn/remote/sharded.rs` | 7 |
| `crates/larql-inference/src/ffn/moe_remote/stream.rs` | 6 |
| `crates/larql-inference/src/forward/predict/types.rs` | 6 |
| `crates/larql-inference/src/layer_graph/generate/types.rs` | 6 |
| `crates/larql-inference/src/layer_graph/generate/lm_head.rs` | 6 |
| `crates/larql-inference/src/walker/attention_walker.rs` | 5 |
| `crates/larql-inference/src/engines/test_utils.rs` | 5 |
| `crates/larql-inference/src/layer_graph/generate/cpu.rs` | 5 |
| `crates/larql-inference/src/attention/gpu.rs` | 4 |
| `crates/larql-inference/src/experts/loader.rs` | 4 |
| `crates/larql-inference/src/ffn/moe_remote/router.rs` | 4 |
| `crates/larql-inference/src/forward/predict/ffn.rs` | 4 |
| `crates/larql-inference/src/attention/mod.rs` | 3 |
| `crates/larql-inference/src/engines/kv_engines/markov_residual/q4k.rs` | 3 |
| `crates/larql-inference/src/vindex/walk_ffn/helpers.rs` | 2 |

_(truncated; 16 more files have no inline tests)_

## larql-lql

- Files with no inline tests: **38**
- Public symbols in those files: **144**

| File | pub symbols |
|---|---:|
| `crates/larql-lql/src/parser/helpers.rs` | 25 |
| `crates/larql-lql/src/ast.rs` | 24 |
| `crates/larql-lql/src/executor/remote.rs` | 15 |
| `crates/larql-lql/src/executor/backend.rs` | 9 |
| `crates/larql-lql/src/executor/introspection.rs` | 6 |
| `crates/larql-lql/src/parser/mod.rs` | 5 |
| `crates/larql-lql/src/parser/lifecycle.rs` | 5 |
| `crates/larql-lql/src/parser/mutation.rs` | 5 |
| `crates/larql-lql/src/parser/query.rs` | 5 |
| `crates/larql-lql/src/executor/mod.rs` | 5 |
| `crates/larql-lql/src/parser/patch.rs` | 4 |
| `crates/larql-lql/src/executor/query/select.rs` | 3 |
| `crates/larql-lql/src/executor/mutation/mod.rs` | 3 |
| `crates/larql-lql/src/error.rs` | 2 |
| `crates/larql-lql/src/parser/introspection.rs` | 2 |
| `crates/larql-lql/src/executor/compact.rs` | 2 |
| `crates/larql-lql/src/executor/mutation/insert/balance.rs` | 2 |
| `crates/larql-lql/src/executor/mutation/insert/plan.rs` | 2 |
| `crates/larql-lql/src/parser/trace.rs` | 1 |
| `crates/larql-lql/src/executor/trace.rs` | 1 |
| `crates/larql-lql/src/executor/lifecycle/stats.rs` | 1 |
| `crates/larql-lql/src/executor/lifecycle/diff.rs` | 1 |
| `crates/larql-lql/src/executor/lifecycle/use_cmd.rs` | 1 |
| `crates/larql-lql/src/executor/lifecycle/extract.rs` | 1 |
| `crates/larql-lql/src/executor/query/mod.rs` | 1 |
| `crates/larql-lql/src/executor/query/walk.rs` | 1 |
| `crates/larql-lql/src/executor/query/infer_trace.rs` | 1 |
| `crates/larql-lql/src/executor/query/explain.rs` | 1 |
| `crates/larql-lql/src/executor/query/infer.rs` | 1 |
| `crates/larql-lql/src/executor/query/describe.rs` | 1 |

_(truncated; 8 more files have no inline tests)_

## larql-models

- Files with no inline tests: **20**
- Public symbols in those files: **105**

| File | pub symbols |
|---|---:|
| `crates/larql-models/src/validation.rs` | 41 |
| `crates/larql-models/src/weights.rs` | 16 |
| `crates/larql-models/src/config.rs` | 7 |
| `crates/larql-models/src/quant/ggml/legacy.rs` | 5 |
| `crates/larql-models/src/quant/ggml/q6_k.rs` | 4 |
| `crates/larql-models/src/quant/ggml/q4_k.rs` | 4 |
| `crates/larql-models/src/architectures/qwen.rs` | 2 |
| `crates/larql-models/src/architectures/gpt_oss.rs` | 2 |
| `crates/larql-models/src/architectures/tinymodel.rs` | 2 |
| `crates/larql-models/src/architectures/llama.rs` | 2 |
| `crates/larql-models/src/architectures/starcoder2.rs` | 2 |
| `crates/larql-models/src/architectures/granite.rs` | 2 |
| `crates/larql-models/src/architectures/gemma2.rs` | 2 |
| `crates/larql-models/src/architectures/gemma3.rs` | 2 |
| `crates/larql-models/src/architectures/deepseek.rs` | 2 |
| `crates/larql-models/src/architectures/mistral.rs` | 2 |
| `crates/larql-models/src/architectures/generic.rs` | 2 |
| `crates/larql-models/src/architectures/gemma4.rs` | 2 |
| `crates/larql-models/src/architectures/mixtral.rs` | 2 |
| `crates/larql-models/src/quant/ggml/quantize.rs` | 2 |

## larql-python

- Files with no inline tests: **5**
- Public symbols in those files: **22**

| File | pub symbols |
|---|---:|
| `crates/larql-python/src/trace_py.rs` | 8 |
| `crates/larql-python/src/vindex.rs` | 6 |
| `crates/larql-python/src/lib.rs` | 3 |
| `crates/larql-python/src/walk.rs` | 3 |
| `crates/larql-python/src/session.rs` | 2 |

## larql-server

- Files with no inline tests: **27**
- Public symbols in those files: **100**

| File | pub symbols |
|---|---:|
| `crates/larql-server/src/session.rs` | 14 |
| `crates/larql-server/src/band_utils.rs` | 13 |
| `crates/larql-server/src/routes/openai/schema/ast.rs` | 12 |
| `crates/larql-server/src/http.rs` | 7 |
| `crates/larql-server/src/routes/patches.rs` | 7 |
| `crates/larql-server/src/routes/expert/mod.rs` | 6 |
| `crates/larql-server/src/routes/warmup.rs` | 5 |
| `crates/larql-server/src/routes/relations.rs` | 3 |
| `crates/larql-server/src/routes/insert.rs` | 3 |
| `crates/larql-server/src/routes/describe.rs` | 3 |
| `crates/larql-server/src/routes/walk.rs` | 3 |
| `crates/larql-server/src/routes/select.rs` | 3 |
| `crates/larql-server/src/routes/mod.rs` | 2 |
| `crates/larql-server/src/routes/stats.rs` | 2 |
| `crates/larql-server/src/routes/expert/multi_layer_batch.rs` | 2 |
| `crates/larql-server/src/routes/expert/warmup.rs` | 2 |
| `crates/larql-server/src/routes/expert/cpu.rs` | 2 |
| `crates/larql-server/src/routes/expert/single.rs` | 2 |
| `crates/larql-server/src/grpc.rs` | 1 |
| `crates/larql-server/src/grpc_expert.rs` | 1 |
| `crates/larql-server/src/auth.rs` | 1 |
| `crates/larql-server/src/error.rs` | 1 |
| `crates/larql-server/src/routes/models.rs` | 1 |
| `crates/larql-server/src/routes/health.rs` | 1 |
| `crates/larql-server/src/routes/expert/metal.rs` | 1 |
| `crates/larql-server/src/routes/expert/batch_legacy.rs` | 1 |
| `crates/larql-server/src/routes/openai/schema/mask.rs` | 1 |

## larql-vindex

- Files with no inline tests: **37**
- Public symbols in those files: **193**

| File | pub symbols |
|---|---:|
| `crates/larql-vindex/src/index/mutate/mod.rs` | 15 |
| `crates/larql-vindex/src/index/types.rs` | 14 |
| `crates/larql-vindex/src/format/weights/write_layers.rs` | 12 |
| `crates/larql-vindex/src/clustering/pair_matching/database.rs` | 11 |
| `crates/larql-vindex/src/format/weights/write_f32.rs` | 11 |
| `crates/larql-vindex/src/index/compute/gate_knn/hnsw_lifecycle.rs` | 7 |
| `crates/larql-vindex/src/index/compute/gate_knn/dispatch.rs` | 7 |
| `crates/larql-vindex/src/index/storage/ffn_store/interleaved_q4k.rs` | 7 |
| `crates/larql-vindex/src/index/storage/lm_head/loaders.rs` | 7 |
| `crates/larql-vindex/src/format/huggingface/publish.rs` | 7 |
| `crates/larql-vindex/src/extract/build_helpers.rs` | 6 |
| `crates/larql-vindex/src/index/storage/ffn_store/interleaved_q4.rs` | 6 |
| `crates/larql-vindex/src/index/storage/ffn_store/interleaved.rs` | 6 |
| `crates/larql-vindex/src/format/weights/load.rs` | 6 |
| `crates/larql-vindex/src/index/compute/hnsw.rs` | 5 |
| `crates/larql-vindex/src/index/compute/router.rs` | 5 |
| `crates/larql-vindex/src/index/storage/ffn_store/q4k_cache.rs` | 5 |
| `crates/larql-vindex/src/index/storage/ffn_store/fp4.rs` | 5 |
| `crates/larql-vindex/src/format/down_meta.rs` | 4 |
| `crates/larql-vindex/src/index/storage/ffn_store/down.rs` | 4 |
| `crates/larql-vindex/src/index/storage/lm_head/knn.rs` | 4 |
| `crates/larql-vindex/src/format/weights/write_q4k/feature_major_down.rs` | 4 |
| `crates/larql-vindex/src/mmap_util.rs` | 3 |
| `crates/larql-vindex/src/vindexfile/mod.rs` | 3 |
| `crates/larql-vindex/src/index/compute/gate_knn/scores_batch.rs` | 3 |
| `crates/larql-vindex/src/index/storage/ffn_store/up.rs` | 3 |
| `crates/larql-vindex/src/index/storage/ffn_store/gate_q4.rs` | 3 |
| `crates/larql-vindex/src/format/huggingface/mod.rs` | 3 |
| `crates/larql-vindex/src/format/huggingface/download.rs` | 3 |
| `crates/larql-vindex/src/extract/callbacks.rs` | 2 |

_(truncated; 7 more files have no inline tests)_

## model-compute

- Files with no inline tests: **4**
- Public symbols in those files: **15**

| File | pub symbols |
|---|---:|
| `crates/model-compute/src/wasm/runtime.rs` | 8 |
| `crates/model-compute/src/wasm/session.rs` | 4 |
| `crates/model-compute/src/native/mod.rs` | 2 |
| `crates/model-compute/src/wasm/error.rs` | 1 |

