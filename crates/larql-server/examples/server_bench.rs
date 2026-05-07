//! Server benchmark — measures endpoint handler latency with synthetic data.
//!
//! Run: cargo run -p larql-server --example server_bench --release

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

use std::time::Instant;

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            larql_models::TopKEntry {
                token: token.to_string(),
                token_id: id,
                logit: score,
            },
            larql_models::TopKEntry {
                token: "also".to_string(),
                token_id: id + 1,
                logit: score * 0.5,
            },
        ],
    }
}

/// Build a realistic-ish index for benchmarking.
/// 8 layers, 1024 features/layer, 256 hidden dims.
fn bench_index() -> VectorIndex {
    let hidden = 256;
    let num_features = 1024;
    let num_layers = 8;

    let mut gate_vectors = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Random-ish gate vectors (deterministic seed via layer index)
        let mut g = Array2::<f32>::zeros((num_features, hidden));
        for f in 0..num_features {
            // Each feature has a primary direction + noise
            let primary = (f * 7 + layer * 13) % hidden;
            g[[f, primary]] = 1.0;
            for d in 0..hidden {
                let noise = ((f * 31 + d * 17 + layer * 53) % 100) as f32 / 10000.0;
                g[[f, d]] += noise;
            }
        }
        gate_vectors.push(Some(g));

        let metas: Vec<Option<FeatureMeta>> = (0..num_features)
            .map(|f| {
                let token = format!("tok_L{}_F{}", layer, f);
                let score = 0.3 + ((f * 7 + layer * 3) % 70) as f32 / 100.0;
                Some(make_meta(&token, f as u32 + layer as u32 * 10000, score))
            })
            .collect();
        down_meta.push(Some(metas));
    }

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

fn bench<F: Fn() -> R, R>(name: &str, warmup: usize, iters: usize, f: F) {
    // Warmup
    for _ in 0..warmup {
        let _ = f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    let throughput = iters as f64 / elapsed.as_secs_f64();

    println!(
        "  {:<30} {:>8.3}ms/op  {:>10.0} ops/sec  ({} iters, {:.1}ms total)",
        name,
        per_iter,
        throughput,
        iters,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!("larql-server benchmark — synthetic vindex operations\n");
    println!("Building index: 8 layers × 1024 features × 256 hidden...");

    let start = Instant::now();
    let index = bench_index();
    println!(
        "  Built in {:.0}ms\n",
        start.elapsed().as_secs_f64() * 1000.0
    );

    let patched = PatchedVindex::new(index);

    // Build some test queries
    let hidden = 256;
    let query_strong = {
        let mut q = Array1::<f32>::zeros(hidden);
        q[0] = 1.0;
        q[1] = 0.5;
        q
    };
    let query_spread = {
        let mut q = Array1::<f32>::zeros(hidden);
        for i in 0..hidden {
            q[i] = ((i * 7) % 100) as f32 / 100.0;
        }
        q
    };

    println!("── Gate KNN (single layer) ──");
    bench("gate_knn L0 top-5", 100, 10000, || {
        patched.gate_knn(0, &query_strong, 5)
    });
    bench("gate_knn L0 top-20", 100, 10000, || {
        patched.gate_knn(0, &query_strong, 20)
    });
    bench("gate_knn L4 spread query", 100, 10000, || {
        patched.gate_knn(4, &query_spread, 10)
    });

    println!("\n── Walk (multi-layer) ──");
    let all_layers = patched.loaded_layers();
    bench("walk 8 layers top-5", 50, 5000, || {
        patched.walk(&query_strong, &all_layers, 5)
    });
    bench("walk 8 layers top-20", 50, 5000, || {
        patched.walk(&query_strong, &all_layers, 20)
    });
    let knowledge_layers: Vec<usize> = (2..6).collect();
    bench("walk 4 layers (knowledge) top-10", 50, 5000, || {
        patched.walk(&query_strong, &knowledge_layers, 10)
    });

    println!("\n── Walk-FFN (decoupled inference) ──");
    bench("walk-ffn single layer", 100, 10000, || {
        patched.gate_knn(4, &query_strong, 8092)
    });
    bench("walk-ffn batched 8 layers", 50, 5000, || {
        let mut results = Vec::with_capacity(8);
        for &l in &all_layers {
            results.push(patched.gate_knn(l, &query_strong, 8092));
        }
        results
    });

    println!("\n── Describe simulation (walk + aggregate) ──");
    bench("describe (walk + edge merge)", 50, 2000, || {
        let trace = patched.walk(&query_strong, &all_layers, 20);
        let mut edges: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (_, hits) in &trace.layers {
            for hit in hits {
                let entry = edges.entry(hit.meta.top_token.clone()).or_insert(0.0);
                if hit.gate_score > *entry {
                    *entry = hit.gate_score;
                }
            }
        }
        let mut ranked: Vec<_> = edges.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(20);
        ranked
    });

    println!("\n── Select simulation (metadata scan) ──");
    bench("select scan L0 (1024 features)", 100, 10000, || {
        let metas = patched.down_meta_at(0).unwrap();
        let count = metas.iter().filter(|m| m.is_some()).count();
        count
    });
    bench("select scan all layers", 50, 2000, || {
        let mut total = 0;
        for l in &all_layers {
            if let Some(metas) = patched.down_meta_at(*l) {
                total += metas.iter().filter(|m| m.is_some()).count();
            }
        }
        total
    });
    bench("select with filter (score > 0.7)", 50, 2000, || {
        let mut matches = Vec::new();
        for l in &all_layers {
            if let Some(metas) = patched.down_meta_at(*l) {
                for (i, m) in metas.iter().enumerate() {
                    if let Some(meta) = m {
                        if meta.c_score > 0.7 {
                            matches.push((*l, i, meta.top_token.clone()));
                        }
                    }
                }
            }
        }
        matches.truncate(20);
        matches
    });

    println!("\n── Feature lookup ──");
    bench("feature_meta(0, 512)", 1000, 100000, || {
        patched.feature_meta(0, 512)
    });
    bench("feature_meta(7, 1023)", 1000, 100000, || {
        patched.feature_meta(7, 1023)
    });

    println!("\n── Probe label lookup ──");
    // Build synthetic probe labels (10% of features labelled)
    let mut probe_labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    for l in 0..8 {
        for f in (0..1024).step_by(10) {
            probe_labels.insert((l, f), format!("rel_L{}_F{}", l, f));
        }
    }
    println!("  {} probe labels loaded", probe_labels.len());

    bench("probe_label hit", 1000, 100000, || {
        probe_labels.get(&(4, 500))
    });
    bench("probe_label miss", 1000, 100000, || {
        probe_labels.get(&(4, 501))
    });
    bench("describe + label merge", 20, 1000, || {
        let trace = patched.walk(&query_strong, &all_layers, 20);
        let mut edges: Vec<(String, f32, Option<&str>)> = Vec::new();
        for (layer, hits) in &trace.layers {
            for hit in hits {
                let label = probe_labels.get(&(*layer, hit.feature)).map(|s| s.as_str());
                edges.push((hit.meta.top_token.clone(), hit.gate_score, label));
            }
        }
        edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        edges.truncate(20);
        edges
    });

    println!("\n── Relations simulation (token aggregation) ──");
    bench("relations (scan knowledge layers)", 20, 500, || {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for l in 2..6 {
            if let Some(metas) = patched.down_meta_at(l) {
                for meta in metas.iter().flatten() {
                    if meta.c_score >= 0.2 {
                        *counts.entry(meta.top_token.clone()).or_default() += 1;
                    }
                }
            }
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(50);
        sorted
    });

    println!("\n── Patch operations ──");
    let test_patch = || larql_vindex::VindexPatch {
        version: 1,
        base_model: "bench".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };
    // Measure apply+remove on a fresh PatchedVindex (reuses existing base via clone).
    // Note: clone cost dominates in debug builds. Run with --release for accurate numbers.
    bench("apply + remove patch (1 op)", 20, 200, || {
        let mut p = PatchedVindex::new(patched.base().clone());
        p.apply_patch(test_patch());
        p.remove_patch(0);
    });

    println!("\n── Cache simulation ──");
    // Simulate DESCRIBE cache behavior
    let mut cache: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();

    bench("cache miss (HashMap lookup)", 1000, 100000, || {
        cache.get("model:France:knowledge:20:5")
    });

    // Populate cache
    for i in 0..1000 {
        let key = format!("model:entity{}:knowledge:20:5", i);
        cache.insert(key, serde_json::json!({"entity": format!("entity{}", i)}));
    }

    bench("cache hit (HashMap lookup)", 1000, 100000, || {
        cache.get("model:entity500:knowledge:20:5")
    });

    bench("cache key construction", 1000, 100000, || {
        format!("{}:{}:{}:{}:{}", "model", "France", "knowledge", 20, 5)
    });

    println!("\n── Session simulation ──");
    bench("session clone + patch", 10, 200, || {
        let mut session = PatchedVindex::new(patched.base().clone());
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: "bench".into(),
            base_checksum: None,
            created_at: "2026-04-01".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: vec![
                larql_vindex::PatchOp::Delete {
                    layer: 0,
                    feature: 0,
                    reason: None,
                },
                larql_vindex::PatchOp::Delete {
                    layer: 1,
                    feature: 1,
                    reason: None,
                },
            ],
        };
        session.apply_patch(patch);
        session
    });

    bench("session walk (after patch)", 50, 2000, || {
        patched.walk(&query_strong, &all_layers, 10)
    });

    println!("\n── JSON serialization ──");
    let sample_response = serde_json::json!({
        "entity": "France",
        "model": "google/gemma-3-4b-it",
        "edges": [
            {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe"},
            {"target": "French", "gate_score": 35.2, "layer": 24},
            {"target": "Europe", "gate_score": 14.4, "layer": 25},
        ],
        "latency_ms": 12.3
    });

    bench("JSON serialize (describe resp)", 1000, 50000, || {
        serde_json::to_string(&sample_response).unwrap()
    });

    bench("JSON serialize (small)", 1000, 100000, || {
        serde_json::to_string(&serde_json::json!({"status": "ok"})).unwrap()
    });

    println!("\n── Embed service — token lookup ──");
    // Simulate the embed endpoint: index into the embedding table for each token.
    // In production the table is mmap'd; here we use a heap Array2 of the same
    // shape (Gemma 3 4B: 262208 × 2560 f32 = 2.68 GB).
    let embed_vocab = 262208usize;
    let embed_hidden = hidden; // use same hidden as bench index (256)
    let embed_table = {
        let mut e = Array2::<f32>::zeros((embed_vocab.min(8192), embed_hidden));
        // Populate first 8K rows with recognizable patterns
        for i in 0..e.shape()[0] {
            e[[i, i % embed_hidden]] = 1.0;
        }
        e
    };
    let vocab_cap = embed_table.shape()[0];
    let embed_scale = (embed_hidden as f32).sqrt(); // Gemma scale

    bench("embed single token (decode step)", 1000, 100_000, || {
        let tok_id = 9515usize % vocab_cap;
        let row = embed_table.row(tok_id);
        row.iter().map(|&v| v * embed_scale).sum::<f32>()
    });
    bench("embed 512-token prefill", 100, 5_000, || {
        let mut h = Array2::<f32>::zeros((512, embed_hidden));
        for (i, row) in h.rows_mut().into_iter().enumerate() {
            let tok_id = (i * 7 + 13) % vocab_cap;
            let src = embed_table.row(tok_id);
            for (dst, &src) in row.into_iter().zip(src.iter()) {
                *dst = src * embed_scale;
            }
        }
        h
    });
    bench(
        "embed 1-token binary encode (request)",
        1000,
        1_000_000,
        || {
            let mut buf = Vec::with_capacity(8);
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&9515u32.to_le_bytes());
            buf
        },
    );
    bench(
        "embed binary response encode (seq=1, hidden=256)",
        1000,
        100_000,
        || {
            let mut buf = Vec::with_capacity(8 + embed_hidden * 4);
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&(embed_hidden as u32).to_le_bytes());
            let row = embed_table.row(0);
            for &v in row.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf
        },
    );

    println!("\n── Embed service — logits projection ──");
    // Simulate /v1/logits: one matmul residual @ lm_head.T
    // At 256 hidden (bench size), this is cheaper than production.
    // Real Gemma 3 4B: 262208 × 2560 ~ 2ms CPU. Scale shown in note.
    let small_vocab = 1024usize; // representative sub-vocab for bench
    let lm_head = embed_table.slice(larql_vindex::ndarray::s![..small_vocab, ..]);
    let query = {
        let mut q = Array1::<f32>::zeros(embed_hidden);
        q[0] = 1.0;
        q[1] = 0.5;
        q[5] = 0.3;
        q
    };

    bench("logits dot (1024 vocab, hidden=256)", 100, 50_000, || {
        let mut scores: Vec<f32> = Vec::with_capacity(small_vocab);
        for row in lm_head.rows() {
            scores.push(row.iter().zip(query.iter()).map(|(&e, &r)| e * r).sum());
        }
        // Partial top-5 sort (representative of production argmax)
        if scores.len() >= 5 {
            scores.select_nth_unstable_by(5, |a, b| b.partial_cmp(a).unwrap());
            scores.truncate(5);
        }
        scores
    });

    bench(
        "logits binary response encode (5 tokens)",
        1000,
        500_000,
        || {
            let top5 = [
                (9515u32, 0.801f32),
                (235, 0.042),
                (100, 0.012),
                (5, 0.008),
                (1, 0.003),
            ];
            let resp = serde_json::json!({
                "top_k": top5.iter().map(|(id, p)| serde_json::json!({"token_id": id, "prob": p})).collect::<Vec<_>>(),
                "latency_ms": 2.1f32,
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    println!("  Note: production Gemma 3 4B logits = 262208 × 2560 ~ 2ms CPU, ~0.1ms Metal");

    // ── OpenAI-compat envelopes (encode-only synthetic timings) ──────────
    //
    // The OpenAI N0 endpoints add an envelope around the existing /v1/embed
    // and /v1/logits compute. These benches measure the JSON encode cost
    // for the envelope alone — total endpoint latency = compute time
    // (above) + envelope cost (below). Useful for validating the wire
    // shape doesn't dominate.
    println!("\n── OpenAI-compat envelopes (encode-only) ──");

    bench(
        "/v1/models OpenAI-shape JSON serialize",
        1000,
        100_000,
        || {
            let resp = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "gemma-3-4b-it",
                    "object": "model",
                    "created": 1746094800u64,
                    "owned_by": "larql",
                    "path": "/v1",
                    "features": 348160usize,
                    "loaded": true,
                }]
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    bench(
        "/v1/embeddings serialize (single, hidden=256)",
        1000,
        50_000,
        || {
            let emb: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
            let resp = serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "embedding": emb, "index": 0}],
                "model": "gemma-3-4b-it",
                "usage": {"prompt_tokens": 1, "total_tokens": 1}
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    bench(
        "/v1/embeddings serialize (batch=8, hidden=256)",
        500,
        20_000,
        || {
            let emb: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
            let data: Vec<serde_json::Value> = (0..8)
                .map(|i| serde_json::json!({"object": "embedding", "embedding": &emb, "index": i}))
                .collect();
            let resp = serde_json::json!({
                "object": "list",
                "data": data,
                "model": "gemma-3-4b-it",
                "usage": {"prompt_tokens": 8, "total_tokens": 8}
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    bench(
        "/v1/completions serialize (max_tokens=10)",
        1000,
        100_000,
        || {
            let resp = serde_json::json!({
                "id": "cmpl-abc123def456",
                "object": "text_completion",
                "created": 1746094800u64,
                "model": "gemma-3-4b-it",
                "choices": [{
                    "text": " Paris is the capital of France.",
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": null,
                }],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 7,
                    "total_tokens": 13,
                }
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    bench(
        "/v1/completions request validation (stream=true → 400)",
        1000,
        100_000,
        || {
            // Simulate the cheap path: parse body, check stream flag, return.
            let body = br#"{"prompt":"hi","max_tokens":1,"stream":true}"#;
            let req: serde_json::Value = serde_json::from_slice(body).unwrap();
            req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false)
        },
    );

    bench(
        "/v1/chat/completions serialize (assistant content)",
        1000,
        100_000,
        || {
            let resp = serde_json::json!({
                "id": "chatcmpl-abc123def456",
                "object": "chat.completion",
                "created": 1746094800u64,
                "model": "gemma-3-4b-it",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " Paris is the capital of France.",
                    },
                    "finish_reason": "stop",
                    "logprobs": null,
                }],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": 7,
                    "total_tokens": 23,
                }
            });
            serde_json::to_string(&resp).unwrap()
        },
    );

    bench(
        "/v1/chat/completions render gemma multi-turn (3 messages)",
        1000,
        100_000,
        || {
            // Mirror the rendering path for slice 2 chat templates —
            // measures string concat cost, not tokenisation.
            let messages = [
                ("system", "You are concise."),
                ("user", "Capital of France?"),
                ("assistant", "Paris."),
            ];
            let mut out = String::with_capacity(256);
            for (role, content) in messages {
                let role = if role == "assistant" { "model" } else { role };
                out.push_str(&format!("<start_of_turn>{role}\n{content}<end_of_turn>\n"));
            }
            out.push_str("<start_of_turn>model\n");
            out
        },
    );

    // ── Constrained decoding (slice 4 / N0.6) ────────────────────────────
    //
    // Fixed cost added to constrained-decoding requests over plain
    // sampling. Token-level mask cost (per-step `O(vocab × avg_token_len)`)
    // lives in the generate loop and isn't bench-able here without a
    // real backend.
    use larql_server::routes::openai::schema::{
        parse_schema_with, resolve_tool_choice, synth_tools_schema, Fsm, ObjectSchema,
        ParseOptions, Schema, ToolMode,
    };

    bench(
        "/v1/chat/completions FSM step Schema::Any (50-char object)",
        5_000,
        100_000,
        || {
            let mut fsm = Fsm::any();
            let _ = fsm.step_str(r#"{"name":"Alice","age":30,"role":"admin"}"#);
            fsm.is_complete()
        },
    );

    bench(
        "/v1/chat/completions FSM step strict Person schema",
        5_000,
        100_000,
        || {
            let schema = Schema::object(ObjectSchema {
                properties: [
                    ("name".to_string(), Schema::string()),
                    ("age".to_string(), Schema::integer()),
                ]
                .into_iter()
                .collect(),
                required: vec!["name".into(), "age".into()],
                additional: None,
            });
            let mut fsm = Fsm::new(schema);
            let _ = fsm.step_str(r#"{"name":"Bob","age":42}"#);
            fsm.is_complete()
        },
    );

    bench(
        "/v1/chat/completions parse_schema (Person, strict)",
        5_000,
        100_000,
        || {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"}
                },
                "required": ["name", "age"]
            });
            parse_schema_with(&schema, ParseOptions { strict: true }).unwrap()
        },
    );

    bench(
        "/v1/chat/completions synth_tools_schema (2 functions)",
        5_000,
        50_000,
        || {
            let tools = serde_json::json!([
                {"type": "function", "function": {"name": "calc",
                    "parameters": {"type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"]}}},
                {"type": "function", "function": {"name": "search",
                    "parameters": {"type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"]}}}
            ]);
            let names = vec!["calc".to_string(), "search".to_string()];
            let mode = resolve_tool_choice(true, None, &names).unwrap();
            synth_tools_schema(&tools, &mode).unwrap()
        },
    );

    bench(
        "/v1/chat/completions FSM tool-call OneOf (commit on name)",
        5_000,
        50_000,
        || {
            // Two tools distinguishable by `name` const — exercises
            // OneOf's parallel-branch tracking + commit-on-disambiguation.
            let tools = serde_json::json!([
                {"type": "function", "function": {"name": "calc",
                    "parameters": {"type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"]}}},
                {"type": "function", "function": {"name": "search",
                    "parameters": {"type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"]}}}
            ]);
            let names = ["calc".to_string(), "search".to_string()];
            let (schema, _) = synth_tools_schema(&tools, &ToolMode::Any).unwrap().unwrap();
            let mut fsm = Fsm::new(schema);
            let _ = fsm.step_str(r#"{"name":"calc","arguments":{"a":12,"b":30}}"#);
            (fsm.is_complete(), names.len())
        },
    );

    // ── Sampling extras (F18, F19, slice 4.10) ───────────────────────────

    bench(
        "Sampler with frequency_penalty (history N=8, vocab=256)",
        5_000,
        100_000,
        || {
            // Full-vocab logit slice with a small history triggers the
            // penalty path. Greedy under penalty so RNG cost is zero.
            let logits: Vec<f32> = (0..256u32).map(|i| i as f32 * 0.01).collect();
            let cfg = larql_inference::SamplingConfig::greedy()
                .with_frequency_penalty(0.5)
                .with_presence_penalty(0.3);
            let mut s = larql_inference::Sampler::new(cfg);
            let history = [10u32, 20, 30, 10, 200, 150, 99, 50];
            s.sample_with_history(&logits, &history)
        },
    );

    bench(
        "Sampler with temperature + top-p (no penalty)",
        5_000,
        50_000,
        || {
            let logits: Vec<f32> = (0..256u32).map(|i| i as f32 * 0.01).collect();
            let cfg = larql_inference::SamplingConfig::temperature(0.8)
                .with_top_p(0.9)
                .with_seed(42);
            let mut s = larql_inference::Sampler::new(cfg);
            s.sample(&logits)
        },
    );

    println!(
        "  Note: OpenAI envelope adds ~10-20 µs over the underlying compute.\n\
         Total /v1/embeddings latency = embed lookup (above) + ~5 µs encode.\n\
         Constrained-decoding fixed cost = parse_schema (~µs) + per-step\n\
         FSM clone+replay (~ns × token surface chars). Per-token mask cost\n\
         (vocab iteration) is dominated by the generate loop, not the FSM.\n\
         Repetition penalties add a HashMap-build + per-id subtraction\n\
         pass — negligible vs the lm_head matvec."
    );

    println!("\n── Summary ──");
    let total_features: usize = all_layers.iter().map(|l| patched.num_features(*l)).sum();
    println!(
        "  Index: {} layers, {} features/layer, {} total, hidden={}",
        all_layers.len(),
        1024,
        total_features,
        hidden
    );
    println!("  All times include full operation (KNN + sort + truncate + metadata)");
    println!("\n  Expected server latency = operation time + serialization + network RTT");
    println!("  Embed endpoint: dominated by table lookup (~O(1) with hot cache)");
    println!("  Logits endpoint: dominated by matmul (~2ms CPU / ~0.1ms Metal on 31B)");
}
