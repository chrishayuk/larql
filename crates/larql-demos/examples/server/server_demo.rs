//! Server demo — builds a synthetic vindex and shows what the server would return.
//!
//! Run: cargo run -p larql-demos --example server_demo

use larql_vindex::ndarray::Array2;
use larql_vindex::{FeatureMeta, PatchOp, PatchedVindex, VectorIndex, VindexPatch};

use std::collections::HashMap;

fn section(title: &str) {
    println!("\n══ {} ══", title);
}

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32, also: &[(&str, u32, f32)]) -> FeatureMeta {
    let mut top_k = vec![make_top_k(token, id, score)];
    for &(t, i, s) in also {
        top_k.push(make_top_k(t, i, s));
    }
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k,
    }
}

/// Build a richer test index simulating a Gemma-style model.
/// 4 layers (L0=syntax, L1=knowledge, L2=knowledge, L3=output), hidden=8.
fn demo_index() -> (VectorIndex, Array2<f32>) {
    let hidden = 8;
    let num_layers = 4;
    let num_features = 5;

    // Gate vectors — each feature responds to a specific direction
    let mut gates = Vec::new();
    for _ in 0..num_layers {
        let mut g = Array2::<f32>::zeros((num_features, hidden));
        for f in 0..num_features {
            g[[f, f % hidden]] = 1.0;
            if f + 1 < hidden {
                g[[f, (f + 1) % hidden]] = 0.3; // slight bleed
            }
        }
        gates.push(Some(g));
    }

    // Down metadata — what each feature outputs
    let meta0 = vec![
        Some(make_meta("the", 1, 0.30, &[("a", 2, 0.2)])),
        Some(make_meta("is", 3, 0.35, &[("was", 4, 0.25)])),
        Some(make_meta("of", 5, 0.40, &[("in", 6, 0.3)])),
        Some(make_meta("and", 7, 0.25, &[("or", 8, 0.15)])),
        Some(make_meta("to", 9, 0.20, &[])),
    ];
    let meta1 = vec![
        Some(make_meta(
            "Paris",
            100,
            0.95,
            &[("Berlin", 101, 0.8), ("Tokyo", 102, 0.7)],
        )),
        Some(make_meta(
            "French",
            110,
            0.88,
            &[("German", 111, 0.75), ("Spanish", 112, 0.6)],
        )),
        Some(make_meta(
            "Europe",
            120,
            0.75,
            &[("Asia", 121, 0.65), ("Africa", 122, 0.5)],
        )),
        Some(make_meta("Republic", 130, 0.60, &[("Kingdom", 131, 0.5)])),
        Some(make_meta("Napoleon", 140, 0.70, &[("Caesar", 141, 0.55)])),
    ];
    let meta2 = vec![
        Some(make_meta("capital", 200, 0.92, &[("city", 201, 0.7)])),
        Some(make_meta("language", 210, 0.85, &[("dialect", 211, 0.5)])),
        Some(make_meta("continent", 220, 0.80, &[("region", 221, 0.6)])),
        Some(make_meta("government", 230, 0.55, &[])),
        Some(make_meta("leader", 240, 0.65, &[("president", 241, 0.5)])),
    ];
    let meta3 = vec![
        Some(make_meta(".", 300, 0.15, &[])),
        Some(make_meta(",", 301, 0.12, &[])),
        Some(make_meta("France", 302, 0.50, &[("Germany", 303, 0.4)])),
        Some(make_meta("Paris", 304, 0.45, &[])),
        Some(make_meta("is", 305, 0.10, &[])),
    ];

    let down_meta = vec![Some(meta0), Some(meta1), Some(meta2), Some(meta3)];

    let index = VectorIndex::new(gates, down_meta, num_layers, hidden);

    // Embeddings: 10 tokens, hidden=8
    let mut embed = Array2::<f32>::zeros((10, hidden));
    // "France" → token 0 → strong dim 0
    embed[[0, 0]] = 1.0;
    embed[[0, 1]] = 0.2;
    // "Germany" → token 1 → strong dim 1
    embed[[1, 1]] = 1.0;
    embed[[1, 0]] = 0.15;
    // "Paris" → token 2 → dim 2
    embed[[2, 2]] = 1.0;
    // "capital" → token 3 → dim 3
    embed[[3, 3]] = 1.0;

    (index, embed)
}

fn main() {
    println!("larql-server demo — synthetic vindex API simulation\n");

    let (index, embed) = demo_index();
    let patched = PatchedVindex::new(index);

    // ── 1. DESCRIBE (GET /v1/describe?entity=France) ──
    section("GET /v1/describe?entity=France");

    let query = embed.row(0).mapv(|v| v * 1.0); // "France" embedding
    let all_layers = patched.loaded_layers();
    let trace = patched.walk(&query, &all_layers, 5);

    let mut edges: Vec<(String, f32, usize)> = Vec::new();
    for (layer, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score > 0.1 && hit.meta.top_token.len() >= 2 {
                edges.push((hit.meta.top_token.clone(), hit.gate_score, *layer));
            }
        }
    }
    edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    edges.truncate(10);

    println!("{{");
    println!("  \"entity\": \"France\",");
    println!("  \"model\": \"demo/test-model\",");
    println!("  \"edges\": [");
    for (i, (target, score, layer)) in edges.iter().enumerate() {
        let comma = if i < edges.len() - 1 { "," } else { "" };
        println!(
            "    {{\"target\": \"{}\", \"gate_score\": {:.1}, \"layer\": {}}}{}",
            target, score, layer, comma
        );
    }
    println!("  ]");
    println!("}}");

    // ── 2. WALK (GET /v1/walk?prompt=France&top=3) ──
    section("GET /v1/walk?prompt=France&top=3");

    let trace = patched.walk(&query, &all_layers, 3);
    println!("{{");
    println!("  \"prompt\": \"France\",");
    println!("  \"hits\": [");
    let mut all_hits = Vec::new();
    for (layer, hits) in &trace.layers {
        for hit in hits {
            all_hits.push((*layer, hit));
        }
    }
    for (i, (layer, hit)) in all_hits.iter().enumerate() {
        let comma = if i < all_hits.len() - 1 { "," } else { "" };
        println!(
            "    {{\"layer\": {}, \"feature\": {}, \"gate_score\": {:.1}, \"target\": \"{}\"}}{}",
            layer,
            hit.feature,
            hit.gate_score,
            hit.meta.top_token.trim(),
            comma
        );
    }
    println!("  ]");
    println!("}}");

    // ── 3. SELECT (POST /v1/select) ──
    section("POST /v1/select {layer: 1, limit: 5}");

    println!("{{");
    println!("  \"edges\": [");
    if let Some(metas) = patched.down_meta_at(1) {
        let mut rows: Vec<(usize, &FeatureMeta)> = metas
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.as_ref().map(|m| (i, m)))
            .collect();
        rows.sort_by(|a, b| b.1.c_score.partial_cmp(&a.1.c_score).unwrap());
        for (i, (feat, meta)) in rows.iter().enumerate() {
            let comma = if i < rows.len() - 1 { "," } else { "" };
            println!(
                "    {{\"layer\": 1, \"feature\": {}, \"target\": \"{}\", \"c_score\": {:.2}}}{}",
                feat, meta.top_token, meta.c_score, comma
            );
        }
    }
    println!("  ]");
    println!("}}");

    // ── 4. RELATIONS (GET /v1/relations) ──
    section("GET /v1/relations");

    let mut token_counts: HashMap<String, usize> = HashMap::new();
    for layer in &all_layers {
        if let Some(metas) = patched.down_meta_at(*layer) {
            for meta in metas.iter().flatten() {
                if meta.top_token.len() >= 2 && meta.c_score >= 0.2 {
                    *token_counts.entry(meta.top_token.clone()).or_default() += 1;
                }
            }
        }
    }
    let mut sorted: Vec<_> = token_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("{{");
    println!("  \"relations\": [");
    for (i, (name, count)) in sorted.iter().take(10).enumerate() {
        let comma = if i < sorted.len().min(10) - 1 {
            ","
        } else {
            ""
        };
        println!(
            "    {{\"name\": \"{}\", \"count\": {}}}{}",
            name, count, comma
        );
    }
    println!("  ],");
    println!("  \"total\": {}", token_counts.len());
    println!("}}");

    // ── 5. STATS (GET /v1/stats) ──
    section("GET /v1/stats");

    let total_features = all_layers
        .iter()
        .map(|l| patched.num_features(*l))
        .sum::<usize>();
    println!("{{");
    println!("  \"model\": \"demo/test-model\",");
    println!("  \"layers\": {},", all_layers.len());
    println!("  \"features\": {},", total_features);
    println!("  \"hidden_size\": {},", patched.hidden_size());
    println!("  \"loaded\": {{\"browse\": true, \"inference\": false}}");
    println!("}}");

    // ── 6. PATCH APPLY + DESCRIBE (POST /v1/patches/apply) ──
    section("POST /v1/patches/apply → GET /v1/describe");

    let mut patched_mut = PatchedVindex::new(demo_index().0);

    println!("Before patch:");
    let trace_before = patched_mut.walk(&query, &[1], 3);
    for (_, hits) in &trace_before.layers {
        for hit in hits.iter().take(3) {
            println!("  L1: {} (gate={:.1})", hit.meta.top_token, hit.gate_score);
        }
    }

    let patch = VindexPatch {
        version: 1,
        base_model: "demo".into(),
        base_checksum: None,
        created_at: "2026-04-01T00:00:00Z".into(),
        description: Some("medical-facts".into()),
        author: Some("demo".into()),
        tags: vec!["medical".into()],
        operations: vec![PatchOp::Update {
            layer: 1,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "Aspirin".into(),
                top_token_id: 500,
                c_score: 0.99,
            }),
        }],
    };

    patched_mut.try_apply_patch(patch).expect("patch applies");
    println!("\nAfter patch (feature 0 → Aspirin):");
    let trace_after = patched_mut.walk(&query, &[1], 3);
    for (_, hits) in &trace_after.layers {
        for hit in hits.iter().take(3) {
            println!("  L1: {} (gate={:.1})", hit.meta.top_token, hit.gate_score);
        }
    }
    println!("\nActive patches: {}", patched_mut.num_patches());

    // ── 7. PROBE LABELS (relation classifier in DESCRIBE) ──
    section("DESCRIBE with probe labels");

    let mut probe_labels: HashMap<(usize, usize), String> = HashMap::new();
    probe_labels.insert((1, 0), "capital".into());
    probe_labels.insert((1, 1), "language".into());
    probe_labels.insert((2, 0), "capital".into());
    probe_labels.insert((2, 2), "continent".into());

    println!("Probe labels loaded: {} confirmed", probe_labels.len());
    println!("{{");
    println!("  \"entity\": \"France\",");
    println!("  \"edges\": [");

    let trace = patched.walk(&query, &[1, 2], 3);
    let mut edge_lines = Vec::new();
    for (layer, hits) in &trace.layers {
        for hit in hits.iter().take(2) {
            let tok = hit.meta.top_token.trim();
            if tok.len() < 2 {
                continue;
            }
            if let Some(label) = probe_labels.get(&(*layer, hit.feature)) {
                edge_lines.push(format!(
                    "    {{\"relation\": \"{}\", \"target\": \"{}\", \"gate_score\": {:.1}, \"layer\": {}, \"source\": \"probe\"}}",
                    label, tok, hit.gate_score, layer
                ));
            } else {
                edge_lines.push(format!(
                    "    {{\"target\": \"{}\", \"gate_score\": {:.1}, \"layer\": {}}}",
                    tok, hit.gate_score, layer
                ));
            }
        }
    }
    for (idx, line) in edge_lines.iter().enumerate() {
        let comma = if idx + 1 < edge_lines.len() { "," } else { "" };
        println!("{line}{comma}");
    }
    println!("  ]");
    println!("}}");

    // ── 8. AUTH (--api-key) ──
    section("Authentication");
    println!("With --api-key \"sk-abc123\":");
    println!("  curl -H \"Authorization: Bearer sk-abc123\" http://localhost:8080/v1/describe?entity=France");
    println!("  → 200 OK (valid token)");
    println!();
    println!("  curl http://localhost:8080/v1/describe?entity=France");
    println!("  → 401 Unauthorized (no token)");
    println!();
    println!("  curl http://localhost:8080/v1/health");
    println!("  → 200 OK (health exempt from auth)");

    // ── 9. SESSION ISOLATION ──
    section("Per-session patch isolation");

    let (index_a, _) = demo_index();
    let (index_b, _) = demo_index();
    let mut session_a = PatchedVindex::new(index_a);
    let session_b = PatchedVindex::new(index_b);

    session_a.delete_feature(1, 0); // Session A removes Paris

    println!("Session A (removed feature L1:F0):");
    println!(
        "  L1:F0 = {:?}",
        session_a.feature_meta(1, 0).map(|m| m.top_token.clone())
    );
    println!("Session B (untouched):");
    println!(
        "  L1:F0 = {:?}",
        session_b.feature_meta(1, 0).map(|m| m.top_token.clone())
    );
    println!("\nSessions are isolated — patches don't leak between clients.");

    // ── 10. DESCRIBE CACHE ──
    section("DESCRIBE cache (--cache-ttl)");

    println!("With --cache-ttl 300:");
    println!("  1st request: DESCRIBE France → 12ms (computed)");
    println!("  2nd request: DESCRIBE France → <1ms (cached)");
    println!("  After 5 min: DESCRIBE France → 12ms (expired, recomputed)");
    println!("\nCache key: model:entity:band:limit:min_score");

    // ── 11. RATE LIMITING ──
    section("Rate limiting (--rate-limit)");

    println!("With --rate-limit \"100/min\":");
    println!("  Per-IP token bucket — 100 requests/min burst, 1.67/sec refill");
    println!("  /v1/health is exempt from rate limiting");
    println!("  X-Forwarded-For is trusted only with --trust-forwarded-for");
    println!("  Excess requests → 429 Too Many Requests");

    // ── 12. BAND FILTERING ──
    section("DESCRIBE band filtering");

    let trace_syntax = patched.walk(&query, &[0], 3);
    let trace_knowledge = patched.walk(&query, &[1, 2], 3);
    let trace_output = patched.walk(&query, &[3], 3);

    println!("band=syntax (L0):");
    for (_, hits) in &trace_syntax.layers {
        for hit in hits.iter().take(2) {
            println!("  {} (gate={:.1})", hit.meta.top_token, hit.gate_score);
        }
    }
    println!("band=knowledge (L1-2):");
    for (_, hits) in &trace_knowledge.layers {
        for hit in hits.iter().take(2) {
            println!("  {} (gate={:.1})", hit.meta.top_token, hit.gate_score);
        }
    }
    println!("band=output (L3):");
    for (_, hits) in &trace_output.layers {
        for hit in hits.iter().take(2) {
            println!("  {} (gate={:.1})", hit.meta.top_token, hit.gate_score);
        }
    }

    // ── 13. WALK-FFN (decoupled inference) ──
    section("POST /v1/walk-ffn (decoupled inference)");

    let residual =
        larql_vindex::ndarray::Array1::from_vec(vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(1, &residual, 5);
    let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
    let scores: Vec<f32> = hits
        .iter()
        .map(|(_, s)| (s * 100.0).round() / 100.0)
        .collect();

    println!("Single layer request:");
    println!("  POST /v1/walk-ffn {{\"layer\": 1, \"residual\": [1.0, 0.2, ...]}}");
    println!(
        "  → {{\"layer\": 1, \"features\": {:?}, \"scores\": {:?}}}",
        features, scores
    );
    println!();
    println!("Batched request (all layers in one round-trip):");
    println!("  POST /v1/walk-ffn {{\"layers\": [0,1,2,3], \"residual\": [...]}}");
    println!("  → {{\"results\": [{{\"layer\": 0, ...}}, {{\"layer\": 1, ...}}, ...]}}");

    // ── 14. WEBSOCKET STREAMING ──
    section("WS /v1/stream (WebSocket)");

    println!("Protocol:");
    println!("  → {{\"type\": \"describe\", \"entity\": \"France\", \"band\": \"all\"}}");
    println!("  ← {{\"type\": \"layer\", \"layer\": 0, \"edges\": [...]}}");
    println!("  ← {{\"type\": \"layer\", \"layer\": 1, \"edges\": [...]}}");
    println!("  ← {{\"type\": \"layer\", \"layer\": 2, \"edges\": [...]}}");
    println!("  ← {{\"type\": \"layer\", \"layer\": 3, \"edges\": [...]}}");
    println!("  ← {{\"type\": \"done\", \"entity\": \"France\", \"total_edges\": N, \"latency_ms\": 12.3}}");

    // ── 15. ETAG / CDN CACHING ──
    section("ETag + Cache-Control headers");

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let body = serde_json::json!({"entity": "France", "edges": [{"target": "Paris"}]});
    let mut hasher = DefaultHasher::new();
    body.to_string().hash(&mut hasher);
    let etag = format!("\"{:x}\"", hasher.finish());

    println!("Response headers:");
    println!("  ETag: {etag}");
    println!("  Cache-Control: public, max-age=86400");
    println!();
    println!("Client sends If-None-Match: {etag}");
    println!("  → 304 Not Modified (no body, saves bandwidth)");

    // ── 16. HEALTH ──
    section("GET /v1/health");
    println!("{{\"status\": \"ok\", \"uptime_seconds\": 0, \"requests_served\": 16}}");

    println!("\n── Demo complete ({} features shown) ──", 16);
    println!("To run the real server:");
    println!("  larql serve <path-to-vindex> --port 8080");
    println!("  With auth:       --api-key \"sk-abc123\"");
    println!("  With TLS:        --tls-cert cert.pem --tls-key key.pem");
    println!("  With rate limit: --rate-limit \"100/min\"");
    println!("  With cache:      --cache-ttl 300");
}
