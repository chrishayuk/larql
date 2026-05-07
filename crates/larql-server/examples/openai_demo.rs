//! Live OpenAI-compat demo — boots an in-process larql server and
//! exercises `/v1/models`, `/v1/embeddings`, `/v1/completions`,
//! `/v1/chat/completions` end-to-end against the loaded vindex.
//!
//! Usage:
//!   cargo run -p larql-server --example openai_demo -- <vindex_path>
//!
//! ## Vindex compatibility
//!
//! Both **f16** and **Q4_K** vindexes produce correct, intelligible
//! output now that the KV-cached generation path is wired up
//! (e.g. "The capital of France is" → " Paris.").
//!
//! ```bash
//! # f16 (fastest, KV-cached):
//! cargo run --release -p larql-server --example openai_demo -- \
//!   output/gemma3-4b-f16.vindex
//!
//! # Q4_K (correct output; CPU per-step Q4_K decode is O(N²) so
//! # high `max_tokens` runs are slow):
//! cargo run --release -p larql-server --example openai_demo -- \
//!   output/gemma3-4b-q4k-streaming.vindex
//! ```
//!
//! Pattern mirrors `bench_embed_server` / `bench_expert_server`: build
//! the router via `tower::ServiceExt::oneshot`, no port binding, no
//! external HTTP client. The wire shapes are real — captured from the
//! same router that the production binary uses.

use std::path::PathBuf;
use std::sync::{atomic::AtomicU64, Arc};
use std::time::Instant;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use serde_json::Value;
use tower::ServiceExt;

use larql_server::{
    bootstrap::{load_single_vindex, LoadVindexOptions},
    cache::DescribeCache,
    routes::single_model_router,
    session::SessionManager,
    state::{AppState, LoadedModel},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn section(title: &str) {
    println!("\n══ {title} ══");
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<serialize error>".into())
}

/// Trim large arrays (embeddings) to the first N + "...total: K" so the
/// printed JSON stays readable. Recursive.
fn trim_arrays_for_print(v: &Value, head: usize) -> Value {
    match v {
        Value::Array(a) if a.len() > head + 2 => {
            let mut head_vals: Vec<Value> = a.iter().take(head).cloned().collect();
            head_vals.push(Value::String(format!(
                "...{} more elements (total: {})",
                a.len() - head,
                a.len()
            )));
            Value::Array(head_vals)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| trim_arrays_for_print(x, head)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| (k.clone(), trim_arrays_for_print(x, head)))
                .collect(),
        ),
        other => other.clone(),
    }
}

async fn get_json(app: &Router, path: &str) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .expect("oneshot get");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn post_json(app: &Router, path: &str, body: &Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("oneshot post");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

// ── Server boot ───────────────────────────────────────────────────────────────

fn make_app_state(model: LoadedModel) -> Arc<AppState> {
    Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
    })
}

fn load_default(path: &str) -> Result<LoadedModel, Box<dyn std::error::Error + Send + Sync>> {
    let opts = LoadVindexOptions {
        no_infer: false,
        ffn_only: false,
        embed_only: false,
        layer_range: None,
        max_gate_cache_layers: 0,
        max_q4k_cache_layers: 0,
        hnsw: None,
        warmup_hnsw: false,
        release_mmap_after_request: false,
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
    };
    load_single_vindex(path, opts)
}

// ── Demos ─────────────────────────────────────────────────────────────────────

async fn demo_models(app: &Router) {
    section("GET /v1/models");
    let t = Instant::now();
    let (status, body) = get_json(app, "/v1/models").await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: `id`, `object`, `created`, `owned_by` are the OpenAI required\n\
         fields. `path`, `features`, `loaded` are larql-specific extras —\n\
         OpenAI SDKs ignore unknown fields."
    );
}

async fn demo_embeddings(app: &Router, model_id: &str) {
    section("POST /v1/embeddings — single string");
    let req = serde_json::json!({"model": model_id, "input": "France"});
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&trim_arrays_for_print(&body, 4)));
    let dim = body
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|e| e.get("embedding"))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if dim > 0 {
        println!("\n→ {dim}-dim mean-pooled lookup vector");
    }

    section("POST /v1/embeddings — string array");
    let req = serde_json::json!({"model": model_id, "input": ["France", "Germany", "Japan"]});
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&trim_arrays_for_print(&body, 3)));

    section("POST /v1/embeddings — base64 encoding");
    let req = serde_json::json!({
        "model": model_id,
        "input": "France",
        "encoding_format": "base64",
    });
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/embeddings", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    // Don't pretty-print the full base64 string — just show the head
    // and length so the demo output stays scannable.
    if let Some(arr) = body.get("data").and_then(|d| d.as_array()) {
        if let Some(s) = arr
            .first()
            .and_then(|e| e.get("embedding"))
            .and_then(|v| v.as_str())
        {
            println!(
                "data[0].embedding: \"{}…\" (length {} chars, ~{} f32s)",
                &s[..s.len().min(48)],
                s.len(),
                // base64 → 4 bytes per 3 chars; 4 bytes per f32.
                s.len() * 3 / 16,
            );
        }
    }
    println!(
        "\nNote: same vector as the float form, encoded as little-endian\n\
         f32 bytes, base64-stringified. ~33% smaller wire than the JSON\n\
         array. Many production OpenAI clients default to base64."
    );
}

async fn demo_completions(app: &Router, model_id: &str) {
    section("POST /v1/completions — non-streaming");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.0
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: generation runs through the KV-cached path\n\
         (`larql_inference::layer_graph::generate_with_sampling`) on\n\
         f16 vindexes, with a per-step Q4_K fallback on CPU+Q4_K\n\
         vindexes. Output text quality depends on the base model."
    );

    section("POST /v1/completions — temperature + top_p + seed (reproducible)");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "Once upon a time",
        "max_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 42
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: seed=42 + temperature>0 makes output reproducible —\n\
         re-running with the same prompt and seed yields the same\n\
         tokens. Drop the seed to get a fresh sample each call."
    );

    section("POST /v1/completions — n=3 (returns 400)");
    let req = serde_json::json!({
        "model": model_id,
        "prompt": "x",
        "max_tokens": 1,
        "n": 3
    });
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/completions", &req).await;
    println!("Status: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
}

async fn demo_chat_completions(app: &Router, model_id: &str) {
    section("POST /v1/chat/completions — non-streaming (slice 2)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user",   "content": "What is the capital of France?"}
        ],
        "max_tokens": 8,
        "temperature": 0.0
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: messages render through the model's chat template\n\
         (Gemma / Llama / ChatML / Mistral / Plain) before going into\n\
         the KV-cached generation loop. Sampling fields\n\
         (temperature, top_p, seed, stop) plumb through the same way\n\
         /v1/completions wires them."
    );

    section("POST /v1/chat/completions — response_format: json_object");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Respond in JSON."},
            {"role": "user",   "content": "Give me a tiny user profile."}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 32
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: any structurally-valid JSON object. The constrained\n\
         decoder masks every token whose surface chars would break\n\
         JSON, and EOS is masked while the object is still open."
    );

    section("POST /v1/chat/completions — response_format: json_schema (strict)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Output JSON only."},
            {"role": "user",   "content": "Describe Alice, age 30, who is admin."}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age":  {"type": "integer"},
                        "role": {"type": "string", "enum": ["user", "admin", "guest"]}
                    },
                    "required": ["name", "age", "role"]
                }
            }
        },
        "max_tokens": 64
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: strict mode flips additionalProperties=false by default,\n\
         so unknown keys are rejected. `enum` becomes a oneOf-of-Const\n\
         branches in the FSM and commits as soon as the literal string\n\
         disambiguates."
    );

    section("POST /v1/chat/completions — tools (function calling)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "user", "content": "What is the weather in London?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units":    {"type": "string", "enum": ["C", "F"]}
                    },
                    "required": ["location"]
                }
            }
        }],
        "max_tokens": 64
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: each tool synthesises a `{{name=Const, arguments=<args>}}`\n\
         schema branch; multiple tools become a discriminated OneOf.\n\
         Output is parsed back into `message.tool_calls[]` with\n\
         `finish_reason: \"tool_calls\"`. Tools + stream=true is wired\n\
         too — buffered constrained gen, single delta chunk for the\n\
         tool_calls payload, then a final finish-reason chunk."
    );

    section("POST /v1/chat/completions — tool-result replay (multi-turn)");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Weather in London?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather", "arguments": "{\"location\":\"London\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "23 C, sunny"}
        ],
        "max_tokens": 32
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));

    section("POST /v1/chat/completions — logprobs + repetition penalties");
    let req = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "Once upon a time"}],
        "max_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 42,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "logprobs": true,
        "top_logprobs": 3
    });
    println!("Request body:\n{}", pretty(&req));
    let t = Instant::now();
    let (status, body) = post_json(app, "/v1/chat/completions", &req).await;
    println!("\nStatus: {status}  ({} ms)", t.elapsed().as_millis());
    println!("{}", pretty(&body));
    println!(
        "\nNote: temperature/top_p/seed are honoured by the sampler;\n\
         frequency/presence penalties subtract from logits before softmax\n\
         (clamped to [-2, 2]); logprobs:true populates choices[i].logprobs\n\
         with one entry per emitted token."
    );
}

fn print_client_snippets(model_id: &str) {
    section("Equivalent client code");
    println!(
        "Python (openai SDK):\n\
         \n\
             from openai import OpenAI\n\
             client = OpenAI(\n\
                 base_url=\"http://localhost:8080/v1\",\n\
                 api_key=\"sk-anything\",  # required by SDK; matched against --api-key if set\n\
             )\n\
             # /v1/models\n\
             models = client.models.list()\n\
             # /v1/embeddings\n\
             emb = client.embeddings.create(\n\
                 model=\"{model_id}\",\n\
                 input=\"France\",\n\
             )\n\
             # /v1/completions\n\
             resp = client.completions.create(\n\
                 model=\"{model_id}\",\n\
                 prompt=\"The capital of France is\",\n\
                 max_tokens=10,\n\
             )\n\
             # /v1/chat/completions\n\
             chat = client.chat.completions.create(\n\
                 model=\"{model_id}\",\n\
                 messages=[\n\
                     {{\"role\": \"system\", \"content\": \"You are concise.\"}},\n\
                     {{\"role\": \"user\",   \"content\": \"Capital of France?\"}},\n\
                 ],\n\
                 max_tokens=10,\n\
             )\n\
             print(chat.choices[0].message.content)\n\
             \n\
             # base64 embeddings\n\
             emb_b64 = client.embeddings.create(\n\
                 model=\"{model_id}\",\n\
                 input=\"France\",\n\
                 encoding_format=\"base64\",\n\
             )\n\
             \n\
             # Structured outputs — strict JSON Schema\n\
             schema = {{\n\
                 \"name\": \"Person\",\n\
                 \"strict\": True,\n\
                 \"schema\": {{\n\
                     \"type\": \"object\",\n\
                     \"properties\": {{\n\
                         \"name\": {{\"type\": \"string\"}},\n\
                         \"age\":  {{\"type\": \"integer\"}}\n\
                     }},\n\
                     \"required\": [\"name\", \"age\"]\n\
                 }}\n\
             }}\n\
             person = client.chat.completions.create(\n\
                 model=\"{model_id}\",\n\
                 messages=[{{\"role\": \"user\", \"content\": \"Describe Bob, 42.\"}}],\n\
                 response_format={{\"type\": \"json_schema\", \"json_schema\": schema}},\n\
             )\n\
             import json; data = json.loads(person.choices[0].message.content)\n\
             \n\
             # Function calling\n\
             tools = [{{\n\
                 \"type\": \"function\",\n\
                 \"function\": {{\n\
                     \"name\": \"get_weather\",\n\
                     \"parameters\": {{\n\
                         \"type\": \"object\",\n\
                         \"properties\": {{\"location\": {{\"type\": \"string\"}}}},\n\
                         \"required\": [\"location\"]\n\
                     }}\n\
                 }}\n\
             }}]\n\
             call = client.chat.completions.create(\n\
                 model=\"{model_id}\",\n\
                 messages=[{{\"role\": \"user\", \"content\": \"Weather in Paris?\"}}],\n\
                 tools=tools,\n\
             )\n\
             # call.choices[0].message.tool_calls[0].function.{{name,arguments}}\n\
             # Multi-turn: append the tool_call message and a {{role:tool, tool_call_id, content}}\n\
             # message, then call again to let the model formulate the answer.\n\
             \n\
             # Sampling + repetition penalties + logprobs\n\
             chat = client.chat.completions.create(\n\
                 model=\"{model_id}\",\n\
                 messages=[{{\"role\": \"user\", \"content\": \"Once upon a time\"}}],\n\
                 max_tokens=20,\n\
                 temperature=0.8,\n\
                 top_p=0.9,\n\
                 seed=42,\n\
                 frequency_penalty=0.5,\n\
                 presence_penalty=0.3,\n\
                 logprobs=True,\n\
                 top_logprobs=3,\n\
             )\n\
         \n\
         curl:\n\
         \n\
             curl http://localhost:8080/v1/models\n\
             curl -X POST http://localhost:8080/v1/embeddings \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"input\": \"France\"}}'\n\
             curl -X POST http://localhost:8080/v1/completions \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 5}}'\n\
             curl -X POST http://localhost:8080/v1/chat/completions \\\n\
                  -H 'Content-Type: application/json' \\\n\
                  -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Capital of France?\"}}], \"max_tokens\": 5}}'"
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,larql_server=info")),
        )
        .with_target(false)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: openai_demo <vindex_path>\n\n\
             Boots an in-process larql server (no port binding, no external\n\
             HTTP client) and exercises the OpenAI-compat endpoints end-to-\n\
             end against the loaded vindex.\n\n\
             Examples:\n\
               cargo run --release -p larql-server --example openai_demo -- \\\n\
                 output/gemma3-4b-q4k-streaming.vindex"
        );
        std::process::exit(1);
    }
    let vindex_path = PathBuf::from(&args[1]);

    println!("── larql-server OpenAI-compat live demo ──");
    println!("Vindex: {}", vindex_path.display());

    let t = Instant::now();
    let model = load_default(&args[1])?;
    let model_id = model.id.clone();
    let hidden = model.config.hidden_size;
    let num_layers = model.config.num_layers;
    println!(
        "Loaded {} ({} layers, hidden={}) in {} ms",
        model_id,
        num_layers,
        hidden,
        t.elapsed().as_millis(),
    );

    let state = make_app_state(model);
    let app = single_model_router(state);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        demo_models(&app).await;
        demo_embeddings(&app, &model_id).await;
        demo_completions(&app, &model_id).await;
        demo_chat_completions(&app, &model_id).await;
    });

    print_client_snippets(&model_id);

    section("Done");
    println!(
        "Boot a public server with the same vindex and the same endpoints\n\
         become reachable from any OpenAI SDK:\n\
         \n\
           larql-server {} --port 8080\n\
         \n\
         Then point `base_url=\"http://localhost:8080/v1\"` and your\n\
         existing OpenAI Python or JS client works unmodified.",
        vindex_path.display()
    );
    Ok(())
}
