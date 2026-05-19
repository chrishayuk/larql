//! Ollama side-by-side bench. Shells out to `curl` against the local Ollama
//! REST endpoint; the orchestration and JSON parsing are pure helpers behind
//! a `Fetcher` indirection so they can be unit-tested without a live server.

use super::row::BenchRow;

const OLLAMA_URL: &str = "http://localhost:11434/api/generate";

/// Query a local Ollama server for a one-shot generate at `n` tokens.
/// Reports tok/s based on Ollama's own `eval_duration` / `eval_count`
/// (GPU wall time on its end, excludes HTTP overhead).
pub(super) fn run_ollama(model: &str, prompt: &str, num_predict: usize) -> BenchRow {
    run_ollama_with(model, prompt, num_predict, curl_fetch)
}

/// Test-friendly variant of `run_ollama` that takes an injectable fetcher.
/// In production the fetcher shells out to `curl`; tests pass a closure that
/// returns canned strings to exercise every branch without touching the
/// network.
pub(super) fn run_ollama_with<F>(
    model: &str,
    prompt: &str,
    num_predict: usize,
    fetch: F,
) -> BenchRow
where
    F: Fn(&str) -> Option<String>,
{
    // Warmup call — discarded. Lets Ollama hot-load the model without that
    // first-token latency leaking into the measurement.
    let _ = fetch(&build_ollama_body(model, "Hi", 5));

    let body = build_ollama_body(model, prompt, num_predict);
    match fetch(&body) {
        Some(text) => parse_ollama_response(model, &text),
        None => unreachable_row(model),
    }
}

/// Production fetcher: shells out to `curl -s -d <body>` against the local
/// Ollama HTTP endpoint and returns stdout. `None` if the process couldn't
/// be spawned.
fn curl_fetch(body: &str) -> Option<String> {
    let out = std::process::Command::new("curl")
        .args(["-s", OLLAMA_URL, "-d", body])
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build the `curl -d` JSON body. Pure; safe to test without a running
/// Ollama server. Escapes `"` in the prompt so embedded quotes don't break
/// the inline JSON.
pub(super) fn build_ollama_body(model: &str, prompt: &str, num_predict: usize) -> String {
    format!(
        r#"{{"model":"{model}","prompt":"{}","stream":false,"options":{{"num_predict":{num_predict}}}}}"#,
        prompt.replace('"', "\\\""),
    )
}

/// Parse Ollama's `/api/generate` non-streaming JSON response into a
/// `BenchRow`. On malformed JSON or missing eval fields, returns the
/// "not reachable" row so the table still renders sensibly.
pub(super) fn parse_ollama_response(model: &str, text: &str) -> BenchRow {
    let val: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return unreachable_row(model),
    };

    let eval_count = val["eval_count"].as_f64().unwrap_or(0.0);
    let eval_dur_ns = val["eval_duration"].as_f64().unwrap_or(0.0);
    let prompt_dur_ns = val["prompt_eval_duration"].as_f64().unwrap_or(0.0);

    let mut row = unreachable_row(model);
    if eval_count > 0.0 && eval_dur_ns > 0.0 {
        let avg_ms = eval_dur_ns / 1e6 / eval_count;
        row.avg_decode_ms = avg_ms;
        row.tok_per_s = 1000.0 / avg_ms;
        row.prefill_ms = prompt_dur_ns / 1e6;
        row.n_steps = eval_count as usize;
        row.note = String::new();
    }
    row
}

/// The default "Ollama unreachable" placeholder row. Made standalone so we
/// can return it from any failure path without duplicating field
/// initialisation.
fn unreachable_row(model: &str) -> BenchRow {
    BenchRow {
        backend: format!("ollama {model}"),
        prefill_ms: 0.0,
        avg_decode_ms: 0.0,
        p50_ms: 0.0,
        p99_ms: 0.0,
        tok_per_s: 0.0,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: 0,
        note: "not reachable (ollama serve on :11434?)".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn build_body_escapes_embedded_quotes() {
        let body = build_ollama_body("gemma3:4b", r#"say "hi""#, 10);
        assert!(body.contains(r#"\"hi\""#), "body: {body}");
        assert!(body.contains(r#""model":"gemma3:4b""#));
        assert!(body.contains(r#""num_predict":10"#));
    }

    #[test]
    fn build_body_preserves_unicode_prompts() {
        let body = build_ollama_body("m", "Привіт", 1);
        assert!(body.contains("Привіт"));
    }

    #[test]
    fn parse_response_returns_unreachable_on_invalid_json() {
        let row = parse_ollama_response("gemma3:4b", "not json");
        assert_eq!(row.backend, "ollama gemma3:4b");
        assert_eq!(row.tok_per_s, 0.0);
        assert!(row.note.contains("not reachable"));
    }

    #[test]
    fn parse_response_returns_unreachable_when_eval_fields_missing() {
        let row = parse_ollama_response("gemma3:4b", r#"{"hello":"world"}"#);
        assert_eq!(row.tok_per_s, 0.0);
        assert!(row.note.contains("not reachable"));
    }

    #[test]
    fn parse_response_returns_unreachable_when_eval_count_zero() {
        let row = parse_ollama_response(
            "m",
            r#"{"eval_count":0,"eval_duration":1000000,"prompt_eval_duration":500000}"#,
        );
        assert_eq!(row.tok_per_s, 0.0);
        assert!(!row.note.is_empty());
    }

    #[test]
    fn parse_response_computes_tok_per_s_from_eval_fields() {
        let body = r#"{
            "eval_count": 50,
            "eval_duration": 5000000000,
            "prompt_eval_duration": 250000000
        }"#;
        let row = parse_ollama_response("gemma3:4b", body);
        assert!((row.tok_per_s - 10.0).abs() < 1e-6);
        assert!((row.avg_decode_ms - 100.0).abs() < 1e-6);
        assert!((row.prefill_ms - 250.0).abs() < 1e-6);
        assert_eq!(row.n_steps, 50);
        assert!(row.note.is_empty(), "successful parse clears the note");
    }

    #[test]
    fn run_ollama_with_fetcher_routes_response_through_parser() {
        // Captures both calls — the warmup call (first) and the real one
        // (second). The fetcher returns Some(success-JSON) on the real call.
        let calls = RefCell::new(Vec::<String>::new());
        let fetch = |body: &str| -> Option<String> {
            calls.borrow_mut().push(body.to_owned());
            if calls.borrow().len() == 1 {
                // Warmup: response is irrelevant.
                Some("{}".into())
            } else {
                Some(
                    r#"{
                        "eval_count": 25,
                        "eval_duration": 2500000000,
                        "prompt_eval_duration": 0
                    }"#
                    .into(),
                )
            }
        };
        let row = run_ollama_with("m", "p", 25, fetch);
        assert_eq!(row.n_steps, 25);
        assert!((row.tok_per_s - 10.0).abs() < 1e-6);
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 2, "warmup + real call");
        assert!(calls[0].contains(r#""num_predict":5"#));
        assert!(calls[1].contains(r#""num_predict":25"#));
    }

    #[test]
    fn run_ollama_with_fetcher_returns_unreachable_when_fetch_fails() {
        let fetch = |_: &str| -> Option<String> { None };
        let row = run_ollama_with("m", "p", 25, fetch);
        assert_eq!(row.tok_per_s, 0.0);
        assert!(row.note.contains("not reachable"));
    }

    #[test]
    fn run_ollama_with_fetcher_returns_unreachable_on_malformed_json() {
        let fetch = |body: &str| -> Option<String> {
            // Warmup gets "{}"; real call gets garbage.
            if body.contains(r#""num_predict":5"#) {
                Some("{}".into())
            } else {
                Some("not json".into())
            }
        };
        let row = run_ollama_with("m", "p", 10, fetch);
        assert_eq!(row.tok_per_s, 0.0);
        assert!(row.note.contains("not reachable"));
    }
}
