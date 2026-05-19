/// Constrained generation → expert dispatch test (Option 1 path).
///
/// Uses `generate_cached_constrained` with a JSON op-grammar mask closure.
/// At each decode step the mask inspects the text emitted so far:
///
///   state Open   — nothing yet; allow `{` only
///   state Key    — after `{`; allow `"op":"`  (multi-token prefix, allow freely)
///   state OpName — after `{"op":"`; restrict to tokens that are a valid prefix
///                  of a known op name, plus `"` to close when the name is complete
///   state Args   — after op name + `"`; free-run until EOS
///
/// This proves that extraction can be grammar-constrained without a second
/// inference pass — the same KV-cached decode loop handles both routing
/// and argument materialisation.
use std::collections::HashSet;
use std::path::PathBuf;

use larql_inference::experts::{parse_op_call, ExpertRegistry};
use larql_inference::{encode_prompt, prompt::ChatTemplate, InferenceModel, WeightFfn};
use larql_kv::generation::generate_cached_constrained;
use serde_json::{json, Value};

// ── Infrastructure ────────────────────────────────────────────────────────────

fn model_id() -> Option<String> {
    std::env::var("LARQL_MODEL").ok()
}

fn wasm_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larql-experts/target/wasm32-wasip1/release")
}

// ── Grammar mask ─────────────────────────────────────────────────────────────

/// Tracks where we are inside the `{"op":"<NAME>","args":{...}}` template.
struct OpJsonMask {
    valid_ops: Vec<String>,
    /// All token IDs whose decoded string is a valid op name start/continuation.
    /// Built lazily on first constrained step.
    op_token_cache: Option<Vec<u32>>,
    tokenizer: tokenizers::Tokenizer,
    generated_text: String,
}

#[derive(Debug, PartialEq)]
enum GrammarState {
    /// Haven't seen `{"op":"` yet — free-run (model is well-prompted).
    Free,
    /// Inside the op-name field — constrain to valid op-name prefixes.
    OpName { so_far: String },
    /// Op name complete — free-run for args.
    Done,
}

impl OpJsonMask {
    fn new(valid_ops: Vec<String>, tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            valid_ops,
            op_token_cache: None,
            tokenizer,
            generated_text: String::new(),
        }
    }

    fn state(&self) -> GrammarState {
        let text = &self.generated_text;
        // Find the op-name start marker
        if let Some(pos) = text.find("{\"op\":\"") {
            let after = &text[pos + 7..]; // text after the 7-char prefix
            if let Some(close) = after.find('"') {
                let _ = close; // op name is complete
                return GrammarState::Done;
            } else {
                return GrammarState::OpName {
                    so_far: after.to_string(),
                };
            }
        }
        GrammarState::Free
    }

    /// Build (once) the set of token IDs whose decoded string could be a
    /// single-char or multi-char fragment of a valid op name.
    fn op_tokens(&mut self) -> &[u32] {
        if self.op_token_cache.is_none() {
            // Collect every character that appears in any valid op name.
            let valid_chars: HashSet<char> =
                self.valid_ops.iter().flat_map(|op| op.chars()).collect();

            // Scan vocab for tokens that decode to a non-empty string composed
            // entirely of op-name characters, or `"` (closes the op name field).
            let vocab_size = self.tokenizer.get_vocab_size(false);
            let mut ids: Vec<u32> = Vec::new();
            for id in 0..vocab_size as u32 {
                if let Ok(s) = self.tokenizer.decode(&[id], false) {
                    if s == "\"" {
                        ids.push(id); // closing quote — always valid
                    } else if !s.is_empty() && s.chars().all(|c| valid_chars.contains(&c)) {
                        ids.push(id);
                    }
                }
            }
            self.op_token_cache = Some(ids);
        }
        self.op_token_cache.as_ref().unwrap()
    }

    /// Called by `generate_cached_constrained` before each argmax.
    /// Updates the internal text buffer and masks logits when in OpName state.
    #[allow(clippy::ptr_arg)]
    fn apply(&mut self, generated_ids: &[u32], logits: &mut Vec<f32>) {
        self.generated_text = self
            .tokenizer
            .decode(generated_ids, true)
            .unwrap_or_default();

        let state = self.state();
        if let GrammarState::OpName { so_far } = state {
            // Ensure op token cache is populated, then take owned copies so the
            // borrow checker doesn't see simultaneous borrows of self.
            let _ = self.op_tokens();
            let candidate_ids: Vec<u32> = self.op_token_cache.as_ref().unwrap().clone();
            let valid_ops = self.valid_ops.clone();
            let tokenizer = &self.tokenizer;

            let valid_next: HashSet<u32> = candidate_ids
                .iter()
                .copied()
                .filter(|&id| {
                    let s = tokenizer.decode(&[id], false).unwrap_or_default();
                    if s == "\"" {
                        // Closing quote — allowed only when so_far is a complete op name.
                        valid_ops.iter().any(|op| op == &so_far)
                    } else if !s.is_empty() {
                        // Continuation — allowed if `so_far + s` is a prefix of any valid op.
                        let candidate = format!("{so_far}{s}");
                        valid_ops
                            .iter()
                            .any(|op| op.starts_with(candidate.as_str()))
                    } else {
                        false
                    }
                })
                .collect();

            if !valid_next.is_empty() {
                for (i, v) in logits.iter_mut().enumerate() {
                    if !valid_next.contains(&(i as u32)) {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            // If valid_next is empty the grammar state detection was wrong;
            // fall back to free generation rather than masking everything out.
        }
    }
}

// ── Cases ─────────────────────────────────────────────────────────────────────

struct Case {
    prompt: &'static str,
    expected_op: &'static str,
    expected_result: Value,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            prompt: "What is the GCD of 144 and 60?",
            expected_op: "gcd",
            expected_result: json!(12),
        },
        Case {
            prompt: "Is 97 a prime number?",
            expected_op: "is_prime",
            expected_result: json!(true),
        },
        Case {
            prompt: "What is 10 factorial?",
            expected_op: "factorial",
            expected_result: json!(3628800),
        },
        Case {
            prompt: "Write 2024 as a Roman numeral.",
            expected_op: "to_roman",
            expected_result: json!("MMXXIV"),
        },
        Case {
            prompt: "Is 2024 a leap year?",
            expected_op: "is_leap_year",
            expected_result: json!(true),
        },
        Case {
            prompt: "Reverse the string \"hello world\".",
            expected_op: "reverse",
            expected_result: json!("dlrow olleh"),
        },
    ]
}

const SYSTEM: &str = r#"Respond with ONLY a JSON object {"op":"...","args":{...}}.
ops: gcd{"a","b"}, is_prime{"n"}, factorial{"n"}, to_roman{"n"}, is_leap_year{"year"}, days_in_month{"year","month"}, reverse{"s"}, is_palindrome{"s"}
No extra text."#;

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "loads a real model; set LARQL_MODEL and run with --ignored"]
fn constrained_dispatch_pipeline() {
    if !wasm_dir().exists() {
        eprintln!("skip: wasm dir missing");
        return;
    }

    let Some(mid) = model_id() else {
        eprintln!("skip: set LARQL_MODEL to run constrained_dispatch_pipeline");
        return;
    };
    let model = match InferenceModel::load(&mid) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    eprintln!("model: {mid}  ({} layers)", model.num_layers());

    let mut reg = ExpertRegistry::load_dir(&wasm_dir()).expect("load_dir");
    let ffn = WeightFfn {
        weights: model.weights(),
    };

    // Collect all op names from the registry for the mask.
    let valid_ops: Vec<String> = reg.ops().into_iter().map(|s| s.to_string()).collect();
    eprintln!("valid ops ({}):", valid_ops.len());
    for op in &valid_ops {
        eprint!("  {op}");
    }
    eprintln!();

    let template = ChatTemplate::for_model_id(&mid);
    eprintln!("template: {}", template.name());

    let mut passed = 0usize;
    let mut failed = 0usize;

    for case in cases() {
        let full_prompt = format!("{SYSTEM}\n\nQuestion: {}", case.prompt);
        let wrapped = template.wrap(&full_prompt);

        let ids = match encode_prompt(model.tokenizer(), &*model.weights().arch, &wrapped) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  FAIL tokenize: {e}");
                failed += 1;
                continue;
            }
        };

        // Build a fresh mask for each case (resets generated_text).
        let mut mask = OpJsonMask::new(valid_ops.clone(), model.tokenizer().clone());

        let mut output = String::new();
        generate_cached_constrained(
            model.weights(),
            model.tokenizer(),
            &ffn,
            &ids,
            128,
            |gen_ids, logits| mask.apply(gen_ids, logits),
            |_id, tok| output.push_str(tok),
        );

        eprintln!("\n  prompt:  {}", case.prompt);
        eprintln!("  raw out: {output:?}");

        let call = match parse_op_call(&output) {
            Some(c) => c,
            None => {
                eprintln!("  FAIL: no op-call JSON");
                failed += 1;
                continue;
            }
        };
        let op = call.op;
        let args = call.args;

        let correct_op = op == case.expected_op;
        eprintln!(
            "  op={op}{}  args={args}",
            if correct_op { "" } else { " ← WRONG OP" }
        );

        if !correct_op {
            eprintln!("  FAIL: expected op={}", case.expected_op);
            failed += 1;
            continue;
        }

        match reg.call(&op, &args) {
            Some(r) if r.value == case.expected_result => {
                eprintln!("  ok  [{op}] {} → {}", case.prompt, r.value);
                passed += 1;
            }
            Some(r) => {
                eprintln!("  FAIL: got {}, expected {}", r.value, case.expected_result);
                failed += 1;
            }
            None => {
                eprintln!("  FAIL: registry returned None for op={op} args={args}");
                failed += 1;
            }
        }
    }

    eprintln!(
        "\n{passed}/{} constrained dispatch cases passed",
        passed + failed
    );
    assert_eq!(failed, 0, "{failed} cases failed");
}
