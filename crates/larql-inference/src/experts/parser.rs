//! Parse a structured op-call from free-form model output.
//!
//! Models targeted with the prompt
//!
//! > Respond with ONLY a JSON object {"op":"...","args":{...}}
//!
//! return surprisingly diverse text:
//!
//! - Code fences (```json … ```), reasoning preambles, trailing commentary.
//! - Fullwidth punctuation (`，` `：`) from CJK-tuned models.
//! - Mistral occasionally elides the comma before `"args"`, emitting
//!   `…"value"args":…` — one quote shared between the preceding value and
//!   the `args` key.
//!
//! [`parse_op_call`] handles all of these. It walks the text for balanced
//! `{...}` blocks, normalises punctuation, applies the Mistral patch, and
//! returns the first block that parses as JSON and contains a string `op`
//! field. `args` defaults to an empty object when absent.

use serde_json::{Map, Value};

/// A structured op-call extracted from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct OpCall {
    pub op: String,
    pub args: Value,
}

/// Extract the first valid op-call JSON object from `text`.
///
/// Returns `None` if no balanced `{...}` block parses to an object with a
/// string `op` field. The function never panics on malformed input.
pub fn parse_op_call(text: &str) -> Option<OpCall> {
    let normalised = normalise(text);
    let bytes = normalised.as_bytes();

    let mut search_from = 0;
    while search_from < bytes.len() {
        let rel = normalised[search_from..].find('{')?;
        let start = search_from + rel;
        let end = find_matching_brace(&normalised, start)?;
        if let Ok(v) = serde_json::from_str::<Value>(&normalised[start..end]) {
            if let Some(call) = into_op_call(v) {
                return Some(call);
            }
        }
        search_from = start + 1;
    }
    None
}

fn into_op_call(v: Value) -> Option<OpCall> {
    let mut obj = match v {
        Value::Object(m) => m,
        _ => return None,
    };
    let op = match obj.remove("op")? {
        Value::String(s) if !s.is_empty() => s,
        _ => return None,
    };
    let args = obj
        .remove("args")
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(OpCall { op, args })
}

/// Find the byte offset (one past) of the `}` that closes the `{` at `start`.
/// Tracks string boundaries so braces inside string values don't confuse depth.
fn find_matching_brace(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s[start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start + i + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

/// Apply punctuation + Mistral patches before brace-walking.
fn normalise(text: &str) -> String {
    // Sentinel byte used to protect the already-correct `,"args":` form so the
    // bare-`"args":` patch doesn't double-apply. `\x01` cannot appear in valid
    // UTF-8 model output that would otherwise reach the parser intact.
    const SENTINEL: char = '\x01';
    text.replace('，', ",")
        .replace('：', ":")
        .replace(",\"args\":", &SENTINEL.to_string())
        .replace("\"args\":", "\",\"args\":")
        .replace(SENTINEL, ",\"args\":")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_simple_object() {
        let text = r#"{"op":"gcd","args":{"a":12,"b":8}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "gcd");
        assert_eq!(call.args, json!({"a": 12, "b": 8}));
    }

    #[test]
    fn extracts_after_preamble() {
        let text = r#"Sure! Here is the answer:
        {"op":"is_prime","args":{"n":97}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "is_prime");
    }

    #[test]
    fn extracts_from_code_fence() {
        let text = "```json\n{\"op\":\"factorial\",\"args\":{\"n\":10}}\n```";
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "factorial");
        assert_eq!(call.args, json!({"n": 10}));
    }

    #[test]
    fn skips_blocks_without_op() {
        let text = r#"{"unrelated":1}{"op":"reverse","args":{"s":"hi"}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "reverse");
    }

    #[test]
    fn defaults_args_to_empty_object() {
        let text = r#"{"op":"now"}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "now");
        assert_eq!(call.args, json!({}));
    }

    #[test]
    fn nested_objects_in_args() {
        let text = r#"{"op":"add_days","args":{"date":{"year":2026,"month":3,"day":1},"days":30}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "add_days");
        assert_eq!(call.args["date"]["year"], json!(2026));
    }

    #[test]
    fn brace_inside_string_value_does_not_break_depth() {
        let text = r#"{"op":"echo","args":{"s":"abc{def}ghi"}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.args, json!({"s": "abc{def}ghi"}));
    }

    #[test]
    fn escaped_quote_inside_string_does_not_break_depth() {
        let text = r#"{"op":"echo","args":{"s":"she said \"hi\""}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.args["s"], json!("she said \"hi\""));
    }

    #[test]
    fn fullwidth_punctuation_normalised() {
        let text = "{\"op\"：\"gcd\"，\"args\"：{\"a\"：12，\"b\"：8}}";
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "gcd");
        assert_eq!(call.args, json!({"a": 12, "b": 8}));
    }

    #[test]
    fn mistral_missing_comma_before_args_patched() {
        let text = r#"{"op":"gcd"args":{"a":12,"b":8}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "gcd");
        assert_eq!(call.args, json!({"a": 12, "b": 8}));
    }

    #[test]
    fn already_correct_args_form_not_double_patched() {
        let text = r#"{"op":"gcd","args":{"a":12,"b":8}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.op, "gcd");
        assert_eq!(call.args, json!({"a": 12, "b": 8}));
    }

    #[test]
    fn returns_none_when_no_object_present() {
        assert!(parse_op_call("the answer is 42").is_none());
    }

    #[test]
    fn returns_none_when_op_missing() {
        assert!(parse_op_call(r#"{"foo":"bar"}"#).is_none());
    }

    #[test]
    fn returns_none_when_op_not_string() {
        assert!(parse_op_call(r#"{"op":42,"args":{}}"#).is_none());
    }

    #[test]
    fn returns_none_when_op_empty() {
        assert!(parse_op_call(r#"{"op":"","args":{}}"#).is_none());
    }

    #[test]
    fn unbalanced_braces_returns_none() {
        assert!(parse_op_call(r#"{"op":"gcd","args":{"#).is_none());
    }

    #[test]
    fn unicode_in_args_preserved() {
        let text = r#"{"op":"echo","args":{"s":"日本語"}}"#;
        let call = parse_op_call(text).expect("parses");
        assert_eq!(call.args["s"], json!("日本語"));
    }
}
