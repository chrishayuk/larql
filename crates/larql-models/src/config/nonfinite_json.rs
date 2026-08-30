//! The judged non-finite boundary for `config.json`.
//!
//! Python's `json.dumps` writes non-finite floats as the bare literals
//! `Infinity`, `-Infinity` and `NaN` (`allow_nan` defaults on), and the
//! HF ecosystem ships them — `transformers` wrote
//! `"time_step_limit": [0.0, Infinity]` into `mamba2-780m-hf`'s config,
//! which RFC 8259 has no spelling for and a strict parser rightly
//! refuses (ontology drill, "the first live witness", finding zero).
//!
//! The judgment, made here and nowhere else (the `PositionPolicy`
//! zero-theta discipline): a non-finite literal is preserved as the
//! **string it spells** — `Infinity` becomes `"Infinity"` — never
//! impersonated by a large float, because a fabricated value wearing a
//! number is exactly the class of quiet lie the format refuses. A
//! consumer that judges such a key handles the string deliberately; an
//! unconsumed key carries it to the registry as the declaration it was.
//!
//! The strict parse runs first: a conforming config never pays for the
//! scan, and the rewrite is attempted only after strict refusal.

/// Parse a checkpoint `config.json`, tolerating Python's bare
/// non-finite literals by quoting them (see module doc).
pub fn parse_config_json(text: &str) -> serde_json::Result<serde_json::Value> {
    match serde_json::from_str(text) {
        Ok(value) => Ok(value),
        Err(strict_error) => match quote_nonfinite_literals(text) {
            Some(rewritten) => serde_json::from_str(&rewritten),
            None => Err(strict_error),
        },
    }
}

/// Quote bare `Infinity` / `-Infinity` / `NaN` outside strings.
///
/// Returns `None` when the text contains none (so the caller keeps the
/// strict parser's own error for genuinely malformed JSON).
fn quote_nonfinite_literals(text: &str) -> Option<String> {
    const TOKENS: &[&str] = &["-Infinity", "Infinity", "NaN"];
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0;
    let mut in_string = false;
    let mut changed = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b as char);
            if b == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            continue;
        }
        let matched = TOKENS.iter().find(|token| {
            let end = i + token.len();
            bytes[i..].starts_with(token.as_bytes())
                && bytes
                    .get(end)
                    .is_none_or(|next| !next.is_ascii_alphanumeric() && *next != b'_')
        });
        if let Some(token) = matched {
            out.push('"');
            out.push_str(token);
            out.push('"');
            i += token.len();
            changed = true;
            continue;
        }
        out.push(b as char);
        i += 1;
    }
    changed.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::parse_config_json;

    #[test]
    fn strict_json_passes_through_untouched() {
        let value = parse_config_json(r#"{"a": 1.5, "b": "Infinity in a string"}"#).unwrap();
        assert_eq!(value["a"], serde_json::json!(1.5));
        assert_eq!(value["b"], serde_json::json!("Infinity in a string"));
    }

    #[test]
    fn the_mamba2_shape_parses_with_the_literal_preserved_as_a_string() {
        // The live witness's exact shape: transformers' allow_nan output.
        let value =
            parse_config_json(r#"{"time_step_limit": [0.0, Infinity], "model_type": "mamba2"}"#)
                .unwrap();
        assert_eq!(
            value["time_step_limit"],
            serde_json::json!([0.0, "Infinity"]),
            "the declaration is preserved as the string it spells, never a fabricated float"
        );
        assert_eq!(value["model_type"], serde_json::json!("mamba2"));
    }

    #[test]
    fn negative_infinity_and_nan_quote_as_whole_tokens() {
        let value = parse_config_json(r#"{"lo": -Infinity, "x": NaN}"#).unwrap();
        assert_eq!(value["lo"], serde_json::json!("-Infinity"));
        assert_eq!(value["x"], serde_json::json!("NaN"));
    }

    #[test]
    fn genuinely_malformed_json_keeps_the_strict_error() {
        // No non-finite literal to rescue: the strict parser's own error
        // must survive, not a second confusing one from a no-op rewrite.
        assert!(parse_config_json(r#"{"a": }"#).is_err());
    }

    #[test]
    fn identifier_prefixes_do_not_match() {
        // `NaNite` is not `NaN`; a key-adjacent word must not be quoted.
        assert!(parse_config_json(r#"{"a": NaNite}"#).is_err());
    }
}
