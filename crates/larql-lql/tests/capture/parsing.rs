//! How CAPTURE parses, and what the parser says when it cannot.

use larql_lql::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite};
use larql_lql::{parse, CapabilityProfile, Statement};

fn parsed(text: &str) -> Statement {
    parse(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

#[test]
fn the_plainest_form_captures_the_last_position() {
    let Statement::Capture {
        site,
        layers,
        prompt,
        continuation,
        positions,
        output,
    } = parsed(r#"CAPTURE logits FOR "The capital of France is" TO "l.npy";"#)
    else {
        panic!("not Capture")
    };
    assert_eq!(site, CaptureSite::Logits);
    assert_eq!(layers, None);
    assert_eq!(prompt, "The capital of France is");
    assert_eq!(continuation, None);
    assert_eq!(positions, CapturePositions::Last);
    assert_eq!(output, "l.npy");
}

/// A statement and what it parses to: site, layers, continuation, positions.
type Case = (
    &'static str,
    CaptureSite,
    Option<CaptureLayers>,
    Option<CaptureContinuation>,
    CapturePositions,
);

#[test]
fn every_clause_in_any_case() {
    let cases: [Case; 6] = [
        (
            r#"capture ffn.down_input at layers (3, 1) for "p" continue " Paris" positions reply to "a.npy""#,
            CaptureSite::FfnDownInput,
            Some(CaptureLayers::Many(vec![3, 1])),
            Some(CaptureContinuation::Text(" Paris".into())),
            CapturePositions::Reply,
        ),
        (
            r#"CAPTURE residual AT LAYER 0 FOR "p" CONTINUE TOKENS (5, 6) POSITIONS ALL TO "a.npy";"#,
            CaptureSite::Residual,
            Some(CaptureLayers::One(0)),
            Some(CaptureContinuation::Tokens(vec![5, 6])),
            CapturePositions::All,
        ),
        (
            r#"CAPTURE final FOR "p" GENERATE 3 POSITIONS PROMPT_END TO "a.npy";"#,
            CaptureSite::Final,
            None,
            Some(CaptureContinuation::Generate(3)),
            CapturePositions::PromptEnd,
        ),
        (
            r#"CAPTURE post_attention AT LAYER 1 FOR "p" POSITIONS (4, 0) TO "a.npy";"#,
            CaptureSite::PostAttention,
            Some(CaptureLayers::One(1)),
            None,
            CapturePositions::List(vec![4, 0]),
        ),
        (
            r#"CAPTURE FFN.Input AT LAYER 1 FOR "p" POSITIONS LAST TO "a.npy";"#,
            CaptureSite::FfnInput,
            Some(CaptureLayers::One(1)),
            None,
            CapturePositions::Last,
        ),
        (
            r#"CAPTURE ffn.down_output AT LAYER 1 FOR "p" TO "a.npy";"#,
            CaptureSite::FfnDownOutput,
            Some(CaptureLayers::One(1)),
            None,
            CapturePositions::Last,
        ),
    ];
    for (text, want_site, want_layers, want_continuation, want_positions) in cases {
        let Statement::Capture {
            site,
            layers,
            continuation,
            positions,
            ..
        } = parsed(text)
        else {
            panic!("{text}")
        };
        assert_eq!(site, want_site, "{text}");
        assert_eq!(layers, want_layers, "{text}");
        assert_eq!(continuation, want_continuation, "{text}");
        assert_eq!(positions, want_positions, "{text}");
    }
}

#[test]
fn every_site_name_parses_back_to_its_site() {
    for site in CaptureSite::ALL {
        let text = format!(r#"CAPTURE {} FOR "p" TO "a.npy";"#, site.name());
        let Statement::Capture {
            site: parsed_site, ..
        } = parsed(&text)
        else {
            panic!("{text}")
        };
        assert_eq!(parsed_site, site);
    }
}

#[test]
fn a_malformed_statement_says_what_was_expected() {
    for (text, needle) in [
        (
            r#"CAPTURE attention FOR "p" TO "a.npy";"#,
            "expected a site",
        ),
        (r#"CAPTURE ffn. FOR "p" TO "a.npy";"#, "expected a site"),
        (r#"CAPTURE "logits" FOR "p" TO "a.npy";"#, "expected a site"),
        (r#"CAPTURE logits "p" TO "a.npy";"#, "AT or FOR"),
        (
            r#"CAPTURE residual AT 1 FOR "p" TO "a.npy";"#,
            "LAYER or LAYERS",
        ),
        (
            r#"CAPTURE residual AT LAYER 1 "p" TO "a.npy";"#,
            "expected FOR",
        ),
        (
            r#"CAPTURE logits FOR "p" "a.npy";"#,
            "CONTINUE, GENERATE, POSITIONS or TO",
        ),
        (
            r#"CAPTURE logits FOR "p" GENERATE 2 "a.npy";"#,
            "POSITIONS or TO",
        ),
        (r#"CAPTURE logits FOR "p" CONTINUE 5 TO "a.npy";"#, "TOKENS"),
        (
            r#"CAPTURE logits FOR "p" POSITIONS FIRST TO "a.npy";"#,
            "PROMPT_END",
        ),
        (r#"CAPTURE logits FOR "p" POSITIONS () TO "a.npy";"#, ""),
        (r#"CAPTURE logits FOR p TO "a.npy";"#, "string"),
    ] {
        let err = parse(text).unwrap_err().to_string();
        assert!(err.contains(needle), "{text}: {err}");
        if !needle.is_empty() && !err.contains("string") {
            assert!(err.contains("HELP CAPTURE;"), "{text}: {err}");
        }
    }
}

#[test]
fn it_is_named_writes_an_artifact_and_is_outside_the_public_profile() {
    let capture = parsed(r#"CAPTURE logits FOR "p" TO "a.npy";"#);
    assert_eq!(capture.verb(), "CAPTURE");
    assert!(capture.writes_artifact());
    let err = CapabilityProfile::PublicExplorer
        .check(&capture)
        .unwrap_err()
        .to_string();
    assert!(err.contains("CAPTURE"), "{err}");
}
