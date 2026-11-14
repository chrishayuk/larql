//! Unit tests for `speak_serve`: request parsing, block draining and
//! the reporting helpers.

use super::*;
use larql_inference::speech::moss_realtime::FrameStages;
use std::cell::RefCell;

/// Run `drain` over a canned request stream, returning the `text:` of
/// every request that actually reached the server. Replies go to
/// stdout because no block names a `reply:`.
fn drained(input: &str) -> Vec<String> {
    let seen = RefCell::new(Vec::new());
    let serve = |request: Request| -> Result<String, BoxErr> {
        seen.borrow_mut()
            .push(request.text.clone().unwrap_or_default());
        Ok("ok frames=1".to_string())
    };
    drain(BufReader::new(input.as_bytes()), &serve);
    seen.into_inner()
}

#[test]
fn absorb_fills_every_known_key() {
    let mut request = Request::default();
    for line in [
        "text: Good evening Sam.",
        "voice: /voices/jarvis.tokens",
        "tokens: /out/frames.txt",
        "stream: /out/frames.fifo",
        "seed: 5",
        "max-frames: 200",
        "greedy: true",
        "reply: /out/reply.fifo",
    ] {
        request.absorb(line).expect("known key");
    }
    assert_eq!(request.text.as_deref(), Some("Good evening Sam."));
    assert_eq!(request.voice.as_deref(), Some("/voices/jarvis.tokens"));
    assert_eq!(request.tokens, Some(PathBuf::from("/out/frames.txt")));
    assert_eq!(request.stream, Some(PathBuf::from("/out/frames.fifo")));
    assert_eq!(request.seed, Some(5));
    assert_eq!(request.max_frames, Some(200));
    assert_eq!(request.greedy, Some(true));
    assert_eq!(request.reply, Some(PathBuf::from("/out/reply.fifo")));
}

/// The whole point of rejecting unknown keys: a typo'd `voise:` must
/// not be silently dropped, or it synthesises the default voice and
/// looks like a model bug.
#[test]
fn absorb_rejects_an_unknown_key() {
    let error = Request::default()
        .absorb("voise: /voices/jarvis.tokens")
        .expect_err("typo must not be a silent no-op");
    assert!(error.to_string().contains("voise"), "{error}");
}

#[test]
fn absorb_rejects_a_line_without_a_colon() {
    let error = Request::default()
        .absorb("text is missing its colon")
        .expect_err("header lines are `key: value`");
    assert!(error.to_string().contains("malformed"), "{error}");
}

#[test]
fn absorb_rejects_an_unparseable_number() {
    assert!(Request::default().absorb("seed: soon").is_err());
    assert!(Request::default().absorb("max-frames: lots").is_err());
}

#[test]
fn absorb_trims_around_the_separator() {
    let mut request = Request::default();
    request
        .absorb("  text  :   Good evening.  ")
        .expect("trimmed");
    assert_eq!(request.text.as_deref(), Some("Good evening."));
}

/// `voice: -` must survive parsing as the sentinel so `serve_one` can
/// tell "explicitly unconditioned" from "inherit the server default".
#[test]
fn absorb_keeps_the_unconditioned_sentinel() {
    let mut request = Request::default();
    request.absorb("voice: -").expect("sentinel");
    assert_eq!(request.voice.as_deref(), Some(UNCONDITIONED));
    assert!(Request::default().voice.is_none(), "absent != sentinel");
}

#[test]
fn absorb_reads_greedy_as_a_flag() {
    for truthy in ["1", "true", "yes"] {
        let mut request = Request::default();
        request.absorb(&format!("greedy: {truthy}")).expect("flag");
        assert_eq!(request.greedy, Some(true), "{truthy}");
    }
    let mut request = Request::default();
    request.absorb("greedy: no").expect("flag");
    assert_eq!(request.greedy, Some(false));
}

#[test]
fn drain_settles_one_block_per_blank_line() {
    assert_eq!(
        drained("text: first\n\ntext: second\n\n"),
        ["first", "second"]
    );
}

#[test]
fn drain_settles_a_trailing_block_at_eof() {
    // No terminating blank line — EOF ends the block instead.
    assert_eq!(drained("text: only\n"), ["only"]);
}

#[test]
fn drain_ignores_runs_of_blank_lines() {
    assert_eq!(drained("\n\ntext: first\n\n\n\n"), ["first"]);
}

#[test]
fn drain_does_not_serve_a_malformed_block() {
    assert!(drained("voise: typo\ntext: unheard\n\n").is_empty());
}

/// A bad block must not take the server down with it — the next one
/// still gets served.
#[test]
fn drain_keeps_serving_after_a_malformed_block() {
    assert_eq!(
        drained("voise: typo\n\ntext: still here\n\n"),
        ["still here"]
    );
}

/// …and neither must a request that fails inside the model.
#[test]
fn drain_keeps_serving_after_a_failing_request() {
    let served = RefCell::new(0usize);
    let serve = |_request: Request| -> Result<String, BoxErr> {
        *served.borrow_mut() += 1;
        Err("generation produced no frames before EOS".into())
    };
    drain(
        BufReader::new("text: one\n\ntext: two\n\n".as_bytes()),
        &serve,
    );
    assert_eq!(served.into_inner(), 2);
}

#[test]
fn stage_split_is_blank_without_stages() {
    assert_eq!(stage_split(&[]), "");
}

#[test]
fn stage_split_reports_medians_in_milliseconds() {
    let stages = [
        // Frame 0 rides the prefill, so its backbone time is absent
        // (0.0) and must not drag the median down.
        FrameStages {
            backbone_seconds: 0.0,
            depth_seconds: 0.030,
        },
        FrameStages {
            backbone_seconds: 0.012,
            depth_seconds: 0.032,
        },
        FrameStages {
            backbone_seconds: 0.014,
            depth_seconds: 0.034,
        },
    ];
    assert_eq!(
        stage_split(&stages),
        " | backbone p50 14 ms | depth p50 32 ms"
    );
}

#[test]
fn stage_split_is_blank_when_a_stage_never_recorded() {
    let stages = [FrameStages {
        backbone_seconds: 0.0,
        depth_seconds: 0.030,
    }];
    assert_eq!(stage_split(&stages), "");
}

#[test]
fn truncate_leaves_short_text_alone() {
    assert_eq!(truncate("Good evening.", 60), "Good evening.");
}

#[test]
fn truncate_marks_what_it_cut() {
    assert_eq!(truncate("abcdef", 3), "abc…");
}

/// Counts characters, not bytes — a byte slice would panic here.
#[test]
fn truncate_is_multibyte_safe() {
    assert_eq!(truncate("héllo wörld", 4), "héll…");
}
