use super::*;

/// A trace survives a write/read round trip unchanged — provenance
/// included. Losing the header on a round trip would strip a replay
/// result of what it replays.
#[test]
fn round_trip_preserves_decisions_and_provenance() {
    let mut t = Trace::new("bwc5 layer=20 lookahead=6 prompt_idx=3");
    t.record(20, 0);
    t.record(20, 5);
    t.record(20, 1);
    let back = Trace::parse(&t.render()).expect("parses");
    assert_eq!(back, t);
    assert_eq!(back.len(), 3);
    assert_eq!(
        back.source.as_deref(),
        Some("bwc5 layer=20 lookahead=6 prompt_idx=3")
    );
}

/// The written form is deterministic and sorted, so two traces from two
/// runs diff cleanly instead of differing by insertion order.
#[test]
fn rendered_pairs_are_sorted_and_deterministic() {
    let mut a = Trace::new("s");
    for (l, s) in [(21, 4), (20, 9), (20, 1)] {
        a.record(l, s);
    }
    let mut b = Trace::new("s");
    for (l, s) in [(20, 1), (21, 4), (20, 9)] {
        b.record(l, s);
    }
    assert_eq!(a.render(), b.render());
    let rendered = a.render();
    let body: Vec<&str> = rendered
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.trim())
        .collect();
    assert_eq!(body, vec!["20 1", "20 9", "21 4"]);
}

/// Duplicates collapse, so `len` is the number of distinct skips a
/// replay should produce — the count a caller compares its observed
/// skips against to detect that the two runs diverged.
#[test]
fn duplicate_decisions_collapse() {
    let mut t = Trace::new("s");
    t.record(20, 0);
    t.record(20, 0);
    assert_eq!(t.len(), 1);
}

/// A file with no format marker is refused. Reading an arbitrary text
/// file as a trace would otherwise yield an empty (or partial) policy
/// that replays as canonical and reads as "the policy did nothing".
#[test]
fn a_file_without_the_marker_is_refused() {
    let err = Trace::parse("20 0\n21 1\n").expect_err("must refuse");
    assert!(err.contains("format marker"), "{err}");
}

/// A malformed line is an ERROR, not a silently skipped line. Dropping
/// half a trace's decisions would produce a replay that looks like a
/// weaker policy rather than a broken file — a wrong answer, not a
/// visible failure.
#[test]
fn a_malformed_line_is_an_error_naming_the_line() {
    for (body, want) in [
        ("20\n", "expected `layer step`"),
        ("20 1 2\n", "expected `layer step`"),
        ("twenty 1\n", "is not an integer"),
        ("20 first\n", "is not an integer"),
    ] {
        let text = format!("# larql-exec-trace v1\n{body}");
        let err = Trace::parse(&text).expect_err("must refuse {body}");
        assert!(err.contains(want), "for {body:?} got {err}");
        assert!(err.contains("line 2"), "must name the line: {err}");
    }
}

/// Blank lines, comments and surrounding whitespace are tolerated — the
/// file is meant to be hand-editable when a replay disagrees with its
/// source run.
#[test]
fn comments_and_blank_lines_are_tolerated() {
    let text = "# larql-exec-trace v1\n\n# a note\n  20 3  \n\n21 4\n";
    let t = Trace::parse(text).expect("parses");
    assert_eq!(t.len(), 2);
    assert!(t.skips.contains(&(20, 3)));
    assert!(t.skips.contains(&(21, 4)));
}

/// A trace with no provenance line reads, but says so — `None`, never a
/// fabricated description. A result quoted from it cannot claim to know
/// which policy it replays.
#[test]
fn a_trace_without_provenance_reports_none() {
    let t = Trace::parse("# larql-exec-trace v1\n20 0\n").expect("parses");
    assert_eq!(t.source, None);
    assert_eq!(t.len(), 1);
}

/// Round trip through the filesystem, including the error path for a
/// file that is not there — the message must name the path, since the
/// operator typed it.
#[test]
fn file_round_trip_and_missing_file_names_the_path() {
    let dir = std::env::temp_dir().join(format!("larql-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("t.trace");
    let mut t = Trace::new("unit test");
    t.record(7, 2);
    t.write(&path).expect("writes");
    assert_eq!(Trace::read(&path).expect("reads"), t);

    let missing = dir.join("nope.trace");
    let err = Trace::read(&missing).expect_err("must fail");
    assert!(err.contains("nope.trace"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
