//! VINDEX3 generation-aware download completeness — tests for [`super`].
//!
//! Split out from `tests.rs` (vindex3-registry-publishing-design cleanup,
//! 2026-08-23) so the pre-existing hf_hub-plumbing suite and this
//! VINDEX3-specific acceptance suite each stay under the file-size cap
//! without losing either one's shape.

use super::test_support::{mock_hf_file_resolve, HfTestEnv, NoOpProgress};
use super::*;
use serial_test::serial;

// ─── VINDEX3 generation-aware completeness (vindex3-registry 2C) ───────
//
// The acceptance test that rung cares about most: not "these expected
// filenames were requested" (a hardcoded-list assertion the original
// bug would have passed too, since the list just needed the *wrong*
// names) but "the pulled result actually opens as a complete VINDEX3
// container." A real, self-encoded fixture is served over a mocked HF
// endpoint; every file it holds is discovered dynamically by walking
// the fixture on disk, never hardcoded here — so this test cannot
// silently drift the way the fixed VINDEX2 list did the day VINDEX3
// grew a payload it didn't know about.

/// Recursively collect every file under `root` as `(relative POSIX
/// path, bytes)`, sorted. Drives both the mock (what exists, what bytes
/// it holds) and the `repo.info()` siblings list — from the same
/// source, so the test can't accidentally list files the mock doesn't
/// actually serve or vice versa.
fn walk_fixture_files(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/");
                out.push((rel, std::fs::read(&path).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_downloads_a_complete_v3_container() {
    // A real, self-encoded VINDEX3 fixture — segments, moe_manifest.json,
    // the actual container shape — not a hand-written `{"version": 3}`
    // stub like a filename-list test would settle for.
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::encode_fixture_container(
        crate::format::vindex3::fixtures::miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "pull-fixture",
    );
    let files = walk_fixture_files(container.path());
    assert!(
        files.len() > 1,
        "fixture must carry more than index.json for this test to mean anything"
    );

    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());

    // `repo.info()` drives the V3 branch's enumeration — list exactly
    // what the fixture wrote, discovered above, never hardcoded here.
    let siblings: Vec<_> = files
        .iter()
        .map(|(rel, _)| serde_json::json!({"rfilename": rel}))
        .collect();
    let info_body = serde_json::json!({"sha": "deadbeef", "siblings": siblings}).to_string();
    let _info = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/api/models/owner/repo".into()),
        )
        .with_status(200)
        .with_header("Content-Type", "application/json")
        .with_body(info_body)
        .create();

    let mocks: Vec<_> = files
        .iter()
        .map(|(rel, bytes)| {
            let regex = rel.replace('.', r"\.");
            mock_hf_file_resolve(&mut server, &regex, rel, bytes)
        })
        .collect();

    let dir = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
        .expect("a complete VINDEX3 repo must download in full");

    for (mock, (rel, _)) in mocks.iter().zip(&files) {
        assert!(mock[0].matched(), "{rel} must have been fetched");
    }

    // The acceptance test itself: open the PULLED result through the
    // real VINDEX3 loader, not a filename checklist. `encode_fixture_container`
    // produces a system-graph container, not a routed-MoE one — its
    // real loader is `inspect_container` (`Vindex3Container::open`
    // itself refuses non-MoE containers by name, directing callers
    // here; see its own error text). `verify_payloads: true` also
    // checks the downloaded segment bytes are readable and
    // correctly shaped, not just that the manifest parses.
    crate::format::vindex3::inspect::inspect_container(&dir, true)
        .expect("a fully-downloaded VINDEX3 container must open and verify cleanly");
    assert!(
        dir.join(super::payloads::VERIFIED_MARKER).exists(),
        "the pull itself verified the payloads and recorded it"
    );
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_v3_refuses_segments_its_index_does_not_seal() {
    // Every listed file is served, complete and the right length — but
    // one segment's bytes are not the bytes `index.json` seals. The hub
    // delivers that intact; only the container's own digest can see it.
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::encode_fixture_container(
        crate::format::vindex3::fixtures::miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "pull-fixture",
    );
    let mut files = walk_fixture_files(container.path());
    let (damaged, bytes) = files
        .iter_mut()
        .find(|(rel, _)| {
            rel.ends_with(&format!(
                ".{}",
                crate::format::vindex3::encode::SEGMENT_BIN_EXT
            ))
        })
        .expect("the fixture carries a segment");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    let damaged = damaged.clone();

    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());
    let siblings: Vec<_> = files
        .iter()
        .map(|(rel, _)| serde_json::json!({"rfilename": rel}))
        .collect();
    let info_body = serde_json::json!({"sha": "deadbeef", "siblings": siblings}).to_string();
    let _info = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/api/models/owner/repo".into()),
        )
        .with_status(200)
        .with_header("Content-Type", "application/json")
        .with_body(info_body)
        .create();
    for (rel, bytes) in &files {
        let regex = rel.replace('.', r"\.");
        mock_hf_file_resolve(&mut server, &regex, rel, bytes);
    }

    let err = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
        .expect_err("a segment its index does not seal must refuse the pull");
    assert!(
        matches!(err, VindexError::PayloadDigestMismatch { .. })
            && err.to_string().contains(&damaged),
        "the refusal must be the digest check naming `{damaged}`: {err}"
    );
}

#[test]
#[serial]
fn resolve_hf_vindex_with_progress_v3_hard_fails_when_a_listed_file_is_missing() {
    // `repo.info()` lists a segment file the mock never actually
    // serves (a corrupt/incomplete upload) — must be a hard failure
    // naming the file, never a silently-incomplete "success".
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::encode_fixture_container(
        crate::format::vindex3::fixtures::miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "pull-fixture",
    );
    let files = walk_fixture_files(container.path());

    let mut server = mockito::Server::new();
    let _g = HfTestEnv::new(&server.url());

    // No catch-all mock — as elsewhere in this file, mockito's default
    // response for an unmatched request (the deliberately-unserved file
    // below) is sufficient; an explicit `Matcher::Any` mock risks
    // shadowing the specific per-file mocks registered around it.

    let mut siblings: Vec<_> = files
        .iter()
        .map(|(rel, _)| serde_json::json!({"rfilename": rel}))
        .collect();
    // Claim a file the mock will never serve.
    siblings.push(serde_json::json!({"rfilename": "routed/layer_999.lyrw"}));
    let info_body = serde_json::json!({"sha": "deadbeef", "siblings": siblings}).to_string();
    let _info = server
        .mock(
            "GET",
            mockito::Matcher::Regex(r"/api/models/owner/repo".into()),
        )
        .with_status(200)
        .with_header("Content-Type", "application/json")
        .with_body(info_body)
        .create();
    for (rel, bytes) in &files {
        let regex = rel.replace('.', r"\.");
        mock_hf_file_resolve(&mut server, &regex, rel, bytes);
    }

    let err = resolve_hf_vindex_with_progress("hf://owner/repo", |_| NoOpProgress)
        .expect_err("a repo missing a listed file must hard-fail, not silently succeed");
    assert!(
        err.to_string().contains("routed/layer_999.lyrw"),
        "error must name the missing file: {err}"
    );
}
