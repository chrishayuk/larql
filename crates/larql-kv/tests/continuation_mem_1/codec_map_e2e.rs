//! CONTINUATION-CODEC-MAP-1: the reconnaissance run module EXECUTED end to
//! end on fixtures (CODEC-2 W2's lesson: a run module that only compiles is
//! untested), and known answers for the tail report.

use std::collections::BTreeSet;

use super::codec_2_store::Checkpoints;
use super::codec_map_report::{explained, jaccard, tail_persistence, worst};
use super::codec_map_run::{run_recon, Recon};
use super::codec_map_sim::CompressMap;
use super::*;

const FIXTURE: Recon = Recon {
    rung: 40,
    resume: 4,
    decode: 32,
};

fn ids() -> Vec<u32> {
    (0..48).map(|i| ((i * 7 + 3) % 29) as u32).collect()
}

fn recon_into(dir: &std::path::Path, specs: &[&str]) -> Value {
    let exact = subjects::fixture(miniature_glimmer, "cmap-e2e-exact");
    let yard = subjects::fixture(miniature_glimmer, "cmap-e2e-yard");
    let backend = ReferenceBackend::new();
    let (eo, yo) = (exact.prepare(&backend), yard.prepare(&backend));
    let specs: Vec<String> = specs.iter().map(|s| s.to_string()).collect();
    let maps: Vec<(CompressMap, String)> = specs
        .iter()
        .map(|s| (CompressMap::parse(s).unwrap(), s.clone()))
        .collect();
    let store = Checkpoints::open(dir, &json!({"implementation_git_sha": "fixture"}));
    run_recon(
        (&exact, &eo),
        (&yard, &yo),
        &backend,
        &ids(),
        &FIXTURE,
        &maps,
        &store,
    )
}

#[test]
fn the_reconnaissance_runs_resumes_and_reports_end_to_end() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let report = recon_into(dir.path(), &["k=all:k:all", "v=all:v:all"]);
    assert_eq!(report["sim0"]["passed"], true, "{}", report["sim0"]);
    let maps = report["maps"].as_array().unwrap();
    let names: Vec<&str> = maps.iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["ch_full", "k", "v"]);
    assert_eq!(
        maps[0]["explained_wide"].as_f64().unwrap(),
        1.0,
        "full CH explains all of its own tail (and the fixture's codec is not a no-op)"
    );
    for m in &maps[1..] {
        let f = m["exact_fraction"].as_f64().unwrap();
        assert!(
            (0.49..=0.51).contains(&f),
            "{}: one of K/V compressed keeps half exact, got {f}",
            m["name"]
        );
    }

    let files = Checkpoints::open_as_run(dir.path()).0.files();
    let unique: BTreeSet<&String> = files.iter().collect();
    assert_eq!(unique.len(), files.len());

    // Stage 2 adds a map; stage 0 and earlier maps are not rerun.
    let before: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter(|f| f.as_str() != "report.json")
        .map(|f| (f.clone(), std::fs::read(dir.path().join(f)).unwrap()))
        .collect();
    let report = recon_into(dir.path(), &["k=all:k:all", "recent=all:kv:older8"]);
    assert_eq!(
        report["maps"].as_array().unwrap().len(),
        4,
        "every checkpointed map is reported"
    );
    for (f, bytes) in before {
        assert_eq!(
            std::fs::read(dir.path().join(&f)).unwrap(),
            bytes,
            "{f} rewritten"
        );
    }

    let refused = std::panic::catch_unwind(|| recon_into(dir.path(), &["k=all:v:all"]));
    assert!(
        refused.is_err(),
        "a map name is never reused for another spec"
    );
}

#[test]
fn the_tail_report_has_known_answers() {
    let d_ch = [0.0, 5.0, 1.0, 4.0, 0.5, 3.0, 0.0, 0.0, 0.0, 2.0];
    assert_eq!(worst(&d_ch, 0.2), BTreeSet::from([1, 3]));
    assert_eq!(
        worst(&d_ch, 0.01),
        BTreeSet::from([1]),
        "at least one position"
    );
    let set = worst(&d_ch, 0.3);
    assert_eq!(explained(&d_ch, &d_ch, &set), 1.0);
    let half: Vec<f64> = d_ch.iter().map(|x| x / 2.0).collect();
    assert_eq!(explained(&half, &d_ch, &set), 0.5);
    assert_eq!(
        explained(&[0.0; 10], &[0.0; 10], &set),
        0.0,
        "no CH tail explains nothing"
    );
    assert_eq!(
        jaccard(&BTreeSet::from([1, 2]), &BTreeSet::from([2, 3])),
        1.0 / 3.0
    );
    let mut shifted = d_ch.to_vec();
    shifted.swap(1, 0);
    let persistence = tail_persistence(&d_ch, &[d_ch.to_vec(), shifted]);
    assert_eq!(
        persistence,
        vec![(1, 1)],
        "position 1 stays worst in one of two maps"
    );
}
