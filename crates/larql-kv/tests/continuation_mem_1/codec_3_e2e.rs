//! CONTINUATION-CODEC-3 reconnaissance: the run module EXECUTED end to
//! end on fixtures, against a CODEC-MAP-1 store the map harness wrote, and
//! the selection rule's known answers.

use super::codec_2_store::{scores_name, Checkpoints};
use super::codec_3::protection_spec;
use super::codec_3_run::{choose_window, map_parity_arm, run_codec_3_recon, SATURATION};
use super::codec_map_report::MapLine;
use super::codec_map_run::{run_recon, Recon};
use super::codec_map_sim::CompressMap;
use super::*;

const FIXTURE: Recon = Recon {
    rung: 40,
    resume: 4,
    decode: 32,
};
const PARITY: usize = 8;
const WINDOWS: [usize; 3] = [4, PARITY, 16];

fn ids() -> Vec<u32> {
    (0..48).map(|i| ((i * 7 + 3) % 29) as u32).collect()
}

fn subjects_() -> (Subject, Subject, ReferenceBackend) {
    (
        subjects::fixture(miniature_glimmer, "cr3-e2e-exact"),
        subjects::fixture(miniature_glimmer, "cr3-e2e-yard"),
        ReferenceBackend::new(),
    )
}

/// The CODEC-MAP-1 harness's own store, with the protection map at PARITY.
fn map_store(dir: &std::path::Path) {
    let (exact, yard, backend) = subjects_();
    let (eo, yo) = (exact.prepare(&backend), yard.prepare(&backend));
    // Written as the real harness writes it: the spec is the whole
    // `<name>=<clauses>` string the run was given (LARQL_CMAP_MAPS).
    let name = map_parity_arm(PARITY);
    let spec = format!(
        "{}={}",
        name.trim_start_matches("map-"),
        protection_spec(PARITY)
    );
    let map = CompressMap::parse(&spec).unwrap();
    let store = Checkpoints::open(dir, &json!({"implementation_git_sha": "fixture-map"}));
    run_recon(
        (&exact, &eo),
        (&yard, &yo),
        &backend,
        &ids(),
        &FIXTURE,
        &[(map, spec)],
        &store,
    );
}

fn recon_into(dir: &std::path::Path, map_dir: &std::path::Path, windows: &[usize]) -> Value {
    let (exact, yard, backend) = subjects_();
    let (eo, yo) = (exact.prepare(&backend), yard.prepare(&backend));
    let store = Checkpoints::open(dir, &json!({"implementation_git_sha": "fixture"}));
    let (map, _) = Checkpoints::open_as_run(map_dir);
    run_codec_3_recon(
        (&exact, &eo),
        (&yard, &yo),
        &backend,
        &ids(),
        &FIXTURE,
        (windows, PARITY),
        (&store, &map),
    )
}

#[test]
fn the_reconnaissance_runs_resumes_and_stops_on_a_parity_break() {
    let _serial = serial();
    let (map_dir, dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    map_store(map_dir.path());
    let report = recon_into(dir.path(), map_dir.path(), &WINDOWS[..2]);
    assert_eq!(report["stage0"]["passed"], true, "{}", report["stage0"]);
    let arms = report["arms"].as_array().unwrap();
    let names: Vec<&str> = arms.iter().map(|a| a["arm"].as_str().unwrap()).collect();
    assert_eq!(names, ["ch", "w4", "w8"]);
    for a in &arms[1..] {
        assert_eq!(a["residency_holds"], true, "{a}");
        assert_eq!(a["trace_identical_to_ch"], true, "{a}");
        assert!(
            a["resident_bytes"].as_f64().unwrap() > arms[0]["resident_bytes"].as_f64().unwrap()
        );
    }
    assert_eq!(
        arms[2]["sim_w_scores_identical"], true,
        "SIM-W at the parity window"
    );
    assert!(
        arms[1]["sim_w_scores_identical"].is_null(),
        "only the parity window is checked"
    );

    // A later stage adds a window and rewrites nothing already held.
    let files = Checkpoints::open_as_run(dir.path()).0.files();
    let before: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter(|f| f.as_str() != "report.json")
        .map(|f| (f.clone(), std::fs::read(dir.path().join(f)).unwrap()))
        .collect();
    let report = recon_into(dir.path(), map_dir.path(), &WINDOWS);
    assert_eq!(report["arms"].as_array().unwrap().len(), 4);
    assert!(
        report["chosen_window"].is_u64(),
        "{}",
        report["chosen_window"]
    );
    for (f, bytes) in before {
        assert_eq!(
            std::fs::read(dir.path().join(&f)).unwrap(),
            bytes,
            "{f} rewritten"
        );
    }

    // A reference store whose protection map scored otherwise stops SIM-W.
    let arm = scores_name(FIXTURE.rung, &map_parity_arm(PARITY));
    let path = map_dir.path().join(&arm);
    let mut held: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    held["positions"][0]["kl"] = json!(held["positions"][0]["kl"].as_f64().unwrap() + 1e-12);
    std::fs::write(&path, serde_json::to_string(&held).unwrap()).unwrap();
    let fresh = tempfile::tempdir().unwrap();
    let stopped = std::panic::catch_unwind(|| recon_into(fresh.path(), map_dir.path(), &[PARITY]))
        .expect_err("a moved reference must stop the run");
    let message = stopped
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("SIM-W failed at w 8"), "{message}");
}

fn line(wide: f64, narrow: f64) -> MapLine {
    MapLine {
        name: String::new(),
        spec: String::new(),
        exact_fraction: 0.0,
        top1_agreement: 0.0,
        mean_d: 0.0,
        p99_d: 0.0,
        explained_wide: 1.0 - wide,
        explained_narrow: 1.0 - narrow,
        tail_overlap_with_ch: 0.0,
    }
}

#[test]
fn the_selection_rule_takes_the_smallest_window_saturating_both_tails() {
    let _serial = serial();
    let curve = [
        (512, line(0.96, 1.0)),
        (64, line(0.40, 0.30)),
        (256, line(0.94, 1.0)),
        (128, line(0.90, 0.85)),
    ];
    assert_eq!(
        choose_window(&curve),
        Some(256),
        "128 saturates the wide tail, not the narrow"
    );
    let both = [
        (64, line(0.40, 0.30)),
        (128, line(SATURATION, SATURATION)),
        (256, line(1.0, 1.0)),
    ];
    assert_eq!(choose_window(&both), Some(128), "the boundary is inclusive");
    assert_eq!(choose_window(&[]), None);
}
