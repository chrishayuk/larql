//! **UNCERTAINTY-2 — offline sequence-level resampling over the two
//! REAL-EVIDENCE-1 observation streams.**
//!
//! Freeze: `docs/represent/forecasts/represent-uncertainty-2.json`. No
//! model runs. Both streams pass three gates before a number exists:
//! artifact integrity (hash + count), experimental identity (the
//! manifest's resolved scope against the committed declaration) and
//! replay (the rederived full bank equals the run's own report).
//!
//! ```text
//! cargo run --release -p larql-vindex --example represent_uncertainty_2 -- \
//!   --selection ~/chris-models/streams/stream-flagship-selection-8192 \
//!   --heldout   ~/chris-models/streams/stream-flagship-heldout-8192 \
//!   --declaration docs/represent/forecasts/represent-real-evidence-1-identity.json \
//!   --out docs/represent/forecasts/represent-uncertainty-2-result.json
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::represent::bank::PositionObservation;
use larql_vindex::format::vindex3::represent::experiment_identity::DeclaredIdentity;
use larql_vindex::format::vindex3::represent::observation_stream::{read_stream, StreamManifest};
use larql_vindex::format::vindex3::represent::quality::kimi_logit_v3;
use larql_vindex::format::vindex3::represent::resampling::blocks::{independent, Blocks};
use larql_vindex::format::vindex3::represent::resampling::dependence::{
    describe, readings, Dependence,
};
use larql_vindex::format::vindex3::represent::resampling::ladder::{progressive, Ladder};
use larql_vindex::format::vindex3::represent::resampling::summary::{summarise, DepthSummary};
use larql_vindex::format::vindex3::represent::resampling::transfer::{coverage, Coverage};
use larql_vindex::format::vindex3::represent::resampling::{
    seed_for, Analysis, SequenceSet, CLUSTERING_PERMUTATIONS, INDEPENDENT_PARTITIONS,
    PROGRESSIVE_DRAWS, STATISTICS,
};
use larql_vindex::format::vindex3::represent::stream_replay::replay_matches_report;

const PROGRAMME: &str = "REPRESENT-UNCERTAINTY-2";
const FREEZE: &str = "docs/represent/forecasts/represent-uncertainty-2.json";
const REPORT: &str = "report.json";
/// The only producer whose streams are the historical arm.
const PRODUCER: &str = "kda_q8_real measurement arm";
const SELECTION: usize = 0;
const HELDOUT: usize = 1;
const BANK_NAMES: [&str; 2] = ["selection", "heldout"];

struct Args {
    dirs: [PathBuf; 2],
    declaration: PathBuf,
    out: PathBuf,
}

fn args() -> Args {
    let mut m: BTreeMap<String, String> = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        m.insert(k, v);
    }
    let get = |k: &str| PathBuf::from(m.get(k).unwrap_or_else(|| panic!("missing {k}")));
    Args {
        dirs: [get("--selection"), get("--heldout")],
        declaration: get("--declaration"),
        out: get("--out"),
    }
}

/// A stream that has passed all three gates.
struct Admitted {
    manifest: StreamManifest,
    observations: Vec<PositionObservation>,
}

fn admit(dir: &Path, declared: &DeclaredIdentity) -> Admitted {
    let (manifest, observations) =
        read_stream(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    assert!(
        manifest.scope.same_transition(&declared.scope),
        "{}: scope {} is not the declared {}",
        dir.display(),
        manifest.scope.describe(),
        declared.scope.describe()
    );
    assert_eq!(
        Some(manifest.candidate_identity.as_str()),
        declared.expert_candidate.as_deref()
    );
    assert_eq!(manifest.producer, PRODUCER, "{}", dir.display());
    let witness = replay_matches_report(dir, &dir.join(REPORT))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    assert_eq!(
        witness.run.as_deref(),
        Some(declared.run.as_str()),
        "{}: the report is not the declared run",
        dir.display()
    );
    eprintln!(
        "[admit] {} — {} observations, scope {}, replay EXACT, sha {}",
        dir.display(),
        manifest.observations,
        manifest.scope.describe(),
        &manifest.stream_sha256[..12]
    );
    Admitted {
        manifest,
        observations,
    }
}

struct BankResult {
    full: [f64; 2],
    ladder: Ladder,
    blocks: Blocks,
    dependence: Dependence,
}

fn analyse(bank_index: usize, a: &Admitted) -> BankResult {
    let set = SequenceSet::new(&a.observations);
    let full = set.statistics(&set.ids());
    let t = std::time::Instant::now();
    let ladder = progressive(
        &set,
        PROGRESSIVE_DRAWS,
        seed_for(bank_index as u64, Analysis::Progressive),
    );
    eprintln!(
        "[{}] ladder: {} draws in {:.0}s",
        BANK_NAMES[bank_index],
        PROGRESSIVE_DRAWS,
        t.elapsed().as_secs_f64()
    );
    let t = std::time::Instant::now();
    let blocks = independent(
        &set,
        INDEPENDENT_PARTITIONS,
        seed_for(bank_index as u64, Analysis::Independent),
    );
    eprintln!(
        "[{}] blocks: {} partitions in {:.0}s",
        BANK_NAMES[bank_index],
        INDEPENDENT_PARTITIONS,
        t.elapsed().as_secs_f64()
    );
    let dependence = describe(
        &readings(&a.observations),
        a.manifest.positions_per_sequence as usize,
        CLUSTERING_PERMUTATIONS,
        seed_for(bank_index as u64, Analysis::Clustering),
    );
    BankResult {
        full,
        ladder,
        blocks,
        dependence,
    }
}

fn limit_for(statistic: usize) -> Option<f64> {
    (STATISTICS[statistic] == larql_vindex::format::vindex3::represent::statistic::Statistic::KlP99)
        .then(|| kimi_logit_v3().kl_p99_max)
}

fn ladder_summaries(r: &BankResult, positions: usize) -> BTreeMap<String, Vec<DepthSummary>> {
    STATISTICS
        .iter()
        .enumerate()
        .map(|(s, stat)| {
            let rows = r
                .ladder
                .depths
                .iter()
                .enumerate()
                .map(|(di, d)| {
                    let steps = r.ladder.steps(di, s, r.full[s]);
                    summarise(
                        *d,
                        positions,
                        &r.ladder.at(di, s),
                        r.full[s],
                        limit_for(s),
                        steps.as_deref(),
                    )
                })
                .collect();
            (stat.label().to_string(), rows)
        })
        .collect()
}

fn block_summaries(r: &BankResult, positions: usize) -> BTreeMap<String, Vec<DepthSummary>> {
    STATISTICS
        .iter()
        .enumerate()
        .map(|(s, stat)| {
            let rows = r
                .blocks
                .depths
                .iter()
                .enumerate()
                .map(|(di, d)| {
                    summarise(
                        *d,
                        positions,
                        &r.blocks.at(di, s),
                        r.full[s],
                        limit_for(s),
                        None,
                    )
                })
                .collect();
            (stat.label().to_string(), rows)
        })
        .collect()
}

fn transfers(
    source: &BankResult,
    target: &BankResult,
) -> (
    BTreeMap<String, Vec<Coverage>>,
    BTreeMap<String, Vec<Coverage>>,
) {
    let per = |src: &dyn Fn(usize, usize) -> Vec<f64>,
               tgt: &dyn Fn(usize, usize) -> Vec<f64>,
               depths: &[usize]| {
        STATISTICS
            .iter()
            .enumerate()
            .map(|(s, stat)| {
                let rows = depths
                    .iter()
                    .enumerate()
                    .map(|(di, d)| {
                        coverage(*d, &src(di, s), source.full[s], &tgt(di, s), target.full[s])
                    })
                    .collect();
                (stat.label().to_string(), rows)
            })
            .collect()
    };
    (
        per(
            &|di, s| source.blocks.at(di, s),
            &|di, s| target.blocks.at(di, s),
            &source.blocks.depths,
        ),
        per(
            &|di, s| source.ladder.at(di, s),
            &|di, s| target.ladder.at(di, s),
            &source.ladder.depths,
        ),
    )
}

fn print_table(name: &str, rows: &BTreeMap<String, Vec<DepthSummary>>) {
    eprintln!("--- {name}");
    for (stat, depths) in rows {
        eprintln!("  {stat}");
        for d in depths {
            eprintln!(
                "    n={:>3} pos={:>4}  rel p05 {:+.3} p50 {:+.3} p95 {:+.3}  |rel| {:.3}  ±10% {:.2} ±25% {:.2}{}",
                d.sequences, d.positions, d.relative.p05, d.relative.p50, d.relative.p95, d.mean_abs_relative,
                d.within["±10%"], d.within["±25%"],
                d.pass_fraction.map(|p| format!("  pass {p:.3}")).unwrap_or_default()
            );
        }
    }
}

fn main() {
    let args = args();
    let declared = DeclaredIdentity::load(&args.declaration).unwrap_or_else(|e| panic!("{e}"));
    let admitted: Vec<Admitted> = args.dirs.iter().map(|d| admit(d, &declared)).collect();
    let results: Vec<BankResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = admitted
            .iter()
            .enumerate()
            .map(|(i, a)| scope.spawn(move || analyse(i, a)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("analysis thread"))
            .collect()
    });
    let positions = admitted[SELECTION].manifest.positions_per_sequence as usize;
    let (a3_blocks, a3_ladder) = transfers(&results[SELECTION], &results[HELDOUT]);

    let mut out = serde_json::Map::new();
    out.insert("programme".into(), PROGRAMME.into());
    out.insert("freeze".into(), FREEZE.into());
    out.insert("written".into(), "2026-09-19".into());
    let mut streams = serde_json::Map::new();
    for (i, a) in admitted.iter().enumerate() {
        streams.insert(
            BANK_NAMES[i].into(),
            serde_json::json!({
                "dir": args.dirs[i].display().to_string(),
                "stream_sha256": a.manifest.stream_sha256,
                "bank_identity": a.manifest.bank_identity,
                "code_identity": a.manifest.code_identity,
                "observations": a.manifest.observations,
                "sequences": a.manifest.sequences,
                "gates": "integrity (hash+count), identity (declared scope/candidate/producer), replay (exact)",
            }),
        );
    }
    out.insert("streams".into(), streams.into());
    out.insert("parameters".into(), serde_json::json!({
        "depths": results[SELECTION].ladder.depths,
        "progressive_draws": PROGRESSIVE_DRAWS,
        "independent_partitions": INDEPENDENT_PARTITIONS,
        "clustering_permutations": CLUSTERING_PERMUTATIONS,
        "seeds": {
            "selection": {"progressive": results[SELECTION].ladder.seed, "independent": results[SELECTION].blocks.seed},
            "heldout": {"progressive": results[HELDOUT].ladder.seed, "independent": results[HELDOUT].blocks.seed},
        },
        "kl_p99_limit": kimi_logit_v3().kl_p99_max,
        "statistics": STATISTICS.iter().map(|s| s.label()).collect::<Vec<_>>(),
    }));
    let mut full = serde_json::Map::new();
    for (i, r) in results.iter().enumerate() {
        full.insert(
            BANK_NAMES[i].into(),
            serde_json::json!({
                STATISTICS[0].label(): r.full[0], STATISTICS[1].label(): r.full[1],
            }),
        );
    }
    out.insert("full_values".into(), full.into());
    for (key, f) in [
        (
            "A1_progressive",
            ladder_summaries as fn(&BankResult, usize) -> _,
        ),
        ("A2_independent_blocks", block_summaries),
    ] {
        let mut per_bank = serde_json::Map::new();
        for (i, r) in results.iter().enumerate() {
            let rows = f(r, positions);
            print_table(&format!("{key} / {}", BANK_NAMES[i]), &rows);
            per_bank.insert(BANK_NAMES[i].into(), serde_json::to_value(rows).unwrap());
        }
        out.insert(key.into(), per_bank.into());
    }
    out.insert(
        "A3_transfer_blocks".into(),
        serde_json::to_value(&a3_blocks).unwrap(),
    );
    out.insert(
        "A3_transfer_ladder".into(),
        serde_json::to_value(&a3_ladder).unwrap(),
    );
    for (stat, rows) in &a3_blocks {
        eprintln!("--- A3 blocks transfer / {stat}");
        for c in rows {
            eprintln!(
                "    n={:>3}  relative coverage {:.3}  absolute coverage {:.3}  (nominal {:.2})",
                c.sequences, c.relative_coverage, c.absolute_coverage, c.nominal
            );
        }
    }
    let mut dep = serde_json::Map::new();
    for (i, r) in results.iter().enumerate() {
        eprintln!("--- dependence / {}: {:?}", BANK_NAMES[i], r.dependence);
        dep.insert(
            BANK_NAMES[i].into(),
            serde_json::to_value(&r.dependence).unwrap(),
        );
    }
    out.insert("dependence".into(), dep.into());
    std::fs::write(
        &args.out,
        serde_json::to_vec_pretty(&serde_json::Value::Object(out)).unwrap(),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", args.out.display()));
    eprintln!("[done] {}", args.out.display());
}
