//! **The KDA candidate, compiled from the real container.**
//!
//! Gated on `LARQL_KIMI_VINDEX3`. The scope is `LARQL_KDA_MAP` — the
//! same inclusive-band spelling the expert driver reads
//! (`"20-22:Q8_0,24-25:Q8_0"`), over KDA layers only: an MLA layer in
//! the band is refused naturally, because MLA ships no separate
//! k/v_proj and its q_proj geometry disagrees with the placement's.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.s6.vindex3 \
//! LARQL_KDA_MAP="20-22:Q8_0,24-25:Q8_0" \
//! LARQL_KDA_CANDIDATE_OUT=/tmp/kimi-kda-l20-25q80.vindex3 \
//!   cargo test -p larql-vindex --features gpu --release \
//!   compile_the_kda_candidate -- --nocapture
//! ```

use std::path::PathBuf;

use super::super::compile_real_tests::{source_dependency, SegmentSource};
use super::super::compiler::CandidateIndex;
use super::super::map::{Exception, PrecisionMap};
use super::{compile_kda_bank, KDA_PROJECTIONS};

const CONTAINER_ENV: &str = "LARQL_KIMI_VINDEX3";
const MAP_ENV: &str = "LARQL_KDA_MAP";
const OUT_ENV: &str = "LARQL_KDA_CANDIDATE_OUT";
const OBJECT: &str = "target.kda_bank";

fn bands() -> Option<Vec<((u32, u32), String)>> {
    let spec = std::env::var(MAP_ENV).ok().filter(|v| !v.is_empty())?;
    Some(
        spec.split(',')
            .map(|band| {
                let (range, enc) = band
                    .split_once(':')
                    .unwrap_or_else(|| panic!("{MAP_ENV}: `{band}` is not `LAYERS:ENCODING`"));
                let (lo, hi) = match range.split_once('-') {
                    Some((a, b)) => (a.trim().parse().unwrap(), b.trim().parse().unwrap()),
                    None => {
                        let l: u32 = range.trim().parse().unwrap();
                        (l, l)
                    }
                };
                assert!(lo <= hi, "{MAP_ENV}: band `{band}` is inverted");
                ((lo, hi), enc.trim().to_string())
            })
            .collect(),
    )
}

#[test]
fn compile_the_kda_candidate() {
    let Some(container) = std::env::var_os(CONTAINER_ENV).map(PathBuf::from) else {
        eprintln!("skipped: set {CONTAINER_ENV} to the source .vindex3");
        return;
    };
    let Some(bands) = bands() else {
        eprintln!("skipped: set {MAP_ENV} to a band spec like 20-22:Q8_0");
        return;
    };
    let out_dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("kimi-kda-candidate.vindex3"));
    std::fs::create_dir_all(out_dir.join("segments")).expect("out dir");
    let bank = out_dir.join("segments").join(format!("{OBJECT}.bin"));

    let layers: Vec<u32> = bands.iter().flat_map(|((lo, hi), _)| *lo..=*hi).collect();
    let (source, tensors) =
        SegmentSource::open_object(&container, "target.decoder_stack", &layers, |name| {
            KDA_PROJECTIONS
                .iter()
                .any(|p| name.ends_with(&format!("self_attn.{p}.weight")))
        })
        .expect("decoder stack opens");
    assert_eq!(
        tensors.len(),
        layers.len() * 4,
        "each KDA layer must hold exactly q/k/v/o_proj — an MLA layer in the band \
         breaks this count (MLA ships no separate k/v_proj)"
    );

    let name = format!(
        "kimi-kda-{}",
        bands
            .iter()
            .map(|((lo, hi), enc)| {
                let tag = enc.to_lowercase().replace('_', "");
                if lo == hi {
                    format!("l{lo}{tag}")
                } else {
                    format!("l{lo}-{hi}{tag}")
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    );
    let map = PrecisionMap {
        name: name.clone(),
        encoding: "BF16".into(),
        roles: vec!["decoder-linear".into()],
        exceptions: bands
            .iter()
            .map(|((lo, hi), enc)| Exception {
                projection: None,
                layers: Some((*lo, *hi)),
                encoding: Some(enc.clone()),
            })
            .collect(),
    };
    let index_path = out_dir.join("index.json");
    let mut index = std::fs::read(&index_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<CandidateIndex>(&b).ok())
        .filter(|i: &CandidateIndex| i.map.name == map.name)
        .unwrap_or_else(|| {
            CandidateIndex::new(
                "Kimi-Linear-48B-A3B-Instruct",
                source_dependency(&container).expect("source index"),
                OBJECT,
                map,
            )
        });

    let start = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let outcome = compile_kda_bank(
        &source,
        &tensors,
        OBJECT,
        &bank,
        Some((&index_path, 8)),
        &mut index,
        &mut |o| {
            if last.elapsed().as_secs() >= 5 {
                eprintln!(
                    "[kda-compile]   {} sealed, {} resumed, {:.2} GB written",
                    o.sealed,
                    o.resumed,
                    o.bytes_written as f64 / 1e9
                );
                last = std::time::Instant::now();
            }
        },
    )
    .expect("compiles");
    let _ =
        crate::format::vindex3::represent::compiler::write_index_atomically(&index, &index_path);
    let on_disk = std::fs::metadata(&bank).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[kda-compile] {name}: {} sealed, {} resumed; {:.3} GB in {:.1}s; bank file {:.3} GB",
        outcome.sealed,
        outcome.resumed,
        outcome.bytes_written as f64 / 1e9,
        start.elapsed().as_secs_f64(),
        on_disk as f64 / 1e9,
    );
}
