//! CONTINUATION-CODEC-3 W2's entry point: the frozen arms on gemma3-4b-it
//! over bank 3, resumably, under four authorities (code, binary,
//! containers with what each bound, bank). Built in W1; run in W2 from the
//! merged W1 SHA.

use std::path::Path;

use super::codec_1_c3::{file_sha256, yardstick_guard, YARDSTICK_ENCODING};
use super::codec_2::full_precision_guard;
use super::codec_2_arms::Ladder;
use super::codec_2_store::Checkpoints;
use super::codec_3_freeze::{bank3, freeze, Binding, BANK_IDS_SHA256};
use super::codec_3_freeze_arms::{record, run_arms};
use super::*;

const RECORD: &str = "codec3-gemma3-4b.json";

fn authorities(exact: (&str, &Subject), yard: (&str, &Subject)) -> Value {
    let binary = std::env::current_exe().expect("the running test binary");
    let container = |(dir, subject): (&str, &Subject)| {
        json!({
            "dir_name": Path::new(dir).file_name().map(|n| n.to_string_lossy().into_owned()),
            "index_json_sha256": file_sha256(&Path::new(dir).join("index.json")),
            "bound": format!("{:?}", subject.store.selection()),
        })
    };
    json!({
        "implementation_git_sha": git_sha(),
        "binary_sha256": file_sha256(&binary),
        "containers": {"exact": container(exact), "yardstick": container(yard)},
        "token_bank_ids_sha256": BANK_IDS_SHA256,
    })
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

/// CODEC-3 W2 on gemma3-4b-it: containers from LARQL_CODEC3_F32 and
/// LARQL_CODEC3_Q4K; checkpoints and the record in LARQL_CODEC3_RUN_DIR.
#[test]
#[ignore = "real containers: LARQL_CODEC3_F32 + LARQL_CODEC3_Q4K + LARQL_CODEC3_RUN_DIR (CODEC-3 W2; long, resumable)"]
fn real_codec_3_arms_gemma3_4b() {
    let _serial = serial();
    let freeze = freeze();
    let binding = Binding::from_freeze(&freeze).unwrap_or_else(|e| panic!("{e}"));
    let ids = bank3(&freeze).expect("the frozen bank 3");
    let (f32_dir, q4k_dir) = (env("LARQL_CODEC3_F32"), env("LARQL_CODEC3_Q4K"));
    let exact = subjects::open(Path::new(&f32_dir), "gemma3-4b-it");
    let yard = subjects::open_bound(
        Path::new(&q4k_dir),
        "gemma3-4b-it.q4k",
        Some(YARDSTICK_ENCODING),
    );
    full_precision_guard(exact.store.selection()).unwrap_or_else(|stop| panic!("{stop}"));
    yardstick_guard(yard.store.selection()).unwrap_or_else(|stop| panic!("{stop}"));
    let stamp = authorities((&f32_dir, &exact), (&q4k_dir, &yard));
    let store = Checkpoints::open(Path::new(&env("LARQL_CODEC3_RUN_DIR")), &stamp);
    let backend = ProductionBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let ladder = Ladder::frozen();
    run_arms(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids,
        &ladder,
        &store,
    );
    let record = record(&stamp, &binding, &store, &ladder);
    store.write(RECORD, &record);
    eprintln!(
        "wrote {}; verdict ({}) {}",
        store.path(RECORD).display(),
        binding.primary,
        record["adjudication"]["verdict"]["verdict"]
    );
}
