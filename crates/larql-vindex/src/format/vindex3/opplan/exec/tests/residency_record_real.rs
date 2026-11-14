//! The residency record of a REAL container prepared on the production CPU backend, under each representation
//! source, written out for an independent verifier. Instrument evidence: it runs no prompt and judges nothing.
//! Ignored by default:
//!
//! ```sh
//! LARQL_V3_CONTAINER=~/chris-models/gemma3-4b-it.vindex3 \
//! LARQL_RESIDENCY_RECORD_DIR=/path/to/dir \
//!   cargo test --release -p larql-vindex --lib residency_record_real -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::residency_record::{self as record, RULE_Q8_LOAD};
use crate::format::vindex3::opplan::plan_component_ops;

const CONTAINER_VAR: &str = "LARQL_V3_CONTAINER";
const OUT_VAR: &str = "LARQL_RESIDENCY_RECORD_DIR";
const INDEX_JSON: &str = "index.json";

fn record_under(container: &std::path::Path, source: RepresentationSource) -> serde_json::Value {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open_for(container, &inspection, None, source).unwrap();
    store.record_residency();
    let backend = ProductionBackend::new();
    PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let operands = store.take_residency_record();

    let mut by_rule: BTreeMap<&str, usize> = BTreeMap::new();
    for operand in &operands {
        *by_rule.entry(operand.rule).or_default() += 1;
    }
    println!(
        "{}: {} matrix operands; by rule {by_rule:?}; {} counted as quantised at load",
        record::source_name(source),
        operands.len(),
        store.runtime_quantised()
    );
    assert_eq!(
        by_rule.get(RULE_Q8_LOAD).copied().unwrap_or(0) as u64,
        store.runtime_quantised(),
        "every load the store counted as manufactured is in the record, and no other"
    );
    let index_sha256: String = Sha256::digest(std::fs::read(container.join(INDEX_JSON)).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    record::document(&container.display().to_string(), &index_sha256, &operands)
}

#[test]
#[ignore = "needs a real container in LARQL_V3_CONTAINER"]
fn residency_record_real_under_each_source() {
    let container = std::path::PathBuf::from(
        std::env::var(CONTAINER_VAR).unwrap_or_else(|_| panic!("set {CONTAINER_VAR}")),
    );
    let out = std::env::var(OUT_VAR).ok().map(std::path::PathBuf::from);
    for source in [RepresentationSource::Stored, RepresentationSource::Auto] {
        let doc = record_under(&container, source);
        if source == RepresentationSource::Stored {
            assert!(
                doc["operands"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|o| o["manufactured"] == false),
                "under stored nothing is manufactured"
            );
        }
        if let Some(dir) = &out {
            let path = dir.join(format!("residency-{}.json", record::source_name(source)));
            std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
            println!("written {}", path.display());
        }
    }
}
