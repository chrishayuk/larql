//! A region's residency read keeps its outcome, and a mapped store names
//! its file. Pins the public contract `resident_bytes` has always had —
//! by its values, not by restating its definition — beside the new read.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::compile::{CompilationLedger, OperandSeal};
use super::super::physical::PhysicalStore;
#[cfg(not(unix))]
use super::super::physical::ResidencyReadError;

const STORE_ID: &str = "compiled-bank";
const OBJECT: &str = "target.decoder_stack";
const TENSOR: &str = "0.mlp.up_proj.weight";
/// Bytes before the operand, so the region's file offset is not zero.
const LEADING: usize = 8;
const OPERAND_LEN: usize = 16;

fn ledger() -> CompilationLedger {
    let seal = OperandSeal {
        object: OBJECT.to_string(),
        tensor: TENSOR.to_string(),
        encoding: "f32".to_string(),
        source_hash: String::new(),
        target_hash: String::new(),
        target_offset: LEADING as u64,
        target_len: OPERAND_LEN as u64,
    };
    CompilationLedger {
        map_name: "f32".to_string(),
        sealed: BTreeMap::from([(format!("{OBJECT}\u{1}{TENSOR}"), seal)]),
    }
}

fn mapped_store(dir: &std::path::Path) -> (std::path::PathBuf, Arc<PhysicalStore>, Vec<u8>) {
    let path = dir.join("bank.bin");
    let bytes: Vec<u8> = (0..(LEADING + OPERAND_LEN) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let store = Arc::new(PhysicalStore::map_compiled(STORE_ID, &path, &ledger()).unwrap());
    (path, store, bytes)
}

#[test]
fn a_mapped_store_names_its_file_and_an_owned_one_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let (path, store, _) = mapped_store(dir.path());
    assert_eq!(store.source_path(), Some(path.as_path()));
    let region = store.whole(TENSOR).unwrap();
    assert_eq!(region.source_path(), Some(path.as_path()));

    let owned = Arc::new(PhysicalStore::owned(
        STORE_ID,
        vec![0u8; 4],
        BTreeMap::new(),
    ));
    assert_eq!(owned.source_path(), None);
    assert_eq!(owned.span(0, 4).unwrap().source_path(), None);
}

#[test]
fn a_mapped_region_offset_is_a_file_offset() {
    let dir = tempfile::tempdir().unwrap();
    let (path, store, _) = mapped_store(dir.path());
    let region = store.whole(TENSOR).unwrap();
    assert_eq!(region.store_offset(), LEADING as u64);
    let file = std::fs::read(&path).unwrap();
    let at = region.store_offset() as usize;
    assert_eq!(region.bytes(), &file[at..at + region.len() as usize]);
}

#[cfg(unix)]
#[test]
fn a_touched_mapped_region_reads_resident_and_keeps_the_old_contract() {
    let dir = tempfile::tempdir().unwrap();
    let (_, store, bytes) = mapped_store(dir.path());
    let region = store.whole(TENSOR).unwrap();
    // Touch the region so its page is resident.
    assert_eq!(region.bytes(), &bytes[LEADING..]);
    let resident = region.residency().expect("a live mapping is readable");
    assert!(resident > 0 && resident <= region.len());
    // Clipped to the region, so a whole resident page reads as its length.
    assert_eq!(resident, OPERAND_LEN as u64);
    assert_eq!(region.resident_bytes(), Some(OPERAND_LEN as u64));
}

#[cfg(unix)]
#[test]
fn an_empty_region_reads_zero() {
    let store = Arc::new(PhysicalStore::owned(
        STORE_ID,
        vec![0u8; 4],
        BTreeMap::new(),
    ));
    let empty = store.span(0, 0).unwrap();
    assert_eq!(empty.residency(), Ok(0));
    assert_eq!(empty.resident_bytes(), Some(0));
}

#[cfg(not(unix))]
#[test]
fn without_a_residency_report_every_region_is_unsupported_even_an_empty_one() {
    let store = Arc::new(PhysicalStore::owned(
        STORE_ID,
        vec![0u8; 4],
        BTreeMap::new(),
    ));
    for region in [store.span(0, 0).unwrap(), store.span(0, 4).unwrap()] {
        assert_eq!(region.residency(), Err(ResidencyReadError::Unsupported));
        assert_eq!(region.resident_bytes(), None);
    }
}
