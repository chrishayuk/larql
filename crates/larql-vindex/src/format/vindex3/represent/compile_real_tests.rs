//! **The layer-1 Q6_K candidate, compiled from the real container.**
//!
//! Gated on `LARQL_KIMI_VINDEX3` pointing at the 92 GB source. This is
//! the first end-to-end demonstration of the physical claim the whole
//! REPRESENT design rests on:
//!
//! ```text
//! source semantic operands -> REPRESENT -> execution-shaped Q6_K bank
//!                                       -> mmap straight into the
//!                                          grouped Metal kernel
//! ```
//!
//! No gather, no transient repack, no separate benchmark layout. Only
//! layer 1 is compiled, because `ExpertWeight / layer 1 / Q6_K` is the
//! only candidate with any evidence behind it — compiling all 26 would
//! be materialising a precision decision that has not earned selection
//! authority.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use super::arena::{SourceOperands, StoredOperand};
use super::compiler::{compile_expert_bank, CandidateIndex, CompileOptions, SourceTensor};
use super::map::{Exception, PrecisionMap};
use super::policy::Role;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

const CONTAINER_ENV: &str = "LARQL_KIMI_VINDEX3";
const OUT_ENV: &str = "LARQL_Q6_CANDIDATE_OUT";
const OBJECT: &str = "target.expert_bank";
const EXPERTS: u32 = 256;
const LAYER: u32 = 1;

/// Reads operands straight out of a segment file.
struct SegmentSource {
    path: PathBuf,
    payload_start: u64,
    offsets: std::collections::BTreeMap<String, (u64, u64)>,
}

impl SegmentSource {
    fn open(container: &std::path::Path) -> Result<(Self, Vec<SourceTensor>), VindexError> {
        let path = container.join("segments").join(format!("{OBJECT}.bin"));
        let (header, payload_start) =
            crate::format::vindex3::encode::segment::read_segment_header(&path)?;
        let mut offsets = std::collections::BTreeMap::new();
        let mut tensors = Vec::new();
        for t in header.tensors {
            if crate::format::vindex3::represent::policy::layer_of(&t.name) == Some(LAYER) {
                tensors.push(SourceTensor {
                    name: t.name.clone(),
                    shape: t.shape.to_vec(),
                });
            }
            offsets.insert(t.name, (t.offset, t.len));
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        Ok((
            Self {
                path,
                payload_start,
                offsets,
            },
            tensors,
        ))
    }
}

impl SourceOperands for SegmentSource {
    fn load_stored(&self, operand: &OperandRef) -> Result<StoredOperand, VindexError> {
        let (offset, len) = *self
            .offsets
            .get(&operand.tensor)
            .ok_or_else(|| VindexError::Parse(format!("no tensor `{}`", operand.tensor)))?;
        let mut f = std::fs::File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.payload_start + offset))?;
        let mut bytes = vec![0u8; len as usize];
        f.read_exact(&mut bytes)?;
        Ok(StoredOperand {
            dtype: "BF16".into(),
            bytes,
        })
    }
}

/// The precision map under test: layer 1's expert weights at Q6_K,
/// everything else at source precision.
fn layer_one_q6() -> PrecisionMap {
    PrecisionMap {
        name: "kimi-expertweight-layer1-q6k".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((LAYER, LAYER)),
                encoding: Some("Q6_K".into()),
            },
            // Everything outside the scope stays where it is. Written
            // explicitly rather than relying on the role not being
            // named, so the map states its own boundary.
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    }
}

/// The source's own identity, plus where it was found.
fn source_dependency(
    container: &std::path::Path,
) -> Result<super::compiler::SourceDependency, VindexError> {
    Ok(super::compiler::SourceDependency {
        identity: super::compiler::read_source_identity(container)?,
        locator_hint: container.to_string_lossy().into_owned(),
    })
}

#[test]
fn compile_the_layer_one_q6_candidate() {
    let Some(container) = std::env::var_os(CONTAINER_ENV).map(PathBuf::from) else {
        eprintln!("skipped: set {CONTAINER_ENV} to the source .vindex3");
        return;
    };
    let out_dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("kimi-q6-candidate.vindex3"));
    std::fs::create_dir_all(out_dir.join("segments")).expect("out dir");
    let bank = out_dir.join("segments").join(format!("{OBJECT}.bin"));

    let (source, tensors) = SegmentSource::open(&container).expect("source segment");
    eprintln!(
        "[compile] layer {LAYER}: {} source tensors from {}",
        tensors.len(),
        container.display()
    );
    assert_eq!(
        tensors.len(),
        (EXPERTS * 3) as usize,
        "layer {LAYER} must hold 3 projections for each of {EXPERTS} experts"
    );

    let index_path = out_dir.join("index.json");
    let mut index = std::fs::read(&index_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<CandidateIndex>(&b).ok())
        .unwrap_or_else(|| {
            CandidateIndex::new(
                "Kimi-Linear-48B-A3B-Instruct",
                source_dependency(&container).expect("source index"),
                OBJECT,
                layer_one_q6(),
            )
        });

    let start = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let outcome = compile_expert_bank(
        &source,
        &tensors,
        &CompileOptions {
            object: OBJECT,
            role: Role::ExpertWeight,
            experts: EXPERTS,
            out: &bank,
            // Durable every 64 seals: a compile killed part-way must
            // resume from what it had done, not from nothing.
            checkpoint: Some((&index_path, 64)),
        },
        &mut index,
        |o| {
            if last.elapsed().as_secs() >= 5 {
                eprintln!(
                    "[compile]   {} sealed, {} resumed, {:.2} GB written",
                    o.sealed,
                    o.resumed,
                    o.bytes_written as f64 / 1e9
                );
                last = std::time::Instant::now();
            }
        },
    )
    .expect("compiles");
    let secs = start.elapsed().as_secs_f64();

    let on_disk = std::fs::metadata(&bank).expect("stat").len();
    eprintln!(
        "[compile] layer {LAYER} Q6_K: {} sealed, {} resumed, {} left at source; \
         {:.2} GB in {secs:.1}s ({:.0} MB/s); bank file {:.2} GB",
        outcome.sealed,
        outcome.resumed,
        outcome.source_precision,
        outcome.bytes_written as f64 / 1e9,
        outcome.bytes_written as f64 / 1e6 / secs.max(1e-9),
        on_disk as f64 / 1e9,
    );
    eprintln!(
        "[compile] CAN_REPRESENT_AS {:?}; SELECTED_REPRESENTATION `{}` \
         (compiled bytes are not authority); depends on {} source segments",
        index.can_represent_as,
        index.selected_representation,
        index.source.identity.segments.len()
    );
    // The overlay must accept the container it was compiled against, and
    // only that one.
    let actual = super::compiler::read_source_identity(&container).expect("source identity");
    index.source.verify(&actual).expect("same source");
    // Identity is CONTENT: the same container found somewhere else still
    // verifies, while altered content does not.
    let mut relocated = index.source.clone();
    relocated.locator_hint = "/some/other/disk".into();
    relocated
        .verify(&actual)
        .expect("a moved container is still the same container");
    let mut altered = actual.clone();
    altered.graph_hash = "0".repeat(64);
    assert!(
        index.source.verify(&altered).is_err(),
        "identical payloads under a different graph must still be refused"
    );

    assert_eq!(
        outcome.sealed + outcome.resumed,
        (EXPERTS * 3) as usize,
        "every layer-{LAYER} operand must be sealed or already sealed"
    );
    assert!(index.ledger.overlaps().is_empty(), "banks must not collide");
    assert!(
        !index.is_authoritative(),
        "no quality bank has run, so these bytes must NOT be selected"
    );
    // The compiled population, against the format's own geometry.
    let expected = index.ledger.compiled_bytes();
    assert!(
        (1.4e9..1.7e9).contains(&(expected as f64)),
        "layer {LAYER} at Q6_K should be ~1.5 GB, got {:.2} GB",
        expected as f64 / 1e9
    );
}
