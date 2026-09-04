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
/// Which layer to compile. Parameterised because the SCOPE is the
/// experiment: `layer 1 / all projections` is one candidate and
/// `layer 26 / w2 only` is another, and the compiler is generic over
/// the map precisely so neither needs new code.
const LAYER_ENV: &str = "LARQL_Q6_LAYER";
/// Which projection, in the checkpoint's own spelling (`w1` gate, `w3`
/// up, `w2` down). Unset compiles all three.
const PROJECTION_ENV: &str = "LARQL_Q6_PROJECTION";
/// Which encoding to compile the scoped operands into. Unset is Q6_K —
/// the driver's historical default. The precision ladder sets `Q8_0`:
/// the depth sweep answered "how deep can Q6_K go" (only layer 26), so
/// the live question is what an ~8-bit representation admits, and that
/// is a different candidate under the SAME driver, not a new one.
const ENCODING_ENV: &str = "LARQL_Q6_ENCODING";

fn scope_layer() -> u32 {
    std::env::var(LAYER_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

/// Refuses an encoding no grouped kernel reads, loudly, before any
/// bytes are written — a candidate that compiles but cannot execute
/// would burn a bank run to discover a typo.
fn scope_encoding() -> String {
    let enc = std::env::var(ENCODING_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "Q6_K".into());
    assert!(
        super::physical::ExpertEncoding::parse(&enc).is_some(),
        "{ENCODING_ENV}={enc} names no encoding a grouped kernel reads"
    );
    enc
}

/// A COMPOSED map: `"20-25:Q8_0,26:Q6_K"` — inclusive layer bands, each
/// at its own encoding, one candidate. When set it overrides the
/// single-layer/encoding envs, because a composed map IS the scope.
const MAP_ENV: &str = "LARQL_Q6_MAP";

fn scope_composed() -> Option<Vec<((u32, u32), String)>> {
    let spec = std::env::var(MAP_ENV).ok().filter(|v| !v.is_empty())?;
    let bands = spec
        .split(',')
        .map(|band| {
            let (range, enc) = band
                .split_once(':')
                .unwrap_or_else(|| panic!("{MAP_ENV}: `{band}` is not `LAYERS:ENCODING`"));
            let (lo, hi) = match range.split_once('-') {
                Some((a, b)) => (
                    a.trim().parse().expect("band start"),
                    b.trim().parse().expect("band end"),
                ),
                None => {
                    let l = range.trim().parse().expect("band layer");
                    (l, l)
                }
            };
            assert!(lo <= hi, "{MAP_ENV}: band `{band}` is inverted");
            assert!(
                super::physical::ExpertEncoding::parse(enc.trim()).is_some(),
                "{MAP_ENV}: `{enc}` names no encoding a grouped kernel reads"
            );
            ((lo, hi), enc.trim().to_string())
        })
        .collect::<Vec<_>>();
    // Bands must not overlap — one layer, one encoding.
    for (i, ((lo_a, hi_a), _)) in bands.iter().enumerate() {
        for ((lo_b, hi_b), _) in bands.iter().skip(i + 1) {
            assert!(
                hi_a < lo_b || hi_b < lo_a,
                "{MAP_ENV}: bands overlap at layers {}..={} vs {}..={}",
                lo_a,
                hi_a,
                lo_b,
                hi_b
            );
        }
    }
    Some(bands)
}

/// Every layer a composed spec names, ascending.
fn composed_layers(bands: &[((u32, u32), String)]) -> Vec<u32> {
    let mut layers: Vec<u32> = bands.iter().flat_map(|((lo, hi), _)| *lo..=*hi).collect();
    layers.sort_unstable();
    layers
}

/// The composed precision map: one exception per band, catch-all source.
fn composed_map(bands: &[((u32, u32), String)]) -> PrecisionMap {
    let name = format!(
        "kimi-map-{}",
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
    PrecisionMap {
        name,
        encoding: bands[0].1.clone(),
        roles: vec!["expert-weight".into()],
        exceptions: bands
            .iter()
            .map(|((lo, hi), enc)| Exception {
                projection: None,
                layers: Some((*lo, *hi)),
                encoding: Some(enc.clone()),
            })
            .chain([Exception {
                projection: None,
                layers: None,
                encoding: None,
            }])
            .collect(),
    }
}

/// One or more projections, comma-separated. Empty compiles all three.
///
/// A LIST because the interesting candidate is rarely one projection:
/// "gate and up at Q6, down protected" is a single precision map, and
/// the sweep that motivated per-projection scoping needs to express it.
fn scope_projections() -> Vec<String> {
    std::env::var(PROJECTION_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.split(',').map(|p| p.trim().to_string()).collect())
        .unwrap_or_default()
}

/// Reads operands straight out of a segment file.
pub(super) struct SegmentSource {
    path: PathBuf,
    payload_start: u64,
    offsets: std::collections::BTreeMap<String, (u64, u64)>,
}

impl SegmentSource {
    fn open(
        container: &std::path::Path,
        layers: &[u32],
    ) -> Result<(Self, Vec<SourceTensor>), VindexError> {
        Self::open_object(container, OBJECT, layers, |_| true)
    }

    /// The same reader over any segment, with a NAME filter — the KDA
    /// candidate reads `target.decoder_stack`, whose tensors are mostly
    /// not in any candidate's scope.
    pub(super) fn open_object(
        container: &std::path::Path,
        object: &str,
        layers: &[u32],
        keep: impl Fn(&str) -> bool,
    ) -> Result<(Self, Vec<SourceTensor>), VindexError> {
        let path = container.join("segments").join(format!("{object}.bin"));
        let (header, payload_start) =
            crate::format::vindex3::encode::segment::read_segment_header(&path)?;
        let mut offsets = std::collections::BTreeMap::new();
        let mut tensors = Vec::new();
        for t in header.tensors {
            if crate::format::vindex3::represent::policy::layer_of(&t.name)
                .is_some_and(|l| layers.contains(&l))
                && keep(&t.name)
            {
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

/// The precision map under test: one layer's expert weights at Q6_K —
/// optionally only SOME projections of them — everything else at
/// source precision.
///
/// The projection selector is what turns "Q6 on this layer failed" into
/// "which part of this layer's FFN is the sensitive one": gate and up
/// feed the nonlinear activation, down writes back to the residual
/// stream, and they need not be equally safe to quantise.
fn layer_q6(layer: u32, projections: &[String], encoding: &str) -> PrecisionMap {
    // "Q6_K" -> "q6k", "Q8_0" -> "q80": the map's name carries its
    // encoding, so two candidates for one layer never collide.
    let tag = encoding.to_lowercase().replace('_', "");
    let name = if projections.is_empty() {
        format!("kimi-expertweight-layer{layer}-{tag}")
    } else {
        format!(
            "kimi-expertweight-layer{layer}-{}-{tag}",
            projections.join("+")
        )
    };
    // One exception per scoped projection; none means the whole layer.
    let scoped: Vec<Exception> = if projections.is_empty() {
        vec![Exception {
            projection: None,
            layers: Some((layer, layer)),
            encoding: Some(encoding.into()),
        }]
    } else {
        projections
            .iter()
            .map(|p| Exception {
                projection: Some(p.clone()),
                layers: Some((layer, layer)),
                encoding: Some(encoding.into()),
            })
            .collect()
    };
    PrecisionMap {
        name,
        encoding: encoding.into(),
        roles: vec!["expert-weight".into()],
        exceptions: scoped
            .into_iter()
            // Everything outside the scope stays where it is. Written
            // explicitly rather than relying on the role not being
            // named, so the map states its own boundary.
            .chain([Exception {
                projection: None,
                layers: None,
                encoding: None,
            }])
            .collect(),
    }
}

/// The source's own identity, plus where it was found.
pub(super) fn source_dependency(
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
    let projections = scope_projections();
    // A composed spec is one candidate over several layer bands; the
    // single-layer envs are the degenerate one-band form of the same
    // thing. The map NAME differs (layer_q6's spelling is what existing
    // candidates resume under), so composition is opt-in via the env,
    // never a silent rename.
    let composed = scope_composed();
    let bands = composed
        .clone()
        .unwrap_or_else(|| vec![((scope_layer(), scope_layer()), scope_encoding())]);
    let layers = composed_layers(&bands);
    if composed.is_some() {
        assert!(
            projections.is_empty(),
            "a composed map compiles whole layers; {PROJECTION_ENV} does not apply"
        );
    }
    let out_dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("kimi-q6-candidate.vindex3"));
    std::fs::create_dir_all(out_dir.join("segments")).expect("out dir");
    let bank = out_dir.join("segments").join(format!("{OBJECT}.bin"));

    let (source, tensors) = SegmentSource::open(&container, &layers).expect("source segment");
    eprintln!(
        "[compile] layers {layers:?}{}: {} source tensors from {}",
        if projections.is_empty() {
            String::new()
        } else {
            format!(" / {} only", projections.join("+"))
        },
        tensors.len(),
        container.display()
    );
    assert_eq!(
        tensors.len(),
        layers.len() * (EXPERTS * 3) as usize,
        "each of layers {layers:?} must hold 3 projections for each of {EXPERTS} experts"
    );

    let index_path = out_dir.join("index.json");
    let map = match &composed {
        Some(bands) => composed_map(bands),
        None => layer_q6(layers[0], &projections, &bands[0].1),
    };
    let map_name = map.name.clone();
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
        &mut |o| {
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
        "[compile] {}: {} sealed, {} resumed, {} left at source; \
         {:.2} GB in {secs:.1}s ({:.0} MB/s); bank file {:.2} GB",
        map_name,
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
        index.source.identity.segments().len()
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
    altered.semantic.graph_hash = "0".repeat(64);
    assert!(
        index.source.verify(&altered).is_err(),
        "identical payloads under a different graph must still be refused"
    );

    // Every operand the scope covers is sealed; the rest were left at
    // source and never written. A projection-scoped map compiles a
    // THIRD of the layer, which is the point of scoping it.
    let per_layer_in_scope = if projections.is_empty() {
        (EXPERTS * 3) as usize
    } else {
        EXPERTS as usize * projections.len()
    };
    let in_scope = per_layer_in_scope * layers.len();
    assert_eq!(
        outcome.sealed + outcome.resumed,
        in_scope,
        "every in-scope operand of layers {layers:?} must be sealed or already sealed"
    );
    assert_eq!(
        outcome.source_precision,
        layers.len() * (EXPERTS * 3) as usize - in_scope,
        "everything outside the scope stays at source precision, unwritten"
    );
    assert!(index.ledger.overlaps().is_empty(), "banks must not collide");
    assert!(
        !index.is_authoritative(),
        "no quality bank has run, so these bytes must NOT be selected"
    );
    // The compiled population, against the format's own geometry: every
    // projection of this layer is 1024 x 2304 elements (gate/up
    // [inter, hidden], down [hidden, inter] — same product), so the
    // ledger's byte count is EXACTLY the encoding's per-matrix size
    // times the in-scope operand count. Derived from the same
    // `matrix_bytes` the layout uses, so a drift here is a compiler
    // fault, never a stale constant.
    let expected = index.ledger.compiled_bytes();
    let want: u64 = bands
        .iter()
        .map(|((lo, hi), enc)| {
            let per_matrix = super::compile::LayerBankLayout::matrix_bytes(enc, 1024, 2304)
                .expect("the driver's geometry is encodable");
            per_matrix * per_layer_in_scope as u64 * u64::from(hi - lo + 1)
        })
        .sum();
    assert_eq!(
        expected,
        want,
        "`{}`{} should be exactly {:.3} GB, got {:.3} GB",
        map_name,
        if projections.is_empty() {
            String::new()
        } else {
            format!(" / {}", projections.join("+"))
        },
        want as f64 / 1e9,
        expected as f64 / 1e9
    );
}
