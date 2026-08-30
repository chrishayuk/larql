//! **The compiler driver: source operands → candidate physical VINDEX.**
//!
//! Driven entirely by a [`PrecisionMap`]. Nothing here knows which
//! layers or projections are under test — `layers: [1]` is today's map,
//! and `down_proj / layers 20..26` is tomorrow's, with no code change
//! between them. That is the whole reason the evidence and the policy
//! share one selector vocabulary.
//!
//! ## The bytes exist before they are authoritative
//!
//! What this produces is a CANDIDATE. Its index records both
//!
//! ```text
//! CAN_REPRESENT_AS        Q6_K      <- these bytes exist and are addressable
//! SELECTED_REPRESENTATION source    <- and are not yet what executes
//! ```
//!
//! and only [`super::selection::promote`], given a quality bank through
//! a named gate, may change the second. Keeping them apart is what stops
//! "we compiled it" from quietly becoming "we shipped it" — and it means
//! that when the evidence does arrive, **the exact bytes that were
//! measured become the selected bytes**, with no post-validation rebuild.

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::arena::{RepresentationArena, SourceOperands};
use super::compile::{
    hash_bytes, CandidatePlacement, CompilationLedger, LayerBankLayout, OperandSeal, Pending,
};
use super::map::{Precision, PrecisionMap};
use super::policy::{layer_of, projection_of, Role};
use super::selection::Promotion;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

/// One tensor the compiler was asked to consider.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

/// What one compilation pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompileOutcome {
    /// Encoded and sealed in this pass.
    pub sealed: usize,
    /// Already sealed against these exact source bytes — resume's whole
    /// purpose.
    pub resumed: usize,
    /// Left at source precision by the map. Not written at all: the
    /// candidate container carries only what it changes, and everything
    /// else resolves from the source container it names.
    pub source_precision: usize,
    pub bytes_written: u64,
}

/// What a source container IS, by content.
///
/// Three levels, because a sparse overlay depends on all three and they
/// can disagree independently:
///
/// * `manifest_hash` — the index: which objects exist, at which
///   encodings, in which segments.
/// * `graph_hash` — the semantic graph: operand identities, shapes,
///   roles. Two containers could carry byte-identical payload segments
///   under different semantic metadata, and an overlay composed against
///   one would be silently wrong on the other.
/// * `segments` — the payload bytes themselves.
///
/// Verifying only the payloads would close the narrowest of the three.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub manifest_hash: String,
    pub graph_hash: String,
    /// Segment name → the source index's own `payload_sha256`.
    pub segments: BTreeMap<String, String>,
}

/// Read a container's identity from its own metadata.
pub fn read_source_identity(container: &Path) -> Result<SourceIdentity, VindexError> {
    let index_bytes = std::fs::read(container.join("index.json"))?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| VindexError::Parse(format!("source index: {e}")))?;
    let graph_name = index["system_graph"]
        .as_str()
        .unwrap_or("system_graph.json");
    let graph_bytes = std::fs::read(container.join(graph_name))?;
    let mut segments = BTreeMap::new();
    if let Some(reps) = index["representations"].as_object() {
        for entry in reps.values() {
            if let (Some(seg), Some(hash)) =
                (entry["segment"].as_str(), entry["payload_sha256"].as_str())
            {
                segments.insert(seg.to_string(), hash.to_string());
            }
        }
    }
    Ok(SourceIdentity {
        manifest_hash: hash_bytes(&index_bytes),
        graph_hash: hash_bytes(&graph_bytes),
        segments,
    })
}

/// The source container a sparse candidate cannot execute without.
///
/// First-class and checked, because this artifact is an OVERLAY: it
/// holds only what its precision map compiled, and every other operand
/// still resolves from the source. Attaching it to a different container
/// — a re-export, another quantisation, a different revision — would
/// compose a hybrid nobody measured, silently, and the numbers would
/// look entirely reasonable.
///
/// **Identified by CONTENT, not by path.** A pathname is where the
/// container was last seen; it is not what the container is. Both files
/// must survive being moved to another disk, so the locator is a hint
/// for FINDING the source and never what verification resolves on.
///
/// The per-operand `source_hash` seals catch a changed operand during
/// COMPILATION. This catches a changed container at LOAD, which is the
/// other half and the one a resumed or shipped artifact needs.
///
/// A future standalone compiled VINDEX depends on nothing and omits this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDependency {
    pub identity: SourceIdentity,
    /// Where the source was when this was compiled. A hint, never the
    /// identity.
    pub locator_hint: String,
}

impl SourceDependency {
    /// Refuse a source that is not the one compiled against.
    ///
    /// Checked semantic-first: a manifest or graph mismatch is reported
    /// as such rather than as whichever segment happened to differ, so
    /// the reader learns the container is a different MODEL rather than
    /// hunting a byte difference.
    pub fn verify(&self, actual: &SourceIdentity) -> Result<(), VindexError> {
        if actual.manifest_hash != self.identity.manifest_hash {
            return Err(VindexError::Parse(format!(
                "source manifest is {} but this candidate was compiled against {} — a                  different container index, so its object/encoding layout may not match",
                short(&actual.manifest_hash),
                short(&self.identity.manifest_hash)
            )));
        }
        if actual.graph_hash != self.identity.graph_hash {
            return Err(VindexError::Parse(format!(
                "source semantic graph is {} but this candidate was compiled against {} —                  identical payloads under a different graph are still a different model",
                short(&actual.graph_hash),
                short(&self.identity.graph_hash)
            )));
        }
        for (segment, want) in &self.identity.segments {
            match actual.segments.get(segment) {
                Some(got) if got == want => {}
                Some(got) => {
                    return Err(VindexError::Parse(format!(
                        "source segment `{segment}` has payload {} but this candidate was                          compiled against {} — it is an overlay on a DIFFERENT container",
                        short(got),
                        short(want)
                    )))
                }
                None => {
                    return Err(VindexError::Parse(format!(
                        "source container is missing segment `{segment}`, which this candidate                          depends on for every operand it left at source precision"
                    )))
                }
            }
        }
        Ok(())
    }
}

fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}

/// A compiled candidate representation, and its standing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateIndex {
    pub model: String,
    /// The container these bytes were compiled FROM, and which still
    /// holds every operand this one left at source precision.
    pub source: SourceDependency,
    pub object: String,
    pub map: PrecisionMap,
    /// Encodings whose physical bytes exist here.
    pub can_represent_as: Vec<String>,
    /// What actually executes. `"source"` until a quality bank earns
    /// otherwise — see [`Self::apply_promotion`].
    pub selected_representation: String,
    pub ledger: CompilationLedger,
}

impl CandidateIndex {
    pub fn new(
        model: impl Into<String>,
        source: SourceDependency,
        object: impl Into<String>,
        map: PrecisionMap,
    ) -> Self {
        // Every encoding the map can put bytes in: the default plus
        // each exception's. A composed map (Q8_0 band + a Q6_K layer)
        // states both, so a promotion can select either honestly.
        let mut can_represent_as = vec![map.encoding.clone()];
        for e in &map.exceptions {
            if let Some(enc) = &e.encoding {
                if !can_represent_as.contains(enc) {
                    can_represent_as.push(enc.clone());
                }
            }
        }
        Self {
            model: model.into(),
            source,
            object: object.into(),
            ledger: CompilationLedger::new(map.name.clone()),
            map,
            can_represent_as,
            // Compiled bytes are not authority. Nothing but a promotion
            // moves this.
            selected_representation: "source".into(),
        }
    }

    /// Adopt a promotion's verdict.
    ///
    /// Refuses to select anything the promotion did not actually
    /// promote, so a caller cannot pass an empty `Promotion` and get a
    /// selection anyway.
    pub fn apply_promotion(&mut self, promotion: &Promotion) -> Result<(), VindexError> {
        let selected = promotion
            .verdicts
            .iter()
            .filter(|v| v.outcome.is_ok())
            .find(|v| self.can_represent_as.contains(&v.target));
        match selected {
            Some(v) => {
                self.selected_representation = v.target.clone();
                Ok(())
            }
            None => Err(VindexError::Parse(format!(
                "no promoted candidate matches this container's compiled encodings {:?} — \
                 the bytes stay compiled and unselected",
                self.can_represent_as
            ))),
        }
    }

    pub fn is_authoritative(&self) -> bool {
        self.selected_representation != "source"
    }
}

/// How a compilation run is parameterised.
pub struct CompileOptions<'a> {
    pub object: &'a str,
    pub role: Role,
    pub experts: u32,
    /// Where the execution-shaped bank is written.
    pub out: &'a Path,
    /// Where the index is persisted MID-RUN, and after how many seals.
    ///
    /// Without this the ledger only reaches disk when the run finishes,
    /// so a compile killed at 60 % resumes from nothing — the durability
    /// the ledger promises would exist only for runs that did not need
    /// it. Written to a temporary beside the target and renamed, so an
    /// interruption during the write cannot leave a half-parsed index.
    pub checkpoint: Option<(&'a Path, usize)>,
}

/// Compile every tensor the map puts in scope, writing at layout
/// offsets and sealing as it goes.
///
/// Idempotent and interruptible: a second call with the same ledger
/// re-does only what is unsealed or whose source moved.
/// `progress` is `&mut dyn` rather than `impl FnMut` deliberately.
/// This body is large, and a generic callback makes one full copy of it
/// per closure type at the call sites — which llvm-cov counts
/// separately, so every new caller with a different closure lowers the
/// file's measured coverage while covering strictly more of it. The
/// indirect call costs one dispatch per sealed operand, against
/// encoding and hashing a multi-megabyte block.
pub fn compile_expert_bank(
    source: &dyn SourceOperands,
    tensors: &[SourceTensor],
    opts: &CompileOptions<'_>,
    index: &mut CandidateIndex,
    progress: &mut dyn FnMut(&CompileOutcome),
) -> Result<CompileOutcome, VindexError> {
    let (object, role, experts, out) = (opts.object, opts.role, opts.experts, opts.out);
    let arena = RepresentationArena::new(index.map.clone());
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(out)?;
    let mut outcome = CompileOutcome::default();

    // The compiled layer set decides each layer's base in the segment,
    // so it is derived ONCE, from the whole scope this run was handed —
    // a per-tensor derivation would let two runs over different subsets
    // of one map write the same layer at two bases.
    let mut compiled_layers: Vec<u32> = Vec::new();
    let mut geometry: Option<(usize, usize)> = None;
    for t in tensors {
        // Geometry is a property of the family, not of the scope: a
        // w2-only map still needs the gate/up shape to place banks.
        if let Some(proj) = projection_of(&t.name) {
            if matches!(proj, "w1" | "w3" | "gate_proj" | "up_proj") {
                let [rows, cols] = t.shape[..] else {
                    // Refuse the malformed tensor BY NAME here rather
                    // than skipping it — a skip would surface later as
                    // "no gate/up tensor", a message that denies this
                    // tensor exists.
                    return Err(VindexError::Parse(format!(
                        "tensor `{}` is not a matrix: {:?}",
                        t.name, t.shape
                    )));
                };
                geometry = Some((cols, rows)); // (hidden, inter)
            }
        }
        if matches!(index.map.resolve(role, &t.name), Precision::Source) {
            continue;
        }
        if let Some(layer) = layer_of(&t.name) {
            compiled_layers.push(layer);
        }
    }
    compiled_layers.sort_unstable();
    compiled_layers.dedup();
    let placement = match (&compiled_layers[..], geometry) {
        ([], _) => None,
        (_, Some((hidden, inter))) => Some(CandidatePlacement::resolve(
            &index.map,
            role,
            &compiled_layers,
            experts,
            hidden,
            inter,
        )?),
        (_, None) => {
            return Err(VindexError::Parse(
                "the scope compiles operands but holds no gate/up tensor to take the \
                 bank geometry from"
                    .into(),
            ))
        }
    };

    for t in tensors {
        let encoding = match index.map.resolve(role, &t.name) {
            Precision::Source => {
                outcome.source_precision += 1;
                continue;
            }
            Precision::Compiled(enc) => enc.to_string(),
        };
        let (Some(layer), Some(projection)) = (layer_of(&t.name), projection_of(&t.name)) else {
            return Err(VindexError::Parse(format!(
                "tensor `{}` carries no layer/projection, so it has no place in an \
                 execution-shaped bank",
                t.name
            )));
        };
        let expert = expert_of(&t.name).ok_or_else(|| {
            VindexError::Parse(format!("tensor `{}` names no expert index", t.name))
        })?;
        let [rows, cols] = t.shape[..] else {
            return Err(VindexError::Parse(format!(
                "tensor `{}` is not a matrix: {:?}",
                t.name, t.shape
            )));
        };
        let placement = placement
            .as_ref()
            .expect("a compiled tensor implies a placement");
        // Strides come from the projection SHAPES, so gate/up and down
        // are distinguished by geometry rather than by trusting a name —
        // and they must agree with the placement's own derivation, or
        // this tensor is not the shape its layer's bank was placed for.
        let layout = LayerBankLayout::new(layer, &encoding, experts, cols, rows)?;
        let placed = placement.layout(layer)?;
        if *placed != layout {
            return Err(VindexError::Parse(format!(
                "tensor `{}`: shape/encoding disagree with the placement's layout for \
                 layer {layer} ({placed:?} vs {layout:?})",
                t.name
            )));
        }
        let slot = layout.slot(projection, expert)?;

        let operand = OperandRef {
            object: object.to_string(),
            tensor: t.name.clone(),
            dtype: String::new(),
            shape: t.shape.clone(),
        };
        let stored = source.load_stored(&operand)?;
        let source_hash = hash_bytes(&stored.bytes);
        // Layer base first (composed maps hold several layers), then
        // the per-projection bank base so w1/w2/w3 do not overlap.
        let offset = placement.layer_base(layer)? + bank_base(&layout, projection)? + slot.offset;

        // A seal written under a DIFFERENT placement — a rerun handed a
        // subset of the map's layers would recompute every base — must
        // refuse, not silently overwrite another layer's bytes.
        if let Some(seal) = index.ledger.get(object, &t.name) {
            if seal.target_offset != offset {
                return Err(VindexError::Parse(format!(
                    "tensor `{}` is sealed at offset {} but this run's placement puts it \
                     at {offset} — the scope differs from the one the ledger was written \
                     under; pass the map's full layer scope or start a fresh candidate",
                    t.name, seal.target_offset
                )));
            }
        }

        match index
            .ledger
            .pending(object, &t.name, &source_hash, &encoding)
        {
            None => {
                outcome.resumed += 1;
                progress(&outcome);
                continue;
            }
            Some(Pending::Absent | Pending::SourceChanged | Pending::EncodingChanged) => {}
        }

        let materialised = arena.resolve(source, role, &operand)?;
        if materialised.bytes.len() as u64 != slot.len {
            return Err(VindexError::Parse(format!(
                "tensor `{}`: encoder produced {} bytes, layout reserved {}",
                t.name,
                materialised.bytes.len(),
                slot.len
            )));
        }
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(&materialised.bytes)?;
        index.ledger.seal(OperandSeal {
            object: object.to_string(),
            tensor: t.name.clone(),
            encoding: encoding.clone(),
            source_hash,
            target_hash: hash_bytes(&materialised.bytes),
            target_offset: offset,
            target_len: slot.len,
        });
        outcome.sealed += 1;
        outcome.bytes_written += slot.len;
        if let Some((path, every)) = opts.checkpoint {
            if every > 0 && outcome.sealed.is_multiple_of(every) {
                file.flush()?;
                write_index_atomically(index, path)?;
            }
        }
        progress(&outcome);
    }
    file.flush()?;
    if let Some((path, _)) = opts.checkpoint {
        write_index_atomically(index, path)?;
    }
    Ok(outcome)
}

/// Persist the index without ever leaving a partial one behind.
///
/// Write-then-rename: a resumed run that read a truncated index would
/// silently recompile everything, which is the failure this whole
/// mechanism exists to avoid.
pub fn write_index_atomically(index: &CandidateIndex, path: &Path) -> Result<(), VindexError> {
    let tmp = path.with_extension("json.partial");
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(index)
            .map_err(|e| VindexError::Parse(format!("index does not serialise: {e}")))?,
    )?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Where a projection's bank starts, so the three do not overlap.
///
/// `pub(crate)` because a LOADER of the compiled bank must place its
/// views at exactly the offsets the compiler wrote to — one definition,
/// or the two drift.
pub(crate) fn bank_base(layout: &LayerBankLayout, projection: &str) -> Result<u64, VindexError> {
    let gate_up = layout.bank_bytes("w1")?;
    Ok(match projection {
        "w1" | "gate_proj" => 0,
        "w3" | "up_proj" => gate_up,
        "w2" | "down_proj" => 2 * gate_up,
        other => {
            return Err(VindexError::Parse(format!(
                "`{other}` is not an expert projection"
            )))
        }
    })
}

/// The expert index inside a name like
/// `1.block_sparse_moe.experts.137.w2.weight`.
pub fn expert_of(tensor: &str) -> Option<u32> {
    let parts: Vec<&str> = tensor.split('.').collect();
    let i = parts.iter().position(|p| *p == "experts")?;
    parts.get(i + 1)?.parse().ok()
}

#[cfg(test)]
#[path = "compiler_tests.rs"]
mod tests;
