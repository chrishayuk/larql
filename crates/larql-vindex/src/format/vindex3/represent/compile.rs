//! **Compiling a persistent, execution-shaped physical representation.**
//!
//! The output of REPRESENT is not benchmark debris. It is the artifact
//! you intend to keep:
//!
//! ```text
//! Kimi-Linear.vindex3            source authority, BF16
//!         │  REPRESENT candidate scopes
//!         ▼
//! Kimi-Linear-q6-candidate.vindex3
//!         ├── source semantic graph + provenance
//!         ├── BF16 protected operands
//!         ├── Q6_K expert banks IN EXECUTION LAYOUT
//!         └── RepresentationExperiment evidence
//! ```
//!
//! Those Q6 edges are **candidate** physical representations. They
//! become `SELECTED_REPRESENTATION` only when a quality bank puts them
//! through a named gate and [`super::selection::promote`] says so — and
//! if it does, no rebuild is needed, because these are the exact bytes
//! that were measured.
//!
//! ## Disk layout IS the execution layout
//!
//! The source container stores experts the way the checkpoint did — one
//! tensor per `(layer, expert, projection)`. The grouped Metal kernel
//! wants the opposite: per layer, three CONTIGUOUS banks with a byte
//! offset table, so expert identity travels as `identity → row range →
//! output slice`. Compiling into that shape means
//!
//! ```text
//! disk representation = mmap representation = GPU execution representation
//! ```
//!
//! with no gather at load. That matters more as models grow: a
//! source-oriented layout plus a runtime-transformed copy is two
//! hundreds-of-gigabytes artifacts where one will do.
//!
//! ## Sealing, so a long compilation is resumable
//!
//! ~95 GB of expert weight does not compile in one operation that may
//! not be interrupted. Each operand is sealed independently with the
//! hash of the bytes it was compiled FROM, so a resumed run skips what
//! is already done — and re-does anything whose source has changed
//! underneath it, which a "already present" check would miss.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VindexError;

/// Where one expert's payload sits inside a compiled projection bank.
///
/// Deterministic from the geometry, so the layout can be computed
/// before a single byte is encoded — which is what lets a resumed run
/// write into the right place without replaying what came before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BankSlot {
    pub expert: u32,
    pub offset: u64,
    pub len: u64,
}

/// One layer's three projection banks, in the shape the grouped kernel
/// reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerBankLayout {
    pub layer: u32,
    pub encoding: String,
    /// Bytes one expert occupies in a gate/up bank.
    pub gate_up_stride: u64,
    /// Bytes one expert occupies in a down bank.
    pub down_stride: u64,
    pub experts: u32,
}

impl LayerBankLayout {
    /// Bytes an `[n, k]` matrix occupies in `encoding`.
    pub fn matrix_bytes(encoding: &str, n: usize, k: usize) -> Result<u64, VindexError> {
        let elems = n * k;
        match encoding {
            "BF16" => Ok(elems as u64 * 2),
            "Q6_K" | "Q4_K" => {
                if !k.is_multiple_of(256) {
                    return Err(VindexError::Parse(format!(
                        "k={k} is not a whole number of 256-element superblocks, so rows \
                         would share a scale in {encoding}"
                    )));
                }
                let per_sb = if encoding == "Q6_K" { 210 } else { 144 };
                Ok((elems / 256) as u64 * per_sb)
            }
            "Q8_0" => {
                if !k.is_multiple_of(32) {
                    return Err(VindexError::Parse(format!(
                        "k={k} is not a whole number of 32-element blocks, so rows \
                         would share a scale in Q8_0"
                    )));
                }
                Ok((elems / 32) as u64 * 34)
            }
            other => Err(VindexError::Parse(format!(
                "no compiled layout for encoding `{other}`"
            ))),
        }
    }

    pub fn new(
        layer: u32,
        encoding: &str,
        experts: u32,
        hidden: usize,
        inter: usize,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            layer,
            encoding: encoding.to_string(),
            gate_up_stride: Self::matrix_bytes(encoding, inter, hidden)?,
            down_stride: Self::matrix_bytes(encoding, hidden, inter)?,
            experts,
        })
    }

    /// Where `expert` lands in the named projection's bank.
    ///
    /// Experts are laid out at their OWN index, not at a position in a
    /// resident subset. A compiled bank holds every expert, so the
    /// offset table is a pure function of identity and a route to any
    /// expert resolves — including one no baseline ever selected.
    pub fn slot(&self, projection: &str, expert: u32) -> Result<BankSlot, VindexError> {
        if expert >= self.experts {
            return Err(VindexError::Parse(format!(
                "expert {expert} is outside layer {}'s {} experts",
                self.layer, self.experts
            )));
        }
        // Both vocabularies: the checkpoint calls Kimi's expert
        // projections `w1`/`w2`/`w3`, and that is what appears in tensor
        // names and therefore in a precision map's exceptions. The
        // canonical spellings are accepted too so a map written against
        // another family still resolves.
        let stride = match projection {
            "w1" | "w3" | "gate_proj" | "up_proj" => self.gate_up_stride,
            "w2" | "down_proj" => self.down_stride,
            other => {
                return Err(VindexError::Parse(format!(
                    "`{other}` is not an expert projection"
                )))
            }
        };
        Ok(BankSlot {
            expert,
            offset: u64::from(expert) * stride,
            len: stride,
        })
    }

    /// Total bytes of one projection bank.
    pub fn bank_bytes(&self, projection: &str) -> Result<u64, VindexError> {
        let last = self.slot(projection, self.experts - 1)?;
        Ok(last.offset + last.len)
    }

    /// Total bytes of this layer's THREE projection banks — the
    /// layer's whole extent in a candidate segment.
    pub fn layer_bytes(&self) -> Result<u64, VindexError> {
        Ok(2 * self.bank_bytes("w1")? + self.bank_bytes("w2")?)
    }
}

/// Where each compiled layer's bank begins in a candidate segment that
/// holds SEVERAL layers — the composed-map extension of `bank_base`.
///
/// Layers are placed ascending by index; each layer's extent is its
/// OWN encoding's three projection banks, so one candidate can hold
/// L20..=25 at Q8_0 beside L26 at Q6_K. A single-layer placement puts
/// its layer at base 0, byte-identical to the layout every existing
/// candidate was compiled with — this is an extension, never a
/// migration.
///
/// ONE definition, used by the compiler, the completeness verifier and
/// the loader, for the same reason `bank_base` is `pub(crate)` in one
/// place: two derivations of the same offset WILL drift.
#[derive(Debug, Clone)]
pub struct CandidatePlacement {
    /// `(layout, base)` per layer, ascending by layer index.
    placed: Vec<(LayerBankLayout, u64)>,
}

impl CandidatePlacement {
    /// Resolve placement for `layers` under `map`.
    ///
    /// Each layer's encoding is the map's OWN answer for that layer's
    /// operands (`PrecisionMap::resolve` over the three projections).
    /// A layer whose compiled projections resolve to DIFFERENT
    /// encodings is refused by name — per-projection precision within
    /// one layer's identity-addressed bank is not placeable, because
    /// identity addressing carries one stride. A layer the map does
    /// not compile at all is refused too: placing it would reserve
    /// bytes no seal will ever cover.
    pub fn resolve(
        map: &super::map::PrecisionMap,
        role: super::policy::Role,
        layers: &[u32],
        experts: u32,
        hidden: usize,
        inter: usize,
    ) -> Result<Self, VindexError> {
        let mut sorted: Vec<u32> = layers.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.is_empty() {
            return Err(VindexError::Parse(
                "a candidate placement over no layers places nothing".into(),
            ));
        }
        let mut placed = Vec::with_capacity(sorted.len());
        let mut base = 0u64;
        for layer in sorted {
            let mut encodings: Vec<(&str, String)> = Vec::new();
            for proj in ["w1", "w3", "w2"] {
                let tensor = format!("{layer}.block_sparse_moe.experts.0.{proj}.weight");
                if let super::map::Precision::Compiled(enc) = map.resolve(role, &tensor) {
                    encodings.push((proj, enc.to_string()));
                }
            }
            let Some((_, encoding)) = encodings.first() else {
                return Err(VindexError::Parse(format!(
                    "layer {layer} is in the placement but the map `{}` compiles none of \
                     its projections",
                    map.name
                )));
            };
            if let Some((proj, other)) = encodings.iter().find(|(_, e)| e != encoding) {
                return Err(VindexError::Parse(format!(
                    "layer {layer}: `{}` resolves to {encoding} but `{proj}` to {other} — \
                     an identity-addressed layer bank carries ONE encoding; per-projection \
                     precision within a layer is not placeable",
                    encodings[0].0
                )));
            }
            let layout = LayerBankLayout::new(layer, encoding, experts, hidden, inter)?;
            let extent = layout.layer_bytes()?;
            placed.push((layout, base));
            base = base
                .checked_add(extent)
                .ok_or_else(|| VindexError::Parse("candidate placement overflows u64".into()))?;
        }
        Ok(Self { placed })
    }

    /// The layers this placement covers, ascending.
    pub fn layers(&self) -> impl Iterator<Item = u32> + '_ {
        self.placed.iter().map(|(l, _)| l.layer)
    }

    /// This layer's layout, or a refusal naming the layer.
    pub fn layout(&self, layer: u32) -> Result<&LayerBankLayout, VindexError> {
        self.placed
            .iter()
            .find(|(l, _)| l.layer == layer)
            .map(|(l, _)| l)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "layer {layer} is not in this candidate's placement"
                ))
            })
    }

    /// Where this layer's bank begins in the segment.
    pub fn layer_base(&self, layer: u32) -> Result<u64, VindexError> {
        self.placed
            .iter()
            .find(|(l, _)| l.layer == layer)
            .map(|(_, b)| *b)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "layer {layer} is not in this candidate's placement"
                ))
            })
    }
}

/// What compiling one operand produced, and what it was compiled FROM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperandSeal {
    pub object: String,
    pub tensor: String,
    pub encoding: String,
    /// Hash of the SOURCE bytes. Resume compares against this, so an
    /// operand whose source changed is recompiled rather than trusted.
    pub source_hash: String,
    pub target_hash: String,
    pub target_offset: u64,
    pub target_len: u64,
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// The resumable record of a compilation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CompilationLedger {
    /// The precision map this compilation is executing.
    pub map_name: String,
    /// Sealed operands, keyed `object\u{1}tensor`.
    pub sealed: BTreeMap<String, OperandSeal>,
}

/// Why an operand still needs work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pending {
    /// Never compiled.
    Absent,
    /// Compiled, but from different source bytes than are there now.
    SourceChanged,
    /// Compiled, but into a different encoding than the map now asks for.
    EncodingChanged,
}

impl CompilationLedger {
    pub fn new(map_name: impl Into<String>) -> Self {
        Self {
            map_name: map_name.into(),
            sealed: BTreeMap::new(),
        }
    }

    fn key(object: &str, tensor: &str) -> String {
        format!("{object}\u{1}{tensor}")
    }

    /// Whether this operand needs compiling, and why.
    ///
    /// `None` means the seal covers exactly these source bytes at
    /// exactly this encoding, and a resumed run may skip it. Anything
    /// else recompiles: a seal is a claim about specific inputs, and
    /// trusting it when they changed is how a compiled artifact silently
    /// stops matching the model it claims to represent.
    pub fn pending(
        &self,
        object: &str,
        tensor: &str,
        source_hash: &str,
        encoding: &str,
    ) -> Option<Pending> {
        match self.sealed.get(&Self::key(object, tensor)) {
            None => Some(Pending::Absent),
            Some(s) if s.source_hash != source_hash => Some(Pending::SourceChanged),
            Some(s) if s.encoding != encoding => Some(Pending::EncodingChanged),
            Some(_) => None,
        }
    }

    pub fn seal(&mut self, seal: OperandSeal) {
        self.sealed
            .insert(Self::key(&seal.object, &seal.tensor), seal);
    }

    pub fn get(&self, object: &str, tensor: &str) -> Option<&OperandSeal> {
        self.sealed.get(&Self::key(object, tensor))
    }

    /// Compiled bytes so far — what a resumed run has already earned.
    pub fn compiled_bytes(&self) -> u64 {
        self.sealed.values().map(|s| s.target_len).sum()
    }

    /// Two seals must never claim the same region of the same bank.
    ///
    /// A resumed run writes at offsets computed from the layout rather
    /// than by appending, so an overlap would mean two operands
    /// silently overwriting each other — undetectable in the output and
    /// fatal to it.
    pub fn overlaps(&self) -> Vec<(String, String)> {
        let mut by_object: BTreeMap<&str, Vec<&OperandSeal>> = BTreeMap::new();
        for s in self.sealed.values() {
            by_object.entry(&s.object).or_default().push(s);
        }
        let mut clashes = Vec::new();
        for seals in by_object.values() {
            let mut sorted = seals.clone();
            sorted.sort_by_key(|s| s.target_offset);
            for w in sorted.windows(2) {
                if w[0].target_offset + w[0].target_len > w[1].target_offset {
                    clashes.push((w[0].tensor.clone(), w[1].tensor.clone()));
                }
            }
        }
        clashes
    }
}

#[cfg(test)]
#[path = "compile_tests.rs"]
mod tests;
