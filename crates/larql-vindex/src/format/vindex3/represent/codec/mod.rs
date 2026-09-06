//! **The representation/execution contract** — what every stored encoding
//! must declare before LARQL will plan, bind, or execute it.
//!
//! This is not a quantisation trait. It is extracted from the encodings
//! the container already carries — float tensor tables, the K-quants,
//! NVFP4, MXFP4, and the LYRW v2 banks that pair their scales — and its
//! test is that nothing about any of them is privileged in it. Six
//! things it gets right, each already paid for once in this tree:
//!
//! ```text
//! identity        an ABI family + revision the file names independently
//!                 of whichever implementation is registered
//! streams         an operand is a NAMED SET of streams, plus the
//!                 represented objects it may depend on — not one &[u8]
//! capabilities    access granularity, logical grouping, physical
//!                 alignment: declared, checked in preflight, refused by name
//! extents         depth, with a certificate per admissible prefix; every
//!                 codec today answers depth 0
//! decode          MANDATORY, range-aware, to canonical f32 — the universal
//!                 correctness surface; kernels are acceleration, never
//!                 semantics
//! realizations    residency is declared per REALIZATION: the decode
//!                 realization and each direct kernel carry their own
//!                 profile, so a fallback is a different realization with
//!                 a different declared cost, never a quiet substitution
//! ```
//!
//! Two rules follow. *Adding a codec requires proving representation
//! correctness; adding a kernel must not change it.* And a codec with no
//! direct realization is not a defect: it executes through the reference
//! path, flagged — the `representable-but-no-kernel` rung of spec §11,
//! made structural.
//!
//! The programme this opens — progressive, codebook-dependent and
//! entropy-coded representations as forcing tests of the abstraction —
//! is in `docs/represent-codec-contract.md`.

pub mod auxiliary;
pub mod capability;
pub mod codecs;
pub mod conformance;
pub mod error;
pub mod extent;
pub mod fidelity;
pub mod geometry;
pub mod registry;
pub mod residency;
pub mod streams;

#[cfg(test)]
mod tests;

use std::ops::Range;

pub use auxiliary::{admit_auxiliary_names, AuxiliaryMetadata, AuxiliarySpec};
pub use capability::{AccessGranularity, CodecCapabilities, RequiredAccess};
pub use error::CodecError;
pub use extent::{ExtentCertificate, RepresentationExtent};
pub use fidelity::{DomainId, FidelityCertificate, MetricId, SemanticId};
pub use geometry::RowGeometry;
pub use registry::CodecRegistry;
pub use residency::{Acceleration, AccelerationBackend, ResidencyClass, ResidencyProfile};
pub use streams::{AuxiliaryOperands, CodecOperands, NamedStreams, StreamRole, StreamSpec};

use super::nvfp4_pack::CodecIdentity;

/// One stored encoding's contract with the planner and the executor.
pub trait RepresentationCodec: Send + Sync {
    /// The label a container writes — the segment `dtype`, the graph's
    /// `Representation.encoding`, a precision map's `encoding`.
    fn encoding_label(&self) -> &'static str;

    /// The decode ABI: family, revision, geometry. What `index.json`
    /// carries so the file names its contract independently of the
    /// implementation registered to serve it.
    fn identity(&self) -> CodecIdentity;

    /// The streams one encoded operand is made of, in declaration order.
    fn streams(&self) -> &'static [StreamSpec];

    /// What this encoding can be asked for.
    fn capabilities(&self) -> CodecCapabilities;

    /// Every admissible extent, the base first. A terminal codec has one.
    fn extents(&self) -> Vec<ExtentCertificate>;

    /// Bytes a tensor of `shape` occupies at `extent`, or why it cannot.
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError>;

    /// Bind one contiguous payload — a tensor-table row — onto this
    /// codec's streams.
    ///
    /// The default serves every single-stream codec and refuses every
    /// other: a codec whose streams are stored apart cannot be handed one
    /// slice, which is the wall `QuantMatVec` hit and this trait exists
    /// to remove. A codec that packs several streams into one row with a
    /// derivable split overrides this.
    fn bind_packed<'a>(
        &self,
        payload: &'a [u8],
        shape: &[usize],
        tensor: &str,
    ) -> Result<NamedStreams<'a>, CodecError> {
        let _ = shape;
        match self.streams() {
            [only] => Ok(NamedStreams::single(*only, payload)),
            many => Err(CodecError::StreamsStoredApart {
                tensor: tensor.into(),
                label: self.encoding_label().into(),
                streams: many.iter().map(|s| s.name.to_string()).collect(),
            }),
        }
    }

    /// Refuse operands inconsistent with `shape` at `extent`.
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError>;

    /// **The mandatory realization.** Decode `rows` of a tensor of `shape`
    /// at `extent` into `dst`, which holds exactly `rows.len() * k` floats.
    ///
    /// Range-aware from the start, because a decode that can only answer
    /// "everything" makes the universal fallback unusable for a sparse
    /// plan and inexpressible for a progressive one.
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        extent: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError>;

    /// What the decode realization makes resident. Required, so an
    /// undeclared path is unrepresentable rather than discouraged.
    fn decode_residency(&self) -> ResidencyProfile;

    /// Direct realizations, each with its own declared residency.
    /// None is a legitimate answer: the codec executes through decode,
    /// flagged.
    fn accelerations(&self) -> Vec<Acceleration> {
        Vec::new()
    }

    // ── Provided ──────────────────────────────────────────────────────

    /// The deepest extent this codec declares — the one that reconstructs
    /// everything the representation holds.
    ///
    /// For a terminal codec that is depth 0 and the question is trivial;
    /// for a progressive one it is the exact extent, and a caller that
    /// wants full fidelity must ask rather than assume. Depth 0 is
    /// [`RepresentationExtent::BASE`] and means the opposite.
    fn terminal_extent(&self) -> RepresentationExtent {
        self.extents()
            .iter()
            .map(|c| c.extent)
            .max()
            .unwrap_or(RepresentationExtent::BASE)
    }

    /// The other represented objects this codec needs to decode at
    /// `extent`, by the names a container's reference table keys on.
    ///
    /// Asked PER EXTENT, so a codec whose deeper extents need a dependency
    /// its base does not can say exactly that. Empty for every codec that
    /// depends on nothing — which is every codec that only reads its own
    /// bytes, and the default.
    ///
    /// The names are part of this codec REVISION's semantics, like its
    /// stream names: a stored container's references are keyed by them, so
    /// renaming one changes what that container means and needs a revision
    /// bump. And a requirement is SEMANTIC — it says the decode needs the
    /// object to mean anything. Whether that object stays resident, or is
    /// touched while serving, is the selected realization's declaration.
    fn required_auxiliaries(&self, extent: RepresentationExtent) -> &'static [AuxiliarySpec] {
        let _ = extent;
        &[]
    }

    /// Judge one resolved auxiliary from the container's METADATA — its
    /// label, shape and codec identity — before any payload is opened.
    ///
    /// The codec owns this check because only it knows what its dependency
    /// has to be. The default refuses: a codec that requires an object and
    /// judges nothing about it would accept any tensor whose name happened
    /// to be in the table, which is the failure the reference table exists
    /// to prevent. Codecs requiring nothing never reach it.
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let _ = (target, shape, extent);
        Err(CodecError::AuxiliaryUnjudged {
            tensor: tensor.into(),
            label: self.encoding_label().into(),
            name: name.into(),
        })
    }

    /// The certificate a caller may rely on at `extent`, GIVEN what this
    /// codec's dependencies certify.
    ///
    /// A representation that decodes through another one inherits its
    /// error: a codebook read at a shallow extent moves every value that
    /// indexes it. So the bound a planner evaluates a quality floor
    /// against is the COMPOSED one — this codec's own radius widened by
    /// each dependency's, in the shared metric over the shared domain —
    /// and never the two read side by side.
    ///
    /// It is UNAVAILABLE, not optimistic, whenever a term is missing:
    ///
    /// * this codec declares no radius of its own. For a vector quantiser
    ///   that is the normal case rather than an omission — the assignment
    ///   error is a fact about the artifact and the encoder that made it,
    ///   and a decoder holding codes and a codebook cannot derive what the
    ///   original tensor was. Such a codec cannot satisfy a quality floor
    ///   at all until something else attests its instance's error;
    /// * a dependency certifies nothing, so there is nothing to add;
    /// * the terms differ, which [`FidelityCertificate::widened_by`]
    ///   refuses by name.
    ///
    /// `auxiliaries` is keyed by the names this codec declared, and each
    /// certificate is the one its dependency's SELECTED extent carries —
    /// the caller resolves that, because the caller is what pinned it.
    fn composed_certificate(
        &self,
        extent: RepresentationExtent,
        auxiliaries: &std::collections::BTreeMap<String, FidelityCertificate>,
        tensor: &str,
    ) -> Result<FidelityCertificate, CodecError> {
        let label = self.encoding_label();
        let mut composed = self.certificate_at(extent, tensor)?.radius.ok_or_else(|| {
            CodecError::CertificateUnavailable {
                tensor: tensor.into(),
                label: label.into(),
                depth: extent.depth,
                why: "it declares no radius of its own, so there is nothing for a dependency's \
                      bound to widen"
                    .into(),
            }
        })?;
        for spec in self.required_auxiliaries(extent) {
            let certificate =
                auxiliaries
                    .get(spec.name)
                    .ok_or_else(|| CodecError::CertificateUnavailable {
                        tensor: tensor.into(),
                        label: label.into(),
                        depth: extent.depth,
                        why: format!("its dependency `{}` certifies nothing", spec.name),
                    })?;
            composed = composed.widened_by(certificate, tensor, label)?;
        }
        Ok(composed)
    }

    /// The streams `extent` reads, in declaration order.
    ///
    /// Every stream that is not a refinement, plus each refinement whose
    /// declared depth the extent reaches. The rule is the DECLARATION's,
    /// not any codec's, so a loader can open exactly the streams an extent
    /// needs without knowing what a particular stream means — and a
    /// refinement deeper than the extent is never opened at all.
    fn streams_at(
        &self,
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<Vec<StreamSpec>, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok(self
            .streams()
            .iter()
            .copied()
            .filter(|spec| match spec.role {
                StreamRole::Refinement { depth } => depth <= extent.depth,
                _ => true,
            })
            .collect())
    }

    /// The certificate for `extent`, or a refusal naming what is declared.
    fn certificate_at(
        &self,
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<ExtentCertificate, CodecError> {
        let extents = self.extents();
        extents
            .iter()
            .find(|c| c.extent == extent)
            .cloned()
            .ok_or_else(|| CodecError::ExtentUnavailable {
                tensor: tensor.into(),
                label: self.encoding_label().into(),
                depth: extent.depth,
                available: extents.len() as u32,
            })
    }

    /// Decode every row at `extent`.
    fn decode_all(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<Vec<f32>, CodecError> {
        let label = self.encoding_label();
        let geometry = RowGeometry::of(shape, label, tensor)?;
        let mut out = vec![0.0f32; geometry.elements(label, tensor)?];
        self.decode_rows(operands, shape, 0..geometry.rows, extent, &mut out, tensor)?;
        Ok(out)
    }

    /// Bind, validate and decode one contiguous payload — the path a
    /// tensor-table operand takes to f32.
    fn decode_packed(
        &self,
        payload: &[u8],
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<Vec<f32>, CodecError> {
        let operands = CodecOperands::from_streams(self.bind_packed(payload, shape, tensor)?);
        self.validate(&operands, shape, extent, tensor)?;
        self.decode_all(&operands, shape, extent, tensor)
    }
}

/// A codec is named by its label and its ABI in every diagnostic.
impl std::fmt::Debug for dyn RepresentationCodec + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let id = self.identity();
        write!(
            f,
            "codec {} ({} r{})",
            self.encoding_label(),
            id.family,
            id.revision
        )
    }
}
