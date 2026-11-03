//! Loading, decoding and accounting operand bytes.

use super::super::super::OperandRef;
use crate::error::VindexError;
use crate::format::vindex3::auxiliary_references::OperandAddress;
use crate::format::vindex3::encode::segment::read_exact_at;
use crate::format::vindex3::represent::codec::{
    admit_auxiliary_names, AuxiliaryMetadata, CodecOperands, NamedStreams, RepresentationCodec,
    RepresentationExtent,
};

#[allow(unused_imports)]
use super::*;

impl OperandStore {
    /// Load one operand as f32 values — the mandatory decode realization
    /// of whichever codec the stored dtype names.
    ///
    /// One dispatch, through the codec registry, for every encoding a
    /// segment can hold: a float widens, a K-quant or an NVFP4 pack
    /// decodes through its own layout, and a dtype no codec is registered
    /// for is refused naming the ones that are. Consumers that want the
    /// compact form for a kernel take [`Self::load_raw`]; this is for the
    /// ones that need values, which on a recurrence is most of them.
    ///
    /// The decode is lossy in exactly the way the representation is:
    /// a pack's 4-bit values widened, not the checkpoint's originals.
    /// That is the point — it is what makes a compact representation
    /// measurable on a stack the device cannot run.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let raw = self.load_raw(operand)?;
        let codec = self.registry.resolve(&raw.dtype, &operand.tensor)?;
        // Everything the representation holds: this loader is asked for
        // values, not for a fidelity, so it asks the codec for its deepest
        // extent rather than assuming depth 0 is the whole of it.
        let extent = codec.terminal_extent();
        self.decode_at(operand, codec, raw, extent)
    }

    /// One operand's values at `extent` — the loader a caller with a PINNED
    /// extent uses, and the only path that reads fewer streams than the
    /// container holds.
    ///
    /// A shallower extent is not a smaller container: every stream the
    /// artifact holds is still on disk, and what changes is which of them
    /// are opened. So this reads the streams the extent needs and no
    /// others, which is a physical fact a test can check on the store's
    /// own read ledger rather than a claim about intent.
    pub fn load_at(
        &self,
        operand: &OperandRef,
        extent: RepresentationExtent,
    ) -> Result<Vec<f32>, VindexError> {
        self.load_with(operand, extent, &AuxiliaryExtents::whole())
    }

    /// [`Self::load_at`], reading each dependency at the extent
    /// `auxiliaries` names for it.
    ///
    /// The owner's own extent and its dependencies' are separate
    /// decisions: the codes of a vector-quantised tensor do not change
    /// when its codebook is read at another depth, and what the values
    /// MEAN does. Both are the caller's to state, because both are pins.
    pub fn load_with(
        &self,
        operand: &OperandRef,
        extent: RepresentationExtent,
        auxiliaries: &AuxiliaryExtents,
    ) -> Result<Vec<f32>, VindexError> {
        let raw = self.load_raw(operand)?;
        let codec = self.registry.resolve(&raw.dtype, &operand.tensor)?;
        self.decode_guarded(operand, codec, raw, extent, auxiliaries, &mut Vec::new())
    }

    /// The tensor a stream after the first is stored in: the operand's own
    /// tensor, suffixed with the stream's declared name.
    ///
    /// The convention is the codec's declaration made physical, so a codec
    /// with streams stored apart needs no container support of its own and
    /// no loader knows what any particular stream means.
    pub fn sibling_stream_tensor(tensor: &str, stream: &str) -> String {
        format!("{tensor}.{stream}")
    }

    /// The shape the container records for an address, or `None` when it
    /// holds no such tensor.
    pub fn stored_shape(&self, address: &OperandAddress) -> Option<Vec<usize>> {
        self.segments
            .get(&address.object)?
            .tensors
            .get(&address.tensor)
            .map(|tensor| tensor.shape.clone())
    }

    /// The container's declared dependencies — what a closure admission
    /// walks, and what a decode resolves through.
    pub fn references(&self) -> &crate::format::vindex3::auxiliary_references::ReferenceTable {
        &self.references
    }

    /// What this container measures about its own representations.
    pub fn attestations(
        &self,
    ) -> &crate::format::vindex3::representation_attestations::AttestationTable {
        &self.attestations
    }

    /// Whose measurements this build acts on.
    pub fn recognised(
        &self,
    ) -> &crate::format::vindex3::representation_attestations::recognition::RecognisedMethods {
        &self.recognised
    }

    /// Every dependency `operand`'s codec requires at `extent`, resolved:
    /// each target decoded through ITS own codec at ITS terminal extent.
    ///
    /// `visiting` is the cycle guard. Admission refuses a cyclic table
    /// before any of this runs, but the loader does not get to assume
    /// someone ran admission: a cycle here would be a stack overflow, and
    /// a refusal is what a store owes its caller.
    pub(super) fn resolve_auxiliaries(
        &self,
        operand: &OperandRef,
        codec: &'static dyn RepresentationCodec,
        extent: RepresentationExtent,
        auxiliaries: &AuxiliaryExtents,
        visiting: &mut Vec<OperandAddress>,
    ) -> Result<Vec<LoadedAuxiliary>, VindexError> {
        let required = codec.required_auxiliaries(extent);
        let owner = OperandAddress::new(&operand.object, &operand.tensor);
        let provided = self.references.auxiliaries_of(&owner);
        let names: Vec<&str> = provided.iter().map(|(name, _)| *name).collect();
        admit_auxiliary_names(
            required,
            &names,
            codec.encoding_label(),
            &operand.tensor,
            extent,
        )?;
        let mut resolved = Vec::with_capacity(required.len());
        for spec in required {
            let target = self
                .references
                .target(&owner, spec.name)
                .expect("an admitted name is a provided one")
                .clone();
            if visiting.contains(&target) {
                return Err(VindexError::Parse(format!(
                    "auxiliary resolution: {} is already being resolved — the container's \
                     declared dependencies form a cycle",
                    target.describe()
                )));
            }
            let (label, shape) = self.tensor_metadata(&target)?;
            let target_codec = self.registry.resolve(&label, &target.tensor)?;
            codec.validate_auxiliary(
                spec.name,
                &AuxiliaryMetadata {
                    object: target.object.clone(),
                    tensor: target.tensor.clone(),
                    label,
                    shape: shape.clone(),
                    identity: Some(target_codec.identity()),
                },
                &operand.shape,
                extent,
                &operand.tensor,
            )?;
            let reference = OperandRef {
                object: target.object.clone(),
                tensor: target.tensor.clone(),
                dtype: String::new(),
                shape: shape.clone(),
            };
            // Whole unless the caller said otherwise; the dependency's
            // own codec decides what "whole" means.
            let read_at = auxiliaries
                .get(spec.name)
                .unwrap_or_else(|| target_codec.terminal_extent());
            visiting.push(target);
            let values = self.load_guarded(&reference, read_at, visiting);
            visiting.pop();
            resolved.push(LoadedAuxiliary {
                name: spec.name.to_string(),
                shape,
                values: values?,
            });
        }
        Ok(resolved)
    }

    /// The label and shape the container records for an address.
    pub(super) fn tensor_metadata(
        &self,
        address: &OperandAddress,
    ) -> Result<(String, Vec<usize>), VindexError> {
        self.segments
            .get(&address.object)
            .and_then(|segment| segment.tensors.get(&address.tensor))
            .map(|tensor| (tensor.dtype.clone(), tensor.shape.clone()))
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "auxiliary resolution: {} is referenced and the container holds no such \
                     tensor",
                    address.describe()
                ))
            })
    }

    /// [`Self::load_at`] under a cycle guard.
    pub(super) fn load_guarded(
        &self,
        operand: &OperandRef,
        extent: RepresentationExtent,
        visiting: &mut Vec<OperandAddress>,
    ) -> Result<Vec<f32>, VindexError> {
        let raw = self.load_raw(operand)?;
        let codec = self.registry.resolve(&raw.dtype, &operand.tensor)?;
        // A dependency's OWN dependencies are read whole: nothing has
        // stated a choice for them, and the loader does not invent one.
        self.decode_guarded(
            operand,
            codec,
            raw,
            extent,
            &AuxiliaryExtents::whole(),
            visiting,
        )
    }

    /// Bind the streams `extent` needs and decode them.
    pub(super) fn decode_at(
        &self,
        operand: &OperandRef,
        codec: &'static dyn RepresentationCodec,
        first: RawOperand,
        extent: RepresentationExtent,
    ) -> Result<Vec<f32>, VindexError> {
        self.decode_guarded(
            operand,
            codec,
            first,
            extent,
            &AuxiliaryExtents::whole(),
            &mut Vec::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_guarded(
        &self,
        operand: &OperandRef,
        codec: &'static dyn RepresentationCodec,
        first: RawOperand,
        extent: RepresentationExtent,
        auxiliaries: &AuxiliaryExtents,
        visiting: &mut Vec<OperandAddress>,
    ) -> Result<Vec<f32>, VindexError> {
        let specs = codec.streams();
        let Some((values, apart)) = specs.split_first() else {
            return Err(VindexError::Parse(format!(
                "`{}` declares no streams",
                codec.encoding_label()
            )));
        };
        // The dependencies first, in dependency order: a codec is handed
        // what its dependency MEANS, never where it lives.
        let auxiliaries =
            self.resolve_auxiliaries(operand, codec, extent, auxiliaries, visiting)?;
        // Whether the streams share one stored tensor is the codec's fact,
        // not a function of how many streams it declares: NVFP4 declares
        // three (codes, group scales, tensor scale) and stores them as one
        // row, so counting streams sent every NVFP4 pack down the
        // stored-apart path looking for siblings that do not exist. Ask the
        // codec to bind the payload; only a codec that answers "stored
        // apart" takes the sibling path.
        match codec.bind_packed(&first.bytes, &operand.shape, &operand.tensor) {
            Ok(bound) => {
                let operands = attach_auxiliaries(CodecOperands::from_streams(bound), &auxiliaries);
                codec.validate(&operands, &operand.shape, extent, &operand.tensor)?;
                return Ok(codec.decode_all(&operands, &operand.shape, extent, &operand.tensor)?);
            }
            Err(crate::format::vindex3::represent::codec::CodecError::StreamsStoredApart {
                ..
            }) if !apart.is_empty() => {}
            Err(e) => return Err(e.into()),
        }
        // Streams stored apart. Only those the extent reads are opened —
        // a refinement stream the extent does not reach is never touched,
        // and a codec that needs one says so by refusing.
        let needed = codec.streams_at(extent, &operand.tensor)?;
        let mut siblings: Vec<RawOperand> = Vec::with_capacity(needed.len());
        for spec in needed.iter().skip(1) {
            siblings.push(self.load_raw(&OperandRef {
                object: operand.object.clone(),
                tensor: Self::sibling_stream_tensor(&operand.tensor, spec.name),
                dtype: operand.dtype.clone(),
                shape: operand.shape.clone(),
            })?);
        }
        let mut streams = NamedStreams::new().with(*values, &first.bytes);
        for (spec, sibling) in needed.iter().skip(1).zip(&siblings) {
            streams = streams.with(*spec, &sibling.bytes);
        }
        let operands = attach_auxiliaries(CodecOperands::from_streams(streams), &auxiliaries);
        codec.validate(&operands, &operand.shape, extent, &operand.tensor)?;
        Ok(codec.decode_all(&operands, &operand.shape, extent, &operand.tensor)?)
    }

    /// This store's process-unique identity.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// How many operands have been read out of this store since it was
    /// opened. The residency gate reads this.
    pub fn load_count(&self) -> u64 {
        self.loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// PHYSICAL bytes read from disk — the observed figure a preparation
    /// ledger is held against.
    pub fn bytes_read(&self) -> u64 {
        self.read_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether this object's bytes are on disk.
    ///
    /// `false` for an object the container describes but has not
    /// hydrated, and for one it does not describe at all — the caller
    /// asking this question wants to know if a load would succeed, and
    /// both answers are "no".
    pub fn is_resident(&self, object: &str) -> bool {
        self.segments.contains_key(object)
    }

    /// Objects described by the container whose bytes are not here.
    pub fn absent_objects(&self) -> &std::collections::BTreeSet<String> {
        &self.absent
    }

    /// Test-only observation of tensors actually read or mapped.
    #[cfg(test)]
    pub(crate) fn touched_operand_addresses(&self) -> std::collections::BTreeSet<(String, String)> {
        self.touched_operands.lock().unwrap().clone()
    }

    /// The objects this store has resolved an operand out of.
    ///
    /// Measured, not predicted. A hydration set computed by folding over
    /// a plan is a claim about what an execution will ask for; this is
    /// what it did ask for, and the two agreeing on a real model is the
    /// only thing that makes the fold trustworthy.
    pub fn touched_objects(&self) -> std::collections::BTreeSet<String> {
        self.touched.lock().unwrap().clone()
    }

    /// The dtype the container stores this operand as — tensor-table
    /// metadata only, no payload read.
    ///
    /// Separate from [`Self::load_raw`] because the residency policy has
    /// to know what a 100 MB matrix is BEFORE deciding how to hold it,
    /// and a query that read the matrix to answer would load the model
    /// twice.
    pub fn stored_dtype(&self, operand: &OperandRef) -> Option<&str> {
        self.segments
            .get(&operand.object)?
            .tensors
            .get(&operand.tensor)
            .map(|t| t.dtype.as_str())
    }

    /// The length the container RECORDS for this operand's stored bytes —
    /// tensor-table metadata only, no payload read. An instance fact: for
    /// an entropy-coded operand it is not a function of the shape, which
    /// is exactly why the stored footprint reads it here and never from
    /// a codec.
    /// The operand's whole stored footprint: its own tensor, plus any
    /// sibling stream the codec declares apart from it.
    ///
    /// Every plane a progressive artifact holds counts here whatever
    /// extent execution later selects — the footprint is what the
    /// container stores, and an extent decides what is READ, not what is
    /// on disk. (For a codec whose streams are stored some other way the
    /// sum is over what is found, so this is exactly the old reading.)
    pub fn stored_len(&self, operand: &OperandRef) -> Option<u64> {
        let segment = self.segments.get(&operand.object)?;
        let tensor = segment.tensors.get(&operand.tensor)?;
        let mut total = tensor.len;
        if let Some(codec) = self.registry.by_label(&tensor.dtype) {
            for spec in codec.streams().iter().skip(1) {
                let sibling = Self::sibling_stream_tensor(&operand.tensor, spec.name);
                total += segment.tensors.get(&sibling).map_or(0, |t| t.len);
            }
        }
        Some(total)
    }

    /// Load one operand's stored bytes and dtype, unwidened — for a
    /// caller that converts to a representation other than f32 (and for
    /// [`Self::load`] itself, so there is exactly one resolution path).
    pub fn load_raw(&self, operand: &OperandRef) -> Result<RawOperand, VindexError> {
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.touched.lock().unwrap().insert(operand.object.clone());
        #[cfg(test)]
        self.touched_operands
            .lock()
            .unwrap()
            .insert((operand.object.clone(), operand.tensor.clone()));
        let segment = self.segments.get(&operand.object).ok_or_else(|| {
            if self.absent.contains(&operand.object) {
                return VindexError::Parse(format!(
                    "object `{}` is described by this container but its segment is \
                     not resident — it was not hydrated",
                    operand.object
                ));
            }
            VindexError::Parse(format!("no segment for object `{}`", operand.object))
        })?;
        let tensor = segment.tensors.get(&operand.tensor).ok_or_else(|| {
            VindexError::Parse(format!(
                "no tensor `{}` in `{}`'s segment",
                operand.tensor, operand.object
            ))
        })?;
        let mut bytes = vec![0u8; tensor.len as usize];
        read_exact_at(
            &segment.file,
            &mut bytes,
            segment.payload_start + tensor.offset,
        )?;
        segment
            .copied
            .store(true, std::sync::atomic::Ordering::Relaxed);
        // Counted AFTER the read succeeds and from the buffer that was
        // filled, so the figure is what came off the disk rather than
        // what the tensor table said would.
        self.read_bytes
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(RawOperand {
            dtype: tensor.dtype.clone(),
            bytes,
        })
    }
}
