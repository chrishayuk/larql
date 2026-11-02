//! Opening an operand store and asking what it holds.

use super::super::super::super::encode::segment::read_segment_header;
use super::super::super::super::encode::REPRESENTATION_ID_SEP;
use super::super::super::super::inspect::SystemInspection;
use super::super::super::OperandRef;
use crate::error::VindexError;
use crate::format::vindex3::represent::codec::CodecRegistry;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

impl OperandStore {
    /// Read an explicitly declared byte window of one tensor. The full stored
    /// length is checked before the seek; expert workers never copy an entire
    /// packed bank merely to discard unowned experts afterwards.
    pub fn load_raw_range(
        &self,
        operand: &OperandRef,
        expected_total: u64,
        offset: u64,
        len: u64,
    ) -> Result<RawOperand, VindexError> {
        let segment = self
            .segments
            .get(&operand.object)
            .ok_or_else(|| VindexError::Parse(format!("missing object {}", operand.object)))?;
        let tensor = segment
            .tensors
            .get(&operand.tensor)
            .ok_or_else(|| VindexError::Parse(format!("missing tensor {}", operand.tensor)))?;
        if tensor.len != expected_total
            || offset.checked_add(len).is_none_or(|end| end > tensor.len)
        {
            return Err(VindexError::Parse(format!(
                "{}: expert byte range or stored length disagrees with declaration",
                operand.tensor
            )));
        }
        let absolute = segment
            .payload_start
            .checked_add(tensor.offset)
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| VindexError::Parse("expert file offset overflow".into()))?;
        let mut file = std::fs::File::open(&segment.path)?;
        file.seek(SeekFrom::Start(absolute))?;
        let mut bytes = vec![
            0;
            usize::try_from(len).map_err(|_| VindexError::Parse(
                "expert byte length exceeds address space".into()
            ))?
        ];
        file.read_exact(&mut bytes)?;
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.read_bytes
            .fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        self.touched.lock().unwrap().insert(operand.object.clone());
        #[cfg(test)]
        self.touched_operands
            .lock()
            .unwrap()
            .insert((operand.object.clone(), operand.tensor.clone()));
        Ok(RawOperand {
            dtype: tensor.dtype.clone(),
            bytes,
        })
    }

    /// Open every canonical segment of every object in the inspection.
    pub fn open(root: &Path, inspection: &SystemInspection) -> Result<Self, VindexError> {
        Self::open_for(root, inspection, None, RepresentationSource::Auto)
    }

    /// Open each object at the representation `want` selects, subject to
    /// `source`.
    ///
    /// `want` is the execution encoding a profile asked for. `None` keeps
    /// every object on its canonical representation, which is what every
    /// caller predating compiled packs meant.
    pub fn open_for(
        root: &Path,
        inspection: &SystemInspection,
        want: Option<&str>,
        source: RepresentationSource,
    ) -> Result<Self, VindexError> {
        Self::open_in(root, inspection, want, source, CodecRegistry::builtin())
    }

    /// [`Self::open_for`], decoding through `registry` from the first
    /// byte: the pack admission at open, selection, provider identity and
    /// decode all read this one registry.
    ///
    /// This is the constructor an external provider needs. A pack whose
    /// identity names a family only that provider registers is admitted
    /// here, and would have been refused by name — before the provider's
    /// registry was ever consulted — had the store been opened through the
    /// built-in one and re-pointed afterwards.
    pub fn open_in(
        root: &Path,
        inspection: &SystemInspection,
        want: Option<&str>,
        source: RepresentationSource,
        registry: &'static CodecRegistry,
    ) -> Result<Self, VindexError> {
        let mut segments = BTreeMap::new();
        let mut absent: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut selected = BTreeMap::new();
        let mut precision_map: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for object in &inspection.graph.objects {
            let Some(canonical) = object.representations.first() else {
                continue;
            };

            // A compiled pack is used only when one was asked for and one
            // exists. `Transient` never looks, so it stays an oracle.
            let packed_id = want.map(|enc| format!("{}{REPRESENTATION_ID_SEP}{}", object.id, enc));
            let stored_entry = match (source, &packed_id) {
                (RepresentationSource::Transient, _) | (_, None) => None,
                (_, Some(id)) => inspection.index.representations.get(id).map(|e| (id, e)),
            };

            // Under `transient` the pack's BYTES are deliberately ignored,
            // but its DECISIONS are not: which tensors a precision map
            // quantised is a property of the compiled representation, and
            // an oracle that re-decided would be measuring a different
            // program. Read the map from the pack's header even when the
            // canonical bytes will be bound.
            if source == RepresentationSource::Transient {
                if let Some(id) = &packed_id {
                    if let Some(pack) = inspection.index.representations.get(id) {
                        if let Ok((header, _)) = read_segment_header(&root.join(&pack.segment)) {
                            precision_map.insert(
                                object.id.clone(),
                                header
                                    .tensors
                                    .into_iter()
                                    .map(|t| (t.name, t.dtype))
                                    .collect(),
                            );
                        }
                    }
                }
            }

            let (id, entry, is_stored) = match stored_entry {
                Some((id, entry)) => {
                    // Bytes compiled by another build under a decode
                    // contract this one may not implement must be refused
                    // here, before anything reads them.
                    if let Some(codec) = &entry.codec {
                        codec.admit_in(registry)?;
                    }
                    (id.clone(), entry, true)
                }
                None => {
                    // No pack for this object. That is not yet a problem —
                    // it becomes one only if execution asks for a format
                    // these bytes are not already in, which the load path
                    // catches by name.
                    let id = format!("{}{REPRESENTATION_ID_SEP}{}", object.id, canonical.encoding);
                    let Some(entry) = inspection.index.representations.get(&id) else {
                        continue;
                    };
                    (id, entry, false)
                }
            };
            let _ = &id;
            selected.insert(
                object.id.clone(),
                SelectedRepresentation {
                    codec: entry.codec.clone(),
                    encoding: entry.encoding.clone(),
                    stored: is_stored,
                },
            );
            let path = root.join(&entry.segment);
            // A described object whose bytes are not here is NOT a
            // malformed container. `index.json` and the system graph say
            // what the model is, and that description is complete whether
            // or not every segment has been hydrated yet — which is the
            // whole basis on which a hydration set can be a SUBSET.
            //
            // Reading eagerly and propagating made a partly resident
            // container unopenable, so the refusal moves to the load
            // path, where it can name the object and say the true thing.
            // A segment that exists but cannot be read is still an error
            // here: that is corruption, not absence.
            if !path.exists() {
                absent.insert(object.id.clone());
                continue;
            }
            let (header, payload_start) = read_segment_header(&path)?;
            segments.insert(
                object.id.clone(),
                SegmentMap {
                    path,
                    declared: entry.segment.clone(),
                    payload_start,
                    tensors: header
                        .tensors
                        .into_iter()
                        .map(|t| (t.name.clone(), t))
                        .collect(),
                },
            );
        }
        // The reference table, if the index names one. A table it names
        // and the container does not hold is a refusal here rather than a
        // surprise at the first decode that needs it.
        let references = match &inspection.index.auxiliary_references {
            Some(name) => {
                crate::format::vindex3::auxiliary_references::AuxiliaryReferences::read(root, name)?
            }
            None => crate::format::vindex3::auxiliary_references::ReferenceTable::empty(),
        };
        // The attestation table, on the same terms as the reference table
        // above: named-but-absent is a refusal here rather than a silent
        // loss of every guarantee at the first floor that needed one.
        let attestations = match &inspection.index.representation_attestations {
            Some(name) => {
                crate::format::vindex3::representation_attestations::RepresentationAttestations::read(
                    root, name,
                )?
            }
            None => crate::format::vindex3::representation_attestations::AttestationTable::empty(),
        };
        Ok(Self {
            root: root.to_path_buf(),
            registry,
            references,
            attestations,
            recognised:
                crate::format::vindex3::representation_attestations::recognition::RecognisedMethods::none(),
            mapped: std::sync::Mutex::new(BTreeMap::new()),
            regions: std::sync::atomic::AtomicUsize::new(0),
            segments,
            absent,
            selected,
            precision_map,
            program: inspection.index.precision_map.clone(),
            // Resolved once at open, so every conformance check in the
            // load path reads the same answer the compiler wrote.
            plan_roles: crate::format::vindex3::represent::plan_roles::plan_roles(root, inspection),
            source,
            id: next_identity(),
            loads: std::sync::atomic::AtomicU64::new(0),
            read_bytes: std::sync::atomic::AtomicU64::new(0),
            runtime_quantised: std::sync::atomic::AtomicU64::new(0),
            stored_precision: std::sync::atomic::AtomicU64::new(0),
            touched: std::sync::Mutex::new(std::collections::BTreeSet::new()),
            #[cfg(test)]
            touched_operands: std::sync::Mutex::new(std::collections::BTreeSet::new()),
        })
    }

    /// The same store, decoding through `registry` instead of the one it
    /// was opened with. Selection, provider identity and decode all read
    /// this one registry, so a codec registered here is executable end to
    /// end and a codec absent from it is refused everywhere.
    ///
    /// Every pack the open admitted is admitted AGAIN, against the new
    /// registry: a store cannot be re-pointed at a registry that does not
    /// implement the contract its bytes were written under. A pack the
    /// open could not admit never reaches here — open the store through
    /// [`Self::open_in`] with the registry that knows it.
    pub fn with_registry(mut self, registry: &'static CodecRegistry) -> Result<Self, VindexError> {
        for selected in self.selected.values() {
            if let (true, Some(codec)) = (selected.stored, &selected.codec) {
                codec.admit_in(registry)?;
            }
        }
        self.registry = registry;
        Ok(self)
    }

    /// The same store, acting on measurements from these authorities and
    /// methods. Trust is the caller's to declare — a container cannot
    /// make itself believed by attesting more loudly.
    pub fn with_recognised(
        mut self,
        recognised: crate::format::vindex3::representation_attestations::recognition::RecognisedMethods,
    ) -> Self {
        self.recognised = recognised;
        self
    }

    /// The codecs this store decodes through.
    pub fn registry(&self) -> &'static CodecRegistry {
        self.registry
    }

    /// Resolve token-bank identity against this store's own container, not a
    /// caller-supplied tokenizer label. Ordinary weight loading needs no tokenizer.
    pub fn tokenizer_sha256(&self) -> Result<String, VindexError> {
        crate::format::vindex3::represent::token_bank::container_tokenizer_sha256(&self.root)
            .map_err(|error| VindexError::Parse(format!("tokenizer authority refused: {error}")))
    }

    /// How many objects' segments this store has mapped — the physical
    /// bindings a prepared image holds, one per object however many
    /// regions were taken from it.
    pub fn mapped_objects(&self) -> usize {
        self.mapped.lock().unwrap().len()
    }

    /// How many regions were bound through [`Self::map_region`]: the
    /// logical operands served by the mappings, none of them read.
    pub fn mapped_regions(&self) -> usize {
        self.regions.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What each object was bound to.
    pub fn selection(&self) -> &BTreeMap<String, SelectedRepresentation> {
        &self.selected
    }

    /// How many tensors this session quantised at load.
    ///
    /// Session-scoped rather than process-global so concurrent runs and
    /// tests cannot contaminate each other's count. Under
    /// [`RepresentationSource::Stored`] a non-zero value is an invariant
    /// violation, not a performance observation: it means the runtime
    /// manufactured a representation the caller required to be already
    /// compiled.
    pub fn runtime_quantised(&self) -> u64 {
        self.runtime_quantised
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Where this store was allowed to source representations from.
    pub fn representation_source(&self) -> RepresentationSource {
        self.source
    }

    /// What encoding a compiled precision map gives this tensor, when the
    /// store is reproducing one.
    ///
    /// `None` means no map is in force — either the store is not the
    /// transient oracle, or no pack exists — and the caller decides as it
    /// always did. `Some(enc)` is the compiled program's decision for this
    /// tensor, and the oracle honours it rather than re-deciding.
    pub fn mapped_encoding(&self, object: &str, tensor: &str) -> Option<&str> {
        self.precision_map
            .get(object)?
            .get(tensor)
            .map(String::as_str)
    }

    /// The container's precision program, when it declares one.
    pub fn program(&self) -> Option<&crate::format::vindex3::represent::map::PrecisionMap> {
        self.program.as_ref()
    }

    /// Whether an NVFP4 request for `operand`, stored as `stored_dtype`,
    /// binds at SOURCE precision (narrowed to f16) instead of an NVFP4
    /// image — the one derivation of that fact, read by the loader when
    /// it binds and by preparation when a backend selects, so the pinned
    /// realization and the resident bytes cannot disagree.
    ///
    /// True when the stored bytes are not an NVFP4 pack AND either the
    /// store may not manufacture a representation (`stored`), or the
    /// container's precision program holds the tensor at source. A
    /// compiled pack is a precision map, and a backend arm names a format
    /// per class — attention, FFN, head — which cannot express one; the
    /// map wins, at a precision higher than the arm asked for, and
    /// nothing is manufactured.
    ///
    /// The declared program is the authority. Only a container written
    /// before the map was explicit falls back to what its pack's tensor
    /// table happens to say.
    pub fn nvfp4_request_binds_at_source(&self, operand: &OperandRef, stored_dtype: &str) -> bool {
        use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
        if stored_dtype == DTYPE_NVFP4 {
            return false;
        }
        if self.source == RepresentationSource::Stored {
            return true;
        }
        match self.program() {
            Some(program) => {
                use crate::format::vindex3::represent::map::Precision;
                use crate::format::vindex3::represent::policy::classify;
                let role = classify(&operand.object, &operand.tensor, &operand.shape);
                matches!(program.resolve(role, &operand.tensor), Precision::Source)
            }
            None => matches!(
                self.mapped_encoding(&operand.object, &operand.tensor),
                Some(enc) if enc != DTYPE_NVFP4
            ),
        }
    }

    /// Whether an f16 request for an operand stored as `stored_dtype`
    /// binds the compiled NVFP4 image as stored — the mirror of
    /// [`Self::nvfp4_request_binds_at_source`], and likewise the one
    /// derivation read by the loader and by selection.
    ///
    /// True when the stored bytes ARE an NVFP4 pack. A pack replaces the
    /// tensor, so the container holds no source bytes to narrow: no
    /// realization is more faithful than the compiled one, and widening it
    /// to f16 would compute the same values at four times the bytes. The
    /// map wins in this direction too — below the precision the arm asked
    /// for, because nothing above it exists — and nothing is manufactured.
    pub fn f16_request_binds_compiled_nvfp4(stored_dtype: &str) -> bool {
        use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
        stored_dtype == DTYPE_NVFP4
    }

    /// The role the plan binds this tensor to, falling back to the name
    /// heuristics for anything the plan does not cover — the same order
    /// the representation compiler resolves in.
    pub fn role_of(
        &self,
        object: &str,
        tensor: &str,
        shape: &[usize],
    ) -> crate::format::vindex3::represent::policy::Role {
        self.plan_roles
            .get(&(object.to_string(), tensor.to_string()))
            .copied()
            .unwrap_or_else(|| {
                crate::format::vindex3::represent::policy::classify(object, tensor, shape)
            })
    }

    /// Whether this object's bytes come from a compiled pack.
    ///
    /// Conformance is a claim about a *pack*. Under `transient` the bound
    /// bytes are the canonical ones by design, and they are expected not to
    /// match a map describing the pack — checking them against it would
    /// refuse the oracle for doing exactly its job.
    pub fn is_stored(&self, object: &str) -> bool {
        self.selected.get(object).is_some_and(|s| s.stored)
    }

    /// How many tensors ran at their stored precision instead of the
    /// format the backend asked for.
    ///
    /// A compiled pack is a precision map: it may store `gate_proj` as
    /// NVFP4 and `q_proj` as BF16 because a policy decided to spend bytes
    /// there. Backend arms declare a format per *class* — attention, FFN,
    /// head — which is a coarser instrument than the map, so under
    /// [`RepresentationSource::Stored`] the stored encoding wins and the
    /// arm's request acts as a ceiling rather than a demand.
    ///
    /// This is never a silent downgrade: honouring the map means running
    /// *higher* precision than asked, and the count says how often.
    pub fn bound_at_stored_precision(&self) -> u64 {
        self.stored_precision
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record one such binding.
    pub fn note_stored_precision(&self) {
        self.stored_precision
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record that a tensor is about to be quantised at load, and refuse
    /// under [`RepresentationSource::Stored`].
    ///
    /// The refusal is the gate: it makes "no runtime quantisation" an
    /// invariant the run enforces rather than a timing an operator infers.
    /// Called by the weight loader, the only place quantisation can happen.
    pub fn note_runtime_quantisation(&self, tensor: &str) -> Result<(), VindexError> {
        if self.source == RepresentationSource::Stored {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}` would be quantised at load, and \
                 `--representation-source stored` forbids manufacturing a \
                 representation. Compile one with `larql vindex3 represent`, \
                 or ask for `auto`."
            )));
        }
        self.runtime_quantised
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
