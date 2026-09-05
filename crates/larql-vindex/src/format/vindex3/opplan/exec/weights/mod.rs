//! Format-aware matrix-operand loading for the plan executor.
//!
//! The interpreter asks the backend which [`WeightFormat`] it computes
//! in and loads every matrix operand through [`load_weight`]; backends
//! receive slices, never operand references, exactly as before. The f16
//! path exists for device residency: a device buffer cache keyed by
//! `(pointer, length)` sees the same allocation on every call and keeps
//! the weight resident instead of re-uploading it per forward.
//!
//! **The bf16 → f16 conversion is exact for every normal-range value.**
//! bf16 carries 7 mantissa bits and f16 carries 10, so any bf16 value
//! whose magnitude lies in f16's normal range converts without rounding.
//! Overflow (|x| ≥ 65520, unrepresentable in f16) fails closed naming
//! the tensor — it would silently become infinity. Values below f16's
//! normal range land on subnormals and may round in the last bits; that
//! tail is a bounded realisation choice, and the parity gates against
//! the f32 backends and the upstream trace are its judge.

use super::backend::{WeightFormat, WeightSlice};
use super::narrow::{bf16_bytes_to_f16, f32_bytes_to_f16};
use super::operands::{OperandSource, RepresentationSource};
use super::quantise::{quantise_q4, quantise_q8, Q4_BLOCK, Q8_BLOCK};
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::kquant::{self, KQuant};
use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use crate::format::vindex3::represent::physical::WeightRegion;
// MXFP4 group geometry — per row, `k/32` groups of 16 packed bytes (lo
// nibble first) plus one e8m0 scale byte each — is the models crate's,
// so the kernel's layout contract and the codec's have one definition.
use larql_models::quant::mxfp4::{e8m0_to_f32, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS, MXFP4_TABLE};
use staged::StagedF32;

/// Alignment (and length granularity) of f16 weight allocations:
/// the Apple-GPU page size. A page-aligned, page-multiple allocation
/// lets a Metal device wrap the memory zero-copy instead of copying it
/// into a private buffer; any other device simply sees ordinary bytes.
pub const DEVICE_PAGE_ALIGN: usize = 16384;

/// Safetensors dtypes this loader can narrow to f16. bf16 converts
/// exactly (normal range); f32 rounds to nearest-even.
const DTYPE_BF16: &str = "BF16";
const DTYPE_F32: &str = "F32";

/// A page-aligned, page-multiple, zero-padded byte buffer.
///
/// [`AlignedBytes::as_slice`] returns the *padded* slice on purpose:
/// callers hand the whole allocation to a device so the buffer length
/// stays page-multiple; matrix geometry always travels separately.
#[derive(Debug)]
pub struct AlignedBytes {
    ptr: std::ptr::NonNull<u8>,
    /// Allocation length — `logical` rounded up to the page.
    padded: usize,
    /// Meaningful bytes at the front of the allocation.
    logical: usize,
}

// The buffer is plain owned bytes; nothing about the raw pointer ties
// it to a thread.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate a zeroed, page-aligned buffer holding `logical` bytes.
    pub fn zeroed(logical: usize) -> Self {
        let padded = logical.div_ceil(DEVICE_PAGE_ALIGN).max(1) * DEVICE_PAGE_ALIGN;
        let layout = std::alloc::Layout::from_size_align(padded, DEVICE_PAGE_ALIGN)
            .expect("page-aligned layout is always valid");
        // SAFETY: layout has non-zero size (padded >= one page).
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = std::ptr::NonNull::new(raw).unwrap_or_else(|| {
            std::alloc::handle_alloc_error(layout);
        });
        Self {
            ptr,
            padded,
            logical,
        }
    }

    /// A page-aligned copy of `bytes` — how a natively stored quantised
    /// operand (an MXFP4 expert's blocks or scales) is bound without a
    /// numeric transform: the bytes are the checkpoint's, only the
    /// alignment is ours.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut aligned = Self::zeroed(bytes.len());
        aligned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        aligned
    }

    /// The full padded allocation — page-aligned pointer, page-multiple
    /// length, zero beyond `logical_len`.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is `padded` bytes, initialised (zeroed
        // at alloc, fronts overwritten by the converter).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.padded) }
    }

    /// Writable, for the converters that FILL a buffer — narrowing to
    /// f16, quantising to Q8. Not public beyond the executor: a caller
    /// that could rewrite a resident weight in place would be editing the
    /// model behind the operand store.
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` guarantees uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.padded) }
    }

    /// Meaningful bytes at the front of the allocation.
    pub fn logical_len(&self) -> usize {
        self.logical
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.padded, DEVICE_PAGE_ALIGN)
            .expect("layout validated at allocation");
        // SAFETY: allocated with exactly this layout in `zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// The stored forms the CPU executor runs in place over a whole matrix,
/// and therefore the only forms a mapped binding can take: bf16 through
/// the fused bf16 matvec, f32 through BLAS. A stored form outside this
/// enum needs a decode, which a mapping by definition does not do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedForm {
    Bf16,
    F32,
}

impl MappedForm {
    pub fn format(self) -> WeightFormat {
        match self {
            Self::Bf16 => WeightFormat::Bf16,
            Self::F32 => WeightFormat::F32,
        }
    }

    /// Bytes per element of the stored form.
    pub fn width(self) -> usize {
        match self {
            Self::Bf16 => std::mem::size_of::<u16>(),
            Self::F32 => std::mem::size_of::<f32>(),
        }
    }

    /// The form a resident representation maps as, when it maps at all.
    pub fn of(format: WeightFormat) -> Option<Self> {
        match format {
            WeightFormat::Bf16 => Some(Self::Bf16),
            WeightFormat::F32 => Some(Self::F32),
            _ => None,
        }
    }
}

/// One loaded matrix operand, owning its bytes in the format the
/// backend declared.
#[derive(Debug)]
pub enum LoadedWeight {
    F32(StagedF32),
    /// Symmetric int8 codes plus one f32 scale per [`Q8_BLOCK`]
    /// elements. The only LOSSY residency format on this path: the values
    /// resident are not the values stored.
    Q8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
        /// Per-[`SUM_BLOCK`] sums of the codes — a materialised execution
        /// index for the asymmetric-activation path, EMPTY where no arm
        /// consumes it. Not model semantics: it is derivable from
        /// `codes`, and is stored only because recomputing it every token
        /// costs a second integer reduction per block.
        sums: Vec<i16>,
    },
    /// Symmetric int4 codes packed two per byte, plus one f32 scale per
    /// [`Q4_BLOCK`] elements. 4.5 bits/weight — half of Q8's bytes and
    /// 18.1x its quantisation step.
    Q4 {
        packed: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Stored bf16 code units, byte-for-byte as the checkpoint holds
    /// them. The cheapest possible load: no conversion at all.
    Bf16(AlignedBytes),
    /// The stored bytes themselves, bound as a region of the container's
    /// mapped segment: nothing copied, nothing converted, paged in as
    /// touched. Only the two forms the CPU executes in place over a
    /// whole matrix can be mapped, and the form is part of the variant so
    /// no other can be constructed.
    Mapped {
        region: WeightRegion,
        form: MappedForm,
    },
    F16(AlignedBytes),
    Mxfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
    },
    Nvfp4 {
        packed: AlignedBytes,
        scales: AlignedBytes,
        tensor_scale: f32,
    },
    /// A compiled ggml K-quant pack, byte-for-byte as the container holds
    /// it, with the codec that names its layout.
    ///
    /// The cheapest load after [`Self::Bf16`]: the bytes are read and
    /// kept. No decode, no requantise — PARETO-1's v3 arm, whose whole
    /// claim is that the bytes the kernel reads are the artifact's own.
    /// Plain owned bytes rather than [`AlignedBytes`]: no device wraps a
    /// K-quant, so page alignment would buy nothing here.
    KQuant {
        blocks: Vec<u8>,
        codec: KQuant,
    },
}

pub mod staged;
pub use staged::StagedF32 as F32Image;

impl LoadedWeight {
    /// The borrowed view a call struct carries.
    /// Bytes this operand OCCUPIES — the allocation, page padding
    /// included, because that is what the process holds.
    ///
    /// Not the matrix's logical size: a census that reported geometry
    /// would agree with itself no matter how much memory was really in
    /// use, which is the one thing a residency claim must not do.
    pub fn resident_bytes(&self) -> usize {
        match self {
            LoadedWeight::F32(w) => std::mem::size_of_val(&w[..]),
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => {
                codes.len() + std::mem::size_of_val(&scales[..]) + std::mem::size_of_val(&sums[..])
            }
            LoadedWeight::Q4 { packed, scales } => {
                packed.len() + std::mem::size_of_val(&scales[..])
            }
            LoadedWeight::Bf16(b) | LoadedWeight::F16(b) => b.as_slice().len(),
            LoadedWeight::Mapped { region, .. } => region.len() as usize,
            LoadedWeight::Mxfp4 { packed, scales } => {
                packed.as_slice().len() + scales.as_slice().len()
            }
            LoadedWeight::Nvfp4 { packed, scales, .. } => {
                packed.as_slice().len() + scales.as_slice().len()
            }
            LoadedWeight::KQuant { blocks, .. } => blocks.len(),
        }
    }

    /// Every backing allocation this operand holds: `(address, bytes)`.
    ///
    /// Plural because the compact formats are not one buffer. Q8 keeps
    /// codes and scales in separate allocations, so a model resident as
    /// Q8 holds roughly twice as many as the same model as bf16 — and
    /// where those land is invisible to a kernel benchmark that allocates
    /// one matrix and reuses it.
    pub fn allocations(&self) -> Vec<(usize, usize)> {
        let of = |p: *const u8, n: usize| (p as usize, n);
        match self {
            LoadedWeight::F32(w) => vec![of(w.as_ptr().cast(), std::mem::size_of_val(&w[..]))],
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => {
                let mut v = vec![
                    of(codes.as_ptr().cast(), codes.len()),
                    of(scales.as_ptr().cast(), std::mem::size_of_val(&scales[..])),
                ];
                if !sums.is_empty() {
                    v.push(of(sums.as_ptr().cast(), std::mem::size_of_val(&sums[..])));
                }
                v
            }
            LoadedWeight::Q4 { packed, scales } => vec![
                of(packed.as_ptr().cast(), packed.len()),
                of(scales.as_ptr().cast(), std::mem::size_of_val(&scales[..])),
            ],
            // The mapping IS the resident object: its pages, as touched.
            LoadedWeight::Mapped { region, .. } => {
                vec![of(region.bytes().as_ptr(), region.len() as usize)]
            }
            LoadedWeight::Bf16(b) | LoadedWeight::F16(b) => {
                vec![of(b.as_slice().as_ptr(), b.as_slice().len())]
            }
            LoadedWeight::Mxfp4 { packed, scales } | LoadedWeight::Nvfp4 { packed, scales, .. } => {
                vec![
                    of(packed.as_slice().as_ptr(), packed.as_slice().len()),
                    of(scales.as_slice().as_ptr(), scales.as_slice().len()),
                ]
            }
            LoadedWeight::KQuant { blocks, .. } => vec![of(blocks.as_ptr(), blocks.len())],
        }
    }

    /// Whether these bytes are the checkpoint's own, or a widened image
    /// of them.
    ///
    /// The distinction the whole rung turns on: `F32` over a bf16
    /// checkpoint means the loader DOUBLED the model, and no total alone
    /// can say where that happened.
    pub fn is_widened_f32(&self) -> bool {
        matches!(self, LoadedWeight::F32(_))
    }

    /// How many of this operand's allocations are page-padded — the
    /// `AlignedBytes` buffers, whose length rounds up to the page. A
    /// reconciliation of declared against resident bytes tolerates one
    /// page of padding per such allocation and nothing for the rest.
    pub fn padded_allocations(&self) -> usize {
        match self {
            LoadedWeight::F32(_)
            | LoadedWeight::Q8 { .. }
            | LoadedWeight::Q4 { .. }
            | LoadedWeight::Mapped { .. } => 0,
            LoadedWeight::Bf16(_) | LoadedWeight::F16(_) => 1,
            LoadedWeight::Mxfp4 { .. } | LoadedWeight::Nvfp4 { .. } => 2,
            LoadedWeight::KQuant { .. } => 0,
        }
    }

    /// The representation these bytes are resident in — what a pinned
    /// realization is checked against.
    pub fn format(&self) -> WeightFormat {
        match self {
            LoadedWeight::F32(_) => WeightFormat::F32,
            LoadedWeight::Bf16(_) => WeightFormat::Bf16,
            LoadedWeight::Mapped { form, .. } => form.format(),
            LoadedWeight::F16(_) => WeightFormat::F16,
            LoadedWeight::Q8 { .. } => WeightFormat::Q8,
            LoadedWeight::Q4 { .. } => WeightFormat::Q4,
            LoadedWeight::Mxfp4 { .. } => WeightFormat::Mxfp4,
            LoadedWeight::Nvfp4 { .. } => WeightFormat::Nvfp4,
            LoadedWeight::KQuant { .. } => WeightFormat::KQuant,
        }
    }

    pub fn slice(&self) -> WeightSlice<'_> {
        match self {
            LoadedWeight::F32(w) => WeightSlice::F32(w),
            LoadedWeight::Q8 {
                codes,
                scales,
                sums,
            } => WeightSlice::Q8 {
                codes,
                scales,
                sums,
                block: Q8_BLOCK,
            },
            LoadedWeight::Q4 { packed, scales } => WeightSlice::Q4 {
                packed,
                scales,
                block: Q4_BLOCK,
            },
            // SAFETY: `AlignedBytes` is page-aligned, so u16 alignment
            // holds; the length is even because the load arm rejects a
            // byte count that is not.
            LoadedWeight::Bf16(b) => {
                let bytes = b.as_slice();
                WeightSlice::Bf16(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2)
                })
            }
            LoadedWeight::F16(b) => WeightSlice::F16(b.as_slice()),
            // SAFETY: a segment's payload offsets are multiples of
            // `WEIGHT_BINDING_ALIGN` (4) over a page-aligned mapping, and
            // `map_region` refused a length that is not a whole number of
            // elements, so both casts are aligned and exact.
            LoadedWeight::Mapped { region, form } => {
                let bytes = region.bytes();
                match form {
                    MappedForm::Bf16 => WeightSlice::Bf16(unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2)
                    }),
                    MappedForm::F32 => WeightSlice::F32(unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
                    }),
                }
            }
            LoadedWeight::Mxfp4 { packed, scales } => WeightSlice::Mxfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
            },
            LoadedWeight::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => WeightSlice::Nvfp4 {
                packed: packed.as_slice(),
                scales: scales.as_slice(),
                tensor_scale: *tensor_scale,
            },
            LoadedWeight::KQuant { blocks, codec } => WeightSlice::KQuant {
                blocks,
                codec: *codec,
            },
        }
    }
}

/// Load one matrix operand in `format`, through the closure-verified
/// path only.
pub fn load_weight(
    store: OperandSource<'_>,
    operand: &OperandRef,
    format: WeightFormat,
) -> Result<LoadedWeight, VindexError> {
    match format {
        WeightFormat::F32 => Ok(LoadedWeight::F32(StagedF32::stage(store.load(operand)?)?)),
        WeightFormat::Q8 => {
            let in_dim = operand.shape.get(1).copied().ok_or_else(|| {
                VindexError::Parse(format!(
                    "tensor `{}` has shape {:?}; q8 residency blocks along the INPUT axis and \
                     needs a `[out, in]` matrix to know where the blocks are",
                    operand.tensor, operand.shape
                ))
            })?;
            // **CPU5-K1 was FALSIFIED and is opt-in.** Precomputing the
            // weight-code sums removes a second `SDOT` from the
            // asymmetric row, but measured 868 ms against the 757 ms it
            // was meant to beat: the extra ~12% of compact traffic, and a
            // third memory stream the prefetcher has to track, cost more
            // than a `vdotq` over codes already in registers. Kept
            // reachable so the negative result stays reproducible.
            let indexed = super::cpu::integer::weight_index_enabled()
                && matches!(
                    super::cpu::integer::activation_code(),
                    super::cpu::integer::ActivationCode::Asymmetric
                );
            Ok(quantise_q8(&store.load(operand)?, in_dim, indexed))
        }
        WeightFormat::Q4 => {
            let in_dim = operand.shape.get(1).copied().ok_or_else(|| {
                VindexError::Parse(format!(
                    "tensor `{}` has shape {:?}; q4 residency blocks along the INPUT axis and \
                     needs a `[out, in]` matrix to know where the blocks are",
                    operand.tensor, operand.shape
                ))
            })?;
            // Two codes to the byte, so an odd input axis has no packing.
            // Refusing is the only honest answer: the alternative is a
            // silently dropped final weight in every row.
            if in_dim % 2 != 0 {
                return Err(VindexError::Parse(format!(
                    "tensor `{}` has an odd input axis {in_dim}; q4 packs two codes per byte \
                     and cannot represent a ragged half-byte",
                    operand.tensor
                )));
            }
            Ok(quantise_q4(&store.load(operand)?, in_dim))
        }
        WeightFormat::Bf16 => {
            let raw = store.load_raw(operand)?;
            if raw.dtype.as_str() != DTYPE_BF16 {
                // No widening or narrowing here on purpose. This format
                // means "the stored bytes ARE the resident bytes"; a
                // checkpoint holding something else needs a judged
                // conversion, and inventing one silently would make the
                // format a lie.
                return Err(VindexError::Parse(format!(
                    "tensor `{}` is `{}`, not bf16 — the bf16 residency format copies stored \
                     bytes and performs no conversion",
                    operand.tensor, raw.dtype
                )));
            }
            Ok(LoadedWeight::Bf16(AlignedBytes::from_bytes(&raw.bytes)))
        }
        WeightFormat::Mxfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            store.store().note_runtime_quantisation(&operand.tensor)?;
            let values = store.load(operand)?;
            quantize_mxfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::Nvfp4 => {
            let rows = operand.shape.first().copied().unwrap_or(0);
            let k = operand.shape.get(1).copied().unwrap_or(0);
            // A compiled pack is already in the grid the kernel wants, so
            // the whole load is a read: no widening to f32, no requantise,
            // no arithmetic at all. That is the point of persisting it.
            let raw = store.load_raw(operand)?;
            check_pack_conforms(store, operand, &raw.dtype)?;
            if raw.dtype == DTYPE_NVFP4 {
                return nvfp4_from_stored(&raw.bytes, rows, k, &operand.tensor);
            }
            // A compiled pack is a precision map, and a backend arm names a
            // format per class — attention, FFN, head — which cannot express
            // one. Under `stored` the map wins: a tensor its policy held at
            // source precision runs at source precision, which is higher
            // than the arm asked for and manufactures nothing.
            let src_policy = store.store().representation_source();
            // A compiled map protected this tensor: honour that under
            // `stored` (bind what is there) and under `transient` (bind the
            // canonical bytes at the same precision, manufacturing nothing).
            // The two arms must run the same precision program or the
            // parity claim stops meaning anything the moment a map is mixed.
            // The declared program is the authority. Only a container
            // written before the map was explicit falls back to what its
            // pack's tensor table happens to say.
            let map_protects = match store.store().program() {
                Some(program) => {
                    use crate::format::vindex3::represent::map::Precision;
                    use crate::format::vindex3::represent::policy::classify;
                    let role = classify(&operand.object, &operand.tensor, &operand.shape);
                    matches!(program.resolve(role, &operand.tensor), Precision::Source)
                }
                None => matches!(
                    store.store().mapped_encoding(&operand.object, &operand.tensor),
                    Some(enc) if enc != DTYPE_NVFP4
                ),
            };
            if src_policy == RepresentationSource::Stored || map_protects {
                store.store().note_stored_precision();
                return narrow_to_f16(&raw, &operand.tensor);
            }
            store.store().note_runtime_quantisation(&operand.tensor)?;
            let values = widen_raw(&raw, &operand.tensor)?;
            quantize_nvfp4(&values, rows, k, &operand.tensor)
        }
        WeightFormat::KQuant => kquant_from_stored(store, operand),
        WeightFormat::F16 => {
            let raw = store.load_raw(operand)?;
            match raw.dtype.as_str() {
                DTYPE_BF16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                DTYPE_F32 => Ok(LoadedWeight::F16(f32_bytes_to_f16(
                    &raw.bytes,
                    &operand.tensor,
                )?)),
                other => Err(VindexError::Parse(format!(
                    "tensor `{}`: no judged f16 narrowing for dtype `{other}`",
                    operand.tensor
                ))),
            }
        }
    }
}

/// Refuse a stored tensor whose dtype the container's precision map does
/// not permit.
///
/// The map is the authority a pack is supposed to satisfy, so `stored`
/// checks conformance rather than taking the bytes' word for what program
/// they implement. A pack that compiled a tensor the map protects is not
/// a pack for this program, and silently executing it would mean running
/// something other than what the container declares. Shared by every arm
/// that binds stored compiled bytes, so the NVFP4 and K-quant paths
/// cannot drift in what they check.
fn check_pack_conforms(
    store: OperandSource<'_>,
    operand: &OperandRef,
    dtype: &str,
) -> Result<(), VindexError> {
    if let (Some(program), true) = (
        store.store().program(),
        store.store().is_stored(&operand.object),
    ) {
        // Resolved through the store, which reads the plan first and the
        // name heuristics second — the same order the compiler used.
        // Calling `classify` directly here was the miss that made a
        // correctly compiled pack fail its own conformance check.
        let role = store
            .store()
            .role_of(&operand.object, &operand.tensor, &operand.shape);
        if !program.conforms(role, &operand.tensor, dtype) {
            return Err(VindexError::Parse(format!(
                "tensor `{}` is stored as `{dtype}`, which the container's precision map \
                 `{}` does not permit — the pack does not implement the program the \
                 container declares",
                operand.tensor, program.name
            )));
        }
    }
    Ok(())
}

/// Bind a compiled K-quant pack: read the stored blocks and keep them.
///
/// No decode happens here and none may: if this path ever widened a
/// block or re-encoded one, the executed bytes would be a derivative of
/// the artifact rather than the artifact, and the v3 gate's claim — the
/// SAME stored bytes, two arithmetics — would be void. The geometry is
/// checked against the codec's own plan of the operand's shape, so a
/// stream of the wrong length for `[out, in]` is refused here by name
/// rather than read at the wrong stride by a kernel.
fn kquant_from_stored(
    store: OperandSource<'_>,
    operand: &OperandRef,
) -> Result<LoadedWeight, VindexError> {
    let raw = store.load_raw(operand)?;
    let Some(codec) = kquant::lookup(&raw.dtype) else {
        return Err(VindexError::Parse(format!(
            "tensor `{}` is `{}`, not a K-quant this executor runs in place ({}) — direct \
             K-quant residency binds stored blocks and performs no conversion",
            operand.tensor,
            raw.dtype,
            kquant::compilable_names()
        )));
    };
    check_pack_conforms(store, operand, &raw.dtype)?;
    let want = codec.plan(&operand.shape, &operand.tensor)?;
    if raw.bytes.len() != want {
        return Err(VindexError::Parse(format!(
            "tensor `{}`: {} bytes of {} do not describe shape {:?}, which needs {want} — \
             refusing rather than reading rows at the wrong stride",
            operand.tensor,
            raw.bytes.len(),
            codec.name,
            operand.shape
        )));
    }
    Ok(LoadedWeight::KQuant {
        blocks: raw.bytes,
        codec,
    })
}

/// Bind a compiled NVFP4 pack: copy each region into a page-aligned
/// buffer the device can take, and read the tensor scale.
///
/// No quantisation happens here and none may: if this path ever needed to
/// compute a scale or round an element, the pack would not have been a
/// compiled representation.
fn nvfp4_from_stored(
    bytes: &[u8],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    use crate::format::vindex3::represent::nvfp4_pack::{split, PackLayout};

    let layout = PackLayout::derive(&[rows, k], name)?;
    let (packed_src, scales_src, tensor_scale) = split(bytes, &layout, name)?;

    let mut packed = AlignedBytes::zeroed(packed_src.len());
    packed.as_mut_slice()[..packed_src.len()].copy_from_slice(packed_src);
    let mut scales = AlignedBytes::zeroed(scales_src.len());
    scales.as_mut_slice()[..scales_src.len()].copy_from_slice(scales_src);

    Ok(LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    })
}

/// Bind an already-read float operand as f16, the narrowing the F16 arm
/// performs. Shared so the stored-precision path cannot drift from it.
fn narrow_to_f16(
    raw: &super::operands::RawOperand,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    match raw.dtype.as_str() {
        DTYPE_BF16 => Ok(LoadedWeight::F16(bf16_bytes_to_f16(&raw.bytes, name)?)),
        DTYPE_F32 => Ok(LoadedWeight::F16(f32_bytes_to_f16(&raw.bytes, name)?)),
        other => Err(VindexError::Parse(format!(
            "tensor `{name}`: no judged f16 narrowing for stored dtype `{other}`"
        ))),
    }
}

/// Widen an already-read raw operand, so the NVFP4 path can inspect the
/// stored dtype without paying for a second read of the same bytes.
fn widen_raw(raw: &super::operands::RawOperand, name: &str) -> Result<Vec<f32>, VindexError> {
    super::operands::widen(&raw.dtype, &raw.bytes, name)
}

/// e2m1's largest magnitude; the shared exponent is chosen so the
/// group's max maps at or below it, saturating the rare overshoot.
const MXFP4_MAX_MAG: f32 = 6.0;
/// Exponent of [`MXFP4_MAX_MAG`]'s leading bit: `floor(log2(6)) = 2`.
const MXFP4_EMAX: i32 = 2;

/// Quantise one `[rows, k]` f32 matrix to MXFP4 — the OCP microscaling
/// rule: per 32-element group, shared scale `2^(floor(log2(max|x|)) -
/// 2)` as e8m0, elements rounded to the nearest e2m1 grid value
/// (ties to the even code index), saturating at ±6.
///
/// A lossy realisation by construction; the parity gates against the
/// f16/f32 anchors and the upstream trace are its judge. Layout is the
/// kernel's, and the nibble-order control in the tests pins it against
/// the independent `larql-models` decoder.
pub fn quantize_mxfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    if !k.is_multiple_of(MXFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the MXFP4 32-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / MXFP4_GROUP_ELEMS;
    let mut packed = AlignedBytes::zeroed(rows * groups * MXFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        packed_dst[..rows * groups * MXFP4_GROUP_BYTES]
            .par_chunks_mut(groups * MXFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                for g in 0..groups {
                    let group = &row_values[g * MXFP4_GROUP_ELEMS..(g + 1) * MXFP4_GROUP_ELEMS];
                    let max_abs = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale_byte = if max_abs == 0.0 {
                        0u8 // decodes to 0.0; all codes zero
                    } else {
                        let exponent = max_abs.log2().floor() as i32 - MXFP4_EMAX;
                        (exponent + 127).clamp(1, 254) as u8
                    };
                    row_scales[g] = scale_byte;
                    let scale = e8m0_to_f32(scale_byte);
                    let inv = if scale == 0.0 { 0.0 } else { scale.recip() };
                    let bytes = &mut row_packed[g * MXFP4_GROUP_BYTES..(g + 1) * MXFP4_GROUP_BYTES];
                    for (b, pair) in group.chunks_exact(2).enumerate() {
                        let lo = nearest_mxfp4_code(pair[0] * inv);
                        let hi = nearest_mxfp4_code(pair[1] * inv);
                        bytes[b] = lo | (hi << 4);
                    }
                }
            });
    }
    Ok(LoadedWeight::Mxfp4 { packed, scales })
}

/// Quantise one `[rows, k]` f32 matrix to NVFP4 into page-aligned
/// buffers, delegating the numerics to `larql_models::quant::nvfp4` so
/// the format has exactly one definition — the CPU reference, this
/// loader, and the Metal kernel all read that module's contract.
///
/// The only thing added here is residency: the same page-aligned
/// allocation MXFP4 uses, so a device can wrap the buffers zero-copy.
pub fn quantize_nvfp4(
    values: &[f32],
    rows: usize,
    k: usize,
    name: &str,
) -> Result<LoadedWeight, VindexError> {
    use larql_models::quant::nvfp4::{
        quantize_row_into, tensor_scale_for, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
    };
    if !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: k={k} is not a multiple of the NVFP4 \
             {NVFP4_GROUP_ELEMS}-element group"
        )));
    }
    if values.len() != rows * k {
        return Err(VindexError::Parse(format!(
            "tensor `{name}`: {} values do not fill [{rows}, {k}]",
            values.len()
        )));
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    // The tensor scale is a property of the whole matrix, so it is chosen
    // once before any row is encoded — rows cannot each pick their own
    // and still decode under one shared scale.
    let tensor_scale = tensor_scale_for(values);
    let mut packed = AlignedBytes::zeroed(rows * groups * NVFP4_GROUP_BYTES);
    let mut scales = AlignedBytes::zeroed(rows * groups);
    {
        use rayon::prelude::*;
        let packed_dst = packed.as_mut_slice();
        let scales_dst = scales.as_mut_slice();
        // Rows are independent given the tensor scale, so the parallelism
        // lives here while the numerics stay in one place
        // (`quant::nvfp4::quantize_row_into`), shared with the CPU
        // reference the kernel is judged against.
        packed_dst[..rows * groups * NVFP4_GROUP_BYTES]
            .par_chunks_mut(groups * NVFP4_GROUP_BYTES)
            .zip(scales_dst[..rows * groups].par_chunks_mut(groups))
            .zip(values.par_chunks(k))
            .for_each(|((row_packed, row_scales), row_values)| {
                quantize_row_into(row_values, tensor_scale, row_packed, row_scales);
            });
    }
    Ok(LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    })
}

/// The e2m1 code nearest to `v` (ties to the even code index),
/// saturating at ±6.
fn nearest_mxfp4_code(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0 };
    let mag = v.abs().min(MXFP4_MAX_MAG);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (code, value) in MXFP4_TABLE.iter().enumerate().take(8) {
        let err = (mag - value).abs();
        if err < best_err || (err == best_err && code.is_multiple_of(2)) {
            best = code as u8;
            best_err = err;
        }
    }
    if best == 0 {
        0 // ±0 collapse to +0: the table's -0.0 encodes nothing extra
    } else {
        sign | best
    }
}

#[cfg(test)]
mod staged_tests;
#[cfg(test)]
mod tests;
