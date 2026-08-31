//! **Native storage for KDA projections — the second candidate family.**
//!
//! A KDA candidate is its OWN candidate object (`target.kda_bank`,
//! its own index, map and seals), never a variant grafted into the
//! expert bank's placement: that machinery is expert-shaped through
//! and through (expert slots, identity addressing over a population),
//! while a KDA layer holds exactly four matrices. The loader composes
//! the two candidate kinds side by side — the same architecture the
//! transient-requant probe proved behaviourally, made physical.
//!
//! Layout per compiled layer, in fixed order:
//!
//! ```text
//! q_proj  [width, hidden]   base 0
//! k_proj  [width, hidden]   base 1·qkv_stride
//! v_proj  [width, hidden]   base 2·qkv_stride
//! o_proj  [hidden, width]   base 3·qkv_stride
//! ```
//!
//! The two strides are computed separately even where an encoding
//! makes them equal — `o_proj` transposes the reduction axis, and an
//! encoding whose block constraint holds for `hidden` but not `width`
//! (or vice versa) must refuse on the axis that actually violates it.
//!
//! Deliberately NOT stored: the convolutions, the low-rank decay and
//! output gates, `b_proj`, `A_log`, `dt_bias`, `o_norm`. They are a
//! few megabytes a layer against ~75, they feed the numerically
//! delicate recurrence, and every behavioural result this candidate
//! kind rests on was earned with them untouched.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use super::arena::{RepresentationArena, SourceOperands};
use super::compile::{hash_bytes, LayerBankLayout, OperandSeal, Pending};
use super::compiler::{write_index_atomically, CandidateIndex, CompileOutcome, SourceTensor};
use super::map::{Precision, PrecisionMap};
use super::policy::{layer_of, projection_of, Role};
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

/// The four projections a KDA layer stores, in bank order.
pub const KDA_PROJECTIONS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// One compiled KDA layer's shape in the bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaLayerLayout {
    pub layer: u32,
    pub encoding: String,
    /// Bytes one `[width, hidden]` projection occupies.
    pub qkv_stride: u64,
    /// Bytes the `[hidden, width]` output projection occupies.
    pub o_stride: u64,
}

impl KdaLayerLayout {
    pub fn new(
        layer: u32,
        encoding: &str,
        width: usize,
        hidden: usize,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            layer,
            encoding: encoding.to_string(),
            qkv_stride: LayerBankLayout::matrix_bytes(encoding, width, hidden)?,
            o_stride: LayerBankLayout::matrix_bytes(encoding, hidden, width)?,
        })
    }

    /// A projection's `(offset, len)` within this layer's extent.
    pub fn slot(&self, projection: &str) -> Result<(u64, u64), VindexError> {
        match projection {
            "q_proj" => Ok((0, self.qkv_stride)),
            "k_proj" => Ok((self.qkv_stride, self.qkv_stride)),
            "v_proj" => Ok((2 * self.qkv_stride, self.qkv_stride)),
            "o_proj" => Ok((3 * self.qkv_stride, self.o_stride)),
            other => Err(VindexError::Parse(format!(
                "`{other}` is not a KDA projection this bank stores"
            ))),
        }
    }

    /// This layer's whole extent in the candidate segment.
    pub fn layer_bytes(&self) -> u64 {
        3 * self.qkv_stride + self.o_stride
    }
}

/// Where each compiled layer's four-slot bank begins — the KDA
/// counterpart of `CandidatePlacement`, and like it the ONE definition
/// the compiler, the completeness verifier and the loader all share.
#[derive(Debug, Clone)]
pub struct KdaPlacement {
    placed: Vec<(KdaLayerLayout, u64)>,
}

impl KdaPlacement {
    /// Resolve placement for `layers` under `map`, at the family's
    /// projection geometry.
    ///
    /// A layer whose four projections resolve to different encodings is
    /// refused by name — one layer, one encoding, same as the expert
    /// bank. A layer the map does not compile is refused too.
    pub fn resolve(
        map: &PrecisionMap,
        role: Role,
        layers: &[u32],
        width: usize,
        hidden: usize,
    ) -> Result<Self, VindexError> {
        let mut sorted: Vec<u32> = layers.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.is_empty() {
            return Err(VindexError::Parse(
                "a KDA placement over no layers places nothing".into(),
            ));
        }
        let mut placed = Vec::with_capacity(sorted.len());
        let mut base = 0u64;
        for layer in sorted {
            let mut encodings: Vec<(&str, String)> = Vec::new();
            for proj in KDA_PROJECTIONS {
                let tensor = format!("{layer}.self_attn.{proj}.weight");
                if let Precision::Compiled(enc) = map.resolve(role, &tensor) {
                    encodings.push((proj, enc.to_string()));
                }
            }
            let Some((_, encoding)) = encodings.first() else {
                return Err(VindexError::Parse(format!(
                    "layer {layer} is in the KDA placement but map `{}` compiles none of \
                     its projections",
                    map.name
                )));
            };
            if let Some((proj, other)) = encodings.iter().find(|(_, e)| e != encoding) {
                return Err(VindexError::Parse(format!(
                    "layer {layer}: `{}` resolves to {encoding} but `{proj}` to {other} — \
                     a KDA layer bank carries ONE encoding",
                    encodings[0].0
                )));
            }
            let layout = KdaLayerLayout::new(layer, encoding, width, hidden)?;
            let extent = layout.layer_bytes();
            placed.push((layout, base));
            base = base
                .checked_add(extent)
                .ok_or_else(|| VindexError::Parse("KDA placement overflows u64".into()))?;
        }
        Ok(Self { placed })
    }

    pub fn layers(&self) -> impl Iterator<Item = u32> + '_ {
        self.placed.iter().map(|(l, _)| l.layer)
    }

    pub fn layout(&self, layer: u32) -> Result<&KdaLayerLayout, VindexError> {
        self.placed
            .iter()
            .find(|(l, _)| l.layer == layer)
            .map(|(l, _)| l)
            .ok_or_else(|| {
                VindexError::Parse(format!("layer {layer} is not in this KDA placement"))
            })
    }

    pub fn layer_base(&self, layer: u32) -> Result<u64, VindexError> {
        self.placed
            .iter()
            .find(|(l, _)| l.layer == layer)
            .map(|(_, b)| *b)
            .ok_or_else(|| {
                VindexError::Parse(format!("layer {layer} is not in this KDA placement"))
            })
    }
}

/// Compile a KDA candidate bank: the same seal/resume/checkpoint
/// discipline as `compile_expert_bank`, over the four-slot layout.
///
/// `tensors` is the scope — every `{layer}.self_attn.{q,k,v,o}_proj`
/// the map might compile. Anything else in the list is refused by
/// name: a KDA bank that silently skipped a tensor would leave the
/// caller believing it was compiled.
pub fn compile_kda_bank(
    source: &dyn SourceOperands,
    tensors: &[SourceTensor],
    object: &str,
    out: &Path,
    checkpoint: Option<(&Path, u64)>,
    index: &mut CandidateIndex,
    progress: &mut dyn FnMut(&CompileOutcome),
) -> Result<CompileOutcome, VindexError> {
    let role = Role::DecoderLinear;
    let arena = RepresentationArena::new(index.map.clone());
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(out)?;
    let mut outcome = CompileOutcome::default();

    // Placement is derived ONCE from the whole handed scope, exactly as
    // the expert compiler does — a per-tensor derivation would let two
    // runs over subsets write one layer at two bases.
    let mut compiled_layers: Vec<u32> = Vec::new();
    let mut geometry: Option<(usize, usize)> = None;
    for t in tensors {
        match projection_of(&t.name) {
            Some("q_proj" | "k_proj" | "v_proj") => {
                let [rows, cols] = t.shape[..] else {
                    return Err(VindexError::Parse(format!(
                        "tensor `{}` is not a matrix: {:?}",
                        t.name, t.shape
                    )));
                };
                geometry = Some((rows, cols)); // (width, hidden)
            }
            Some("o_proj") => {}
            other => {
                return Err(VindexError::Parse(format!(
                    "tensor `{}` (projection {other:?}) is not a KDA projection — a KDA \
                     scope holds q/k/v/o_proj and nothing else",
                    t.name
                )));
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
        (_, Some((width, hidden))) => Some(KdaPlacement::resolve(
            &index.map,
            role,
            &compiled_layers,
            width,
            hidden,
        )?),
        (_, None) => {
            return Err(VindexError::Parse(
                "the scope compiles operands but holds no q/k/v tensor to take the bank \
                 geometry from"
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
                "tensor `{}` carries no layer/projection, so it has no place in a KDA bank",
                t.name
            )));
        };
        let placement = placement
            .as_ref()
            .expect("a compiled tensor implies a placement");
        let layout = placement.layout(layer)?;
        // The tensor's own shape must produce the placed strides — a
        // transposed or truncated projection refuses here, by name,
        // instead of landing plausibly at the wrong extent.
        let [rows, cols] = t.shape[..] else {
            return Err(VindexError::Parse(format!(
                "tensor `{}` is not a matrix: {:?}",
                t.name, t.shape
            )));
        };
        let expect = LayerBankLayout::matrix_bytes(&encoding, rows, cols)?;
        let (slot_offset, slot_len) = layout.slot(projection)?;
        if expect != slot_len {
            return Err(VindexError::Parse(format!(
                "tensor `{}`: shape {:?} occupies {expect} bytes in {encoding} but the \
                 placement reserved {slot_len} — the shape is not the one this layer's \
                 bank was placed for",
                t.name, t.shape
            )));
        }
        let offset = placement.layer_base(layer)? + slot_offset;

        let operand = OperandRef {
            object: object.to_string(),
            tensor: t.name.clone(),
            dtype: String::new(),
            shape: t.shape.clone(),
        };
        let stored = source.load_stored(&operand)?;
        let source_hash = hash_bytes(&stored.bytes);

        if let Some(seal) = index.ledger.get(object, &t.name) {
            if seal.target_offset != offset {
                return Err(VindexError::Parse(format!(
                    "tensor `{}` is sealed at offset {} but this run's placement puts it \
                     at {offset} — pass the map's full layer scope or start a fresh \
                     candidate",
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
        if materialised.bytes.len() as u64 != slot_len {
            return Err(VindexError::Parse(format!(
                "tensor `{}`: encoder produced {} bytes, layout reserved {slot_len}",
                t.name,
                materialised.bytes.len(),
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
            target_len: slot_len,
        });
        outcome.sealed += 1;
        outcome.bytes_written += slot_len;
        if let Some((path, every)) = checkpoint {
            if every > 0 && (outcome.sealed as u64).is_multiple_of(every) {
                file.flush()?;
                write_index_atomically(index, path)?;
            }
        }
        progress(&outcome);
    }
    file.flush()?;
    if let Some((path, _)) = checkpoint {
        write_index_atomically(index, path)?;
    }
    Ok(outcome)
}

#[cfg(test)]
#[path = "kda_candidate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kda_candidate_real.rs"]
mod real;
