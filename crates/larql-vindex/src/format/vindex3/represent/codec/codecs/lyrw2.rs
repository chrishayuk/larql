//! How a LYRW v2 bank's region schema reaches the contract.
//!
//! A bank region declares a format tag, a packing and, for a split
//! codec, the `pair_id` that binds its values region to its scales
//! region. None of that is a codec: it is the STORAGE ARRANGEMENT that
//! supplies a codec's streams. So this module answers two questions and
//! nothing else — which registered codec a format tag names, and how a
//! packing supplies that codec's streams — and refuses, by tag, what this
//! build recognises but cannot execute. Reading preserves such tags;
//! capability checking refuses them (spec §6.5, §11).

use super::super::error::CodecError;
use super::super::extent::RepresentationExtent;
use super::super::streams::{CodecOperands, NamedStreams, GROUP_SCALES, VALUES};
use super::super::RepresentationCodec;
use super::float::{BF16, F16, F32};
use super::kquant::{Q4_K, Q6_K, Q8_0};
use super::mxfp4::DTYPE_MXFP4;
use super::nvfp4::NVFP4;
use crate::format::lyrw2::region_format::{Packing, RegionFormat};

/// The codec label a region format names, or `None` where this build
/// recognises the tag and registers no codec for it.
///
/// `None` is not "unknown": `Q4_0`, `Fp4Larql` and `Mxfp8` are tags this
/// reader preserves faithfully and this executor cannot serve, and a
/// caller that needs them gets a refusal naming the tag, not a guess.
pub fn codec_label(format: RegionFormat) -> Option<&'static str> {
    Some(match format {
        RegionFormat::F32 => F32.encoding_label(),
        RegionFormat::F16 => F16.encoding_label(),
        RegionFormat::BF16 => BF16.encoding_label(),
        RegionFormat::Q4K => Q4_K.encoding_label(),
        RegionFormat::Q6K => Q6_K.encoding_label(),
        RegionFormat::Q8_0 => Q8_0.encoding_label(),
        RegionFormat::Mxfp4 => DTYPE_MXFP4,
        RegionFormat::Nvfp4 => NVFP4.encoding_label(),
        RegionFormat::Q4_0
        | RegionFormat::Fp4Larql
        | RegionFormat::Mxfp8
        | RegionFormat::Unknown(_) => return None,
    })
}

/// How a region's packing supplies a codec's streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionBinding {
    /// The region alone is the whole operand: one contiguous payload.
    Single,
    /// The region holds codes; the scales are in the partner named by
    /// `pair_id`.
    PairedValues,
    /// The region holds scales for the partner named by `pair_id`.
    PairedScales,
}

pub fn region_binding(packing: Packing) -> Option<RegionBinding> {
    match packing {
        Packing::RowMajor | Packing::BlocksWithScalesInline => Some(RegionBinding::Single),
        Packing::BlocksValues => Some(RegionBinding::PairedValues),
        Packing::BlocksScales => Some(RegionBinding::PairedScales),
        Packing::Unknown(_) => None,
    }
}

/// Bind a bank region — with its paired scales region, if the packing
/// names one — onto `codec`'s streams, and validate the result.
///
/// Without a scales region the codec binds the payload as one row; a
/// codec whose streams are stored apart then refuses by name. With one,
/// the two regions are the codes and group-scale streams, and a codec
/// that declares no scales stream refuses a partner it cannot consume.
///
/// A region is the WHOLE of a tensor of `shape`, so its streams must
/// total exactly what the codec declares for that shape. For a fixed-size
/// codec the rule is: each stream satisfies its declared minimum, all
/// stream lengths total the declared representation size, therefore no
/// stream can contain unclaimed bytes. A region longer than its shape is
/// a container disagreeing with the declared geometry — a scales operand
/// pointed at some other tensor — not spare bytes. An instance-sized
/// codec has no shape-derived size (`stored_bytes` refuses with
/// `InstanceSized`), so it cannot be bound as a region here at all; its
/// exact length is the bound container record's, read through the
/// operand store, never a declaration.
pub fn bind_region<'a>(
    codec: &dyn RepresentationCodec,
    shape: &[usize],
    values: &'a [u8],
    scales: Option<&'a [u8]>,
    tensor: &str,
) -> Result<CodecOperands<'a>, CodecError> {
    let streams = match scales {
        None => codec.bind_packed(values, shape, tensor)?,
        Some(scales) => {
            if !codec.streams().contains(&GROUP_SCALES) {
                return Err(CodecError::UnexpectedStream {
                    tensor: tensor.into(),
                    label: codec.encoding_label().into(),
                    stream: GROUP_SCALES.name.into(),
                    declared: codec.streams().iter().map(|s| s.name.to_string()).collect(),
                });
            }
            NamedStreams::single(VALUES, values).with(GROUP_SCALES, scales)
        }
    };
    let operands = CodecOperands::from_streams(streams);
    codec.validate(&operands, shape, RepresentationExtent::TERMINAL, tensor)?;
    let have = values.len() + scales.map_or(0, <[u8]>::len);
    let need = codec.stored_bytes(shape, RepresentationExtent::TERMINAL, tensor)?;
    if have as u64 != need {
        return Err(CodecError::Geometry {
            tensor: tensor.into(),
            label: codec.encoding_label().into(),
            shape: shape.to_vec(),
            why: format!(
                "its streams total {have} bytes and the shape declares {need}; the container \
                 and the declared geometry disagree"
            ),
        });
    }
    Ok(operands)
}
