//! Representation compilation over a real encoded container.
//!
//! The gate under test is the one the module header states: compiled bytes
//! must equal, bit for bit, what the runtime would have produced by
//! quantising the same tensor at load. Everything else here guards the
//! properties that make that claim meaningful — the canonical bytes survive,
//! the pack is smaller, and a reader can find its way back to the source.

use super::*;
use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer,
};
use crate::format::vindex3::index::ContainerAuthority;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::weights::load_weight;
use nvfp4_pack::split;

/// Encode the miniature Glimmer fixture, then compile an NVFP4
/// representation of it.
fn compiled_pair(
    tmp: &tempfile::TempDir,
) -> (std::path::PathBuf, std::path::PathBuf, RepresentReport) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let report = compile_representation(&src, &out, &RepresentSpec::nvfp4())
        .expect("the dense fixture is 16-aligned throughout");
    (src, out, report)
}

fn index_of(dir: &std::path::Path) -> Vindex3Index {
    serde_json::from_str(&std::fs::read_to_string(dir.join(INDEX_JSON)).unwrap()).unwrap()
}

/// The dtype the canonical container stores `operand`'s tensor as.
fn packed_store_source_dtype(store: &OperandStore, operand: &OperandRef) -> String {
    store
        .stored_dtype(operand)
        .expect("the canonical container holds the tensor")
        .to_string()
}

//
// Compiling the right bytes is worth nothing if execution does not read
// them. These pin the three claims that make a compiled pack real: it is
// selected, it is used *instead of* quantising, and using it changes no
// value.

use crate::format::vindex3::opplan::exec::operands::RepresentationSource;

/// Load one tensor through a store opened under `source`, returning the
/// bound weight and how many tensors the session quantised at load.
fn load_under(
    dir: &std::path::Path,
    source: RepresentationSource,
    object: &str,
    tensor: &str,
    dtype: &str,
    shape: &[usize],
) -> (LoadedWeight, u64) {
    let inspection = inspect_container(dir, false).unwrap();
    let store = OperandStore::open_for(dir, &inspection, Some(DTYPE_NVFP4), source).unwrap();
    let loaded = load_weight(
        (&store).into(),
        &OperandRef {
            object: object.to_string(),
            tensor: tensor.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        },
        WeightFormat::Nvfp4,
    )
    .unwrap();
    let n = store.runtime_quantised();
    (loaded, n)
}

/// The first compiled tensor of the decoder stack, as (object, name, dtype,
/// shape) in the *source* container.
fn a_compiled_tensor(src: &std::path::Path) -> (String, String, String, Vec<usize>) {
    let index = index_of(src);
    let entry = index
        .representations
        .values()
        .find(|e| e.object.contains("decoder_stack"))
        .unwrap();
    let (header, _) = read_segment_header(&src.join(&entry.segment)).unwrap();
    let t = header
        .tensors
        .iter()
        .find(|t| t.shape.len() == 2 && t.name.contains("q_proj"))
        .expect("the stack has attention projections");
    (
        entry.object.clone(),
        t.name.clone(),
        t.dtype.clone(),
        t.shape.clone(),
    )
}

fn deployment_of(
    tmp: &tempfile::TempDir,
) -> (std::path::PathBuf, std::path::PathBuf, RepresentReport) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("deploy.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let mut spec = RepresentSpec::nvfp4();
    spec.deployment = true;
    let report = compile_representation(&src, &out, &spec).unwrap();
    (src, out, report)
}

//
// The vocabulary in `kquant.rs` is tested directly and thoroughly, and
// that is exactly why these are here: unit-testing the pieces proved the
// codec, not the WIRING. Nothing exercised `compile_representation` down
// the K-quant branch or `OperandStore::load` through the K-quant decode,
// so both were type-checked, reviewed, and never once executed. Coverage
// said 65% on a file with nineteen passing tests, which is what that gap
// looks like from the outside.

fn kquant_spec(encoding: &str) -> RepresentSpec {
    RepresentSpec {
        encoding: encoding.to_string(),
        objects: Vec::new(),
        roles: policy::RolePolicy::default(),
        deployment: false,
        protect: policy::Protections::default(),
    }
}

// NVFP4-Q8-1 at the loader: the Q8 arm binds the SAME stored pack for a
// Q8 activation, keeps every source-precision binding the f32 arm makes,
// and refuses to quantise at load under its name.

fn load_as(
    dir: &std::path::Path,
    source: RepresentationSource,
    op: &OperandRef,
    format: WeightFormat,
) -> Result<(LoadedWeight, OperandStore), VindexError> {
    let inspection = inspect_container(dir, false).unwrap();
    let store = OperandStore::open_for(dir, &inspection, Some(DTYPE_NVFP4), source).unwrap();
    let loaded = load_weight((&store).into(), op, format)?;
    Ok((loaded, store))
}

fn operand_of(src: &std::path::Path) -> OperandRef {
    let (object, tensor, dtype, shape) = a_compiled_tensor(src);
    OperandRef {
        object,
        tensor,
        dtype,
        shape,
    }
}

/// Compile the dense fixture into a registered encoder's pack. The double
/// stores raw little-endian f32, so the fixture's F32 tensors must come
/// back as exactly their source bytes.
fn encoder_pair(
    tmp: &tempfile::TempDir,
) -> (std::path::PathBuf, std::path::PathBuf, RepresentReport) {
    use codec::encoder::tests::RawF32Codec;
    use codec::RepresentationCodec;
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("encoder.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let encoders = EncoderRegistry::new()
        .register(Box::new(RawF32Codec))
        .unwrap();
    let spec = RepresentSpec {
        encoding: RawF32Codec.encoding_label().to_string(),
        ..RepresentSpec::nvfp4()
    };
    let report = compile_representation_with(&src, &out, &spec, &encoders)
        .expect("a registered encoder is a compiler");
    (src, out, report)
}

use codec::RepresentationCodec as _;

/// How a [`QuirkyEncoder`] departs from the honest raw-f32 double.
#[derive(Clone, Copy)]
enum Quirk {
    /// Answers `InstanceSized`: its length is learned by encoding.
    InstanceSized,
    /// Refuses every shape, so every tensor is carried.
    RefusesEveryShape,
    /// States the honest length, then writes one byte more.
    MisstatesItsLength,
    /// Honest, and accepts input-feature weights (it writes raw f32, so
    /// the weights cannot change its bytes; it checks their length).
    AcceptsWeights,
}

/// The raw-f32 double with one quirk in how it reports or writes lengths.
struct QuirkyEncoder(Quirk);

impl codec::RepresentationCodec for QuirkyEncoder {
    fn encoding_label(&self) -> &'static str {
        codec::encoder::tests::RawF32Codec.encoding_label()
    }
    fn identity(&self) -> CodecIdentity {
        codec::encoder::tests::RawF32Codec.identity()
    }
    fn streams(&self) -> &'static [codec::StreamSpec] {
        codec::encoder::tests::RawF32Codec.streams()
    }
    fn capabilities(&self) -> codec::CodecCapabilities {
        codec::encoder::tests::RawF32Codec.capabilities()
    }
    fn extents(&self) -> Vec<codec::ExtentCertificate> {
        codec::encoder::tests::RawF32Codec.extents()
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: codec::RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        match self.0 {
            Quirk::InstanceSized => Err(CodecError::InstanceSized {
                tensor: tensor.into(),
                label: self.encoding_label().into(),
            }),
            Quirk::RefusesEveryShape => Err(CodecError::Destination {
                tensor: tensor.into(),
                need: 0,
                have: 1,
            }),
            Quirk::MisstatesItsLength | Quirk::AcceptsWeights => {
                codec::encoder::tests::RawF32Codec.stored_bytes(shape, extent, tensor)
            }
        }
    }
    fn validate(
        &self,
        operands: &codec::CodecOperands<'_>,
        shape: &[usize],
        extent: codec::RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        codec::encoder::tests::RawF32Codec.validate(operands, shape, extent, tensor)
    }
    fn decode_rows(
        &self,
        operands: &codec::CodecOperands<'_>,
        shape: &[usize],
        rows: std::ops::Range<usize>,
        extent: codec::RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        codec::encoder::tests::RawF32Codec.decode_rows(operands, shape, rows, extent, dst, tensor)
    }
    fn decode_residency(&self) -> codec::ResidencyProfile {
        codec::encoder::tests::RawF32Codec.decode_residency()
    }
}

impl RepresentationEncoder for QuirkyEncoder {
    fn encode_packed(
        &self,
        values: &[f32],
        shape: &[usize],
        extent: codec::RepresentationExtent,
        tensor: &str,
    ) -> Result<Vec<u8>, CodecError> {
        let mut bytes =
            codec::encoder::tests::RawF32Codec.encode_packed(values, shape, extent, tensor)?;
        if matches!(self.0, Quirk::MisstatesItsLength) {
            bytes.push(0);
        }
        Ok(bytes)
    }
    fn encode_packed_weighted(
        &self,
        values: &[f32],
        shape: &[usize],
        extent: codec::RepresentationExtent,
        tensor: &str,
        input_weights: &[f64],
    ) -> Result<Vec<u8>, CodecError> {
        if !matches!(self.0, Quirk::AcceptsWeights) {
            return Err(CodecError::WeightingUnsupported {
                tensor: tensor.into(),
                label: self.encoding_label().into(),
            });
        }
        let row: usize = shape[1..].iter().product();
        if input_weights.len() != row {
            return Err(CodecError::WeightingShape {
                tensor: tensor.into(),
                need: row,
                have: input_weights.len(),
            });
        }
        self.encode_packed(values, shape, extent, tensor)
    }
}

/// Compile the dense fixture with a [`QuirkyEncoder`].
fn compile_quirky(
    tmp: &tempfile::TempDir,
    quirk: Quirk,
) -> (std::path::PathBuf, Result<RepresentReport, VindexError>) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("quirky.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let encoders = EncoderRegistry::new()
        .register(Box::new(QuirkyEncoder(quirk)))
        .unwrap();
    let spec = RepresentSpec {
        encoding: QuirkyEncoder(quirk).encoding_label().to_string(),
        ..RepresentSpec::nvfp4()
    };
    let result = compile_representation_with(&src, &out, &spec, &encoders);
    (out, result)
}

/// Weights for every 2-D tensor of the dense fixture's first compilable
/// object except one, so both the weighted and the unweighted path run.
fn fixture_weights(src: &std::path::Path) -> (InputWeights, usize) {
    let index = index_of(src);
    let mut by_tensor = BTreeMap::new();
    let mut skipped = 0;
    for entry in index.representations.values() {
        let (header, _) = read_segment_header(&src.join(&entry.segment)).unwrap();
        for t in header.tensors.iter().filter(|t| t.shape.len() == 2) {
            if skipped == 0 {
                skipped += 1;
                continue;
            }
            by_tensor.insert(
                (entry.object.clone(), t.name.clone()),
                vec![1.0; t.shape[1]],
            );
        }
    }
    (
        InputWeights {
            by_tensor,
            digest: "fixture-weights".into(),
        },
        skipped,
    )
}

fn compile_weighted(
    tmp: &tempfile::TempDir,
    quirk: Quirk,
    encoding: Option<&str>,
) -> (
    std::path::PathBuf,
    Result<RepresentReport, VindexError>,
    usize,
) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("weighted.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let encoders = EncoderRegistry::new()
        .register(Box::new(QuirkyEncoder(quirk)))
        .unwrap();
    let spec = RepresentSpec {
        encoding: encoding
            .unwrap_or(QuirkyEncoder(quirk).encoding_label())
            .to_string(),
        ..RepresentSpec::nvfp4()
    };
    let (weights, skipped) = fixture_weights(&src);
    let result = compile_representation_weighted(&src, &out, &spec, &encoders, &weights);
    (out, result, skipped)
}

mod deployment_images;
mod physical_residency;
mod physical_store;
mod tests_basics;
mod the_k_quant_path_end_to_end;
mod the_loader_ladder;
