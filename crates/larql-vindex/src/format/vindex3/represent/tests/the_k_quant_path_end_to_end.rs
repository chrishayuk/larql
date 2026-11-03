//! The K-quant path, end to end

use super::*;

/// The fixture's rows are 64 wide (projections) and 256 wide
/// (`down_proj`), so Q8_0's 32-element blocks divide every one of them.
/// This is the full path: plan, encode, seal, re-open, decode.
#[test]
fn a_q8_0_representation_compiles_binds_and_decodes() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("q8.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let report = compile_representation(&src, &out, &kquant_spec("Q8_0"))
        .expect("every fixture row divides by 32");
    assert!(
        !report.compiled_objects.is_empty(),
        "nothing compiled: {report:?}"
    );

    let index = index_of(&out);
    let (rep_id, entry) = index
        .representations
        .iter()
        .find(|(_, e)| e.encoding == "Q8_0")
        .expect("a Q8_0 representation was registered");

    // Provenance: the pack says which contract decodes it and which
    // encoder chose its values, and those are different questions.
    let codec = entry
        .codec
        .as_ref()
        .expect("a K-quant pack declares a codec");
    assert_eq!(*codec, kquant::Q8_0.codec_identity());
    codec
        .admit()
        .expect("this build implements the contract it just wrote");
    // The provenance must name whichever encoder actually ran, and the
    // two builds must not be able to claim each other's. This is the
    // architectural control made observable in the artifact: the feature
    // being compiled in IS the switch, so a comparative campaign that
    // forgot to enable it produces a pack that SAYS so.
    let recipe = entry.encoder.as_ref().expect("an encoder recipe");
    #[cfg(not(feature = "reference-encoder"))]
    {
        assert_eq!(
            recipe,
            &nvfp4_pack::EncoderRecipe::kquant_native_v2(),
            "a default build encodes with LARQL's own K-quant encoder"
        );
        assert!(
            recipe.source.is_none(),
            "a native encode has no upstream to pin"
        );
    }
    #[cfg(feature = "reference-encoder")]
    {
        assert_eq!(
            recipe.algorithm, "kquant-ggml-reference",
            "a reference-encoder build must record that ggml chose the values"
        );
        assert!(
            recipe.source.is_some(),
            "a reference encode records the upstream it was pinned to, or the \
             comparison it supports is not reproducible"
        );
        assert_ne!(
            recipe,
            &nvfp4_pack::EncoderRecipe::kquant_native_v2(),
            "the two encoders must not be able to claim each other's provenance"
        );
    }

    // It is actually smaller: BF16 is 16 bits, Q8_0 is 8.5.
    let source = index
        .representations
        .get(entry.compiled_from.as_ref().expect("compiled_from"))
        .expect("the source representation survives");
    assert!(
        entry.payload_bytes < source.payload_bytes,
        "{rep_id}: {} is not smaller than its source {}",
        entry.payload_bytes,
        source.payload_bytes
    );

    // And the executor can bind it: values come back through the
    // K-quant branch of `OperandStore::load`, not through `widen`.
    let inspection = inspect_container(&out, false).unwrap();
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some("Q8_0"),
        crate::format::vindex3::opplan::exec::operands::RepresentationSource::Stored,
    )
    .expect("the compiled pack binds");

    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    let quantised = header
        .tensors
        .iter()
        .find(|t| t.dtype == "Q8_0")
        .expect("the pack holds at least one Q8_0 tensor");
    let values = store
        .load(&OperandRef {
            object: entry.object.clone(),
            tensor: quantised.name.clone(),
            dtype: quantised.dtype.clone(),
            shape: quantised.shape.clone(),
        })
        .expect("a Q8_0 operand decodes");
    let expected: usize = quantised.shape.iter().product();
    assert_eq!(values.len(), expected, "{}", quantised.name);
    assert!(
        values.iter().any(|v| *v != 0.0),
        "{} decoded to all zeros — the pack is not being read",
        quantised.name
    );
}

/// Q6_K frames 256 values per block, so it fits `down_proj`'s 256-wide
/// rows and NOT the 64-wide projections. The compiler must take the one
/// and carry the others rather than refusing the object whole — the
/// row-geometry rule acting as an eligibility filter, not an error.
#[test]
fn a_q6_k_map_compiles_only_the_rows_whose_geometry_holds_it() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("q6.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let report = compile_representation(&src, &out, &kquant_spec("Q6_K"))
        .expect("down_proj's 256-wide rows hold Q6_K");
    let compiled: usize = report
        .compiled_objects
        .iter()
        .map(|o| o.compiled_tensors)
        .sum();
    let carried: usize = report
        .compiled_objects
        .iter()
        .map(|o| o.carried_tensors)
        .sum();
    assert!(compiled > 0, "down_proj should have compiled: {report:?}");
    assert!(
        carried > 0,
        "the 64-wide projections cannot hold a 256-element block and must be carried: {report:?}"
    );

    // Every compiled tensor's row really does divide by 256.
    let index = index_of(&out);
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == "Q6_K")
        .expect("a Q6_K representation was registered");
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    let mut seen = 0usize;
    for t in header.tensors.iter().filter(|t| t.dtype == "Q6_K") {
        let row = *t.shape.last().unwrap();
        assert_eq!(row % 256, 0, "{} has a {row}-wide row", t.name);
        seen += 1;
    }
    assert!(seen > 0, "no Q6_K tensor in the pack");
}

/// An encoding with no compiler is refused by name, and the refusal
/// lists what IS available rather than a stale literal.
#[test]
fn an_uncompilable_kquant_names_the_ones_that_are() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("q3.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let err = compile_representation(&src, &out, &kquant_spec("Q3_K"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no representation compiler"), "{err}");
    for name in ["Q8_0", "Q6_K", "Q4_K"] {
        assert!(err.contains(name), "the refusal must list {name}: {err}");
    }
}

#[test]
fn the_q8_arm_binds_the_stored_pack_for_a_q8_activation() {
    use crate::format::vindex3::opplan::exec::backend::{Nvfp4Activation, WeightSlice};
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let op = operand_of(&src);
    let (f32_arm, _) =
        load_as(&out, RepresentationSource::Stored, &op, WeightFormat::Nvfp4).unwrap();
    let (q8_arm, store) = load_as(
        &out,
        RepresentationSource::Stored,
        &op,
        WeightFormat::Nvfp4Q8,
    )
    .unwrap();
    assert_eq!(f32_arm.format(), WeightFormat::Nvfp4);
    assert_eq!(q8_arm.format(), WeightFormat::Nvfp4Q8);
    let (
        LoadedWeight::Nvfp4 {
            packed: ap,
            scales: a_s,
            tensor_scale: at,
            ..
        },
        LoadedWeight::Nvfp4 {
            packed: bp,
            scales: bs,
            tensor_scale: bt,
            activation,
        },
    ) = (&f32_arm, &q8_arm)
    else {
        panic!("both arms bind the NVFP4 pack");
    };
    assert_eq!(
        &ap.as_slice()[..ap.logical_len()],
        &bp.as_slice()[..bp.logical_len()]
    );
    assert_eq!(
        &a_s.as_slice()[..a_s.logical_len()],
        &bs.as_slice()[..bs.logical_len()]
    );
    assert_eq!(at.to_bits(), bt.to_bits());
    assert_eq!(*activation, Nvfp4Activation::Q8);
    let WeightSlice::Nvfp4 {
        activation: sliced, ..
    } = q8_arm.slice()
    else {
        panic!("an NVFP4 binding slices as NVFP4");
    };
    assert_eq!(sliced, Nvfp4Activation::Q8);
    assert_eq!(store.runtime_quantised(), 0, "the Q8 arm quantised at load");
}

#[test]
fn the_q8_arm_keeps_a_source_precision_binding() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, _, _) = compiled_pair(&tmp);
    let op = operand_of(&src);
    let (loaded, store) = load_as(
        &src,
        RepresentationSource::Stored,
        &op,
        WeightFormat::Nvfp4Q8,
    )
    .unwrap();
    assert!(matches!(loaded, LoadedWeight::F16(_)));
    assert_eq!(store.runtime_quantised(), 0);
    assert_eq!(store.bound_at_stored_precision(), 1);
}

#[test]
fn the_q8_arm_refuses_to_quantise_at_load() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let op = operand_of(&src);
    let err = match load_as(
        &out,
        RepresentationSource::Transient,
        &op,
        WeightFormat::Nvfp4Q8,
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a transient request would quantise at load and must be refused"),
    };
    assert!(
        err.contains("NVFP4 x Q8") && err.contains("quantised at load"),
        "{err}"
    );
}

#[test]
fn a_registered_encoder_compiles_a_pack_of_its_own_bytes() {
    use codec::encoder::tests::RawF32Codec;
    use codec::RepresentationCodec;
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, report) = encoder_pair(&tmp);
    assert!(!report.compiled_objects.is_empty());

    let label = RawF32Codec.encoding_label();
    let src_index = index_of(&src);
    let out_index = index_of(&out);
    let mut checked = 0usize;
    for entry in out_index.representations.values() {
        if entry.encoding != label {
            continue;
        }
        assert_eq!(entry.codec.as_ref(), Some(&RawF32Codec.identity()));
        assert_eq!(
            entry.encoder.as_ref(),
            Some(&EncoderRecipe::codec(&RawF32Codec.identity()))
        );
        let source_entry = &src_index.representations[entry.compiled_from.as_ref().unwrap()];
        let read = |dir: &std::path::Path, segment: &str| {
            let (header, start) = read_segment_header(&dir.join(segment)).unwrap();
            let bytes = std::fs::read(dir.join(segment)).unwrap();
            header
                .tensors
                .into_iter()
                .map(|t| {
                    let at = (start + t.offset) as usize;
                    (t.name, (t.dtype, bytes[at..at + t.len as usize].to_vec()))
                })
                .collect::<BTreeMap<_, _>>()
        };
        let packed = read(&out, &entry.segment);
        let source = read(&src, &source_entry.segment);
        for (name, (dtype, bytes)) in &packed {
            if dtype == label {
                assert_eq!(source[name].0, "F32", "{name}");
                assert_eq!(
                    bytes, &source[name].1,
                    "{name}: raw f32 is the source bytes"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "no tensor was compiled into the encoder's pack"
    );
}

#[test]
fn a_registered_encoders_pack_is_marked_approximate_in_the_graph() {
    use codec::encoder::tests::RawF32Codec;
    use codec::RepresentationCodec;
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = encoder_pair(&tmp);
    let graph: super::super::super::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    for compiled in &report.compiled_objects {
        let object = graph
            .objects
            .iter()
            .find(|o| o.id == compiled.object)
            .unwrap();
        assert!(object.representations.iter().any(|r| {
            r.encoding == RawF32Codec.encoding_label() && r.fidelity == Fidelity::Approximate
        }));
    }
}

#[test]
fn an_unknown_encoding_names_the_registered_encoders() {
    use codec::encoder::tests::RawF32Codec;
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("out.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let encoders = EncoderRegistry::new()
        .register(Box::new(RawF32Codec))
        .unwrap();
    let spec = RepresentSpec {
        encoding: "NOT_AN_ENCODING".into(),
        ..RepresentSpec::nvfp4()
    };
    let err = compile_representation_with(&src, &out, &spec, &encoders)
        .unwrap_err()
        .to_string();
    assert!(err.contains("registered encoders: TEST_RAW_F32"), "{err}");
    assert!(!out.exists());
}

#[test]
fn an_instance_sized_encoder_is_encoded_while_planning() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result) = compile_quirky(&tmp, Quirk::InstanceSized);
    let report = result.expect("an instance-sized encoder still compiles");
    let compiled: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.compiled_tensors)
        .sum();
    assert!(compiled > 0);
    // The segment table was planned from the encoded lengths, so every
    // compiled tensor's recorded length is what the encoder wrote.
    let index = index_of(&out);
    let label = QuirkyEncoder(Quirk::InstanceSized).encoding_label();
    for entry in index
        .representations
        .values()
        .filter(|e| e.encoding == label)
    {
        let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
        for t in header.tensors.iter().filter(|t| t.dtype == label) {
            let elements: usize = t.shape.iter().product();
            assert_eq!(t.len as usize, elements * 4, "{}", t.name);
        }
    }
}

#[test]
fn an_encoder_that_refuses_every_shape_compiles_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result) = compile_quirky(&tmp, Quirk::RefusesEveryShape);
    let err = result.unwrap_err().to_string();
    assert!(err.contains("eligible"), "{err}");
    assert!(!out.join(INDEX_JSON).exists());
}

#[test]
fn an_encoder_whose_bytes_contradict_its_stated_length_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result) = compile_quirky(&tmp, Quirk::MisstatesItsLength);
    let err = result.unwrap_err().to_string();
    assert!(err.contains("its codec states"), "{err}");
    assert!(!out.join(INDEX_JSON).exists());
}

#[test]
fn weights_reach_the_encoder_and_the_recipe_names_their_digest() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result, skipped) = compile_weighted(&tmp, Quirk::AcceptsWeights, None);
    let report = result.expect("a weighted compile");
    let compiled: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.compiled_tensors)
        .sum();
    let weighted: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.weighted_tensors)
        .sum();
    assert!(weighted > 0);
    assert!(
        weighted <= compiled - skipped.min(compiled),
        "a tensor with no weights is encoded unweighted and not counted"
    );
    let identity = QuirkyEncoder(Quirk::AcceptsWeights).identity();
    let packs: Vec<_> = index_of(&out)
        .representations
        .into_values()
        .filter(|e| e.compiled_from.is_some())
        .collect();
    assert!(!packs.is_empty());
    for e in packs {
        assert_eq!(
            e.encoder,
            Some(EncoderRecipe::codec_weighted(&identity, "fixture-weights"))
        );
    }
}

#[test]
fn an_encoder_without_weighting_refuses_rather_than_ignores_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result, _) = compile_weighted(&tmp, Quirk::InstanceSized, None);
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("does not accept input-feature weights"),
        "{err}"
    );
    assert!(!out.join(INDEX_JSON).exists());
}

#[test]
fn a_shipped_compiler_refuses_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let (out, result, _) = compile_weighted(&tmp, Quirk::AcceptsWeights, Some(DTYPE_NVFP4));
    let err = result.unwrap_err().to_string();
    assert!(err.contains("shipped compiler"), "{err}");
    assert!(!out.exists());
}
