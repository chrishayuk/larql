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

#[test]
fn compiled_bytes_equal_what_the_loader_would_have_quantised() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);

    // The transient path: open the SOURCE container and quantise at load,
    // exactly as `--backend metal-nvfp4` does today.
    let src_inspection = inspect_container(&src, false).unwrap();
    let src_store = OperandStore::open(&src, &src_inspection).unwrap();
    let src_index = index_of(&src);

    let out_index = index_of(&out);
    let mut checked = 0usize;

    for (rep_id, entry) in &out_index.representations {
        if entry.encoding != DTYPE_NVFP4 {
            continue;
        }
        let (header, payload_start) = read_segment_header(&out.join(&entry.segment)).unwrap();
        let mut file = std::fs::File::open(out.join(&entry.segment)).unwrap();

        let from = entry
            .compiled_from
            .as_ref()
            .expect("compiled pack names its source");
        let source_entry = src_index
            .representations
            .get(from)
            .expect("the source representation survives compilation");
        let (src_header, _) = read_segment_header(&src.join(&source_entry.segment)).unwrap();

        // Byte-equality is only claimable against the encoder that wrote
        // the pack. This build wrote it, so the claim holds here — but the
        // precondition is asserted rather than assumed, so a future encoder
        // change turns this into a clear message instead of a diff.
        let recipe = entry
            .encoder
            .as_ref()
            .expect("a compiled pack names its encoder");
        assert!(
            recipe.is_reproducible_by_this_build(),
            "pack was compiled by {}; this build compiles with {} — compare \
             behaviour, not bytes",
            recipe.name(),
            nvfp4_pack::EncoderRecipe::current().name()
        );

        for t in &header.tensors {
            if t.dtype != DTYPE_NVFP4 {
                continue;
            }
            let layout = PackLayout::derive(&t.shape, &t.name).unwrap();

            // Stored.
            let mut payload = vec![0u8; t.len as usize];
            file.seek(SeekFrom::Start(payload_start + t.offset))
                .unwrap();
            file.read_exact(&mut payload).unwrap();
            let (packed, scales, tensor_scale) = split(&payload, &layout, &t.name).unwrap();

            // Transient, from the canonical source representation.
            let src_tensor = src_header
                .tensors
                .iter()
                .find(|s| s.name == t.name)
                .expect("the pack carries the same tensor names");
            let loaded = load_weight(
                (&src_store).into(),
                &OperandRef {
                    object: entry.object.clone(),
                    tensor: src_tensor.name.clone(),
                    dtype: src_tensor.dtype.clone(),
                    shape: src_tensor.shape.clone(),
                },
                WeightFormat::Nvfp4,
            )
            .unwrap();
            let LoadedWeight::Nvfp4 {
                packed: want_packed,
                scales: want_scales,
                tensor_scale: want_scale,
            } = &loaded
            else {
                panic!("asked for NVFP4, got another format");
            };

            // Compare the logical prefixes: `as_slice` exposes the
            // page-aligned allocation, and comparing padded-to-padded would
            // pass even if the pack persisted the padding.
            assert_eq!(
                packed,
                &want_packed.as_slice()[..want_packed.logical_len()],
                "{rep_id}/{}: stored codes differ from load-time codes",
                t.name
            );
            assert_eq!(
                scales,
                &want_scales.as_slice()[..want_scales.logical_len()],
                "{rep_id}/{}: stored group scales differ",
                t.name
            );
            assert_eq!(
                tensor_scale.to_bits(),
                want_scale.to_bits(),
                "{rep_id}/{}: stored tensor scale differs",
                t.name
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "the fixture compiled no NVFP4 tensors to check"
    );
}

#[test]
fn the_canonical_representation_survives_byte_for_byte() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let src_index = index_of(&src);
    let out_index = index_of(&out);

    for (id, src_entry) in &src_index.representations {
        let out_entry = out_index
            .representations
            .get(id)
            .unwrap_or_else(|| panic!("canonical representation {id} was dropped"));
        // Compiling an alternative must not touch what it was derived from:
        // the source stays bit-authoritative and its hashes prove it.
        assert_eq!(out_entry.payload_sha256, src_entry.payload_sha256, "{id}");
        assert_eq!(out_entry.segment_sha256, src_entry.segment_sha256, "{id}");
        assert_eq!(out_entry.encoding, src_entry.encoding, "{id}");
        assert!(out_entry.compiled_from.is_none(), "{id} is not compiled");
    }
}

#[test]
fn the_pack_is_smaller_and_names_its_provenance() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    assert!(!report.compiled_objects.is_empty());
    for compiled in &report.compiled_objects {
        assert!(
            compiled.compiled_bytes < compiled.source_bytes,
            "{}: {} compiled vs {} source",
            compiled.object,
            compiled.compiled_bytes,
            compiled.source_bytes
        );
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .expect("compiled representation is in the index");
        assert_eq!(entry.encoding, DTYPE_NVFP4);
        // Provenance a derived pack cannot otherwise state.
        let from = entry.compiled_from.as_deref().expect("names its source");
        assert!(
            index.representations.contains_key(from),
            "provenance {from} must resolve inside the same container"
        );
    }
}

#[test]
fn the_graph_marks_the_compiled_representation_approximate() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();

    let compiled: BTreeSet<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();
    let mut seen = 0usize;
    for object in &graph.objects {
        if !compiled.contains(object.id.as_str()) {
            continue;
        }
        let rep = object
            .representations
            .iter()
            .find(|r| r.encoding == DTYPE_NVFP4)
            .expect("the graph learns the object gained a materialisation");
        // Canonical is what the release ships as ground truth; a 4-bit
        // re-encoding is not that, and mislabelling it would let a profile
        // select an approximation believing it authoritative.
        assert_eq!(rep.fidelity, Fidelity::Approximate);
        assert!(
            object
                .representations
                .iter()
                .any(|r| r.fidelity == Fidelity::Canonical),
            "{}: the canonical representation is still declared",
            object.id
        );
        seen += 1;
    }
    assert_eq!(seen, compiled.len());
}

#[test]
fn non_matrix_tensors_are_carried_verbatim() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    // Norms and biases are 1-D: NVFP4 does not apply, and dropping them
    // would leave a pack its consumers had to patch from elsewhere.
    let carried: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.carried_tensors)
        .sum();
    assert!(carried > 0, "the fixture has 1-D tensors to carry");

    for compiled in &report.compiled_objects {
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .unwrap();
        let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
        assert_eq!(
            header.tensors.len(),
            compiled.compiled_tensors + compiled.carried_tensors,
            "{}: pack is complete",
            compiled.object
        );
        for t in &header.tensors {
            if t.dtype != DTYPE_NVFP4 {
                assert_ne!(t.shape.len(), 2, "a 2-D tensor was carried, not compiled");
            }
        }
    }
}

#[test]
fn an_unknown_encoding_is_refused_before_anything_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mxfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let spec = RepresentSpec {
        encoding: "MXFP4".into(),
        objects: Vec::new(),
        roles: policy::RolePolicy::default(),
        deployment: false,
        protect: policy::Protections::default(),
    };
    let err = compile_representation(&src, &out, &spec)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no representation compiler"), "{err}");
    assert!(
        !out.exists(),
        "a refused compile leaves no half-written container"
    );
}

#[test]
fn compiling_twice_does_not_quantise_a_quantised_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, first) = compiled_pair(&tmp);
    let twice = tmp.path().join("twice.vindex3");
    let second = compile_representation(&out, &twice, &RepresentSpec::nvfp4()).unwrap_err();

    // Every object already carries the target encoding, so there is nothing
    // left to compile — and re-encoding an approximation would compound the
    // error silently.
    assert!(
        second.to_string().contains("nothing was compiled"),
        "{second}"
    );
    assert!(!first.compiled_objects.is_empty());
}

#[test]
fn an_object_filter_compiles_only_what_it_names() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("one.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let all = index_of(&src);
    let target = all
        .representations
        .values()
        .map(|e| e.object.clone())
        .find(|o| o.contains("decoder_stack"))
        .expect("the fixture has a decoder stack");

    let spec = RepresentSpec {
        encoding: DTYPE_NVFP4.to_string(),
        objects: vec![target.clone()],
        roles: policy::RolePolicy::default(),
        deployment: false,
        protect: policy::Protections::default(),
    };
    let report = compile_representation(&src, &out, &spec).unwrap();
    assert_eq!(report.compiled_objects.len(), 1);
    assert_eq!(report.compiled_objects[0].object, target);
}

#[test]
fn a_model_whose_k_is_not_group_aligned_is_refused_whole() {
    // NVFP4 groups 16 elements. The miniature Glimmer fixture is hidden=12,
    // ffn=20 — nothing in it can be grouped, and the refusal names that
    // rather than compiling a container with an empty pack in it.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(miniature_glimmer, &checkpoint, &src, "target");

    let err = compile_representation(&src, &out, &RepresentSpec::nvfp4())
        .unwrap_err()
        .to_string();
    assert!(err.contains("nothing was compiled"), "{err}");
    assert!(!out.join(INDEX_JSON).exists(), "no container was written");
}

#[test]
fn the_pack_lives_under_segments_and_is_declared() {
    // Two ways a pack goes missing: written outside `segments/`, or written
    // there but absent from `index.segments`, which
    // `Vindex3Container::segment` refuses to resolve.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    for compiled in &report.compiled_objects {
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .unwrap();
        assert!(
            entry.segment.starts_with("segments/"),
            "pack at {} is outside segments/",
            entry.segment
        );
        assert!(entry.segment.ends_with(".bin"), "{}", entry.segment);
        assert!(
            out.join(&entry.segment).is_file(),
            "{} does not exist on disk",
            entry.segment
        );

        let key = entry.segment.trim_end_matches(".bin");
        assert!(
            index.segments.contains_key(key),
            "segment key {key} is not declared; declared: {:?}",
            index.segments.keys().collect::<Vec<_>>()
        );
    }

    // Nothing stray at the container root.
    for e in std::fs::read_dir(&out).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.ends_with(".lyrw") && !name.contains("@"),
            "stray segment at the container root: {name}"
        );
    }
}

#[test]
fn the_default_policy_preserves_the_embedding_and_the_head() {
    // Regression for the shape-based default: an embedding table is a 2-D
    // matrix, and "matrix implies quantise" 4-bits one of the surfaces
    // where 4-bit is least safe. Role decides now.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    let compiled: Vec<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();
    assert!(
        compiled.iter().any(|o| o.contains("decoder_stack")),
        "decoder linear weights are the point of the default policy"
    );

    // The embedding object may be compiled only if some tensor in it was
    // eligible — under the default, none is.
    for c in &report.compiled_objects {
        if c.object.contains("embedding") || c.object.contains("output_head") {
            panic!("{} was compiled under the conservative default", c.object);
        }
    }
    // And no NVFP4 representation exists for it in the index.
    for (id, e) in &index.representations {
        if e.encoding == DTYPE_NVFP4 {
            assert!(
                !id.contains("embedding") && !id.contains("output_head"),
                "{id} should not have a compiled pack by default"
            );
        }
    }
}

#[test]
fn opting_the_embedding_in_compiles_it() {
    // The escape hatch works, and only for the role named.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("aggressive.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.roles = spec.roles.clone().including(policy::Role::Embedding);
    let report = compile_representation(&src, &out, &spec).unwrap();

    assert!(
        report
            .compiled_objects
            .iter()
            .any(|c| c.object.contains("embedding")),
        "explicit opt-in must compile the embedding"
    );
}

#[test]
fn the_report_names_what_the_policy_protected() {
    // A conservative default is only trustworthy if it says what it
    // conserved; a silent policy is indistinguishable from no policy.
    let tmp = tempfile::tempdir().unwrap();
    let (_, _, report) = compiled_pair(&tmp);
    let stack = report
        .compiled_objects
        .iter()
        .find(|c| c.object.contains("decoder_stack"))
        .expect("the stack is compiled");
    assert!(
        !stack.preserved.is_empty(),
        "the decoder stack carries norms the policy preserved"
    );
    let protected: usize = stack.preserved.values().sum();
    assert_eq!(protected, stack.carried_tensors);
}

#[test]
fn a_compiled_pack_records_the_abi_it_was_compiled_against() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, report) = compiled_pair(&tmp);
    let src_index = index_of(&src);
    let index = index_of(&out);

    for c in &report.compiled_objects {
        let entry = index.representations.get(&c.representation_id).unwrap();
        let codec = entry
            .codec
            .as_ref()
            .expect("a compiled pack states its ABI");
        assert_eq!(codec.family, "nvfp4");
        assert_eq!(codec.revision, nvfp4_pack::CodecIdentity::REVISION);
        assert_eq!(codec.group_elems, 16);
        codec.admit().expect("this build implements what it wrote");

        // Provenance survives being copied out of the container.
        let from = entry.compiled_from.as_deref().unwrap();
        let digest = entry
            .source_representation_digest
            .as_deref()
            .expect("a derived pack names the bytes it derives from");
        assert_eq!(digest, src_index.representations[from].payload_sha256);
    }

    // Source-encoded representations carry no ABI: their bytes are the
    // checkpoint's, not this compiler's.
    for (id, e) in &index.representations {
        if e.encoding != DTYPE_NVFP4 {
            assert!(e.codec.is_none(), "{id} should not claim a compiler ABI");
        }
    }
}

#[test]
fn a_future_abi_revision_is_refused_rather_than_decoded() {
    // The whole point: an improved `quantize_nvfp4` must not silently
    // redefine containers already on disk.
    let mut future = nvfp4_pack::CodecIdentity::nvfp4_v1();
    future.revision = nvfp4_pack::CodecIdentity::REVISION + 1;
    let err = future.admit().unwrap_err().to_string();
    assert!(err.contains("another build"), "{err}");
    assert!(err.contains("Recompile"), "{err}");

    // A family nothing registers is refused as unknown. (`mxfp4` was the
    // alien here until the codec registry admitted it as a family of its
    // own; NVFP4's geometry under that name is now a geometry
    // disagreement, asserted below, not an unknown family.)
    let mut alien = nvfp4_pack::CodecIdentity::nvfp4_v1();
    alien.family = "vq-codebook".into();
    assert!(alien.admit().unwrap_err().to_string().contains("is not"));
    let mut misfiled = nvfp4_pack::CodecIdentity::nvfp4_v1();
    misfiled.family = "mxfp4".into();
    let err = misfiled.admit().unwrap_err().to_string();
    assert!(err.contains("disagrees with its own revision"), "{err}");

    // Same revision, disagreeing geometry: a corrupted or hand-edited
    // index, and named differently so it is not mistaken for version skew.
    let mut bad = nvfp4_pack::CodecIdentity::nvfp4_v1();
    bad.group_elems = 32;
    let err = bad.admit().unwrap_err().to_string();
    assert!(err.contains("disagrees with its own revision"), "{err}");
}

#[test]
fn the_encoder_recipe_is_recorded_apart_from_the_codec_abi() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    for c in &report.compiled_objects {
        let e = index.representations.get(&c.representation_id).unwrap();
        let codec = e.codec.as_ref().unwrap();
        let encoder = e.encoder.as_ref().expect("a pack names its encoder");
        assert_eq!(encoder.name(), "nvfp4-nearest-v1");
        assert!(encoder.is_reproducible_by_this_build());
        // Two separate identities, not one: same decode contract, and a
        // recipe that may change under it.
        assert_eq!(codec.family, "nvfp4");
        assert_ne!(codec.family, encoder.algorithm);
    }
}

#[test]
fn a_different_encoder_is_not_a_refusal_only_a_weaker_claim() {
    // A GPTQ pack decodes through the same kernel and is entirely valid;
    // it simply is not byte-reproducible by a nearest-rounding build.
    // Treating that as corruption would make every encoder improvement a
    // breaking change.
    let gptq = nvfp4_pack::EncoderRecipe {
        algorithm: "nvfp4-gptq".into(),
        revision: 1,
        source: None,
    };
    assert!(!gptq.is_reproducible_by_this_build());
    assert_eq!(gptq.name(), "nvfp4-gptq-v1");

    // The decode contract is unaffected — that is the whole point of
    // keeping the two identities apart.
    nvfp4_pack::CodecIdentity::nvfp4_v1()
        .admit()
        .expect("codec admission does not depend on the encoder recipe");

    let newer = nvfp4_pack::EncoderRecipe {
        algorithm: "nvfp4-nearest".into(),
        revision: 2,
        source: None,
    };
    assert!(!newer.is_reproducible_by_this_build());
}

// ── The loader ladder ────────────────────────────────────────────────────
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

#[test]
fn stored_and_transient_bind_identical_weights() {
    // Same encoder recipe wrote the pack, so this is a byte claim, not a
    // KL claim. If it ever fails, something in selection, layout or
    // loading differs — and a small numerical difference would be the
    // *wrong* thing to accept here.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let (stored, stored_n) = load_under(
        &out,
        RepresentationSource::Stored,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    let (transient, transient_n) = load_under(
        &out,
        RepresentationSource::Transient,
        &object,
        &tensor,
        &dtype,
        &shape,
    );

    let (
        LoadedWeight::Nvfp4 {
            packed: sp,
            scales: ss,
            tensor_scale: st,
        },
        LoadedWeight::Nvfp4 {
            packed: tp,
            scales: ts,
            tensor_scale: tt,
        },
    ) = (&stored, &transient)
    else {
        panic!("both arms must bind NVFP4");
    };
    assert_eq!(
        &sp.as_slice()[..sp.logical_len()],
        &tp.as_slice()[..tp.logical_len()],
        "codes differ between the stored pack and a fresh quantisation"
    );
    assert_eq!(
        &ss.as_slice()[..ss.logical_len()],
        &ts.as_slice()[..ts.logical_len()],
        "group scales differ"
    );
    assert_eq!(st.to_bits(), tt.to_bits(), "tensor scale differs");

    // And the counter proves the two arms got there by different routes —
    // otherwise this test would pass even if `stored` silently quantised.
    assert_eq!(stored_n, 0, "stored mode quantised at load");
    assert_eq!(transient_n, 1, "transient mode did not invoke the encoder");
}

#[test]
fn transient_ignores_a_present_pack_and_still_encodes() {
    // `transient` is the oracle the compiler is checked against. An arm
    // that fell through to a convenient pack would silently stop being
    // one, and every parity result after that would be vacuous.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let (_, n) = load_under(
        &out,
        RepresentationSource::Transient,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(n, 1, "the pack exists, and transient must encode anyway");
}

#[test]
fn stored_binds_a_float_tensor_at_its_own_precision_and_manufactures_nothing() {
    // A compiled pack is a precision map, and a backend arm names a format
    // per class, which cannot express one. Under `stored` the map wins: a
    // tensor held at source precision runs there — higher than the arm
    // asked for — and nothing is manufactured.
    //
    // This replaces an earlier refusal. Refusing made a per-projection
    // precision map unrunnable: protecting `q_proj` while the arm demanded
    // NVFP4 attention could never execute, so the map could be compiled and
    // never measured.
    let tmp = tempfile::tempdir().unwrap();
    let (src, _, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let inspection = inspect_container(&src, false).unwrap();
    let store = OperandStore::open_for(
        &src,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .unwrap();

    let loaded = load_weight(
        (&store).into(),
        &OperandRef {
            object,
            tensor,
            dtype,
            shape,
        },
        WeightFormat::Nvfp4,
    )
    .expect("stored precision is a binding, not a refusal");

    assert!(
        matches!(loaded, LoadedWeight::F16(_)),
        "a BF16 tensor binds as f16, not as a fresh quantisation"
    );
    assert_eq!(store.runtime_quantised(), 0, "nothing was manufactured");
    assert_eq!(
        store.bound_at_stored_precision(),
        1,
        "and the run says it ran above the requested format"
    );
}

#[test]
fn stored_binds_a_policy_preserved_object_without_complaint() {
    // The embedding has no pack *by design*. Strict mode must not treat a
    // deliberate protection as a missing artifact, or a conservative role
    // policy and a strict source policy could never be used together.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);
    let inspection = inspect_container(&out, false).unwrap();
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .expect("a preserved object is not a missing pack");

    let sel = store.selection();
    if let Some((_, embed)) = sel.iter().find(|(k, _)| k.contains("embedding")) {
        assert!(!embed.stored, "the embedding has no pack");
        assert_ne!(embed.encoding, DTYPE_NVFP4);
    }
    let stack = sel
        .iter()
        .find(|(k, _)| k.contains("decoder_stack"))
        .unwrap()
        .1;
    assert!(stack.stored, "the stack does have one and must use it");
}

#[test]
fn auto_prefers_the_pack_but_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    // Pack present: used, nothing quantised.
    let (_, n) = load_under(
        &out,
        RepresentationSource::Auto,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(n, 0, "auto did not use the available pack");

    // Pack absent: manufactured rather than refused.
    let (_, n) = load_under(
        &src,
        RepresentationSource::Auto,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(
        n, 1,
        "auto must fall back to encoding when nothing is stored"
    );
}

#[test]
fn selection_reports_which_objects_came_from_a_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);
    let inspection = inspect_container(&out, false).unwrap();

    let stored = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Auto,
    )
    .unwrap();
    let sel = stored.selection();
    let stack = sel
        .iter()
        .find(|(k, _)| k.contains("decoder_stack"))
        .expect("the stack is bound")
        .1;
    assert!(
        stack.stored,
        "the decoder stack has a pack and should use it"
    );
    assert_eq!(stack.encoding, DTYPE_NVFP4);

    // The embedding is preserved by policy, so it has no pack and binds
    // canonically — the selection must say so rather than implying the
    // whole model is 4-bit.
    if let Some((_, embed)) = sel.iter().find(|(k, _)| k.contains("embedding")) {
        assert!(!embed.stored);
        assert_ne!(embed.encoding, DTYPE_NVFP4);
    }
}

// ── Deployment images ────────────────────────────────────────────────────

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

#[test]
fn a_deployment_image_drops_superseded_sources_and_keeps_protected_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = deployment_of(&tmp);
    let index = index_of(&out);

    // Compiled objects: only the pack.
    for c in &report.compiled_objects {
        let reps: Vec<&RepresentationEntry> = index
            .representations
            .values()
            .filter(|e| e.object == c.object)
            .collect();
        assert_eq!(reps.len(), 1, "{} should carry only its pack", c.object);
        assert_eq!(reps[0].encoding, DTYPE_NVFP4);
    }

    // Protected objects still travel — an image missing its BF16 embedding
    // is smaller and does not execute.
    for p in &report.preserved_objects {
        let e = index
            .representations
            .values()
            .find(|e| e.object == p.object)
            .unwrap_or_else(|| panic!("{} must travel with the image", p.object));
        assert_ne!(e.encoding, DTYPE_NVFP4);
        assert!(
            out.join(&e.segment).is_file(),
            "{} bytes are present",
            p.object
        );
    }

    // Every declared segment resolves; nothing points at absent bytes.
    for (id, e) in &index.representations {
        assert!(out.join(&e.segment).is_file(), "{id} -> {}", e.segment);
        let key = e.segment.trim_end_matches(".bin");
        assert!(index.segments.contains_key(key), "{key} undeclared");
    }
}

#[test]
fn a_deployment_image_says_it_is_derived_and_names_its_source() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = deployment_of(&tmp);
    let index = index_of(&out);

    // "Executable, not re-compilable" has to be a claim the artifact makes,
    // not something an operator infers from what is missing.
    assert_eq!(index.authority, ContainerAuthority::Derived);
    assert!(index.derived_from_model.is_some());
    for e in index.representations.values() {
        if e.encoding == DTYPE_NVFP4 {
            assert!(
                e.source_representation_digest.is_some(),
                "a derived pack names the bytes it derives from"
            );
        }
    }
}

#[test]
fn the_graph_of_a_deployment_image_declares_only_present_bytes() {
    // The store binds `representations.first()`. If the graph still listed
    // the dropped canonical encoding, every reader would resolve to a
    // segment that is not in the image.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = deployment_of(&tmp);
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    let index = index_of(&out);
    let compiled: BTreeSet<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();

    for object in &graph.objects {
        if compiled.contains(object.id.as_str()) {
            assert_eq!(object.representations.len(), 1, "{}", object.id);
            assert_eq!(object.representations[0].encoding, DTYPE_NVFP4);
        }
        for r in &object.representations {
            let id = format!("{}@{}", object.id, r.encoding);
            if let Some(e) = index.representations.get(&id) {
                assert!(out.join(&e.segment).is_file(), "{id}");
            }
        }
    }
}

#[test]
fn an_archival_container_is_unaffected_by_the_deployment_switch() {
    // The default must stay archival: canonical bytes present, authority
    // unchanged, so nobody loses their source by omitting a flag.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let index = index_of(&out);
    assert_eq!(index.authority, ContainerAuthority::Canonical);
    assert!(index.derived_from_model.is_none());
    for (id, e) in &index_of(&src).representations {
        assert!(index.representations.contains_key(id), "{id} was dropped");
        assert!(out.join(&e.segment).is_file());
    }
}

#[test]
fn a_protected_projection_is_carried_not_compiled() {
    // The R1 mechanism: an eligible role, held back anyway, so a precision
    // map can be expressed and then measured.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("protected.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("v_proj");
    let report = compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .unwrap();
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    for t in &header.tensors {
        if t.name.contains("v_proj") {
            assert_ne!(t.dtype, DTYPE_NVFP4, "{} was compiled anyway", t.name);
        }
    }
    assert!(header.tensors.iter().any(|t| t.dtype == DTYPE_NVFP4));

    // A protected tensor is bigger than a compiled one, so the pack must
    // grow relative to R0 — the byte cost of the protection is the thing
    // being traded against fidelity.
    let r0_out = tmp.path().join("r0.vindex3");
    let r0 = compile_representation(&src, &r0_out, &RepresentSpec::nvfp4()).unwrap();
    assert!(
        report.compiled_objects[0].compiled_bytes > r0.compiled_objects[0].compiled_bytes,
        "protection costs bytes"
    );
}

#[test]
fn a_protected_depth_range_is_carried_not_compiled() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("early.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().layers(0, 0);
    compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .unwrap();
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    for t in &header.tensors {
        if t.name.starts_with("0.") {
            assert_ne!(t.dtype, DTYPE_NVFP4, "{} is in a protected range", t.name);
        }
    }
    assert!(
        header.tensors.iter().any(|t| t.dtype == DTYPE_NVFP4),
        "later layers still compile"
    );
}

/// The precision map is authority, and both arms must run the SAME
/// program.
///
/// R0 could not see this: its map protects nothing in the decoder stack,
/// so "quantise everything" and "reproduce the map" coincide. The moment a
/// map is mixed they diverge, and the guarantee that stored and transient
/// differ *only* in whether compiled bytes already existed would quietly
/// stop being true.
///
/// The failure this pins is specific: a transient oracle that read
/// `--backend metal-nvfp4-*` as permission to quantise a tensor the map
/// deliberately kept at BF16 would be measuring a different model than the
/// one that was compiled.
#[test]
fn a_mixed_precision_map_runs_identically_on_both_arms() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mixed.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    // q_proj protected, everything else eligible: a genuinely mixed map.
    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("q_proj");
    compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let pack = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .expect("a pack exists");
    let (header, _) = read_segment_header(&out.join(&pack.segment)).unwrap();

    let open = |source| {
        let insp = inspect_container(&out, false).unwrap();
        OperandStore::open_for(&out, &insp, Some(DTYPE_NVFP4), source).unwrap()
    };
    let stored = open(RepresentationSource::Stored);
    let transient = open(RepresentationSource::Transient);

    let mut protected = 0usize;
    let mut compiled = 0usize;

    for t in &header.tensors {
        if t.shape.len() != 2 {
            continue;
        }
        let op = OperandRef {
            object: pack.object.clone(),
            tensor: t.name.clone(),
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
        };
        let a = load_weight((&stored).into(), &op, WeightFormat::Nvfp4).unwrap();
        let b = load_weight((&transient).into(), &op, WeightFormat::Nvfp4).unwrap();

        match (&a, &b) {
            (LoadedWeight::F16(x), LoadedWeight::F16(y)) => {
                // The map protected it: BOTH arms must keep it float, and
                // the bytes must match.
                assert!(t.name.contains("q_proj"), "{} unexpectedly float", t.name);
                assert_eq!(
                    &x.as_slice()[..x.logical_len()],
                    &y.as_slice()[..y.logical_len()],
                    "{}: protected tensor differs between arms",
                    t.name
                );
                protected += 1;
            }
            (
                LoadedWeight::Nvfp4 {
                    packed: p1,
                    scales: s1,
                    tensor_scale: t1,
                },
                LoadedWeight::Nvfp4 {
                    packed: p2,
                    scales: s2,
                    tensor_scale: t2,
                },
            ) => {
                assert!(!t.name.contains("q_proj"), "{} should be protected", t.name);
                assert_eq!(
                    &p1.as_slice()[..p1.logical_len()],
                    &p2.as_slice()[..p2.logical_len()],
                    "{}: codes differ",
                    t.name
                );
                assert_eq!(
                    &s1.as_slice()[..s1.logical_len()],
                    &s2.as_slice()[..s2.logical_len()],
                    "{}: scales differ",
                    t.name
                );
                assert_eq!(
                    t1.to_bits(),
                    t2.to_bits(),
                    "{}: tensor scale differs",
                    t.name
                );
                compiled += 1;
            }
            _ => panic!("{}: arms bound different formats", t.name),
        }
    }

    assert!(
        protected > 0,
        "the fixture must exercise a protected tensor"
    );
    assert!(compiled > 0, "and a compiled one");

    // Manufacture differs by exactly the compiled tensors, and by nothing
    // else — which is the whole content of "the arms differ only in
    // whether the bytes already existed".
    assert_eq!(stored.runtime_quantised(), 0);
    assert_eq!(
        transient.runtime_quantised() as usize,
        compiled,
        "transient manufactured something other than the map's compiled set"
    );
    assert_eq!(stored.bound_at_stored_precision() as usize, protected);
    assert_eq!(transient.bound_at_stored_precision() as usize, protected);
}

#[test]
fn the_container_declares_the_program_that_produced_it() {
    // Authority, not description: a reader can ask *why* a tensor is BF16
    // without having to observe that some pack stored it that way.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mapped.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("v_proj");
    compile_representation(&src, &out, &spec).unwrap();

    let program = index_of(&out)
        .precision_map
        .expect("the container states its program");
    assert_eq!(program.name, "r1-protect-v_proj");
    assert_eq!(program.encoding, DTYPE_NVFP4);
    assert!(program.roles.iter().any(|r| r == "decoder-linear"));

    use crate::format::vindex3::represent::map::Precision;
    assert_eq!(
        program.resolve(policy::Role::DecoderLinear, "0.self_attn.v_proj.weight"),
        Precision::Source
    );
    assert_eq!(
        program.resolve(policy::Role::DecoderLinear, "0.self_attn.k_proj.weight"),
        Precision::Compiled(DTYPE_NVFP4)
    );
    // And the map is a policy, not a transcript: it does not grow with the
    // model it was compiled against.
    assert!(program.exceptions.len() <= 2);
}

#[test]
fn a_pack_that_disagrees_with_the_declared_program_is_refused() {
    // The check `stored` owes. Without it a container could declare one
    // program and execute another, and the declaration would be decoration.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);

    // Claim a program that protects a projection the pack actually compiled.
    let mut index = index_of(&out);
    let mut program = index.precision_map.clone().unwrap();
    program.name = "claims-q-protected".into();
    program.exceptions = policy::Protections::default()
        .projection("q_proj")
        .as_exceptions();
    index.precision_map = Some(program);
    std::fs::write(
        out.join(INDEX_JSON),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();

    let inspection = inspect_container(&out, false).unwrap();
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .unwrap();
    let (object, tensor, dtype, shape) = a_compiled_tensor(&out);
    let err = match load_weight(
        (&store).into(),
        &OperandRef {
            object,
            tensor,
            dtype,
            shape,
        },
        WeightFormat::Nvfp4,
    ) {
        Ok(_) => panic!("a non-conforming pack was executed"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("does not permit"), "{err}");
    assert!(
        err.contains("claims-q-protected"),
        "the refusal names the program: {err}"
    );
}

/// The serialised name and [`Role::name`] are one vocabulary.
///
/// A record persisted by an evidence gate and a role named on a CLI flag
/// have to resolve to the same authority key; two spellings of one role
/// would let a precision map silently fail to govern the tensors an
/// experiment measured.
#[test]
fn the_serde_form_is_the_role_name() {
    for r in super::policy::Role::ALL {
        let json = serde_json::to_string(r).expect("role serialises");
        assert_eq!(
            json,
            format!("\"{}\"", r.name()),
            "{r} serialises differently from its name"
        );
        let back: super::policy::Role = serde_json::from_str(&json).expect("round trips");
        assert_eq!(back, *r);
        assert_eq!(super::policy::Role::parse(r.name()), Some(*r));
    }
}

// ── The K-quant path, end to end ─────────────────────────────────────────
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
            &nvfp4_pack::EncoderRecipe::kquant_native_v1(),
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
            &nvfp4_pack::EncoderRecipe::kquant_native_v1(),
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
