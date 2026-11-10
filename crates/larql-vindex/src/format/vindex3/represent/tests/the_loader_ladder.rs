//! The loader ladder

use super::*;

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
            ..
        },
        LoadedWeight::Nvfp4 {
            packed: tp,
            scales: ts,
            tensor_scale: tt,
            ..
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

/// Open `src` under `source` and ask the loader for `format` on its
/// first compiled tensor.
fn load_compiled_tensor_as(
    source: RepresentationSource,
    format: WeightFormat,
) -> (Result<LoadedWeight, VindexError>, u64) {
    let tmp = tempfile::tempdir().unwrap();
    let (src, _, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);
    let inspection = inspect_container(&src, false).unwrap();
    let store = OperandStore::open_for(&src, &inspection, Some(DTYPE_NVFP4), source).unwrap();
    let loaded = load_weight(
        (&store).into(),
        &OperandRef {
            object,
            tensor,
            dtype,
            shape,
        },
        format,
    );
    (loaded, store.runtime_quantised())
}

#[test]
fn q8_and_q4_residency_are_runtime_quantisation_and_stored_refuses_them() {
    // #665: these arms quantised at load without saying so, so a stored
    // run reported "served entirely from stored bytes" over a head it had
    // re-quantised. They are counted like every other manufactured form,
    // and refused where manufacturing is forbidden.
    for format in [WeightFormat::Q8, WeightFormat::Q4] {
        let (loaded, n) = load_compiled_tensor_as(RepresentationSource::Auto, format);
        loaded.unwrap_or_else(|e| panic!("{format:?} under auto: {e}"));
        assert_eq!(n, 1, "{format:?} under auto is one runtime quantisation");

        let (loaded, n) = load_compiled_tensor_as(RepresentationSource::Stored, format);
        let err = match loaded {
            Ok(_) => panic!("{format:?} under stored must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("would be quantised at load"),
            "{format:?}: {err}"
        );
        assert_eq!(n, 0, "{format:?}: a refusal manufactured nothing");
    }
}
