//! `tests` for [`super`].

use super::*;
use crate::format::vindex3::represent::map::Exception;

fn store(id: &str, tensors: &[(&str, &[u8])]) -> Arc<PhysicalStore> {
    let mut bytes = Vec::new();
    let mut table = BTreeMap::new();
    for (name, payload) in tensors {
        table.insert(name.to_string(), (bytes.len() as u64, payload.len() as u64));
        bytes.extend_from_slice(payload);
    }
    Arc::new(PhysicalStore::owned(id, bytes, table))
}

fn operand(tensor: &str) -> OperandRef {
    OperandRef {
        object: "target.expert_bank".into(),
        tensor: tensor.into(),
        dtype: String::new(),
        shape: vec![1, 1],
    }
}

/// Layer 1's experts at Q6_K; everything else source.
fn overlay_map() -> PrecisionMap {
    PrecisionMap {
        name: "kimi-expertweight-layer1-q6k".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((1, 1)),
                encoding: Some("Q6_K".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    }
}

/// **The composition claim.** A model is assembled from a sparse
/// candidate overlay plus the source container, decided by the precision
/// map — and the counts prove which layer answered.
#[test]
fn a_model_is_composed_from_the_overlay_and_the_source() {
    let candidate = store(
        "candidate",
        &[("1.block_sparse_moe.experts.7.w2.weight", b"Q6")],
    );
    let source = store(
        "source",
        &[
            ("1.block_sparse_moe.experts.7.w2.weight", b"BF16-layer1"),
            ("4.block_sparse_moe.experts.7.w2.weight", b"BF16-layer4"),
        ],
    );
    let layered = LayeredOperands::new(overlay_map(), candidate, source);

    let scoped = layered
        .resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.7.w2.weight"),
        )
        .expect("in scope");
    let fallback = layered
        .resolve(
            Role::ExpertWeight,
            &operand("4.block_sparse_moe.experts.7.w2.weight"),
        )
        .expect("out of scope");

    // **Bank identity**, not merely different bytes: the scoped operand
    // must physically come from the candidate store.
    assert_eq!(scoped.store_id(), "candidate");
    assert_eq!(scoped.region.bytes(), b"Q6");
    assert_eq!(fallback.store_id(), "source");
    assert_eq!(fallback.region.bytes(), b"BF16-layer4");

    let s = layered.stats();
    assert_eq!(s.candidate_hits, 1);
    assert_eq!(s.source_fallback_hits, 1);
    assert_eq!(s.missing, 0);
}

/// **The null arm.** A map that compiles nothing resolves everything
/// from the source, so the baseline is the source container exactly —
/// no overlay can leak into it.
#[test]
fn a_source_only_map_never_touches_the_overlay() {
    let candidate = store(
        "candidate",
        &[("1.block_sparse_moe.experts.7.w2.weight", b"Q6")],
    );
    let source = store(
        "source",
        &[("1.block_sparse_moe.experts.7.w2.weight", b"BF16")],
    );
    let layered = LayeredOperands::new(
        PrecisionMap {
            roles: vec![],
            ..overlay_map()
        },
        candidate,
        source,
    );
    let got = layered
        .resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.7.w2.weight"),
        )
        .expect("resolves");
    assert_eq!(got.store_id(), "source", "the null arm must be pure source");
    assert_eq!(got.region.bytes(), b"BF16");
    assert_eq!(layered.stats().candidate_hits, 0);
}

/// **Route escape.** The candidate may select an expert the baseline
/// never routed to, and it must resolve from the candidate bank — the
/// event the quality bank exists to observe, not an error caused by how
/// the artifact was built.
#[test]
fn an_expert_outside_the_baseline_route_still_resolves_from_the_candidate() {
    let mut tensors = Vec::new();
    let names: Vec<String> = (0..256)
        .map(|e| format!("1.block_sparse_moe.experts.{e}.w2.weight"))
        .collect();
    for n in &names {
        tensors.push((n.as_str(), b"Q6".as_slice()));
    }
    let candidate = store("candidate", &tensors);
    let source = store(
        "source",
        &[("4.block_sparse_moe.experts.0.w2.weight", b"BF16")],
    );
    let layered = LayeredOperands::new(overlay_map(), candidate, source);

    // An expert no baseline route touched.
    let escaped = layered
        .resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.181.w2.weight"),
        )
        .expect("every compiled expert must be addressable");
    assert_eq!(escaped.store_id(), "candidate");
    assert_eq!(layered.stats().missing, 0);
}

/// **Never silently fall back.** An operand the map compiled but the
/// overlay lacks is an error: serving source bytes would execute BF16
/// while every record claimed Q6_K, and the failure would read as
/// "quantisation is free".
#[test]
fn a_compiled_operand_missing_from_the_overlay_is_refused_not_downgraded() {
    let candidate = store(
        "candidate",
        &[("1.block_sparse_moe.experts.0.w2.weight", b"Q6")],
    );
    let source = store(
        "source",
        &[("1.block_sparse_moe.experts.9.w2.weight", b"BF16")],
    );
    let layered = LayeredOperands::new(overlay_map(), candidate, source);

    // The source HAS it — that is precisely the trap.
    let err = layered
        .resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.9.w2.weight"),
        )
        .expect_err("must refuse");
    assert!(format!("{err}").contains("refusing to fall back"), "{err}");
    assert_eq!(layered.stats().missing, 1);
    assert_eq!(layered.stats().source_fallback_hits, 0);
}

/// An operand in neither store is named as such, rather than producing
/// an empty region.
#[test]
fn an_operand_in_neither_store_is_reported_missing() {
    let layered =
        LayeredOperands::new(overlay_map(), store("candidate", &[]), store("source", &[]));
    let err = layered
        .resolve(
            Role::ExpertWeight,
            &operand("9.block_sparse_moe.experts.1.w2.weight"),
        )
        .expect_err("must refuse");
    assert!(format!("{err}").contains("neither"), "{err}");
    assert_eq!(layered.stats().missing, 1);
}

/// Regions borrow from a store they keep alive, so execution can hold a
/// reference without owning or copying the bytes.
#[test]
fn a_region_keeps_its_backing_alive() {
    let region = {
        let s = store("candidate", &[("t", b"payload")]);
        let layered = LayeredOperands::new(overlay_map(), s, store("source", &[]));
        layered
            .resolve(Role::Unknown, &operand("t"))
            .expect_err("Unknown is not a compiled role");
        // Resolve through a map that DOES compile it, then drop everything
        // but the region.
        let s2 = store(
            "candidate",
            &[("1.block_sparse_moe.experts.0.w2.weight", b"payload")],
        );
        let l2 = LayeredOperands::new(overlay_map(), s2, store("source", &[]));
        l2.resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.0.w2.weight"),
        )
        .expect("resolves")
    };
    assert_eq!(region.region.bytes(), b"payload");
    assert_eq!(region.region.len(), 7);
}

/// The compiled bank's layout: every expert at its own index, so a route
/// to any of them resolves without consulting a residency universe.
#[test]
fn an_identity_layout_holds_every_expert_at_its_own_index() {
    let l = ExpertLayout::Identity { experts: 256 };
    assert_eq!(l.slot_of(0), Some(0));
    assert_eq!(l.slot_of(181), Some(181));
    assert_eq!(l.slot_of(255), Some(255));
    assert_eq!(l.slot_of(256), None, "outside the bank is still refused");
    assert_eq!(l.slots(), 256);
}

/// The packed union: a genuine subset, where a missing expert is a real
/// answer rather than an error.
#[test]
fn a_mapped_layout_holds_only_its_own_subset() {
    let l = ExpertLayout::Mapped {
        ids: vec![73, 12, 181],
    };
    assert_eq!(l.slot_of(73), Some(0));
    assert_eq!(l.slot_of(12), Some(1));
    assert_eq!(
        l.slot_of(181),
        Some(2),
        "identity and slot are different facts"
    );
    assert_eq!(l.slot_of(99), None);
    assert_eq!(l.slots(), 3);
}

/// **The control that makes the hybrid convincing.** For a
/// candidate-selected expert, semantic identity, physical slot and
/// backing-object identity must all agree — and a neighbouring layer
/// left at source must come from the source store.
#[test]
fn semantic_id_physical_slot_and_backing_store_all_agree() {
    let candidate = store(
        "candidate",
        &[
            ("1.block_sparse_moe.experts.181.w1.weight", b"g"),
            ("1.block_sparse_moe.experts.181.w3.weight", b"u"),
            ("1.block_sparse_moe.experts.181.w2.weight", b"d"),
        ],
    );
    let source = store(
        "source",
        &[
            ("4.block_sparse_moe.experts.181.w1.weight", b"G"),
            ("4.block_sparse_moe.experts.181.w3.weight", b"U"),
            ("4.block_sparse_moe.experts.181.w2.weight", b"D"),
        ],
    );
    let layered = LayeredOperands::new(overlay_map(), candidate, source);
    let region = |t: &str| layered.resolve(Role::ExpertWeight, &operand(t)).expect(t);

    // Layer 1: compiled, full bank, identity layout.
    let scoped = ExpertBankBinding {
        gate: region("1.block_sparse_moe.experts.181.w1.weight"),
        up: region("1.block_sparse_moe.experts.181.w3.weight"),
        down: region("1.block_sparse_moe.experts.181.w2.weight"),
        layout: ExpertLayout::Identity { experts: 256 },
        extent: ExtentPolicy::Exact,
        shared_branch: false,
    };
    assert_eq!(scoped.layout.slot_of(181), Some(181), "identity layout");
    assert_eq!(scoped.store_id(), "candidate", "backing object");
    assert_eq!(
        scoped.gate.encoding,
        ExpertEncoding::Q6K,
        "the map decided this"
    );
    assert_eq!(scoped.gate.region.bytes(), b"g");

    // Layer 4: untouched, source precision.
    let untouched = ExpertBankBinding {
        gate: region("4.block_sparse_moe.experts.181.w1.weight"),
        up: region("4.block_sparse_moe.experts.181.w3.weight"),
        down: region("4.block_sparse_moe.experts.181.w2.weight"),
        layout: ExpertLayout::Mapped { ids: vec![181] },
        extent: ExtentPolicy::Exact,
        shared_branch: false,
    };
    assert_eq!(untouched.store_id(), "source");
    assert_eq!(untouched.gate.encoding, ExpertEncoding::Bf16);
    assert_eq!(untouched.layout.slot_of(181), Some(0), "packed slot != id");

    assert_eq!(layered.stats().candidate_hits, 3);
    assert_eq!(layered.stats().source_fallback_hits, 3);
    assert_eq!(layered.stats().missing, 0);
}

/// Offsets come from the layout, so the same binding code addresses a
/// packed fixture and a compiled full bank.
#[test]
fn offsets_follow_the_layout_not_the_expert_id() {
    let s = store("s", &[("t", &[0u8; 16])]);
    let region = || {
        let layered = LayeredOperands::new(
            PrecisionMap {
                roles: vec![],
                ..overlay_map()
            },
            s.clone(),
            s.clone(),
        );
        layered
            .resolve(Role::ExpertWeight, &operand("t"))
            .expect("t")
    };
    let identity = ExpertBankBinding {
        gate: region(),
        up: region(),
        down: region(),
        layout: ExpertLayout::Identity { experts: 256 },
        extent: ExtentPolicy::Exact,
        shared_branch: false,
    };
    assert_eq!(identity.gate_up_offset(181, 1000), Some(181_000));

    let packed = ExpertBankBinding {
        gate: region(),
        up: region(),
        down: region(),
        layout: ExpertLayout::Mapped { ids: vec![73, 181] },
        extent: ExtentPolicy::Exact,
        shared_branch: false,
    };
    assert_eq!(packed.gate_up_offset(181, 1000), Some(1000), "slot 1");
    assert_eq!(packed.gate_up_offset(99, 1000), None, "not in this bank");
}

/// **The execution analogue of the no-silent-fallback rule.**
///
/// Bytes that are BF16 dispatched as Q6_K would decode to plausible
/// garbage, and the failure would read as "quantisation is
/// catastrophic" rather than "the wrong kernel ran". A bank must refuse
/// to claim an encoding its bytes are not.
#[test]
fn a_bank_whose_bytes_are_not_its_declared_encoding_is_refused() {
    const HIDDEN: usize = 512;
    const INTER: usize = 256;
    const EXPERTS: u32 = 4;
    let bf16_bank = ExpertEncoding::Bf16
        .matrix_bytes(INTER, HIDDEN)
        .expect("bf16") as usize
        * EXPERTS as usize;
    let q6_bank =
        ExpertEncoding::Q6K.matrix_bytes(INTER, HIDDEN).expect("q6") as usize * EXPERTS as usize;
    assert!(
        bf16_bank > q6_bank,
        "the two sizes must differ to test anything"
    );

    let bank = |bytes: usize, encoding: ExpertEncoding, extent: ExtentPolicy| {
        let s = store("s", &[("bank", &vec![0u8; bytes])]);
        let r = || EncodedRegion {
            region: PhysicalStore::whole(&s, "bank").expect("whole"),
            encoding,
        };
        ExpertBankBinding {
            gate: r(),
            up: r(),
            down: r(),
            layout: ExpertLayout::Identity { experts: EXPERTS },
            extent,
            shared_branch: false,
        }
    };

    // Honest banks pass.
    bank(bf16_bank, ExpertEncoding::Bf16, ExtentPolicy::Exact)
        .validate(HIDDEN, INTER)
        .expect("bf16 bytes as BF16");
    bank(q6_bank, ExpertEncoding::Q6K, ExtentPolicy::Exact)
        .validate(HIDDEN, INTER)
        .expect("q6 bytes as Q6_K");

    // Q6_K bytes claimed as BF16: too small, caught by room alone.
    let err = bank(q6_bank, ExpertEncoding::Bf16, ExtentPolicy::ContainingView)
        .validate(HIDDEN, INTER)
        .expect_err("must refuse");
    assert!(
        format!("{err}").contains("not what this encoding claims"),
        "{err}"
    );

    // BF16 bytes claimed as Q6_K: LARGER than the claim, so room alone
    // would wave it through — the exact-extent check is what catches it.
    assert!(
        bank(bf16_bank, ExpertEncoding::Q6K, ExtentPolicy::ContainingView)
            .validate(HIDDEN, INTER)
            .is_ok(),
        "room alone cannot catch an oversized bank — that is why exact_bank exists"
    );
    let err = bank(bf16_bank, ExpertEncoding::Q6K, ExtentPolicy::Exact)
        .validate(HIDDEN, INTER)
        .expect_err("must refuse");
    assert!(format!("{err}").contains("not this encoding"), "{err}");
}

/// A precision map naming an encoding no grouped kernel reads is
/// refused at resolution, not at dispatch.
#[test]
fn a_map_naming_an_unexecutable_encoding_is_refused() {
    let candidate = store(
        "candidate",
        &[("1.block_sparse_moe.experts.7.w2.weight", b"x")],
    );
    // The EXCEPTION carries the encoding here, so overriding only the
    // map default would leave the exception's Q6_K winning and the test
    // proving nothing.
    let layered = LayeredOperands::new(
        PrecisionMap {
            encoding: "Q3_K".into(),
            exceptions: vec![Exception {
                projection: None,
                layers: Some((1, 1)),
                encoding: Some("Q3_K".into()),
            }],
            ..overlay_map()
        },
        candidate,
        store("source", &[]),
    );
    let err = layered
        .resolve(
            Role::ExpertWeight,
            &operand("1.block_sparse_moe.experts.7.w2.weight"),
        )
        .expect_err("must refuse");
    assert!(
        format!("{err}").contains("no grouped kernel reads"),
        "{err}"
    );
}
