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

    let identity = |r| RoutedProjection {
        region: r,
        addressing: ProjectionAddressing::Identity {
            experts: 256,
            stride: 1,
        },
        extent: ExtentPolicy::Exact,
    };
    let packed = |r| RoutedProjection {
        region: r,
        addressing: ProjectionAddressing::Table(vec![0]),
        extent: ExtentPolicy::Exact,
    };

    // Layer 1: compiled, full bank, identity-addressed.
    let scoped = ExpertBankBinding {
        gate: identity(region("1.block_sparse_moe.experts.181.w1.weight")),
        up: identity(region("1.block_sparse_moe.experts.181.w3.weight")),
        down: identity(region("1.block_sparse_moe.experts.181.w2.weight")),
        shared: None,
    };
    assert_eq!(
        scoped.gate.addressing.offset_of(181),
        Some(181),
        "identity: the address IS the identity"
    );
    assert_eq!(scoped.store_id(), "candidate", "backing object");
    assert!(scoped.stores_agree());
    assert_eq!(
        scoped.gate.encoding(),
        ExpertEncoding::Q6K,
        "the map decided this"
    );
    assert_eq!(scoped.gate.region.region.bytes(), b"g");

    // Layer 4: untouched, source precision, table-addressed.
    let untouched = ExpertBankBinding {
        gate: packed(region("4.block_sparse_moe.experts.181.w1.weight")),
        up: packed(region("4.block_sparse_moe.experts.181.w3.weight")),
        down: packed(region("4.block_sparse_moe.experts.181.w2.weight")),
        shared: None,
    };
    assert_eq!(untouched.store_id(), "source");
    assert_eq!(untouched.gate.encoding(), ExpertEncoding::Bf16);
    assert_eq!(
        untouched.gate.addressing.offset_of(0),
        Some(0),
        "a packed slot is not the id"
    );

    // **The mixed binding the per-projection split exists for**: gate
    // compiled and identity-addressed over the candidate store, up and
    // down still table-addressed over the source. Inexpressible while
    // addressing was bank-wide.
    let mixed = ExpertBankBinding {
        gate: identity(region("1.block_sparse_moe.experts.181.w1.weight")),
        up: packed(region("4.block_sparse_moe.experts.181.w3.weight")),
        down: packed(region("4.block_sparse_moe.experts.181.w2.weight")),
        shared: None,
    };
    assert!(
        !mixed.stores_agree(),
        "a scoped candidate legitimately spans two stores"
    );
    assert_eq!(mixed.gate.store_id(), "candidate");
    assert_eq!(mixed.up.store_id(), "source");
    assert_eq!(mixed.gate.encoding(), ExpertEncoding::Q6K);
    assert_eq!(mixed.up.encoding(), ExpertEncoding::Bf16);
    assert!(matches!(
        mixed.gate.addressing,
        ProjectionAddressing::Identity { .. }
    ));
    assert!(matches!(
        mixed.up.addressing,
        ProjectionAddressing::Table(_)
    ));

    // Six resolves for the two full bindings, plus the three the mixed
    // one re-resolved: one more from the candidate (layer 1's gate) and
    // two more from the source (layer 4's up/down).
    assert_eq!(layered.stats().candidate_hits, 4);
    assert_eq!(layered.stats().source_fallback_hits, 5);
    assert_eq!(
        layered.stats().missing,
        0,
        "every operand the map named was found in the layer it named"
    );
}

/// Offsets come from each projection's OWN addressing, so the same
/// binding code serves a packed fixture and a compiled full bank — and
/// serves both inside one layer.
#[test]
fn offsets_follow_the_addressing_not_the_expert_id() {
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
    let proj = |addressing| RoutedProjection {
        region: region(),
        addressing,
        extent: ExtentPolicy::Exact,
    };
    let ident = || {
        proj(ProjectionAddressing::Identity {
            experts: 256,
            stride: 1000,
        })
    };
    // A packed bank: expert 73 at slot 0, expert 181 at slot 1, and
    // nothing else addressable. `NOT_ADDRESSABLE` is a real answer.
    let table = || {
        let mut t = vec![NOT_ADDRESSABLE; 256];
        t[73] = 0;
        t[181] = 1000;
        proj(ProjectionAddressing::Table(t))
    };

    let identity = ExpertBankBinding {
        gate: ident(),
        up: ident(),
        down: ident(),
        shared: None,
    };
    assert_eq!(identity.gate_offset(181), Some(181_000));
    assert_eq!(identity.down_offset(255), Some(255_000));
    assert_eq!(identity.gate_offset(256), None, "outside the bank");

    let packed = ExpertBankBinding {
        gate: table(),
        up: table(),
        down: table(),
        shared: None,
    };
    assert_eq!(packed.gate_offset(181), Some(1000), "slot 1");
    assert_eq!(packed.gate_offset(73), Some(0), "slot 0");
    assert_eq!(packed.gate_offset(99), None, "not in this bank");

    // **The two inside ONE layer.** Gate identity-addressed, down
    // table-addressed: the same expert resolves to different offsets in
    // the same forward pass, which is what a projection-scoped
    // candidate is.
    let mixed = ExpertBankBinding {
        gate: ident(),
        up: table(),
        down: table(),
        shared: None,
    };
    assert_eq!(mixed.gate_offset(181), Some(181_000));
    assert_eq!(mixed.down_offset(181), Some(1000));
    assert_eq!(
        mixed.gate.addressing.experts(),
        mixed.down.addressing.experts(),
        "both still address the same expert population"
    );
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
        // The stride is the encoding's OWN per-expert size, so the
        // bank's extent and its addressing describe the same object.
        let stride = encoding.matrix_bytes(INTER, HIDDEN).expect("stride") as u32;
        let p = || RoutedProjection {
            region: r(),
            addressing: ProjectionAddressing::Identity {
                experts: EXPERTS,
                stride,
            },
            extent,
        };
        ExpertBankBinding {
            gate: p(),
            up: p(),
            down: p(),
            shared: None,
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

/// A store reports where its payload begins and hands out its whole
/// backing for registration — the two facts a zero-copy backend needs
/// and a `WeightRegion` deliberately cannot express.
#[test]
fn a_store_exposes_its_payload_start_and_its_whole_backing() {
    let s = store("owned", &[("a", &[1, 2, 3, 4]), ("b", &[5, 6])]);
    assert_eq!(s.payload_start(), 0, "an owned store has no header");
    assert_eq!(s.backing_bytes(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(s.payload_len(), 6);
    assert!(s.holds("a") && !s.holds("nope"));
    let r = s.whole("b").expect("named tensor");
    assert_eq!(r.bytes(), &[5, 6]);
    assert_eq!(r.store_offset(), 4);
    assert_eq!(r.len(), 2);
    assert!(!r.is_empty());
    assert!(s.whole("absent").is_none());
    // A span past the end is refused rather than clamped.
    assert!(s.span(4, 99).is_none());
}

/// **Bindability is decided by the region's OFFSET, not its bytes.**
///
/// The offset a zero-copy backend would bind at; an odd one makes the
/// kernel read a misaligned pointer, which on Metal is garbage with a
/// successful command buffer. Every projection is checked, shared
/// branch included, and the refusal names which.
#[test]
fn a_misaligned_region_is_refused_and_names_its_projection() {
    // A store whose first tensor is one byte long, so the SECOND starts
    // at an odd offset — the shape a container with an odd payload
    // start has for every tensor in it.
    let bytes = vec![0u8; 4096];
    let mut table = BTreeMap::new();
    table.insert("aligned".to_string(), (0u64, 512u64));
    table.insert("odd".to_string(), (1u64, 512u64));
    let s = Arc::new(PhysicalStore::owned("src", bytes, table));
    let region = |name: &str| EncodedRegion {
        region: s.whole(name).expect("tensor"),
        encoding: ExpertEncoding::Bf16,
    };
    assert!(region("aligned")
        .region
        .is_bindable_at(WEIGHT_BINDING_ALIGN));
    assert!(!region("odd").region.is_bindable_at(WEIGHT_BINDING_ALIGN));

    // (hidden, inter) = (16, 16) => a 512-byte bf16 projection.
    let (hidden, inter) = (16usize, 16usize);
    let bind = |name: &str, extent| RoutedProjection {
        region: region(name),
        addressing: ProjectionAddressing::Identity {
            experts: 1,
            stride: 512,
        },
        extent,
    };
    let ok = ExpertBankBinding {
        gate: bind("aligned", ExtentPolicy::Exact),
        up: bind("aligned", ExtentPolicy::Exact),
        down: bind("aligned", ExtentPolicy::Exact),
        shared: None,
    };
    ok.validate(hidden, inter)
        .expect("an aligned bank validates");

    for (which, mut bad) in [
        ("routed gate", ok.clone()),
        ("routed up", ok.clone()),
        ("routed down", ok.clone()),
    ] {
        // The misaligned region is a byte shorter, so THAT projection
        // relaxes to a containing view — the point under test is its
        // OFFSET. Per-projection extent means relaxing it here no
        // longer disables the exact check on its two siblings, which is
        // precisely what a bank-wide flag used to do.
        match which {
            "routed gate" => bad.gate = bind("odd", ExtentPolicy::ContainingView),
            "routed up" => bad.up = bind("odd", ExtentPolicy::ContainingView),
            _ => bad.down = bind("odd", ExtentPolicy::ContainingView),
        }
        let err = bad
            .validate(hidden, inter)
            .expect_err("a misaligned region must be refused");
        let text = format!("{err}");
        assert!(
            text.contains(which),
            "the refusal must name {which}: {text}"
        );
        assert!(text.contains("zero-copy"), "{text}");
    }

    // The shared branch is checked too — it is a separate region and
    // could be misaligned while every routed one is fine.
    let mut shared_bad = ok.clone();
    shared_bad.shared = Some(SharedExpertBinding {
        gate: region("aligned"),
        up: region("odd"),
        down: region("aligned"),
    });
    let err = shared_bad
        .validate(hidden, inter)
        .expect_err("a misaligned shared region must be refused");
    assert!(format!("{err}").contains("shared up"), "{err}");

    // And an aligned shared branch passes.
    let mut shared_ok = ok.clone();
    shared_ok.shared = Some(SharedExpertBinding {
        gate: region("aligned"),
        up: region("aligned"),
        down: region("aligned"),
    });
    shared_ok
        .validate(hidden, inter)
        .expect("an aligned shared branch validates");
    assert_eq!(shared_ok.store_id(), "src");
    assert_eq!(shared_ok.gate_offset(0), Some(0));
    assert_eq!(shared_ok.down_offset(0), Some(0));
}

/// `Debug` names the store and the encoding, never the bytes — what a
/// reader needs from a failure over a multi-gigabyte region.
#[test]
fn debug_names_the_store_and_encoding() {
    let s = store("candidate", &[("t", &[0u8; 8])]);
    let r = EncodedRegion {
        region: s.whole("t").expect("t"),
        encoding: ExpertEncoding::Q4K,
    };
    let text = format!("{r:?}");
    assert!(
        text.contains("candidate") && text.contains("Q4_K"),
        "{text}"
    );
    assert_eq!(r.store_id(), "candidate");
    assert_eq!(ExpertEncoding::Q4K.name(), "Q4_K");
    assert_eq!(ExpertEncoding::Q6K.name(), "Q6_K");
    assert_eq!(ExpertEncoding::Q80.name(), "Q8_0");
    assert_eq!(ExpertEncoding::Bf16.name(), "BF16");
    // Q-format geometry refuses a k that is not whole superblocks.
    assert!(ExpertEncoding::Q6K.matrix_bytes(4, 255).is_err());
    assert_eq!(ExpertEncoding::Q6K.matrix_bytes(4, 256).unwrap(), 4 * 210);
    assert_eq!(ExpertEncoding::Q4K.matrix_bytes(4, 256).unwrap(), 4 * 144);
    // Q8_0's block is 32 elements, so a k Q6_K refuses can still encode —
    // and a k that is not whole 32-blocks refuses too.
    assert!(ExpertEncoding::Q80.matrix_bytes(4, 33).is_err());
    assert_eq!(ExpertEncoding::Q80.matrix_bytes(4, 96).unwrap(), 4 * 3 * 34);
    assert_eq!(ExpertEncoding::Bf16.matrix_bytes(4, 8).unwrap(), 64);
    // The name round-trips through parse for every encoding a grouped
    // kernel reads.
    for enc in [
        ExpertEncoding::Bf16,
        ExpertEncoding::Q80,
        ExpertEncoding::Q6K,
        ExpertEncoding::Q4K,
    ] {
        assert_eq!(ExpertEncoding::parse(enc.name()), Some(enc));
    }
}

/// **Each projection answers about ITS OWN addressing**, including the
/// two questions only one addressing mode can answer.
///
/// `identity_stride` is `None` for a table on purpose: a packed bank's
/// entries need not be evenly spaced, and inventing a stride from two
/// of them would state a layout nobody declared. `max_offset` is the
/// extent question — the highest byte a projection can be asked to
/// read — and for a table that is derived from the OFFSETS, not from
/// the entry count, because a table has an entry per scored expert
/// while its bank holds only the addressable subset.
#[test]
fn each_addressing_answers_stride_and_extent_for_itself() {
    let identity = ProjectionAddressing::Identity {
        experts: 4,
        stride: 100,
    };
    assert_eq!(identity.identity_stride(), Some(100));
    assert_eq!(identity.max_offset(), Some(300), "the LAST expert's offset");
    assert_eq!(identity.experts(), 4);

    // 256 entries, 2 addressable — the fixture's shape.
    let mut t = vec![NOT_ADDRESSABLE; 256];
    t[73] = 0;
    t[181] = 500;
    let table = ProjectionAddressing::Table(t);
    assert_eq!(
        table.identity_stride(),
        None,
        "a packed bank declares no stride"
    );
    assert_eq!(
        table.max_offset(),
        Some(500),
        "the extent follows the OFFSETS, not the 256 entries"
    );
    assert_eq!(table.experts(), 256, "but it can be ASKED about 256");

    // Degenerate cases answer rather than panicking.
    assert_eq!(
        ProjectionAddressing::Table(vec![NOT_ADDRESSABLE; 8]).max_offset(),
        None,
        "a table addressing nothing has no extent"
    );
    assert_eq!(
        ProjectionAddressing::Identity {
            experts: 0,
            stride: 8
        }
        .max_offset(),
        None
    );

    // And a binding answers per projection, so one layer can mix them.
    let s = store("s", &[("t", &[0u8; 16])]);
    let proj = |addressing| RoutedProjection {
        region: EncodedRegion {
            region: PhysicalStore::whole(&s, "t").expect("t"),
            encoding: ExpertEncoding::Bf16,
        },
        addressing,
        extent: ExtentPolicy::ContainingView,
    };
    let mixed = ExpertBankBinding {
        gate: proj(ProjectionAddressing::Identity {
            experts: 4,
            stride: 100,
        }),
        up: proj(ProjectionAddressing::Table(vec![0, 8])),
        down: proj(ProjectionAddressing::Table(vec![8, 0])),
        shared: None,
    };
    assert_eq!(mixed.gate_offset(1), Some(100));
    assert_eq!(mixed.up_offset(1), Some(8));
    assert_eq!(
        mixed.down_offset(1),
        Some(0),
        "down's own table, not a sibling's"
    );
    assert!(
        mixed.stores_agree(),
        "same store here, different addressing"
    );
}
