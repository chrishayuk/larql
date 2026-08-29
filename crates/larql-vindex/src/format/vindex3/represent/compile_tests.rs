//! `tests` for [`super`].

use super::*;

const HIDDEN: usize = 2304;
const INTER: usize = 1024;
const EXPERTS: u32 = 256;

fn layout(encoding: &str) -> LayerBankLayout {
    LayerBankLayout::new(1, encoding, EXPERTS, HIDDEN, INTER).expect("layout")
}

/// The compiled sizes are the formats' own block geometry, at Kimi's
/// real expert shape.
#[test]
fn bank_strides_follow_the_encodings_block_geometry() {
    let bf16 = layout("BF16");
    assert_eq!(bf16.gate_up_stride, (INTER * HIDDEN * 2) as u64);
    assert_eq!(bf16.down_stride, (HIDDEN * INTER * 2) as u64);

    let q6 = layout("Q6_K");
    assert_eq!(q6.gate_up_stride, (INTER * HIDDEN / 256) as u64 * 210);
    // The measured Q1 ratio, arrived at independently here.
    let ratio = bf16.gate_up_stride as f64 / q6.gate_up_stride as f64;
    assert!((ratio - 2.438).abs() < 1e-3, "{ratio}");

    let q4 = layout("Q4_K");
    assert_eq!(q4.gate_up_stride, (INTER * HIDDEN / 256) as u64 * 144);
}

/// **Experts sit at their own index, not at a position in a resident
/// subset.** That is what makes a route to any expert resolvable —
/// including one the baseline never selected.
#[test]
fn every_expert_has_a_slot_and_the_bank_is_gapless() {
    let l = layout("Q6_K");
    let mut expected = 0u64;
    for e in 0..EXPERTS {
        let s = l.slot("down_proj", e).expect("slot");
        assert_eq!(s.offset, expected, "expert {e} is not contiguous");
        assert_eq!(s.len, l.down_stride);
        expected += s.len;
    }
    assert_eq!(l.bank_bytes("down_proj").expect("bytes"), expected);
    // The last expert is addressable; one past it is refused by name.
    assert!(l.slot("down_proj", EXPERTS - 1).is_ok());
    let err = l.slot("down_proj", EXPERTS).expect_err("out of range");
    assert!(format!("{err}").contains("outside layer 1"), "{err}");
}

/// A shape that would let two rows share a superblock scale is refused
/// at LAYOUT time, before anything is encoded.
#[test]
fn a_non_superblock_shape_is_refused_before_encoding() {
    let err = LayerBankLayout::new(1, "Q6_K", 4, 100, 100).expect_err("must refuse");
    assert!(format!("{err}").contains("superblock"), "{err}");
    // BF16 has no such constraint.
    assert!(LayerBankLayout::new(1, "BF16", 4, 100, 100).is_ok());
}

fn seal(tensor: &str, source_hash: &str, encoding: &str, offset: u64, len: u64) -> OperandSeal {
    OperandSeal {
        object: "target.expert_bank".into(),
        tensor: tensor.into(),
        encoding: encoding.into(),
        source_hash: source_hash.into(),
        target_hash: hash_bytes(tensor.as_bytes()),
        target_offset: offset,
        target_len: len,
    }
}

/// **Resume.** A sealed operand compiled from the same source at the
/// same encoding is skipped; everything else is redone.
#[test]
fn resume_skips_only_what_is_sealed_against_the_same_source() {
    let mut led = CompilationLedger::new("q2-candidate");
    let src = hash_bytes(b"source bytes v1");
    led.seal(seal(
        "1.mlp.experts.7.down_proj.weight",
        &src,
        "Q6_K",
        0,
        210,
    ));

    assert_eq!(
        led.pending(
            "target.expert_bank",
            "1.mlp.experts.7.down_proj.weight",
            &src,
            "Q6_K"
        ),
        None,
        "an unchanged operand must be skipped on resume"
    );
    assert_eq!(
        led.pending(
            "target.expert_bank",
            "1.mlp.experts.8.down_proj.weight",
            &src,
            "Q6_K"
        ),
        Some(Pending::Absent)
    );
}

/// **The property a plain "already present" check would miss.** A seal
/// is a claim about SPECIFIC source bytes; if the source changed, the
/// compiled bytes no longer represent the model and must be redone.
#[test]
fn a_changed_source_invalidates_its_seal() {
    let mut led = CompilationLedger::new("q2-candidate");
    let v1 = hash_bytes(b"source bytes v1");
    led.seal(seal(
        "1.mlp.experts.7.down_proj.weight",
        &v1,
        "Q6_K",
        0,
        210,
    ));

    let v2 = hash_bytes(b"source bytes v2");
    assert_eq!(
        led.pending(
            "target.expert_bank",
            "1.mlp.experts.7.down_proj.weight",
            &v2,
            "Q6_K"
        ),
        Some(Pending::SourceChanged)
    );
    // And re-targeting the same operand at another encoding also redoes it,
    // so switching a scope from Q6_K to Q4_K cannot inherit stale bytes.
    assert_eq!(
        led.pending(
            "target.expert_bank",
            "1.mlp.experts.7.down_proj.weight",
            &v1,
            "Q4_K"
        ),
        Some(Pending::EncodingChanged)
    );
}

/// Two seals must never claim the same region: a resumed run writes at
/// layout-computed offsets, so an overlap would be two operands
/// overwriting each other invisibly.
#[test]
fn overlapping_seals_are_detectable() {
    let src = hash_bytes(b"s");
    let mut led = CompilationLedger::new("q2-candidate");
    led.seal(seal("a", &src, "Q6_K", 0, 210));
    led.seal(seal("b", &src, "Q6_K", 210, 210));
    assert!(led.overlaps().is_empty(), "a gapless bank must not clash");

    led.seal(seal("c", &src, "Q6_K", 100, 210));
    let clashes = led.overlaps();
    assert!(!clashes.is_empty(), "an overlap must be caught");
}

/// A ledger round-trips, because resume reads it from disk after the
/// process that wrote it is gone.
#[test]
fn a_ledger_round_trips_through_json() {
    let src = hash_bytes(b"s");
    let mut led = CompilationLedger::new("q2-candidate");
    for e in 0..4u32 {
        let l = layout("Q6_K");
        let slot = l.slot("down_proj", e).expect("slot");
        led.seal(seal(
            &format!("1.mlp.experts.{e}.down_proj.weight"),
            &src,
            "Q6_K",
            slot.offset,
            slot.len,
        ));
    }
    assert_eq!(led.compiled_bytes(), 4 * layout("Q6_K").down_stride);

    let json = serde_json::to_string(&led).expect("serialises");
    let back: CompilationLedger = serde_json::from_str(&json).expect("round trips");
    assert_eq!(back, led);
    assert_eq!(
        back.pending(
            "target.expert_bank",
            "1.mlp.experts.2.down_proj.weight",
            &src,
            "Q6_K"
        ),
        None,
        "a resumed run must trust a ledger it read from disk"
    );
}

/// The whole compiled expert population, sized from the real geometry —
/// the number that decides whether this artifact is worth keeping.
#[test]
fn report_the_compiled_expert_population_size() {
    const MOE_LAYERS: u64 = 26;
    for enc in ["BF16", "Q6_K", "Q4_K"] {
        let l = layout(enc);
        let per_layer =
            2 * l.bank_bytes("gate_proj").expect("gate") + l.bank_bytes("down_proj").expect("down");
        eprintln!(
            "[compile] {enc:<5} experts: {:.1} GB per layer x {MOE_LAYERS} = {:.1} GB",
            per_layer as f64 / 1e9,
            (per_layer * MOE_LAYERS) as f64 / 1e9
        );
    }
    let bf16 = layout("BF16").bank_bytes("gate_proj").expect("g") * 3 * MOE_LAYERS;
    let q6 = layout("Q6_K").bank_bytes("gate_proj").expect("g") * 3 * MOE_LAYERS;
    assert!(
        (94.0..96.0).contains(&(bf16 as f64 / 1e9)),
        "the BF16 population must match the container's own 94.2 GB expert segment"
    );
    assert!(
        (38.0..40.0).contains(&(q6 as f64 / 1e9)),
        "{}",
        q6 as f64 / 1e9
    );
}
