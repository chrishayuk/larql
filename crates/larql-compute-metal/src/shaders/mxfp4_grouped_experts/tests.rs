use super::*;

#[test]
fn the_interleaved_superblock_is_four_and_a_sixteenth_bits_per_weight() {
    let bpw = (SB_BYTES * 8) as f64 / (GROUPS_PER_SB * GROUP_ELEMS) as f64;
    assert!((bpw - 4.0625).abs() < 1e-12, "got {bpw}");
    // ...against the split layout's 4.25.
    let split = ((GROUP_BYTES + 1) * 8) as f64 / GROUP_ELEMS as f64;
    assert!((split - 4.25).abs() < 1e-12, "got {split}");
}

#[test]
fn the_pair_table_decodes_each_byte_to_its_two_nibbles() {
    // Guards the generated table against LUT drift, without eyeballing 256
    // literals: spot-check the corners and the sign boundary.
    for b in [0usize, 0x37, 0x8F, 0xFF] {
        let want = (LUT[b & 0x0F], LUT[b >> 4]);
        let text = format!("float2({:.1}f, {:.1}f)", want.0, want.1);
        assert!(pair_table().contains(&text), "byte {b:#04x} -> {text}");
    }
}

#[test]
fn the_pair_table_has_exactly_two_hundred_and_fifty_six_entries() {
    assert_eq!(pair_table().matches("float2(").count(), 256);
}

#[test]
fn every_arm_appears_in_the_emitted_source_exactly_once() {
    let src = shader();
    for name in [
        "mxfp4g_split_lut16",
        "mxfp4g_split_lut16_vec",
        "mxfp4g_inter_lut16",
        "mxfp4g_inter_pair",
        "mxfp4g_inter_magsign",
        "mxfp4g_inter_bits",
        "mxfp4g_inter_affine",
        "mxfp4g_inter_nox",
    ] {
        // The `(` closes the name: `mxfp4g_split_lut16` must not also
        // count its `_vec` sibling.
        assert_eq!(
            src.matches(&format!("kernel void {name}(")).count(),
            1,
            "{name}"
        );
    }
}

#[test]
fn only_the_split_arm_binds_scale_offsets_and_a_row_walk() {
    let src = shader();
    // The binding table in `ExpertScaleBinding`'s docs is what call sites
    // encode against, so pin it at the source rather than trusting prose.
    // Whitespace is column alignment, not contract — collapse it first.
    let flat = KERNEL_A.split_whitespace().collect::<Vec<_>>().join(" ");
    for (name, slot) in [
        ("Wp", 0),
        ("offsets", 1),
        ("Ws", 2),
        ("s_offsets", 3),
        ("X", 4),
        ("out", 5),
        ("N", 6),
        ("K", 7),
        ("XSTRIDE", 8),
        ("ROWBASE", 9),
        ("ROWSTRIDE", 10),
    ] {
        assert!(
            flat.contains(&format!("{name} [[buffer({slot})]]")),
            "arm A must bind {name} at buffer({slot})"
        );
    }
    // Arm A2 binds the identical table — same slots, same names — so the
    // two split arms are interchangeable at every call site.
    let flat_vec = KERNEL_A2.split_whitespace().collect::<Vec<_>>().join(" ");
    for (name, slot) in [("s_offsets", 3), ("ROWBASE", 9), ("ROWSTRIDE", 10)] {
        assert!(
            flat_vec.contains(&format!("{name} [[buffer({slot})]]")),
            "arm A2 must bind {name} at buffer({slot})"
        );
    }
    // The interleaved arms deliberately do NOT carry either: they keep the
    // shared inline-scale arity, which is also why they can only serve a
    // contiguous-halves bank. A call site holding an interleaved bank must
    // refuse rather than pick one of them.
    // A2x2 binds the same table too — it substitutes for A2 wherever
    // the alignment holds.
    let flat_x2 = KERNEL_A2X2.split_whitespace().collect::<Vec<_>>().join(" ");
    for (name, slot) in [("s_offsets", 3), ("ROWBASE", 9), ("ROWSTRIDE", 10)] {
        assert!(
            flat_x2.contains(&format!("{name} [[buffer({slot})]]")),
            "arm A2x2 must bind {name} at buffer({slot})"
        );
    }
    // The VERIFY-N block arms (A2x2m, A2x2s) are x2 over a block's slots:
    // same table at 0–10, their slot grouping after it.
    for (arm, kernel) in [("A2x2m", KERNEL_A2X2M), ("A2x2s", KERNEL_A2X2S)] {
        let flat = kernel.split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, slot) in [("s_offsets", 3), ("ROWBASE", 9), ("ROWSTRIDE", 10)] {
            assert!(
                flat.contains(&format!("{name} [[buffer({slot})]]")),
                "arm {arm} must bind {name} at buffer({slot})"
            );
        }
    }
    let interleaved_src: String = src
        .replace(KERNEL_A, "")
        .replace(KERNEL_A2, "")
        .replace(KERNEL_A2X2M, "")
        .replace(KERNEL_A2X2S, "")
        .replace(KERNEL_A2X2, "")
        .replace(KERNEL_A2X2GU, "")
        .replace(KERNEL_A2DC, "")
        .replace(KERNEL_A2X2P, "")
        .replace(KERNEL_A2X4, "");
    assert!(!interleaved_src.contains("s_offsets"));
    assert!(!interleaved_src.contains("ROWSTRIDE"));
}

/// Every interleaved kernel — the three candidates plus the bit-math arm and
/// the two ceiling probes.
const INTERLEAVED_ARMS: usize = 6;

#[test]
fn all_interleaved_arms_share_one_addressing_body() {
    // C-B, D-B, G-D and the probe subtractions are only pure decode effects
    // if the addressing, tiling and scale decode are byte-identical.
    let src = shader();
    assert_eq!(
        src.matches("const uint sb   = g / MXG_GROUPS_PER_SB;")
            .count(),
        INTERLEAVED_ARMS
    );
    // Each decode strategy appears in exactly one arm.
    assert_eq!(src.matches("MXG_PAIR[blk[b]]").count(), 1);
    assert_eq!(src.matches("MXG_MAG[lo & 7u]").count(), 1);
    assert_eq!(
        src.matches("as_type<float>(((c & 8u) << 28u) | mag)")
            .count(),
        1
    );
}

#[test]
fn the_ceiling_probes_are_the_only_arms_that_change_the_arithmetic() {
    // E drops the fp4 grid, F drops X entirely. Nothing else may.
    let src = shader();
    assert_eq!(src.matches("- 8.0f)").count(), 2, "affine probe only");
    let no_x = src.matches("part += float(byte & 0x0Fu) + float((byte >> 4u) & 0x0Fu);");
    assert_eq!(no_x.count(), 1, "exactly one arm skips the X gather");
}

#[test]
fn the_lut_matches_the_fp4_value_set() {
    assert_eq!(LUT[7], 6.0);
    assert_eq!(LUT[15], -6.0);
    assert_eq!(LUT[8], -0.0);
}
