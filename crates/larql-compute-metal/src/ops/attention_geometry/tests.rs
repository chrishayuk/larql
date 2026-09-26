use super::*;

const GPT_OSS: AttentionGeometryQuery = AttentionGeometryQuery {
    head_dim: 64,
    num_q_heads: 64,
    num_kv_heads: 8,
    span: 0,
};
const GLIMMER: AttentionGeometryQuery = AttentionGeometryQuery {
    head_dim: 128,
    num_q_heads: 32,
    num_kv_heads: 2,
    span: 0,
};

fn at(mut q: AttentionGeometryQuery, span: u32) -> AttentionGeometryQuery {
    q.span = span;
    q
}

/// The gpt-oss row IS the KV-B1 policy: for every span the planner under
/// an unset request agrees with `slices_for(Unset, 64, span)`, so moving
/// the decode path onto the planner changes nothing the A/B/C ladder
/// licensed.
#[test]
fn gpt_oss_row_is_the_kv_b1_policy() {
    for span in [
        1u32, 36, 128, 255, 256, 511, 512, 513, 767, 768, 1023, 1024, 2048, 4096,
    ] {
        let planner = choose_attention_geometry(SeqparRequest::Unset, &at(GPT_OSS, span)).slices();
        let policy = slices_for(SeqparRequest::Unset, 64, span);
        assert_eq!(planner, policy, "span {span}");
    }
}

/// The Glimmer row: serial below 1024 (the 512 block was direction-only,
/// so unlicensed), 8 slices — KV-B1's ceiling at head_dim 128 — from
/// 1024 up, where the 1K/2K/4K blocks all had 8 ahead.
#[test]
fn glimmer_row_is_serial_short_and_eight_slices_from_1k() {
    for span in [1u32, 128, 512, 1023] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Unset, &at(GLIMMER, span)),
            AttentionGeometry::Serial,
            "span {span}"
        );
    }
    for span in [1024u32, 2048, 4000, 4096] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Unset, &at(GLIMMER, span)),
            AttentionGeometry::SeqPar { slices: 8 },
            "span {span}"
        );
    }
}

/// An unmeasured geometry runs serial under an unset request — an
/// unmeasured policy is not a default.
#[test]
fn unmeasured_geometry_is_serial_when_unset() {
    let odd = AttentionGeometryQuery {
        head_dim: 64,
        num_q_heads: 16,
        num_kv_heads: 16,
        span: 2048,
    };
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Unset, &odd),
        AttentionGeometry::Serial,
        "same head_dim as a measured row is not the same geometry"
    );
}

#[test]
fn off_is_serial_everywhere() {
    for q in [at(GPT_OSS, 4096), at(GLIMMER, 4096)] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Off, &q),
            AttentionGeometry::Serial
        );
    }
}

#[test]
fn explicit_slices_are_honoured_and_bounded() {
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(4), &at(GLIMMER, 512)),
        AttentionGeometry::SeqPar { slices: 4 }
    );
    // 16 x 128 = 2048 threads exceeds the kernel's tg_partial bound: clamped
    // to 8, not refused and not overrun.
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(16), &at(GLIMMER, 512)),
        AttentionGeometry::SeqPar { slices: 8 }
    );
    // One slice partitions nothing.
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(1), &at(GLIMMER, 512)),
        AttentionGeometry::Serial
    );
    // A head_dim past the bound cannot host two slices at all.
    let wide = AttentionGeometryQuery {
        head_dim: 1024,
        num_q_heads: 8,
        num_kv_heads: 8,
        span: 4096,
    };
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(8), &wide),
        AttentionGeometry::Serial
    );
}

/// `auto` is the occupancy heuristic on any geometry: at head_dim 128 the
/// 512/768/1024-thread tiers mean 4/6/8 slices.
#[test]
fn auto_is_the_occupancy_heuristic_on_any_geometry() {
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 256)),
        AttentionGeometry::SeqPar { slices: 4 }
    );
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 768)),
        AttentionGeometry::SeqPar { slices: 6 }
    );
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 2048)),
        AttentionGeometry::SeqPar { slices: 8 }
    );
}

#[test]
fn slices_accessor_matches_variant() {
    assert_eq!(AttentionGeometry::Serial.slices(), 0);
    assert_eq!(AttentionGeometry::SeqPar { slices: 6 }.slices(), 6);
}

const GEMMA3: AttentionGeometryQuery = AttentionGeometryQuery {
    head_dim: 256,
    num_q_heads: 8,
    num_kv_heads: 4,
    span: 0,
};

/// The Gemma 3 split-K row: none below 128 (the 32-span block drifted, so
/// unlicensed), 8 chunks x 2 slices from 128, 16 x 2 from 512.
#[test]
fn gemma3_splitk_row_tiers() {
    for span in [1u32, 32, 127] {
        assert_eq!(
            choose_splitk(SeqparRequest::Unset, &at(GEMMA3, span)),
            None,
            "span {span}"
        );
    }
    for (span, chunks) in [
        (128u32, 8usize),
        (511, 8),
        (512, 16),
        (1024, 16),
        (4096, 16),
    ] {
        assert_eq!(
            choose_splitk(SeqparRequest::Unset, &at(GEMMA3, span)),
            Some(SplitKGeometry { chunks, slices: 2 }),
            "span {span}"
        );
    }
}

/// Any explicit request keeps the intra-threadgroup kernels — the A/B arm
/// against split-K is `LARQL_KV_SEQPAR=<n>`.
#[test]
fn explicit_request_never_selects_splitk() {
    for req in [
        SeqparRequest::Off,
        SeqparRequest::Auto,
        SeqparRequest::Slices(4),
    ] {
        assert_eq!(choose_splitk(req, &at(GEMMA3, 2048)), None, "{req:?}");
    }
}

/// No split-K row → none, however long the span.
#[test]
fn unmeasured_geometry_has_no_splitk() {
    for g in [GPT_OSS, GLIMMER] {
        assert_eq!(choose_splitk(SeqparRequest::Unset, &at(g, 4096)), None);
    }
}

/// Every chosen chunk count respects the kernel floor and the scratch
/// ceiling at every span the long kernel can serve.
#[test]
fn splitk_chunks_respect_kernel_bounds() {
    use crate::ops::kv_splitk::{min_chunks, SPLITK_MAX_CHUNKS};
    for span in 1u32..=4096 {
        if let Some(g) = choose_splitk(SeqparRequest::Unset, &at(GEMMA3, span)) {
            assert!(
                g.chunks >= min_chunks(span) && g.chunks <= SPLITK_MAX_CHUNKS,
                "span {span}"
            );
            assert!(g.chunks <= span as usize, "span {span}");
        }
    }
}
