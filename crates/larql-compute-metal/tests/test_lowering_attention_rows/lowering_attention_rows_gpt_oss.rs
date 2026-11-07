//! gpt-oss's attention through the VERIFY-N block: all four projection
//! biases, sinks and YaRN rope together (no QK norm, post norm or
//! softcap), at its own geometry (64 q / 8 kv heads of 64) and its
//! sliding-window width. The basics cover each feature on some geometry;
//! nothing covered them together, or the window's effect on the arm a
//! block takes at gpt-oss's 512 slice tier.

use super::*;

const SEQPAR_ONE_OP: Witness = Witness {
    serial: 0,
    seqpar: ONE_OP,
    splitk: 0,
};
const SEQPAR_PER_ROW: Witness = Witness {
    serial: 0,
    seqpar: PER_ROW,
    splitk: 0,
};

#[test]
fn gpt_oss_attention_block_is_one_seqpar_dispatch_at_parity() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Unwindowed and windowed (narrower than the cache, so every row's
    // span is clamped) — both below the 512 tier, both one dispatch.
    for window in [None, Some(WINDOW)] {
        check(
            &gpu,
            Case {
                geom: GPT_OSS,
                base: SHORT_BASE,
                window,
                variant: Variant::GptOss,
                splitk_rows: None,
            },
            SEQPAR_ONE_OP,
            SEQPAR_PER_ROW,
        );
    }
}

#[test]
fn gpt_oss_sliding_layer_at_the_512_tier_is_one_dispatch_where_full_loops() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // The tier is chosen from the WINDOWED span: a sliding layer's rows
    // at 510..=514 all clamp to 128 and share one slice count, so the
    // block stays one dispatch. The full layer at the same positions
    // straddles 8 -> 12 slices and must loop per row.
    check(
        &gpu,
        Case {
            geom: GPT_OSS,
            base: TIER_512_BASE,
            window: Some(GPT_OSS_WINDOW),
            variant: Variant::GptOss,
            splitk_rows: None,
        },
        SEQPAR_ONE_OP,
        SEQPAR_PER_ROW,
    );
    check(
        &gpu,
        Case {
            geom: GPT_OSS,
            base: TIER_512_BASE,
            window: None,
            variant: Variant::GptOss,
            splitk_rows: None,
        },
        SEQPAR_PER_ROW,
        SEQPAR_PER_ROW,
    );
}
