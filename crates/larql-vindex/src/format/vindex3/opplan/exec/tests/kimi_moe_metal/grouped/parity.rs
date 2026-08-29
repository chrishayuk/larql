//! Correctness: grouped must reproduce the per-expert dispatches, and expert identity must come from the offset table.
//!
//! See [`super`] for the hypothesis this rung tests and what the
//! grouped dispatch does and does not change.

use larql_compute_metal::trait_impl::grouped_experts::InputLayout;

use super::*;

/// **The contract.** Grouping changes where the work is scheduled, never
/// what it computes: bit-identical to the per-expert dispatches, at the
/// real geometry, for all three stages including the per-slot one.
///
/// Exact equality rather than a tolerance, deliberately — a tolerance
/// would let a numerics change hide inside an occupancy result.
#[test]
fn grouped_reproduces_the_per_expert_dispatches_bit_for_bit() {
    let Some((metal, fx)) = setup() else {
        return;
    };

    let gate_bank = Bank::build(&fx, Stage::Gate);
    let gate_grouped = metal
        .bf16_grouped_experts(
            &gate_bank.bytes,
            &gate_bank.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("grouped gate");
    assert_eq!(
        gate_grouped,
        per_expert_stage(&metal, &fx, Stage::Gate, &fx.x),
        "gate: grouping changed a value"
    );

    let up_bank = Bank::build(&fx, Stage::Up);
    let up_grouped = metal
        .bf16_grouped_experts(
            &up_bank.bytes,
            &up_bank.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("grouped up");
    assert_eq!(
        up_grouped,
        per_expert_stage(&metal, &fx, Stage::Up, &fx.x),
        "up: grouping changed a value"
    );

    let h = down_inputs(&fx, &gate_grouped, &up_grouped);
    let down_bank = Bank::build(&fx, Stage::Down);
    let down_grouped = metal
        .bf16_grouped_experts(
            &down_bank.bytes,
            &down_bank.offsets,
            &h,
            fx.hidden,
            fx.inter,
            InputLayout::PerSlot,
        )
        .expect("grouped down");
    assert_eq!(
        down_grouped,
        per_expert_stage(&metal, &fx, Stage::Down, &h),
        "down: grouping changed a value"
    );
}

/// The same gate rung 1 passes, run through the grouped path: every
/// selected expert plus the shared branch, scored against
/// `modeling_kimi.py`'s own per-expert output.
///
/// Slot `i` must hold expert `selected_ids_order[i]`'s answer — the
/// identity claim, at real geometry. The unit test
/// `the_offset_table_decides_which_expert_a_slot_computes` proves the
/// table rather than row position carries it; this proves the table this
/// caller builds is the right one.
#[test]
fn the_grouped_moe_ffn_matches_the_checkpoints_own_output() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let gate_bank = Bank::build(&fx, Stage::Gate);
    let up_bank = Bank::build(&fx, Stage::Up);
    let down_bank = Bank::build(&fx, Stage::Down);

    let gate = metal
        .bf16_grouped_experts(
            &gate_bank.bytes,
            &gate_bank.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("grouped gate");
    let up = metal
        .bf16_grouped_experts(
            &up_bank.bytes,
            &up_bank.offsets,
            &fx.x,
            fx.inter,
            fx.hidden,
            InputLayout::Shared,
        )
        .expect("grouped up");
    let h = down_inputs(&fx, &gate, &up);
    let out = metal
        .bf16_grouped_experts(
            &down_bank.bytes,
            &down_bank.offsets,
            &h,
            fx.hidden,
            fx.inter,
            InputLayout::PerSlot,
        )
        .expect("grouped down");

    for (slot, e) in fx.experts.iter().enumerate() {
        let got = &out[slot * fx.hidden..(slot + 1) * fx.hidden];
        let rel = rel_err(got, &e.oracle);
        eprintln!("[{:>9}] grouped-vs-hf {rel:.3e}", e.id);
        assert!(
            rel < REL_TOLERANCE,
            "slot {slot} ({}): grouped vs the checkpoint's own output, rel {rel:e}",
            e.id
        );
    }
}

/// **Control.** Rotating the offset table must rotate which expert each
/// slot answers for, at real geometry — so a slot's identity comes from
/// the table this caller built, not from where the bytes happen to sit.
///
/// Without it, a bank whose experts are laid out in selection order
/// would satisfy the parity test above even if the kernel ignored the
/// table entirely, and the first caller to feed a resident-only or
/// reordered bank would be silently mis-served.
#[test]
fn rotating_the_offset_table_rotates_which_expert_answers() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let bank = Bank::build(&fx, Stage::Gate);
    let (n, k) = (fx.inter, fx.hidden);

    let forward = metal
        .bf16_grouped_experts(&bank.bytes, &bank.offsets, &fx.x, n, k, InputLayout::Shared)
        .expect("forward");
    let mut rotated = bank.offsets.clone();
    rotated.rotate_left(1);
    let after = metal
        .bf16_grouped_experts(&bank.bytes, &rotated, &fx.x, n, k, InputLayout::Shared)
        .expect("rotated");

    let slots = bank.slots();
    for slot in 0..slots {
        let source = (slot + 1) % slots;
        assert_eq!(
            &after[slot * n..(slot + 1) * n],
            &forward[source * n..(source + 1) * n],
            "slot {slot} should now hold what slot {source} held"
        );
    }
    assert_ne!(forward, after, "control: the experts must differ at all");
}
