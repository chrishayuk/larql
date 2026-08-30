//! The position-policy surface the QW-3.5/3.8 work added: the M-RoPE
//! axis table, the `MRope` accessor arms, and the linear-attention
//! topology facts. These are consumed heavily from larql-vindex, but a
//! consumer in another crate is not coverage here — the contract each
//! function states is asserted in the crate that owns it.

use larql_models::config::position::{mrope_axis_table, PositionPolicy, RotaryFrequencyBasis};
use larql_models::inventory::report::{LinearAttentionTopology, RecurrentStateDtype};

// ═══════════════════════════════════════════════════════════════
// mrope_axis_table — transcribed from HF `apply_interleaved_mrope`
// ═══════════════════════════════════════════════════════════════

/// Qwen3.8's real geometry: section [11, 11, 10] over 32 frequencies.
/// HF starts from all-T and overwrites `slice(1, section[1]*3, 3)` with
/// H and `slice(2, section[2]*3, 3)` with W — so the table must read
/// `T H W T H W …` with H stopping after 11 slots and W after 10.
#[test]
fn interleaved_axis_table_matches_hf_slice_semantics() {
    let axes = mrope_axis_table([11, 11, 10], true, 32);
    assert_eq!(axes.len(), 32);
    for (slot, axis) in axes.iter().enumerate() {
        let expected = match slot % 3 {
            1 if slot < 33 => 1, // H: slots 1,4,…,31 — all under 11·3
            2 if slot < 30 => 2, // W: slots 2,5,…,29 — under 10·3
            _ => 0,
        };
        assert_eq!(*axis, expected, "slot {slot}");
    }
    assert_eq!(axes.iter().filter(|a| **a == 1).count(), 11);
    assert_eq!(axes.iter().filter(|a| **a == 2).count(), 10);
}

/// Slots past `n_freqs` must not be written: a table narrower than the
/// section's own span truncates rather than panics.
#[test]
fn interleaved_axis_table_truncates_at_n_freqs() {
    let axes = mrope_axis_table([2, 2, 2], true, 4);
    // T at 0 and 3; H at 1 (next H slot 4 >= n_freqs); W at 2.
    assert_eq!(axes, vec![0, 1, 2, 0]);
}

/// The non-interleaved layout is contiguous blocks: `T… H… W…`.
#[test]
fn sectioned_axis_table_is_contiguous_blocks() {
    let axes = mrope_axis_table([2, 3, 1], false, 6);
    assert_eq!(axes, vec![0, 0, 1, 1, 1, 2]);
}

/// A section that over-declares past `n_freqs` stops at the table's
/// edge on whichever axis it runs out.
#[test]
fn sectioned_axis_table_truncates_mid_axis() {
    let axes = mrope_axis_table([2, 3, 3], false, 6);
    assert_eq!(axes, vec![0, 0, 1, 1, 1, 2]);
    // And a T block wider than the whole table leaves everything T.
    assert_eq!(mrope_axis_table([8, 1, 1], false, 4), vec![0, 0, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════
// PositionPolicy — the MRope accessor arms
// ═══════════════════════════════════════════════════════════════

fn qwen38_mrope(basis: RotaryFrequencyBasis) -> PositionPolicy {
    PositionPolicy::MRope {
        theta: 5_000_000.0,
        rotary_fraction: 0.25,
        basis,
        section: [11, 11, 10],
        interleaved: true,
    }
}

#[test]
fn mrope_accessors_answer_the_declared_facts() {
    let policy = qwen38_mrope(RotaryFrequencyBasis::RotaryWidth);
    assert_eq!(policy.mrope(), Some(([11, 11, 10], true)));
    assert_eq!(policy.rotary_fraction(), Some(0.25));
    // Qwen3.8 declares `rope_type: "default"` — the multi-axis facts
    // live in `mrope_section`, so the default basis answers no
    // `rope_type` spelling at all.
    assert_eq!(policy.declared_rope_type(), None);
}

#[test]
fn mrope_head_width_basis_spells_proportional() {
    let policy = qwen38_mrope(RotaryFrequencyBasis::HeadWidth);
    assert_eq!(policy.declared_rope_type(), Some("proportional"));
}

#[test]
fn non_mrope_policies_declare_no_axis_split() {
    assert_eq!(PositionPolicy::Rope { theta: 10_000.0 }.mrope(), None);
    assert_eq!(PositionPolicy::None.mrope(), None);
}

// ═══════════════════════════════════════════════════════════════
// RecurrentStateDtype — only the declared-and-executed value exists
// ═══════════════════════════════════════════════════════════════

#[test]
fn recurrent_state_dtype_reads_both_spellings_and_refuses_the_rest() {
    assert_eq!(
        RecurrentStateDtype::from_declared("float32"),
        Some(RecurrentStateDtype::Float32)
    );
    assert_eq!(
        RecurrentStateDtype::from_declared("f32"),
        Some(RecurrentStateDtype::Float32)
    );
    // An undeclared spelling is None — refused, never defaulted.
    assert_eq!(RecurrentStateDtype::from_declared("bfloat16"), None);
    assert_eq!(RecurrentStateDtype::Float32.declared_name(), "float32");
}

// ═══════════════════════════════════════════════════════════════
// LinearAttentionTopology — Qwen3.8's declared geometry closes
// ═══════════════════════════════════════════════════════════════

#[test]
fn qwen38_geometry_closes_on_the_observed_projection_widths() {
    let topo = LinearAttentionTopology {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
        state_dtype: Some(RecurrentStateDtype::Float32),
    };
    // 2·16·128 + 48·128 — the observed in_proj_qkv row count.
    assert_eq!(topo.qkv_channels(), 10_240);
    assert_eq!(topo.value_width(), 6_144);
}

#[test]
fn from_config_requires_every_dimension() {
    let full = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 48, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"
    }));
    let topo =
        LinearAttentionTopology::from_config(full.config()).expect("every dimension declared");
    assert_eq!(topo.qkv_channels(), 10_240);
    assert_eq!(topo.state_dtype, Some(RecurrentStateDtype::Float32));

    // A partially declared recurrence is refused, not completed.
    let partial = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "linear_num_key_heads": 16
    }));
    assert_eq!(LinearAttentionTopology::from_config(partial.config()), None);
}
