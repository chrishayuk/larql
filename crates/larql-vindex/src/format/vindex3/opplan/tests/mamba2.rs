//! The pure-SSM witness, in miniature (schema 6): a Mamba2 checkpoint
//! admits with ZERO fabricated surfaces, encodes with its operands
//! closing at encode time, reads back as 48-fold smaller honest layer
//! programs, and refuses execution by name.
//!
//! Mirrors the real `mamba2-780m-hf` witness at awkward miniature
//! dimensions — every width distinct so no broadcasting accident can
//! hide: hidden 12, d_inner 24, 4 heads × 6, state 5, conv 3, in_proj
//! rows 2·24 + 2·5 + 4 = 62, conv rows 24 + 2·5 = 34.

use std::path::Path;

use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{ContinuationProvider, RowKvState};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};
use crate::format::vindex3::plan::plan_system;
use larql_models::inventory::build_inventory;

const M_HIDDEN: usize = 12;
const M_D_INNER: usize = 24; // expand 2 × hidden
const M_HEADS: usize = 4;
const M_HEAD_DIM: usize = 6; // 4 × 6 = d_inner
const M_STATE: usize = 5;
const M_CONV: usize = 3;
const M_VOCAB: usize = 29;
const M_LAYERS: usize = 2;
const M_CONV_DIM: usize = M_D_INNER + 2 * M_STATE; // 34
const M_IN_PROJ_ROWS: usize = 2 * M_D_INNER + 2 * M_STATE + M_HEADS; // 62

/// Write the miniature pure-SSM checkpoint. `skip_suffix` drops every
/// tensor whose name ends with it — the lever the closure-at-encode test
/// pulls.
pub(crate) fn miniature_mamba2(dir: &Path, skip_suffix: Option<&str>) {
    // Rendered with the bare `Infinity` transformers actually writes, so
    // the fixture exercises the judged non-finite boundary end-to-end
    // rather than the pre-quoted string form.
    let config = serde_json::json!({
        "architectures": ["Mamba2ForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "mamba2",
        "hidden_size": M_HIDDEN,
        "num_hidden_layers": M_LAYERS,
        "vocab_size": M_VOCAB,
        "state_size": M_STATE,
        "num_heads": M_HEADS,
        "head_dim": M_HEAD_DIM,
        "expand": 2,
        "conv_kernel": M_CONV,
        "n_groups": 1,
        "chunk_size": 8,
        "time_step_limit": [0.0, "Infinity"],
        "time_step_floor": 1e-4,
        "time_step_min": 0.001,
        "time_step_max": 0.1,
        "time_step_rank": 8,
        "rescale_prenorm_residual": false,
        "rms_norm": true,
        "use_bias": false,
        "use_conv_bias": true,
        "hidden_act": "silu",
        "layer_norm_epsilon": 1e-5,
        "residual_in_fp32": true,
        "tie_word_embeddings": true
    })
    .to_string()
    .replace("\"Infinity\"", "Infinity");
    std::fs::write(dir.join("config.json"), config).unwrap();

    let mut shard = ShardBuilder::new();
    let mut push = |name: String, shape: &[usize], values: Vec<f32>| {
        if skip_suffix.is_some_and(|suffix| name.ends_with(suffix)) {
            return;
        }
        shard.push(&name, shape, &values);
    };
    push(
        "backbone.embeddings.weight".into(),
        &[M_VOCAB, M_HIDDEN],
        lcg_values(M_VOCAB * M_HIDDEN, 1),
    );
    push(
        "backbone.norm_f.weight".into(),
        &[M_HIDDEN],
        norm_values(M_HIDDEN, 2),
    );
    for layer in 0..M_LAYERS {
        let seed = 300 + layer as u64 * 20;
        let p = format!("backbone.layers.{layer}");
        push(
            format!("{p}.norm.weight"),
            &[M_HIDDEN],
            norm_values(M_HIDDEN, seed),
        );
        push(
            format!("{p}.mixer.in_proj.weight"),
            &[M_IN_PROJ_ROWS, M_HIDDEN],
            lcg_values(M_IN_PROJ_ROWS * M_HIDDEN, seed + 1),
        );
        push(
            format!("{p}.mixer.conv1d.weight"),
            &[M_CONV_DIM, 1, M_CONV],
            lcg_values(M_CONV_DIM * M_CONV, seed + 2),
        );
        push(
            format!("{p}.mixer.conv1d.bias"),
            &[M_CONV_DIM],
            lcg_values(M_CONV_DIM, seed + 3),
        );
        push(
            format!("{p}.mixer.A_log"),
            &[M_HEADS],
            lcg_values(M_HEADS, seed + 4),
        );
        push(
            format!("{p}.mixer.D"),
            &[M_HEADS],
            lcg_values(M_HEADS, seed + 5),
        );
        push(
            format!("{p}.mixer.dt_bias"),
            &[M_HEADS],
            lcg_values(M_HEADS, seed + 6),
        );
        push(
            format!("{p}.mixer.norm.weight"),
            &[M_D_INNER],
            norm_values(M_D_INNER, seed + 7),
        );
        push(
            format!("{p}.mixer.out_proj.weight"),
            &[M_HIDDEN, M_D_INNER],
            lcg_values(M_HIDDEN * M_D_INNER, seed + 8),
        );
    }
    shard.write(dir);
}

/// **A pure-SSM checkpoint admits with zero fabricated surfaces.** The
/// census names both mixer layers, the plan carries no attention and no
/// FFN group, the mixer surface is complete with the judged unbounded dt
/// clamp, and the init-only `time_step_*` keys grade declaration-only
/// rather than blocking or vanishing.
#[test]
fn a_pure_ssm_checkpoint_admits_with_no_fabricated_surfaces() {
    let dir = tempfile::tempdir().unwrap();
    miniature_mamba2(dir.path(), None);
    let inventory = build_inventory(dir.path()).unwrap();
    let plan = plan_system(&[("mamba2-mini".to_string(), inventory)]);
    let blocking: Vec<String> = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking: {blocking:?}");

    let census = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("census finding");
    assert!(
        census.detail.contains("2 Mamba2 recurrent"),
        "{}",
        census.detail
    );
    assert!(
        census.detail.contains("0 sliding / 0 full"),
        "{}",
        census.detail
    );

    let target = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    let surface = target.execution.as_ref().expect("surface built");
    assert!(surface.attention.is_none(), "no attention was fabricated");
    assert!(surface.ffn.is_none(), "no FFN was fabricated");
    let mixer = surface.mamba2.expect("the mixer surface is present");
    assert_eq!(mixer.geometry.state_size, M_STATE);
    assert_eq!(
        mixer.geometry.dt_limit_max,
        larql_models::config::DtBound::Unbounded,
        "the bare Infinity arrived as a declared unbounded side"
    );
    assert_eq!(
        surface.norm.placement,
        Some(crate::format::vindex3::graph::NormPlacement::PreMixer)
    );
    assert_eq!(surface.residual_in_fp32, Some(true));

    // Init-only keys: declaration-only, never blocking, never silently
    // consumed.
    let time_step_min = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject.ends_with("time_step_min"))
        .expect("declared key reported");
    assert!(!time_step_min.blocks());
}

/// **Encode closes at encode, the container reads back honestly, and
/// the generic path EXECUTES it:** every layer a `LayerAttention::Mamba2`
/// with 9/9 operands, no FFN op, no pre-FFN norm — then the mixer runs
/// through the ordinary prefill traversal with state prepared from the
/// plan's own declared geometry. Three properties an SSM cannot fake:
/// finite logits, bitwise determinism across a fresh state, and prefix
/// equivalence — `0..n` in one batch equals `0..k` then `k..n` with the
/// state (including the conv history crossing the batch boundary)
/// persisted.
#[test]
fn a_pure_ssm_container_encodes_closes_and_executes_generically() {
    let dir = tempfile::tempdir().unwrap();
    miniature_mamba2(dir.path(), None);
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("mamba2-mini.vindex3");
    encode_checkpoint(dir.path(), &container).expect("the witness encodes");

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("closure held");
    assert_eq!(plan.layers.len(), M_LAYERS);
    for layer in &plan.layers {
        let LayerAttention::Mamba2(op) = &layer.attention else {
            panic!(
                "layer {} is not a mixer: {:?}",
                layer.layer, layer.attention
            );
        };
        assert_eq!(op.state_elements(), M_HEADS * M_HEAD_DIM * M_STATE);
        assert!(op.conv1d_bias.is_some(), "use_conv_bias: true");
        assert!(op.gated_norm.is_some(), "rms_norm: true");
        assert!(layer.ffn.is_none(), "no FFN exists to plan");
        assert!(layer.pre_ffn_norm.is_none());
        assert_eq!(layer.operands_accounted, 9);
    }
    // Tied embeddings: the head reuses the embedding object.
    assert!(plan.output.is_some());

    let store = OperandStore::open(&container, &inspection).unwrap();
    // Preparation succeeds — the operator has an executor now.
    PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the mixer prepares");

    let geometry = plan_continuation_geometry(&plan).expect("state geometry is declared");
    let prefill = |tokens: &[u32], provider: &mut RowKvState| -> Vec<f32> {
        crate::format::vindex3::opplan::exec::prefill_plan(
            &plan,
            &store,
            tokens,
            &ReferenceBackend,
            provider,
        )
        .expect("the mixer executes")
        .logits
        .expect("the fixture carries an output head")
    };
    let fresh = || {
        let mut p = RowKvState::default();
        p.prepare_continuation(&geometry).unwrap();
        p
    };

    let mut provider = fresh();
    let logits = prefill(&[3, 17, 5], &mut provider);
    assert_eq!(logits.len(), M_VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "finite logits");

    // Bitwise determinism across a fresh state — state reset is real.
    let mut again = fresh();
    assert_eq!(prefill(&[3, 17, 5], &mut again), logits);

    // Prefix equivalence: the recurrence and its conv history survive the
    // batch boundary exactly. This executor is sequential in both shapes,
    // so the equality is bitwise, not approximate.
    let mut resumed = fresh();
    prefill(&[3, 17], &mut resumed);
    assert_eq!(prefill(&[5], &mut resumed), logits);

    // And the state genuinely moved: a mixer layer holds recurrent
    // buffers, never KV rows.
    let state = provider.recurrent_state(0).expect("a recurrent layer");
    assert!(state.buffer(0).cells().iter().any(|v| *v != 0.0));
    assert_eq!(provider.keys(0).len(), 0, "no KV row exists anywhere");

    // **Prefill-by-one equivalence, across the code-path seam.** The
    // batch above ran `execute_layer`; stepping the same tokens through
    // a `DecodeSession` runs the single-token decode path with its own
    // state carriage. For an SSM this is the continuation contract in
    // one assertion — recurrence, conv history and dt discretisation
    // must all survive being advanced one position at a time — and both
    // paths run the same sequential arithmetic, so the agreement is
    // bitwise, not approximate.
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    let mut stepped = None;
    for token in [3u32, 17, 5] {
        stepped = session.step(token).unwrap().logits;
    }
    assert_eq!(session.position(), 3);
    assert_eq!(
        stepped.expect("head present"),
        logits,
        "the decode step path diverged from batch prefill"
    );
}

/// **Closure at encode is a gate, not a report (drill F4):** a checkpoint
/// that passes the plan but ships an incomplete operand estate must not
/// leave a container behind. Dropping `mixer.D` starves closure of one
/// per-head operand; the encode refuses, names the missing role, and
/// removes its output.
#[test]
fn an_encode_whose_operands_do_not_close_is_removed_and_refused() {
    let dir = tempfile::tempdir().unwrap();
    miniature_mamba2(dir.path(), Some("mixer.D"));
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("mamba2-broken.vindex3");
    let err =
        encode_checkpoint(dir.path(), &container).expect_err("closure must refuse the encode");
    let text = err.to_string();
    assert!(
        text.contains("operands do not close") && text.contains("Mamba2D"),
        "{text}"
    );
    assert!(
        !container.exists(),
        "a refused encode must not leave a container behind"
    );
}

/// **The census fails closed on generic-plus-silence (drill F3):** an
/// unregistered family that declares neither per-layer topology nor an
/// attention shape blocks on its layer census instead of resolving every
/// layer to softmax/full by default.
#[test]
fn an_undeclared_family_with_no_attention_shape_fails_the_census_closed() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "model_type": "mystery_recurrence",
            "hidden_size": M_HIDDEN,
            "num_hidden_layers": M_LAYERS,
            "intermediate_size": 4 * M_HIDDEN,
            "vocab_size": M_VOCAB
        })
        .to_string(),
    )
    .unwrap();
    let mut shard = ShardBuilder::new();
    shard.push(
        "backbone.embeddings.weight",
        &[M_VOCAB, M_HIDDEN],
        &lcg_values(M_VOCAB * M_HIDDEN, 7),
    );
    shard.write(dir.path());
    let inventory = build_inventory(dir.path()).unwrap();
    let plan = plan_system(&[("mystery".to_string(), inventory)]);
    let census = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "layer_census")
        .expect("the fail-closed census finding");
    assert!(census.blocks());
    assert!(census.detail.contains("fails closed"), "{}", census.detail);
    assert!(!plan.admissible);
}

/// **F16 widens to f32 exactly.** The real witness container is F16
/// throughout (the prior corpus was BF16), and every IEEE half value is
/// exactly representable in f32 — checked at the values whose bit
/// patterns differ between the two half formats.
#[test]
fn f16_operands_widen_to_f32_exactly() {
    use crate::format::vindex3::opplan::exec::operands::widen;
    let values = [0.0f32, 1.0, -1.5, 0.099975586, 65504.0, -6.1035156e-5];
    let bytes: Vec<u8> = larql_models::quant::half::encode_f16(&values);
    let widened = widen("F16", &bytes, "probe").unwrap();
    assert_eq!(widened, values);
    // BF16 bytes read as F16 would be wrong values, not an error — which
    // is why the dtype label, not a guess, selects the decoder; and an
    // unjudged label still refuses.
    assert!(widen("F8_E4M3", &bytes, "probe").is_err());
}
