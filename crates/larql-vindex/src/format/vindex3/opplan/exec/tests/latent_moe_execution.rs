//! **The latent routed branch as the ROUTED PATH runs it**
//! (K3-LATENTMOE-1) — a whole `KimiSparseMoeBlock` forward, against the
//! reference's own `output`.
//!
//! `latent_moe_parity` drives `enter_latent` and `exit_latent` directly.
//! That establishes the two functions and says nothing about whether the
//! routed path COMPOSES them correctly: in what order, how many times,
//! and on which vector. That composition is where a wrapper is applied
//! to the wrong thing with every shape still closing, and until this
//! module existed nothing executed those arms at all.
//!
//! # Why the oracle rather than a synthetic control
//!
//! The obvious cheap witness is an identity wrapper — declare `down` and
//! `up` as identities at `latent == hidden` and require the routed
//! output to be unchanged. It is far too weak, and every one of these
//! defects survives it:
//!
//! ```text
//! router reads down(h) instead of h        I·h == h
//! the down projection is applied twice     I(I·h) == h
//! the shared branch reads the bottleneck   same vector either way
//! shared is summed before up, not after    I(r + s) == I(r) + s
//! ```
//!
//! A non-identity `D` with `U = D⁻¹` fixes the first three and still
//! cannot give an expected value, because `U·experts(D·x) != experts(x)`
//! — the experts are not equivariant under `D`, so there is nothing to
//! compare the answer TO without recomputing the branch.
//!
//! So the comparison is against the reference itself. The oracle's
//! `arms.latent.output` was computed by `transformers`' own forward with
//! the router on the un-projected input, the shared branch outside the
//! bottleneck, the norm on the weighted aggregate and each projection
//! applied once. Every one of those facts is load-bearing in the number,
//! and any of them misplaced here moves it. The container is built from
//! the oracle's OWN weights, so this is the production path executing
//! the reference's model, not a fixture agreeing with itself.
//!
//! # The arm was made to fail before it was believed
//!
//! A parity assertion that has never gone red is a claim about nothing.
//! Four defects were introduced into `RoutedOperands::apply` in turn and
//! `the_routed_path_reproduces_the_references_whole_block` caught every
//! one, with the other three arms staying green — so the failure was the
//! composition and not the fixture:
//!
//! ```text
//! router routed on the bottleneck        (router_input dropped)   RED
//! shared branch fed the bottleneck       (x -> expert_x)          RED
//! norm skipped on the aggregate          (norm -> None)           RED
//! down projection applied twice          (enter_latent twice)     RED
//! ```
//!
//! The first two are the placement facts the ORACLE could not build
//! mutants for: at its own geometry they fail on shape, so it recorded
//! them `structurally_unreachable` and witnessed them positively by
//! cross-form bit-identity. Here they are reachable, because the
//! executor binds `[experts, hidden]` against whatever vector it is
//! handed rather than raising — which is exactly why the composition
//! needed its own witness.
//!
//! # What the geometry buys
//!
//! `hidden = 10`, `latent = 7`, `intermediate = 4`, five experts, top-2.
//! The widths are pairwise distinct and `hidden / 2 = 5` is none of them,
//! so the routed bank cannot be bound at a width derived rather than
//! declared — the bank here is stored at 7 and the residual stream is 10,
//! which is the whole geometry this rung is about.

use serde_json::{json, Value};

use super::super::backend::WeightFormat;
use super::super::experts::FfnOperands;
use super::super::operands::OperandStore;
use super::super::production::ProductionBackend;
use super::ShardBuilder;
use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, LayerFfn, OperandRef, RoutedFfnOp};

/// The committed oracle — the same document `latent_moe_parity` reads.
const ORACLE: &str = include_str!("kimi_latent_moe_oracle.json");

/// Attention geometry, chosen only so a one-layer transformer closes;
/// `Q_HEADS * HEAD_DIM` must be the hidden width.
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 5;
const VOCAB: usize = 12;
/// TWO layers, and not for realism. With one, the expert bank's object
/// prefix absorbs the layer index and every expert tensor arrives
/// relative to the bank as `0.w1.weight` — the expert index and the
/// layer index become the same digit and the role table can classify
/// neither. Both layers carry the same weights; the assertions read
/// layer 0.
const LAYERS: usize = 2;

/// f32 over widths of 7 and 10, through a container round-trip. The
/// reference computed in f64 and the export wrote f32, so the floor is
/// the accumulated arithmetic difference, not a fitted band.
const TOLERANCE: f32 = 2e-6;

struct Oracle(Value);

impl Oracle {
    fn load() -> Self {
        Self(serde_json::from_str(ORACLE).expect("the oracle parses"))
    }

    fn usize_at(&self, key: &str) -> usize {
        self.0[key].as_u64().unwrap_or_else(|| panic!("{key}")) as usize
    }

    fn floats(node: &Value) -> Vec<f32> {
        node.as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_f64().expect("number") as f32)
            .collect()
    }

    fn weight(&self, name: &str) -> Vec<f32> {
        Self::floats(&self.0["weights"][name])
    }

    fn expert(&self, index: usize, matrix: &str) -> Vec<f32> {
        Self::floats(&self.0["weights"]["latent_experts"][index.to_string()][matrix])
    }

    fn input(&self) -> Vec<f32> {
        Self::floats(&self.0["input"])
    }

    fn latent_arm(&self, boundary: &str) -> Vec<f32> {
        Self::floats(&self.0["arms"]["latent"][boundary])
    }
}

/// The oracle's declaration, as a checkpoint would state it.
fn config(o: &Oracle) -> Value {
    json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "torch_dtype": "float32",
        "hidden_size": o.usize_at("hidden"),
        "num_hidden_layers": LAYERS,
        "intermediate_size": o.usize_at("hidden") * 2,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": o.0["latent_norm_eps"].as_f64().expect("eps"),
        "hidden_act": "silu",
        // Every layer is plain full attention, 1-indexed as the
        // checkpoint writes them. This rung is the routed branch; KDA or
        // MLA geometry here would test another rung's machinery.
        "linear_attn_config": {
            "kda_layers": [],
            "full_attn_layers": (1..=LAYERS).collect::<Vec<_>>()
        },
        "first_k_dense_replace": 0,
        "num_experts": o.usize_at("experts"),
        "num_experts_per_token": o.usize_at("top_k"),
        "num_shared_experts": 1,
        "moe_intermediate_size": o.usize_at("intermediate"),
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": o.0["moe_renormalize"].as_bool().expect("renormalize"),
        "routed_scaling_factor": o.0["routed_scaling_factor"].as_f64().expect("scale"),
        // The rung's two leaves, exactly as the oracle's own arm ran.
        "routed_expert_hidden_size": o.usize_at("latent"),
        "latent_moe_use_norm": true,
    })
}

/// A checkpoint carrying the oracle's weights verbatim.
fn shard(o: &Oracle) -> ShardBuilder {
    let hidden = o.usize_at("hidden");
    let latent = o.usize_at("latent");
    let inter = o.usize_at("intermediate");
    let experts = o.usize_at("experts");
    let mut shard = ShardBuilder::new();
    // The stack's own operands. Only the routed block's values matter to
    // the assertion; these exist so the layer is a transformer layer.
    let filler = |n: usize| vec![0.5f32; n];
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, hidden],
        &filler(VOCAB * hidden),
    );
    shard.push("model.norm.weight", &[hidden], &filler(hidden));
    shard.push("lm_head.weight", &[VOCAB, hidden], &filler(VOCAB * hidden));

    let q_rows = Q_HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
    // Two layers — see [`LAYERS`]. One collapses the layer and expert
    // indices into the same `0.*` spelling relative to the bank object,
    // so a one-layer fixture would exercise a naming accident instead of
    // the role classifier.
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![q_rows, hidden]),
            ("self_attn.k_proj.weight", vec![kv_rows, hidden]),
            ("self_attn.v_proj.weight", vec![kv_rows, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, q_rows]),
            ("input_layernorm.weight", vec![hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
        ] {
            let n = shape.iter().product();
            shard.push(&format!("{p}.{name}"), &shape, &filler(n));
        }

        // The routed block, from the oracle. The router is `[experts,
        // hidden]` — the un-projected width, which is the placement fact a
        // shape cannot express here and the reference's own geometry.
        let moe = format!("{p}.block_sparse_moe");
        shard.push(
            &format!("{moe}.gate.weight"),
            &[experts, hidden],
            &o.weight("router"),
        );
        shard.push(
            &format!("{moe}.gate.e_score_correction_bias"),
            &[experts],
            &o.weight("router_bias"),
        );
        // The shared branch, also at the un-projected width, in the
        // checkpoint's own gate/up/down spelling of the reference's w1/w3/w2.
        shard.push(
            &format!("{moe}.shared_experts.gate_proj.weight"),
            &[inter, hidden],
            &o.weight("shared_w1"),
        );
        shard.push(
            &format!("{moe}.shared_experts.up_proj.weight"),
            &[inter, hidden],
            &o.weight("shared_w3"),
        );
        shard.push(
            &format!("{moe}.shared_experts.down_proj.weight"),
            &[hidden, inter],
            &o.weight("shared_w2"),
        );
        // The wrapper: down crosses INTO the bottleneck, up crosses out.
        shard.push(
            &format!("{moe}.routed_expert_down_proj.weight"),
            &[latent, hidden],
            &o.weight("routed_expert_down_proj"),
        );
        shard.push(
            &format!("{moe}.routed_expert_norm.weight"),
            &[latent],
            &o.weight("routed_expert_norm"),
        );
        shard.push(
            &format!("{moe}.routed_expert_up_proj.weight"),
            &[hidden, latent],
            &o.weight("routed_expert_up_proj"),
        );
        // And the bank, stored at the LATENT width — the geometry the
        // declaration governs.
        for e in 0..experts {
            shard.push(
                &format!("{moe}.experts.{e}.w1.weight"),
                &[inter, latent],
                &o.expert(e, "w1"),
            );
            shard.push(
                &format!("{moe}.experts.{e}.w3.weight"),
                &[inter, latent],
                &o.expert(e, "w3"),
            );
            shard.push(
                &format!("{moe}.experts.{e}.w2.weight"),
                &[latent, inter],
                &o.expert(e, "w2"),
            );
        }
    }
    shard
}

struct Staged {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    store: OperandStore,
    op: RoutedFfnOp,
}

fn stage(o: &Oracle) -> Staged {
    let src = tempfile::tempdir().unwrap();
    std::fs::write(src.path().join("config.json"), config(o).to_string()).unwrap();
    shard(o).write(src.path());

    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("kimi-latent".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        outcome.closed(),
        "the oracle's own estate must close: {:?}",
        outcome.defects
    );
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let op = match &outcome.plan.expect("the op plan").layers[0].ffn {
        Some(LayerFfn::Routed(op)) => (**op).clone(),
        other => panic!("layer 0 planned {other:?}"),
    };
    Staged {
        _src: src,
        _container: container,
        store,
        op,
    }
}

fn load(op: &RoutedFfnOp, store: &OperandStore) -> FfnOperands {
    let format = WeightFormat::F32;
    FfnOperands::load(
        &LayerFfn::Routed(Box::new(op.clone())),
        store.into(),
        &|_: &OperandRef| Ok(format),
        format.into(),
        &|_: &OperandRef| Ok(format),
    )
    .expect("the routed operands load")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "boundary widths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// **The plan reads the declaration, and the bank is bound at the width
/// it names.**
///
/// Asserted before the arithmetic, because a plan that had quietly sized
/// the bank from `hidden` would fail to load rather than compute a wrong
/// answer, and the failure would name bytes rather than the declaration.
#[test]
fn the_planned_bank_is_bound_at_the_declared_latent_width() {
    let o = Oracle::load();
    let staged = stage(&o);
    let latent = staged
        .op
        .latent
        .as_ref()
        .expect("the declaration reaches the op");
    assert_eq!(latent.width, o.usize_at("latent"));
    assert_ne!(latent.width, o.usize_at("hidden"));
    assert_ne!(latent.width, o.usize_at("hidden") / 2);
    assert_ne!(latent.width, o.usize_at("intermediate"));
    assert_eq!(
        latent.down.shape,
        vec![o.usize_at("latent"), o.usize_at("hidden")],
        "down crosses INTO the bottleneck"
    );
    assert_eq!(
        latent.up.shape,
        vec![o.usize_at("hidden"), o.usize_at("latent")],
        "and up crosses out of it"
    );
    assert_eq!(
        staged.op.router.shape,
        vec![o.usize_at("experts"), o.usize_at("hidden")],
        "the router reads the un-projected block input"
    );
    // The layer's epsilon as the LAYER has it: every norm in the
    // component reaches its value through `ModelArchitecture::norm_eps`,
    // which narrows to f32. Reading the config's f64 for this one norm
    // alone would give it a better derivation of "the layer's epsilon"
    // than the layer's other norms have — one fact, two authorities,
    // differing in the eighth digit. The claim this rung makes is about
    // WHICH epsilon, and that claim is a factor of ten wide.
    let declared = o.0["latent_norm_eps"].as_f64().expect("eps");
    let carried = latent.norm.as_ref().expect("the flag declares one").eps;
    assert_eq!(carried, declared as f32 as f64);
    assert_ne!(
        carried,
        o.0["class_default_eps_not_used_here"]
            .as_f64()
            .expect("the oracle exports the epsilon it does NOT use"),
        "the norm must not have borrowed the class default the two preceding K3 rungs found"
    );
}

/// **The whole block, against the reference's `output`.**
///
/// `RoutedOperands::apply` returns `routed_out + shared_output`, which is
/// exactly what `latent_moe_block_forward` returns. Every placement fact
/// of the operator is load-bearing in this one number: the router on the
/// un-projected input, one down-projection, the norm on the weighted
/// aggregate, one up-projection, and the shared branch summed after it.
#[test]
fn the_routed_path_reproduces_the_references_whole_block() {
    let o = Oracle::load();
    let staged = stage(&o);
    let x = o.input();
    let got = load(&staged.op, &staged.store)
        .apply(
            &LayerFfn::Routed(Box::new(staged.op.clone())),
            &ProductionBackend::new(),
            &x,
            o.usize_at("hidden"),
        )
        .expect("the routed layer executes");

    let want = o.latent_arm("output");
    assert_eq!(
        got.len(),
        o.usize_at("hidden"),
        "back at the residual width"
    );
    let drift = max_abs_diff(&got, &want);
    assert!(
        drift < TOLERANCE,
        "the routed path differs from the reference by {drift}"
    );
}

/// **The band gate.** The reference's routed and shared contributions
/// must both be large enough for the comparison above to mean anything:
/// if the shared branch dominated, a wrapper applied to the wrong vector
/// would move the sum by less than the tolerance and the arm would pass
/// on a broken build.
///
/// Read off the ORACLE's own boundaries, so this is a property of the
/// fixture rather than of the implementation under test.
#[test]
fn the_fixture_can_see_a_misplaced_wrapper() {
    let o = Oracle::load();
    let routed_out = o.latent_arm("routed_out");
    let shared = o.latent_arm("shared_output");
    let scale = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        scale(&routed_out) > TOLERANCE * 1e3,
        "the routed contribution is too small to witness anything: {routed_out:?}"
    );
    assert!(
        scale(&shared) > TOLERANCE * 1e3,
        "the shared contribution is too small to witness its own placement: {shared:?}"
    );
    // And the two are genuinely different vectors, so summing them in
    // the wrong space cannot coincide with summing them in the right one.
    assert!(
        max_abs_diff(&routed_out, &shared) > TOLERANCE * 1e3,
        "the two branches are indistinguishable in this fixture"
    );
}

/// Operands and op must agree about whether a bottleneck exists. The
/// loader builds one from the other, so a disagreement is an interpreter
/// defect rather than anything a checkpoint can cause — and it is
/// refused rather than silently run at the wrong width.
#[test]
fn operands_and_op_that_disagree_about_the_wrapper_refuse() {
    let o = Oracle::load();
    let staged = stage(&o);
    let mut without = staged.op.clone();
    without.latent = None;
    let err = load(&staged.op, &staged.store)
        .apply(
            &LayerFfn::Routed(Box::new(without)),
            &ProductionBackend::new(),
            &o.input(),
            o.usize_at("hidden"),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("disagree about the latent branch"), "{err}");
}
