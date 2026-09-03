//! Post-norm placement EXECUTES at its declared causal position.
//!
//! The claim under test is not "a norm runs" but "the norm MOVED":
//!
//! ```text
//! pre-norm:   h + branch(norm(h))     the norm conditions the INPUT
//! post-norm:  h + norm(branch(h))     the norm conditions the OUTPUT
//! ```
//!
//! Two containers carry the SAME numeric weights and differ only in
//! which norm names they ship, so any behavioural difference between
//! them is placement and nothing else. That is what makes the
//! coincidence impossible: a build that ignored placement would produce
//! identical logits from the two, and the margin assertion below fails
//! loudly if the fixture is ever weakened into one where placement does
//! not matter.
//!
//! The sharpest arm is exact rather than statistical. Under post-norm
//! placement nothing at all is applied to the residual before attention,
//! so the attention input at layer 0 must be the embedding row **bit for
//! bit** — and the fixture generates that row from a known sequence, so
//! the test can state it rather than observe it.

use std::path::Path;

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::{
    lcg_values, norm_values, ShardBuilder, DENSE_HEAD_DIM, DENSE_HIDDEN, DENSE_INTERMEDIATE,
    DENSE_KV_HEADS, DENSE_LAYERS, DENSE_Q_HEADS, DENSE_VOCAB,
};
use crate::format::vindex3::graph::NormPlacement;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::observe::{InputSite, StepObserver};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

const TOKEN: u32 = 7;
/// The embedding seed the fixture uses, so the test can state the row.
const EMBED_SEED: u64 = 1;

/// Where a layer's two trunk norms are stored. The WEIGHTS are identical
/// in both — only the names, and therefore the judged placement, differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Estate {
    /// `input_layernorm` + `post_attention_layernorm` — the two-norm
    /// Llama estate, where the second name means the PRE-FFN norm.
    PreNorm,
    /// `post_attention_layernorm` + `post_feedforward_layernorm` — the
    /// OLMo-2 estate, where the first name means a true post-norm.
    PostNorm,
}

fn model(dir: &Path, estate: Estate) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "llama",
            "tie_word_embeddings": false,
            "hidden_size": DENSE_HIDDEN,
            "num_hidden_layers": DENSE_LAYERS,
            "intermediate_size": DENSE_INTERMEDIATE,
            "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS,
            "head_dim": DENSE_HEAD_DIM,
            "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        })
        .to_string(),
    )
    .unwrap();

    let q_rows = DENSE_Q_HEADS * DENSE_HEAD_DIM;
    let kv_rows = DENSE_KV_HEADS * DENSE_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, EMBED_SEED),
    );
    shard.push(
        "model.norm.weight",
        &[DENSE_HIDDEN],
        &norm_values(DENSE_HIDDEN, 2),
    );
    shard.push(
        "lm_head.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 3),
    );
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        for (leaf, shape, values) in [
            ("self_attn.q_proj.weight", vec![q_rows, DENSE_HIDDEN], seed),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, DENSE_HIDDEN],
                seed + 1,
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, DENSE_HIDDEN],
                seed + 2,
            ),
            (
                "self_attn.o_proj.weight",
                vec![DENSE_HIDDEN, q_rows],
                seed + 3,
            ),
            (
                "mlp.gate_proj.weight",
                vec![DENSE_INTERMEDIATE, DENSE_HIDDEN],
                seed + 6,
            ),
            (
                "mlp.up_proj.weight",
                vec![DENSE_INTERMEDIATE, DENSE_HIDDEN],
                seed + 7,
            ),
            (
                "mlp.down_proj.weight",
                vec![DENSE_HIDDEN, DENSE_INTERMEDIATE],
                seed + 8,
            ),
        ] {
            let n = shape.iter().product();
            shard.push(&format!("{prefix}.{leaf}"), &shape, &lcg_values(n, values));
        }
        // The two trunk norms. Same numbers, same order, different names
        // — this is the entire difference between the two containers.
        let (first, second) = match estate {
            Estate::PreNorm => ("input_layernorm", "post_attention_layernorm"),
            Estate::PostNorm => ("post_attention_layernorm", "post_feedforward_layernorm"),
        };
        shard.push(
            &format!("{prefix}.{first}.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 4),
        );
        shard.push(
            &format!("{prefix}.{second}.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 5),
        );
    }
    shard.write(dir);
}

fn plan_of(estate: Estate) -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let src = tempfile::tempdir().unwrap();
    model(src.path(), estate);
    let inventory =
        larql_models::inventory::build_inventory(src.path()).expect("fixture inventory");
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("target-artifact".to_string(), inventory)],
        container.path(),
    )
    .expect("fixture encodes");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome
        .plan
        .unwrap_or_else(|| panic!("{estate:?} must lower: {:?}", outcome.defects));
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Captures the values entering one site of one layer.
#[derive(Default)]
struct SiteCapture {
    layer: usize,
    site: Option<InputSite>,
    values: Vec<f32>,
}

impl StepObserver for SiteCapture {
    fn event(&mut self, _event: crate::format::vindex3::opplan::exec::observe::StepEvent) {}
    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if layer == self.layer && Some(site) == self.site {
            self.values = values.to_vec();
        }
    }
}

fn site_input(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    layer: usize,
    site: InputSite,
) -> Vec<f32> {
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(plan, store, &backend).unwrap();
    let mut capture = SiteCapture {
        layer,
        site: Some(site),
        values: Vec::new(),
    };
    session.step_observed(TOKEN, &mut capture).unwrap();
    capture.values
}

/// **The FFN must RUN.** Under post-norm placement the FFN has no
/// pre-norm, and gating the sublayer on that norm's presence skipped it
/// entirely — a stack that computes attention and no FFN, which still
/// produces finite planes and still differs from the pre-norm container,
/// so the margin arm below passed anyway. Real-checkpoint parity caught
/// it; this is the assertion that would have.
#[test]
fn a_post_norm_layer_still_runs_its_ffn_over_the_raw_residual() {
    let (_c, post, store) = plan_of(Estate::PostNorm);
    for layer in 0..post.layers.len() {
        assert!(
            post.layers[layer].pre_ffn_norm.is_none(),
            "premise: a post-norm layer carries no pre-FFN norm"
        );
        let ffn_in = site_input(&post, &store, layer, InputSite::Ffn);
        assert!(
            !ffn_in.is_empty(),
            "layer {layer}: the FFN sublayer did not run — its absent pre-norm is not \
             a reason to skip it"
        );
    }

    // And the control: a pre-norm stack observes the site too, so an
    // empty capture cannot be explained by the tap never firing.
    let (_c2, pre, store2) = plan_of(Estate::PreNorm);
    assert!(!site_input(&pre, &store2, 0, InputSite::Ffn).is_empty());
}

fn attention_input(plan: &ComponentOpPlan, store: &OperandStore, layer: usize) -> Vec<f32> {
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(plan, store, &backend).unwrap();
    let mut capture = SiteCapture {
        layer,
        site: Some(InputSite::Attention),
        values: Vec::new(),
    };
    session.step_observed(TOKEN, &mut capture).unwrap();
    assert!(!capture.values.is_empty(), "no attention input observed");
    capture.values
}

/// The embedding row the fixture stores for [`TOKEN`], stated from the
/// generator rather than read back — so the assertion below compares the
/// executor against the checkpoint, not against itself.
fn embedding_row() -> Vec<f32> {
    let all = lcg_values(DENSE_VOCAB * DENSE_HIDDEN, EMBED_SEED);
    let start = TOKEN as usize * DENSE_HIDDEN;
    all[start..start + DENSE_HIDDEN].to_vec()
}

/// The two estates judge two different placements, and the post-norm one
/// carries no pre-sublayer norm at all.
#[test]
fn the_two_estates_judge_two_placements_and_carry_different_norm_sites() {
    let (_c, pre, _s) = plan_of(Estate::PreNorm);
    let (_c2, post, _s2) = plan_of(Estate::PostNorm);

    for layer in &pre.layers {
        assert!(
            layer.pre_attention_norm.is_some(),
            "pre-norm keeps its input norm"
        );
        assert!(
            layer.pre_ffn_norm.is_some(),
            "pre-norm keeps its pre-FFN norm"
        );
        assert!(layer.post_attention_norm.is_none());
        assert!(layer.post_ffn_norm.is_none());
    }
    for layer in &post.layers {
        assert!(
            layer.pre_attention_norm.is_none(),
            "a post-norm layer must carry NO pre-attention norm"
        );
        assert!(
            layer.pre_ffn_norm.is_none(),
            "a post-norm layer must carry NO pre-FFN norm"
        );
        assert!(layer.post_attention_norm.is_some());
        assert!(layer.post_ffn_norm.is_some());
        // The epsilon the QK norm runs at no longer rides on a norm site
        // that this placement does not have.
        assert!(layer.declared_norm_eps > 0.0);
    }
    assert!(NormPlacement::PostOnly.unimplemented_reason().is_none());
}

/// **The exact arm.** Under post-norm placement nothing conditions the
/// residual before attention, so layer 0's attention input IS the
/// embedding row. Under pre-norm placement it is not — which is the
/// control proving the assertion can fail.
#[test]
fn a_post_norm_layer_feeds_attention_the_raw_residual() {
    let (_c, post, store) = plan_of(Estate::PostNorm);
    let observed = attention_input(&post, &store, 0);
    assert_eq!(
        observed,
        embedding_row(),
        "post-norm attention must read the RAW residual, bit for bit"
    );

    let (_c2, pre, store2) = plan_of(Estate::PreNorm);
    let observed_pre = attention_input(&pre, &store2, 0);
    assert_ne!(
        observed_pre,
        embedding_row(),
        "the pre-norm control must condition its attention input — if it \
         does not, the assertion above proves nothing"
    );
}

/// **The margin arm.** Same weights, different placement, materially
/// different logits. A build that ignored placement would produce two
/// identical rows here.
#[test]
fn placement_is_load_bearing_end_to_end() {
    let (_c, post, store) = plan_of(Estate::PostNorm);
    let (_c2, pre, store2) = plan_of(Estate::PreNorm);
    let backend = ReferenceBackend::new();

    let a = DecodeSession::new(&post, &store, &backend)
        .unwrap()
        .step(TOKEN)
        .unwrap()
        .logits;
    let b = DecodeSession::new(&pre, &store2, &backend)
        .unwrap()
        .step(TOKEN)
        .unwrap()
        .logits;
    let a = a.expect("post-norm step returns logits");
    let b = b.expect("pre-norm step returns logits");
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().all(|v| v.is_finite()),
        "post-norm logits must be finite"
    );

    let max_gap = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    assert!(
        max_gap > scale * 1e-3,
        "the two placements produced near-identical logits (max gap {max_gap}, \
         scale {scale}); this fixture can no longer tell them apart, so every \
         other assertion in this file is worthless"
    );
}
