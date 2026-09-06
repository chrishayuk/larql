//! **`token_forward` on a tiny synthetic model** — the composition
//! `embed → stack → final norm → lm_head`, checked without a
//! multi-gigabyte fixture.
//!
//! `token_real.rs`/`token2_real.rs` already gate this path against the
//! real checkpoint, but both are env-gated on a fixture that is not in
//! the repo, so nothing exercised the module in an ordinary run. What
//! is proven here is what the module itself adds — the embedding
//! gather, the final norm, the `lm_head` matvec, the argmax, and the
//! state threading across positions — with a one-layer dense stack so
//! the already-gated attention and FFN arithmetic is not re-derived.

use larql_models::config::{KdaGeometry, NormType};

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda::{zero_state, KdaOutputGateWeights, KdaWeights};
use crate::format::vindex3::opplan::exec::kernels::norm;
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;
use crate::format::vindex3::opplan::exec::stack::{
    LayerAttention, LayerFfn, LayerSpec, LayerState,
};
use crate::format::vindex3::opplan::exec::token::{embed, token_forward, EmbeddingRow};

const HIDDEN: usize = 8;
const INTER: usize = 4;
const HEADS: usize = 2;
const DIM: usize = 4;
const KERNEL: usize = 2;
const VOCAB: usize = 5;
const EPS: f64 = 1e-5;

fn geometry() -> KdaGeometry {
    KdaGeometry {
        num_heads: HEADS,
        head_dim: DIM,
        conv_kernel: KERNEL,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.25)
        .collect()
}

fn codes(n: usize, seed: f32) -> Vec<u16> {
    synth(n, seed)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect()
}

/// One dense KDA layer's owned weights.
struct Tiny {
    kda_f32: Vec<Vec<f32>>,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
    embeddings: Vec<Vec<f32>>,
}

impl Tiny {
    fn new() -> Self {
        let width = HEADS * DIM;
        Self {
            kda_f32: vec![
                synth(width * KERNEL, 0.5),
                synth(width * KERNEL, 1.5),
                synth(width * KERNEL, 2.5),
                synth(DIM * HIDDEN, 6.1),
                synth(width * DIM, 7.2),
                synth(DIM * HIDDEN, 8.3),
                synth(width * DIM, 9.4),
                synth(HEADS * HIDDEN, 10.5),
                synth(HEADS, 11.6),
                synth(width, 12.7),
                synth(DIM, 13.8).iter().map(|v| v + 1.0).collect(),
            ],
            q: codes(width * HIDDEN, 0.1),
            k: codes(width * HIDDEN, 1.3),
            v: codes(width * HIDDEN, 2.7),
            o: codes(HIDDEN * width, 5.5),
            gate: codes(INTER * HIDDEN, 3.0),
            up: codes(INTER * HIDDEN, 9.0),
            down: codes(HIDDEN * INTER, 15.0),
            input_norm: synth(HIDDEN, 2.2).iter().map(|v| v + 1.0).collect(),
            post_norm: synth(HIDDEN, 3.3).iter().map(|v| v + 1.0).collect(),
            final_norm: synth(HIDDEN, 4.4).iter().map(|v| v + 1.0).collect(),
            lm_head: synth(VOCAB * HIDDEN, 5.1),
            embeddings: (0..VOCAB).map(|i| synth(HIDDEN, 0.7 + i as f32)).collect(),
        }
    }

    fn rows(&self) -> Vec<EmbeddingRow<'_>> {
        self.embeddings
            .iter()
            .enumerate()
            .map(|(id, vector)| EmbeddingRow { id, vector })
            .collect()
    }

    fn spec(&self) -> LayerSpec<'_> {
        let f = &self.kda_f32;
        LayerSpec {
            attention: LayerAttention::Kda(
                KdaWeights {
                    q_proj: WeightRows::Bf16(&self.q),
                    k_proj: WeightRows::Bf16(&self.k),
                    v_proj: WeightRows::Bf16(&self.v),
                    q_conv1d: &f[0],
                    k_conv1d: &f[1],
                    v_conv1d: &f[2],
                    f_a_proj: &f[3],
                    f_b_proj: &f[4],
                    output_gate: KdaOutputGateWeights::LowRank {
                        g_a_proj: &f[5],
                        g_b_proj: &f[6],
                    },
                    b_proj: &f[7],
                    a_log: &f[8],
                    dt_bias: &f[9],
                    o_norm: &f[10],
                    o_proj: WeightRows::Bf16(&self.o),
                    norm_eps: EPS as f32,
                    // The rank the gate factorisations meet at — this fixture's
                    // own `f_a_proj`, not the head dim the executor used to assume.
                    gate_rank: f[3].len() / HIDDEN,
                },
                geometry(),
            ),
            ffn: LayerFfn::Dense {
                weights: ExpertWeights {
                    gate: &self.gate,
                    up: &self.up,
                    down: &self.down,
                },
                inter: INTER,
            },
            input_norm_weight: &self.input_norm,
            post_attention_norm_weight: &self.post_norm,
            norm_eps: EPS,
        }
    }

    fn forward(&self, token: usize, states: &mut [LayerState]) -> super::super::token::TokenTrace {
        let rows = self.rows();
        let spec = self.spec();
        token_forward(
            token,
            HIDDEN,
            &rows,
            std::slice::from_ref(&spec),
            states,
            &self.final_norm,
            EPS,
            &self.lm_head,
            VOCAB,
        )
    }
}

fn states() -> Vec<LayerState> {
    vec![LayerState::Kda(zero_state(geometry()))]
}

/// The embedding gather is a LOOKUP: it returns the loaded row for that
/// id, never a computed one, and never a neighbouring row.
#[test]
fn the_embedding_gather_returns_the_row_the_token_names() {
    let t = Tiny::new();
    let rows = t.rows();
    for id in 0..VOCAB {
        assert_eq!(embed(&rows, id), t.embeddings[id].as_slice());
    }
    // Sparse loading: only the rows a caller supplies exist, and the
    // gather finds them by ID rather than by position.
    let sparse = vec![
        EmbeddingRow {
            id: 4,
            vector: &t.embeddings[4],
        },
        EmbeddingRow {
            id: 1,
            vector: &t.embeddings[1],
        },
    ];
    assert_eq!(embed(&sparse, 1), t.embeddings[1].as_slice());
    assert_eq!(embed(&sparse, 4), t.embeddings[4].as_slice());
}

/// A token id with no loaded row is a construction error, not a zero
/// vector — a silently-zero embedding would produce a plausible
/// distribution from a token the fixture never loaded.
#[test]
#[should_panic(expected = "has no loaded embedding row")]
fn a_token_without_a_loaded_row_panics_rather_than_returning_zeros() {
    let t = Tiny::new();
    let rows = vec![EmbeddingRow {
        id: 0,
        vector: &t.embeddings[0],
    }];
    embed(&rows, 3);
}

/// **The composition is what it claims**: every boundary the trace
/// reports is recomputed from the one before it, so a reordering or a
/// dropped stage cannot pass.
#[test]
fn each_traced_boundary_follows_from_the_previous_one() {
    let t = Tiny::new();
    let mut s = states();
    let trace = t.forward(2, &mut s);

    assert_eq!(trace.embedding, t.embeddings[2], "embed(token 2)");
    assert_eq!(trace.layers.len(), 1, "one layer, one trace entry");
    assert_eq!(
        trace.stack_output(),
        trace.layers[0].layer_output.as_slice(),
        "stack_output names the last layer's output"
    );

    // final_normed = RMSNorm(stack_output) with the final norm weight.
    let want_norm = norm(
        NormType::RmsNorm,
        trace.stack_output(),
        &t.final_norm,
        0.0,
        EPS,
    );
    assert_eq!(trace.final_normed, want_norm);

    // logits = lm_head @ final_normed, a SEPARATE matrix from the
    // embedding table (`tie_word_embeddings=False`).
    assert_eq!(trace.logits.len(), VOCAB);
    for (v, row) in trace.logits.iter().zip(t.lm_head.chunks_exact(HIDDEN)) {
        let want: f32 = row
            .iter()
            .zip(&trace.final_normed)
            .map(|(a, b)| a * b)
            .sum();
        assert!((v - want).abs() < 1e-5, "logit {v} vs {want}");
    }
    let want_argmax = trace
        .logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .expect("non-empty")
        .0;
    assert_eq!(trace.argmax, want_argmax);
    assert!(trace.logits.iter().all(|v| v.is_finite()));
}

/// **State is threaded across positions.** The same token fed twice
/// must not produce the same trace, because the recurrent state has
/// advanced — and a freshly zeroed state must reproduce the first call
/// exactly.
#[test]
fn state_carries_across_positions_and_resets_reproduce_the_first() {
    let t = Tiny::new();
    let mut s = states();
    let first = t.forward(1, &mut s);
    let second = t.forward(1, &mut s);
    assert_ne!(
        first.logits, second.logits,
        "a carried recurrent state must change the second call's answer"
    );

    let mut fresh = states();
    let again = t.forward(1, &mut fresh);
    assert_eq!(
        first.logits, again.logits,
        "a zeroed state must reproduce the first position exactly"
    );
    assert_eq!(first, again, "every boundary, not only the logits");
}
