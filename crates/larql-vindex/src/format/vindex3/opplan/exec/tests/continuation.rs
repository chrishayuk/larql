//! QW-3: heterogeneous continuation state, over Qwen3.8's real topology.
//!
//! The claim is not that the enum has the right variants. It is that the
//! **memory model** differs: a recurrent layer's footprint is constant in
//! context length and a KV layer's is not. That is checked as arithmetic at
//! two very different context lengths, not asserted from a label.
//!
//! The 64-layer plan here is assembled directly rather than read from a
//! container, because Qwen3.8 is not yet encodable (24 blocking findings,
//! mostly vision). Its geometry and cadence are the real checkpoint's,
//! verified against it in `vindex3 plan`; what is synthetic is the plan
//! object, not the architecture it describes.

use super::super::continuation::{
    plan_continuation_geometry, ContinuationState, LayerContinuationGeometry,
    LayerContinuationState, RecurrentBufferGeometry, RecurrentGeometry, RecurrentState,
    StateInitialization,
};
use super::super::kv::LayerKvGeometry;
use super::super::kv::try_plan_kv_geometry;
use crate::format::vindex3::opplan::{ComponentOpPlan, GatedDeltaOp, LayerAttention};
use larql_models::inventory::report::RecurrentStateDtype;

/// Qwen3.8-27B, from its own config.
const LAYERS: usize = 64;
const FULL_ATTENTION_INTERVAL: usize = 4;
const VALUE_HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
/// `2*Hk*Dk + Hv*Dv` — the fused projection's channel count, which the
/// convolution runs over. 10240 on Qwen3.8.
const CONV_CHANNELS: usize = 2 * KEY_HEADS * HEAD_DIM + VALUE_HEADS * HEAD_DIM;
/// Full-attention layers: GQA 24/4 at head_dim 256.
const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 256;

/// 48 · 128 · 128 · 4 bytes ≈ 3 MB per recurrent layer.
const STATE_ELEMENTS_PER_LAYER: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;

fn gated_delta_op() -> GatedDeltaOp {
    let operand = || crate::format::vindex3::opplan::OperandRef {
        object: "target.decoder_stack".into(),
        tensor: "synthetic".into(),
        dtype: "BF16".into(),
        shape: vec![1],
    };
    GatedDeltaOp {
        num_key_heads: KEY_HEADS,
        num_value_heads: VALUE_HEADS,
        key_head_dim: HEAD_DIM,
        value_head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        state_dtype: Some(RecurrentStateDtype::Float32),
        in_proj_qkv: operand(),
        in_proj_a: operand(),
        in_proj_b: operand(),
        in_proj_z: operand(),
        conv1d: operand(),
        a_log: operand(),
        dt_bias: operand(),
        norm: operand(),
        out_proj: operand(),
    }
}

/// A 64-layer hybrid on Qwen3.8's LLLF cadence.
fn qwen_like_plan() -> ComponentOpPlan {
    let mut plan =
        super::plan_fixtures::softmax_plan_with_layers(LAYERS, FULL_KV_HEADS, FULL_HEAD_DIM);
    for (index, layer) in plan.layers.iter_mut().enumerate() {
        if index % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1 {
            layer.attention = LayerAttention::GatedDelta(Box::new(gated_delta_op()));
        }
    }
    plan
}

/// The census, on the real cadence.
#[test]
fn the_hybrid_topology_resolves_to_the_right_mix() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).expect("every layer declares its state");
    assert_eq!(geometry.len(), LAYERS);

    let recurrent = geometry.iter().filter(|g| g.recurrent().is_some()).count();
    let kv = geometry.iter().filter(|g| g.kv().is_some()).count();
    assert_eq!((recurrent, kv), (48, 16), "Qwen3.8 is 48 recurrent + 16 KV");

    let cadence: String = geometry
        .iter()
        .map(|g| if g.recurrent().is_some() { 'L' } else { 'F' })
        .collect();
    assert!(
        cadence.as_bytes().chunks(4).all(|c| c == b"LLLF"),
        "cadence is not uniform LLLF: {cadence}"
    );

    let r = geometry[0].recurrent().expect("layer 0 is recurrent");
    // TWO buffers, not one. QW-3.6: a DeltaNet layer's durable state is
    // the delta matrix AND the causal convolution's history, and this
    // assertion used to describe only the first while calling it the
    // layer's state.
    assert_eq!(r.buffers.len(), 2, "delta matrix + conv history");

    let matrix = &r.buffers[0];
    assert_eq!(matrix.shape, vec![VALUE_HEADS, HEAD_DIM, HEAD_DIM]);
    assert_eq!(matrix.dtype, RecurrentStateDtype::Float32);
    assert_eq!(matrix.initialization, StateInitialization::Zeros);
    assert_eq!(matrix.elements(), STATE_ELEMENTS_PER_LAYER);

    let conv = &r.buffers[1];
    assert_eq!(conv.shape.len(), 2, "channels x kernel");
    assert_eq!(conv.shape[1], CONV_KERNEL);
    assert_eq!(conv.initialization, StateInitialization::Zeros);

    // The layer's total is the sum, and the matrix alone is not it.
    assert_eq!(r.elements(), matrix.elements() + conv.elements());
    assert!(r.elements() > STATE_ELEMENTS_PER_LAYER);
}

/// Claim 1: no fake KV. Claim 2: no phantom recurrence.
///
/// Both are the same failure in opposite directions — a planner that sized
/// every layer the same way would pass a census and still allocate wrongly.
#[test]
fn neither_kind_of_layer_allocates_the_other_kind_of_state() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    for (index, g) in geometry.iter().enumerate() {
        let recurrent_layer = index % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1;
        if recurrent_layer {
            assert!(
                g.kv().is_none(),
                "layer {index} allocated KV for a recurrence"
            );
        } else {
            assert!(
                g.recurrent().is_none(),
                "layer {index} allocated recurrent state for softmax attention"
            );
        }
    }
}

/// Claims 3 and 4, and the whole architectural point: the two mechanisms
/// respond differently to context length.
#[test]
fn only_the_softmax_layers_grow_with_context() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let total = |positions: usize, pick: fn(&LayerContinuationGeometry) -> bool| -> usize {
        geometry
            .iter()
            .filter(|g| pick(g))
            .map(|g| g.elements_at(positions))
            .sum()
    };
    let is_recurrent = |g: &LayerContinuationGeometry| g.recurrent().is_some();
    let is_kv = |g: &LayerContinuationGeometry| g.kv().is_some();

    let (short, long) = (128usize, 32_768usize);
    let rec_short = total(short, is_recurrent);
    let rec_long = total(long, is_recurrent);
    let kv_short = total(short, is_kv);
    let kv_long = total(long, is_kv);

    assert_eq!(
        rec_short, rec_long,
        "recurrent footprint moved with context: {rec_short} -> {rec_long}"
    );
    assert!(
        kv_long > kv_short,
        "KV footprint did not grow with context: {kv_short} -> {kv_long}"
    );
    assert_eq!(
        kv_long / kv_short,
        long / short,
        "KV should scale linearly in positions"
    );

    // 48 layers, whatever the context. The total is the DELTA MATRIX plus
    // the convolution history — 144 MiB + 7.5 MiB. It used to be asserted
    // as the matrix alone, which is what a missing buffer looks like from
    // the outside: a number that adds up against itself.
    let matrix = 48 * STATE_ELEMENTS_PER_LAYER;
    let conv = 48 * CONV_CHANNELS * CONV_KERNEL;
    assert_eq!(rec_short, matrix + conv);
    let mib = |n: usize| n * 4 / (1024 * 1024);
    assert_eq!(mib(matrix), 144, "delta matrices");
    assert_eq!(mib(conv), 7, "convolution histories (7.5 MiB)");
    assert!(
        (150..=153).contains(&mib(rec_short)),
        "recurrent total is {} MiB, expected ~151.5",
        mib(rec_short)
    );
}

/// The compatibility seam must fail loudly on a hybrid, not answer for the
/// layers it happens to understand.
#[test]
fn the_kv_adapter_refuses_a_hybrid_model() {
    let err = try_plan_kv_geometry(&qwen_like_plan())
        .expect_err("a model with recurrent layers has no flat KV geometry");
    assert!(
        err.contains("plan_continuation_geometry"),
        "the refusal must name the seam that can describe this model: {err}"
    );
}

/// The control: a wholly-softmax model is unchanged, and the adapter still
/// answers for it exactly as before.
#[test]
fn a_pure_softmax_model_is_untouched() {
    let plan = super::plan_fixtures::softmax_plan_with_layers(LAYERS, FULL_KV_HEADS, FULL_HEAD_DIM);
    let geometry = plan_continuation_geometry(&plan).unwrap();
    assert_eq!(geometry.iter().filter(|g| g.kv().is_some()).count(), LAYERS);
    assert!(geometry.iter().all(|g| g.recurrent().is_none()));

    let flat = try_plan_kv_geometry(&plan).expect("a softmax model still flattens");
    assert_eq!(flat.len(), LAYERS);
    for (g, kv) in geometry.iter().zip(&flat) {
        assert_eq!(g.kv().unwrap(), kv, "the adapter changed a KV geometry");
    }
}

/// The type-level control: `RecurrentGeometry` must not have quietly
/// learned what a Gated DeltaNet is.
///
/// Qwen3.8's state happens to be rank-3 and head-shaped. If any of that had
/// leaked into the type — a head count, a rank assumption, a square
/// `Dk × Dv` — then KDA, whose state is a different shape under a different
/// update rule, would need a second variant and this abstraction would have
/// failed at its first real test. So the control states states that are
/// deliberately nothing like DeltaNet's and requires the same arithmetic to
/// hold.
#[test]
fn recurrent_geometry_does_not_assume_the_operator() {
    use super::super::continuation::RecurrentGeometry;

    let shapes: [Vec<usize>; 4] = [
        vec![7],                               // rank 1: a plain vector of state
        vec![13, 5],                           // rank 2, non-square, no head axis
        vec![2, 3, 5, 7],                      // rank 4
        vec![VALUE_HEADS, HEAD_DIM, HEAD_DIM], // and DeltaNet's, for contrast
    ];
    for shape in shapes {
        let expected: usize = shape.iter().product();
        let g = RecurrentGeometry::single(RecurrentBufferGeometry {
            shape: shape.clone(),
            dtype: RecurrentStateDtype::Float32,
            initialization: StateInitialization::Zeros,
        });
        assert_eq!(g.elements(), expected, "{shape:?}");
        assert_eq!(g.bytes(), expected * 4, "{shape:?}");

        // The defining property, independent of shape: constant in context.
        let layer = LayerContinuationGeometry::Recurrent(g);
        assert_eq!(layer.elements_at(1), expected, "{shape:?}");
        assert_eq!(layer.elements_at(1_000_000), expected, "{shape:?}");
    }
}

/// A recurrence whose state precision was never declared cannot be sized,
/// and the planner says so rather than choosing one.
#[test]
fn an_undeclared_state_precision_refuses_rather_than_defaulting() {
    let mut plan = qwen_like_plan();
    if let LayerAttention::GatedDelta(op) = &mut plan.layers[0].attention {
        op.state_dtype = None;
    }
    let err =
        plan_continuation_geometry(&plan).expect_err("an unsized recurrence must not be planned");
    assert!(
        err.contains("layer 0") && err.contains("precision"),
        "the refusal must name the layer and the missing fact: {err}"
    );
}

/// The runtime mirror allocates exactly what the geometry asked for.
///
/// The geometry gates proved the PLAN is honest. This proves the
/// allocation follows it — a planner that described 48 recurrent layers and
/// a runtime that then allocated 64 KV stores would pass every earlier test
/// in this file.
#[test]
fn the_runtime_state_allocates_what_the_geometry_asked_for() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let state = ContinuationState::prepare(&geometry);
    assert_eq!(state.len(), LAYERS);
    assert_eq!(state.position(), 0, "a fresh state continues from 0");

    let mut recurrent = 0;
    let mut kv = 0;
    for index in 0..LAYERS {
        match state.layer(index) {
            LayerContinuationState::Recurrent(r) => {
                recurrent += 1;
                assert_eq!(r.buffer(0).shape(), [VALUE_HEADS, HEAD_DIM, HEAD_DIM]);
                assert_eq!(r.buffer(0).cells().len(), STATE_ELEMENTS_PER_LAYER);
                assert!(
                    (0..r.len()).all(|i| r.buffer(i).cells().iter().all(|c| *c == 0.0)),
                    "layer {index} did not start from zeros"
                );
            }
            LayerContinuationState::Kv(rows) => {
                kv += 1;
                assert!(
                    rows.keys().is_empty() && rows.values().is_empty(),
                    "layer {index} preallocated KV rows before any position"
                );
            }
            LayerContinuationState::Hybrid { .. } => {
                panic!("layer {index} is hybrid; this fixture declares none")
            }
            LayerContinuationState::Stateless => panic!("layer {index} is stateless"),
        }
    }
    assert_eq!((recurrent, kv), (48, 16));
}

/// The same asymmetry the geometry proved, now in the allocated state: a
/// recurrent layer holds a buffer from step zero, a KV layer holds nothing
/// until positions arrive.
#[test]
fn the_two_kinds_of_state_grow_differently() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let mut state = ContinuationState::prepare(&geometry);

    // Nothing has been appended, so only the recurrent buffers exist.
    let resident: usize = (0..LAYERS)
        .filter_map(|i| {
            state.layer(i).recurrent().map(|r| {
                (0..r.len())
                    .map(|b| r.buffer(b).cells().len())
                    .sum::<usize>()
            })
        })
        .sum();
    assert_eq!(
        resident,
        48 * (STATE_ELEMENTS_PER_LAYER + CONV_CHANNELS * CONV_KERNEL),
        "delta matrices AND convolution histories"
    );

    // Append one position to every KV layer; the recurrent buffers must not
    // move.
    let width = geometry
        .iter()
        .find_map(|g| g.kv().map(|kv| kv.kv_dim))
        .unwrap();
    for index in 0..LAYERS {
        if let Some(rows) = state.layer_mut(index).kv_mut() {
            rows.append(vec![0.0; width], vec![0.0; width]);
        }
    }
    let after: usize = (0..LAYERS)
        .filter_map(|i| {
            state.layer(i).recurrent().map(|r| {
                (0..r.len())
                    .map(|b| r.buffer(b).cells().len())
                    .sum::<usize>()
            })
        })
        .sum();
    assert_eq!(after, resident, "a KV append resized a recurrent buffer");
    assert_eq!(state.elements_at(1), resident + 16 * width * 2);

    state.set_position(1);
    assert_eq!(state.position(), 1, "position is owned, not derived");
}

/// A recurrent buffer only accepts cells its geometry can hold.
#[test]
fn a_recurrent_state_refuses_the_wrong_number_of_cells() {
    use super::super::continuation::{RecurrentBuffer, RecurrentBufferGeometry};
    let g = RecurrentBufferGeometry {
        shape: vec![4, 5],
        dtype: RecurrentStateDtype::Float32,
        initialization: StateInitialization::Zeros,
    };
    assert!(RecurrentBuffer::from_cells(&g, vec![0.0; 20]).is_ok());
    let err = RecurrentBuffer::from_cells(&g, vec![0.0; 19]).expect_err("19 != 4*5");
    assert!(err.contains("19") && err.contains("20"), "{err}");
}

// ── The two-region shape (KvAndRecurrent) — the hybrid conv-QKV
//    attention layer's continuation, in the vocabulary's own terms ──

fn two_region_geometry() -> LayerContinuationGeometry {
    LayerContinuationGeometry::KvAndRecurrent {
        kv: LayerKvGeometry {
            kv_dim: 4,
            window: None,
        },
        recurrent: RecurrentGeometry::single(RecurrentBufferGeometry {
            shape: vec![16, 2],
            dtype: RecurrentStateDtype::Float32,
            initialization: StateInitialization::Zeros,
        }),
    }
}

/// **The two-region layer's arithmetic is the sum of its claims**: the
/// KV half scales with positions, the conv history does not — the same
/// asymmetry the pure variants state, on one layer.
#[test]
fn a_two_region_layer_grows_on_one_side_only() {
    let g = two_region_geometry();
    assert_eq!(g.elements_at(0), 32, "the buffer exists before any position");
    assert_eq!(g.elements_at(10), 10 * 4 * 2 + 32);
    // And the pure variants keep their own answers beside it.
    let kv = LayerContinuationGeometry::Kv(LayerKvGeometry {
        kv_dim: 4,
        window: None,
    });
    assert_eq!(kv.elements_at(10), 80);
    assert_eq!(LayerContinuationGeometry::Stateless.elements_at(10), 0);
    let r = RecurrentGeometry::single(RecurrentBufferGeometry {
        shape: vec![16, 2],
        dtype: RecurrentStateDtype::Float32,
        initialization: StateInitialization::Zeros,
    });
    assert_eq!(r.elements(), 32);
    assert_eq!(r.bytes(), 128, "f32 cells");
    assert_eq!(
        LayerContinuationGeometry::Recurrent(r).elements_at(10),
        32
    );
}

/// **`kv()` refuses the two-region layer; `kv_side()` serves it.** The
/// split is the fail-closed contract: `kv()` is what the KV-only
/// provider default projects through, and answering the hybrid's KV
/// half there would allocate the cache while silently dropping the conv
/// history — half a continuation that looks whole. A provider that
/// genuinely holds both asks through `kv_side()`.
#[test]
fn the_kv_accessor_refuses_what_the_kv_side_accessor_serves() {
    let g = two_region_geometry();
    assert!(g.kv().is_none(), "kv() must not hand out half a continuation");
    assert_eq!(g.kv_side().map(|kv| kv.kv_dim), Some(4));
    assert_eq!(
        g.recurrent().map(|r| r.elements()),
        Some(32),
        "the buffer half answers through the same accessor a pure recurrence uses"
    );
    // A pure KV layer answers both spellings identically.
    let kv = LayerContinuationGeometry::Kv(LayerKvGeometry {
        kv_dim: 4,
        window: None,
    });
    assert_eq!(kv.kv().map(|k| k.kv_dim), Some(4));
    assert_eq!(kv.kv_side().map(|k| k.kv_dim), Some(4));
    assert!(kv.recurrent().is_none());
}

/// **A KV-only provider fails CLOSED on the two-region layer** — the
/// default `prepare_continuation` projection refuses rather than
/// allocating the cache without the conv history. The failure the
/// accessor split above exists to force.
#[test]
fn a_kv_only_provider_refuses_the_two_region_layer() {
    use super::super::kv::{ContinuationError, ContinuationProvider};

    struct RowsOnly;
    impl ContinuationProvider for RowsOnly {
        fn prepare(&mut self, _layers: &[LayerKvGeometry]) {}
        fn append(&mut self, _layer: usize, _key: Vec<f32>, _value: Vec<f32>) {}
        fn keys(&self, _layer: usize) -> &[Vec<f32>] {
            &[]
        }
        fn values(&self, _layer: usize) -> &[Vec<f32>] {
            &[]
        }
        fn position(&self) -> usize {
            0
        }
        fn set_position(&mut self, _position: usize) {}
        fn recurrent_state(
            &mut self,
            layer: usize,
        ) -> Result<&mut RecurrentState, ContinuationError> {
            Err(ContinuationError::RecurrentUnsupported {
                provider: "RowsOnly",
                layer,
            })
        }
    }

    let layers = [
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: 4,
            window: None,
        }),
        two_region_geometry(),
    ];
    let err = RowsOnly
        .prepare_continuation(&layers)
        .expect_err("half a continuation must not be allocated");
    assert_eq!(
        err,
        ContinuationError::RecurrentUnsupported {
            provider: "a KV-only provider",
            layer: 1,
        }
    );
    let rendered = err.to_string();
    assert!(rendered.contains("refusing"), "{rendered}");
}

/// The runtime state allocates BOTH regions for the two-region layer —
/// empty rows and a zeroed buffer — and counts them the way the
/// geometry promised.
#[test]
fn the_two_region_state_allocates_rows_and_buffer_together() {
    let geometry = [
        two_region_geometry(),
        LayerContinuationGeometry::Stateless,
    ];
    let state = ContinuationState::prepare(&geometry);
    let LayerContinuationState::Hybrid { rows, state: buffer } = state.layer(0) else {
        panic!("layer 0 must hold both regions: {:?}", state.layer(0));
    };
    assert!(rows.keys().is_empty() && rows.values().is_empty());
    assert_eq!(buffer.buffer(0).cells().len(), 32);
    assert!(buffer.buffer(0).cells().iter().all(|c| *c == 0.0));
    assert!(matches!(
        state.layer(1),
        LayerContinuationState::Stateless
    ));
    // The runtime state's KV width is learned from appended rows (the
    // same convention the pure-KV arm holds), so a fresh state counts
    // only the buffer — the region that exists from step zero.
    assert_eq!(state.elements_at(5), 32);
}
