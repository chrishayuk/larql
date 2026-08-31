//! What has to survive between decode steps.
//!
//! LARQL no longer has a KV-cache abstraction. It has a **continuation
//! state** abstraction, of which KV is one form. That is not a rename: a
//! hybrid checkpoint carries two genuinely different mechanisms at once,
//! and Qwen3.8-27B carries 48 of one and 16 of the other.
//!
//! ```text
//! softmax layer     grows a sequence-indexed K/V store
//! recurrent layer   carries a fixed-size matrix, constant in sequence length
//! ```
//!
//! # The division of labour
//!
//! Continuation geometry says **what must persist**. The operator says
//! **how it changes**. Nothing here knows what a Gated DeltaNet is, and
//! that is the point: the next consumer of [`RecurrentGeometry`] is Kimi's
//! KDA, a different update rule over the same kind of durable storage. If
//! this type had been shaped around DeltaNet's semantics, KDA would need a
//! second variant and the abstraction would have failed at its first real
//! test.
//!
//! # Why an enum rather than `Option<LayerKvGeometry>`
//!
//! `Option` would collapse four distinct facts into one `None`: no state at
//! all, recurrent state instead of KV, unknown, and unsupported. A planner
//! reading `None` could not tell "this layer needs nothing" from "this
//! layer needs something I cannot describe", and would size memory for both
//! identically.

use larql_models::inventory::report::RecurrentStateDtype;

use super::super::{ComponentOpPlan, LayerAttention};
use super::kv::LayerKvGeometry;

/// How a recurrent state begins a sequence.
///
/// Recorded rather than assumed. A zero start is what the judged operators
/// declare, but "the reference starts from zeros" and "any recurrence must
/// start from zeros" are different claims, and only the first is evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateInitialization {
    Zeros,
}

/// ONE durable buffer whose size does not depend on how many positions
/// have been seen.
///
/// Deliberately says nothing about the operator that owns it — no head
/// semantics, no update rule, no family name. `shape` is whatever that
/// operator's buffer is; for a Gated DeltaNet layer buffer 0 happens to be
/// `[value_heads, key_head_dim, value_head_dim]`, but this type does not
/// know that and must not learn it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentBufferGeometry {
    pub shape: Vec<usize>,
    pub dtype: RecurrentStateDtype,
    pub initialization: StateInitialization,
}

impl RecurrentBufferGeometry {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.elements()
            * match self.dtype {
                RecurrentStateDtype::Float32 => 4,
            }
    }
}

/// A recurrent layer's COMPLETE durable state: one or more buffers.
///
/// Plural because a recurrence is not always one tensor, and discovering
/// that late is expensive. Gated DeltaNet keeps two — the delta matrix and
/// a causal-convolution history — and HF's cache holds them as separate
/// `recurrent_states` and `conv_states` fields for exactly that reason.
/// Modelling only the matrix made the layer's persistent state look
/// complete while a whole buffer was missing, and no whole-prefix forward
/// could notice: the convolution history is reconstructible from the batch
/// when the batch IS the prefix, and only becomes load-bearing at
/// `seq_len == 1`.
///
/// Which buffer means what is the OPERATOR's knowledge. This type, and
/// everything below it in the storage layer, still knows nothing about
/// Gated DeltaNet, KDA, or any other rule — it knows sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentGeometry {
    pub buffers: Vec<RecurrentBufferGeometry>,
}

impl RecurrentGeometry {
    /// A single-buffer recurrence, for operators that keep one tensor.
    pub fn single(buffer: RecurrentBufferGeometry) -> Self {
        Self {
            buffers: vec![buffer],
        }
    }

    pub fn elements(&self) -> usize {
        self.buffers
            .iter()
            .map(RecurrentBufferGeometry::elements)
            .sum()
    }

    pub fn bytes(&self) -> usize {
        self.buffers
            .iter()
            .map(RecurrentBufferGeometry::bytes)
            .sum()
    }
}

/// ONE row per position, of a width the OPERATOR decides — a cache that
/// grows with the sequence but is not a K/V pair.
///
/// MLA is the case that forced it: what an MLA layer retains is the
/// compressed latent plus one shared rope-K, `kv_lora_rank + rope`
/// elements per position, RAW — decompression into per-head K and V
/// happens at read time and is never itself cached. Sizing that as
/// [`LayerKvGeometry`] would claim two rows where the model keeps one,
/// and would have to name a `kv_dim` no operand has; sizing it as
/// [`RecurrentGeometry`] would claim a fixed-size buffer for something
/// that grows with the prefix. It is a third fact, so it gets a third
/// variant — the same argument the enum's own doc comment makes about
/// `Option<LayerKvGeometry>`, one level along.
///
/// Deliberately no dtype field: the row store is f32 throughout, exactly
/// as [`LayerKvGeometry`]'s rows are, and a precision this schema cannot
/// yet vary must not be given a knob that implies it can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerLatentKvGeometry {
    /// Elements retained per position — one row, not a pair.
    pub width: usize,
}

/// One layer's continuation requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerContinuationGeometry {
    /// Grows with the sequence.
    Kv(LayerKvGeometry),
    /// Grows with the sequence, one operator-defined row per position
    /// rather than a K/V pair. See [`LayerLatentKvGeometry`].
    LatentKv(LayerLatentKvGeometry),
    /// Constant in the sequence.
    Recurrent(RecurrentGeometry),
    /// Grows with the sequence AND keeps fixed buffers beside it — the
    /// conv-QKV attention shape: a real per-position KV cache plus a
    /// conv history over the pre-conv fused QKV. Its own variant, not a
    /// flag on [`Self::Kv`]: a provider that can hold only rows must
    /// fail closed on this layer, never quietly allocate half of it.
    KvAndRecurrent {
        kv: LayerKvGeometry,
        recurrent: RecurrentGeometry,
    },
    /// Nothing survives this layer.
    ///
    /// No judged operator produces this yet — it is here because "this
    /// layer needs nothing" is a distinct fact from "this layer needs
    /// something I cannot describe", and collapsing them is exactly what
    /// `Option<LayerKvGeometry>` did.
    Stateless,
}

impl LayerContinuationGeometry {
    /// Elements this layer must retain after `positions` steps.
    ///
    /// The asymmetry is the whole architectural claim, expressed as
    /// arithmetic rather than as a label: a KV layer's answer scales with
    /// `positions` and a recurrent layer's does not.
    pub fn elements_at(&self, positions: usize) -> usize {
        match self {
            // K and V, one row of `kv_dim` each, per position.
            Self::Kv(kv) => kv.kv_dim * 2 * positions,
            // ONE row of `width` per position — the compression IS the
            // difference from the arm above, and it shows here.
            Self::LatentKv(latent) => latent.width * positions,
            Self::Recurrent(r) => r.elements(),
            Self::KvAndRecurrent { kv, recurrent } => {
                kv.kv_dim * 2 * positions + recurrent.elements()
            }
            Self::Stateless => 0,
        }
    }

    /// The KV geometry of a layer that keeps ONLY rows. Deliberately
    /// `None` on [`Self::KvAndRecurrent`]: this accessor is what the
    /// KV-only provider default projects through, and answering the
    /// hybrid's KV half there would allocate the cache while silently
    /// dropping the conv history — half a continuation that looks whole.
    pub fn kv(&self) -> Option<&LayerKvGeometry> {
        match self {
            Self::Kv(kv) => Some(kv),
            _ => None,
        }
    }

    /// The KV side wherever one exists — [`Self::Kv`] and
    /// [`Self::KvAndRecurrent`] alike. For providers and consumers that
    /// genuinely serve both regions; a KV-only path must keep using
    /// [`Self::kv`].
    pub fn kv_side(&self) -> Option<&LayerKvGeometry> {
        match self {
            Self::Kv(kv) | Self::KvAndRecurrent { kv, .. } => Some(kv),
            _ => None,
        }
    }

    pub fn recurrent(&self) -> Option<&RecurrentGeometry> {
        match self {
            Self::Recurrent(r) | Self::KvAndRecurrent { recurrent: r, .. } => Some(r),
            _ => None,
        }
    }

    /// The latent-cache geometry of a layer that keeps operator-defined
    /// rows. `None` everywhere else — including on [`Self::Kv`], whose
    /// rows are a K/V pair and are served by [`Self::kv`].
    pub fn latent_kv(&self) -> Option<LayerLatentKvGeometry> {
        match self {
            Self::LatentKv(latent) => Some(*latent),
            _ => None,
        }
    }
}

/// What a layer keeps, in words — the one place this vocabulary lives.
///
/// A refusal that says "recurrent" for a layer holding a growing latent
/// cache sends its reader to the wrong seam, so the naming is a function
/// rather than a phrase repeated at each refusal site.
pub fn region_name(geometry: &LayerContinuationGeometry) -> &'static str {
    match geometry {
        LayerContinuationGeometry::Kv(_) => "sequence-indexed KV rows",
        LayerContinuationGeometry::LatentKv(_) => {
            "a per-position LATENT cache, one operator-defined row per position"
        }
        LayerContinuationGeometry::Recurrent(_) => "recurrent continuation state",
        LayerContinuationGeometry::KvAndRecurrent { .. } => "KV rows AND recurrent buffers",
        LayerContinuationGeometry::Stateless => "no continuation state at all",
    }
}

/// Every layer's continuation requirement, in layer order.
///
/// The authoritative seam. [`plan_kv_geometry`](super::kv::plan_kv_geometry)
/// remains only as a compatibility adapter and refuses any model this
/// cannot be flattened for.
pub fn plan_continuation_geometry(
    plan: &ComponentOpPlan,
) -> Result<Vec<LayerContinuationGeometry>, String> {
    plan.layers
        .iter()
        .map(|layer| match &layer.attention {
            LayerAttention::Softmax(op) => Ok(LayerContinuationGeometry::Kv(LayerKvGeometry {
                kv_dim: op.num_kv_heads * op.head_dim,
                window: op.window,
            })),
            // KDA's state geometry is known — one `Dk × Dv` matrix per
            // head — but the precision to hold it at is not: KDA declares
            // no `mamba_ssm_dtype`, and picking one here would run the
            // recurrence at a precision its author never chose. That is
            // the refusal `GatedDelta` makes below for an undeclared
            // dtype, and KDA is in that position for every checkpoint
            // observed so far. Execution is out of scope for the rung that
            // introduced this operator; refusing is what keeps it out.
            // Conv-QKV attention's continuation is TWO regions: a real
            // per-position KV cache (K/V cached post-conv, post-rotary)
            // AND a conv history holding the last `conv_kernel` positions
            // of the PRE-conv fused QKV — the reference's own cache
            // shape (`conv_states[layer]`: full kernel width,
            // left-padded). fp32 for the same transcribed reason the
            // mixer's regions are: the reference executor computes in
            // f32 over widened operands.
            LayerAttention::ConvQkv(op) => Ok(LayerContinuationGeometry::KvAndRecurrent {
                kv: LayerKvGeometry {
                    kv_dim: op.geometry.num_kv_heads * op.geometry.head_dim,
                    window: None,
                },
                recurrent: super::conv_qkv::conv_history_geometry(op),
            }),
            // KDA's state geometry is known — one `Dk × Dv` matrix per
            // head, plus the three convolution windows — and since the
            // execution rung the precision is known too: no checkpoint
            // declares one, and the reference recurrence
            // (`fla`'s `naive_recurrent_kda`, transcribed and sha-pinned
            // in `scripts/kda_reference.py`) holds its state in fp32, so
            // the state is held at the precision the reference computes
            // at. That judgment is a transcription and lives in exactly
            // one place, `exec::kda::state_geometry` — the same footing
            // as Mamba2's below, and the reason this arm no longer
            // refuses.
            LayerAttention::Kda(op) => Ok(LayerContinuationGeometry::Recurrent(
                super::kda::state_geometry(op.geometry()),
            )),
            // MLA retains a real per-position cache (compressed, not
            // absent — see `MlaOp::compressed_kv_width`), so sizing it is
            // a real question with a real answer, unlike KDA's above. It
            // is refused anyway: the operand-closure rung that introduced
            // this operator deliberately does not reach execution, and
            // answering `LayerContinuationGeometry::Kv` here would need a
            // KV shape this planner's `LayerKvGeometry` cannot state
            // (`compressed_kv_width` per position, not `num_kv_heads ×
            // head_dim`) — inventing one now is exactly the kind of
            // plausible-but-unbuilt executor KDA's own refusal exists to
            // avoid becoming.
            // MLA retains a real per-position cache — compressed, not
            // absent — and the schema can now state it: ONE row of
            // `compressed_kv_width` per position, which is what the
            // operator caches (raw, pre-norm) and what it decompresses at
            // read time. It is NOT `LayerContinuationGeometry::Kv`: that
            // arm means a K/V pair of `kv_dim` each, and answering it here
            // would claim two rows the model does not keep, at a width no
            // operand has. The third arm is the fact.
            LayerAttention::Mla(op) => {
                Ok(LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry {
                    width: op.compressed_kv_width(),
                }))
            }
            // Mamba2's geometry is fully declared, and the execution rung
            // brought the precision judgment with it: the reference's own
            // naive path computes the scan in fp32 (explicit `.float()`
            // casts), so the state is held at the precision the reference
            // computes at — a transcription, not a planner's pick. The
            // judgment lives once, in `exec::mamba2::state_geometry`.
            LayerAttention::Mamba2(op) => Ok(LayerContinuationGeometry::Recurrent(
                super::mamba2::state_geometry(op),
            )),
            LayerAttention::GatedDelta(op) => {
                // A recurrence must be held at SOME precision, and the
                // planner does not get to pick one. An undeclared state
                // dtype means the checkpoint never said, or said it in a
                // spelling this build does not represent — either way,
                // sizing it at the model's bulk dtype would run the
                // recurrence at a precision its author did not choose, and
                // the whole reason `mamba_ssm_dtype` exists is that the two
                // differ. Refuse instead.
                let dtype = op.state_dtype.ok_or_else(|| {
                    format!(
                        "layer {} carries a recurrence whose state precision the \
                         checkpoint never declared; continuation state cannot be \
                         sized without it",
                        layer.layer
                    )
                })?;
                Ok(LayerContinuationGeometry::Recurrent(RecurrentGeometry {
                    buffers: vec![
                        // Buffer 0 — the delta matrix, at the precision
                        // the checkpoint declared for its recurrence.
                        RecurrentBufferGeometry {
                            shape: vec![op.num_value_heads, op.key_head_dim, op.value_head_dim],
                            dtype,
                            initialization: StateInitialization::Zeros,
                        },
                        // Buffer 1 — the causal convolution's history:
                        // the last `conv_kernel` positions of the fused
                        // projection, per channel.
                        //
                        // Its precision is NOT `mamba_ssm_dtype`. That
                        // key governs the recurrence; HF seeds this
                        // buffer from the projection activations
                        // (`F.pad(mixed_qkv, …)`) and so holds it at the
                        // activation dtype. The two are independently
                        // determined and only coincide here because the
                        // reference path is f32 throughout — recorded so
                        // a later bf16 path does not inherit the wrong
                        // one from a shared field.
                        RecurrentBufferGeometry {
                            shape: vec![op.qkv_channels(), op.conv_kernel],
                            dtype: RecurrentStateDtype::Float32,
                            initialization: StateInitialization::Zeros,
                        },
                    ],
                }))
            }
        })
        .collect()
}

/// One layer's durable state, mirroring [`LayerContinuationGeometry`].
///
/// Storage only. There is deliberately no update method, no callback and no
/// operator handle on any variant: the state owns what survives between
/// steps, and the operator owns how it changes. A state that knew how a
/// Gated DeltaNet updates itself would have to learn KDA's rule too, and
/// then every future recurrence's — which is how a storage type becomes a
/// dispatch table.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerContinuationState {
    /// Sequence-indexed rows, appended per position.
    Kv(LayerKvRows),
    /// Sequence-indexed rows of ONE operator-defined species, appended
    /// per position — see [`LayerLatentKvGeometry`].
    LatentKv(LatentKvRows),
    /// A fixed-size buffer the operator reads and rewrites in place.
    Recurrent(RecurrentState),
    /// Both regions, on one layer — rows AND a fixed buffer, the
    /// conv-QKV attention shape.
    Hybrid {
        rows: LayerKvRows,
        state: RecurrentState,
    },
    Stateless,
}

/// One latent-cache layer's retained rows: one row per position, of the
/// width its operator declared.
///
/// Storage only, and deliberately unnamed as to CONTENT — this type does
/// not learn that MLA's row is a compressed latent followed by a shared
/// rope-K, any more than [`RecurrentBuffer`] learns that buffer 1 is a
/// convolution history. The operator that wrote the geometry is the one
/// that knows how to split the row.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LatentKvRows {
    rows: Vec<Vec<f32>>,
}

impl LatentKvRows {
    pub fn append(&mut self, row: Vec<f32>) {
        self.rows.push(row);
    }

    /// Every row appended, position-ordered from 0.
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// One softmax layer's retained rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerKvRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl LayerKvRows {
    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
    }

    pub fn keys(&self) -> &[Vec<f32>] {
        &self.keys
    }

    pub fn values(&self) -> &[Vec<f32>] {
        &self.values
    }
}

/// ONE of a recurrence's durable buffers.
///
/// Carries its shape so a consumer can index it, and nothing else. It does
/// not know which operator owns it, nor which buffer of that operator it
/// is: the same type serves Gated DeltaNet's delta matrix, its convolution
/// history, and KDA's buffers unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentBuffer {
    shape: Vec<usize>,
    cells: Vec<f32>,
}

impl RecurrentBuffer {
    /// The zero start a sequence begins from.
    pub fn zeros(geometry: &RecurrentBufferGeometry) -> Self {
        match geometry.initialization {
            StateInitialization::Zeros => Self {
                shape: geometry.shape.clone(),
                cells: vec![0.0; geometry.elements()],
            },
        }
    }

    /// Adopt existing cells — a captured state, or one being resumed.
    pub fn from_cells(geometry: &RecurrentBufferGeometry, cells: Vec<f32>) -> Result<Self, String> {
        if cells.len() != geometry.elements() {
            return Err(format!(
                "state has {} cells, this geometry needs {}",
                cells.len(),
                geometry.elements()
            ));
        }
        Ok(Self {
            shape: geometry.shape.clone(),
            cells,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [f32] {
        &mut self.cells
    }
}

/// A recurrent layer's complete durable state, mirroring
/// [`RecurrentGeometry`].
///
/// Indexed, not named: the storage layer does not learn that buffer 1 is a
/// convolution history. The operator that wrote the geometry is the one
/// that knows, and it is the only thing that indexes.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentState {
    buffers: Vec<RecurrentBuffer>,
}

impl RecurrentState {
    pub fn zeros(geometry: &RecurrentGeometry) -> Self {
        Self {
            buffers: geometry
                .buffers
                .iter()
                .map(RecurrentBuffer::zeros)
                .collect(),
        }
    }

    /// A single-buffer state, for operators that keep one tensor.
    pub fn single(buffer: RecurrentBuffer) -> Self {
        Self {
            buffers: vec![buffer],
        }
    }

    pub fn buffer(&self, index: usize) -> &RecurrentBuffer {
        &self.buffers[index]
    }

    pub fn buffer_mut(&mut self, index: usize) -> &mut RecurrentBuffer {
        &mut self.buffers[index]
    }

    /// Two buffers at once, for an operator that reads one and writes the
    /// other in the same step. Panics if the indices are equal, which
    /// would alias.
    pub fn buffer_pair_mut(
        &mut self,
        a: usize,
        b: usize,
    ) -> (&mut RecurrentBuffer, &mut RecurrentBuffer) {
        assert_ne!(a, b, "a buffer cannot be borrowed mutably twice");
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (left, right) = self.buffers.split_at_mut(hi);
        let (x, y) = (&mut left[lo], &mut right[0]);
        if a < b {
            (x, y)
        } else {
            (y, x)
        }
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// A component's whole continuation state, one entry per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationState {
    layers: Vec<LayerContinuationState>,
    position: usize,
}

impl ContinuationState {
    /// Allocate exactly what each layer's geometry asks for — and nothing
    /// for the layers that ask for nothing.
    pub fn prepare(geometry: &[LayerContinuationGeometry]) -> Self {
        Self {
            layers: geometry
                .iter()
                .map(|g| match g {
                    LayerContinuationGeometry::Kv(_) => {
                        LayerContinuationState::Kv(LayerKvRows::default())
                    }
                    LayerContinuationGeometry::LatentKv(_) => {
                        LayerContinuationState::LatentKv(LatentKvRows::default())
                    }
                    LayerContinuationGeometry::Recurrent(r) => {
                        LayerContinuationState::Recurrent(RecurrentState::zeros(r))
                    }
                    LayerContinuationGeometry::KvAndRecurrent { recurrent, .. } => {
                        LayerContinuationState::Hybrid {
                            rows: LayerKvRows::default(),
                            state: RecurrentState::zeros(recurrent),
                        }
                    }
                    LayerContinuationGeometry::Stateless => LayerContinuationState::Stateless,
                })
                .collect(),
            position: 0,
        }
    }

    pub fn layer(&self, index: usize) -> &LayerContinuationState {
        &self.layers[index]
    }

    pub fn layer_mut(&mut self, index: usize) -> &mut LayerContinuationState {
        &mut self.layers[index]
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The logical position this state continues from. Owned explicitly,
    /// never derived from a row count — the same contract
    /// [`KvState`](super::kv::KvState) states, and for the same reason: a
    /// recurrent layer retains no rows at all, so a count could not answer
    /// it even in principle.
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Total elements retained after `positions` steps.
    pub fn elements_at(&self, positions: usize) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerContinuationState::Kv(rows) => {
                    rows.keys.first().map_or(0, |r| r.len()) * 2 * positions
                }
                LayerContinuationState::LatentKv(rows) => {
                    rows.rows.first().map_or(0, |r| r.len()) * positions
                }
                LayerContinuationState::Recurrent(r) => {
                    (0..r.len()).map(|i| r.buffer(i).cells().len()).sum()
                }
                LayerContinuationState::Hybrid { rows, state } => {
                    rows.keys.first().map_or(0, |r| r.len()) * 2 * positions
                        + (0..state.len())
                            .map(|i| state.buffer(i).cells().len())
                            .sum::<usize>()
                }
                LayerContinuationState::Stateless => 0,
            })
            .sum()
    }
}

impl LayerContinuationState {
    pub fn kv_mut(&mut self) -> Option<&mut LayerKvRows> {
        match self {
            Self::Kv(rows) => Some(rows),
            _ => None,
        }
    }

    pub fn recurrent_mut(&mut self) -> Option<&mut RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }

    pub fn recurrent(&self) -> Option<&RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }

    pub fn latent_kv_mut(&mut self) -> Option<&mut LatentKvRows> {
        match self {
            Self::LatentKv(rows) => Some(rows),
            _ => None,
        }
    }

    pub fn latent_kv(&self) -> Option<&LatentKvRows> {
        match self {
            Self::LatentKv(rows) => Some(rows),
            _ => None,
        }
    }
}
