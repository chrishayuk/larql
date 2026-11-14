//! Caller-owned continuation state for the executor's traversals
//! (VI3-INF-2/3).
//!
//! [`DecodeSession`](super::decode::DecodeSession) used to own its K/V
//! rows as private per-layer `Vec<Vec<f32>>`. That was right for
//! proving decode semantics and wrong as the production
//! continuation-state architecture: the state a conversation carries
//! between steps is *policy* (residency, quantisation, windowing,
//! checkpointing), and policy composes outside the executor. This
//! module is the seam: the session drives a [`KvState`] provider and
//! owns none of the rows.
//!
//! The provider learns its geometry **from the plan** —
//! [`plan_kv_geometry`] reads each layer's KV row width and attention
//! window out of the executable program itself. No head-count
//! inference from a family registry, no `ModelArchitecture` questions:
//! sliding/full and head dims are explicit properties of the program.
//!
//! Contract notes:
//!
//! - Rows are stored exactly as the backend returned them (post-norm,
//!   post-rope) and lent through a logical view
//!   ([`KvView`](super::kv_view::KvView)) over absolute positions
//!   `[base, end)`, in whatever representation the provider holds them
//!   (CONTINUATION-VIEW-1). The plan's retention authority
//!   ([`HistoryRange`]) says which positions a step may still read; a
//!   provider must hold every one of those and may drop anything before.
//!   A step lacking a required row is refused by name when it is built
//!   ([`AttentionStepCall::new`](super::backend::AttentionStepCall::new)).

use std::ops::Range;

use super::super::super::graph::policy::AttentionSpan;
use super::super::{AttentionOp, ComponentOpPlan};
use super::continuation::{LatentKvRows, LayerContinuationGeometry, RecurrentState};
use super::continuation_authority::ContinuationConfig;
use super::continuation_identity::ContinuationIdentity;
use super::continuation_registry::{BoxedContinuation, ContinuationFactory, ContinuationRegion};

/// Which history a step may still read: the SINGLE retention authority
/// (CONTINUATION-VIEW-1, V1).
///
/// The plan declares the earliest logical position a step can still
/// request. A provider must retain everything from there on and is never
/// required to retain anything earlier. Every backend takes its source
/// floor from [`required_range`](Self::required_range); nothing else
/// derives one from a span or a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRange {
    /// Every position from 0 remains readable.
    Full,
    /// Only the trailing `w` positions (the current one included) remain
    /// readable.
    Trailing(usize),
}

impl HistoryRange {
    /// The policy, written once. Exhaustive over the span vocabulary on
    /// purpose: a `_` arm would let the next span kind mean "whole
    /// prefix", or worse "these positions are dead", without anyone
    /// deciding it.
    pub fn of_span(span: AttentionSpan, window: Option<usize>) -> Self {
        match (span, window) {
            (AttentionSpan::Sliding, Some(window)) => Self::Trailing(window),
            // A sliding layer with no declared window has no bound to apply.
            (AttentionSpan::Sliding, None) | (AttentionSpan::Full, _) => Self::Full,
            // A spatial window's extent is not a position count, so no
            // position is dead because of it (see `AttentionSpan::Windowed`).
            (AttentionSpan::Windowed, _) => Self::Full,
        }
    }

    /// The first position a step at `position` may read.
    pub fn required_start(self, position: usize) -> usize {
        match self {
            Self::Full => 0,
            Self::Trailing(window) => (position + 1).saturating_sub(window),
        }
    }

    /// The positions a step at `position` may read: everything a provider
    /// must still hold for it, and nothing more.
    pub fn required_range(self, position: usize) -> Range<usize> {
        self.required_start(position)..position + 1
    }
}

impl AttentionOp {
    /// This layer's retention authority, from its declared span and window.
    pub fn history(&self) -> HistoryRange {
        HistoryRange::of_span(self.span, self.window)
    }
}

/// One layer's continuation-state geometry, read from the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvGeometry {
    /// Row width of one position's K (and V): `num_kv_heads * head_dim`.
    pub kv_dim: usize,
    /// Width of one head within that row, as the plan's op declares it —
    /// never inferred from `kv_dim`, which a block size could otherwise
    /// silently span heads with.
    pub head_dim: usize,
    /// The layer's attention window in positions; `None` = full span.
    /// A geometry fact, NOT a retention permission: it is copied from the
    /// op whatever its span, so a full or spatially windowed layer may
    /// carry one. Retention is [`history`](Self::history).
    pub window: Option<usize>,
    /// Which history a step may still read — the retention authority.
    pub history: HistoryRange,
}

impl LayerKvGeometry {
    /// The positions a step at `position` may read on this layer.
    pub fn required_range(&self, position: usize) -> Range<usize> {
        self.history.required_range(position)
    }
}

/// Every layer's [`LayerKvGeometry`], in layer order.
///
/// **Compatibility adapter.** The authoritative seam is
/// [`plan_continuation_geometry`](super::continuation::plan_continuation_geometry),
/// which describes what each layer must retain without assuming it is KV.
/// This flattens that back to the older shape and exists only until its
/// callers migrate.
///
/// It REFUSES a model carrying any non-KV continuation rather than
/// answering for the layers it happens to understand. Returning only the
/// softmax layers would silently renumber them; returning a zero-width row
/// for a recurrence would claim a KV store the layer does not have and
/// mis-size every allocation downstream. Qwen3.8 is 48 of 64 such layers,
/// so this is the common case for a hybrid, not an edge.
pub fn plan_kv_geometry(plan: &ComponentOpPlan) -> Vec<LayerKvGeometry> {
    try_plan_kv_geometry(plan).unwrap_or_else(|e| panic!("{e}"))
}

/// [`plan_kv_geometry`] without the panic, for callers that can refuse.
pub fn try_plan_kv_geometry(plan: &ComponentOpPlan) -> Result<Vec<LayerKvGeometry>, String> {
    let continuation = super::continuation::plan_continuation_geometry(plan)?;
    continuation
        .iter()
        .enumerate()
        .map(|(index, geometry)| {
            geometry.kv().cloned().ok_or_else(|| {
                // Name the region actually there. "Recurrent" was the
                // only alternative to rows when this adapter was
                // written; a latent cache is a third answer, and a
                // caller told the wrong one looks for the wrong seam.
                let region = super::continuation::region_name(geometry);
                format!(
                    "layer {index} carries {region}, not a plain KV cache; this model \
                     needs `plan_continuation_geometry`, which describes every form"
                )
            })
        })
        .collect()
}

/// Per-layer K/V continuation state, owned by the caller for its
/// entire lifetime — execution modes merely consume and update it.
/// The batch prefill ([`prefill_plan`](super::prefill_plan)) and the
/// incremental [`DecodeSession`](super::decode::DecodeSession) drive
/// the **same** provider; there is no batch-state → decode-state
/// translation anywhere.
///
/// Every traversal calls [`prepare`](Self::prepare) before driving the
/// provider, reads earlier rows, and appends new positions' pairs.
/// The decode step appends layer 0..n for position p, then again for
/// p+1; the batch prefill appends all positions for layer 0, then all
/// for layer 1 — a provider must not assume one interleaving.
pub trait ContinuationProvider {
    /// Announce the traversal's geometry before any append. An
    /// announcement, **not** a reset: a provider already holding rows
    /// (a prefilled state being resumed) keeps them.
    fn prepare(&mut self, layers: &[LayerKvGeometry]);

    /// Append one position's K and V rows for `layer`.
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>);

    /// The K and V rows `layer` holds, lent as a logical view over
    /// ABSOLUTE positions `[base, end)`: `end` is every row ever appended
    /// (never a physical count), `base` the first still held. A provider
    /// lends its storage as it holds it — one allocation per row, a
    /// contiguous matrix, or anything behind
    /// [`KvRows`](super::kv_view::KvRows) — and may drop rows only before
    /// the plan's required range
    /// ([`HistoryRange`](HistoryRange)); a step lacking one is refused.
    fn rows(&self, layer: usize) -> super::kv_view::KvView<'_>;

    /// Make `layer`'s held rows readable: called immediately before every
    /// [`rows`](Self::rows) for that layer. A provider that lends what it
    /// already holds has nothing to do; one that stores rows in another
    /// form (compressed) materialises the layer's live range here, into
    /// storage it owns and reuses. Required, with no default, so a
    /// wrapper that forwards the trait cannot silently skip it — and a
    /// read it did not precede is a view whose `end` is not the step's
    /// position, which the step refuses.
    fn prepare_layer(&mut self, layer: usize);

    /// The logical continuation position: the next position this state
    /// continues from. Owned explicitly by the provider — **never**
    /// derived from a physical row count, because a windowed or
    /// compressed provider may one day retain fewer rows than the
    /// positions it logically represents. A session resuming over this
    /// state starts here; there is no separate start-position argument
    /// anywhere that could disagree with it.
    fn position(&self) -> usize;

    /// Record that the state now continues from `position`. The
    /// driving traversal is the only writer, and calls are monotonic —
    /// once per consumed position on the decode path, once at the end
    /// of a batch prefill. Moving the position BACK is never done here:
    /// that is [`truncate`](Self::truncate), which also restores the rows.
    fn set_position(&mut self, position: usize);

    /// Discard every position at or after the ABSOLUTE `position`, so the
    /// state continues from `position` exactly as if it had never advanced
    /// further. `position` is a sequence position, not a count of rows to
    /// drop.
    ///
    /// This restores LOGICAL state, not a counter. After it returns `Ok`,
    /// every later read must see what a provider that only ever advanced
    /// to `position` would show, including every earlier row a sliding
    /// window may still reach. A provider that has evicted or overwritten
    /// such rows cannot honour that and must refuse.
    ///
    /// All or nothing: the whole rewind is validated, across every layer,
    /// before anything is changed. An `Err` leaves the provider exactly as
    /// it was. Rewinding to the current position is permitted and discards
    /// any rows held beyond it; rewinding forward is refused.
    ///
    /// The default refuses. Rollback is a capability a provider states by
    /// implementing it, never one it inherits.
    fn truncate(&mut self, position: usize) -> Result<(), ContinuationError> {
        Err(ContinuationError::TruncateUnsupported { position })
    }

    /// Announce the FULL continuation geometry, KV and recurrent alike.
    ///
    /// Separate from [`prepare`](Self::prepare) so a provider that holds
    /// only KV needs no change: the default projects to the KV subset and
    /// delegates. A provider that holds recurrent buffers overrides this
    /// and allocates them here.
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        let kv: Vec<LayerKvGeometry> = layers.iter().filter_map(|g| g.kv().cloned()).collect();
        // Providers are indexed by ABSOLUTE layer index. Filtering to the
        // KV subset preserves that only when every layer keeps rows —
        // true for the KV-only providers this default exists for, and
        // false the moment a stack is hybrid, which is why a hybrid plan
        // must have been refused by `recurrent_state` before reaching a
        // provider that took this default.
        if kv.len() != layers.len() {
            let layer = layers.iter().position(|g| g.kv().is_none()).unwrap_or(0);
            // Name the region actually missing. Reporting every
            // non-KV layer as "no recurrent state" would send a serving
            // cache after the wrong half of a hybrid.
            return Err(match layers.get(layer) {
                Some(LayerContinuationGeometry::LatentKv(_)) => {
                    ContinuationError::LatentUnsupported {
                        provider: "a KV-only provider",
                        layer,
                    }
                }
                _ => ContinuationError::RecurrentUnsupported {
                    provider: "a KV-only provider",
                    layer,
                },
            });
        }
        self.prepare(&kv);
        Ok(())
    }

    /// This layer's durable recurrent buffers.
    ///
    /// **Required, and returns a Result rather than an Option.** A
    /// provider that cannot hold recurrent state must say so — an
    /// `Option` here would make "I have no buffers" and "this operator
    /// needs none" the same answer, which is precisely the ambiguity
    /// [`LayerContinuationGeometry::Stateless`] exists to prevent one
    /// level down. Every implementor states its position explicitly;
    /// nothing inherits an answer by omission.
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError>;

    /// This layer's per-position latent cache — the rows an operator
    /// whose continuation is ONE row per position reads and extends (an
    /// MLA layer's compressed latent + shared rope-K).
    ///
    /// **Required, and a Result**, for the same reason
    /// [`recurrent_state`](Self::recurrent_state) is: a provider that
    /// cannot hold these rows must say so rather than answer with an
    /// empty store, which an operator would read as "position 0 of a
    /// fresh sequence" and continue from — silently, with every shape
    /// closing. `&mut` because this operator reads its whole prefix and
    /// appends the current position in one call, and the store, not the
    /// operator, owns the rows either way.
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError>;
}

/// Why a continuation provider could not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuationError {
    /// This provider holds no recurrent buffers at all. Not a defect in
    /// the plan — a statement about the provider, which is why it names
    /// itself: a hybrid model reaching a KV-only serving cache must fail
    /// closed and say which side is missing.
    RecurrentUnsupported {
        provider: &'static str,
        layer: usize,
    },
    /// The provider holds recurrent buffers, but not for this layer —
    /// a softmax layer was asked for a recurrence, or the reverse.
    NotRecurrent {
        provider: &'static str,
        layer: usize,
    },
    /// This provider holds no per-position latent cache at all. Its own
    /// variant rather than [`Self::RecurrentUnsupported`]: a latent cache
    /// grows with the prefix and a recurrence does not, so a provider
    /// missing one is not missing the other, and a serving cache told
    /// "no recurrent state" when the real gap is MLA's latent rows would
    /// be sent looking in the wrong place.
    LatentUnsupported {
        provider: &'static str,
        layer: usize,
    },
    /// The provider holds latent rows, but not for this layer.
    NotLatent {
        provider: &'static str,
        layer: usize,
    },
    /// The provider can hold this region but not in this layer's shape —
    /// a representation with its own geometry constraints (a codec whose
    /// block must be a power of two, say) refusing at announcement rather
    /// than failing on the first row.
    GeometryUnsupported {
        provider: &'static str,
        layer: usize,
        reason: String,
    },
    /// This provider cannot rewind at all: it does not implement
    /// [`ContinuationProvider::truncate`].
    TruncateUnsupported { position: usize },
    /// The provider can rewind, but not this rewind. Nothing was changed.
    TruncateRefused {
        provider: &'static str,
        position: usize,
        reason: String,
    },
}

impl std::fmt::Display for ContinuationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RecurrentUnsupported { provider, layer } => write!(
                f,
                "continuation provider `{provider}` holds no recurrent state, which layer \
                 {layer} requires; refusing rather than running the layer stateless"
            ),
            Self::NotRecurrent { provider, layer } => write!(
                f,
                "layer {layer} is not a recurrent layer in `{provider}`'s geometry; asking \
                 it for recurrent state is a dispatch bug"
            ),
            Self::LatentUnsupported { provider, layer } => write!(
                f,
                "continuation provider `{provider}` holds no per-position latent cache, \
                 which layer {layer} requires; refusing rather than continuing from an \
                 empty prefix"
            ),
            Self::NotLatent { provider, layer } => write!(
                f,
                "layer {layer} keeps no latent cache in `{provider}`'s geometry; asking \
                 it for latent rows is a dispatch bug"
            ),
            Self::GeometryUnsupported {
                provider,
                layer,
                reason,
            } => write!(
                f,
                "continuation provider `{provider}` cannot hold layer {layer}: {reason}; \
                 refusing before any row is stored"
            ),
            Self::TruncateUnsupported { position } => write!(
                f,
                "this continuation provider cannot rewind to position {position}: it does \
                 not support rollback; refusing with its state unchanged"
            ),
            Self::TruncateRefused {
                provider,
                position,
                reason,
            } => write!(
                f,
                "continuation provider `{provider}` cannot rewind to position {position}: \
                 {reason}; refusing with its state unchanged"
            ),
        }
    }
}

impl From<ContinuationError> for crate::error::VindexError {
    fn from(value: ContinuationError) -> Self {
        Self::Parse(value.to_string())
    }
}

/// The seam's previous name.
///
/// `KvState` described the runtime model when every layer kept rows. It no
/// longer does: Qwen3.8 keeps a delta matrix and a convolution history on
/// 48 of its 64 layers and no rows at all. The alias exists so the serving
/// stack and its KV-1 bit-identity gate keep compiling through QW-3.6b;
/// it goes away at STATE-CONSOLIDATE, when `ContinuationState` becomes the
/// authoritative seam and KV becomes its projection.
pub use ContinuationProvider as KvState;

/// [`RowKvState`]'s family ([`RowKvState::identity`]):
/// plain per-layer row vectors. Revision 1 is the store as it serves rows
/// today; it moves when the same appended history would be served back
/// differently, never for a faster store serving the same bits.
pub const ROW_IDENTITY_FAMILY: &str = "row";
pub const ROW_IDENTITY_REVISION: u32 = 1;

/// The default provider: plain per-layer row vectors — exactly the
/// state [`DecodeSession`](super::decode::DecodeSession) used to own
/// privately, now behind the seam. The decode-vs-batch parity gates
/// pin that this indirection changed nothing.
#[derive(Clone, Default)]
pub struct RowKvState {
    layers: Vec<LayerRows>,
    /// Durable recurrent buffers, one slot per layer, `None` on layers
    /// that keep rows instead. Allocated by
    /// [`prepare_continuation`](ContinuationProvider::prepare_continuation)
    /// from the plan's geometry — never lazily on first use, because a
    /// buffer conjured mid-traversal would start from zeros in the middle
    /// of a sequence and look like a working continuation.
    recurrent: Vec<Option<RecurrentState>>,
    /// Per-position latent rows, one slot per layer, `None` on layers
    /// that keep something else. Allocated with the rest of the
    /// continuation, for the same reason: a cache conjured mid-traversal
    /// would look like the start of a sequence.
    latent: Vec<Option<LatentKvRows>>,
    position: usize,
}

#[derive(Clone, Default)]
struct LayerRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl RowKvState {
    /// This provider's continuation identity (C1 of
    /// CONTINUATION-PLUGIN-1) — `row/v1`, read from this module's
    /// constants.
    pub fn identity() -> ContinuationIdentity {
        ContinuationIdentity::new(ROW_IDENTITY_FAMILY, ROW_IDENTITY_REVISION)
    }
}

/// Builds [`RowKvState`] for the continuation registry (C2 of
/// CONTINUATION-PLUGIN-1): every region the plan vocabulary describes, no
/// options.
#[derive(Debug, Clone, Copy, Default)]
pub struct RowFactory;

impl ContinuationFactory for RowFactory {
    fn identity(&self) -> ContinuationIdentity {
        RowKvState::identity()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &ContinuationRegion::ALL
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        Box::new(RowKvState::default())
    }
}

impl KvState for RowKvState {
    /// Exact at any position: this provider keeps every row of every layer
    /// and never evicts, and a sliding window is applied when attention
    /// reads, not when rows are stored. Dropping the rows at and after
    /// `position` therefore leaves exactly the rows a state that stopped at
    /// `position` holds. Recurrent buffers and latent rows are refused: a
    /// recurrence's state at an earlier position is not recoverable from
    /// its state at a later one.
    fn truncate(&mut self, position: usize) -> Result<(), ContinuationError> {
        let refuse = |reason: String| ContinuationError::TruncateRefused {
            provider: ROW_IDENTITY_FAMILY,
            position,
            reason,
        };
        // Everything is checked before anything is changed.
        if position > self.position {
            return Err(refuse(format!(
                "the state continues from position {}; a rewind cannot move forward",
                self.position
            )));
        }
        if let Some(layer) = self.recurrent.iter().position(Option::is_some) {
            return Err(refuse(format!(
                "layer {layer} keeps recurrent buffers, which cannot be rewound"
            )));
        }
        if let Some(layer) = self.latent.iter().position(Option::is_some) {
            return Err(refuse(format!(
                "layer {layer} keeps a latent cache, whose rewind is not implemented"
            )));
        }
        for (layer, rows) in self.layers.iter().enumerate() {
            if rows.keys.len() != rows.values.len() {
                return Err(refuse(format!(
                    "layer {layer} holds {} keys and {} values",
                    rows.keys.len(),
                    rows.values.len()
                )));
            }
            // A layer that keeps rows holds one per position, so it must
            // reach the target. A layer holding none keeps no rows at all.
            if !rows.keys.is_empty() && rows.keys.len() < position {
                return Err(refuse(format!(
                    "layer {layer} holds {} rows, fewer than the target position",
                    rows.keys.len()
                )));
            }
        }
        for rows in &mut self.layers {
            rows.keys.truncate(position);
            rows.values.truncate(position);
        }
        self.position = position;
        Ok(())
    }

    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            self.layers = layers.iter().map(|_| LayerRows::default()).collect();
        } else {
            // A held state is being resumed; it must be state for a
            // program of this shape. Fail loud — silently reshaping
            // continuation state would be a wrong-conversation bug.
            assert_eq!(
                self.layers.len(),
                layers.len(),
                "resumed KV state holds {} layers but the plan declares {}",
                self.layers.len(),
                layers.len()
            );
        }
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let rows = &mut self.layers[layer];
        rows.keys.push(key);
        rows.values.push(value);
    }

    fn rows(&self, layer: usize) -> super::kv_view::KvView<'_> {
        let rows = &self.layers[layer];
        super::kv_view::KvView::over_rows(&rows.keys, &rows.values)
    }
    fn prepare_layer(&mut self, _layer: usize) {}

    fn position(&self) -> usize {
        self.position
    }

    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        // KV rows keep their own announcement contract (an announcement,
        // not a reset), so the recurrent side follows the same rule: a
        // resumed state keeps its buffers and only their SHAPE is
        // re-checked.
        // Sized by the FULL layer count, not the KV subset: this
        // provider is indexed by absolute layer index, and a stack whose
        // layer 3 is the only softmax one would otherwise write its rows
        // at slot 0.
        if self.layers.is_empty() {
            self.layers = layers.iter().map(|_| LayerRows::default()).collect();
        } else {
            assert_eq!(
                self.layers.len(),
                layers.len(),
                "resumed KV state holds {} layers but the plan declares {}",
                self.layers.len(),
                layers.len()
            );
        }
        if self.recurrent.is_empty() {
            self.recurrent = layers
                .iter()
                .map(|g| g.recurrent().map(RecurrentState::zeros))
                .collect();
        } else {
            assert_eq!(
                self.recurrent.len(),
                layers.len(),
                "resumed continuation state holds {} layers but the plan declares {}",
                self.recurrent.len(),
                layers.len()
            );
        }
        if self.latent.is_empty() {
            self.latent = layers
                .iter()
                .map(|g| g.latent_kv().map(|_| LatentKvRows::default()))
                .collect();
        } else {
            assert_eq!(
                self.latent.len(),
                layers.len(),
                "resumed continuation state holds {} layers but the plan declares {}",
                self.latent.len(),
                layers.len()
            );
        }
        Ok(())
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.recurrent
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or(ContinuationError::NotRecurrent {
                provider: "RowKvState",
                layer,
            })
    }

    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.latent
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or(ContinuationError::NotLatent {
                provider: "RowKvState",
                layer,
            })
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }
}
