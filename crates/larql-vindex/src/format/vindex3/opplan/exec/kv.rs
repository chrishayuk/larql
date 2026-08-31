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
//!   post-rope) and returned position-ordered from position 0. A
//!   provider must hold **every** appended row — the span logic, not
//!   the store, excludes positions a window masks out (the cache may
//!   hold a position the span must exclude; dropping it is a policy
//!   the executor has not been taught to coordinate with).
//! - The `&[Vec<f32>]` row-slice shape mirrors
//!   [`AttentionStepCall`](super::backend::AttentionStepCall) and
//!   changes only with it; a flat or device-resident representation is
//!   a later rung tied to that backend contract.

use super::super::ComponentOpPlan;
use super::continuation::{LatentKvRows, LayerContinuationGeometry, RecurrentState};

/// One layer's continuation-state geometry, read from the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvGeometry {
    /// Row width of one position's K (and V): `num_kv_heads * head_dim`.
    pub kv_dim: usize,
    /// The layer's attention window in positions; `None` = full span.
    /// Informational for sizing/policy — see the module contract: the
    /// store still holds every row.
    pub window: Option<usize>,
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

    /// All K rows appended for `layer`, position-ordered from 0.
    fn keys(&self, layer: usize) -> &[Vec<f32>];

    /// All V rows appended for `layer`, position-ordered from 0.
    fn values(&self, layer: usize) -> &[Vec<f32>];

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
    /// of a batch prefill.
    fn set_position(&mut self, position: usize);

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

/// The default provider: plain per-layer row vectors — exactly the
/// state [`DecodeSession`](super::decode::DecodeSession) used to own
/// privately, now behind the seam. The decode-vs-batch parity gates
/// pin that this indirection changed nothing.
#[derive(Default)]
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

#[derive(Default)]
struct LayerRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvState for RowKvState {
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

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].keys
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].values
    }

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
