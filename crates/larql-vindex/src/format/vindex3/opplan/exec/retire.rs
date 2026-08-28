//! A continuation state that can retire part of its own past.
//!
//! This is the boundary-context mechanism's storage half (STATE-2): the
//! caller decides that a span of already-consumed positions is no longer
//! part of the continuation — typically everything in a context chunk
//! *behind* its boundary — and every later attention step excludes those
//! rows via [`ContinuationProvider::retired_spans`]. The rows stay
//! physically present and position-aligned, per the store contract; what
//! changes is the continuation the state *declares*, which is exactly
//! the separation the STATE-1 representation seam names: the semantic
//! state and its physical holding are different facts.
//!
//! Retirement here is a policy the CALLER composes — the provider does
//! not decide what to retire, and the executor only honours (or
//! refuses) the declaration. That keeps the division of labour the kv
//! module states: the store holds, the span logic excludes, and policy
//! lives outside the executor.

use std::ops::Range;

use super::continuation::{LayerContinuationGeometry, RecurrentState};
use super::kv::{ContinuationError, ContinuationProvider, LayerKvGeometry, RowKvState};

/// [`RowKvState`] plus a caller-declared set of retired position spans.
///
/// Everything a plain row state does is delegated unchanged; the only
/// added behaviour is [`retire`](Self::retire) and the non-`None`
/// [`retired_spans`](ContinuationProvider::retired_spans) answer. With
/// nothing retired the two providers are indistinguishable — the gate
/// tests pin that as bit-identity, not as intent.
#[derive(Clone, Default)]
pub struct RetiringKvState {
    inner: RowKvState,
    retired: Vec<Range<usize>>,
}

impl RetiringKvState {
    /// Adopt an already-populated state — a prefilled prefix about to
    /// have part of its past retired.
    pub fn from_state(inner: RowKvState) -> Self {
        Self {
            inner,
            retired: Vec::new(),
        }
    }

    /// Declare that `span`'s positions are no longer part of the
    /// continuation.
    ///
    /// Only the already-consumed past can be retired — the span must end
    /// at or before the current logical position — and a span must be
    /// non-empty and disjoint from every span already retired: retiring
    /// a position twice is a bookkeeping error worth naming, not a
    /// no-op.
    pub fn retire(&mut self, span: Range<usize>) -> Result<(), String> {
        if span.start >= span.end {
            return Err(format!("span {}..{} is empty", span.start, span.end));
        }
        if span.end > self.inner.position() {
            return Err(format!(
                "span {}..{} reaches beyond position {}; only the consumed past can be retired",
                span.start,
                span.end,
                self.inner.position()
            ));
        }
        if let Some(overlap) = self
            .retired
            .iter()
            .find(|r| span.start < r.end && r.start < span.end)
        {
            return Err(format!(
                "span {}..{} overlaps already-retired {}..{}",
                span.start, span.end, overlap.start, overlap.end
            ));
        }
        self.retired.push(span);
        Ok(())
    }

    /// Positions retired so far — the footprint a physical compaction
    /// would reclaim, counted rather than claimed.
    pub fn retired_positions(&self) -> usize {
        self.retired.iter().map(|r| r.end - r.start).sum()
    }
}

impl ContinuationProvider for RetiringKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.inner.prepare(layers);
    }

    // Forwarded rather than inherited: the trait default projects to the
    // KV subset and refuses hybrids, but the inner state genuinely holds
    // recurrent buffers, and losing that by taking the default would
    // make wrapping a hybrid state silently stricter than owning one.
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.inner.prepare_continuation(layers)
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.inner.append(layer, key, value);
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        self.inner.keys(layer)
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        self.inner.values(layer)
    }

    fn position(&self) -> usize {
        self.inner.position()
    }

    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position);
    }

    fn retired_spans(&self) -> Option<&[Range<usize>]> {
        if self.retired.is_empty() {
            None
        } else {
            Some(&self.retired)
        }
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.inner.recurrent_state(layer)
    }
}
