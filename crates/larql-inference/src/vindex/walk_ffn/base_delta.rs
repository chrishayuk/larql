//! Base+delta patched-FFN execution — exact override handling on top
//! of the fast dense path (2026-07-30 review, item 16).
//!
//! The gated FFN is feature-separable, so for a layer with patched
//! slots `P`:
//!
//! ```text
//! y_patched = y_base + Σ_{i∈P} [aᵢ_new(x)·dᵢ_new − aᵢ_old(x)·dᵢ_old]
//! aᵢ(x) = φ(gᵢ·x)·(uᵢ·x)        φ = SiLU or GELU-tanh per arch
//! ```
//!
//! is *exact* — override cost drops from O(N) forced-sparse to O(|P|)
//! extra dots on the dense base. Inserts into slots the base has no
//! row for are pure additions; tombstones (DELETE) pure subtractions.
//!
//! Exactness conditions (all enforced by `base_delta_preconditions`,
//! which declines — returns `Err` naming the failed condition —
//! whenever one fails, routing the layer to the pre-existing
//! sparse/fallback override arm; the planner reports the same string
//! as the rung's skip reason):
//!
//! 1. The OLD contribution is computed from the SAME quantised bytes
//!    the dense base consumed, via the unified `ffn_row_dot` /
//!    `ffn_row_scaled_add` dispatch — never an f32-recomputed twin.
//!    Subtracting a mismatched old term would inject quantisation
//!    error into every patched slot. Hence the storage-coherence
//!    checks: exactly one of {Q4K interleaved, f32 interleaved} may
//!    be present, with no FP4/Q4_0/full-mmap/feature-major-f32
//!    storage that would divert either the base ladder or the row
//!    dispatch onto different bytes.
//! 2. Old down rows come from the feature-major Q4K down sidecar
//!    (`down_features_q4k`, required for the Q4K shape — the
//!    transposed interleaved down is not row-gatherable) or from the
//!    native f32 interleaved down.
//! 3. The dense base is produced by `forward_unpatched_whole_layer` —
//!    the very ladder body (steps 4–9) the unpatched routing runs, so
//!    base numerics are identical to the unpatched model.

use ndarray::Array2;

use super::observe::Observe;
use super::WalkFfn;
use crate::ffn::FfnActivations;
use larql_vindex::{OverrideSlot, FFN_DOWN, FFN_GATE, FFN_UP};

// Precondition-failure reasons — `Err` payloads of
// `base_delta_preconditions`, surfaced verbatim by the planner as the
// `override:base_delta` rung's skip reason (item 18).
pub(super) const BD_NOT_GATED: &str = "base+delta needs a gated FFN (aᵢ = φ(g·x)·(u·x))";
pub(super) const BD_SPARSE_CONFIG: &str =
    "explicit sparse K — the ladder would run the walk, so there is no dense base to correct";
pub(super) const BD_BAD_SHAPE: &str = "input shape unsupported (empty seq or hidden-size mismatch)";
pub(super) const BD_DIVERGENT_STORAGE: &str =
    "FP4/Q4_0/full-mmap/feature-major-down storage would divert base and row dispatch onto different bytes";
pub(super) const BD_NO_COHERENT_BASE: &str =
    "need exactly one of {Q4K, f32} interleaved storage for a byte-coherent dense base";
pub(super) const BD_UP_SIDECAR: &str =
    "f32 up sidecar would hijack the up row dot away from the Q4K base bytes";
pub(super) const BD_NO_DOWN_SIDECAR: &str =
    "old-down subtraction needs the feature-major Q4K down sidecar";
pub(super) const BD_NO_SLOT_ENUMERATION: &str =
    "index cannot enumerate override slots (point-query only)";

impl<'a> WalkFfn<'a> {
    /// Check every base+delta precondition for `layer` and return the
    /// enumerated patch slots when the path can run exactly. `Err`
    /// names the first failed condition — see the module doc for the
    /// condition list; the planner reports the string as the rung's
    /// skip reason.
    pub(super) fn base_delta_preconditions(
        &self,
        layer: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Vec<OverrideSlot>, &'static str> {
        // Gated FFNs only: the aᵢ(x) = φ(g·x)(u·x) decomposition is
        // the gated form. (Activation dispatch below mirrors the
        // sparse walk / kquant-native paths exactly: GeluTanh|Gelu →
        // gelu_tanh, everything else → SiLU.)
        if self.weights.arch.ffn_type() != larql_models::FfnType::Gated {
            return Err(BD_NOT_GATED);
        }
        // An explicit sparse K means the unpatched ladder would route
        // the (override-aware) walk, not a whole-layer dense path —
        // there is no dense base to correct.
        if self.config.is_sparse(layer) {
            return Err(BD_SPARSE_CONFIG);
        }
        if seq_len == 0 || hidden != self.weights.hidden_size {
            return Err(BD_BAD_SHAPE);
        }
        // Storage coherence (exactness conditions 1–2). FP4 has no
        // whole-layer dense path; Q4_0 has no `ffn_row_dot` arm;
        // full-mmap / feature-major f32 / `up_features` divert either
        // the base ladder or the row dispatch onto different bytes
        // than the other would use.
        if self.index.has_fp4_storage()
            || self.index.has_interleaved_q4()
            || self.index.has_full_mmap_ffn()
            || self.index.has_down_features()
        {
            return Err(BD_DIVERGENT_STORAGE);
        }
        let kquant = self.index.has_interleaved_kquant();
        let native = self.index.has_interleaved();
        if kquant == native {
            // Both → the ladder's base (Q4K, step 4) and the row
            // dispatch (native f32 first) would read different bytes.
            // Neither → no whole-layer base at all.
            return Err(BD_NO_COHERENT_BASE);
        }
        if kquant {
            // A feature-major f32 up sidecar would hijack the up row
            // dot away from the Q4K bytes the base consumed.
            if self.index.up_layer_matrix(layer).is_some() {
                return Err(BD_UP_SIDECAR);
            }
            // Old-down subtraction needs the feature-major sidecar
            // (exactness condition 2).
            if self.index.down_features_q4k_layer_data(layer).is_none() {
                return Err(BD_NO_DOWN_SIDECAR);
            }
        }
        // Enumeration must be exhaustive; point-query-only indexes
        // (`override_slots_at` default `None`) decline.
        self.index
            .override_slots_at(layer)
            .ok_or(BD_NO_SLOT_ENUMERATION)
    }

    /// Exact patched forward: dense base + per-slot corrections.
    /// Returns `None` (decline) when preconditions fail or a slot's
    /// storage turns out inconsistent mid-computation — the caller
    /// then takes the pre-existing sparse/fallback override routes.
    ///
    /// Observation: the base's activation matrix is intrinsic to the
    /// dense ladder, so `Record` mode reports it (with the patched
    /// slots' post-patch activations written in) and `Skip` drops it.
    pub(super) fn walk_ffn_base_delta(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> Option<(Array2<f32>, Option<FfnActivations>)> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        let slots = self.base_delta_preconditions(layer, seq_len, hidden).ok()?;

        // Dense base — the SAME whole-layer ladder body the unpatched
        // routing runs (exactness condition 3). On a patched index the
        // whole-layer paths read only base bytes, so this is the
        // pre-patch output the delta corrects.
        let (mut out, base_activation) = self.forward_unpatched_whole_layer(layer, x)?;
        let mut activation = observe.recording().then_some(base_activation);

        let use_gelu = self.weights.arch.gate_up_is_gelu_tanh();
        let phi = |g: f32| {
            if use_gelu {
                crate::ffn::gelu_tanh(g)
            } else {
                g * crate::ffn::sigmoid(g)
            }
        };

        let mut delta = vec![0.0f32; hidden];
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_owned = x_row.to_vec();
                &x_owned
            };
            delta.fill(0.0);

            for slot in &slots {
                let feat = slot.feature;

                // ── OLD term — same quantised bytes as the dense base.
                let g_old = self.index.ffn_row_dot(layer, FFN_GATE, feat, x_slice);
                let u_old = self.index.ffn_row_dot(layer, FFN_UP, feat, x_slice);
                let a_old = match (g_old, u_old) {
                    (Some(g), Some(u)) => Some(phi(g) * u),
                    // No base row at all: a pure insert — nothing to
                    // subtract, the base never included this slot.
                    (None, None) => None,
                    // Half-covered slot: inconsistent storage. Decline
                    // rather than subtract a wrong old term.
                    _ => return None,
                };
                if let Some(a_old) = a_old {
                    if a_old != 0.0
                        && !self
                            .index
                            .ffn_row_scaled_add(layer, FFN_DOWN, feat, -a_old, &mut delta)
                    {
                        return None;
                    }
                }

                // ── NEW term — override vectors where present, the
                //    base-byte dots otherwise. Tombstoned slots
                //    contribute nothing post-patch.
                let a_new = if slot.tombstoned {
                    0.0
                } else {
                    // φ(0) = 0 for both SiLU and GELU-tanh, so a slot
                    // with no gate anywhere correctly activates to 0.
                    let g_new = match self.index.gate_override(layer, feat) {
                        Some(gv) if gv.len() == hidden => ndarray::ArrayView1::from(gv).dot(&x_row),
                        Some(_) => return None, // wrong-width override
                        None => g_old.unwrap_or(0.0),
                    };
                    let u_new = match self.index.up_override(layer, feat) {
                        Some(uv) if uv.len() == hidden => ndarray::ArrayView1::from(uv).dot(&x_row),
                        Some(_) => return None,
                        None => u_old.unwrap_or(0.0),
                    };
                    phi(g_new) * u_new
                };

                match self.index.down_override(layer, feat) {
                    Some(dv) if dv.len() == hidden => {
                        if a_new != 0.0 {
                            for (d, v) in delta.iter_mut().zip(dv) {
                                *d += a_new * v;
                            }
                        }
                    }
                    Some(_) => return None,
                    None => {
                        if a_new != 0.0 {
                            // No down override → the new term projects
                            // through the base down row, which must
                            // exist (a_old proved it does).
                            if a_old.is_none()
                                || !self
                                    .index
                                    .ffn_row_scaled_add(layer, FFN_DOWN, feat, a_new, &mut delta)
                            {
                                return None;
                            }
                        }
                    }
                }

                // Post-patch activation for the slot (0 for tombstones
                // and inserts beyond the base activation width).
                if let Some(act) = activation.as_mut() {
                    if feat < act.shape()[1] {
                        act[[s, feat]] = a_new;
                    }
                }
            }

            for (o, d) in out.row_mut(s).iter_mut().zip(&delta) {
                *o += *d;
            }
        }

        self.trace_path(layer, "override:base_delta");
        Some((out, activation.map(FfnActivations::Dense)))
    }
}
