//! The layer traversal loop.

use super::super::ComponentOpPlan;
use crate::error::VindexError;
use backend::{NormCall, PlanBackend};
use hyper_connection::{Bundle, Mutation};
use kv::KvState;
use observe::CarrierTransition;
use prepared::PreparedOperands;

#[allow(unused_imports)]
use super::*;

/// The one traversal in this module (see [`execute_plan_streaming`]'s
/// doc). `kv` switches the attention realisation: `None` runs the
/// backend's batched attention; `Some` runs the decode step's
/// per-position arithmetic into the provider — same numbers (the
/// decode-vs-batch parity gates are the guarantee), plus the rows.
/// `kv` and `resume` DO combine, and must: a resumed stack with
/// recurrent or latent layers cannot run without a provider. The provider
/// then holds nothing for the layers the resume skipped, which is why a
/// one-shot entry consumes it rather than lending it (RESIDUAL-BUS-2).
#[allow(clippy::too_many_arguments)]
pub(super) fn traverse<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mut kv: Option<&mut K>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    let embedding = plan.embedding.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "component `{}` has no embedding op — external hidden-state input is a later rung",
            plan.component
        ))
    })?;
    let hidden = embedding.table.shape[1];
    let topology = ops.hyper_connection();
    // The declared block period, `None` on every other topology — the
    // ONE fact the attention-residual schedule needs, and the same fact
    // the decode traversal reads.
    let block_size = ops.attention_residual_block_size();

    // **Refuse before any output.** A recurrence needs durable buffers,
    // and discovering at layer 63 that nobody can hold them would mean
    // every earlier layer had already been emitted — a caller left
    // holding 16 of 64 layers cannot tell that from a finished model.
    // QW-1 put this refusal up front for exactly that reason; the
    // question it asks has changed (from "can this run at all" to "can
    // this provider hold the state"), its position must not.
    //
    // Scoped to the layers this slice will actually execute. For `Full`
    // that is every layer and the check is unchanged; a reduced-depth
    // draft must not be refused — or charged state — for a recurrence in
    // a layer it never runs.
    //
    // Driven by the CONTINUATION GEOMETRY, not by "is this layer
    // softmax": the question is which region a layer needs, and there
    // are now three answers. An earlier form asked every non-softmax
    // layer for recurrent buffers, which was right while a recurrence
    // was the only alternative to rows and became wrong the moment MLA
    // executed — it keeps a per-position latent cache and no recurrence
    // at all, so the pre-flight would have refused a state the provider
    // was holding perfectly well.
    let executed = ops.first_layer()..ops.first_layer() + ops.layers().len();
    let regions = continuation::plan_continuation_geometry(plan).map_err(|e| {
        VindexError::Parse(format!(
            "component `{}` declares continuation state this build cannot size: {e}",
            plan.component
        ))
    })?;
    for (offset, layer) in plan.layers.iter().enumerate() {
        if !executed.contains(&offset) {
            continue;
        }
        let region = &regions[offset];
        if matches!(
            region,
            continuation::LayerContinuationGeometry::Kv(_)
                | continuation::LayerContinuationGeometry::Stateless
        ) {
            continue;
        }
        let Some(provider) = kv.as_mut() else {
            return Err(VindexError::Parse(format!(
                "layer {} carries `{}`, which keeps durable continuation state, and this \
                 traversal was given no provider to hold it",
                layer.layer,
                layer.attention.declared_name(),
            )));
        };
        let named = |e: kv::ContinuationError| {
            VindexError::Parse(format!(
                "layer {} carries `{}`: {e}",
                layer.layer,
                layer.attention.declared_name(),
            ))
        };
        match region {
            continuation::LayerContinuationGeometry::Recurrent(_)
            | continuation::LayerContinuationGeometry::KvAndRecurrent { .. } => {
                provider.recurrent_state(offset).map_err(named)?;
            }
            continuation::LayerContinuationGeometry::LatentKv(_) => {
                provider.latent_state(offset).map_err(named)?;
            }
            continuation::LayerContinuationGeometry::Kv(_)
            | continuation::LayerContinuationGeometry::Stateless => {}
        }
    }

    // Every mixer's dense operands against this backend's projector, for
    // the whole plan, before any layer runs (refuse before commit).
    dense_rows::refuse_unconsumable_plan(plan, ops, backend.dense_projector())?;

    // Where this batch sits in the sequence: a provider that already holds
    // state continues from its position, and every transition is named at
    // `base + row`, as decode names it (RESIDUAL-BUS-2 A2). With no
    // provider the batch is the whole sequence and starts at zero.
    let base = kv.as_ref().map_or(0, |state| state.position());
    let (start_layer, mut h) = match resume {
        Some(point) => {
            if point.next_layer > plan.layers.len() {
                return Err(VindexError::Parse(format!(
                    "resume point at layer {} is past the plan's {} layers",
                    point.next_layer,
                    plan.layers.len()
                )));
            }
            if point.hidden.positions() != tokens.len() {
                return Err(VindexError::Parse(format!(
                    "resume state carries {} positions but the fixture has {} tokens",
                    point.hidden.positions(),
                    tokens.len()
                )));
            }
            // The plane's kind must be the component's residual: a row
            // plane on a hyper-connected component would run the bundle
            // as one stream, and a bundle plane on a single stream has
            // no reading at all.
            match (&point.hidden, topology) {
                // `topology` names hyper-connections only, so an
                // attention-residual component reaches this arm too. Its
                // carrier is a history, and rows here would run every
                // site as a single stream: no reduction, no boundary, no
                // exit reduction. A plausible plane, and the wrong model.
                (Plane::Rows(_), None) if ops.carries_attention_residual() => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` carries an attention-residual history; a resume point \
                         of rows cannot enter it, and a history resume point is not supported",
                        plan.component
                    )));
                }
                (Plane::Rows(rows), None) => {
                    if rows.iter().any(|row| row.len() != hidden) {
                        return Err(VindexError::Parse(format!(
                            "resume state rows do not match the plan's hidden size {hidden}"
                        )));
                    }
                }
                (Plane::Bundles(bundles), Some(hc)) => {
                    if bundles
                        .iter()
                        .any(|b| b.streams() != hc.streams || b.hidden() != hidden)
                    {
                        return Err(VindexError::Parse(format!(
                            "resume bundles do not match the component's {} streams x {hidden}",
                            hc.streams
                        )));
                    }
                }
                (Plane::Rows(_), Some(hc)) => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` declares {} residual streams; a resume point for it \
                         carries bundles, not rows",
                        plan.component, hc.streams
                    )));
                }
                // A history resume point is REFUSED, not supported.
                // Carrying a typed state through a traversal and being
                // able to reconstruct it from an external representation
                // are different capabilities: nothing reads a serialised
                // prefix-plus-snapshots state back, and inventing a
                // format for one nothing consumes is the addressability
                // -without-execution mistake in a new place. It returns
                // when a real reader is built.
                (Plane::Histories(histories), _) => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` declares the attention-residual topology; a resume \
                         point carrying {} residual histories is not supported — nothing \
                         reads a snapshot history back from an external representation, and \
                         this build will not invent a format for one",
                        plan.component,
                        histories.len()
                    )));
                }
                (Plane::Bundles(_), None) => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` declares a single residual stream; a resume point for \
                         it carries rows, not bundles",
                        plan.component
                    )));
                }
            }
            each_position(
                sink,
                point.next_layer,
                base,
                point.hidden.form(),
                point.hidden.positions(),
                CarrierTransition::Enter,
            )?;
            (point.next_layer, point.hidden)
        }
        None => {
            let table = ops.embed_table().ok_or_else(|| {
                VindexError::Parse(
                    "this prepared image carries no embedding table — a layer-range slice \
                     consumes hidden states, not token ids"
                        .to_string(),
                )
            })?;
            let mut h: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| backend.embed(table, hidden, t, embedding.scale))
                .collect();
            // The judged embedding normalisation, when the plan carries
            // one. It is weightless — no operand, hence the empty weight
            // slice — and it runs *after* any embedding scale, matching
            // the upstream order in which the scale belongs to the table
            // and the norm to the lookup.
            if let Some(norm) = embedding.norm {
                for row in h.iter_mut() {
                    *row = backend.norm(NormCall {
                        kind: norm.kind,
                        x: row,
                        weight: &[],
                        weight_offset: 0.0,
                        eps: norm.eps,
                    });
                }
            }
            // The embedding enters a hyper-connected stack replicated
            // into every stream, per position — after its scale and
            // norm, which belong to the lookup.
            let h = match (topology, block_size) {
                (Some(hc), _) => Plane::Bundles(
                    h.iter()
                        .map(|row| Bundle::replicate(row, hc.streams))
                        .collect(),
                ),
                // Each position enters as its OWN first prefix with an
                // EMPTY history — nothing is replicated and nothing is
                // shared. Every later divergence between positions
                // begins here.
                (None, Some(_)) => Plane::Histories(
                    h.into_iter()
                        .map(attention_residual::History::new)
                        .collect(),
                ),
                (None, None) => Plane::Rows(h),
            };
            sink(PlaneEvent::Embedded(&h))?;
            each_position(
                sink,
                ops.first_layer(),
                base,
                h.form(),
                h.positions(),
                CarrierTransition::Enter,
            )?;
            (0, h)
        }
    };

    let first = ops.first_layer();
    for (offset, prepared_layer) in ops.layers().iter().enumerate() {
        let index = first + offset;
        if index < start_layer {
            continue;
        }
        let capture = kv.as_mut().map(|state| (&mut **state, index));
        let trace = execute_layer(
            &plan.layers[index],
            prepared_layer,
            &mut h,
            hidden,
            backend,
            capture,
            topology,
            block_size,
            index,
            base,
            sink,
            mutation,
        )?;
        sink(PlaneEvent::Layer { index, trace })?;
    }

    // ── The exit ──
    //
    // A bundle leaves the stack through the head's OWN reduction (a
    // different operation from a site's — no Sinkhorn) when the image
    // carries one, and a whole-stack image of a hyper-connected
    // component always does (preparation refuses otherwise). A
    // layer-range image has no exit: the bundle after its last layer IS
    // its output, and it produces no logits.
    let empty = || VindexError::Parse("cannot execute over an empty token sequence".to_string());
    let (exit, logits) = match h {
        Plane::Rows(rows) => {
            let last = rows.last().ok_or_else(empty)?;
            let final_hidden = match ops.final_norm() {
                Some(norm) => norm.apply(backend, last),
                None => last.clone(),
            };
            let logits = ops.head_over_normed(backend, &final_hidden)?;
            (FinalState::Hidden(final_hidden), logits)
        }
        // The attention-residual exit: the same reduction a site runs,
        // once per position, over that position's WHOLE history plus its
        // prefix, before the final norm. Required by the declaration —
        // preparation refuses a whole-stack image without the pair — so
        // the `None` arm is the layer-range image, whose output IS the
        // history it hands on.
        Plane::Histories(mut histories) => {
            let last = histories.pop().ok_or_else(empty)?;
            match ops.attention_residual_exit() {
                Some(exit) if mutation != Mutation::AttnResExitSkipped => {
                    let pair = match mutation {
                        // The control that makes "the SHIPPED pair" a
                        // claim: reduce with a layer's pair instead.
                        Mutation::AttnResExitUsesALayerPair => ops.layers()[0]
                            .attention_residual
                            .as_ref()
                            .map(|sites| sites.ffn.pair())
                            .unwrap_or_else(|| exit.pair()),
                        _ => exit.pair(),
                    };
                    let reduced =
                        attention_residual::reduce(&last, pair, exit.norm_eps(), mutation)?.mixed;
                    let final_hidden = match ops.final_norm() {
                        Some(norm) => norm.apply(backend, &reduced),
                        None => reduced,
                    };
                    let logits = ops.head_over_normed(backend, &final_hidden)?;
                    (FinalState::Hidden(final_hidden), logits)
                }
                Some(_) => {
                    let prefix = last.clone().into_prefix()?;
                    let final_hidden = match ops.final_norm() {
                        Some(norm) => norm.apply(backend, &prefix),
                        None => prefix,
                    };
                    let logits = ops.head_over_normed(backend, &final_hidden)?;
                    (FinalState::Hidden(final_hidden), logits)
                }
                None => {
                    if ops.final_norm().is_some() || ops.output().is_some() {
                        return Err(VindexError::Parse(
                            "an attention-residual history reached a whole-stack exit with no \
                             exit reduction; preparation should have refused the image"
                                .to_string(),
                        ));
                    }
                    (FinalState::History(last), None)
                }
            }
        }
        Plane::Bundles(mut bundles) => {
            let last = bundles.pop().ok_or_else(empty)?;
            match (ops.hyper_connection_head(), topology) {
                (Some(head), Some(hc)) => {
                    let reduced = hyper_connection::head_reduce(
                        last.as_flat(),
                        last.streams(),
                        hidden,
                        &head.weights(),
                        head.norm_eps(),
                        hc.sinkhorn_eps,
                    );
                    let final_hidden = match ops.final_norm() {
                        Some(norm) => norm.apply(backend, &reduced),
                        None => reduced,
                    };
                    let logits = ops.head_over_normed(backend, &final_hidden)?;
                    (FinalState::Hidden(final_hidden), logits)
                }
                _ => {
                    if ops.final_norm().is_some() || ops.output().is_some() {
                        return Err(VindexError::Parse(
                            "a hyper-connected bundle reached a whole-stack exit with no head \
                             reduction; preparation should have refused the image"
                                .to_string(),
                        ));
                    }
                    (FinalState::Bundle(last), None)
                }
            }
        }
    };
    Ok(FinalOutput { exit, logits })
}

/// Scale a sublayer's own output before its residual add, when the plan
/// declares a residual-scale operation (`LayerPlan::residual_scale`).
/// `None` leaves `delta` untouched — absence is not an identity multiply,
/// the same discipline every other optional op in the surface follows.
/// Backend-agnostic by design: every `PlanBackend` shares this one step
/// rather than each reimplementing the same multiply. `pub(in super::super)`: the
/// stateful single-token driver in [`decode`] applies the same op at its
/// own residual-add sites, on the incremental decode path this batch path
/// does not cover.
pub(in super::super) fn scale_residual_delta(scale: Option<f32>, delta: &mut [f32]) {
    if let Some(scale) = scale {
        for x in delta.iter_mut() {
            *x *= scale;
        }
    }
}
