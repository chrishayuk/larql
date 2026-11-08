//! Executing one layer of a prepared plan.

use self::attention_residual::BoundaryPhase;
use super::super::LayerPlan;
use crate::error::VindexError;
use backend::PlanBackend;
use dense_rows::consuming_projector;
use hyper_connection::{Bundle, Mutation};
use kv::KvState;
use kv_view::KvView;
use larql_models::config::HyperConnection;
use observe::{CarrierForm, CarrierTransition, HcSite};
use prepared::{PreparedAttention, PreparedLayer};
use std::borrow::Cow;

#[allow(unused_imports)]
use super::*;

/// One decoder layer: norms and residuals exactly where the plan puts
/// them — placement is data, not code structure.
///
/// **Wave 19b.** The residual is a [`Plane`]. Each sublayer is a SITE:
/// on a bundle plane it enters through `hyper_connection::reduce`, per
/// position, and leaves through `hyper_connection::update`, per
/// position — while the ordinary operator in between runs exactly once
/// over `[positions, hidden]`, as it always has. On a row plane the site
/// is the identity in and the residual add out. `hidden` means the same
/// thing on both; the stream count is a different dimension and never
/// reaches the operator.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_layer<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    layer: &LayerPlan,
    prepared: &PreparedLayer,
    h: &mut Plane,
    hidden: usize,
    backend: &B,
    kv: Option<(&mut K, usize)>,
    topology: Option<HyperConnection>,
    // The declared block period, `None` on every other topology — the
    // ONE fact the attention-residual schedule needs.
    block_size: Option<usize>,
    index: usize,
    // The provider's position when the traversal began; batch row `i`
    // is absolute position `base + i` (RESIDUAL-BUS-2 A2).
    base: usize,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mutation: Mutation,
) -> Result<LayerTrace, VindexError> {
    // ── Attention site: enter ──
    //
    // The attention input is normalised here, once, and handed to the
    // backend — the judged gate reads the same vector, so producing it
    // in one place is what keeps the two from drifting apart. On a
    // bundle the norm reads the REDUCED vector, never the bundle.
    //
    // Position loops below run in parallel. Each position's arithmetic
    // is untouched and rows are disjoint, so the result is bit-identical
    // to the serial order — parallelism here is an execution strategy,
    // never a reassociation.
    // Under post-norm placement the sublayer reads the RAW residual; the
    // wrap norm applies to its output before the update. Same program as
    // the decode path, which must not be able to disagree with this one.
    // The entering prefix states, captured BEFORE the site reads
    // anything: they are what the boundary event snapshots, per
    // position, and capturing them later would snapshot whatever the
    // site had already done.
    let entering_prefixes: Option<Vec<Vec<f32>>> = match &*h {
        Plane::Histories(histories) => Some(
            histories
                .iter()
                .map(|history| history.prefix().unwrap_or_default().to_vec())
                .collect(),
        ),
        _ => None,
    };
    let boundary = matches!(&*h, Plane::Histories(_))
        && block_size.is_some_and(|size| attention_residual::is_block_boundary(index, size));
    if boundary {
        batch_boundary_event(
            h,
            BoundaryPhase::BeforeAttentionReduce,
            entering_prefixes.as_deref().unwrap_or_default(),
            &[],
            index,
            base,
            mutation,
            sink,
        )?;
    }
    // The attention site, and — on a history plane — the boundary event
    // that falls between its reduction and its branch. Written as two
    // paths rather than one because of a borrow that is not incidental:
    // on a ROW plane the site's inputs BORROW the plane (the branch reads
    // the residual itself, and cloning `[positions, hidden]` twice per
    // layer is megabytes of copying on a real model), while the event
    // needs the plane mutably. On a HISTORY plane the inputs are already
    // owned, so taking them out of the `Cow` ends the borrow and costs
    // nothing. The row path keeps its borrow and never has a boundary.
    let (branch_inputs, reductions, attn_res) = if boundary {
        let entry = enter_batch_site(
            h,
            BatchSite {
                layer: index,
                which: HcSite::Attention,
                hyper_connection: prepared.hyper_connection.as_ref().map(|hc| &hc.attention),
                attention_residual: prepared.attention_residual.as_ref().map(|a| &a.attention),
            },
            topology,
            layer.declared_norm_eps,
            mutation,
        )?;
        // Destructured in full, so the borrowing `Cow` field is CONSUMED
        // here rather than living on inside `entry` across the mutable
        // call below.
        let BatchSiteEntry {
            branch_inputs,
            reductions,
            attn_res,
        } = entry;
        let inputs = branch_inputs.into_owned();
        // The mixed vectors come from the OWNED reductions, which is
        // also the exact source: where the site did not reduce (layer
        // 0), there are none and the event falls back to the entering
        // prefix — which IS what the branch receives there.
        let mixed: Vec<Vec<f32>> = attn_res
            .as_ref()
            .map(|entry| entry.reductions.iter().map(|r| r.mixed.clone()).collect())
            .unwrap_or_default();
        // **Where the reference puts it** — between the attention site's
        // reduction and the attention branch.
        batch_boundary_event(
            h,
            BoundaryPhase::AfterAttentionReduce,
            entering_prefixes.as_deref().unwrap_or_default(),
            &mixed,
            index,
            base,
            mutation,
            sink,
        )?;
        (Cow::Owned(inputs), reductions, attn_res)
    } else {
        let entry = enter_batch_site(
            h,
            BatchSite {
                layer: index,
                which: HcSite::Attention,
                hyper_connection: prepared.hyper_connection.as_ref().map(|hc| &hc.attention),
                attention_residual: prepared.attention_residual.as_ref().map(|a| &a.attention),
            },
            topology,
            layer.declared_norm_eps,
            mutation,
        )?;
        (entry.branch_inputs, entry.reductions, entry.attn_res)
    };
    // The (c) controls feed the pre-attention norm a stream instead.
    let norm_source: Cow<'_, [Vec<f32>]> = match (mutation, &*h) {
        (Mutation::PreNormOnStreamZero, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(|x| x.stream(0).to_vec()).collect())
        }
        (Mutation::PreNormOnStreamMean, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(Bundle::stream_mean).collect())
        }
        _ => Cow::Borrowed(&branch_inputs),
    };
    let inputs: Vec<Vec<f32>> = match &prepared.pre_attention {
        Some(norm) => norm_source
            .par_iter()
            .map(|row| norm.apply(backend, row))
            .collect(),
        None => norm_source.into_owned(),
    };
    // V3-SERVE-2: the attention realisation and the K/V behaviour are
    // separate decisions. Wanting a populated provider does not mean
    // wanting per-position arithmetic — the batched pass computes the
    // same conditioned rows and now returns them, so it can populate the
    // provider from the one traversal it already performs.
    //
    // The exception is not a preference but an expressibility limit: a
    // batched pass conditions position `p` as the `p`-th token of the
    // sequence it is given, so it cannot serve a prefill that resumes
    // part-way through one. Extending a populated provider therefore
    // still steps.
    // Dispatch on the OPERATOR the prepared layer holds. There is no
    // `softmax_or_refuse` here any more: a recurrence is something this
    // traversal runs, not something it declines. Both arms produce the
    // attention block's output for every position, and the residual and
    // FFN below are shared — the operator changes what attention IS, not
    // what a decoder layer does around it.
    let attn_out = match &prepared.attention {
        PreparedAttention::GatedDelta(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            // Resolved BEFORE any arithmetic. A provider that cannot hold
            // these buffers must not see a half-updated layer — the same
            // refuse-before-commit contract QW-1 established for the plan.
            let weights = ops.weights()?;
            let projector = consuming_projector(backend.dense_projector(), &weights, layer.layer)?;
            let state = provider.recurrent_state(layer_index)?;
            gated_delta::layer_forward_with(
                &ops.op,
                &weights,
                &inputs,
                state,
                gated_delta::Mutation::None,
                projector,
            )
            .output
        }
        PreparedAttention::Mamba2(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            let weights = ops.weights()?;
            let projector = consuming_projector(backend.dense_projector(), &weights, layer.layer)?;
            let state = provider.recurrent_state(layer_index)?;
            mamba2::layer_forward_with(&ops.op, &weights, &inputs, state, projector).output
        }
        PreparedAttention::Kda(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            let weights = ops.weights()?;
            let projector = consuming_projector(backend.dense_projector(), &weights, layer.layer)?;
            let state = provider.recurrent_state(layer_index)?;
            // The batch is the sequence, flat: KDA's reference consumes
            // positions one after another because the recurrence IS
            // sequential, and this traversal differs from the decode
            // path only in how many positions it hands over.
            let flat: Vec<f32> = inputs.concat();
            let hidden_width = inputs.first().map_or(0, Vec::len);
            let planes = kda::layer_forward_with(
                &kda::BackendKdaProjections(projector),
                &flat,
                hidden_width,
                weights,
                ops.op.geometry(),
                state,
                kda::Mutation::None,
            );
            planes
                .output
                .chunks_exact(hidden_width.max(1))
                .map(<[f32]>::to_vec)
                .collect()
        }
        PreparedAttention::Mla(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} keeps a per-position latent cache, and this traversal was \
                     given no provider to hold it",
                    layer.layer
                )));
            };
            let weights = ops.weights()?;
            let geometry = ops.op.geometry();
            let projector = consuming_projector(backend.dense_projector(), &weights, layer.layer)?;
            let latent = provider.latent_state(layer_index)?;
            // Position by position, appending each one's latent before
            // reading the prefix back — the same call the decode path
            // makes, run `inputs.len()` times. A whole-sequence form
            // would need its own explicit causal mask; this one's
            // causality is the append-then-read order itself.
            inputs
                .iter()
                .map(|x| {
                    mla::mla_forward_with(
                        projector,
                        x,
                        x.len(),
                        weights,
                        geometry,
                        latent,
                        mla::Mutation::None,
                    )
                    .output
                })
                .collect()
        }
        PreparedAttention::ConvQkv(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} keeps a KV cache and a conv history, and this traversal \
                     was given no provider to hold either",
                    layer.layer
                )));
            };
            // Two regions, one provider, borrowed in phases: the rows
            // already persisted are copied out first (the reference
            // executor is deliberately literal — speed is a later
            // rung's problem), the conv history is advanced by the
            // forward, and the batch's new rows are appended after.
            let weights = ops.weights()?;
            let projector = consuming_projector(backend.dense_projector(), &weights, layer.layer)?;
            let base = provider.position();
            provider.prepare_layer(layer_index);
            let held = provider.rows(layer_index);
            // Conv-QKV reads its whole history (HistoryRange::Full).
            held.covers(0..base)?;
            let (past_keys, past_values) = held.to_owned_rows();
            let past = KvView::over_rows(&past_keys, &past_values);
            let state = provider.recurrent_state(layer_index)?;
            let planes = conv_qkv::layer_forward_with(
                &ops.op, &weights, &inputs, state, past, base, projector,
            );
            for (key, value) in planes.keys.into_iter().zip(planes.values) {
                provider.append(layer_index, key, value);
            }
            planes.output
        }
        PreparedAttention::Softmax(ops) => {
            let attention_op = layer
                .attention
                .softmax()
                .expect("prepared softmax operands imply a softmax op");
            match kv {
                Some((kv, layer_index)) if kv.position() == 0 => {
                    let out = backend.attention(ops.call(
                        attention_op,
                        &inputs,
                        layer.declared_norm_eps,
                        hidden,
                    ))?;
                    for (key, value) in out.keys.into_iter().zip(out.values) {
                        kv.append(layer_index, key, value);
                    }
                    out.outputs
                }
                Some((kv, layer_index)) => {
                    let _site = cpu::ledger::in_site(cpu::ledger::Site::Attention);
                    attention_into_kv(
                        attention_op,
                        ops,
                        &inputs,
                        layer.declared_norm_eps,
                        hidden,
                        backend,
                        kv,
                        layer_index,
                    )?
                }
                None => {
                    backend
                        .attention(ops.call(
                            attention_op,
                            &inputs,
                            layer.declared_norm_eps,
                            hidden,
                        ))?
                        .outputs
                }
            }
        }
    };
    // The tail every position shares: post-norm and residual scaling,
    // then the site's update — the residual add on rows, stage five on
    // bundles.
    let deltas: Vec<Vec<f32>> = attn_out
        .into_par_iter()
        .map(|out| {
            let mut out = match &prepared.post_attention {
                Some(norm) => norm.apply(backend, &out),
                None => out,
            };
            scale_residual_delta(layer.residual_scale, &mut out);
            out
        })
        .collect();
    leave_batch_site(
        backend,
        h,
        deltas,
        reductions,
        attn_res,
        BatchSiteContext {
            layer: index,
            site: HcSite::Attention,
            mutation,
            layer_scale: None,
            base,
        },
        sink,
    )?;
    if boundary {
        batch_boundary_event(
            h,
            BoundaryPhase::AfterAttentionBranch,
            entering_prefixes.as_deref().unwrap_or_default(),
            &[],
            index,
            base,
            mutation,
            sink,
        )?;
    }
    let post_attention = {
        let _t = timing::timed(timing::OpClass::PlaneTrace);
        h.clone()
    };

    // A mixer-only (Mamba2) layer carries no FFN program: its one
    // residual update happened above, and the layer is complete.
    // Presence follows the program at execution time too.
    // The FFN OP is the discriminator, never its pre-norm — see the
    // decode path's note. A post-norm layer has no pre-FFN norm and must
    // still run its FFN, over the raw residual.
    let (Some(ffn), Some(ffn_op)) = (&prepared.ffn, &layer.ffn) else {
        let post_layer = {
            let _t = timing::timed(timing::OpClass::PlaneTrace);
            h.clone()
        };
        return Ok(LayerTrace {
            post_attention,
            ffn_input: Vec::new(),
            post_layer,
        });
    };
    // ── FFN site: enter ──
    let BatchSiteEntry {
        branch_inputs: ffn_branch_inputs,
        reductions: ffn_reductions,
        attn_res: ffn_attn_res,
    } = enter_batch_site(
        h,
        BatchSite {
            layer: index,
            which: HcSite::Ffn,
            hyper_connection: prepared.hyper_connection.as_ref().map(|hc| &hc.ffn),
            attention_residual: prepared.attention_residual.as_ref().map(|a| &a.ffn),
        },
        topology,
        layer.declared_norm_eps,
        mutation,
    )?;
    // The normed FFN inputs are computed once here (same values the
    // in-loop computation produced — one deterministic norm per row)
    // so the trace can carry the tap without a second norm pass.
    let ffn_inputs: Vec<Vec<f32>> = match &prepared.pre_ffn {
        Some(pre_ffn) => ffn_branch_inputs
            .par_iter()
            .map(|row| pre_ffn.apply(backend, row))
            .collect(),
        None => ffn_branch_inputs.to_vec(),
    };
    // The FFN's raw residual — what a hybrid's router and expert
    // pre-norm read — is the vector the branch sees: the reduced one on
    // a bundle, never a stream of it (the (d) controls hand it one).
    let residual_source: Cow<'_, [Vec<f32>]> = match (mutation, &*h) {
        (Mutation::HybridResidualFromStreamZero, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(|x| x.stream(0).to_vec()).collect())
        }
        (Mutation::HybridResidualFromStreamMean, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(Bundle::stream_mean).collect())
        }
        _ => Cow::Borrowed(&ffn_branch_inputs),
    };
    let ffn_outs: Vec<Vec<f32>> = if multi_position_ffn() {
        // **CPU-7C2.** One call for every position, rather than a parallel
        // loop over positions each re-entering the executor.
        //
        // The previous shape ran positions through `par_iter_mut`, so
        // every projection inside saw `caller_owns_the_machine` and
        // collapsed to a single worker. CPU-7C1 measured that as
        // `slabs/call` 5.03 -> 2.81 and a 42% loss against serial decode.
        // Here the executor partitions ROWS across its workers and the
        // positions live inside that traversal, which is the ownership
        // rule this module already states.
        let _site = cpu::ledger::in_site(cpu::ledger::Site::Ffn);
        let residuals: Vec<&[f32]> = residual_source.iter().map(Vec::as_slice).collect();
        let normed: Vec<&[f32]> = ffn_inputs.iter().map(Vec::as_slice).collect();
        ffn.apply_from_residual_many(ffn_op, backend, &residuals, &normed, hidden)?
    } else {
        // **Arm B.** The pre-CPU-7C2 shape, kept in the SAME binary so the
        // regression it exhibits is measured beside its fix rather than
        // carried in from another run — the anchor defect CPU-5's G1 was.
        residual_source
            .par_iter()
            .zip(&ffn_inputs)
            .map(|(residual, normed)| {
                // Inside the closure on purpose: this body runs on a
                // rayon worker with its own thread-local, so a guard
                // taken by the caller would attribute none of it.
                let _site = cpu::ledger::in_site(cpu::ledger::Site::Ffn);
                ffn.apply_from_residual(ffn_op, backend, residual, normed, hidden)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    // The glue AFTER it stays position-parallel: norms, scaling and the
    // update are elementwise and issue no projections, so there is no
    // ownership to collapse.
    let deltas: Vec<Vec<f32>> = ffn_outs
        .into_par_iter()
        .map(|out| {
            let mut out = match &prepared.post_ffn {
                Some(norm) => norm.apply(backend, &out),
                None => out,
            };
            scale_residual_delta(layer.residual_scale, &mut out);
            out
        })
        .collect();
    leave_batch_site(
        backend,
        h,
        deltas,
        ffn_reductions,
        ffn_attn_res,
        BatchSiteContext {
            layer: index,
            site: HcSite::Ffn,
            mutation,
            layer_scale: prepared.layer_scale,
            base,
        },
        sink,
    )?;
    if let Some(scale) = prepared.layer_scale {
        match h {
            Plane::Rows(rows) => {
                rows.par_iter_mut()
                    .for_each(|row| backend.scale_row(row, scale));
                each_position(
                    sink,
                    index,
                    base,
                    CarrierForm::Single,
                    rows.len(),
                    CarrierTransition::Scale,
                )?;
            }
            // Preparation refuses this combination; reaching it is an
            // executor bug, not a model.
            Plane::Bundles(_) | Plane::Histories(_) => {
                return Err(VindexError::Parse(format!(
                    "layer {index} carries a layer scale on a component whose residual is not \
                     one vector; preparation should have refused it"
                )))
            }
        }
    }
    let post_layer = {
        let _t = timing::timed(timing::OpClass::PlaneTrace);
        h.clone()
    };
    Ok(LayerTrace {
        post_attention,
        ffn_input: ffn_inputs,
        post_layer,
    })
}
