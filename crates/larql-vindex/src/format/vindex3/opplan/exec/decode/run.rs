//! The decode session step body.

use super::super::address::CarrierAddress;
use super::super::attention_residual::{self, BoundaryPhase};
use super::super::backend::{AttentionStepCall, PlanBackend};
use super::super::hyper_connection::{self, Bundle, Mutation};
use super::super::intervene::{Firing, InterventionPlan};
use super::super::intervene_heads::{HeadFiring, HeadInterventionPlan};
use super::super::observe::{
    AttnResSiteRecord, CarrierForm, CarrierTransition, HcSite, InputSite, StepEvent, StepObserver,
};
use crate::error::VindexError;
use std::borrow::Cow;

#[allow(unused_imports)]
use super::*;

impl<'a, B: PlanBackend> DecodeSession<'a, B> {
    pub(super) fn run(
        &mut self,
        entry: Entry,
        observer: &mut dyn StepObserver,
        mutation: Mutation,
        interventions: &InterventionPlan,
        head_interventions: &HeadInterventionPlan,
    ) -> Result<StepRun, VindexError> {
        let mut profile = super::super::profile::Token::start(
            self.kv.state().position(),
            match &entry {
                Entry::Token(token) => Some(*token),
                _ => None,
            },
        );
        let ops = self.ops.get();
        let mut firings: Vec<Firing> = Vec::new();
        let mut head_firings: Vec<HeadFiring> = Vec::new();
        let hidden = ops.hidden();
        let topology = ops.hyper_connection();
        // The ONE declared fact the attention-residual schedule needs:
        // which layers carry the boundary event. `None` on every other
        // topology.
        let block_size = ops.attention_residual_block_size();
        let position = self.kv.state().position();
        // The form the entering carrier takes once the topology wraps it,
        // so `Enter` is addressed as batch addresses it (RESIDUAL-BUS-2 A1).
        let entering_form = match (topology, block_size) {
            (Some(_), _) => CarrierForm::Bundle,
            (None, Some(_)) => CarrierForm::History,
            (None, None) => CarrierForm::Single,
        };
        let entering_address = |position: usize| {
            CarrierAddress::of(
                position,
                ops.first_layer(),
                entering_form,
                &CarrierTransition::Enter,
            )
        };
        let mut carrier = match entry {
            Entry::Single(values) => {
                observer.entering_carrier(position, &values);
                // A layer-range entry is single-stream by refusal upstream.
                observer.transition(
                    CarrierAddress::of(
                        position,
                        ops.first_layer(),
                        CarrierForm::Single,
                        &CarrierTransition::Enter,
                    ),
                    CarrierTransition::Enter,
                );
                Carrier::Single(values)
            }
            Entry::Token(token) => {
                let h = ops.embed_token(self.plan, self.backend, token)?;
                observer.event(StepEvent::Embedded { position });
                // The first link of the carrier chain (V3-OBS-1, C6):
                // what enters layer 0, before any topology wraps it.
                observer.entering_carrier(position, &h);
                observer.transition(entering_address(position), CarrierTransition::Enter);
                // The embedding enters a hyper-connected stack replicated
                // into every stream (`Transformer.forward`'s repeat) —
                // after its scale and norm, which belong to the lookup.
                match (topology, block_size) {
                    (Some(hc), _) => Carrier::Bundle(Bundle::replicate(&h, hc.streams)),
                    // The embedding enters an attention-residual stack as
                    // the FIRST PREFIX, with an EMPTY history — the
                    // reference starts `block_residual` at
                    // `new_zeros(tokens, 0, hidden)` and nothing is
                    // replicated. That emptiness is what gives layer 0's
                    // attention site nothing to read.
                    (None, Some(_)) => Carrier::History(attention_residual::History::new(h)),
                    (None, None) => Carrier::Single(h),
                }
            }
            Entry::Hidden(h) => {
                observer.entering_carrier(position, &h);
                observer.transition(entering_address(position), CarrierTransition::Enter);
                match (topology, block_size) {
                    (Some(hc), _) => Carrier::Bundle(Bundle::replicate(&h, hc.streams)),
                    (None, Some(_)) => Carrier::History(attention_residual::History::new(h)),
                    (None, None) => Carrier::Single(h),
                }
            }
            #[cfg(test)]
            Entry::Bundle(bundle) => {
                let Some(hc) = topology else {
                    return Err(VindexError::Parse(
                        "a bundle entered a single-stream component".to_string(),
                    ));
                };
                if bundle.streams() != hc.streams || bundle.hidden() != hidden {
                    return Err(VindexError::Parse(format!(
                        "the entering bundle is {} x {}; the component is {} x {}",
                        bundle.streams(),
                        bundle.hidden(),
                        hc.streams,
                        hidden
                    )));
                }
                Carrier::Bundle(bundle)
            }
        };

        let first = ops.first_layer();
        for (offset, state) in ops.layers().iter().enumerate() {
            let index = first + offset;
            profile.layer(index);
            let layer = &self.plan.layers[index];
            profile.phase(super::super::profile::Phase::Attention);
            // ── Attention site ──
            //
            // On a bundle the site reduces first: the ordinary operator
            // consumes the reduced `[hidden]` vector, and so does the
            // pre-attention norm — the norm normalises what the branch
            // sees, never the bundle. On the single stream the "reduced"
            // vector is the residual itself, as it always was.
            // The entering prefix state, captured BEFORE the site reads
            // anything: it is what the boundary event snapshots, and
            // capturing it later would snapshot whatever the site had
            // already done to the carrier.
            let entering_prefix: Option<Vec<f32>> = match &carrier {
                Carrier::History(history) => history.prefix().map(<[f32]>::to_vec),
                _ => None,
            };
            let boundary = match (block_size, &carrier) {
                (Some(size), Carrier::History(_)) => {
                    attention_residual::is_block_boundary(index, size)
                }
                _ => false,
            };
            let boundary_context = SiteContext {
                layer: index,
                site: HcSite::Attention,
                position,
                mutation,
                intervention: None,
                layer_scale: None,
            };
            // Phase one of three: before the attention site reads. The
            // reference does NOTHING here; one control moves the
            // snapshot to this point so the reduction below sees the new
            // set instead of the old one.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::BeforeAttentionReduce,
                    entering,
                    entering,
                    boundary_context,
                    observer,
                );
            }
            let attention_site = enter(
                &carrier,
                state.hyper_connection.as_ref().map(|hc| &hc.attention),
                state.attention_residual.as_ref(),
                HcSite::Attention,
                topology,
                layer.declared_norm_eps,
                index,
                mutation,
            )?;
            // Phase two: between the attention site's reduction and the
            // attention branch. **Where the reference puts it** — the
            // point wave 19's two-point seam cannot express.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::AfterAttentionReduce,
                    entering,
                    &attention_site.branch_input,
                    boundary_context,
                    observer,
                );
            }
            // Under post-norm placement there is no pre-attention norm and
            // the sublayer reads the RAW vector — the norm applies to its
            // output, below, before the update. Cloning rather than
            // normalising by an identity keeps the absent op absent.
            let norm_input: Cow<'_, [f32]> = match (mutation, &carrier) {
                (Mutation::PreNormOnStreamZero, Carrier::Bundle(x)) => Cow::Borrowed(x.stream(0)),
                (Mutation::PreNormOnStreamMean, Carrier::Bundle(x)) => Cow::Owned(x.stream_mean()),
                _ => Cow::Borrowed(&attention_site.branch_input),
            };
            let inputs = [match &state.pre_attention {
                Some(norm) => norm.apply(self.backend, &norm_input),
                None => norm_input.into_owned(),
            }];
            // The sensitivity tap from main, kept ahead of the operator
            // dispatch: it observes the attention INPUT, which both
            // operators read, so it belongs to neither branch.
            observer.operand_input(index, InputSite::Attention, inputs[0].as_slice());
            // One position, either operator. A recurrence appends no KV
            // row — it rewrites its own buffers in place, which is only
            // correct because those buffers are DURABLE: the convolution
            // history spans the step boundary, and a single-position call
            // that reconstructed it from the batch would see a window of
            // one. That is the whole point of QW-3.6a.
            let _attention_stage =
                super::super::stages::stage(super::super::stages::Stage::Attention);
            let raw_attn = match &state.attention {
                super::super::prepared::PreparedAttention::GatedDelta(delta) => {
                    let weights = delta.weights()?;
                    let projector = super::super::dense_rows::consuming_projector(
                        self.backend.dense_projector(),
                        &weights,
                        layer.layer,
                    )?;
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let mut planes = super::super::gated_delta::layer_forward_with(
                        &delta.op,
                        &weights,
                        &inputs,
                        recurrent,
                        super::super::gated_delta::Mutation::None,
                        projector,
                    );
                    planes.output.remove(0)
                }
                super::super::prepared::PreparedAttention::Mamba2(mixer) => {
                    let weights = mixer.weights()?;
                    let projector = super::super::dense_rows::consuming_projector(
                        self.backend.dense_projector(),
                        &weights,
                        layer.layer,
                    )?;
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let mut planes = super::super::mamba2::layer_forward_with(
                        &mixer.op, &weights, &inputs, recurrent, projector,
                    );
                    planes.output.remove(0)
                }
                super::super::prepared::PreparedAttention::Kda(ops) => {
                    let weights = ops.weights()?;
                    let projector = super::super::dense_rows::consuming_projector(
                        self.backend.dense_projector(),
                        &weights,
                        layer.layer,
                    )?;
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let mut planes = super::super::kda::layer_forward_with(
                        &super::super::kda::BackendKdaProjections(projector),
                        &inputs[0],
                        inputs[0].len(),
                        weights,
                        ops.op.geometry(),
                        recurrent,
                        super::super::kda::Mutation::None,
                    );
                    // One position in, one position out: the planes are
                    // flat and this call contributed exactly `hidden`.
                    planes.output.truncate(inputs[0].len());
                    std::mem::take(&mut planes.output)
                }
                super::super::prepared::PreparedAttention::Mla(ops) => {
                    // MLA appends its own position to the latent cache
                    // and reads the whole prefix back — the cache IS the
                    // continuation, and the provider owns it across the
                    // step boundary exactly as it owns KV rows.
                    let hidden = inputs[0].len();
                    let weights = ops.weights()?;
                    let projector = super::super::dense_rows::consuming_projector(
                        self.backend.dense_projector(),
                        &weights,
                        layer.layer,
                    )?;
                    let geometry = ops.op.geometry();
                    let latent = self.kv.state_mut().latent_state(index)?;
                    super::super::mla::mla_forward_with(
                        projector,
                        &inputs[0],
                        hidden,
                        weights,
                        geometry,
                        latent,
                        super::super::mla::Mutation::None,
                    )
                    .output
                }
                super::super::prepared::PreparedAttention::ConvQkv(ops) => {
                    // Two regions, borrowed in phases: past rows copied
                    // out, conv history advanced by the forward, the
                    // step's row appended after — same choreography as
                    // the batch path, at one position.
                    let base = position;
                    self.kv.state_mut().prepare_layer(index);
                    let held = self.kv.state().rows(index);
                    // Conv-QKV reads its whole history (HistoryRange::Full).
                    held.covers(0..base)?;
                    let (past_keys, past_values) = held.to_owned_rows();
                    let past = super::super::kv_view::KvView::over_rows(&past_keys, &past_values);
                    let weights = ops.weights()?;
                    let projector = super::super::dense_rows::consuming_projector(
                        self.backend.dense_projector(),
                        &weights,
                        layer.layer,
                    )?;
                    let recurrent = self.kv.state_mut().recurrent_state(index)?;
                    let mut planes = super::super::conv_qkv::layer_forward_with(
                        &ops.op, &weights, &inputs, recurrent, past, base, projector,
                    );
                    let key = planes.keys.remove(0);
                    let value = planes.values.remove(0);
                    self.kv.state_mut().append(index, key, value);
                    planes.output.remove(0)
                }
                super::super::prepared::PreparedAttention::Softmax(sops) => {
                    let call = sops.call(
                        layer
                            .attention
                            .softmax()
                            .expect("prepared softmax operands imply a softmax op"),
                        &inputs,
                        layer.declared_norm_eps,
                        hidden,
                    );
                    let _site = super::super::cpu::ledger::in_site(
                        super::super::cpu::ledger::Site::Attention,
                    );
                    self.kv.state_mut().prepare_layer(index);
                    let step = AttentionStepCall::new(call, position, self.kv.state().rows(index))?;
                    // V3-HEAD-OBS-1: the same step, with the per-head tap
                    // armed when the observer asked for it. The tap fires
                    // inside the kernel; the structural event closes it
                    // before the write so a reader of the write knows
                    // the heads preceded it.
                    //
                    // V3-INTERVENE-2: when a head intervention addresses
                    // THIS layer, the step routes through the intervened
                    // form instead — the tap (if armed) still fires on
                    // the uninintervened `ctx_h` first (J3); declaring
                    // nothing at this layer costs nothing (J4/JF1).
                    let out = if head_interventions.touches_layer(index) {
                        let heads = step.op.num_q_heads;
                        let wants_heads = observer.wants_attention_heads_at(index, position);
                        let mut fired_heads: Vec<HeadFiring> = Vec::new();
                        let mut head_intervene = |head: usize, ctx_h: &mut [f32]| {
                            if let Some(intervention) = head_interventions.at(index, head, position)
                            {
                                intervention.apply(ctx_h);
                                fired_heads.push(HeadFiring {
                                    layer: index,
                                    head,
                                    position,
                                    kind: intervention.kind(),
                                });
                            }
                        };
                        let out = if wants_heads {
                            self.backend.attention_step_intervened(
                                step,
                                Some(&mut |record| observer.attention_head(index, record)),
                                &mut head_intervene,
                            )?
                        } else {
                            self.backend.attention_step_intervened(
                                step,
                                None,
                                &mut head_intervene,
                            )?
                        };
                        if wants_heads {
                            observer.event(StepEvent::HeadsObserved {
                                layer: index,
                                heads,
                            });
                        }
                        for fired in &fired_heads {
                            observer.event(StepEvent::HeadIntervened {
                                layer: fired.layer,
                                head: fired.head,
                                kind: fired.kind,
                            });
                        }
                        head_firings.extend(fired_heads);
                        out
                    } else if observer.wants_attention_heads_at(index, position) {
                        let heads = step.op.num_q_heads;
                        let out = self.backend.attention_step_observed(step, &mut |record| {
                            observer.attention_head(index, record)
                        })?;
                        observer.event(StepEvent::HeadsObserved {
                            layer: index,
                            heads,
                        });
                        out
                    } else {
                        self.backend.attention_step(step)?
                    };
                    self.kv.state_mut().append(index, out.key, out.value);
                    out.output
                }
            };
            // V3-HEAD-OBS-1, property A6: a layer without softmax heads is
            // named as uncovered on this step, never refused as a plan.
            if observer.wants_attention_heads()
                && !matches!(
                    state.attention,
                    super::super::prepared::PreparedAttention::Softmax(_)
                )
            {
                observer.event(StepEvent::HeadsUncovered { layer: index });
            }
            drop(_attention_stage);
            profile.phase(super::super::profile::Phase::Reentry);
            observer.attention_output(index, position, &raw_attn);
            let mut attn_out = match &state.post_attention {
                Some(norm) => norm.apply(self.backend, &raw_attn),
                None => raw_attn,
            };
            super::super::scale_residual_delta(layer.residual_scale, &mut attn_out);
            leave_site(
                self.backend,
                &mut carrier,
                attn_out,
                attention_site.reduction,
                attention_site.attn_res,
                SiteContext {
                    layer: index,
                    site: HcSite::Attention,
                    position,
                    mutation,
                    intervention: interventions.at(index, HcSite::Attention, position),
                    layer_scale: None,
                },
                observer,
            )?;
            if let Some(fired) = interventions.at(index, HcSite::Attention, position) {
                firings.push(Firing {
                    layer: index,
                    site: HcSite::Attention,
                    position,
                    kind: fired.kind(),
                });
            }
            // Phase three: after the attention branch. The reference does
            // NOTHING here; one control moves the snapshot to this point
            // so it carries the post-attention prefix instead of the
            // entering state.
            if boundary {
                let entering = entering_prefix.as_deref().unwrap_or_default();
                boundary_event(
                    &mut carrier,
                    BoundaryPhase::AfterAttentionBranch,
                    entering,
                    entering,
                    boundary_context,
                    observer,
                );
            }
            observer.event(StepEvent::AttentionDone { layer: index });

            // A mixer-only (Mamba2) layer carries no FFN program: its one
            // residual update happened above, and running a fabricated FFN
            // stage here would be the schema-6 fabrication re-enacted at
            // execution time. Presence follows the program here too.
            //
            // The discriminator is the FFN OP, never its pre-norm. Under
            // post-norm placement the FFN reads the raw residual and has
            // no pre-norm at all, and gating the whole sublayer on that
            // norm's presence silently ran OLMo-2 as attention-only —
            // caught by real-checkpoint parity, not by any synthetic
            // fixture, because a stack missing every FFN still produces
            // fluent-looking planes.
            if let (Some(ffn), Some(ffn_op)) = (&state.ffn, &layer.ffn) {
                let ffn_site = enter(
                    &carrier,
                    state.hyper_connection.as_ref().map(|hc| &hc.ffn),
                    state.attention_residual.as_ref(),
                    HcSite::Ffn,
                    topology,
                    layer.declared_norm_eps,
                    index,
                    mutation,
                )?;
                let normed = match &state.pre_ffn {
                    Some(pre_ffn) => pre_ffn.apply(self.backend, &ffn_site.branch_input),
                    None => ffn_site.branch_input.clone(),
                };
                observer.operand_input(index, InputSite::Ffn, normed.as_slice());
                // The FFN's raw residual — what a hybrid's router and
                // expert pre-norm read — is the vector the branch sees:
                // the reduced one on a bundle, never a stream of it.
                let residual: Cow<'_, [f32]> = match (mutation, &carrier) {
                    (Mutation::HybridResidualFromStreamZero, Carrier::Bundle(x)) => {
                        Cow::Borrowed(x.stream(0))
                    }
                    (Mutation::HybridResidualFromStreamMean, Carrier::Bundle(x)) => {
                        Cow::Owned(x.stream_mean())
                    }
                    _ => Cow::Borrowed(&ffn_site.branch_input),
                };
                let _site =
                    super::super::cpu::ledger::in_site(super::super::cpu::ledger::Site::Ffn);
                profile.phase(super::super::profile::Phase::Ffn);
                let ffn_out = if observer.wants_ffn_down_input(index) {
                    ffn.apply_observed(ffn_op, self.backend, &normed, hidden, &mut |values| {
                        observer.ffn_down_input(index, values)
                    })?
                } else {
                    ffn.apply_from_residual(ffn_op, self.backend, &residual, &normed, hidden)?
                };
                profile.phase(super::super::profile::Phase::Reentry);
                drop(residual);
                observer.operand_input(index, InputSite::FfnOutput, ffn_out.as_slice());
                let mut ffn_out = match &state.post_ffn {
                    Some(norm) => norm.apply(self.backend, &ffn_out),
                    None => ffn_out,
                };
                super::super::scale_residual_delta(layer.residual_scale, &mut ffn_out);
                leave_site(
                    self.backend,
                    &mut carrier,
                    ffn_out,
                    ffn_site.reduction,
                    ffn_site.attn_res,
                    SiteContext {
                        layer: index,
                        site: HcSite::Ffn,
                        position,
                        mutation,
                        intervention: interventions.at(index, HcSite::Ffn, position),
                        layer_scale: state.layer_scale,
                    },
                    observer,
                )?;
                if let Some(fired) = interventions.at(index, HcSite::Ffn, position) {
                    firings.push(Firing {
                        layer: index,
                        site: HcSite::Ffn,
                        position,
                        kind: fired.kind(),
                    });
                }
                if let Some(scale) = state.layer_scale {
                    match &mut carrier {
                        Carrier::Single(h) => {
                            self.backend.scale_row(h, scale);
                            observer.transition(
                                CarrierAddress::of(
                                    position,
                                    index,
                                    CarrierForm::Single,
                                    &CarrierTransition::Scale,
                                ),
                                CarrierTransition::Scale,
                            );
                        }
                        // Preparation refuses this combination; reaching
                        // it is an executor bug, not a model.
                        // Preparation refuses both of these; reaching
                        // one is an executor bug, not a model.
                        Carrier::Bundle(_) | Carrier::History(_) => {
                            return Err(VindexError::Parse(format!(
                                "layer {index} carries a layer scale on a component whose \
                                 residual is not one vector; preparation should have refused it"
                            )))
                        }
                    }
                }
            }
            observer.event(StepEvent::FfnDone { layer: index });
        }

        profile.phase(super::super::profile::Phase::Other);
        // ── The exit ──
        //
        // A bundle leaves the stack through the head's OWN reduction
        // (a different operation from a site's — no Sinkhorn) when the
        // image carries one, and a whole-stack image of a hyper-connected
        // component always does (preparation refuses otherwise). A
        // layer-range image has no exit: the bundle after its last layer
        // IS its output, and it produces no logits.
        let exit = match carrier {
            Carrier::Single(h) => Exit {
                hidden: Some(h),
                #[cfg(test)]
                bundle: None,
            },
            // The attention-residual exit: the same reduction a site
            // runs, once, over the WHOLE snapshot history plus the
            // prefix, before the final norm. Required by the
            // declaration — preparation refuses a whole-stack image
            // without one — so a layer-range image is the only way to
            // reach the `None` arm, and its output is the history it
            // hands on rather than a `[hidden]` vector.
            Carrier::History(history) => {
                let reduced = match ops.attention_residual_exit() {
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
                        let reduction =
                            attention_residual::reduce(&history, pair, exit.norm_eps(), mutation)?;
                        observer.attention_residual_site(AttnResSiteRecord {
                            layer: self.plan.layers.len(),
                            site: HcSite::Ffn,
                            position,
                            candidate_count: history.candidate_count(),
                            snapshot_count_before: history.snapshot_count(),
                            probs: &reduction.probs,
                            mixed_vector: &reduction.mixed,
                            prefix_before: history.prefix().unwrap_or(&reduction.mixed),
                            prefix_after: &reduction.mixed,
                        });
                        Some(reduction.mixed)
                    }
                    Some(_) => Some(history.clone().into_prefix()?),
                    None => None,
                };
                Exit {
                    hidden: reduced,
                    #[cfg(test)]
                    bundle: None,
                }
            }
            Carrier::Bundle(x) => {
                let reduced = match (ops.hyper_connection_head(), topology) {
                    (Some(head), Some(hc)) => Some(hyper_connection::head_reduce(
                        x.as_flat(),
                        x.streams(),
                        hidden,
                        &head.weights(),
                        head.norm_eps(),
                        hc.sinkhorn_eps,
                    )),
                    _ => None,
                };
                Exit {
                    hidden: reduced,
                    #[cfg(test)]
                    bundle: Some(x),
                }
            }
        };
        #[cfg(test)]
        let witness_exit = exit.hidden.clone();
        let logits = match exit.hidden {
            // The one head path (V3-LENS-1): the same function a logit lens
            // calls on an intermediate carrier.
            Some(exit_hidden) => {
                ops.head_logits_observed(self.backend, &exit_hidden, &mut |v| {
                    observer.head_input(position, v)
                })?
            }
            None => {
                if ops.final_norm().is_some() || ops.output().is_some() {
                    return Err(VindexError::Parse(
                        "a hyper-connected bundle reached a whole-stack exit with no head \
                         reduction; preparation should have refused the image"
                            .to_string(),
                    ));
                }
                None
            }
        };
        if let Some(logits) = &logits {
            observer.event(StepEvent::Logits {
                vocab: logits.len(),
            });
        }
        self.kv.state_mut().set_position(position + 1);
        profile.complete();
        Ok(StepRun {
            logits,
            firings,
            head_firings,
            #[cfg(test)]
            exit: witness_exit,
            #[cfg(test)]
            bundle: exit.bundle,
        })
    }
}
