//! Residency, providers and lowering accounting for prepared operands.

use super::super::super::ComponentOpPlan;
use super::super::accounting::{
    expectations, reconcile, BlockGeometry, Bound, Expectation, Observed, Reconciliation,
};
use super::super::backend::{MatrixClass, PlanBackend, WeightFormat, WeightSlice};
use super::super::lowering::{LoweringIdentity, LoweringRegistry};
use super::super::operands::{OperandSource, SourceStamp};
use super::super::realization::{lowerings_stand_in, PinnedAuthorities, RealizationRecord};
use super::super::weights::LoadedWeight;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::represent::codec::CodecRegistry;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

#[allow(unused_imports)]
use super::*;

impl PreparedOperands {
    /// Hand every matrix operand to the backend once, so a device
    /// backend can hold the model resident for this image's lifetime.
    pub(super) fn place<B: PlanBackend + ?Sized>(&self, backend: &B) {
        let mut weights: Vec<WeightSlice<'_>> = Vec::new();
        for layer in &self.layers {
            weights.extend(layer.attention.weight_slices());
            if let Some(ffn) = &layer.ffn {
                weights.extend(ffn.weight_slices());
            }
        }
        if let Some((_, projection)) = &self.output {
            weights.push(projection.slice());
        }
        backend.prepare(&weights);
    }

    /// **What this image actually occupies, by site and representation.**
    ///
    /// Site by site rather than one total, because a total cannot fail
    /// usefully. The claim CPU-2A makes is not "the model is smaller" but
    /// "every streaming matrix kept the checkpoint's own bytes" — and a
    /// single number is satisfied just as well by a stack that halved its
    /// FFN and left 11 GB of recurrence widened.
    ///
    /// The embedding table is the one f32 population that is EXPECTED:
    /// decode gathers a single row from it per token, so it is residency
    /// without traffic, and no kernel here consumes a compact one.
    /// One pinned realization per planned operand this image executes:
    /// the representation resolved, the candidates considered, the one
    /// selected, its reason and its declared residency.
    /// The registry this image was prepared through.
    pub fn registry(&self) -> &'static CodecRegistry {
        self.registry
    }

    pub fn realizations(&self) -> &[RealizationRecord] {
        &self.realizations
    }

    #[cfg(test)]
    pub(in super::super) fn realizations_mut(&mut self) -> &mut Vec<RealizationRecord> {
        &mut self.realizations
    }

    /// Every resident matrix holds the representation its pinned
    /// realization named — checked per layer and site as a multiset, so
    /// the executor, which reads its kernel off the resident bytes, can
    /// run nothing the plan did not pin. A packed bank's per-expert slices
    /// are a different realization and are checked by the bank loader.
    pub fn verify_pins(&self) -> Result<(), VindexError> {
        let pinned =
            |layer: Option<usize>, want: &dyn Fn(Operation) -> bool| -> Vec<WeightFormat> {
                let mut out: Vec<WeightFormat> = self
                    .realizations
                    .iter()
                    .filter(|r| r.planned.layer == layer && want(r.planned.operation))
                    .map(|r| r.selection.realization.format())
                    .collect();
                out.sort_by_key(|f| format!("{f:?}"));
                out
            };
        let observed = |weights: Vec<&LoadedWeight>| -> Vec<WeightFormat> {
            let mut out: Vec<WeightFormat> = weights.iter().map(|w| w.format()).collect();
            out.sort_by_key(|f| format!("{f:?}"));
            out
        };
        let mismatch = |site: String, pinned: Vec<WeightFormat>, resident: Vec<WeightFormat>| {
            VindexError::Parse(format!(
                "{site}: the resident representations {resident:?} are not the pinned \
                 realizations {pinned:?} — the loader drifted from the selector"
            ))
        };
        if let Some(ffns) = &self.dense_ffns {
            let mut expected: Vec<_> = self
                .realizations
                .iter()
                .map(|r| r.selection.realization.format())
                .collect();
            expected.sort_by_key(|f| format!("{f:?}"));
            let resident = observed(ffns.matrices());
            if expected != resident {
                return Err(mismatch("dense FFN worker".into(), expected, resident));
            }
        }
        for (offset, layer) in self.layers.iter().enumerate() {
            let index = self.first_layer + offset;
            let attention = pinned(Some(index), &|o| {
                o == Operation::Project(MatrixClass::AttentionProjection)
            });
            let resident = observed(layer.attention.matrices());
            if attention != resident {
                return Err(mismatch(
                    format!("layer {index} attention"),
                    attention,
                    resident,
                ));
            }
            let ffn = pinned(Some(index), &|o| {
                o == Operation::Project(MatrixClass::FfnProjection)
            });
            let resident = observed(
                layer
                    .ffn
                    .as_ref()
                    .map(|f| f.dense_matrices())
                    .unwrap_or_default(),
            );
            if ffn != resident {
                return Err(mismatch(format!("layer {index} ffn"), ffn, resident));
            }
        }
        let head = pinned(None, &|o| o == Operation::OutputHead);
        let resident = observed(self.output.iter().map(|(_, w)| w).collect());
        if head != resident {
            return Err(mismatch("output head".to_string(), head, resident));
        }
        Ok(())
    }

    /// Every planned operand paired with the object(s) the loader bound
    /// for it — the OBSERVATION side of the accounting, read off the
    /// resident objects and never off a declaration.
    pub fn bound(&self, plan: &ComponentOpPlan) -> Result<Vec<Observed>, VindexError> {
        if let Some(experts) = &self.routed_experts {
            return experts.bound();
        }
        if let Some(ffns) = &self.dense_ffns {
            return ffns.bound(plan);
        }
        let mut out = Vec::new();
        if let (Some(embedding), Some(table)) = (&plan.embedding, &self.embed_table) {
            out.push(Observed {
                operand: embedding.table.clone(),
                operation: Operation::Embed,
                layer: None,
                format: WeightFormat::F32,
                resident_bytes: std::mem::size_of_val(&table[..]) as u64,
                mapped_bytes: 0,
                allocations: 0,
            });
        }
        for (offset, prepared) in self.layers.iter().enumerate() {
            let index = self.first_layer + offset;
            let layer = plan.layers.get(index).ok_or_else(|| {
                VindexError::Parse(format!("layer {index}: prepared but not in the plan"))
            })?;
            for bound in prepared.attention.bound(&layer.attention)? {
                out.push(bound.observed(
                    Operation::Project(MatrixClass::AttentionProjection),
                    Some(index),
                )?);
            }
            if let (Some(ffn), Some(op)) = (&prepared.ffn, &layer.ffn) {
                // The loader names the operation it bound each object for.
                for (operation, bound) in ffn.bound(op)? {
                    out.push(bound.observed(operation, Some(index))?);
                }
            }
        }
        if let Some((op, weight)) = &self.output {
            out.push(Bound::one(&op.projection, weight).observed(Operation::OutputHead, None)?);
        }
        Ok(out)
    }

    /// What every pinned realization DECLARES it costs, priced from the
    /// pin, the codec's declared residency, `geometry`, and the container's
    /// recorded lengths — the EXPECTATION side, which reads no object.
    pub fn expectations(
        &self,
        store: OperandSource<'_>,
        geometry: BlockGeometry,
    ) -> Vec<Expectation> {
        expectations(
            &self.realizations,
            |op| {
                let full = store.stored_len(op)?;
                if let ExecutionSlice::RoutedExperts {
                    expert_start,
                    expert_end,
                    ..
                } = self.slice
                {
                    let count = *op.shape.first()?;
                    Some(full / count as u64 * (expert_end - expert_start) as u64)
                } else {
                    Some(full)
                }
            },
            geometry,
        )
    }

    /// Bind the declarations AGAINST the observations: every pin meets
    /// exactly one resident object in the pinned representation holding
    /// the declared bytes, and nothing resident is unpinned.
    pub fn reconcile(
        &self,
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
    ) -> Result<Reconciliation, VindexError> {
        reconcile(
            &self.expectations(store, BlockGeometry::executor()),
            &self.bound(plan)?,
        )
    }

    /// The providers this image was prepared against, by representation
    /// label, with the identity each resolved to. Since 3d every record
    /// names a registered codec — a packed bank's carrier label resolves
    /// to the codec the plan declares — so a `None` here is an overlay
    /// edit's f32-space fact, never a label a loader judged for itself.
    pub fn providers(&self) -> Vec<(String, Option<CodecIdentity>)> {
        let mut out: Vec<(String, Option<CodecIdentity>)> = Vec::new();
        let mut note = |label: &str, identity: &Option<CodecIdentity>| {
            if !out.iter().any(|(seen, _)| seen == label) {
                out.push((label.to_string(), identity.clone()));
            }
        };
        for r in &self.realizations {
            note(&r.representation, &r.codec_provider);
            // A DEPENDENCY's provider is a provider. An image prepared
            // while a codebook's codec was registered is not executable
            // once that codec is gone, however well the codes' own
            // provider survives — the values would be unobtainable, not
            // merely differently obtained.
            for dependency in &r.dependencies {
                note(&dependency.label, &dependency.provider);
            }
        }
        out
    }

    /// The lowering providers that qualified this image's pins, once
    /// each, in pin order (LOWERING-PLUGIN-1, L4).
    ///
    /// One today: an image is prepared by one provider. A list because
    /// nothing in the contract says it must stay one, and because the
    /// check below should not have to change if it stops being one.
    pub fn lowerings(&self) -> Vec<LoweringIdentity> {
        let mut out: Vec<LoweringIdentity> = Vec::new();
        for r in &self.realizations {
            if !out.contains(&r.lowering_provider) {
                out.push(r.lowering_provider.clone());
            }
        }
        out
    }

    /// Both authorities this image was pinned under, as one value — what
    /// was DECIDED, holding neither the codecs nor the providers that
    /// decided it.
    pub fn authorities(&self) -> PinnedAuthorities {
        PinnedAuthorities {
            codecs: self.providers(),
            lowerings: self.lowerings(),
        }
    }

    /// Refuse to execute this image against a registry that no longer
    /// resolves every provider it was prepared with to the same identity
    /// — a provider that disappeared or changed invalidates the
    /// preparation; nothing falls back.
    pub fn ensure_providers_in(&self, registry: &CodecRegistry) -> Result<(), VindexError> {
        self.authorities().ensure_codecs_in(registry)
    }

    /// Refuse to execute this image on a provider that did not prepare
    /// it (LOWERING-PLUGIN-1, L4).
    ///
    /// The registry check above asks whether the pinned provider still
    /// EXISTS. This asks the question available where no registry is —
    /// at the execution seam, which is handed a provider directly —
    /// namely whether the provider about to run these pins is the one
    /// that made them. A pin is a decision one provider took from one
    /// set of declared facts; another provider running it is that
    /// decision reinterpreted, which is exactly what this wave forbids.
    pub fn ensure_lowered_by<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
    ) -> Result<(), VindexError> {
        let executing = backend.identity();
        for pinned in self.lowerings() {
            if pinned != executing {
                return Err(VindexError::Parse(format!(
                    "this image's realizations were pinned by lowering provider `{pinned}` and \
                     `{executing}` is executing them; re-prepare rather than run a pin another \
                     provider decided"
                )));
            }
        }
        Ok(())
    }

    /// The same question of the LOWERING plane: is the provider that
    /// qualified these pins still in `registry`, under exactly the
    /// identity the pins recorded (LOWERING-PLUGIN-1, L4)?
    ///
    /// The registry is the caller's — a session's current authority —
    /// because a lowering registry is a carried value and not a
    /// `&'static` the image can hold. What the image holds is the
    /// decision, never the provider: an image that owned its provider
    /// would keep a removed one alive and could never be invalidated by
    /// its disappearance.
    pub fn ensure_lowerings_in(&self, registry: &LoweringRegistry) -> Result<(), VindexError> {
        // The lowering half only: an image asking whether its provider
        // still exists has no reason to build its codec half first.
        lowerings_stand_in(&self.lowerings(), registry)
    }

    pub fn residency_census(&self) -> ResidencyCensus {
        let mut census = ResidencyCensus::default();
        if let Some(ffns) = &self.dense_ffns {
            for weight in ffns.matrices() {
                census.ffn.add(weight);
            }
        }
        if let Some(experts) = &self.routed_experts {
            for weight in experts.matrices() {
                census.ffn.add(weight);
            }
            census.glue.widened_f32 += experts.bias_bytes();
        }
        if let Some(table) = &self.embed_table {
            census.embedding.widened_f32 += std::mem::size_of_val(&table[..]);
        }
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                }
                PreparedAttention::GatedDelta(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                PreparedAttention::Mamba2(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                PreparedAttention::ConvQkv(ops) => {
                    // Attention matrix traffic — the block attends, and
                    // its fused QKV/out projections are what a device
                    // backend would hold resident on that site.
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                // KDA is a recurrence: its four wide projections are
                // counted where the other recurrences' are, so a
                // hybrid's census still separates "the model attends"
                // from "the model recurs".
                PreparedAttention::Kda(ops) => {
                    for w in ops.loaded_matrices() {
                        census.delta.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
                // MLA attends — over a compressed cache, but it attends.
                PreparedAttention::Mla(ops) => {
                    for w in ops.loaded_matrices() {
                        census.attention.add(w);
                    }
                    census.glue.widened_f32 += ops.glue_bytes();
                }
            }
            if let Some(ffn) = &layer.ffn {
                for w in ffn.loaded_matrices() {
                    census.ffn.add(w);
                }
            }
            census.glue.widened_f32 += layer.glue_bytes();
        }
        if let Some(norm) = &self.final_norm {
            census.glue.widened_f32 += std::mem::size_of_val(&norm.weight[..]);
        }
        if let Some(head) = &self.hyper_connection_head {
            census.glue.widened_f32 += head.glue_bytes();
        }
        if let Some((_, projection)) = &self.output {
            census.head.add(projection);
        }
        census
    }

    /// Where this image's allocations landed. See [`AllocationCensus`].
    /// The image's mappings, and how much of them is physically resident
    /// at this moment: address space summed once per bound region, pages
    /// resident as the OS reports them now. Cheap enough to ask between
    /// tokens, which is what a residency curve is. Derived from one
    /// [`Self::observe_residency`] capture, so a failed region is counted,
    /// never summed as zero.
    pub fn mapped_residency(&self) -> MappedResidency {
        self.observe_residency().summary()
    }

    /// One scan of every mapped region of this image, each read once, in
    /// the image's fixed traversal order, with the outcome of every read
    /// kept. The aggregate and the per-region report both come from it.
    pub fn observe_residency(&self) -> ResidencyObservation {
        let mut regions = Vec::new();
        self.for_each_weight(|w| {
            if let Some(region) = w.mapped_region() {
                regions.push(region.clone());
            }
        });
        ResidencyObservation::capture(&regions)
    }

    /// Every loaded weight of the image, in one fixed order: the dense
    /// FFNs, the routed experts, then per layer its attention and FFN
    /// matrices, then the output projection. The single definition of
    /// "the image's weights" that its residency views traverse.
    fn for_each_weight(&self, mut add: impl FnMut(&LoadedWeight)) {
        if let Some(ffns) = &self.dense_ffns {
            ffns.matrices().iter().for_each(|w| add(w));
        }
        if let Some(experts) = &self.routed_experts {
            experts.matrices().iter().for_each(|w| add(w));
        }
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::GatedDelta(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Mamba2(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::ConvQkv(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Kda(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::Mla(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
            }
            if let Some(ffn) = &layer.ffn {
                ffn.loaded_matrices().iter().for_each(|w| add(w));
            }
        }
        if let Some((_, projection)) = &self.output {
            add(projection);
        }
    }

    pub fn allocation_census(&self) -> AllocationCensus {
        let mut census = AllocationCensus::default();
        let mut add = |w: &LoadedWeight| {
            for (address, bytes) in w.allocations() {
                census.add(address, bytes);
            }
        };
        if let Some(ffns) = &self.dense_ffns {
            ffns.matrices().iter().for_each(|w| add(w));
        }
        if let Some(experts) = &self.routed_experts {
            experts.matrices().iter().for_each(|w| add(w));
        }
        for layer in &self.layers {
            match &layer.attention {
                PreparedAttention::Softmax(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::GatedDelta(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Mamba2(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::ConvQkv(ops) => {
                    ops.loaded_matrices().iter().for_each(|w| add(w))
                }
                PreparedAttention::Kda(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
                PreparedAttention::Mla(ops) => ops.loaded_matrices().iter().for_each(|w| add(w)),
            }
            if let Some(ffn) = &layer.ffn {
                ffn.loaded_matrices().iter().for_each(|w| add(w));
            }
        }
        if let Some((_, projection)) = &self.output {
            add(projection);
        }
        census
    }

    /// The slice this image was prepared for.
    pub fn slice(&self) -> &ExecutionSlice {
        &self.slice
    }

    /// The effective source this image was compiled from.
    pub fn source_stamp(&self) -> SourceStamp {
        self.stamp
    }

    /// Whether this image still describes `source`.
    ///
    /// False after any overlay mutation, and for a different store or a
    /// different override set. A caller that has the source in hand
    /// should ask before reusing a cached image; one that does not
    /// (the serve path, which holds only its own image) is safe by
    /// ownership — it has nothing else to confuse it with.
    pub fn is_current_for(&self, source: &OperandSource<'_>) -> bool {
        self.stamp == source.stamp()
    }

    /// [`Self::is_current_for`] as a refusal, for callers that would
    /// otherwise execute a stale image.
    pub fn ensure_current_for(&self, source: &OperandSource<'_>) -> Result<(), VindexError> {
        if self.is_current_for(source) {
            return Ok(());
        }
        Err(VindexError::Parse(
            "this prepared image was compiled from a different effective operand source — \
             the overlay changed, or it belongs to another container. Re-prepare rather than \
             executing a stale compilation of the model."
                .to_string(),
        ))
    }
}
