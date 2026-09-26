use super::identity::image_digest;
use super::statistic::Accumulator;
use super::{
    refused, Boundary, CalibrationArtifact, CalibrationBank, CalibrationKey, CalibrationSite,
    Projection, StatisticKind, CAPTURE_REVISION,
};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::{
    continuation_registry::SelectedContinuation,
    decode::DecodeSession,
    lowering::{LoweringIdentity, LoweringRegistry, SharedProvider},
    observe::{InputSite, StepEvent, StepObserver},
    operands::OperandSource,
    prepared::{ExecutionSlice, PreparedOperands},
    provenance::ExecutionProvenance,
};
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerFfn};

/// Offline CPU capture, using the actual decode interpreter. The target layer
/// is included in the sealed image; later layers and the output head are not.
/// No public constructor accepts an asserted prefix digest or a foreign session.
pub struct PreparedCalibration {
    plan: ComponentOpPlan,
    operands: PreparedOperands,
    backend: SharedProvider,
    site: CalibrationSite,
    source_image_sha256: String,
    candidate_prefix_sha256: String,
    execution_sha256: String,
    tokenizer_sha256: String,
}

impl PreparedCalibration {
    pub fn prepare<'s>(
        plan: &ComponentOpPlan,
        source: impl Into<OperandSource<'s>>,
        layer: usize,
        projection: Projection,
    ) -> Result<Self, VindexError> {
        let source = source.into();
        let tokenizer_sha256 = source.store().tokenizer_sha256()?;
        let site = CalibrationSite::resolve(plan, layer, projection)?;
        if !plan.residual_topology.is_single_stream()
            || plan.layers[..=layer].iter().enumerate().any(|(i, l)| {
                l.layer != i
                    || l.attention.softmax().is_none()
                    || !matches!(&l.ffn, Some(LayerFfn::Dense(_)))
            })
        {
            return Err(refused(
                "CAL-1.1 supports contiguous dense softmax single-stream prefixes only",
            ));
        }
        let mut prefix = plan.clone();
        prefix.layers.truncate(layer + 1);
        prefix.final_norm = None;
        prefix.output = None;
        let source_image_sha256 = image_digest(&prefix, source.store().into())?;
        let candidate_prefix_sha256 =
            if source.stamp() == OperandSource::from(source.store()).stamp() {
                source_image_sha256.clone()
            } else {
                image_digest(&prefix, source)?
            };
        // The CPU executor by request, from the shipped registry — not by
        // privileged construction (LOWERING-PLUGIN-1, L3).
        let backend =
            LoweringRegistry::shipped().provider_shared(&LoweringIdentity::cpu_production())?;
        let operands = PreparedOperands::load(&prefix, source, &backend, ExecutionSlice::Full)?;
        let execution_sha256 = ExecutionProvenance::of(&operands).fingerprint();
        Ok(Self {
            plan: prefix,
            operands,
            backend,
            site,
            source_image_sha256,
            candidate_prefix_sha256,
            execution_sha256,
            tokenizer_sha256,
        })
    }

    pub fn key(
        &self,
        bank: &CalibrationBank,
        statistic: StatisticKind,
    ) -> Result<CalibrationKey, VindexError> {
        if bank.tokenizer_sha256 != self.tokenizer_sha256 {
            return Err(refused(
                "bank tokenizer does not match the source container",
            ));
        }
        Ok(CalibrationKey {
            source_image_sha256: self.source_image_sha256.clone(),
            candidate_prefix_sha256: self.candidate_prefix_sha256.clone(),
            execution_sha256: self.execution_sha256.clone(),
            site: self.site.clone(),
            bank_name: bank.name.clone(),
            bank_sha256: bank.sha256()?,
            tokenizer_sha256: bank.tokenizer_sha256.clone(),
            token_bank_id: bank.token_bank_id.clone(),
            population: bank.population,
            statistic,
            samples: bank.sample_count(),
            capture_revision: CAPTURE_REVISION.into(),
        })
    }

    /// Reuse this immutable prefix image for another projection in the same
    /// target layer. Each capture still accumulates only one site's statistics.
    pub fn select_projection(&mut self, projection: Projection) -> Result<(), VindexError> {
        self.site = CalibrationSite::resolve(&self.plan, self.site.layer, projection)?;
        Ok(())
    }

    /// Execute each sequence from a fresh continuation state, built from
    /// the caller's selection; masked rows still supply context. Retain
    /// only a single site's sufficient statistics, not X.
    pub fn capture(
        &self,
        bank: &CalibrationBank,
        statistic: StatisticKind,
        continuation: &SelectedContinuation,
    ) -> Result<CalibrationArtifact, VindexError> {
        let key = self.key(bank, statistic)?;
        if ExecutionProvenance::of(&self.operands).fingerprint() != self.execution_sha256 {
            return Err(refused(
                "execution arithmetic changed since capture preparation",
            ));
        }
        let mut tap = CaptureTap {
            site: &self.site,
            accumulator: Accumulator::new(statistic, self.site.width)?,
            include: false,
            seen: 0,
            error: None,
        };
        for sequence in &bank.sequences {
            let mut state = continuation.build();
            let mut session = DecodeSession::over_prepared(
                &self.plan,
                &self.operands,
                &self.backend,
                &mut *state,
            )?;
            for (&token, &include) in sequence.tokens.iter().zip(&sequence.include) {
                tap.include = include;
                tap.seen = 0;
                session.step_observed(token, &mut tap)?;
                if let Some(error) = tap.error.take() {
                    return Err(error);
                }
                if tap.seen != 1 {
                    return Err(refused(format!(
                        "expected one site input per position, observed {}",
                        tap.seen
                    )));
                }
            }
        }
        if tap.accumulator.samples != bank.sample_count() {
            return Err(refused("captured sample count mismatch"));
        }
        CalibrationArtifact::new(key, tap.accumulator.values)
    }
}

struct CaptureTap<'a> {
    site: &'a CalibrationSite,
    accumulator: Accumulator,
    include: bool,
    seen: usize,
    error: Option<VindexError>,
}

impl CaptureTap<'_> {
    fn observe(&mut self, values: &[f32]) {
        self.seen += 1;
        if self.include && self.error.is_none() {
            if let Err(error) = self.accumulator.push(values) {
                self.error = Some(error);
            }
        }
    }
}

impl StepObserver for CaptureTap<'_> {
    fn event(&mut self, _: StepEvent) {}

    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if layer == self.site.layer
            && matches!(
                (self.site.boundary, site),
                (Boundary::AttentionInput, InputSite::Attention)
                    | (Boundary::FfnInput, InputSite::Ffn)
            )
        {
            self.observe(values);
        }
    }

    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        layer == self.site.layer && self.site.boundary == Boundary::FfnDownInput
    }

    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        if self.wants_ffn_down_input(layer) {
            self.observe(values);
        }
    }
}
