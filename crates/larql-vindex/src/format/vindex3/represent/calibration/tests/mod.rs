use super::*;
use crate::format::vindex3::opplan::exec::{
    backend::{PlanBackend, ProjectCall, WeightSlice},
    continuation::{
        plan_continuation_geometry, LatentKvRows, LayerContinuationGeometry, RecurrentState,
    },
    continuation_authority::ContinuationConfig,
    continuation_identity::ContinuationIdentity,
    continuation_registry::{
        BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
        SelectedContinuation,
    },
    decode::DecodeSession,
    kv::{ContinuationError, ContinuationProvider, LayerKvGeometry, RowFactory, RowKvState},
    observe::{InputSite, StepEvent, StepObserver},
    operands::{OperandEdit, OperandOverrides, OperandSource, OperandStore},
    production::ProductionBackend,
    reference::ReferenceBackend,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerFfn};
use crate::format::vindex3::{encode::encode_system, fixtures, inspect::inspect_container};

/// The built-in row provider, reached the way production reaches it:
/// selected by identity from a registry for this plan's geometry.
fn row(plan: &ComponentOpPlan) -> SelectedContinuation {
    let mut registry = ContinuationRegistry::new();
    registry.register(Box::new(RowFactory)).unwrap();
    registry
        .select(
            &RowFactory.identity(),
            &ContinuationConfig::empty(),
            &plan_continuation_geometry(plan).unwrap(),
        )
        .unwrap()
}

fn tokenizer_bytes() -> Vec<u8> {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab((0..128).map(|i| (format!("t{i}"), i)).collect())
        .unk_token("t0".into())
        .build()
        .unwrap();
    serde_json::to_vec(&tokenizers::Tokenizer::new(model)).unwrap()
}

fn fixture(glimmer: bool) -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let weights = tempfile::tempdir().unwrap();
    if glimmer {
        fixtures::miniature_glimmer(weights.path());
    } else {
        fixtures::dense_f32_model(weights.path());
    }
    let inventory = larql_models::inventory::build_inventory(weights.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("calibration-fixture".into(), inventory)],
        container.path(),
    )
    .unwrap();
    std::fs::write(container.path().join("tokenizer.json"), tokenizer_bytes()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn bank() -> CalibrationBank {
    CalibrationBank::new(
        "synthetic-mechanism-only".into(),
        digest(&tokenizer_bytes()),
        Population::Calibration,
        vec![
            CalibrationSequence {
                tokens: vec![3, 17, 8, 0, 11],
                include: vec![true, false, true, true, true],
            },
            CalibrationSequence {
                tokens: vec![11, 3],
                include: vec![false, true],
            },
        ],
    )
    .unwrap()
}

#[derive(Default)]
struct Rows {
    layer: usize,
    attention: Vec<Vec<f32>>,
    ffn: Vec<Vec<f32>>,
    down: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
}
impl StepObserver for Rows {
    fn event(&mut self, _: StepEvent) {}
    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if layer == self.layer {
            match site {
                InputSite::Attention => &mut self.attention,
                InputSite::Ffn => &mut self.ffn,
                InputSite::FfnOutput => &mut self.output,
            }
            .push(values.to_vec());
        }
    }
    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        layer == self.layer
    }
    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        assert_eq!(layer, self.layer);
        self.down.push(values.to_vec());
    }
}

fn direct_rows<B: PlanBackend>(
    plan: &ComponentOpPlan,
    source: OperandSource<'_>,
    bank: &CalibrationBank,
    backend: &B,
) -> Rows {
    let mut result = Rows {
        layer: 1,
        ..Rows::default()
    };
    for sequence in &bank.sequences {
        let mut session =
            DecodeSession::new(plan, source, backend, Box::new(RowKvState::default())).unwrap();
        for (&token, &include) in sequence.tokens.iter().zip(&sequence.include) {
            let mut tap = Rows {
                layer: 1,
                ..Rows::default()
            };
            session.step_observed(token, &mut tap).unwrap();
            if include {
                result.attention.extend(tap.attention);
                result.ffn.extend(tap.ffn);
                result.down.extend(tap.down);
                result.output.extend(tap.output);
            }
        }
    }
    result
}

fn gram(rows: &[Vec<f32>]) -> Vec<f64> {
    let width = rows[0].len();
    (0..width)
        .flat_map(|i| {
            (0..width).map(move |j| rows.iter().map(|x| f64::from(x[i]) * f64::from(x[j])).sum())
        })
        .collect()
}

fn tap_parity<B: PlanBackend>(backend: &B) {
    let (_dir, plan, store) = fixture(false);
    let bank = bank();
    let mut observed =
        DecodeSession::new(&plan, &store, backend, Box::new(RowKvState::default())).unwrap();
    let mut plain =
        DecodeSession::new(&plan, &store, backend, Box::new(RowKvState::default())).unwrap();
    let down = &plan.layers[1].ffn.as_ref().unwrap().dense().unwrap().down;
    let weights = store.load(down).unwrap();
    for &token in &bank.sequences[0].tokens {
        let mut tap = Rows {
            layer: 1,
            ..Rows::default()
        };
        assert_eq!(
            observed.step_observed(token, &mut tap).unwrap().logits,
            plain.step(token).unwrap().logits
        );
        assert_eq!(tap.down.len(), 1);
        let output = backend
            .project(ProjectCall {
                weight: WeightSlice::F32(&weights),
                x: &tap.down[0],
                out_dim: down.shape[0],
                in_dim: down.shape[1],
            })
            .unwrap();
        assert_eq!(
            output, tap.output[0],
            "captured down input must reconstruct the executor's raw output"
        );
    }
}

/// Builds the row state under a non-built-in identity and counts builds,
/// so a capture that ignored the caller's selection would be visible.
struct CountingRow(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl ContinuationFactory for CountingRow {
    fn identity(&self) -> ContinuationIdentity {
        ContinuationIdentity::new("counting-row", 1)
    }

    fn regions(&self) -> &[ContinuationRegion] {
        RowFactory.regions()
    }

    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        RowFactory.build(config)
    }
}

/// Row state that refuses every continuation announcement, as a provider
/// lacking the plan's state form would. Everything else delegates.
struct RefusingState(RowKvState);

impl ContinuationProvider for RefusingState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.0.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.0.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> crate::format::vindex3::opplan::exec::kv_view::KvView<'_> {
        self.0.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.0.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.0.position()
    }
    fn set_position(&mut self, position: usize) {
        self.0.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.0.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.0.invalidated()
    }
    fn prepare_continuation(
        &mut self,
        _: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        Err(ContinuationError::RecurrentUnsupported {
            provider: "refusing-row",
            layer: 0,
        })
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.0.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.0.latent_state(layer)
    }
}

struct RefusingRow;

impl ContinuationFactory for RefusingRow {
    fn identity(&self) -> ContinuationIdentity {
        ContinuationIdentity::new("refusing-row", 1)
    }
    fn regions(&self) -> &[ContinuationRegion] {
        RowFactory.regions()
    }
    fn build(&self, _: &ContinuationConfig) -> BoxedContinuation {
        Box::new(RefusingState(RowKvState::default()))
    }
}

mod tests_basics;
mod tests_basics_2;
