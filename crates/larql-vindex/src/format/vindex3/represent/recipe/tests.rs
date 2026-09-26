use super::super::{candidate_authority::read_candidate_evidence, compiler::CandidateIndex};
use super::*;
use crate::format::vindex3::opplan::exec::{
    continuation::plan_continuation_geometry,
    continuation_authority::ContinuationConfig,
    continuation_registry::{ContinuationFactory, ContinuationRegistry},
    decode::DecodeSession,
    kv::{RowFactory, RowKvState},
    observe::{InputSite, StepEvent, StepObserver},
    operands::{OperandEdit, RepresentationSource},
    production::ProductionBackend,
};
use crate::format::vindex3::represent::calibration::{CalibrationManifest, CalibrationSequence};
use crate::format::vindex3::{
    fixtures,
    opplan::{ComponentOpPlan, OperandRef},
};

struct Fixture {
    dir: tempfile::TempDir,
    src: PathBuf,
    plan: ComponentOpPlan,
    store: OperandStore,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("weights");
        std::fs::create_dir(&ckpt).unwrap();
        let src = dir.path().join("source");
        fixtures::encode_fixture_container(fixtures::dense_f32_model, &ckpt, &src, "target");
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab((0..128).map(|i| (format!("t{i}"), i)).collect())
            .unk_token("t0".into())
            .build()
            .unwrap();
        std::fs::write(
            src.join("tokenizer.json"),
            serde_json::to_vec(&tokenizers::Tokenizer::new(model)).unwrap(),
        )
        .unwrap();
        let inspection = inspect_container(&src, false).unwrap();
        let plan = plan_component_ops(&inspection, &src, "target")
            .unwrap()
            .plan
            .unwrap();
        let store = OperandStore::open(&src, &inspection).unwrap();
        Self {
            dir,
            src,
            plan,
            store,
        }
    }
    fn bank(&self) -> CalibrationBank {
        CalibrationBank::new(
            "cal12-fixture".into(),
            self.store.tokenizer_sha256().unwrap(),
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
    fn request(&self, artifacts: &Path) -> GptqRequest {
        std::fs::create_dir(artifacts).unwrap();
        let mut sites = BTreeMap::new();
        for l in &self.plan.layers {
            let a = l.attention.softmax().unwrap();
            let f = l.ffn.as_ref().unwrap().dense().unwrap();
            for op in [&a.q, &a.k, &a.v, f.gate.as_ref().unwrap(), &f.up, &f.down] {
                sites.insert(
                    (op.object.clone(), op.tensor.clone()),
                    CalibrationInput::CaptureTo(artifacts.join(format!("site-{}", sites.len()))),
                );
            }
        }
        GptqRequest {
            component: "target".into(),
            bank: self.bank(),
            sites,
            continuation: self.continuation(),
        }
    }
    /// The built-in row provider, selected by identity as production callers do.
    fn continuation(&self) -> SelectedContinuation {
        let mut registry = ContinuationRegistry::new();
        registry.register(Box::new(RowFactory)).unwrap();
        registry
            .select(
                &RowFactory.identity(),
                &ContinuationConfig::empty(),
                &plan_continuation_geometry(&self.plan).unwrap(),
            )
            .unwrap()
    }
    fn out(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}
fn candidate(path: &Path) -> CandidateIndex {
    serde_json::from_slice(&std::fs::read(path.join("candidate.json")).unwrap()).unwrap()
}
fn stored(path: &Path) -> OperandStore {
    let insp = inspect_container(path, false).unwrap();
    OperandStore::open_for(path, &insp, Some("NVFP4"), RepresentationSource::Stored).unwrap()
}
fn logits(plan: &ComponentOpPlan, store: &OperandStore) -> Vec<Vec<f32>> {
    let backend = ProductionBackend::new();
    let mut session =
        DecodeSession::new(plan, store, &backend, Box::new(RowKvState::default())).unwrap();
    [3, 17, 8, 0, 11]
        .iter()
        .map(|t| session.step(*t).unwrap().logits.unwrap())
        .collect()
}
fn op_for(f: &Fixture, t: &TensorDerivation) -> OperandRef {
    let a = f
        .plan
        .layers
        .iter()
        .flat_map(|l| {
            let a = l.attention.softmax().unwrap();
            let d = l.ffn.as_ref().unwrap().dense().unwrap();
            vec![
                &a.q,
                &a.k,
                &a.v,
                &a.o,
                d.gate.as_ref().unwrap(),
                &d.up,
                &d.down,
            ]
        })
        .find(|op| op.object == t.object && op.tensor == t.tensor)
        .unwrap();
    a.clone()
}
#[derive(Default)]
struct Tap {
    layer: usize,
    site: Option<Projection>,
    rows: Vec<Vec<f32>>,
}
impl StepObserver for Tap {
    fn event(&mut self, _: StepEvent) {}
    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if layer == self.layer
            && matches!(
                (self.site, site),
                (
                    Some(Projection::Query | Projection::Key | Projection::Value),
                    InputSite::Attention
                ) | (Some(Projection::Gate | Projection::Up), InputSite::Ffn)
            )
        {
            self.rows.push(values.to_vec());
        }
    }
    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        layer == self.layer && self.site == Some(Projection::Down)
    }
    fn ffn_down_input(&mut self, _: usize, values: &[f32]) {
        self.rows.push(values.to_vec());
    }
}
fn independent_gram(f: &Fixture, edits: &OperandOverrides, c: &CalibrationManifest) -> Vec<f64> {
    let mut rows = Vec::new();
    let backend = ProductionBackend::new();
    for (tokens, mask) in [
        (vec![3, 17, 8, 0, 11], vec![true, false, true, true, true]),
        (vec![11, 3], vec![false, true]),
    ] {
        let mut session = DecodeSession::new(
            &f.plan,
            OperandSource::overlaid(&f.store, edits),
            &backend,
            Box::new(RowKvState::default()),
        )
        .unwrap();
        for (token, selected) in tokens.into_iter().zip(mask) {
            let mut tap = Tap {
                layer: c.key.site.layer,
                site: Some(c.key.site.projection),
                ..Tap::default()
            };
            session.step_observed(token, &mut tap).unwrap();
            assert_eq!(tap.rows.len(), 1);
            if selected {
                rows.extend(tap.rows);
            }
        }
    }
    let k = c.key.site.width;
    (0..k)
        .flat_map(|i| {
            let rows = &rows;
            (0..k).map(move |j| rows.iter().map(|r| f64::from(r[i]) * f64::from(r[j])).sum())
        })
        .collect()
}

#[test]
fn cal12_sequential_stored_candidate_has_frozen_scales_and_independent_derivation() {
    let f = Fixture::new();
    let artifacts = f.out("calibration");
    let request = f.request(&artifacts);
    let out = f.out("gptq");
    let nearest_out = f.out("nearest");
    let spec = RepresentSpec::nvfp4();
    compile_representation_recipe(&f.src, &nearest_out, &spec, Nvfp4Recipe::Nearest).unwrap();
    compile_representation_recipe(&f.src, &out, &spec, Nvfp4Recipe::Gptq(&request)).unwrap();
    let ci = candidate(&out);
    let record = ci.derivation.as_ref().unwrap();
    record.validate(&ci).unwrap();
    let authority = read_candidate_evidence(&out).unwrap();
    let output = stored(&out);
    let control = stored(&nearest_out);
    let mut edits = OperandOverrides::new();
    let mut decoded_edits = OperandOverrides::new();
    let mut changed = 0;
    let mut later_falsified = false;
    let mut down_falsified = false;
    for t in &record.tensors {
        let op = op_for(&f, t);
        let layout = PackLayout::derive(&op.shape, &op.tensor).unwrap();
        let raw = output.load_raw(&op).unwrap();
        let a = control.load_raw(&op).unwrap();
        assert_eq!(raw.bytes.len(), a.bytes.len());
        assert_eq!(
            &raw.bytes[layout.scales_offset()..],
            &a.bytes[layout.scales_offset()..],
            "every scale byte frozen"
        );
        changed += usize::from(raw.bytes != a.bytes);
        assert_eq!(
            super::super::compile::hash_bytes(&raw.bytes),
            t.payload_sha256
        );
        if let Some(c) = &t.calibration {
            assert_eq!(t.recipe, EncoderRecipe::gptq_v1());
            let CalibrationInput::CaptureTo(path) =
                &request.sites[&(t.object.clone(), t.tensor.clone())]
            else {
                unreachable!()
            };
            let prepared = PreparedCalibration::prepare(
                &f.plan,
                OperandSource::overlaid(&f.store, &edits),
                c.key.site.layer,
                c.key.site.projection,
            )
            .unwrap();
            let expected = prepared
                .key(&request.bank, StatisticKind::DenseGram)
                .unwrap();
            let captured = CalibrationArtifact::read(path, &expected).unwrap();
            assert_eq!(captured.values(), independent_gram(&f, &edits, c));
            let canonical = PreparedCalibration::prepare(
                &f.plan,
                &f.store,
                c.key.site.layer,
                c.key.site.projection,
            )
            .unwrap()
            .capture(
                &request.bank,
                StatisticKind::DenseGram,
                &request.continuation,
            )
            .unwrap();
            if c.key.site.layer == 1 && c.key.site.projection == Projection::Query {
                assert_ne!(captured.values(), canonical.values());
                later_falsified = true;
                let decoded = PreparedCalibration::prepare(
                    &f.plan,
                    OperandSource::overlaid(&f.store, &decoded_edits),
                    1,
                    Projection::Query,
                )
                .unwrap();
                assert_ne!(
                    expected.execution_sha256,
                    decoded
                        .key(&request.bank, StatisticKind::DenseGram)
                        .unwrap()
                        .execution_sha256,
                    "packed CPU realization must not be relabeled as decoded f32"
                );
                assert!(CalibrationArtifact::read(path, &canonical.manifest().key).is_err());
            }
            if c.key.site.layer == 0 && c.key.site.projection == Projection::Down {
                assert_ne!(captured.values(), canonical.values());
                down_falsified = true;
            }
        } else {
            assert_eq!(t.recipe, EncoderRecipe::nearest_v1());
            assert_eq!(raw.bytes, a.bytes);
        }
        // Bind the final container's actual physical bytes into the prefix.
        edits.replace_nvfp4(
            &op,
            output.map_region(&op, layout.total_len as u64).unwrap(),
        );
        // Deliberately decoded counterfactual: values match, arithmetic differs.
        let decoded = output.load(&op).unwrap();
        for (index, row) in decoded.chunks_exact(layout.k).enumerate() {
            decoded_edits.push(
                &op,
                OperandEdit::Row {
                    index,
                    values: row.to_vec(),
                },
            );
        }
    }
    assert!(changed > 0 && later_falsified && down_falsified);
    let before = logits(&f.plan, &output);
    let backend = ProductionBackend::new();
    let mut session = DecodeSession::new(
        &f.plan,
        OperandSource::overlaid(&f.store, &edits),
        &backend,
        Box::new(RowKvState::default()),
    )
    .unwrap();
    let from_prefix: Vec<_> = [3, 17, 8, 0, 11]
        .iter()
        .map(|t| session.step(*t).unwrap().logits.unwrap())
        .collect();
    assert_eq!(
        before, from_prefix,
        "stored execution must match the completed candidate prefix"
    );
    drop(output);
    // Replaying only persisted artifacts reproduces the complete candidate.
    let replay_request = GptqRequest {
        component: request.component.clone(),
        continuation: request.continuation.clone(),
        bank: request.bank.clone(),
        sites: request
            .sites
            .iter()
            .map(|(id, input)| {
                let CalibrationInput::CaptureTo(path) = input else {
                    unreachable!()
                };
                (id.clone(), CalibrationInput::Existing(path.clone()))
            })
            .collect(),
    };
    let replay = f.out("replay");
    compile_representation_recipe(&f.src, &replay, &spec, Nvfp4Recipe::Gptq(&replay_request))
        .unwrap();
    assert_eq!(ci, candidate(&replay));
    std::fs::remove_dir_all(&artifacts).unwrap();
    assert!(!out.join(".represent-recipe-work").exists());
    let reopened = stored(&out);
    assert_eq!(before, logits(&f.plan, &reopened));
    assert_eq!(authority, read_candidate_evidence(&out).unwrap());
    assert_eq!(reopened.runtime_quantised(), 0);
    // Metadata and payload tampering cannot retain completed authority.
    let mut tampered = ci.clone();
    tampered.derivation.as_mut().unwrap().tensors[0].recipe = EncoderRecipe::nearest_v1();
    std::fs::write(
        out.join("candidate.json"),
        serde_json::to_vec(&tampered).unwrap(),
    )
    .unwrap();
    assert!(read_candidate_evidence(&out).is_err());
}

#[test]
fn cal12_explicit_nearest_is_legacy_nearest_without_calibration() {
    let f = Fixture::new();
    std::fs::remove_file(f.src.join("tokenizer.json")).unwrap();
    let a = f.out("default");
    let b = f.out("explicit");
    super::super::compile_representation(&f.src, &a, &RepresentSpec::nvfp4()).unwrap();
    compile_representation_recipe(&f.src, &b, &RepresentSpec::nvfp4(), Nvfp4Recipe::Nearest)
        .unwrap();
    assert_eq!(
        std::fs::read(a.join("candidate.json")).unwrap(),
        std::fs::read(b.join("candidate.json")).unwrap()
    );
    assert!(candidate(&a).derivation.is_none());
    assert_eq!(logits(&f.plan, &stored(&a)), logits(&f.plan, &stored(&b)));
}

#[test]
fn cal12_existing_artifacts_require_exact_dense_calibration_key_before_site_encoding() {
    let f = Fixture::new();
    let q = &f.plan.layers[0].attention.softmax().unwrap().q;
    let bank = f.bank();
    let prepared = PreparedCalibration::prepare(&f.plan, &f.store, 0, Projection::Query).unwrap();
    let path = f.out("artifact");
    let artifact = prepared
        .capture(&bank, StatisticKind::DenseGram, &f.continuation())
        .unwrap();
    artifact.write(&path).unwrap();
    let mut request = GptqRequest {
        component: "target".into(),
        continuation: f.continuation(),
        bank,
        sites: BTreeMap::from([(
            (q.object.clone(), q.tensor.clone()),
            CalibrationInput::Existing(path.clone()),
        )]),
    };
    compile_representation_recipe(
        &f.src,
        &f.out("valid"),
        &RepresentSpec::nvfp4(),
        Nvfp4Recipe::Gptq(&request),
    )
    .unwrap();
    let original = artifact.manifest();
    for mutation in 0..7 {
        let mut m = original.clone();
        match mutation {
            0 => m.key.candidate_prefix_sha256 = "1".repeat(64),
            1 => m.key.site.projection = Projection::Key,
            2 => m.key.bank_sha256 = "2".repeat(64),
            3 => m.key.statistic = StatisticKind::DiagonalSecondMoment,
            4 => m.key.execution_sha256 = "3".repeat(64),
            5 => m.key.population = Population::ReconstructionValidation,
            _ => m.key.samples += 1,
        }
        m.binding_sha256 = m.binding().unwrap();
        std::fs::write(path.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        let out = f.out(&format!("bad-{mutation}"));
        let err = compile_representation_recipe(
            &f.src,
            &out,
            &RepresentSpec::nvfp4(),
            Nvfp4Recipe::Gptq(&request),
        )
        .unwrap_err();
        assert!(err.to_string().contains("context mismatch"), "{err}");
        assert!(!out.join("candidate.json").exists());
        assert_eq!(
            std::fs::read_dir(out.join(".represent-recipe-work"))
                .unwrap()
                .count(),
            0
        );
    }
    // An actual valid diagonal artifact also refuses; no implicit dense upgrade.
    let diag = f.out("diagonal");
    prepared
        .capture(
            &request.bank,
            StatisticKind::DiagonalSecondMoment,
            &request.continuation,
        )
        .unwrap()
        .write(&diag)
        .unwrap();
    request
        .sites
        .values_mut()
        .for_each(|v| *v = CalibrationInput::Existing(diag.clone()));
    assert!(compile_representation_recipe(
        &f.src,
        &f.out("bad-diagonal"),
        &RepresentSpec::nvfp4(),
        Nvfp4Recipe::Gptq(&request)
    )
    .is_err());
}
