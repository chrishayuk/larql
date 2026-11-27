use super::*;

#[test]
fn calibration_second_layer_falsifies_the_canonical_prefix_after_nvfp4_replacement() {
    let (_dir, plan, store) = fixture(false);
    let down = &plan.layers[0].ffn.as_ref().unwrap().dense().unwrap().down;
    let original = store.load(down).unwrap();
    let (rows, width) = (down.shape[0], down.shape[1]);
    let packed = larql_models::quant::nvfp4::quantize(&original, rows, width).unwrap();
    let mut decoded = vec![0.; original.len()];
    larql_models::quant::nvfp4::dequantize_into(&packed, rows, width, &mut decoded).unwrap();
    assert_ne!(original, decoded);
    let mut edits = OperandOverrides::new();
    for (index, values) in decoded.chunks_exact(width).enumerate() {
        edits.push(
            down,
            OperandEdit::Row {
                index,
                values: values.to_vec(),
            },
        );
    }
    let candidate = OperandSource::overlaid(&store, &edits);
    let bank = bank();
    let a = PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).unwrap();
    let b = PreparedCalibration::prepare(&plan, candidate, 1, Projection::Query).unwrap();
    let canonical = a
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    let captured = b
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    assert_eq!(
        canonical.manifest().key.source_image_sha256,
        captured.manifest().key.source_image_sha256
    );
    assert_ne!(
        canonical.manifest().key.candidate_prefix_sha256,
        captured.manifest().key.candidate_prefix_sha256
    );
    assert_ne!(
        canonical.values(),
        captured.values(),
        "second-layer fixture must distinguish the candidate prefix"
    );
    let actual = direct_rows(&plan, candidate, &bank, &ProductionBackend::new());
    assert_eq!(captured.values(), gram(&actual.attention));
    // The wrong-prefix mutant produces plausible canonical statistics, but
    // those artifacts cannot be admitted for the independently prepared candidate.
    let out = tempfile::tempdir().unwrap();
    let path = out.path().join("canonical");
    canonical.write(&path).unwrap();
    assert!(
        CalibrationArtifact::read(&path, &b.key(&bank, StatisticKind::DenseGram).unwrap()).is_err()
    );
    // The changed down projection cannot alter layer zero's entering Q input.
    let before_a = PreparedCalibration::prepare(&plan, &store, 0, Projection::Query)
        .unwrap()
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    let before_b = PreparedCalibration::prepare(&plan, candidate, 0, Projection::Query)
        .unwrap()
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    assert_eq!(before_a.values(), before_b.values());
}

#[test]
fn calibration_all_sites_match_actual_inputs_with_masks_resets_and_sliding_window() {
    let (_dir, plan, store) = fixture(true);
    let bank = bank();
    let rows = direct_rows(&plan, (&store).into(), &bank, &ProductionBackend::new());
    let mut prepared = PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).unwrap();
    for projection in [
        Projection::Query,
        Projection::Key,
        Projection::Value,
        Projection::Gate,
        Projection::Up,
        Projection::Down,
    ] {
        prepared.select_projection(projection).unwrap();
        let dense = prepared
            .capture(&bank, StatisticKind::DenseGram, &row(&plan))
            .unwrap();
        let diagonal = prepared
            .capture(&bank, StatisticKind::DiagonalSecondMoment, &row(&plan))
            .unwrap();
        let inputs = match projection {
            Projection::Query | Projection::Key | Projection::Value => &rows.attention,
            Projection::Gate | Projection::Up => &rows.ffn,
            Projection::Down => &rows.down,
        };
        assert_eq!(dense.values(), gram(inputs), "{projection:?}");
        let width = inputs[0].len();
        assert_eq!(
            diagonal.values(),
            (0..width)
                .map(|i| dense.values()[i * width + i])
                .collect::<Vec<_>>()
        );
        assert_eq!(dense.manifest().key.samples, 5);
        assert_eq!(
            dense.values(),
            prepared
                .capture(&bank, StatisticKind::DenseGram, &row(&plan))
                .unwrap()
                .values()
        );
    }
}

#[test]
fn calibration_down_tap_preserves_both_backends_and_shared_handle_forwarding() {
    tap_parity(&ReferenceBackend::new());
    tap_parity(&std::sync::Arc::new(ProductionBackend::new()));
}

#[test]
fn calibration_unsupported_backend_refuses_down_capture() {
    use crate::format::vindex3::opplan::exec::{
        backend::{FfnCall, WeightFormat},
        device::DevicePlanBackend,
    };
    let backend = DevicePlanBackend::new(
        larql_compute::CpuBackend,
        "no-down-capture",
        WeightFormat::F32,
    );
    let mut called = false;
    let error = backend
        .ffn_observed(
            FfnCall {
                x: &[1.],
                hidden: 1,
                intermediate: 1,
                gate: None,
                up: WeightSlice::F32(&[1.]),
                down: WeightSlice::F32(&[1.]),
                activation: larql_models::config::Activation::Silu,
                gate_policy: larql_models::ExpertGatePolicy::default(),
                up_bias: None,
                down_bias: None,
            },
            &mut |_| called = true,
        )
        .unwrap_err();
    assert!(error.to_string().contains("down-input capture unavailable"));
    assert!(!called);
}

#[test]
fn calibration_streaming_statistics_are_raw_uncentered_f64_sums() {
    let mut dense = statistic::Accumulator::new(StatisticKind::DenseGram, 3).unwrap();
    let mut diagonal = statistic::Accumulator::new(StatisticKind::DiagonalSecondMoment, 3).unwrap();
    for x in [[1., 2., 0.], [3., -1., 0.]] {
        dense.push(&x).unwrap();
        diagonal.push(&x).unwrap();
    }
    assert_eq!(dense.values, vec![10., -1., 0., -1., 5., 0., 0., 0., 0.]);
    assert_eq!(diagonal.values, vec![10., 5., 0.]);
    assert_eq!(dense.samples, 2);
    assert!(dense.push(&[1., 2.]).is_err());
    assert!(dense.push(&[1., f32::NAN, 0.]).is_err());
    assert!(statistic::Accumulator::new(StatisticKind::DenseGram, usize::MAX).is_err());
}

#[test]
fn calibration_artifact_roundtrip_binds_context_and_rejects_corruption() {
    let (_dir, plan, store) = fixture(true);
    let prepared = PreparedCalibration::prepare(&plan, &store, 1, Projection::Down).unwrap();
    let bank = bank();
    let key = prepared.key(&bank, StatisticKind::DenseGram).unwrap();
    let artifact = prepared
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("capture");
    artifact.write(&root).unwrap();
    let loaded = CalibrationArtifact::read(&root, &key).unwrap();
    assert_eq!(artifact.manifest(), loaded.manifest());
    assert_eq!(artifact.values(), loaded.values());
    assert!(artifact.write(&root).is_err());
    for field in 0..9 {
        let mut wrong = key.clone();
        match field {
            0 => wrong.source_image_sha256 = "b".repeat(64),
            1 => wrong.candidate_prefix_sha256 = "b".repeat(64),
            2 => wrong.site.width += 1,
            3 => wrong.site.layer = 0,
            4 => wrong.bank_sha256 = "b".repeat(64),
            5 => wrong.tokenizer_sha256 = "b".repeat(64),
            6 => wrong.population = Population::ReconstructionValidation,
            7 => wrong.statistic = StatisticKind::DiagonalSecondMoment,
            _ => wrong.execution_sha256 = "b".repeat(64),
        }
        assert!(
            CalibrationArtifact::read(&root, &wrong).is_err(),
            "context field {field}"
        );
    }
    let manifest_path = root.join("manifest.json");
    let original = std::fs::read(&manifest_path).unwrap();
    for (field, value) in [
        ("schema", "future/v99"),
        ("numeric", "f32-le"),
        ("binding_sha256", "bad"),
    ] {
        let mut json: serde_json::Value = serde_json::from_slice(&original).unwrap();
        json[field] = value.into();
        std::fs::write(&manifest_path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(CalibrationArtifact::read(&root, &key).is_err());
    }
    std::fs::write(&manifest_path, original).unwrap();
    let payload = root.join("statistics.f64");
    let mut bytes = std::fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&payload, &bytes).unwrap();
    assert!(CalibrationArtifact::read(&root, &key).is_err());
    std::fs::write(&payload, &bytes[..bytes.len() - 1]).unwrap();
    assert!(CalibrationArtifact::read(&root, &key).is_err());
}

#[test]
fn calibration_bank_digest_covers_order_masks_boundaries_and_tokenizer() {
    let original = bank();
    for field in 0..5 {
        let mut changed = original.clone();
        match field {
            0 => changed.sequences[0].tokens.swap(0, 1),
            1 => changed.sequences[0].include[1] = true,
            2 => changed.sequences.reverse(),
            3 => changed.tokenizer_sha256 = "b".repeat(64),
            _ => changed.population = Population::ReconstructionValidation,
        }
        assert_ne!(original.sha256().unwrap(), changed.sha256().unwrap());
    }
    assert!(CalibrationBank::new(
        "empty".into(),
        "a".repeat(64),
        Population::Calibration,
        vec![]
    )
    .is_err());
    assert!(CalibrationBank::new(
        "bad mask".into(),
        "a".repeat(64),
        Population::Calibration,
        vec![CalibrationSequence {
            tokens: vec![1],
            include: vec![]
        }]
    )
    .is_err());
}

#[test]
fn calibration_reuses_sealed_token_banks_and_rejects_wrong_tokenizers() {
    use crate::format::vindex3::represent::token_bank::{import_ids, TokenBank};
    let (container, plan, store) = fixture(false);
    let tmp = tempfile::tempdir().unwrap();
    let ids = tmp.path().join("ids.json");
    std::fs::write(&ids, serde_json::to_vec(&serde_json::json!({
        "bank": "cal1-fixture", "samples": [{"id": "a", "category": "mechanism", "ids": [3, 17, 8]}]
    })).unwrap()).unwrap();
    let root = tmp.path().join("bank");
    import_ids(&ids, &container.path().join("tokenizer.json"), &root).unwrap();
    let sealed = TokenBank::open(&root).unwrap();
    let bank = CalibrationBank::from_token_bank(
        &sealed,
        vec![vec![true, false, true]],
        Population::Calibration,
    )
    .unwrap();
    let capture = PreparedCalibration::prepare(&plan, &store, 0, Projection::Query).unwrap();
    let artifact = capture
        .capture(&bank, StatisticKind::DiagonalSecondMoment, &row(&plan))
        .unwrap();
    assert_eq!(
        artifact.manifest().key.token_bank_id.as_deref(),
        Some(sealed.manifest().bank_id.as_str())
    );
    assert_eq!(artifact.manifest().key.samples, 2);
    let mut wrong = bank.clone();
    wrong.tokenizer_sha256 = "b".repeat(64);
    assert!(capture
        .capture(&wrong, StatisticKind::DiagonalSecondMoment, &row(&plan))
        .err()
        .unwrap()
        .to_string()
        .contains("tokenizer"));
    assert!(CalibrationBank::from_token_bank(&sealed, vec![], Population::Calibration).is_err());
    std::fs::write(root.join("seq-000.u32"), [0u8; 12]).unwrap();
    assert!(CalibrationBank::from_token_bank(
        &sealed,
        vec![vec![true; 3]],
        Population::Calibration
    )
    .is_err());
}

#[test]
fn calibration_prefix_binds_glue_and_excludes_later_layers() {
    let (_dir, plan, store) = fixture(true);
    let bank = bank();
    let base = PreparedCalibration::prepare(&plan, &store, 0, Projection::Query)
        .unwrap()
        .key(&bank, StatisticKind::DenseGram)
        .unwrap();
    let mut edits = OperandOverrides::new();
    let late = &plan.layers[1].ffn.as_ref().unwrap().dense().unwrap().down;
    edits.push(
        late,
        OperandEdit::Row {
            index: 0,
            values: vec![0.; late.shape[1]],
        },
    );
    let later = PreparedCalibration::prepare(
        &plan,
        OperandSource::overlaid(&store, &edits),
        0,
        Projection::Query,
    )
    .unwrap()
    .key(&bank, StatisticKind::DenseGram)
    .unwrap();
    assert_eq!(base, later);
    // Bind glue through a plan change as well: epsilon is an execution fact.
    let mut changed = plan.clone();
    changed.layers[0].pre_attention_norm.as_mut().unwrap().eps *= 10.;
    let other = PreparedCalibration::prepare(&changed, &store, 0, Projection::Query)
        .unwrap()
        .key(&bank, StatisticKind::DenseGram)
        .unwrap();
    assert_ne!(base.candidate_prefix_sha256, other.candidate_prefix_sha256);
    assert!(PreparedCalibration::prepare(&plan, &store, 99, Projection::Down).is_err());
    let mut no_ffn = plan.clone();
    no_ffn.layers[0].ffn = None;
    assert!(PreparedCalibration::prepare(&no_ffn, &store, 0, Projection::Down).is_err());
}

#[test]
fn calibration_prefix_digest_includes_bias_contents_not_just_matrices() {
    let capture = |perturb| {
        let weights = tempfile::tempdir().unwrap();
        fixtures::miniature_glimmer_with(
            weights.path(),
            fixtures::MiniatureExtras {
                attention_bias: true,
                sinks: false,
                perturb,
            },
        );
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
        PreparedCalibration::prepare(&plan, &store, 1, Projection::Query)
            .unwrap()
            .capture(&bank(), StatisticKind::DiagonalSecondMoment, &row(&plan))
            .unwrap()
    };
    let a = capture(None);
    let b = capture(Some("self_attn.o_proj.bias"));
    assert_ne!(
        a.manifest().key.source_image_sha256,
        b.manifest().key.source_image_sha256
    );
    assert_ne!(a.values(), b.values());
}

#[test]
fn calibration_rejects_invalid_statistic_values_even_with_a_matching_digest() {
    let (_dir, plan, store) = fixture(true);
    let key = PreparedCalibration::prepare(&plan, &store, 0, Projection::Query)
        .unwrap()
        .key(&bank(), StatisticKind::DenseGram)
        .unwrap();
    let width = key.site.width;
    for mutation in 0..4 {
        let mut values = vec![0.; width * width];
        match mutation {
            0 => values[0] = f64::INFINITY,
            1 => values[0] = -1.,
            2 => values[1] = 1.,
            _ => {
                values[1] = 1.;
                values[width] = 1.;
            }
        }
        assert!(CalibrationArtifact::new(key.clone(), values).is_err());
    }
}

/// A boundary qualification over real weights, not a quality or performance
/// experiment. Uses fresh text, never the R4/Q-bank admission populations.
#[test]
#[ignore = "requires CAL1_CONTAINER pointing to a dense softmax VINDEX3 container"]
fn calibration_real_layer_inputs_match_independent_decode_observation() {
    let root =
        std::path::PathBuf::from(std::env::var("CAL1_CONTAINER").expect("set CAL1_CONTAINER"));
    let inspection = inspect_container(&root, false).unwrap();
    let mut plan = plan_component_ops(&inspection, &root, "target")
        .unwrap()
        .plan
        .unwrap();
    plan.layers.truncate(2);
    plan.final_norm = None;
    plan.output = None;
    let store = OperandStore::open(&root, &inspection).unwrap();
    let tokenizer_bytes = std::fs::read(root.join("tokenizer.json")).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    let tokens = tokenizer
        .encode(
            "Calibration checks the inputs consumed by each projection.",
            true,
        )
        .unwrap()
        .get_ids()
        .to_vec();
    let bank = CalibrationBank::new(
        "cal1-boundary-qualification".into(),
        digest(&tokenizer_bytes),
        Population::Calibration,
        vec![CalibrationSequence {
            include: vec![true; tokens.len()],
            tokens,
        }],
    )
    .unwrap();
    let rows = direct_rows(&plan, (&store).into(), &bank, &ProductionBackend::new());
    let mut prepared = PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).unwrap();
    for projection in [
        Projection::Query,
        Projection::Key,
        Projection::Value,
        Projection::Gate,
        Projection::Up,
        Projection::Down,
    ] {
        prepared.select_projection(projection).unwrap();
        let actual = prepared
            .capture(&bank, StatisticKind::DiagonalSecondMoment, &row(&plan))
            .unwrap();
        let inputs = match projection {
            Projection::Query | Projection::Key | Projection::Value => &rows.attention,
            Projection::Gate | Projection::Up => &rows.ffn,
            Projection::Down => &rows.down,
        };
        let expected: Vec<f64> = (0..inputs[0].len())
            .map(|i| inputs.iter().map(|x| f64::from(x[i]).powi(2)).sum())
            .collect();
        assert_eq!(actual.values(), expected);
        eprintln!(
            "CAL1 boundary {projection:?}: {} positions, width {}, prefix {}",
            actual.manifest().key.samples,
            actual.manifest().key.site.width,
            actual.manifest().key.candidate_prefix_sha256
        );
    }
}

#[test]
fn calibration_builds_one_fresh_state_per_sequence_from_the_callers_selection() {
    let (_dir, plan, store) = fixture(false);
    let prepared = PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).unwrap();
    let bank = bank();
    let built = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Box::new(CountingRow(built.clone())))
        .unwrap();
    let counting = registry
        .select(
            &CountingRow(built.clone()).identity(),
            &ContinuationConfig::empty(),
            &plan_continuation_geometry(&plan).unwrap(),
        )
        .unwrap();
    let selected = prepared
        .capture(&bank, StatisticKind::DenseGram, &counting)
        .unwrap();
    assert_eq!(
        built.load(std::sync::atomic::Ordering::SeqCst),
        bank.sequences.len(),
        "one fresh state per sequence, each from the caller's selection"
    );
    let reference = prepared
        .capture(&bank, StatisticKind::DenseGram, &row(&plan))
        .unwrap();
    assert_eq!(selected.values(), reference.values());
}

/// Capture authority is CAL-1.1's declared scope: a prefix whose layers are
/// not contiguous dense softmax is refused before any image is sealed, even
/// when the target site itself resolves.
#[test]
fn calibration_refuses_prefixes_outside_the_dense_softmax_scope() {
    let (_dir, plan, store) = fixture(false);
    // The target layer 1 Query site still resolves; only the prefix is out of scope.
    let mut no_ffn = plan.clone();
    no_ffn.layers[0].ffn = None;
    let mut discontiguous = plan.clone();
    discontiguous.layers[0].layer = plan.layers.len();
    for changed in [no_ffn, discontiguous] {
        let error = PreparedCalibration::prepare(&changed, &store, 1, Projection::Query)
            .err()
            .expect("an out-of-scope prefix must refuse");
        assert!(
            error
                .to_string()
                .contains("contiguous dense softmax single-stream prefixes only"),
            "{error}"
        );
    }
    // The unchanged plan is inside scope, so the refusal is the prefix's.
    assert!(PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).is_ok());
}

#[test]
fn calibration_site_resolution_refuses_absent_gates_and_non_matrix_operands() {
    let (_dir, plan, _store) = fixture(false);
    let refusal = |plan: &ComponentOpPlan, projection| {
        CalibrationSite::resolve(plan, 0, projection)
            .unwrap_err()
            .to_string()
    };
    let mut no_gate = plan.clone();
    let Some(LayerFfn::Dense(ffn)) = &mut no_gate.layers[0].ffn else {
        panic!("fixture layer 0 is dense");
    };
    ffn.gate = None;
    assert!(refusal(&no_gate, Projection::Gate).contains("no gate projection"));
    // Up still resolves without a gate: the refusal names the missing operand only.
    assert_eq!(
        CalibrationSite::resolve(&no_gate, 0, Projection::Up)
            .unwrap()
            .boundary,
        Boundary::FfnInput
    );
    let mut vector = plan.clone();
    let q = &mut vector.layers[0].attention.softmax_mut().unwrap().q;
    q.shape = vec![q.shape.iter().product()];
    assert!(refusal(&vector, Projection::Query).contains("site is not a matrix"));
    let mut empty = plan.clone();
    empty.layers[0].attention.softmax_mut().unwrap().k.shape[0] = 0;
    assert!(refusal(&empty, Projection::Key).contains("empty site matrix"));
}
