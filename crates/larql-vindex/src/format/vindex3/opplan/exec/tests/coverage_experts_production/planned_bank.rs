//! Rung 3a over the routed miniature: a packed bank is two expert-slice
//! operations requiring row access, a per-expert bank is a set of whole
//! projections, and the production loader's census holds exactly the bytes
//! the view lists for the FFN site. Lives beside the routed fixture because
//! that fixture is scoped here.

use super::fixture::{bf16_carrier_store, routed_fixture, BF16_SUFFIX, BLOCKS_SUFFIX, EXPERTS};
use crate::format::vindex3::opplan::exec::backend::MatrixClass;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{ExpertBank, LayerFfn, OperandRef, SharedExpertOp};
use crate::format::vindex3::represent::codec::RequiredAccess;

const F32_WIDTH: usize = std::mem::size_of::<f32>();

#[test]
fn a_packed_bank_is_two_row_random_slices_and_the_census_agrees() {
    let fixture = routed_fixture();
    let plan = &fixture.plan;
    let slices: Vec<_> = plan
        .planned_operands()
        .into_iter()
        .filter(|p| p.operation == Operation::ExpertBankSlice)
        .collect();
    let routed_layers = plan
        .layers
        .iter()
        .filter(|l| l.ffn.as_ref().is_some_and(|f| f.routed().is_some()))
        .count();
    assert!(routed_layers > 0);
    assert_eq!(
        slices.len(),
        2 * routed_layers,
        "gate/up and down per routed layer"
    );
    assert!(slices.iter().all(|s| s.access == RequiredAccess::RowRandom));
    // The bank's stored operands, exactly — never the router or a bias.
    let ExpertBank::Packed { gate_up, down } = &fixture.op.bank else {
        panic!("the miniature packs its bank");
    };
    let named: Vec<&str> = slices.iter().map(|s| s.operand.tensor.as_str()).collect();
    assert!(named.contains(&gate_up.weights.tensor.as_str()));
    assert!(named.contains(&down.weights.tensor.as_str()));
    assert!(!named.contains(&fixture.op.router.tensor.as_str()));

    let census = PreparedOperands::load(
        plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
    .residency_census();
    let ffn_bytes: usize = plan
        .planned_operands()
        .iter()
        .filter(|p| {
            matches!(
                p.operation,
                Operation::ExpertBankSlice | Operation::Project(MatrixClass::FfnProjection)
            )
        })
        .map(|p| p.elements() * F32_WIDTH)
        .sum();
    assert_eq!(census.ffn.widened_f32 + census.ffn.compact, ffn_bytes);
}

#[test]
fn a_per_expert_bank_is_a_set_of_whole_projections() {
    let fixture = routed_fixture();
    let mut plan = fixture.plan.clone();
    let synthetic = |name: &str| OperandRef {
        object: fixture.op.router.object.clone(),
        tensor: name.to_string(),
        dtype: "F32".into(),
        shape: vec![8, 4],
    };
    let per_expert = ExpertBank::PerExpert {
        gate: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.gate")))
            .collect(),
        up: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.up")))
            .collect(),
        down: (0..EXPERTS)
            .map(|e| synthetic(&format!("experts.{e}.down")))
            .collect(),
    };
    let mut op = fixture.op.clone();
    op.bank = per_expert;
    plan.layers[0].ffn = Some(LayerFfn::Routed(Box::new(op)));
    let listed: Vec<_> = plan
        .planned_operands()
        .into_iter()
        .filter(|p| p.operand.tensor.starts_with("experts."))
        .collect();
    assert_eq!(listed.len(), 3 * EXPERTS);
    assert!(listed.iter().all(
        |p| p.operation == Operation::Project(MatrixClass::FfnProjection)
            && p.access == RequiredAccess::Sequential
    ));
}

/// The plan executes a shared expert beside the routed ones, so the view
/// lists its three projections under their own operation. No backend
/// binds them through the prepared plan yet, and the plan is REFUSED at
/// preparation — before any byte is read — rather than prepared without
/// its shared expert (rung 3b's hard invariant). When a CPU realization
/// for the shared expert arrives (3d), this is the test that changes.
#[test]
fn a_plan_with_a_shared_expert_is_refused_before_any_byte_is_read() {
    let fixture = routed_fixture();
    let mut plan = fixture.plan.clone();
    let mut op = fixture.op.clone();
    let synthetic = |name: &str, shape: Vec<usize>| OperandRef {
        object: op.router.object.clone(),
        tensor: name.to_string(),
        dtype: "F32".into(),
        shape,
    };
    let hidden = op.router.shape[1];
    let inter = 16;
    op.shared = Some(SharedExpertOp {
        intermediate_size: inter,
        activation: op.activation,
        gate_policy: op.gate_policy,
        gate: synthetic("shared.gate", vec![inter, hidden]),
        up: synthetic("shared.up", vec![inter, hidden]),
        down: synthetic("shared.down", vec![hidden, inter]),
        branch_gate: None,
    });
    plan.layers[0].ffn = Some(LayerFfn::Routed(Box::new(op)));
    let planned = plan.planned_operands();
    let shared: Vec<_> = planned
        .iter()
        .filter(|p| p.operand.tensor.starts_with("shared."))
        .collect();
    assert_eq!(shared.len(), 3);
    assert!(shared
        .iter()
        .all(|p| p.operation == Operation::SharedExpertProject
            && p.access == RequiredAccess::Sequential));

    // The synthetic operands are not in the container, so the selector
    // skips them (the loader would refuse them by name); the refusal has
    // to come from a shared expert the container DOES hold. Point the
    // shared projections at the dense-shaped bf16 copies the carrier
    // stores, which exist and are registered.
    let (_dir, _container, carrier) = bf16_carrier_store();
    let mut op = fixture.op.clone();
    let ExpertBank::Packed { gate_up, down } = &op.bank else {
        panic!("packed");
    };
    let existing = |projection: &crate::format::vindex3::opplan::PackedProjection| OperandRef {
        object: projection.weights.object.clone(),
        tensor: projection
            .weights
            .tensor
            .replace(BLOCKS_SUFFIX, BF16_SUFFIX),
        dtype: "BF16".into(),
        shape: projection.weights.shape.clone(),
    };
    op.shared = Some(SharedExpertOp {
        intermediate_size: inter,
        activation: op.activation,
        gate_policy: op.gate_policy,
        gate: existing(gate_up),
        up: existing(gate_up),
        down: existing(down),
        branch_gate: None,
    });
    plan.layers[0].ffn = Some(LayerFfn::Routed(Box::new(op)));
    let before = carrier.load_count();
    let err = PreparedOperands::load(
        &plan,
        &carrier,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("a plan with a shared expert has no realization on the CPU path");
    assert!(err.contains("missing realization"), "{err}");
    assert!(err.contains("shared-expert-project"), "{err}");
    assert_eq!(
        carrier.load_count(),
        before,
        "refused before any byte was read"
    );
}

/// Rung 3c over the routed miniature: two bank pins reconcile with the
/// per-expert objects the loader bound (one operand, `EXPERTS` objects),
/// and the bank's declared residency is the widened image.
#[test]
fn a_packed_bank_s_pin_reconciles_with_its_per_expert_objects() {
    use crate::format::vindex3::opplan::exec::accounting::BlockGeometry;
    use crate::format::vindex3::opplan::exec::backend::WeightFormat;
    let fixture = routed_fixture();
    let ops = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let done = ops
        .reconcile(&fixture.plan, (&fixture.store).into())
        .unwrap();
    assert_eq!(done.matched, ops.realizations().len());
    let observed = ops.bound(&fixture.plan).unwrap();
    let banks: Vec<_> = observed
        .iter()
        .filter(|o| o.operation == Operation::ExpertBankSlice)
        .collect();
    assert!(!banks.is_empty());
    for bank in &banks {
        assert_eq!(bank.format, WeightFormat::F32);
        assert_eq!(bank.allocations, 0, "widened experts are exact vectors");
    }
    let expected = ops.expectations((&fixture.store).into(), BlockGeometry::executor());
    for e in expected
        .iter()
        .filter(|e| e.operation == Operation::ExpertBankSlice)
    {
        assert_eq!(
            e.declared_resident,
            e.logical_elements as u64 * F32_WIDTH as u64
        );
        assert_eq!(
            e.staging, 0,
            "the bank is widened per expert straight into residency"
        );
        assert!(
            e.stored_bytes > 0 && e.stored_bytes < e.declared_resident,
            "an MXFP4 bank is stored compact"
        );
    }
}

/// Rung 3c: a bank stored under a dialect no codec claims is prepared
/// with no provider; a registry that later claims the label is a changed
/// provider, and the image is invalidated rather than executed. And a
/// plan whose FFN is a different program from the prepared one is refused
/// by the pairing.
#[test]
fn an_unregistered_dialect_that_gains_a_provider_invalidates_the_preparation() {
    use super::super::accounting::ProviderStub;
    use crate::format::vindex3::opplan::exec::operands::OperandStore;
    use crate::format::vindex3::represent::codec::CodecRegistry;
    let fixture = routed_fixture();
    let ops = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let providers = ops.providers();
    let u8 = providers
        .iter()
        .find(|(label, _)| label == "U8")
        .expect("the MXFP4 bank is stored as U8");
    assert!(u8.1.is_none(), "no codec claims the dialect");
    ops.ensure_providers_in(CodecRegistry::builtin()).unwrap();
    // Every shipped codec, so the only change is the dialect gaining a
    // claimant; a registry with just the stub would mismatch on F32 first.
    use crate::format::vindex3::represent::codec::codecs::{
        bf16_zlib, float, kquant, mxfp4, nvfp4,
    };
    let claimed = CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .and_then(|r| r.register(Box::new(float::F32)))
        .and_then(|r| r.register(Box::new(kquant::Q4_K)))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .and_then(|r| r.register(Box::new(bf16_zlib::BF16_ZLIB)))
        .and_then(|r| r.register(Box::new(ProviderStub { label: "U8" })))
        .unwrap();
    let err = ops.ensure_providers_in(&claimed).unwrap_err().to_string();
    assert!(
        err.contains("`U8`") && err.contains("no registered codec") && err.contains("stub-U8 r7"),
        "{err}"
    );

    // The dense fixture's prepared image, held against this routed plan:
    // same attention program, different FFN program.
    let dense = {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        crate::format::vindex3::fixtures::encode_fixture_container(
            crate::format::vindex3::fixtures::dense_f32_model,
            src.path(),
            container.path(),
            "dense",
        );
        let inspection =
            crate::format::vindex3::inspect::inspect_container(container.path(), false).unwrap();
        let plan = crate::format::vindex3::opplan::plan_component_ops(
            &inspection,
            container.path(),
            "target",
        )
        .unwrap()
        .plan
        .unwrap();
        let store = OperandStore::open(container.path(), &inspection).unwrap();
        (src, container, plan, store)
    };
    let dense_ops = PreparedOperands::load(
        &dense.2,
        &dense.3,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let mut mixed = dense.2.clone();
    mixed.layers[0].ffn = Some(LayerFfn::Routed(Box::new(fixture.op.clone())));
    let err = dense_ops.bound(&mixed).unwrap_err().to_string();
    assert!(err.contains("different programs"), "{err}");
}
