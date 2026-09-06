//! `vindex3 ops --realizations` — the prepared plan's admission, priced
//! from tensor tables, held against a budget, with no payload byte read.

use super::*;

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    (dir, out)
}

fn ops(container: &std::path::Path, budget_gib: Option<f64>) -> Vindex3Command {
    ops_binding(container, budget_gib, false)
}

fn ops_binding(container: &std::path::Path, budget_gib: Option<f64>, bind: bool) -> Vindex3Command {
    ops_full(container, budget_gib, None, bind)
}

fn ops_full(
    container: &std::path::Path,
    budget_gib: Option<f64>,
    bandwidth_gbs: Option<f64>,
    bind: bool,
) -> Vindex3Command {
    Vindex3Command::Ops(OpsArgs {
        container: container.to_path_buf(),
        component: "target".to_string(),
        layer: None,
        json: false,
        realizations: true,
        budget_gib,
        bind,
        bandwidth_gbs,
        target_tok_s: 20.0,
    })
}

/// A throughput budget no plan can meet refuses before I/O naming the
/// per-token deficit; a generous one passes.
#[test]
fn a_throughput_budget_is_held_per_token() {
    let (_dir, out) = encoded_fixture();
    run(ops_full(&out, Some(1024.0), Some(1_000.0), false)).expect("a TB/s budget passes");
    let err = run(ops_full(&out, Some(1024.0), Some(1e-9), false))
        .unwrap_err()
        .to_string();
    assert!(err.contains("within the budget"), "{err}");
}

/// `--bind` prepares the plan for real and reconciles what was bound
/// against what was declared — and does so only inside the budget.
#[test]
fn bind_reconciles_within_budget_and_is_never_reached_over_it() {
    let (_dir, out) = encoded_fixture();
    run(ops_binding(&out, Some(1024.0), true)).expect("the miniature binds and reconciles");
    let err = run(ops_binding(&out, Some(1e-9), true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("within the budget"), "{err}");
}

/// A closable fixture pins every planned operand and its declared working
/// set sits inside a generous budget; the verb exits clean.
#[test]
fn realizations_of_a_closed_plan_fit_a_generous_budget() {
    let (_dir, out) = encoded_fixture();
    run(ops(&out, Some(1024.0))).expect("a miniature plan fits a 1 TiB budget");
}

/// The machine's own memory is the default budget, and the miniature
/// fits it too.
#[test]
fn the_default_budget_is_the_machine() {
    let (_dir, out) = encoded_fixture();
    run(ops(&out, None)).expect("a miniature plan fits this machine");
}

/// A budget the declared working set cannot meet is a refusal BEFORE
/// any byte — named as such, non-zero exit.
#[test]
fn an_unmeetable_budget_refuses_before_io() {
    let (_dir, out) = encoded_fixture();
    let err = run(ops(&out, Some(1e-9))).unwrap_err().to_string();
    assert!(err.contains("within the budget"), "{err}");
    assert!(err.contains("nothing was read"), "{err}");
}

/// A store bound to a registry that knows none of the container's
/// representations refuses at selection — every operand named, nothing
/// read — and the verb reports that refusal rather than a budget.
#[test]
fn a_registry_without_the_representation_refuses_before_io() {
    use larql_vindex::format::vindex3::inspect::inspect_container;
    use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
    use larql_vindex::format::vindex3::opplan::plan_component_ops;
    use larql_vindex::format::vindex3::represent::codec::CodecRegistry;
    let (_dir, out) = encoded_fixture();
    let inspection = inspect_container(&out, false).unwrap();
    let plan = plan_component_ops(&inspection, &out, "target")
        .unwrap()
        .plan
        .expect("the fixture plans");
    let empty: &'static CodecRegistry = Box::leak(Box::new(CodecRegistry::new()));
    let store = OperandStore::open(&out, &inspection)
        .unwrap()
        .with_registry(empty);
    let budget = larql_vindex::format::vindex3::opplan::exec::accounting::ResidencyBudget::physical(
        1024 * 1024 * 1024 * 1024,
    );
    let err = super::super::realizations::report_through(&plan, &store, &budget)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no admissible preparation"), "{err}");
    assert!(err.contains("nothing was read"), "{err}");
}
