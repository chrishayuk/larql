//! No statement that succeeds loses a pending weight edit, and a weight-patched container survives the rest of
//! the lifecycle: COMPACT, the knowledge COMPILE, and a container that carries its own L0 knowledge.

use std::path::Path;

use larql_vindex::KnnStore;

use crate::support::*;

const LEDGER: &str = "weight_patches.json";
const KNN_STORE: &str = "knn_store.bin";
/// Where the test KNN entry sits, and how wide its key is. Nothing here queries it.
const KNN_LAYER: usize = 1;
const KNN_KEY_WIDTH: usize = 4;

fn compile(s: &mut larql_lql::Session, out: &Path) -> Vec<String> {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)))
}

fn compact(s: &mut larql_lql::Session, out: &Path) -> Result<Vec<String>, String> {
    try_run(s, &format!("COMPACT INTO VINDEX \"{}\";", q(out)))
}

/// Give `container` the L0 knowledge a knowledge COMPILE would have written into it.
fn carry_knowledge(container: &Path) {
    let mut store = KnnStore::default();
    store.add(
        KNN_LAYER,
        vec![1.0; KNN_KEY_WIDTH],
        7,
        "t".into(),
        "carried".into(),
        "r".into(),
        1.0,
    );
    store.save(&container.join(KNN_STORE)).unwrap();
}

#[test]
fn compact_refuses_while_a_weight_edit_is_pending_whatever_form_it_takes() {
    let fx = Fixture::new();
    let out = fx.path("compacted.vindex3");

    // Staged directly.
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    let err = compact(&mut s, &out).unwrap_err();
    assert!(err.contains("pending stored-weight op"), "{err}");
    assert!(!out.exists(), "a refused COMPACT writes nothing");

    // Carried by an applied patch.
    let vlp = save_patch(&fx, "edit.vlp", &COLS, DELTA);
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    let err = compact(&mut s, &out).unwrap_err();
    assert!(err.contains("pending stored-weight op"), "{err}");
    assert!(!out.exists());

    // With nothing pending it compacts.
    let mut s = session_on(&fx.container);
    compact(&mut s, &out).unwrap();
    assert!(is_coherent(&out));
}

#[test]
fn a_weight_patched_container_compacts_with_its_ledger() {
    let fx = Fixture::new();
    let patched = fx.path("patched.vindex3");
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    compile(&mut s, &patched);

    let compacted = fx.path("compacted.vindex3");
    let mut s = session_on(&patched);
    compact(&mut s, &compacted).unwrap();
    assert_eq!(ledger(&patched), ledger(&compacted));
    assert!(is_coherent(&compacted));
    assert_eq!(
        tensor_bytes(&patched, CANONICAL, TENSOR),
        tensor_bytes(&compacted, CANONICAL, TENSOR),
        "the edited weights are carried"
    );
}

#[test]
fn a_knowledge_compile_of_a_weight_patched_container_keeps_its_ledger() {
    let fx = Fixture::new();
    let patched = fx.path("patched.vindex3");
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    compile(&mut s, &patched);

    // Nothing pending on the patched container: the compile carries it whole.
    let again = fx.path("again.vindex3");
    let mut s = session_on(&patched);
    compile(&mut s, &again);
    assert!(again.join(LEDGER).exists());
    assert_eq!(ledger(&patched), ledger(&again));
    assert!(is_coherent(&again));
}

#[test]
fn a_container_that_carries_its_own_knowledge_still_takes_weight_edits() {
    let fx = Fixture::new();
    carry_knowledge(&fx.container);

    let out = fx.path("patched.vindex3");
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    let lines = compile(&mut s, &out);
    assert!(
        lines[0].contains("weight patch of 4 element(s)"),
        "{lines:?}"
    );
    assert!(is_coherent(&out));
    assert_eq!(
        std::fs::read(fx.container.join(KNN_STORE)).unwrap(),
        std::fs::read(out.join(KNN_STORE)).unwrap(),
        "the container's knowledge travels with the edited weights"
    );

    // And with nothing pending, that container compacts: its knowledge is not an edit.
    let compacted = fx.path("compacted.vindex3");
    let mut s = session_on(&fx.container);
    compact(&mut s, &compacted).unwrap();
    assert!(compacted.join(KNN_STORE).exists());
}

#[test]
fn diff_says_a_pending_weight_edit_is_not_part_of_the_comparison() {
    let fx = Fixture::new();
    let diff = format!("DIFF CURRENT \"{}\";", q(&fx.container));

    let mut s = session_on(&fx.container);
    let clean = run(&mut s, &diff);
    assert!(
        !clean.iter().any(|l| l.contains("pending COMPILE")),
        "{clean:?}"
    );

    run(&mut s, &update(&COLS, DELTA));
    let pending = run(&mut s, &diff);
    assert!(
        pending
            .iter()
            .any(|l| l.contains("1 stored-weight op(s) are pending COMPILE")),
        "{pending:?}"
    );
}
