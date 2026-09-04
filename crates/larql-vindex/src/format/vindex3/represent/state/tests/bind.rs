//! **4b-d: can this surface be priced authoritatively at all?**
//!
//! One question, and the tests are the whole contract:
//!
//! ```text
//! every surface tensor present exactly once  → READY
//! one missing                                → MISSING, that exact identity
//! several missing                            → MISSING, the complete list
//! stored facts the surface does not name     → neither satisfy nor damage
//! facts from another container               → refused as FOREIGN, not missing
//! an alias                                   → two entries, two required prices
//! ```
//!
//! No cost is computed here and none may be: 4b-e's `Footprint` takes a
//! `BoundPhysicalAccounting` and is total because this file resolved
//! every absence first.

use super::super::super::compiler::{read_source_identity, SourceIdentity};
use super::super::super::policy::Role;
use super::super::accounting::PhysicalAccountingFacts;
use super::super::accounting::{read_source_storage, TensorIdentity};
use super::super::bind::AccountingBindError;
use super::super::surface::{SurfaceTensor, TensorSurface};
use super::container;

/// A container's identity and storage facts, together.
fn read(container: &std::path::Path) -> (SourceIdentity, PhysicalAccountingFacts) {
    let identity = read_source_identity(container).expect("a container identity");
    let facts = read_source_storage(container, &identity).expect("its storage facts");
    (identity, facts)
}

/// The surface REPRESENT would enumerate if it saw exactly what the
/// container stores.
///
/// Built FROM the facts on purpose: the READY case is the control, and
/// every test below reaches its case by perturbing this one thing. The
/// role and shape are placeholders — binding reads neither, which
/// `binding_is_blind_to_role_and_shape` asserts rather than assumes.
fn surface_of(facts: &PhysicalAccountingFacts) -> TensorSurface {
    TensorSurface::new(
        facts
            .tensors()
            .map(|(id, _)| SurfaceTensor::new(&id.object, &id.tensor, Role::Unknown, vec![1])),
    )
    .expect("the facts key one entry per tensor")
}

fn missing_of(err: AccountingBindError) -> Vec<TensorIdentity> {
    match err {
        AccountingBindError::Incomplete(incomplete) => incomplete.missing,
        other => panic!("expected an incomplete bind, got {other}"),
    }
}

// ------------------------------------------------------------- READY

#[test]
fn a_surface_the_container_stores_entirely_binds() {
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    assert!(surface.len() > 1, "a one-tensor surface would prove little");

    let bound = facts.bind(&model, &surface).expect("READY");
    assert_eq!(bound.len(), surface.len(), "one price per surface tensor");
    assert_eq!(bound.surface_identity(), surface.identity());
    assert_eq!(bound.source(), model.semantic_digest());
    assert!(!bound.is_empty());

    // Every pairing is the price of THAT tensor, checked against the
    // facts independently of the positional zip that produced it.
    for (tensor, price) in bound.prices_for(&surface).expect("the bound surface") {
        let id = TensorIdentity::new(&tensor.object, &tensor.tensor);
        assert_eq!(
            Some(price),
            facts.get(&id),
            "{id} is paired with its own price"
        );
    }
}

#[test]
fn binding_is_blind_to_role_and_shape() {
    // A reclassified role is a different search problem (1a) and a
    // shape is not what anything is priced by (4b-c). Neither may
    // change whether a model can be priced at all.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let plain = surface_of(&facts);
    let reclassified = TensorSurface::new(
        plain
            .entries()
            .iter()
            .map(|t| SurfaceTensor::new(&t.object, &t.tensor, Role::ExpertWeight, vec![9, 9, 9])),
    )
    .expect("distinct tensors");

    assert_ne!(
        plain.identity(),
        reclassified.identity(),
        "the surfaces differ, which is what makes this worth asserting"
    );
    let a = facts.bind(&model, &plain).expect("READY");
    let b = facts.bind(&model, &reclassified).expect("READY");
    assert_eq!(a.len(), b.len());
    assert_ne!(a.surface_identity(), b.surface_identity());
}

// ----------------------------------------------------------- MISSING

#[test]
fn one_surface_tensor_the_container_does_not_store_is_named_exactly() {
    let container = container::dense();
    let (model, facts) = read(container.path());
    let mut entries = surface_of(&facts).entries().to_vec();
    entries.push(SurfaceTensor::new(
        "target.embedding",
        "a_tensor_the_container_does_not_store",
        Role::Unknown,
        vec![1],
    ));
    let surface = TensorSurface::new(entries).expect("distinct tensors");

    let err = facts.bind(&model, &surface).expect_err("MISSING");
    assert!(
        err.to_string()
            .contains("target.embedding/a_tensor_the_container_does_not_store"),
        "the reported error names the tensor, not just the count: {err}"
    );
    assert_eq!(
        missing_of(err),
        vec![TensorIdentity::new(
            "target.embedding",
            "a_tensor_the_container_does_not_store"
        )],
        "the exact identity, and only it"
    );
}

#[test]
fn several_missing_tensors_come_back_as_one_complete_ordered_list() {
    // The whole gap, not the first instance of it, and the same list on
    // every machine — a caller reporting "add these" must not have to
    // re-run to discover the second one.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let mut entries = surface_of(&facts).entries().to_vec();
    for (object, tensor) in [
        ("target.zulu", "w"),
        ("target.alpha", "w"),
        ("target.alpha", "b"),
    ] {
        entries.push(SurfaceTensor::new(object, tensor, Role::Unknown, vec![1]));
    }
    let surface = TensorSurface::new(entries).expect("distinct tensors");

    assert_eq!(
        missing_of(facts.bind(&model, &surface).expect_err("MISSING")),
        vec![
            TensorIdentity::new("target.alpha", "b"),
            TensorIdentity::new("target.alpha", "w"),
            TensorIdentity::new("target.zulu", "w"),
        ]
    );
}

#[test]
fn a_stored_tensor_the_surface_does_not_name_neither_satisfies_nor_damages() {
    // The asymmetry that matters. An extra stored fact must not stand
    // in for a missing one, and must not make a complete surface fail.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let full = surface_of(&facts);

    // A strict subset still binds, and prices only what it names.
    let mut entries = full.entries().to_vec();
    let dropped = entries.pop().expect("more than one tensor");
    let subset = TensorSurface::new(entries).expect("distinct tensors");
    let bound = facts.bind(&model, &subset).expect("READY");
    assert_eq!(bound.len(), subset.len());
    assert_eq!(bound.len(), full.len() - 1, "the extra fact is not priced");
    assert!(
        bound
            .tensors()
            .all(|(id, _)| id != &TensorIdentity::new(&dropped.object, &dropped.tensor)),
        "and does not appear"
    );

    // And with one genuinely absent tensor added, the many extras that
    // the subset leaves unnamed do not paper over it.
    let mut entries = subset.entries().to_vec();
    entries.push(SurfaceTensor::new(
        "target.absent",
        "w",
        Role::Unknown,
        vec![1],
    ));
    let surface = TensorSurface::new(entries).expect("distinct tensors");
    assert_eq!(
        missing_of(facts.bind(&model, &surface).expect_err("MISSING")),
        vec![TensorIdentity::new("target.absent", "w")]
    );
}

// -------------------------------------------------------- the source

#[test]
fn facts_from_another_container_are_foreign_and_not_merely_incomplete() {
    // Re-reading fixes one of these and cannot fix the other, so they
    // are not the same answer. Reported as FOREIGN even though every
    // surface tensor is also, incidentally, unpriceable.
    let a = container::dense();
    let b = container::glimmer();
    let (_, facts) = read(a.path());
    let (other_model, other_facts) = read(b.path());
    let surface = surface_of(&other_facts);

    match facts.bind(&other_model, &surface) {
        Err(AccountingBindError::ForeignSource { facts: f, model: m }) => {
            assert_ne!(f, m);
            assert!(format!(
                "{}",
                AccountingBindError::ForeignSource { facts: f, model: m }
            )
            .contains("pricing one model's surface from another's storage"));
        }
        other => panic!("expected a foreign source, got {other:?}"),
    }
}

#[test]
fn a_reserialised_container_still_binds() {
    // 4b-b2's relation, carried through the bind: re-exporting the
    // index is a different FILE and the same source, so a surface that
    // bound before binds after.
    let container = container::dense();
    let (before, facts) = read(container.path());
    let surface = surface_of(&facts);
    facts.bind(&before, &surface).expect("READY");

    container::reserialise(container.path());
    let after = read_source_identity(container.path()).expect("identity");
    assert_ne!(before.artifact, after.artifact, "a different file");
    facts.bind(&after, &surface).expect("and the same source");
}

// ------------------------------------------------------------ aliases

#[test]
fn an_alias_is_two_entries_and_two_required_prices() {
    // A tied embedding and output head are one payload and two objects.
    // The surface enumerates both, so both must be priced — one stored
    // fact satisfying two enumerated tensors is exactly the merge this
    // keying prevents.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let stored = surface_of(&facts);
    let alias = stored.entries().first().expect("a tensor").clone();

    let mut entries = stored.entries().to_vec();
    entries.push(SurfaceTensor::new(
        "target.tied_head",
        &alias.tensor,
        Role::OutputHead,
        alias.shape.clone(),
    ));
    let surface = TensorSurface::new(entries).expect("a second object, same tensor name");

    assert_eq!(
        missing_of(facts.bind(&model, &surface).expect_err("MISSING")),
        vec![TensorIdentity::new("target.tied_head", &alias.tensor)],
        "the alias is required on its own account, not satisfied by its twin"
    );
}

// ----------------------------------------------- what a proof is good for

#[test]
fn a_bound_accounting_refuses_a_surface_it_did_not_bind() {
    // The one check `prices_for` makes, standing in for the per-tensor
    // ones. Answering over another population would hand back prices
    // paired with the wrong tensors.
    let container = container::glimmer();
    let (model, facts) = read(container.path());
    let surface = surface_of(&facts);
    let bound = facts.bind(&model, &surface).expect("READY");

    let mut entries = surface.entries().to_vec();
    entries.pop();
    let narrower = TensorSurface::new(entries).expect("distinct tensors");

    match bound.prices_for(&narrower) {
        Err(AccountingBindError::ForeignSurface { bound: b, asked }) => {
            assert_eq!(b, surface.identity());
            assert_eq!(asked, narrower.identity());
            let err = AccountingBindError::ForeignSurface { bound: b, asked };
            assert!(
                err.to_string()
                    .contains("completeness was proved for one population"),
                "{err}"
            );
        }
        other => panic!("expected a foreign surface, got {other:?}"),
    }
}

#[test]
fn an_empty_surface_binds_and_prices_nothing() {
    // Degenerate and legal: nothing enumerated is nothing to price. It
    // is READY rather than an error because "cannot be priced" is a
    // statement about a tensor, and there are none.
    let container = container::dense();
    let (model, facts) = read(container.path());
    let empty = TensorSurface::new([]).expect("an empty surface is a surface");
    let bound = facts.bind(&model, &empty).expect("READY");
    assert!(bound.is_empty());
    assert!(bound.prices_for(&empty).expect("bound").is_empty());
}

#[test]
fn the_reported_gap_reads_as_a_sentence() {
    let missing = super::super::bind::AccountingIncomplete {
        missing: vec![
            TensorIdentity::new("target.alpha", "w"),
            TensorIdentity::new("target.zulu", "w"),
        ],
    };
    let text = missing.to_string();
    assert!(text.contains("2 surface tensor(s)"), "{text}");
    assert!(text.contains("target.alpha/w"), "{text}");
    assert!(text.contains("target.zulu/w"), "{text}");
}
