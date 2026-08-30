//! `tests` for [`super`].

use super::*;
use crate::format::vindex3::represent::map::Exception;
use std::collections::BTreeMap;

const K: usize = 256;

/// An in-memory stand-in for the container, so the arena's own
/// behaviour is tested without a 92 GB dependency.
struct FakeContainer {
    tensors: BTreeMap<(String, String), Vec<f32>>,
    loads: std::cell::RefCell<Vec<String>>,
}

impl FakeContainer {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            loads: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn with(mut self, tensor: &str, seed: f32) -> Self {
        let values: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.017 + seed).sin()).collect();
        self.tensors
            .insert(("target.expert_bank".into(), tensor.into()), values);
        self
    }
}

impl SourceOperands for FakeContainer {
    fn load_stored(&self, operand: &OperandRef) -> Result<StoredOperand, VindexError> {
        self.loads.borrow_mut().push(operand.tensor.clone());
        let values = self
            .tensors
            .get(&(operand.object.clone(), operand.tensor.clone()))
            .ok_or_else(|| VindexError::Parse(format!("no tensor `{}`", operand.tensor)))?;
        Ok(StoredOperand {
            dtype: "BF16".into(),
            bytes: values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
        })
    }
}

fn operand(tensor: &str) -> OperandRef {
    OperandRef {
        object: "target.expert_bank".into(),
        tensor: tensor.into(),
        dtype: "BF16".into(),
        shape: vec![1, K],
    }
}

fn scoped_map(exception: Option<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "q2-candidate".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: exception.into_iter().collect(),
    }
}

/// The baseline arm returns the container's OWN bytes, untouched — no
/// decode/re-encode round trip that could perturb the reference the
/// candidate is measured against.
#[test]
fn a_source_precision_arm_returns_the_stored_bytes_unchanged() {
    let c = FakeContainer::new().with("3.mlp.experts.7.down_proj.weight", 0.3);
    // A map naming no roles compiles nothing.
    let arena = RepresentationArena::new(PrecisionMap {
        roles: vec![],
        ..scoped_map(None)
    });
    let got = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("3.mlp.experts.7.down_proj.weight"),
        )
        .expect("resolves");
    let stored = c
        .load_stored(&operand("3.mlp.experts.7.down_proj.weight"))
        .expect("stored");
    assert_eq!(got.encoding, "BF16");
    assert_eq!(*got.bytes, stored.bytes);
    assert_eq!(arena.materialised(), 0, "source precision caches nothing");
}

/// The candidate arm materialises the scoped operand, and the SECOND
/// ask is served from cache — keyed by semantic identity, so it is the
/// same operand that hits, not whatever happened to share an address.
#[test]
fn a_scoped_operand_is_materialised_once_and_reused() {
    let c = FakeContainer::new().with("3.mlp.experts.7.down_proj.weight", 0.3);
    let arena = RepresentationArena::new(scoped_map(None));
    let op = operand("3.mlp.experts.7.down_proj.weight");

    let first = arena.resolve(&c, Role::ExpertWeight, &op).expect("first");
    assert_eq!(first.encoding, "Q6_K");
    assert!(!first.cached);
    // 210 bytes a 256-element superblock.
    assert_eq!(first.bytes.len(), 210);

    let second = arena.resolve(&c, Role::ExpertWeight, &op).expect("second");
    assert!(second.cached);
    assert_eq!(second.bytes, first.bytes);
    assert_eq!(arena.materialised(), 1);
    assert_eq!(
        c.loads.borrow().len(),
        1,
        "a cache hit must not re-read the container"
    );
}

/// **Any expert is addressable, including one no baseline route
/// touched.**
///
/// If the candidate representation moves a routing decision from expert
/// 73 to expert 181, expert 181 must resolve and execute. Backed by the
/// container rather than a pre-exported union, it does — and that event
/// is the one the quality bank exists to observe, not a refusal caused
/// by how the fixture was built.
#[test]
fn an_expert_the_baseline_never_routed_to_still_resolves() {
    let c = FakeContainer::new()
        .with("3.mlp.experts.73.down_proj.weight", 0.3)
        .with("3.mlp.experts.181.down_proj.weight", 1.9);
    let arena = RepresentationArena::new(scoped_map(None));

    let baseline_route = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("3.mlp.experts.73.down_proj.weight"),
        )
        .expect("baseline expert");
    let diverged = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("3.mlp.experts.181.down_proj.weight"),
        )
        .expect("an expert only the CANDIDATE routes to must still resolve");
    assert_ne!(
        baseline_route.bytes, diverged.bytes,
        "two experts must not collide in the cache"
    );
    assert_eq!(arena.materialised(), 2);
}

/// The arena grows to exactly the scope under test. A narrow
/// `RoleScope` must not drag the whole role into memory.
#[test]
fn a_narrow_scope_materialises_only_its_own_region() {
    let c = FakeContainer::new()
        .with("22.mlp.experts.3.down_proj.weight", 0.1)
        .with("22.mlp.experts.3.gate_proj.weight", 0.2)
        .with("4.mlp.experts.3.down_proj.weight", 0.4);
    // down_proj, layers 20..26 at Q6_K; everything else source.
    let arena = RepresentationArena::new(PrecisionMap {
        name: "late-down".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: Some("down_proj".into()),
                layers: Some((20, 26)),
                encoding: Some("Q6_K".into()),
            },
            // Everything else in the role falls back to source.
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    });

    let inside = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("22.mlp.experts.3.down_proj.weight"),
        )
        .expect("in scope");
    assert_eq!(inside.encoding, "Q6_K");
    for outside in [
        "22.mlp.experts.3.gate_proj.weight",
        "4.mlp.experts.3.down_proj.weight",
    ] {
        let got = arena
            .resolve(&c, Role::ExpertWeight, &operand(outside))
            .expect("out of scope");
        assert_eq!(got.encoding, "BF16", "{outside} is outside the scope");
    }
    assert_eq!(
        arena.materialised(),
        1,
        "only the scoped operand was encoded"
    );
}

/// A role the map does not name is untouched, whatever the exceptions
/// say — the map's own fail-safe, exercised through the arena.
#[test]
fn an_unnamed_role_is_never_materialised() {
    let c = FakeContainer::new().with("3.mlp.gate.weight", 0.5);
    let arena = RepresentationArena::new(scoped_map(None));
    let got = arena
        .resolve(&c, Role::Router, &operand("3.mlp.gate.weight"))
        .expect("resolves");
    assert_eq!(got.encoding, "BF16");
    assert_eq!(arena.materialised(), 0);
}

/// **Fail closed.** An encoding with no encoder is refused, never
/// silently served as source bytes — binding BF16 under a name claiming
/// Q3_K would make every downstream record a lie, and the failure would
/// read as "quantisation is free".
#[test]
fn an_unknown_encoding_is_refused_rather_than_passed_through() {
    let c = FakeContainer::new().with("3.mlp.experts.7.down_proj.weight", 0.3);
    let arena = RepresentationArena::new(PrecisionMap {
        encoding: "Q3_K".into(),
        ..scoped_map(None)
    });
    let err = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("3.mlp.experts.7.down_proj.weight"),
        )
        .expect_err("must refuse");
    assert!(format!("{err}").contains("no encoder for `Q3_K`"), "{err}");
}

/// A tensor that is not a whole number of superblocks is refused, since
/// a flat quantisation of it would let two rows share a scale.
#[test]
fn a_shape_that_would_straddle_superblocks_is_refused() {
    let mut c = FakeContainer::new();
    c.tensors.insert(
        (
            "target.expert_bank".into(),
            "3.mlp.experts.7.down_proj.weight".into(),
        ),
        vec![0.5f32; K + 1],
    );
    let arena = RepresentationArena::new(scoped_map(None));
    let err = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("3.mlp.experts.7.down_proj.weight"),
        )
        .expect_err("must refuse");
    assert!(format!("{err}").contains("superblock"), "{err}");
}

/// The arena reports the map it executes and how many operands it has
/// materialised — the per-ARM accounting two arms are compared by.
#[test]
fn the_arena_reports_its_map_and_what_it_has_materialised() {
    let mut c = FakeContainer::new();
    c.tensors.insert(
        (
            "target.expert_bank".into(),
            "1.mlp.experts.0.down_proj.weight".into(),
        ),
        vec![0.25f32; K],
    );
    let arena = RepresentationArena::new(scoped_map(None));
    assert_eq!(arena.map().name, scoped_map(None).name);
    assert_eq!(arena.materialised(), 0, "nothing resolved yet");
    let first = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("1.mlp.experts.0.down_proj.weight"),
        )
        .expect("resolves");
    assert_eq!(arena.materialised(), 1);
    // The cache is keyed on SEMANTIC identity, so a second resolve of
    // the same operand adds nothing and returns the same bytes.
    let again = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("1.mlp.experts.0.down_proj.weight"),
        )
        .expect("resolves again");
    assert_eq!(arena.materialised(), 1, "a cache hit materialises nothing");
    assert_eq!(first.bytes, again.bytes);
    assert_eq!(first.encoding, again.encoding);
}

/// BF16 is a real encoder here, not a pass-through: the arena narrows
/// f32 values to the checkpoint's own codes.
#[test]
fn bf16_is_encoded_from_values_not_passed_through() {
    let mut c = FakeContainer::new();
    let values: Vec<f32> = (0..K).map(|i| (i as f32) * 0.001 - 0.1).collect();
    c.tensors.insert(
        (
            "target.expert_bank".into(),
            "9.mlp.experts.3.down_proj.weight".into(),
        ),
        values.clone(),
    );
    let arena = RepresentationArena::new(PrecisionMap {
        encoding: "BF16".into(),
        exceptions: vec![Exception {
            projection: None,
            layers: None,
            encoding: Some("BF16".into()),
        }],
        ..scoped_map(None)
    });
    let got = arena
        .resolve(
            &c,
            Role::ExpertWeight,
            &operand("9.mlp.experts.3.down_proj.weight"),
        )
        .expect("resolves");
    assert_eq!(got.encoding, "BF16");
    let want: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    assert_eq!(got.bytes.as_slice(), want.as_slice());
}
