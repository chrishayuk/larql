//! The whole dependency closure, admitted or refused from metadata alone.
//!
//! Everything here runs against a container that HAS no payload — the
//! fixture answers `tensor()` and nothing else — so "no byte was read" is
//! not asserted, it is unrepresentable. What the tests do assert is the
//! shape of what the walk produces and the exact refusal for every way a
//! declared dependency can be wrong.

use std::cell::RefCell;
use std::collections::BTreeMap;

use super::super::closure::{admit, ClosureRoot, ContainerMetadata};
use super::super::*;
use super::{address, reference, BOOKS, CODEBOOK, STACK};
use crate::format::vindex3::represent::codec::codecs::float::F32;
use crate::format::vindex3::represent::codec::{
    AuxiliaryMetadata, AuxiliarySpec, CodecCapabilities, CodecError, CodecOperands, CodecRegistry,
    ExtentCertificate, RepresentationCodec, RepresentationExtent, ResidencyProfile,
    ResidencyProfile as Profile, StreamSpec,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// A test container: a tensor table, and a count of how often it was
/// asked. There is no payload here at all.
#[derive(Default)]
struct Tensors {
    by_address: BTreeMap<OperandAddress, (String, Vec<usize>)>,
    asked: RefCell<usize>,
}

impl Tensors {
    fn with(mut self, address: OperandAddress, label: &str, shape: &[usize]) -> Self {
        self.by_address
            .insert(address, (label.to_string(), shape.to_vec()));
        self
    }
}

impl ContainerMetadata for Tensors {
    fn tensor(&self, address: &OperandAddress) -> Option<(String, Vec<usize>)> {
        *self.asked.borrow_mut() += 1;
        self.by_address.get(address).cloned()
    }
}

/// A codec that requires one named dependency and checks its width.
struct Dependent {
    label: &'static str,
    family: &'static str,
    requires: &'static [AuxiliarySpec],
    width: usize,
}

const NEEDS_CODEBOOK: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];
const NEEDS_PALETTE: [AuxiliarySpec; 1] = [AuxiliarySpec::new("palette")];
const WIDTH: usize = 4;

impl RepresentationCodec for Dependent {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: self.family.into(),
            ..F32.identity()
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        F32.streams()
    }
    fn capabilities(&self) -> CodecCapabilities {
        F32.capabilities()
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(8.0)]
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        self.requires
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let entries = target.shape.first().copied().unwrap_or(0);
        target.require_shape(&[entries, self.width], tensor, self.label, name)
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        F32.stored_bytes(shape, RepresentationExtent::BASE, tensor)
    }
    fn validate(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<(), CodecError> {
        Ok(())
    }
    fn decode_rows(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        _: &str,
    ) -> Result<(), CodecError> {
        dst.fill(0.0);
        Ok(())
    }
    fn decode_residency(&self) -> Profile {
        ResidencyProfile::DECODED_F32
    }
}

const CODED: &str = "CODED";
const NESTED: &str = "NESTED";

/// `CODED` needs a codebook; `NESTED` needs a palette, so a codebook
/// stored as `NESTED` gives the closure two levels to order.
fn registry() -> &'static CodecRegistry {
    let registry = CodecRegistry::new()
        .register(Box::new(F32))
        .and_then(|r| {
            r.register(Box::new(Dependent {
                label: CODED,
                family: CODED,
                requires: &NEEDS_CODEBOOK,
                width: WIDTH,
            }))
        })
        .and_then(|r| {
            r.register(Box::new(Dependent {
                label: NESTED,
                family: NESTED,
                requires: &NEEDS_PALETTE,
                width: WIDTH,
            }))
        })
        .expect("three distinct labels");
    Box::leak(Box::new(registry))
}

fn root(object: &str, tensor: &str) -> ClosureRoot {
    ClosureRoot::new(address(object, tensor), RepresentationExtent::BASE)
}

/// Two owners, one codebook: the target is visited ONCE and ordered
/// before both of them.
#[test]
fn a_shared_target_is_reached_once_and_ordered_before_its_owners() {
    let down = address(STACK, "0.mlp.down_proj.weight");
    let up = address(STACK, "0.mlp.up_proj.weight");
    let book = address(BOOKS, "shared.codebook");
    let container = Tensors::default()
        .with(down.clone(), CODED, &[8, 256])
        .with(up.clone(), CODED, &[8, 256])
        .with(book.clone(), "F32", &[256, WIDTH]);
    let table = AuxiliaryReferences::new(vec![
        reference(
            (STACK, "0.mlp.down_proj.weight"),
            CODEBOOK,
            (BOOKS, "shared.codebook"),
        ),
        reference(
            (STACK, "0.mlp.up_proj.weight"),
            CODEBOOK,
            (BOOKS, "shared.codebook"),
        ),
    ])
    .judge()
    .unwrap();

    let closure = admit(
        &[
            root(STACK, "0.mlp.down_proj.weight"),
            root(STACK, "0.mlp.up_proj.weight"),
        ],
        &table,
        registry(),
        &container,
    )
    .expect("a shared codebook is ordinary");

    assert_eq!(closure.len(), 3, "two owners and one target, counted once");
    assert_eq!(closure.order()[0], book, "the dependency comes first");
    assert!(closure.order()[1..].contains(&down) && closure.order()[1..].contains(&up));
    assert_eq!(closure.dependencies_of(&down).len(), 1);
    assert_eq!(closure.dependencies_of(&down)[0].target, book);
    assert!(
        closure.dependencies_of(&book).is_empty(),
        "the codebook depends on nothing"
    );
}

/// A codebook that itself depends on something: the order is deepest
/// first, which is the property a loader relies on.
#[test]
fn a_dependency_of_a_dependency_is_ordered_deepest_first() {
    let owner = address(STACK, "w");
    let book = address(BOOKS, "cb");
    let palette = address(BOOKS, "palette");
    let container = Tensors::default()
        .with(owner.clone(), CODED, &[8, 256])
        .with(book.clone(), NESTED, &[256, WIDTH])
        .with(palette.clone(), "F32", &[256, WIDTH]);
    let table = AuxiliaryReferences::new(vec![
        reference((STACK, "w"), CODEBOOK, (BOOKS, "cb")),
        reference((BOOKS, "cb"), "palette", (BOOKS, "palette")),
    ])
    .judge()
    .unwrap();

    let closure = admit(&[root(STACK, "w")], &table, registry(), &container).unwrap();
    assert_eq!(closure.order(), &[palette, book, owner]);
}

/// Every way a declared dependency can be wrong, refused by name and
/// before anything is read.
#[test]
fn each_way_a_dependency_can_be_wrong_is_refused_by_name() {
    let good_container = || {
        Tensors::default()
            .with(address(STACK, "w"), CODED, &[8, 256])
            .with(address(BOOKS, "cb"), "F32", &[256, WIDTH])
    };
    let points_at_the_book = || {
        AuxiliaryReferences::new(vec![reference((STACK, "w"), CODEBOOK, (BOOKS, "cb"))])
            .judge()
            .unwrap()
    };

    // The codec requires a name the table does not provide.
    let err = admit(
        &[root(STACK, "w")],
        &ReferenceTable::empty(),
        registry(),
        &good_container(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(CODEBOOK) && err.contains("no reference"),
        "{err}"
    );

    // The table provides a name the codec does not require.
    let extra = AuxiliaryReferences::new(vec![
        reference((STACK, "w"), CODEBOOK, (BOOKS, "cb")),
        reference((STACK, "w"), "atlas", (BOOKS, "cb")),
    ])
    .judge()
    .unwrap();
    let err = admit(&[root(STACK, "w")], &extra, registry(), &good_container())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("atlas") && err.contains("does not require"),
        "{err}"
    );

    // The target is not in the container.
    let without_book = Tensors::default().with(address(STACK, "w"), CODED, &[8, 256]);
    let err = admit(
        &[root(STACK, "w")],
        &points_at_the_book(),
        registry(),
        &without_book,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("no such tensor") && err.contains("cb"),
        "{err}"
    );

    // The target's label names no registered codec: a missing provider.
    let foreign = Tensors::default()
        .with(address(STACK, "w"), CODED, &[8, 256])
        .with(address(BOOKS, "cb"), "NOBODY_REGISTERS_THIS", &[256, WIDTH]);
    let err = admit(
        &[root(STACK, "w")],
        &points_at_the_book(),
        registry(),
        &foreign,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("NOBODY_REGISTERS_THIS") && err.contains("not registered"),
        "{err}"
    );

    // The target exists, is registered, and is the wrong shape — judged
    // by the OWNER, from metadata.
    let wrong_width = Tensors::default()
        .with(address(STACK, "w"), CODED, &[8, 256])
        .with(address(BOOKS, "cb"), "F32", &[256, WIDTH + 1]);
    let err = admit(
        &[root(STACK, "w")],
        &points_at_the_book(),
        registry(),
        &wrong_width,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("is unusable") && err.contains(CODEBOOK),
        "{err}"
    );
}

/// A cycle through two operands is refused with the path that closes it.
/// (A one-operand cycle never reaches here: the table refuses a
/// self-reference where it is declared.)
#[test]
fn a_cycle_is_refused_with_the_path_that_closes_it() {
    let first = address(BOOKS, "first");
    let second = address(BOOKS, "second");
    let container = Tensors::default()
        .with(address(STACK, "w"), CODED, &[8, 256])
        .with(first.clone(), CODED, &[256, WIDTH])
        .with(second.clone(), NESTED, &[256, WIDTH]);
    let table = AuxiliaryReferences::new(vec![
        reference((STACK, "w"), CODEBOOK, (BOOKS, "first")),
        reference((BOOKS, "first"), CODEBOOK, (BOOKS, "second")),
        reference((BOOKS, "second"), "palette", (BOOKS, "first")),
    ])
    .judge()
    .unwrap();

    let err = admit(&[root(STACK, "w")], &table, registry(), &container)
        .unwrap_err()
        .to_string();
    assert!(err.contains("cycle"), "{err}");
    assert!(err.contains("first") && err.contains("second"), "{err}");
    assert!(err.contains("→"), "the path is spelled: {err}");
}

/// The walk asks the container about metadata and nothing else — the
/// count is the witness that admission is a metadata pass.
#[test]
fn admission_reads_metadata_and_the_reading_is_bounded() {
    let container = Tensors::default()
        .with(address(STACK, "w"), CODED, &[8, 256])
        .with(address(BOOKS, "cb"), "F32", &[256, WIDTH]);
    let table = AuxiliaryReferences::new(vec![reference((STACK, "w"), CODEBOOK, (BOOKS, "cb"))])
        .judge()
        .unwrap();
    let closure = admit(&[root(STACK, "w")], &table, registry(), &container).unwrap();
    assert_eq!(closure.len(), 2);
    let asked = *container.asked.borrow();
    assert!(
        (2..=4).contains(&asked),
        "each operand described a bounded number of times, not once per edge: {asked}"
    );
}
