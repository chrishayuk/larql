//! Every refusal names itself, and converts to the crate's error without losing the cause.

use larql_vindex::format::vindex3::weight_patch::WeightPatchError as E;
use larql_vindex::VindexError;

fn every_refusal() -> Vec<E> {
    vec![
        E::TooManyElements {
            count: 5000,
            limit: 4096,
        },
        E::Empty,
        E::DuplicateElement { row: 1, col: 2 },
        E::NonFiniteDelta { row: 1, col: 2 },
        E::UnknownTensor("t".into()),
        E::AmbiguousTensor {
            tensor: "t".into(),
            holders: vec!["a@BF16".into(), "b@BF16".into()],
        },
        E::UnsupportedDtype {
            tensor: "t".into(),
            dtype: "F32".into(),
        },
        E::NotAMatrix {
            tensor: "t".into(),
            shape: vec![4],
        },
        E::OutOfBounds {
            tensor: "t".into(),
            row: 9,
            col: 9,
            rows: 4,
            cols: 4,
        },
        E::NonFiniteSource {
            tensor: "t".into(),
            row: 1,
            col: 1,
        },
        E::Overflow { row: 1, col: 1 },
        E::NotCanonical("Derived".into()),
        E::AlreadyPatched,
        E::HasVariants,
        E::DerivedFromOther {
            derived: "d".into(),
            canonical: "c".into(),
        },
        E::UnexpectedEntry("extra".into()),
        E::SourceIncoherent("x".into()),
        E::NotReproducible("x".into()),
        E::OutputExists("/tmp/x".into()),
        E::Incoherent("x".into()),
        E::Conflict {
            tensor: "t".into(),
            row: 1,
            col: 2,
            by: "a patch".into(),
        },
        E::WrongSource {
            expected: "aa / bb".into(),
            found: "cc / dd".into(),
        },
        E::PatchOp("x".into()),
        E::MoreThanOneTensor(vec!["a".into(), "b".into()]),
        E::Io(std::io::Error::other("disk")),
    ]
}

#[test]
fn every_refusal_says_what_it_is() {
    for e in every_refusal() {
        let text = e.to_string();
        assert!(!text.is_empty(), "{e:?} has no message");
        let converted: VindexError = e.into();
        assert!(
            converted.to_string().contains("weight patch refused")
                || matches!(converted, VindexError::Io(_)),
            "{converted}"
        );
    }
}

#[test]
fn a_wrapped_crate_error_passes_through_unchanged() {
    let inner = VindexError::Parse("the original cause".into());
    let wrapped = E::Vindex(inner);
    let back: VindexError = wrapped.into();
    assert_eq!(back.to_string(), "parse error: the original cause");
}
