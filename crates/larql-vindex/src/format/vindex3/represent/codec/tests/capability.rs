//! The access ladder, and refusals that name both sides.

use super::*;

#[test]
fn a_finer_granularity_provides_every_coarser_requirement() {
    use AccessGranularity as G;
    use RequiredAccess as R;
    let ladder = [
        G::Sequential,
        G::BlockRandom { elems: 256 },
        G::RowRandom,
        G::ElementRandom,
    ];
    for (i, provided) in ladder.iter().enumerate() {
        assert!(provided.provides(R::Sequential), "{provided:?}");
        assert_eq!(provided.provides(R::RowRandom), i >= 2, "{provided:?}");
        assert_eq!(provided.provides(R::ElementRandom), i >= 3, "{provided:?}");
    }
    // Block-random is not row-random: whether a row is whole blocks is a
    // fact about the tensor, not the codec.
    assert!(!G::BlockRandom { elems: 256 }.provides(R::RowRandom));
}

#[test]
fn granularities_and_requirements_have_names_a_refusal_can_carry() {
    assert_eq!(AccessGranularity::Sequential.name(), "sequential");
    assert_eq!(
        AccessGranularity::BlockRandom { elems: 256 }.name(),
        "block-random (256 elements)"
    );
    assert_eq!(AccessGranularity::RowRandom.name(), "row-random");
    assert_eq!(AccessGranularity::ElementRandom.name(), "element-random");
    assert_eq!(RequiredAccess::Sequential.name(), "sequential");
    assert_eq!(RequiredAccess::RowRandom.name(), "row-random");
    assert_eq!(RequiredAccess::ElementRandom.name(), "element-random");
}

#[test]
fn a_refusal_names_what_is_provided_and_what_is_required() {
    let sequential = CodecCapabilities {
        access: AccessGranularity::Sequential,
        group_elems: 1,
        row_align_elems: 1,
        physical_align_bytes: 1,
    };
    sequential
        .require(RequiredAccess::Sequential, "zstd-bf16")
        .unwrap();
    let err = sequential
        .require(RequiredAccess::RowRandom, "zstd-bf16")
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "`zstd-bf16` provides sequential access; the plan requires row-random"
    );
}

#[test]
fn row_admission_is_a_whole_number_of_the_alignment_and_zero_admits_nothing() {
    let caps = CodecCapabilities {
        access: AccessGranularity::RowRandom,
        group_elems: 32,
        row_align_elems: 32,
        physical_align_bytes: 1,
    };
    assert!(caps.admits_k(64));
    assert!(!caps.admits_k(48));
    let degenerate = CodecCapabilities {
        row_align_elems: 0,
        ..caps
    };
    assert!(!degenerate.admits_k(64));
}
