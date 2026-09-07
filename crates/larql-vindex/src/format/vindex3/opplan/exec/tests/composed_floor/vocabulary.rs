//! The premises this arm rests on, and the names it addresses them by.
//!
//! One home for every value the tests compare against, so a number can
//! never be spelled twice and drift. The `const` block is the guard: if
//! someone edits these so one term alone settles the outcome, the build
//! fails rather than the witness quietly becoming a tautology.

use crate::format::vindex3::represent::codec::{AuxiliarySpec, StreamRole, StreamSpec};

/// What the OWNER's shallow extent declares on its own.
pub(super) const PARENT: f64 = 0.003;
/// What the coarse codebook certifies.
pub(super) const COARSE: f64 = 0.003;
/// What the fine codebook certifies.
pub(super) const FINE: f64 = 0.001;
/// What execution requires.
pub(super) const FLOOR: f64 = 0.005;
/// What an ENCODER measured on this instance, against the coarse
/// codebook — better than the scheme's declaration, because the scheme
/// has to hold for every tensor and this number was measured on one.
pub(super) const ATTESTED: f64 = 0.001;

pub(super) const AUTHORITY: &str = "larql-encoder";
pub(super) const METHOD: &str = "measured-rms";

/// Neither half decides it, and the composition decides it both ways. If
/// someone edits these so that one term alone settles the outcome, the
/// build fails rather than the test quietly proving nothing.
const _: () = {
    assert!(PARENT < FLOOR);
    assert!(COARSE < FLOOR);
    assert!(FINE < FLOOR);
    assert!(PARENT + COARSE > FLOOR);
    assert!(PARENT + FINE < FLOOR);
    // And the attested arm: the SAME coarse codebook, admitted only
    // because the measured claim replaced the declared one.
    assert!(ATTESTED + COARSE < FLOOR);
};

pub(super) const OWNER_LABEL: &str = "GRADED_PALETTE";
pub(super) const COARSE_LABEL: &str = "COARSEBOOK";
pub(super) const FINE_LABEL: &str = "FINEBOOK";
pub(super) const CODEBOOK: &str = "codebook";
pub(super) const CODEBOOK_TENSOR: &str = "shared.palette.codebook";
pub(super) const ENTRIES: usize = 256;
pub(super) const REFINE_DTYPE: &str = "U8";

/// The projections stored under the owner's codec.
pub(super) const OWNERS: [&str; 2] = ["0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"];

pub(super) const VALUES: StreamSpec = StreamSpec {
    name: "codes",
    role: StreamRole::Values,
};
pub(super) const REFINE: StreamSpec = StreamSpec {
    name: "residual",
    role: StreamRole::Refinement { depth: 1 },
};
pub(super) const OWNER_STREAMS: [StreamSpec; 2] = [VALUES, REFINE];
pub(super) const BOOK_STREAMS: [StreamSpec; 1] = [VALUES];
pub(super) const OWNER_AUXILIARIES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];

pub(super) fn elements(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

/// The quantum a residual byte corrects by. Read by the codec that
/// decodes residuals and by the builder that writes them, so it lives
/// here rather than in either.
pub(super) const PALETTE_STEP: f32 = 1.0 / 512.0;

/// A bound so loose that only an ABSENT claim can fail it — the point
/// being that an absent certificate is not a large radius.
pub(super) const UNREACHABLE_BOUND: f64 = 1.0e9;

/// A floor tighter than the coarse container's TERMINAL composition
/// (`0.0` from the owner widened by the codebook's 0.003), so the plan is
/// refused at the initial pin with a bound that exists and is simply too
/// large — the other half of the refusal, opposite an absent one.
pub(super) const TIGHTER_THAN_COMPOSED: f64 = 0.001;
