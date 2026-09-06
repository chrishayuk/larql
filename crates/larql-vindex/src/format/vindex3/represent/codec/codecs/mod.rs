//! The encodings this build registers, each an implementation of
//! [`super::RepresentationCodec`] over numerics that already existed.
//!
//! Nothing here defines arithmetic. Every decode routes through the
//! decoder the workspace already judged — `larql_models::quant` for the
//! grids, the operand widener for the floats — so registering a codec
//! adds a *declaration*, never a second opinion about what bytes mean.

pub mod bf16_zlib;
pub mod f32_planes;
pub mod float;
pub mod kquant;
pub mod lyrw2;
pub mod mxfp4;
pub mod nvfp4;
pub mod vq8_shared;

/// Vocabulary shared by the identities and capabilities below.
pub(crate) mod vocabulary {
    /// No scale at this level of an identity.
    pub const SCALE_NONE: &str = "none";
    /// Nothing is shared between elements: a group of one.
    pub const UNGROUPED: usize = 1;
    /// A stream of bytes needs no wider alignment than a byte.
    pub const BYTE_ALIGN: usize = 1;
}
