//! The reference table and the closure it describes, with the fixtures
//! they share.

mod closure;
mod table;

use super::*;

pub(super) const STACK: &str = "target.decoder_stack";
pub(super) const BOOKS: &str = "target.codebooks";
pub(super) const CODEBOOK: &str = "codebook";

pub(super) fn address(object: &str, tensor: &str) -> OperandAddress {
    OperandAddress::new(object, tensor)
}

pub(super) fn reference(
    owner: (&str, &str),
    auxiliary: &str,
    target: (&str, &str),
) -> AuxiliaryReference {
    AuxiliaryReference {
        owner: address(owner.0, owner.1),
        auxiliary: auxiliary.to_string(),
        target: address(target.0, target.1),
    }
}
