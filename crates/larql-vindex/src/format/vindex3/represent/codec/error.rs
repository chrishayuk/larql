//! Every way a codec refuses, each naming what was asked and what is there.
//!
//! A refusal that names only one side sends someone to the wrong remedy: a
//! stream missing from a binding is fixed at the binding site, a family
//! nobody registered is fixed by registering it, and a revision this build
//! does not implement is fixed by recompiling the pack. Only the pair
//! distinguishes them, so every variant carries both.

use crate::error::VindexError;

/// Why a codec, or the registry in front of it, cannot honour a request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CodecError {
    #[error(
        "tensor `{tensor}`: representation `{label}` is not registered; registered: [{}]",
        registered.join(", ")
    )]
    UnknownEncoding {
        tensor: String,
        label: String,
        registered: Vec<String>,
    },

    #[error(
        "tensor `{tensor}`: `{label}` needs stream `{stream}`, which was not bound; bound: [{}]",
        bound.join(", ")
    )]
    MissingStream {
        tensor: String,
        label: String,
        stream: String,
        bound: Vec<String>,
    },

    #[error(
        "tensor `{tensor}`: `{label}` stores its streams apart ({}); one payload cannot be \
         bound to it — bind each stream by name",
        streams.join(", ")
    )]
    StreamsStoredApart {
        tensor: String,
        label: String,
        streams: Vec<String>,
    },

    #[error(
        "tensor `{tensor}`: `{label}` declares no stream `{stream}`; declared: [{}]",
        declared.join(", ")
    )]
    UnexpectedStream {
        tensor: String,
        label: String,
        stream: String,
        declared: Vec<String>,
    },

    #[error("tensor `{tensor}`: `{label}` stream `{stream}` is {have} bytes; {need} needed")]
    StreamLength {
        tensor: String,
        label: String,
        stream: String,
        need: usize,
        have: usize,
    },

    #[error(
        "tensor `{tensor}`: `{label}` is entropy-coded, so its stored size is a property of \
         the instance and not of its shape; read the operand length the container records"
    )]
    InstanceSized { tensor: String, label: String },

    #[error("tensor `{tensor}`: shape {shape:?} cannot hold `{label}`: {why}")]
    Geometry {
        tensor: String,
        label: String,
        shape: Vec<usize>,
        why: String,
    },

    #[error(
        "tensor `{tensor}`: `{label}` has no extent at depth {depth}; it declares {available}"
    )]
    ExtentUnavailable {
        tensor: String,
        label: String,
        depth: u32,
        available: u32,
    },

    #[error("tensor `{tensor}`: rows {start}..{end} exceed the {rows} rows `{label}` holds")]
    RowRange {
        tensor: String,
        label: String,
        start: usize,
        end: usize,
        rows: usize,
    },

    #[error("tensor `{tensor}`: destination holds {have} floats; {need} needed")]
    Destination {
        tensor: String,
        need: usize,
        have: usize,
    },

    #[error("tensor `{tensor}`: decoding `{label}`: {detail}")]
    Decode {
        tensor: String,
        label: String,
        detail: String,
    },

    #[error("`{label}` provides {provided} access; the plan requires {required}")]
    AccessRefused {
        label: String,
        provided: String,
        required: String,
    },

    #[error(
        "representation family `{family}` is not registered; registered: [{}]",
        registered.join(", ")
    )]
    UnknownFamily {
        family: String,
        registered: Vec<String>,
    },

    #[error(
        "`{family}` ABI revision {found} was compiled by another build; this one implements \
         revision {implemented}. Recompile the representation from its canonical source \
         rather than decoding it under new rules."
    )]
    AbiRevision {
        family: String,
        found: u32,
        implemented: u32,
    },

    #[error(
        "`{family}` revision {revision} declares geometry this build does not produce \
         ({declared}); the index disagrees with its own revision"
    )]
    AbiGeometry {
        family: String,
        revision: u32,
        declared: String,
    },

    #[error("codec registry: `{label}` is registered twice")]
    DuplicateLabel { label: String },

    #[error(
        "tensor `{tensor}`: `{label}` at depth {depth} decodes finite normal values with \
         relative RMS {measured:.3e}; its certificate declares {declared:.3e}"
    )]
    CertificateViolated {
        tensor: String,
        label: String,
        depth: u32,
        declared: f64,
        measured: f64,
    },

    #[error(
        "tensor `{tensor}`: `{label}` at depth {depth} decodes finite normal values with \
         relative RMS {measured:.3e}, worse than depth {shallower}'s {before:.3e}; a deeper \
         extent must reconstruct at least as well"
    )]
    CertificateNotMonotone {
        tensor: String,
        label: String,
        depth: u32,
        shallower: u32,
        measured: f64,
        before: f64,
    },

    #[error(
        "tensor `{tensor}`: `{label}`'s terminal extent (depth {depth}) reconstructs \
         {differing} of {elements} bit patterns differently from the source; the deepest \
         extent must be exact"
    )]
    TerminalNotExact {
        tensor: String,
        label: String,
        depth: u32,
        differing: usize,
        elements: usize,
    },
}

impl From<CodecError> for VindexError {
    fn from(e: CodecError) -> Self {
        VindexError::Parse(e.to_string())
    }
}
