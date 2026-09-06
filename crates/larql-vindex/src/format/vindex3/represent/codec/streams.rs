//! The operands a decode consumes: named byte streams, and the represented
//! objects a stream may depend on for its meaning.
//!
//! `&[u8]` was the wall two formats already hit — ternary's per-channel
//! scales and MXFP4's e8m0 exponents had nowhere to ride a single slice, so
//! both were routed to dedicated kernels outside the trait that should have
//! served them. A named set of streams is the first repair. The second is
//! [`AuxiliaryOperands`]: a codebook is not a stream of this tensor's bytes,
//! it is another represented object, and a decode that depends on one has
//! admitted a representation dependency. Keeping the two apart is what
//! stops a dependency graph from being smuggled in as a stream set.
//!
//! A stream is bytes and reaches the codec as bytes. An auxiliary is
//! another operand and reaches the codec as VALUES — decoded through its
//! own codec, at its own extent, by whoever resolved the closure. The
//! codec never learns how its dependency was stored, and never acquires
//! it: resolution happens before the decode, in dependency order.

use std::collections::BTreeMap;

use super::error::CodecError;

/// What one stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRole {
    /// The element codes themselves, scales inline or not.
    Values,
    /// One scale per group of elements.
    GroupScales,
    /// One scale for the whole tensor.
    TensorScale,
    /// A refinement of everything before it: the stream a progressive
    /// representation reads to reach `depth`, meaningless on its own and
    /// never read at a shallower extent.
    Refinement { depth: u32 },
}

/// One stream a codec declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamSpec {
    pub name: &'static str,
    pub role: StreamRole,
}

/// The stream every codec has: its codes.
pub const VALUES: StreamSpec = StreamSpec {
    name: "values",
    role: StreamRole::Values,
};
/// Per-group scales stored apart from the codes.
pub const GROUP_SCALES: StreamSpec = StreamSpec {
    name: "group_scales",
    role: StreamRole::GroupScales,
};
/// A whole-tensor scale stored apart from the codes.
pub const TENSOR_SCALE: StreamSpec = StreamSpec {
    name: "tensor_scale",
    role: StreamRole::TensorScale,
};

/// Bound streams, by declared name.
#[derive(Debug, Clone, Default)]
pub struct NamedStreams<'a> {
    streams: Vec<(&'static str, &'a [u8])>,
}

impl<'a> NamedStreams<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// One stream — the shape every inline-scale codec binds.
    pub fn single(spec: StreamSpec, bytes: &'a [u8]) -> Self {
        Self::new().with(spec, bytes)
    }

    /// Bind `spec`, replacing an earlier binding of the same name.
    pub fn with(mut self, spec: StreamSpec, bytes: &'a [u8]) -> Self {
        self.streams.retain(|(name, _)| *name != spec.name);
        self.streams.push((spec.name, bytes));
        self
    }

    pub fn get(&self, spec: StreamSpec) -> Option<&'a [u8]> {
        self.streams
            .iter()
            .find(|(name, _)| *name == spec.name)
            .map(|(_, bytes)| *bytes)
    }

    /// Bound names, in binding order — what a refusal lists.
    pub fn names(&self) -> Vec<String> {
        self.streams.iter().map(|(n, _)| (*n).to_string()).collect()
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// One dependency, RESOLVED: decoded through its own codec, at its own
/// selected extent, into the canonical f32 surface.
///
/// Values rather than an `OperandRef`, and the change is the point. The
/// slot originally held a reference — which is what a container states
/// and what a closure resolves — but no decode can use one: a codec has
/// no store, no registry, and no business acquiring bytes. Resolution
/// happens before the codec is called, in dependency order, and what
/// arrives here is what the dependency MEANS. It also makes an
/// auxiliary's own representation invisible to its owner: a codebook
/// stored as planes and a codebook stored raw reach this struct
/// identically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedAuxiliary<'a> {
    /// The shape the container records for the dependency.
    pub shape: &'a [usize],
    /// Its decoded values, row-major.
    pub values: &'a [f32],
}

/// Other represented objects a decode depends on, by the name the codec
/// declared for each. Empty for every codec that reads only its own bytes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuxiliaryOperands<'a> {
    operands: BTreeMap<String, ResolvedAuxiliary<'a>>,
}

impl<'a> AuxiliaryOperands<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: impl Into<String>, resolved: ResolvedAuxiliary<'a>) -> Self {
        self.operands.insert(name.into(), resolved);
        self
    }

    pub fn get(&self, name: &str) -> Option<ResolvedAuxiliary<'a>> {
        self.operands.get(name).copied()
    }

    /// The dependency `name`, or a refusal listing what was resolved —
    /// the auxiliary twin of [`CodecOperands::stream`].
    pub fn require(
        &self,
        name: &str,
        label: &str,
        tensor: &str,
    ) -> Result<ResolvedAuxiliary<'a>, CodecError> {
        self.get(name).ok_or_else(|| CodecError::MissingAuxiliary {
            tensor: tensor.into(),
            label: label.into(),
            name: name.into(),
            depth: 0,
            required: self.names(),
        })
    }

    /// Resolved names, in name order — what a refusal lists.
    pub fn names(&self) -> Vec<String> {
        self.operands.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.operands.len()
    }
}

/// Everything one decode consumes.
#[derive(Debug, Clone, Default)]
pub struct CodecOperands<'a> {
    pub streams: NamedStreams<'a>,
    pub auxiliaries: AuxiliaryOperands<'a>,
}

impl<'a> CodecOperands<'a> {
    pub fn from_streams(streams: NamedStreams<'a>) -> Self {
        Self {
            streams,
            auxiliaries: AuxiliaryOperands::new(),
        }
    }

    /// The stream `spec` names, or a refusal listing what was bound.
    pub fn stream(
        &self,
        spec: StreamSpec,
        label: &str,
        tensor: &str,
    ) -> Result<&'a [u8], CodecError> {
        self.streams
            .get(spec)
            .ok_or_else(|| CodecError::MissingStream {
                tensor: tensor.into(),
                label: label.into(),
                stream: spec.name.into(),
                bound: self.streams.names(),
            })
    }

    /// The stream `spec` names, refusing one shorter than `need` bytes.
    pub fn stream_of_len(
        &self,
        spec: StreamSpec,
        need: usize,
        label: &str,
        tensor: &str,
    ) -> Result<&'a [u8], CodecError> {
        let bytes = self.stream(spec, label, tensor)?;
        if bytes.len() < need {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: label.into(),
                stream: spec.name.into(),
                need,
                have: bytes.len(),
            });
        }
        Ok(bytes)
    }
}
