//! The codecs this build can serve, addressed by the label a container
//! writes and by the ABI family an index declares.
//!
//! Two keys because they answer different questions. A segment's `dtype`
//! asks "which codec reads these bytes"; a pack's `codec` entry asks "is
//! the contract these bytes were written under the one this build
//! implements". The second is [`Self::admit`], and it is what stops an
//! improved encoder from silently redefining every artifact on disk.

use std::sync::OnceLock;

use super::codecs::{float, kquant, mxfp4, nvfp4};
use super::error::CodecError;
use super::RepresentationCodec;
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// Registered codecs, in registration order.
#[derive(Default)]
pub struct CodecRegistry {
    codecs: Vec<Box<dyn RepresentationCodec>>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a codec, refusing a label or family already taken.
    pub fn register(mut self, codec: Box<dyn RepresentationCodec>) -> Result<Self, CodecError> {
        let label = codec.encoding_label();
        let family = codec.identity().family;
        if self.by_label(label).is_some() {
            return Err(CodecError::DuplicateLabel {
                label: label.into(),
            });
        }
        if self.by_family(&family).is_some() {
            return Err(CodecError::DuplicateLabel { label: family });
        }
        self.codecs.push(codec);
        Ok(self)
    }

    /// The codecs this build ships.
    pub fn builtin() -> &'static CodecRegistry {
        static BUILTIN: OnceLock<CodecRegistry> = OnceLock::new();
        BUILTIN.get_or_init(|| {
            Self::new()
                .register(Box::new(float::BF16))
                .and_then(|r| r.register(Box::new(float::F16)))
                .and_then(|r| r.register(Box::new(float::F32)))
                .and_then(|r| r.register(Box::new(kquant::Q4_K)))
                .and_then(|r| r.register(Box::new(kquant::Q6_K)))
                .and_then(|r| r.register(Box::new(kquant::Q8_0)))
                .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
                .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
                .expect("the built-in codecs carry distinct labels and families")
        })
    }

    pub fn codecs(&self) -> impl Iterator<Item = &dyn RepresentationCodec> {
        self.codecs.iter().map(|c| c.as_ref())
    }

    pub fn by_label(&self, label: &str) -> Option<&dyn RepresentationCodec> {
        self.codecs().find(|c| c.encoding_label() == label)
    }

    pub fn by_family(&self, family: &str) -> Option<&dyn RepresentationCodec> {
        self.codecs().find(|c| c.identity().family == family)
    }

    /// Registered labels, in registration order — what a refusal lists.
    pub fn labels(&self) -> Vec<String> {
        self.codecs()
            .map(|c| c.encoding_label().to_string())
            .collect()
    }

    pub fn families(&self) -> Vec<String> {
        self.codecs().map(|c| c.identity().family).collect()
    }

    /// The codec `label` names, or a refusal naming the registered ones.
    pub fn resolve(
        &self,
        label: &str,
        tensor: &str,
    ) -> Result<&dyn RepresentationCodec, CodecError> {
        self.by_label(label)
            .ok_or_else(|| CodecError::UnknownEncoding {
                tensor: tensor.into(),
                label: label.into(),
                registered: self.labels(),
            })
    }

    /// Refuse a pack this build cannot decode under the rules it was
    /// written under.
    ///
    /// Three refusals, told apart because their remedies differ: an
    /// unregistered family, a revision another build wrote, and a
    /// same-revision identity whose geometry disagrees with what that
    /// revision means — a corrupted or hand-edited index, not version
    /// skew.
    pub fn admit(&self, id: &CodecIdentity) -> Result<&dyn RepresentationCodec, CodecError> {
        let codec = self
            .by_family(&id.family)
            .ok_or_else(|| CodecError::UnknownFamily {
                family: id.family.clone(),
                registered: self.families(),
            })?;
        let want = codec.identity();
        if id.revision != want.revision {
            return Err(CodecError::AbiRevision {
                family: id.family.clone(),
                found: id.revision,
                implemented: want.revision,
            });
        }
        if *id != want {
            return Err(CodecError::AbiGeometry {
                family: id.family.clone(),
                revision: id.revision,
                declared: format!("{id:?}"),
            });
        }
        Ok(codec)
    }
}
