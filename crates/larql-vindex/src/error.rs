use std::path::PathBuf;

use crate::config::ExtractLevel;

#[derive(Debug, thiserror::Error)]
pub enum VindexError {
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("unsupported architecture '{family}' for {surface}: {feature} is not implemented")]
    UnsupportedArchitecture {
        family: String,
        feature: String,
        surface: String,
    },
    #[error("requires extract level '{needed}' but vindex was built at '{have}'")]
    InsufficientExtractLevel {
        needed: ExtractLevel,
        have: ExtractLevel,
    },
    #[error(
        "index.json declares version {found}, which this binary does not support; \
         supported: {supported}"
    )]
    UnknownContainerGeneration { found: u32, supported: String },

    #[error("this is a {found} index; that path requires the {required} loader")]
    WrongContainerGeneration {
        found: &'static str,
        required: &'static str,
    },

    /// A legacy bank-shape container reached a surface that reads only the
    /// normative graph shape (ADR-0027).
    #[error(
        "this is a legacy bank-shape VINDEX3 container (moe_manifest + LYRW segments, no \
         system graph), not a VINDEX3 3.0 container (ADR-0027); {}",
        crate::format::vindex3::LEGACY_BANK_MIGRATION
    )]
    LegacyBankContainer,

    /// A container named a file outside itself — an absolute path, a
    /// `..` component, or a name that would mean a different file on a
    /// different host. Refused before any path is composed from it.
    #[error(
        "the container names `{name}` as one of its files, which is refused: {reason}; \
         every name a container uses must be a relative path inside it"
    )]
    UnsafeContainerPath { name: String, reason: &'static str },

    /// A downloaded container's segment bytes are not the bytes its own
    /// index seals. Transport integrity is the host's; this is the
    /// container's word about itself, checked on arrival.
    #[error(
        "the container at {} does not match its own index: {}; remove it from the \
         download cache and fetch it again",
        container.display(),
        mismatches.join("; ")
    )]
    PayloadDigestMismatch {
        container: PathBuf,
        mismatches: Vec<String>,
    },

    /// A verb that writes a new vindex directory was pointed at a
    /// destination it must not touch: the source, something containing or
    /// inside it, a symlink, or a path that already holds anything.
    /// Refused before any write (`format::fresh_output`).
    #[error("refusing to write output `{}`: {reason}", out.display())]
    UnsafeOutputDestination { out: PathBuf, reason: String },

    /// Staging or promoting a new vindex directory failed at the
    /// filesystem. Nothing was left at the destination.
    #[error("{step} {}: {source}", path.display())]
    OutputStaging {
        step: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model error: {0}")]
    Model(#[from] larql_models::ModelError),
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILY_DEEPSEEK: &str = "deepseek";
    const FEATURE_MLA: &str = "multi-head latent attention (MLA)";
    const SURFACE_Q4K_WEIGHT_WRITER: &str = "q4k weight writer";

    #[test]
    fn not_a_directory_includes_path() {
        let e = VindexError::NotADirectory("/tmp/missing".into());
        let s = e.to_string();
        assert!(s.contains("not a directory"), "{s}");
        assert!(s.contains("missing"), "{s}");
    }

    #[test]
    fn no_safetensors_includes_path() {
        let e = VindexError::NoSafetensors("/data/model".into());
        let s = e.to_string();
        assert!(s.contains("no safetensors"), "{s}");
        assert!(s.contains("model"), "{s}");
    }

    #[test]
    fn missing_tensor_includes_name() {
        let e = VindexError::MissingTensor("model.embed_tokens.weight".into());
        let s = e.to_string();
        assert!(s.contains("missing tensor"), "{s}");
        assert!(s.contains("model.embed_tokens.weight"), "{s}");
    }

    #[test]
    fn parse_error_includes_message() {
        let e = VindexError::Parse("unexpected token at line 5".into());
        assert!(e.to_string().contains("unexpected token at line 5"));
    }

    #[test]
    fn unsupported_dtype_includes_type() {
        let e = VindexError::UnsupportedDtype("bfloat16".into());
        let s = e.to_string();
        assert!(s.contains("unsupported dtype"), "{s}");
        assert!(s.contains("bfloat16"), "{s}");
    }

    #[test]
    fn unsupported_architecture_includes_context() {
        let e = VindexError::UnsupportedArchitecture {
            family: FAMILY_DEEPSEEK.into(),
            feature: FEATURE_MLA.into(),
            surface: SURFACE_Q4K_WEIGHT_WRITER.into(),
        };
        let s = e.to_string();
        assert!(s.contains(FAMILY_DEEPSEEK), "{s}");
        assert!(s.contains(FEATURE_MLA), "{s}");
        assert!(s.contains(SURFACE_Q4K_WEIGHT_WRITER), "{s}");
    }

    #[test]
    fn insufficient_extract_level_shows_both_levels() {
        let e = VindexError::InsufficientExtractLevel {
            needed: ExtractLevel::Inference,
            have: ExtractLevel::Browse,
        };
        let s = e.to_string();
        assert!(s.contains("inference"), "{s}");
        assert!(s.contains("browse"), "{s}");
    }

    #[test]
    fn io_error_from_converts() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "oops");
        let e: VindexError = io.into();
        assert!(e.to_string().contains("IO error"));
    }
}
