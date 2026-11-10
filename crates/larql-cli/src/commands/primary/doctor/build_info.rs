//! What this `larql` binary is: version, target and the compute backends
//! compiled in.

use super::report::{Check, Group};

/// Facts fixed when this binary was compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildFacts {
    pub version: String,
    pub target: String,
    /// Whether the Metal backend is compiled in (`gpu` feature on macOS).
    pub metal: bool,
    /// Whether the `research` tooling (`larql dev`, …) is compiled in.
    pub research: bool,
}

impl BuildFacts {
    /// This binary's facts.
    pub(crate) fn current() -> Self {
        BuildFacts {
            version: env!("CARGO_PKG_VERSION").to_string(),
            target: env!("LARQL_BUILD_TARGET").to_string(),
            metal: cfg!(all(feature = "gpu", target_os = "macos")),
            research: cfg!(feature = "research"),
        }
    }
}

pub(crate) fn checks(build: &BuildFacts) -> Vec<Check> {
    let backend = if build.metal {
        "Metal + CPU build"
    } else {
        "portable CPU build"
    };
    let research = if build.research {
        ", research tools"
    } else {
        ""
    };
    vec![Check::ok(
        Group::Larql,
        "larql",
        format!(
            "{} for {} ({backend}{research})",
            build.version, build.target
        ),
    )]
}
