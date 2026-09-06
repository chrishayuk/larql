//! **What a certificate certifies** — a radius, and the metric and domain
//! that give it meaning.
//!
//! A bare number is not a guarantee. `0.0045` is a relative RMS over
//! finite normal values, or a maximum absolute error over everything, or
//! something a provider this build has never heard of; those are different
//! promises, and adding two of them together produces a promise nobody
//! made. Composition is where that stops being pedantry: a parent
//! representation whose fidelity depends on its dependency's must add the
//! two, and it may only do so when they are stated in the same terms.
//!
//! So a certificate carries three things and validates them at
//! construction:
//!
//! ```text
//! metric   what is measured        relative-rms@1
//! domain   over which values       finite-normals@1
//! radius   the bound, in that metric, over that domain
//! ```
//!
//! The identities are VERSIONED SEMANTIC IDS, not an enum. A closed enum
//! would limit the plugin plane to the metrics this build happens to ship:
//! a provider certifying in a metric LARQL has never heard of must be able
//! to SAY so, and be refused for incompatibility rather than for being
//! unspellable. What this build refuses is a malformed id, an unusable
//! radius, and — at composition — ids that differ.
//!
//! v1's composition rule is EXACT IDENTITY: same metric, same domain, or
//! no composed certificate. That is a rule about what this build INFERS,
//! not a claim about mathematics — a bound over a wider domain may well
//! hold over a narrower one, and often does. v1 declines to derive that,
//! because deciding which domain contains which is a lattice this
//! vocabulary does not yet describe, and a wrong containment silently
//! widens a guarantee. Domain-subset reasoning is future work; until it
//! arrives, a caller who knows two domains are related states the bound
//! in the terms it needs rather than asking composition to convert.

use std::fmt;

use super::error::CodecError;

/// The separator between a semantic id's name and its version.
const VERSION_SEP: char = '@';

/// A versioned semantic identity: `name@version`.
///
/// Constructed through validation and read through accessors, so a later
/// field cannot break every provider that spelled one out.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticId {
    name: String,
    version: u32,
}

impl SemanticId {
    /// Validate and mint an id.
    ///
    /// A name is non-empty, has no whitespace and no `@` of its own — it
    /// has to survive being written into a container and read back as one
    /// token.
    pub fn new(kind: &str, name: impl Into<String>, version: u32) -> Result<Self, CodecError> {
        let name = name.into();
        if name.is_empty()
            || name.chars().any(char::is_whitespace)
            || name.contains(VERSION_SEP)
            || !name.is_ascii()
        {
            return Err(CodecError::MalformedSemanticId {
                kind: kind.into(),
                given: name,
            });
        }
        Ok(Self { name, version })
    }

    /// Read `name@version`.
    pub fn parse(kind: &str, text: &str) -> Result<Self, CodecError> {
        let (name, version) =
            text.rsplit_once(VERSION_SEP)
                .ok_or_else(|| CodecError::MalformedSemanticId {
                    kind: kind.into(),
                    given: text.into(),
                })?;
        let version = version
            .parse::<u32>()
            .map_err(|_| CodecError::MalformedSemanticId {
                kind: kind.into(),
                given: text.into(),
            })?;
        Self::new(kind, name, version)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.version
    }
}

impl fmt::Display for SemanticId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{VERSION_SEP}{}", self.name, self.version)
    }
}

/// What a radius measures.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetricId(SemanticId);

impl MetricId {
    pub const KIND: &'static str = "metric";

    pub fn new(name: impl Into<String>, version: u32) -> Result<Self, CodecError> {
        SemanticId::new(Self::KIND, name, version).map(Self)
    }

    pub fn parse(text: &str) -> Result<Self, CodecError> {
        SemanticId::parse(Self::KIND, text).map(Self)
    }

    /// Root-mean-square of the per-element relative error — what every
    /// certificate this build ships is stated in.
    pub fn relative_rms() -> Self {
        Self::new("relative-rms", 1).expect("a spelled constant is well formed")
    }

    pub fn id(&self) -> &SemanticId {
        &self.0
    }
}

impl fmt::Display for MetricId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Which values a radius is measured over.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DomainId(SemanticId);

impl DomainId {
    pub const KIND: &'static str = "domain";

    pub fn new(name: impl Into<String>, version: u32) -> Result<Self, CodecError> {
        SemanticId::new(Self::KIND, name, version).map(Self)
    }

    pub fn parse(text: &str) -> Result<Self, CodecError> {
        SemanticId::parse(Self::KIND, text).map(Self)
    }

    /// Finite normal values: the domain a relative bound can describe.
    /// Subnormals, zeroes, infinities and NaNs are outside it, and a
    /// codec says what it does with them in its own documentation.
    pub fn finite_normals() -> Self {
        Self::new("finite-normals", 1).expect("a spelled constant is well formed")
    }

    pub fn id(&self) -> &SemanticId {
        &self.0
    }
}

impl fmt::Display for DomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A reconstruction bound, and what it is a bound ON.
///
/// No public fields: a certificate gains metadata over time — a sample
/// size, a confidence, a provenance — and a provider that spelled one out
/// positionally would break on the first addition.
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityCertificate {
    metric: MetricId,
    domain: DomainId,
    radius: f64,
}

impl FidelityCertificate {
    /// Validate and mint a certificate. A radius is finite and not
    /// negative; `0.0` is the exact reconstruction's honest answer.
    pub fn new(metric: MetricId, domain: DomainId, radius: f64) -> Result<Self, CodecError> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(CodecError::MalformedRadius {
                metric: metric.to_string(),
                radius,
            });
        }
        Ok(Self {
            metric,
            domain,
            radius,
        })
    }

    /// The relative-RMS-over-finite-normals certificate this build's own
    /// codecs state — spelled once so they cannot drift apart.
    pub fn relative_rms(radius: f64) -> Result<Self, CodecError> {
        Self::new(MetricId::relative_rms(), DomainId::finite_normals(), radius)
    }

    pub fn metric(&self) -> &MetricId {
        &self.metric
    }

    pub fn domain(&self) -> &DomainId {
        &self.domain
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Whether two certificates are stated in the same terms — the only
    /// question v1 asks before composing.
    pub fn comparable_with(&self, other: &Self) -> bool {
        self.metric == other.metric && self.domain == other.domain
    }

    /// This certificate widened by `other`, in the shared metric and
    /// domain: the triangle bound a dependency's error adds to its
    /// owner's.
    ///
    /// Refuses when the terms differ. Treating a maximum-absolute bound
    /// as an L2 one would manufacture a guarantee nobody made; treating a
    /// bound over a wider domain as one over a narrower domain might well
    /// be sound, and v1 still declines, because it holds no description
    /// of which domain contains which. Exact identity is the rule, and
    /// carrying the metric and domain is what makes it checkable.
    pub fn widened_by(&self, other: &Self, tensor: &str, label: &str) -> Result<Self, CodecError> {
        if !self.comparable_with(other) {
            return Err(CodecError::IncomparableCertificates {
                tensor: tensor.into(),
                label: label.into(),
                own: format!("{} over {}", self.metric, self.domain),
                other: format!("{} over {}", other.metric, other.domain),
            });
        }
        Self::new(
            self.metric.clone(),
            self.domain.clone(),
            self.radius + other.radius,
        )
    }

    /// How a refusal or a report spells a certificate.
    pub fn describe(&self) -> String {
        format!("{:.3e} {} over {}", self.radius, self.metric, self.domain)
    }
}
