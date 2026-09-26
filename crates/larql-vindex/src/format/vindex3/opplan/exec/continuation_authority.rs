//! **What a continuation provider was asked to do, and who answers for
//! the state it produced.**
//!
//! C2 of CONTINUATION-PLUGIN-1
//! (`docs/represent/forecasts/continuation-plugin-1.json`). A provider's
//! identity says which code holds the state; its configuration says what
//! that code was asked to do (`bits=4`, a residency). One implementation
//! under two configurations is two interpretations of the same stored
//! bytes, so the forecast makes configuration a SECOND authority beside
//! the identity rather than folding it into the name.
//! [`ContinuationAuthority`] is the pair; C4 carries it in every durable
//! handoff and refuses a resume under any other.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

use super::continuation_identity::ContinuationIdentity;

/// Domain separator for [`ConfigDigest`]: bump it if the canonical form
/// below ever changes, so an old digest can never collide with a new one.
const DIGEST_DOMAIN: &str = "larql-continuation-config/1";

/// A provider's configuration: `key=value` options, held in key order so
/// the order a caller wrote them in cannot change what they mean.
///
/// Parsing checks only the SHAPE (a key, an `=`, no duplicates, nothing
/// that would make the canonical form ambiguous). Whether a key means
/// anything is the provider's question
/// ([`ContinuationFactory::validate_config`](super::continuation_registry::ContinuationFactory::validate_config)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContinuationConfig {
    options: BTreeMap<String, String>,
}

impl ContinuationConfig {
    /// No options: what every built-in provider takes.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parse `key=value` options, refusing a missing `=`, an empty key, a
    /// key outside `[A-Za-z0-9_-]`, a newline in a value (the canonical
    /// form is newline-separated) and a key given twice (which of the two
    /// was meant has no answer).
    pub fn parse<S: AsRef<str>>(options: &[S]) -> Result<Self, String> {
        let mut parsed = BTreeMap::new();
        for option in options {
            let option = option.as_ref();
            let (key, value) = option
                .split_once('=')
                .ok_or_else(|| format!("continuation option `{option}` is not `key=value`"))?;
            if key.is_empty() {
                return Err(format!("continuation option `{option}` has an empty key"));
            }
            if let Some(bad) = key
                .chars()
                .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
            {
                return Err(format!(
                    "continuation option key `{key}` contains `{bad}`; only ASCII letters, \
                     digits, `_` and `-` name an option"
                ));
            }
            if value.contains('\n') {
                return Err(format!(
                    "continuation option `{key}` has a newline in its value"
                ));
            }
            if parsed.insert(key.to_string(), value.to_string()).is_some() {
                return Err(format!("continuation option `{key}` is given twice"));
            }
        }
        Ok(Self { options: parsed })
    }

    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str)
    }

    /// The option keys, in key order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.options.keys().map(String::as_str)
    }

    /// One `key=value` per line, in key order: the form the digest hashes.
    pub fn canonical_form(&self) -> String {
        self.options
            .iter()
            .map(|(key, value)| format!("{key}={value}\n"))
            .collect()
    }
}

/// SHA-256 over an identity and a configuration's canonical form,
/// displayed `sha256:<hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ConfigDigest(String);

impl ConfigDigest {
    pub fn of(identity: &ContinuationIdentity, config: &ContinuationConfig) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(DIGEST_DOMAIN.as_bytes());
        hasher.update(b"\n");
        hasher.update(identity.to_string().as_bytes());
        hasher.update(b"\n");
        hasher.update(config.canonical_form().as_bytes());
        let hex: String = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Self(format!("sha256:{hex}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConfigDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Who answers for a conversation's continuation state: the provider that
/// holds it and the configuration it was asked to hold it under. Two
/// fields compared separately, never one folded name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ContinuationAuthority {
    pub identity: ContinuationIdentity,
    pub config_digest: ConfigDigest,
}

impl ContinuationAuthority {
    pub fn new(identity: ContinuationIdentity, config: &ContinuationConfig) -> Self {
        let config_digest = ConfigDigest::of(&identity, config);
        Self {
            identity,
            config_digest,
        }
    }
}

impl fmt::Display for ContinuationAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.identity, self.config_digest)
    }
}
