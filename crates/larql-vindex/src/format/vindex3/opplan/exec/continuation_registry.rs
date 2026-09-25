//! **The continuation registry** — continuation providers as a value a
//! caller carries, selected against the plan before any prefill.
//!
//! C2 of CONTINUATION-PLUGIN-1
//! (`docs/represent/forecasts/continuation-plugin-1.json`). The lowering
//! registry holds configured INSTANCES because a lowering is a
//! process-level authority. Continuation state is per conversation, so
//! this registry holds FACTORIES: one registration builds a fresh provider
//! for every conversation that selects it, and no two conversations can
//! share mutable state through the registry.
//!
//! **A value, not a default.** There is no static registry and nothing
//! consults one when a caller says nothing; the codec plane (rung 3, F8)
//! and the lowering plane (F4) both found hidden built-in registries at
//! execution, and each was a place registration could not reach.
//!
//! **Capability, not discovery.** A factory declares which continuation
//! regions it can hold. [`ContinuationRegistry::select`] matches every
//! layer of the plan's geometry against that declaration and refuses
//! BEFORE a provider exists, naming the first layer it cannot hold —
//! rather than the provider discovering the gap mid-prefill through
//! [`ContinuationError`](super::kv::ContinuationError) (F5). Selection
//! compares identities for equality and regions by declaration; it never
//! branches on a family name (F4).

use std::fmt;
use std::sync::Arc;

use super::continuation::{region_name, LayerContinuationGeometry};
use super::continuation_authority::{ContinuationAuthority, ContinuationConfig};
use super::continuation_handoff::ContinuationHandoff;
use super::continuation_identity::ContinuationIdentity;
use super::kv::ContinuationProvider;
use crate::error::VindexError;

/// A provider as a factory builds it: owned by one conversation.
pub type BoxedContinuation = Box<dyn ContinuationProvider + Send>;

/// A kind of continuation state a layer can require — one per
/// [`LayerContinuationGeometry`] variant that requires anything.
///
/// No `Stateless` capability: a stateless layer asks nothing of a
/// provider, so every provider holds it and declaring it would be a
/// capability that could never be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ContinuationRegion {
    /// Sequence-indexed K/V row pairs ([`LayerContinuationGeometry::Kv`]).
    Kv,
    /// One operator-defined row per position ([`LayerContinuationGeometry::LatentKv`]).
    LatentKv,
    /// Fixed-size recurrent buffers ([`LayerContinuationGeometry::Recurrent`]).
    Recurrent,
    /// K/V rows and recurrent buffers on one layer
    /// ([`LayerContinuationGeometry::KvAndRecurrent`]) — its own capability,
    /// because holding each half separately is not holding the layer.
    KvAndRecurrent,
}

impl ContinuationRegion {
    /// Every region, in declaration order: what a provider that holds any
    /// continuation the plan vocabulary can describe declares.
    pub const ALL: [Self; 4] = [
        Self::Kv,
        Self::LatentKv,
        Self::Recurrent,
        Self::KvAndRecurrent,
    ];

    /// The region a layer requires; `None` for a stateless layer.
    pub fn required_by(geometry: &LayerContinuationGeometry) -> Option<Self> {
        match geometry {
            LayerContinuationGeometry::Kv(_) => Some(Self::Kv),
            LayerContinuationGeometry::LatentKv(_) => Some(Self::LatentKv),
            LayerContinuationGeometry::Recurrent(_) => Some(Self::Recurrent),
            LayerContinuationGeometry::KvAndRecurrent { .. } => Some(Self::KvAndRecurrent),
            LayerContinuationGeometry::Stateless => None,
        }
    }
}

impl fmt::Display for ContinuationRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Kv => "kv",
            Self::LatentKv => "latent-kv",
            Self::Recurrent => "recurrent",
            Self::KvAndRecurrent => "kv-and-recurrent",
        })
    }
}

/// What a continuation provider registers: who it is, what it can hold,
/// which configuration it accepts, and how to build one conversation's
/// state.
pub trait ContinuationFactory: Send + Sync {
    /// The provider's identity. Stable for the factory's lifetime.
    fn identity(&self) -> ContinuationIdentity;

    /// The regions a provider built by this factory can hold.
    fn regions(&self) -> &[ContinuationRegion];

    /// Accept or refuse a configuration, with the reason. The default
    /// takes no options at all, so a provider that reads none cannot be
    /// handed one it would silently ignore.
    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        match config.keys().collect::<Vec<_>>().as_slice() {
            [] => Ok(()),
            keys => Err(format!("takes no options; given `{}`", keys.join("`, `"))),
        }
    }

    /// A fresh provider for one conversation, under a configuration
    /// [`Self::validate_config`] accepted.
    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation;
}

/// Why the registry refused.
#[derive(Debug, thiserror::Error)]
pub enum ContinuationRegistryError {
    /// An identity that could not key a registry.
    #[error("{0}")]
    Invalid(VindexError),

    /// The same family and revision registered twice — refused rather than
    /// replaced, so the answer never depends on registration order.
    #[error("continuation provider `{identity}` is already registered")]
    Duplicate { identity: ContinuationIdentity },

    /// No provider under that identity; names every one that IS
    /// registered, so the remedy is in the refusal.
    #[error(
        "no continuation provider `{identity}` is registered; registered: {}",
        list(registered)
    )]
    Unregistered {
        identity: ContinuationIdentity,
        registered: Vec<ContinuationIdentity>,
    },

    /// The provider refused the configuration.
    #[error("continuation provider `{identity}` refused its configuration: {reason}")]
    Config {
        identity: ContinuationIdentity,
        reason: String,
    },

    /// A layer requires a region the provider does not declare — refused
    /// at selection, before any state exists.
    #[error(
        "continuation provider `{identity}` cannot hold layer {layer}, which keeps {requirement}; \
         it declares: {}",
        list(declared)
    )]
    Unsupported {
        identity: ContinuationIdentity,
        layer: usize,
        region: ContinuationRegion,
        requirement: &'static str,
        declared: Vec<ContinuationRegion>,
    },
}

fn list<T: fmt::Display>(items: &[T]) -> String {
    if items.is_empty() {
        return "none".into();
    }
    items
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

impl From<ContinuationRegistryError> for VindexError {
    fn from(e: ContinuationRegistryError) -> Self {
        match e {
            ContinuationRegistryError::Invalid(inner) => inner,
            other => VindexError::Parse(other.to_string()),
        }
    }
}

/// The continuation factories a caller carries, keyed by identity.
#[derive(Default, Clone)]
pub struct ContinuationRegistry {
    factories: Vec<Arc<dyn ContinuationFactory>>,
}

impl fmt::Debug for ContinuationRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuationRegistry")
            .field("identities", &self.identities())
            .finish()
    }
}

impl ContinuationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the identity it states, refusing an
    /// invalid identity or one already registered.
    pub fn register(
        &mut self,
        factory: Box<dyn ContinuationFactory>,
    ) -> Result<(), ContinuationRegistryError> {
        let identity = factory.identity();
        identity
            .validate()
            .map_err(ContinuationRegistryError::Invalid)?;
        if self.factories.iter().any(|f| f.identity() == identity) {
            return Err(ContinuationRegistryError::Duplicate { identity });
        }
        self.factories.push(Arc::from(factory));
        Ok(())
    }

    /// Every registered identity, in registration order.
    pub fn identities(&self) -> Vec<ContinuationIdentity> {
        self.factories.iter().map(|f| f.identity()).collect()
    }

    pub fn len(&self) -> usize {
        self.factories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Resolve `identity` under `config` for a program of `geometry`:
    /// the provider must be registered, accept the configuration, and
    /// declare every region a layer requires. Nothing is built until the
    /// caller asks the returned selection to.
    pub fn select(
        &self,
        identity: &ContinuationIdentity,
        config: &ContinuationConfig,
        geometry: &[LayerContinuationGeometry],
    ) -> Result<SelectedContinuation, ContinuationRegistryError> {
        let factory = self
            .factories
            .iter()
            .find(|f| &f.identity() == identity)
            .ok_or_else(|| ContinuationRegistryError::Unregistered {
                identity: identity.clone(),
                registered: self.identities(),
            })?;
        factory
            .validate_config(config)
            .map_err(|reason| ContinuationRegistryError::Config {
                identity: identity.clone(),
                reason,
            })?;
        let declared = factory.regions();
        for (layer, layer_geometry) in geometry.iter().enumerate() {
            let Some(region) = ContinuationRegion::required_by(layer_geometry) else {
                continue;
            };
            if !declared.contains(&region) {
                return Err(ContinuationRegistryError::Unsupported {
                    identity: identity.clone(),
                    layer,
                    region,
                    requirement: region_name(layer_geometry),
                    declared: declared.to_vec(),
                });
            }
        }
        Ok(SelectedContinuation {
            authority: ContinuationAuthority::new(identity.clone(), config),
            config: config.clone(),
            factory: Arc::clone(factory),
        })
    }
}

/// A provider chosen for a program: its authority, and the means to build
/// one conversation's state under it.
#[derive(Clone)]
pub struct SelectedContinuation {
    authority: ContinuationAuthority,
    config: ContinuationConfig,
    factory: Arc<dyn ContinuationFactory>,
}

impl fmt::Debug for SelectedContinuation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectedContinuation")
            .field("authority", &self.authority)
            .finish()
    }
}

impl SelectedContinuation {
    /// Who answers for state built from this selection.
    pub fn authority(&self) -> &ContinuationAuthority {
        &self.authority
    }

    /// A fresh provider for one conversation.
    pub fn build(&self) -> BoxedContinuation {
        self.factory.build(&self.config)
    }

    /// A fresh provider for one conversation, sealed with this selection's
    /// authority so the state can outlive the call and be resumed only
    /// under the same authority (C4).
    pub fn begin(&self) -> ContinuationHandoff {
        ContinuationHandoff::seal(self.authority.clone(), self.build())
    }
}
