//! The continuation providers this crate ships, as registry factories
//! (C2 of CONTINUATION-PLUGIN-1).

use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
};
use larql_vindex::format::vindex3::opplan::exec::kv::RowFactory;

use super::window::WindowFactory;
use super::CanonicalKvState;

/// Builds [`CanonicalKvState`]: every region the plan vocabulary
/// describes, no options.
#[derive(Debug, Clone, Copy, Default)]
pub struct CanonicalFactory;

impl ContinuationFactory for CanonicalFactory {
    fn identity(&self) -> ContinuationIdentity {
        CanonicalKvState::identity()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &ContinuationRegion::ALL
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        Box::new(CanonicalKvState::new())
    }
}

/// A fresh registry holding the providers LARQL ships — `row/v1`,
/// `canonical/v1` and `window/v1`. A new value on every call, never a process default:
/// a caller that wants more registers into what this returns.
pub fn shipped_continuations() -> ContinuationRegistry {
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Box::new(RowFactory))
        .expect("row/v1 is a valid, unique identity");
    registry
        .register(Box::new(CanonicalFactory))
        .expect("canonical/v1 is a valid, unique identity");
    registry
        .register(Box::new(WindowFactory))
        .expect("window/v1 is a valid, unique identity");
    registry
}
