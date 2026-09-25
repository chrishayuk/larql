//! **The C5 continuation provider, as a plugin** (CONTINUATION-PLUGIN-1
//! C6).
//!
//! A test fixture. The provider is not re-written here: it is the file
//! the C5 proof uses, included by path, so the dylib that C6 loads and
//! the provider C5 held bit-identical to `canonical/v1` are one source.
//! The only code of this crate's own is the registration below, through
//! the exported plugin ABI.

// The C5 file also carries the pieces only C5's controls use (the
// corrupting sibling, the unselected build); a plugin registers one
// factory and leaves them unused.
#[allow(dead_code)]
#[path = "../../larql-kv/tests/external_continuation_provider/provider.rs"]
mod provider;

use larql_vindex::format::vindex3::plugin::PluginRegistrar;

fn register(registrar: &mut PluginRegistrar) {
    registrar.continuation(Box::new(provider::HostileFactory::new(
        provider::HOSTILE_REVISION,
    )));
}

larql_vindex::larql_plugin!(register);
