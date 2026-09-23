//! **Dynamically loaded plugins** — the contract between a host (the
//! `larql` binary) and a shared library that registers representation
//! codecs and lowering providers this build does not ship.
//!
//! The codec plane (`docs/represent-codec-contract.md`) and the lowering
//! plane (LOWERING-PLUGIN-1) already take providers from outside the
//! tree, but only as crates linked at compile time: a caller constructs
//! a [`CodecRegistry`] or a [`LoweringRegistry`] and registers into it.
//! This module is the same registration, reached through `dlopen` instead
//! of the linker, so a CLI can be handed a plugin on its command line.
//!
//! Rust has no stable ABI. A plugin and its host exchange Rust trait
//! objects, which is sound only when both were compiled by the same
//! compiler from the same `larql-vindex` source. [`ABI`] states both,
//! and the host compares it — through a C-ABI function, which is safe to
//! call whatever the plugin was built with — before it calls anything
//! Rust-typed. A plugin built from a different compiler or a different
//! larql commit is refused, never tried. (Uncommitted edits to the same
//! commit are not detected; build the plugin and the host from one
//! checkout.)
//!
//! What crosses is registration only. The plugin keeps its own statically
//! linked copy of this crate, so process-wide state inside it (a
//! `OnceLock` registry, a ledger) is the plugin's copy, not the host's:
//! a plugin reports through the values it returns, never through
//! globals. A loaded library is never unloaded — its vtables and
//! `&'static` labels outlive every registry that holds them.
//!
//! A plugin exports its registration with [`larql_plugin!`]:
//!
//! ```ignore
//! fn register(r: &mut larql_vindex::format::vindex3::plugin::PluginRegistrar) {
//!     r.codec(Box::new(MyCodec));
//!     r.lowering(|| Box::new(MyProvider::new()));
//! }
//! larql_vindex::larql_plugin!(register);
//! ```
//!
//! [`CodecRegistry`]: super::represent::codec::CodecRegistry
//! [`LoweringRegistry`]: super::opplan::exec::lowering::LoweringRegistry

use super::opplan::exec::backend::PlanBackend;
use super::represent::codec::RepresentationCodec;

/// The host/plugin compatibility stamp: contract revision, crate version,
/// compiler, and source commit. NUL-terminated so the C-ABI export can
/// hand it out as-is.
pub const ABI: &str = concat!(
    "larql-plugin/1 larql-vindex/",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("LARQL_PLUGIN_RUSTC"),
    ") commit ",
    env!("LARQL_PLUGIN_COMMIT"),
    "\0"
);

/// [`ABI`] without its terminator, for comparison and display.
pub fn abi() -> &'static str {
    ABI.trim_end_matches('\0')
}

/// Whether `abi` is one this host can trust: identical to its own, and
/// not built from an unknown commit (two unknowns are not evidence of
/// one source).
pub fn abi_compatible(abi: &str) -> bool {
    abi == self::abi() && !abi.ends_with("commit unknown")
}

/// Symbol a plugin exports: `extern "C" fn() -> *const c_char`, its [`ABI`].
pub const ABI_SYMBOL: &[u8] = b"larql_plugin_abi\0";
/// Symbol a plugin exports: [`RegisterFn`], called once after the ABI check.
pub const REGISTER_SYMBOL: &[u8] = b"larql_plugin_register\0";

/// The C-ABI stamp function — safe to call before compatibility is known.
pub type AbiFn = unsafe extern "C" fn() -> *const std::ffi::c_char;
/// The registration function. Rust ABI: called only after [`abi_compatible`].
pub type RegisterFn = fn(&mut PluginRegistrar);

/// Builds a lowering provider. A factory rather than an instance because
/// a host composes a fresh [`LoweringRegistry`] per command and each one
/// owns the providers registered into it.
///
/// [`LoweringRegistry`]: super::opplan::exec::lowering::LoweringRegistry
pub type LoweringFactory = fn() -> Box<dyn PlanBackend + Send>;

/// What a plugin registers, collected by the host.
#[derive(Default)]
pub struct PluginRegistrar {
    codecs: Vec<Box<dyn RepresentationCodec>>,
    lowerings: Vec<LoweringFactory>,
}

impl PluginRegistrar {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a representation codec. Duplicates are refused where the
    /// host composes its [`CodecRegistry`](super::represent::codec::CodecRegistry),
    /// by label and family, exactly as for a linked codec.
    pub fn codec(&mut self, codec: Box<dyn RepresentationCodec>) {
        self.codecs.push(codec);
    }

    /// Register a lowering provider, as the factory that builds it.
    pub fn lowering(&mut self, factory: LoweringFactory) {
        self.lowerings.push(factory);
    }

    /// The registrations, in the order the plugin made them.
    pub fn into_parts(self) -> (Vec<Box<dyn RepresentationCodec>>, Vec<LoweringFactory>) {
        (self.codecs, self.lowerings)
    }
}

/// Export a plugin's [`ABI`] and registration function under the symbols
/// a host looks up. `$register` is a `fn(&mut PluginRegistrar)`.
#[macro_export]
macro_rules! larql_plugin {
    ($register:path) => {
        #[no_mangle]
        pub extern "C" fn larql_plugin_abi() -> *const ::std::ffi::c_char {
            $crate::format::vindex3::plugin::ABI.as_ptr().cast()
        }

        #[no_mangle]
        pub fn larql_plugin_register(
            registrar: &mut $crate::format::vindex3::plugin::PluginRegistrar,
        ) {
            let register: $crate::format::vindex3::plugin::RegisterFn = $register;
            register(registrar)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_is_terminated_and_names_its_parts() {
        assert!(ABI.ends_with('\0'));
        assert_eq!(ABI.matches('\0').count(), 1);
        let abi = abi();
        assert!(abi.starts_with("larql-plugin/1 larql-vindex/"), "{abi}");
        assert!(abi.contains("rustc"), "{abi}");
        assert!(abi.contains(" commit "), "{abi}");
    }

    #[test]
    fn only_the_hosts_own_stamp_is_compatible() {
        assert_eq!(abi_compatible(abi()), !abi().ends_with("commit unknown"));
        assert!(!abi_compatible(""));
        assert!(!abi_compatible(&format!("{} ", abi())));
        assert!(!abi_compatible(
            "larql-plugin/1 larql-vindex/0.0.0 (rustc x) commit unknown"
        ));
    }
}
