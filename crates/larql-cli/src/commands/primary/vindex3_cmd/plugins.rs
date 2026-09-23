//! `--plugin` and `--lowering`: representation codecs and lowering
//! providers this build does not ship, loaded from shared libraries named
//! on the command line.
//!
//! The contract a plugin implements is `larql_vindex`'s
//! (`format::vindex3::plugin`): a C-ABI stamp checked before anything
//! Rust-typed is called, then one registration call. This module is the
//! host half — load, check, collect — and composes what it collected the
//! same way a linked provider is composed: codecs into a
//! [`CodecRegistry`] on top of the shipped ones, lowering providers into
//! the [`LoweringRegistry`](larql_vindex::format::vindex3::opplan::exec::lowering::LoweringRegistry)
//! a `--backend` names, in [`super::prepare::lowerings_with`]. Nothing is
//! discovered: a plugin not named on the command line is not loaded.

use std::ffi::CStr;
use std::path::PathBuf;

use clap::Args;
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::plugin::{
    self, AbiFn, LoweringFactory, PluginRegistrar, RegisterFn,
};
use larql_vindex::format::vindex3::represent::codec::CodecRegistry;

type BoxErr = Box<dyn std::error::Error>;

#[derive(Args, Debug, Clone, Default)]
pub struct PluginArgs {
    /// Load a larql plugin: a shared library (`.dylib`/`.so`) exporting
    /// `larql_plugin_register`, which registers representation codecs and
    /// lowering providers. Repeatable. The plugin must be built by the
    /// same compiler from the same larql commit as this binary, or it is
    /// refused before it runs.
    #[arg(long = "plugin", value_name = "PATH")]
    pub plugins: Vec<PathBuf>,

    /// Execute on the lowering provider with this identity
    /// (`family/vN`, e.g. one a `--plugin` registered) instead of the one
    /// `--backend` names. `--backend` still decides which stored
    /// representation is asked for.
    #[arg(long, value_name = "FAMILY/vN")]
    pub lowering: Option<String>,
}

/// What the command line loaded: the codec registry every open decodes
/// through, the lowering providers to add to whatever `--backend`
/// composes, and the provider `--lowering` asked for.
pub(crate) struct Plugins {
    pub(crate) codecs: &'static CodecRegistry,
    pub(crate) lowerings: Vec<LoweringFactory>,
    pub(crate) select: Option<LoweringIdentity>,
}

impl Plugins {
    /// No plugins: the shipped codecs, the shipped providers.
    pub(crate) fn none() -> Self {
        Self {
            codecs: CodecRegistry::builtin(),
            lowerings: Vec::new(),
            select: None,
        }
    }

    /// Load every `--plugin`, in order, and parse `--lowering`.
    pub(crate) fn load(args: &PluginArgs) -> Result<Self, BoxErr> {
        let select = args
            .lowering
            .as_deref()
            .map(parse_lowering_identity)
            .transpose()?;
        if args.plugins.is_empty() {
            return Ok(Self {
                select,
                ..Self::none()
            });
        }
        let mut codecs = CodecRegistry::shipped();
        let mut lowerings = Vec::new();
        for path in &args.plugins {
            let registrar = load_one(path)?;
            let (plugin_codecs, plugin_lowerings) = registrar.into_parts();
            let labels: Vec<&str> = plugin_codecs.iter().map(|c| c.encoding_label()).collect();
            let identities: Vec<String> = plugin_lowerings
                .iter()
                .map(|f| f().identity().to_string())
                .collect();
            eprintln!(
                "plugin: {} — codecs [{}], lowerings [{}]",
                path.display(),
                labels.join(", "),
                identities.join(", ")
            );
            for codec in plugin_codecs {
                codecs = codecs
                    .register(codec)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
            lowerings.extend(plugin_lowerings);
        }
        Ok(Self {
            codecs: Box::leak(Box::new(codecs)),
            lowerings,
            select,
        })
    }
}

/// Open one library, check its stamp, and collect its registration.
#[cfg(unix)]
fn load_one(path: &std::path::Path) -> Result<PluginRegistrar, BoxErr> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| format!("--plugin {}: path contains a NUL byte", path.display()))?;
    // SAFETY: loading runs the library's initialisers; the user named it.
    // RTLD_LOCAL keeps its symbols out of the global namespace, so two
    // plugins never resolve each other's copies of larql.
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return Err(format!("--plugin {}: {}", path.display(), dl_error()).into());
    }
    let symbol = |name: &[u8]| -> Result<*mut libc::c_void, BoxErr> {
        // SAFETY: `name` is one of the NUL-terminated symbol constants.
        let sym = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
        if sym.is_null() {
            let name = String::from_utf8_lossy(&name[..name.len() - 1]);
            return Err(format!(
                "--plugin {}: not a larql plugin: no `{name}` ({})",
                path.display(),
                dl_error()
            )
            .into());
        }
        Ok(sym)
    };
    let abi = symbol(plugin::ABI_SYMBOL)?;
    // SAFETY: the stamp is a C-ABI function returning a NUL-terminated
    // static string — callable whatever compiler built the plugin.
    let stamp = unsafe {
        let abi: AbiFn = std::mem::transmute::<*mut libc::c_void, AbiFn>(abi);
        CStr::from_ptr(abi()).to_string_lossy().into_owned()
    };
    if !plugin::abi_compatible(&stamp) {
        return Err(format!(
            "--plugin {}: built for `{stamp}`, this binary is `{}` — rebuild the plugin \
             against this larql checkout with the same compiler",
            path.display(),
            plugin::abi()
        )
        .into());
    }
    let register = symbol(plugin::REGISTER_SYMBOL)?;
    let mut registrar = PluginRegistrar::new();
    // SAFETY: the stamps match, so the plugin's `PluginRegistrar` and
    // trait objects have this binary's layout.
    unsafe {
        let register: RegisterFn = std::mem::transmute::<*mut libc::c_void, RegisterFn>(register);
        register(&mut registrar);
    }
    // Never dlclose'd: the registrations hold its vtables and statics.
    Ok(registrar)
}

#[cfg(unix)]
fn dl_error() -> String {
    // SAFETY: dlerror returns a thread-local NUL-terminated string or null.
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "unknown dynamic-loader error".into()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

#[cfg(not(unix))]
fn load_one(path: &std::path::Path) -> Result<PluginRegistrar, BoxErr> {
    Err(format!(
        "--plugin {}: loading plugins is implemented for unix dlopen only",
        path.display()
    )
    .into())
}

/// Parse `family/vN` — the form a [`LoweringIdentity`] displays as.
pub(crate) fn parse_lowering_identity(spec: &str) -> Result<LoweringIdentity, BoxErr> {
    let parsed = spec.rsplit_once("/v").and_then(|(family, revision)| {
        revision
            .parse::<u32>()
            .ok()
            .map(|r| LoweringIdentity::new(family, r))
    });
    let identity = parsed.ok_or_else(|| {
        format!("--lowering `{spec}`: expected `family/vN`, e.g. `cpu-production/v1`")
    })?;
    identity.validate()?;
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowering_identity_round_trips_its_display() {
        let id = LoweringIdentity::new("example-lowering", 1);
        assert_eq!(parse_lowering_identity(&id.to_string()).unwrap(), id);
    }

    #[test]
    fn malformed_lowering_identities_refuse() {
        for bad in [
            "",
            "family",
            "family/v",
            "family/vx",
            "/v1",
            "fam ily/v1",
            "f/v0",
        ] {
            assert!(parse_lowering_identity(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_library_that_is_not_a_plugin_refuses_by_name() {
        let err = Plugins::load(&PluginArgs {
            plugins: vec![PathBuf::from("/nonexistent/libnot-a-plugin.dylib")],
            lowering: None,
        })
        .err()
        .unwrap()
        .to_string();
        assert!(err.contains("libnot-a-plugin"), "{err}");
    }
}
