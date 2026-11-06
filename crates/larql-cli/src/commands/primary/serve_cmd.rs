//! `larql serve`: resolve the model name, then exec into the
//! `larql-server` binary with every other argument untouched.
//!
//! The server's own clap struct (`larql_server::bootstrap::Cli`) is the
//! only definition of its flags. This command used to mirror them in a
//! second struct and rebuild the argv field by field, and the mirror
//! drifted: 18 server flags were unreachable, `--warmup-hnsw` always
//! failed because its required `--hnsw` could not be passed, and the
//! mirror's hardcoded defaults overrode the server's. Forwarding the
//! arguments verbatim makes that drift impossible. `larql serve --help`
//! is forwarded too, so it prints the server's own flag list.
//!
//! The one argument this command reads is the model: when the first
//! argument is not a flag, it is the vindex path or name, and it is
//! resolved (cache shorthands, `hf://`, registry names) before the exec.
//! A model named after a flag is passed through unresolved, because
//! telling a positional from a flag's value needs the server's schema.
//!
//! When the model was given by *name* (a cache shorthand, `owner/name`,
//! `hf://…` or a registry name — anything but an existing directory), the
//! server only ever sees the resolved path, and would answer to nothing but
//! the id inside the container (`Qwen3-0.6B` for a model served as
//! `qwen3-0.6b`). So the name is also forwarded as `--model-alias`, and a
//! request's `"model"` may use either. An explicit `--model-alias` wins.

#[derive(clap::Args)]
pub(crate) struct ServeArgs {
    /// [VINDEX_PATH] [SERVER FLAGS...] — run `larql serve --help` for the
    /// server's flags.
    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub(crate) args: Vec<String>,
}

/// The server flag naming another name the served model answers to.
const MODEL_ALIAS_FLAG: &str = "--model-alias";

/// Whether an argument is a flag rather than the model positional.
fn is_flag(arg: &str) -> bool {
    arg.starts_with('-')
}

/// Whether the caller already chose an alias, as `--model-alias X` or
/// `--model-alias=X`.
fn has_explicit_alias(args: &[String]) -> bool {
    args.iter().any(|a| {
        a == MODEL_ALIAS_FLAG
            || a.strip_prefix(MODEL_ALIAS_FLAG)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

pub(crate) fn serve_command_args(
    args: &ServeArgs,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    forward_args(&args.args, super::serve_resolve::resolve_serve_target)
}

/// Testable core of [`serve_command_args`], with the model resolver
/// injected so the forwarding rules are proven without a cache.
fn forward_args(
    args: &[String],
    resolve: impl FnOnce(&str) -> Result<String, Box<dyn std::error::Error>>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut cmd_args = args.to_vec();
    let explicit_alias = has_explicit_alias(&cmd_args);
    let mut served_name = None;
    if let Some(model) = cmd_args.first_mut().filter(|a| !is_flag(a)) {
        if !std::path::Path::new(model.as_str()).is_dir() {
            served_name = Some(model.clone());
        }
        // Resolve cache shorthands / owner-name / hf:// → actual path so
        // `larql serve gemma3-4b-v2` works the same as `larql run`. A
        // name the VINDEX3 registry has claimed resolves through it
        // exclusively (no fallback on failure); everything else keeps
        // the cache-shorthand/hf:///local-path behaviour, and a real
        // resolution failure propagates instead of being silently
        // replaced by the raw, unresolved string. See
        // `commands::primary::serve_resolve` module docs.
        *model = resolve(model)?;
    }
    if let Some(name) = served_name.filter(|_| !explicit_alias) {
        cmd_args.splice(1..1, [MODEL_ALIAS_FLAG.to_string(), name]);
    }
    Ok(cmd_args)
}

pub(crate) fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cmd_args = serve_command_args(&args)?;
    let exe = std::env::current_exe().ok();
    let server_bin = exe
        .as_ref()
        .and_then(|e| e.parent())
        .map(|d| d.join("larql-server"))
        .filter(|p| p.exists());

    let bin = server_bin
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "larql-server".into());

    let status = std::process::Command::new(&bin).args(&cmd_args).status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("larql-server exited with: {s}").into()),
        Err(e) => {
            eprintln!("Failed to exec larql-server: {e}");
            eprintln!(
                "Make sure larql-server is installed (cargo install --path crates/larql-server)"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests;
