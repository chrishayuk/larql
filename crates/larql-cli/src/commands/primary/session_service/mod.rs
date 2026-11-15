//! **`larql serve-session` — a restricted LARQL session over a Unix socket.**
//!
//! The service binds exactly one VINDEX3 container, named by its policy, at start, and never another. It is not
//! an LQL shell: it accepts no LQL text, only typed requests, one JSON object per line ([`protocol`]), each one
//! of the operations the policy allows. It is the out-of-process transport of the Python bridge
//! (`larql.connect(socket)`): a client gets every weight, activation, logit and generation from it, and changes
//! the model only through it, without being able to open the model files itself, *if* the operating system
//! keeps them from the client. This service does not do that. See `crates/larql-cli/README.md`.
//!
//! What it enforces itself:
//! - the one container, bound read-only at start ([`service::Service::start`]);
//! - the allowed operations ([`policy`]);
//! - every path a request names stays inside the workdir, symlinks and hard links refused ([`paths`]);
//! - the budget of edited-model executions, and the resource limits ([`ops`]);
//! - an append-only audit record of every request and response ([`audit`]);
//! - one client at a time ([`server`]).

pub mod audit;
pub mod npy;
pub mod ops;
pub mod paths;
pub mod policy;
pub mod protocol;
pub mod server;
pub mod service;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::Args;

#[derive(Args, Debug)]
pub struct ServeSessionArgs {
    /// The policy file (JSON, `"policy": "larql-session-policy/v1"`).
    #[arg(long, value_name = "POLICY.json")]
    pub config: PathBuf,

    /// The Unix socket to listen on. It must not exist; it is removed at shutdown.
    #[arg(long, value_name = "PATH")]
    pub socket: PathBuf,
}

/// Set by SIGTERM and SIGINT.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Route SIGTERM and SIGINT to [`SHUTDOWN`].
pub fn install_signal_handlers() {
    let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
    }
}

/// Start, serve until signalled, and shut down: the stop recorded, the socket removed.
pub fn run(args: ServeSessionArgs) -> Result<(), Box<dyn std::error::Error>> {
    let policy = policy::Policy::read(&args.config)?;
    let mut service = service::Service::start(policy)?;
    let socket = server::BoundSocket::bind(&args.socket)?;
    install_signal_handlers();
    eprintln!(
        "larql serve-session: listening on {} ({})",
        args.socket.display(),
        protocol::PROTOCOL
    );
    let stopped = server::serve(&mut service, &socket.listener, &SHUTDOWN);
    service.stop(&stopped.reason());
    socket.remove();
    eprintln!("larql serve-session: stopped ({})", stopped.reason());
    match stopped {
        server::Stopped::Shutdown => Ok(()),
        server::Stopped::AuditFailed(why) => Err(why.into()),
    }
}

#[cfg(test)]
mod tests;
