//! Network-exposure policy for the router's listeners.
//!
//! Every listener the router opens — HTTP (`--port`), the gRPC grid
//! (`--grid-port`) and, with the `quic` feature, the QUIC grid
//! (`--quic-port`) — binds the same `--host`. The router binds loopback by
//! default and refuses to serve a keyless grid on a network address unless
//! the operator opts in with `--insecure-public`: the same convention
//! `larql-server` follows with `--api-key` / `--insecure-public`.
//!
//! The grid key is the router's only credential and it guards the grid
//! (registration, drain, assign). A router with no grid listener has no
//! credential at all, so on a network address it needs the explicit
//! opt-in; a key there would guard nothing and does not count.

use std::net::IpAddr;

/// Loopback by default: a router reachable from the network must be asked
/// for (see [`BindRequest::check_network_exposure`]).
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Flag that accepts a keyless network bind.
pub const INSECURE_PUBLIC_FLAG: &str = "--insecure-public";

/// Flag that supplies the grid key.
pub const GRID_KEY_FLAG: &str = "--grid-key";

/// Environment variable that supplies the grid key.
pub const GRID_KEY_ENV: &str = "LARQL_GRID_KEY";

/// True when `host` is an IP literal on the loopback interface. A hostname
/// that is not an IP literal counts as a network address, as in
/// `larql-server`.
pub fn is_loopback_host(host: &str) -> bool {
    host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The bind-relevant subset of the router's daemon flags.
#[derive(Debug, Clone, Copy)]
pub struct BindRequest<'a> {
    /// `--host`, shared by every listener.
    pub host: &'a str,
    /// Whether a grid listener (`--grid-port`) will be opened.
    pub grid_enabled: bool,
    /// `--grid-key` / `LARQL_GRID_KEY`. An empty key is no key.
    pub grid_key: Option<&'a str>,
    /// `--insecure-public`.
    pub insecure_public: bool,
}

impl BindRequest<'_> {
    fn has_grid_credential(&self) -> bool {
        self.grid_enabled && self.grid_key.is_some_and(|key| !key.is_empty())
    }

    /// Refuse an unauthenticated bind to a network address unless the
    /// operator opted in with `--insecure-public`. Returns the user-facing
    /// message on refusal.
    pub fn check_network_exposure(&self) -> Result<(), String> {
        if is_loopback_host(self.host) || self.insecure_public || self.has_grid_credential() {
            return Ok(());
        }
        let host = self.host;
        Err(if self.grid_enabled {
            format!(
                "refusing to serve the grid unauthenticated on {host}: pass {GRID_KEY_FLAG} \
                 (or set {GRID_KEY_ENV}), bind a loopback --host, or accept the exposure \
                 with {INSECURE_PUBLIC_FLAG}"
            )
        } else {
            format!(
                "refusing to serve unauthenticated on {host}: a router without a grid has \
                 no credential, so bind a loopback --host or accept the exposure with \
                 {INSECURE_PUBLIC_FLAG}"
            )
        })
    }
}
