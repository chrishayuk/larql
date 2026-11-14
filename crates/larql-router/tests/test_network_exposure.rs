//! The router binds loopback by default and refuses to serve its grid
//! unauthenticated on a network address unless explicitly told to — the
//! same convention `larql-server` follows with `--api-key` /
//! `--insecure-public`.

use larql_router::exposure::{
    is_loopback_host, BindRequest, DEFAULT_HOST, GRID_KEY_ENV, INSECURE_PUBLIC_FLAG,
};

const KEY: &str = "shared-secret";
const NETWORK_HOSTS: [&str; 4] = ["0.0.0.0", "::", "192.168.1.10", "myhost.local"];
const LOOPBACK_HOSTS: [&str; 3] = [DEFAULT_HOST, "127.0.0.2", "::1"];

fn grid(host: &str) -> BindRequest<'_> {
    BindRequest {
        host,
        grid_enabled: true,
        grid_key: None,
        insecure_public: false,
    }
}

#[test]
fn the_default_bind_is_loopback() {
    assert!(is_loopback_host(DEFAULT_HOST));
    assert!(grid(DEFAULT_HOST).check_network_exposure().is_ok());
}

#[test]
fn loopback_binds_need_no_key() {
    for host in LOOPBACK_HOSTS {
        assert!(grid(host).check_network_exposure().is_ok(), "{host}");
        let static_only = BindRequest {
            grid_enabled: false,
            ..grid(host)
        };
        assert!(static_only.check_network_exposure().is_ok(), "{host}");
    }
}

#[test]
fn a_keyless_network_grid_bind_is_refused() {
    for host in NETWORK_HOSTS {
        assert!(!is_loopback_host(host), "{host}");
        let err = grid(host).check_network_exposure().unwrap_err();
        assert!(err.contains(INSECURE_PUBLIC_FLAG), "{host}: {err}");
        assert!(err.contains(GRID_KEY_ENV), "{host}: {err}");
        assert!(err.contains(host), "{host}: {err}");
    }
}

#[test]
fn a_network_grid_bind_is_allowed_with_a_key() {
    for host in NETWORK_HOSTS {
        let keyed = BindRequest {
            grid_key: Some(KEY),
            ..grid(host)
        };
        assert!(keyed.check_network_exposure().is_ok(), "{host}");
    }
}

#[test]
fn an_empty_key_is_no_key() {
    let empty = BindRequest {
        grid_key: Some(""),
        ..grid(NETWORK_HOSTS[0])
    };
    assert!(empty.check_network_exposure().is_err());
}

#[test]
fn a_network_bind_is_allowed_with_the_explicit_opt_in() {
    for grid_enabled in [true, false] {
        let opted_in = BindRequest {
            grid_enabled,
            insecure_public: true,
            ..grid(NETWORK_HOSTS[0])
        };
        assert!(opted_in.check_network_exposure().is_ok());
    }
}

/// Without a grid listener the router has no credential at all, so a key
/// cannot stand in for the opt-in: it would guard nothing.
#[test]
fn a_static_only_network_bind_needs_the_opt_in_even_with_a_key() {
    for grid_key in [None, Some(KEY)] {
        let static_only = BindRequest {
            grid_enabled: false,
            grid_key,
            ..grid(NETWORK_HOSTS[0])
        };
        let err = static_only.check_network_exposure().unwrap_err();
        assert!(err.contains(INSECURE_PUBLIC_FLAG), "{err}");
        assert!(!err.contains(GRID_KEY_ENV), "{err}");
    }
}
