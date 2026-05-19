//! Small pure helpers extracted from `main.rs` so the binary entry point
//! is a thin shim and the supporting logic gets unit-test coverage.

/// Accept both `larql-router <args>` and the legacy `larql-router route
/// <args>` invocation. The `route` subcommand was the binary's original
/// name when it served exactly one purpose; the alias keeps deployers'
/// systemd units / Docker entrypoints working unchanged.
pub fn filter_legacy_route_arg(args: Vec<String>) -> Vec<String> {
    if args.len() > 1 && args[1] == "route" {
        std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect()
    } else {
        args
    }
}

/// Daemon startup validation: at least one of `--shards` / `--grid-port`
/// must be supplied. Returns the user-facing error message that
/// `main.rs` prints to stderr.
pub fn validate_daemon_inputs(shards: Option<&str>, grid_port: Option<u16>) -> Result<(), String> {
    if shards.is_none() && grid_port.is_none() {
        return Err("must provide --shards or --grid-port (or both)".into());
    }
    Ok(())
}

/// Build the reqwest client used to talk to backend shards. Lives here
/// (vs main.rs) so the connection-pool / timeout defaults are testable
/// without standing up a daemon.
pub fn build_shard_client(timeout_secs: u64) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(16)
        .build()
        .map_err(|e| format!("reqwest client: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_legacy_route_passes_through_normal_args() {
        let args = vec!["larql-router".into(), "--port".into(), "8080".into()];
        assert_eq!(filter_legacy_route_arg(args.clone()), args);
    }

    #[test]
    fn filter_legacy_route_strips_route_subcommand() {
        let args = vec![
            "larql-router".into(),
            "route".into(),
            "--port".into(),
            "8080".into(),
        ];
        assert_eq!(
            filter_legacy_route_arg(args),
            vec!["larql-router", "--port", "8080"]
        );
    }

    #[test]
    fn filter_legacy_route_handles_bare_route_invocation() {
        let args = vec!["larql-router".into(), "route".into()];
        assert_eq!(filter_legacy_route_arg(args), vec!["larql-router"]);
    }

    #[test]
    fn filter_legacy_route_no_args_returns_empty() {
        // argv[0] is always present, but be paranoid.
        let args = vec!["larql-router".into()];
        assert_eq!(
            filter_legacy_route_arg(args.clone()),
            vec!["larql-router".to_string()]
        );
    }

    #[test]
    fn validate_daemon_inputs_requires_one_of_shards_or_grid() {
        let err = validate_daemon_inputs(None, None).unwrap_err();
        assert!(err.contains("--shards or --grid-port"));
    }

    #[test]
    fn validate_daemon_inputs_accepts_shards_only() {
        assert!(validate_daemon_inputs(Some("0-1=http://a"), None).is_ok());
    }

    #[test]
    fn validate_daemon_inputs_accepts_grid_port_only() {
        assert!(validate_daemon_inputs(None, Some(50052)).is_ok());
    }

    #[test]
    fn validate_daemon_inputs_accepts_both() {
        assert!(validate_daemon_inputs(Some("0-1=http://a"), Some(50052)).is_ok());
    }

    #[test]
    fn build_shard_client_returns_a_client() {
        let client = build_shard_client(30).expect("reqwest builder must succeed");
        // The defaults all build; we don't assert specific timeouts because
        // reqwest::Client doesn't expose them post-construction.
        drop(client);
    }

    #[test]
    fn build_shard_client_accepts_zero_timeout() {
        // 0 seconds is technically allowed by reqwest (no timeout).
        let client = build_shard_client(0).expect("reqwest builder must succeed");
        drop(client);
    }
}
