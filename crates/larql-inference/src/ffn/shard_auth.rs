//! The API key a client sends to a remote FFN or expert shard.
//!
//! A shard server started with `--api-key` refuses every request without
//! `Authorization: Bearer <key>`. The remote clients had no way to send
//! one, so a shard could be reached by a client only if it ran with no key
//! at all. The key is read once, when a shard config is built, from
//! [`SHARD_API_KEY_ENV`], and can be set explicitly with the configs'
//! `with_api_key`.

/// Environment variable holding the key sent to remote shards.
pub const SHARD_API_KEY_ENV: &str = "LARQL_SHARD_API_KEY";

/// Scheme prefix of the `Authorization` header value.
const BEARER: &str = "Bearer ";

/// The key from the environment, if one is set and not empty.
///
/// Read through `larql_compute::options`, so a test can set it for its own
/// thread without touching the process environment.
pub fn shard_api_key_from_env() -> Option<String> {
    larql_compute::options::env_nonempty_value(SHARD_API_KEY_ENV)
}

/// The `Authorization` header value for `key`.
pub(crate) fn bearer(key: &str) -> String {
    format!("{BEARER}{key}")
}

/// Default headers for a `reqwest` client that sends `key`, when there is
/// one. The value is marked sensitive so it stays out of debug output.
pub(crate) fn default_headers(key: Option<&str>) -> Result<reqwest::header::HeaderMap, String> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = key {
        let mut value = reqwest::header::HeaderValue::from_str(&bearer(key))
            .map_err(|_| format!("{SHARD_API_KEY_ENV} is not a valid header value"))?;
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    Ok(headers)
}
