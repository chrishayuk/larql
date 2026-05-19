//! Shared test fixtures for the discovery sibling test suites.

use crate::format::huggingface::publish::protocol::TEST_BASE_ENV;

/// RAII env-var override for `LARQL_HF_TEST_BASE`, plus a fake
/// HF_TOKEN so the discovery functions don't try to read
/// `~/.huggingface/token` during the test. Restored on drop.
pub(super) struct TestEnvGuard {
    prev_base: Option<String>,
    prev_token: Option<String>,
}

impl TestEnvGuard {
    pub(super) fn new(base: &str) -> Self {
        let prev_base = std::env::var(TEST_BASE_ENV).ok();
        let prev_token = std::env::var("HF_TOKEN").ok();
        std::env::set_var(TEST_BASE_ENV, base);
        std::env::set_var("HF_TOKEN", "test-token");
        Self {
            prev_base,
            prev_token,
        }
    }
}

impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        match self.prev_base.take() {
            Some(v) => std::env::set_var(TEST_BASE_ENV, v),
            None => std::env::remove_var(TEST_BASE_ENV),
        }
        match self.prev_token.take() {
            Some(v) => std::env::set_var("HF_TOKEN", v),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn guard_restores_previous_base_on_drop() {
        // Set a "previous" value so the Drop's Some(v) branch fires.
        std::env::set_var(TEST_BASE_ENV, "prior://value");
        std::env::set_var("HF_TOKEN", "prior-token");
        {
            let _g = TestEnvGuard::new("override://value");
            assert_eq!(std::env::var(TEST_BASE_ENV).unwrap(), "override://value");
            assert_eq!(std::env::var("HF_TOKEN").unwrap(), "test-token");
        }
        assert_eq!(std::env::var(TEST_BASE_ENV).unwrap(), "prior://value");
        assert_eq!(std::env::var("HF_TOKEN").unwrap(), "prior-token");
        std::env::remove_var(TEST_BASE_ENV);
        std::env::remove_var("HF_TOKEN");
    }

    #[test]
    #[serial]
    fn guard_clears_when_no_previous_value() {
        // Both env vars unset before construction → Drop's None branch
        // fires, leaving them unset after.
        std::env::remove_var(TEST_BASE_ENV);
        std::env::remove_var("HF_TOKEN");
        {
            let _g = TestEnvGuard::new("scratch://value");
            assert!(std::env::var(TEST_BASE_ENV).is_ok());
            assert!(std::env::var("HF_TOKEN").is_ok());
        }
        assert!(std::env::var(TEST_BASE_ENV).is_err());
        assert!(std::env::var("HF_TOKEN").is_err());
    }
}
