//! Repo existence probes — cheap HEAD against the HF API.

use crate::error::VindexError;

use super::super::publish::get_hf_token;
use super::super::publish::protocol::hf_base;

/// Cheap HEAD probe — returns `Ok(true)` if the dataset repo exists and
/// is readable, `Ok(false)` on 404, `Err` on other failures. Auth is
/// optional; pass-through when available (lets callers see private
/// repos they own).
pub fn dataset_repo_exists(repo_id: &str) -> Result<bool, VindexError> {
    repo_exists(repo_id, "model")
}

pub fn repo_exists(repo_id: &str, repo_type: &str) -> Result<bool, VindexError> {
    let token = get_hf_token().ok();
    let plural = if repo_type == "dataset" {
        "datasets"
    } else {
        "models"
    };
    let url = format!("{}/api/{plural}/{repo_id}", hf_base());
    let client = reqwest::blocking::Client::new();
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF HEAD failed: {e}")))?;
    if resp.status().is_success() {
        Ok(true)
    } else if resp.status().as_u16() == 404 {
        Ok(false)
    } else {
        Err(VindexError::Parse(format!(
            "HF HEAD {repo_id}: {}",
            resp.status()
        )))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::TestEnvGuard;
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn repo_exists_returns_true_on_200() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(200)
            .create();

        assert!(repo_exists("org/repo", "model").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_returns_false_on_404() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/missing/repo")
            .with_status(404)
            .create();

        assert!(!repo_exists("missing/repo", "model").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_dataset_uses_datasets_path() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/datasets/org/repo")
            .with_status(200)
            .create();

        assert!(repo_exists("org/repo", "dataset").unwrap());
        mock.assert();
    }

    #[test]
    #[serial]
    fn repo_exists_other_status_propagates_error() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(500)
            .create();

        let err = repo_exists("org/repo", "model").expect_err("500 must error");
        mock.assert();
        assert!(err.to_string().contains("500"));
    }

    #[test]
    #[serial]
    fn dataset_repo_exists_is_thin_wrapper_over_repo_exists() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("HEAD", "/api/models/org/repo")
            .with_status(200)
            .create();

        assert!(dataset_repo_exists("org/repo").unwrap());
        mock.assert();
    }
}
