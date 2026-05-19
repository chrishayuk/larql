//! Mode B shard downloader — streams a tar from the donor's `/v1/shard`
//! endpoint, optionally verifies the SHA-256 of the byte stream, and
//! unpacks it into `store_path/{model_id}/layers-{start}-{end}/`.
//!
//! The unpack is atomic: the tar is unpacked into a sibling `.tmp` directory
//! that is renamed onto the final path on success. A partial download leaves
//! a `.tmp` directory behind which the next attempt removes.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

const SHARD_ENDPOINT: &str = "/v1/shard";

/// Download a shard tar from `origin_url`, verify the hash, atomically unpack
/// to `store_path/{model_id}/layers-{layer_start}-{layer_end}/`.
pub async fn download_and_load_shard(
    origin_url: &str,
    store_path: &str,
    expected_hash: &str,
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = format!(
        "{}{SHARD_ENDPOINT}/{model_id}/{layer_start}-{layer_end}",
        origin_url.trim_end_matches('/')
    );

    let model_dir = PathBuf::from(store_path).join(model_id);
    let shard_dir = model_dir.join(format!("layers-{layer_start}-{layer_end}"));
    let tmp_dir = model_dir.join(format!(".tmp-layers-{layer_start}-{layer_end}"));

    tokio::fs::create_dir_all(&model_dir).await?;

    // Remove a stale tmp directory from an earlier aborted attempt.
    if tokio::fs::metadata(&tmp_dir).await.is_ok() {
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
    }
    // If the final shard already exists, treat as success (idempotent).
    if tokio::fs::metadata(&shard_dir).await.is_ok() {
        info!(dest = %shard_dir.display(), "Mode B: shard already present — skipping download");
        return Ok(());
    }

    info!(url = %url, dest = %shard_dir.display(), "Mode B: downloading shard tar…");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(format!("shard download failed: HTTP {} from {url}", resp.status()).into());
    }

    let bytes = resp.bytes().await?;
    info!(
        bytes = bytes.len(),
        "Mode B: download complete — unpacking…"
    );

    let skip_hash = expected_hash.is_empty()
        || expected_hash == "0000000000000000"
        || expected_hash.chars().all(|c| c == '0');

    if !skip_hash {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got_hash = format!("{:x}", hasher.finalize());
        if got_hash != expected_hash {
            return Err(
                format!("shard hash mismatch: expected {expected_hash}, got {got_hash}").into(),
            );
        }
        info!("Mode B: hash verified ✓");
    } else {
        warn!("Mode B: hash check skipped (placeholder hash)");
    }

    // Unpack in a blocking task — `tar::Archive` is sync I/O.
    let tmp_dir_for_blocking = tmp_dir.clone();
    let bytes_for_blocking = bytes.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        std::fs::create_dir_all(&tmp_dir_for_blocking)?;
        let cursor = std::io::Cursor::new(bytes_for_blocking);
        let mut archive = tar::Archive::new(cursor);
        archive.unpack(&tmp_dir_for_blocking)?;
        Ok(())
    })
    .await
    .map_err(|e| format!("unpack task join failed: {e}"))??;

    // Atomic rename onto the final path.
    if let Err(e) = tokio::fs::rename(&tmp_dir, &shard_dir).await {
        // Best-effort cleanup of the half-unpacked tmp dir.
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
        return Err(format!(
            "atomic rename {} -> {} failed: {e}",
            tmp_dir.display(),
            shard_dir.display()
        )
        .into());
    }

    info!(dest = %shard_dir.display(), "Mode B: shard unpacked — ready");
    Ok(())
}

#[allow(dead_code)] // exposed for tests + future external callers
pub fn shard_dest_path(store_path: &str, model_id: &str, start: u32, end: u32) -> PathBuf {
    Path::new(store_path)
        .join(model_id)
        .join(format!("layers-{start}-{end}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn build_tar_in_memory(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut tar = tar::Builder::new(&mut buf);
            for (name, content) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(content.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                tar.append_data(&mut header, name, *content).unwrap();
            }
            tar.finish().unwrap();
        }
        buf
    }

    #[test]
    fn shard_dest_path_combines_segments() {
        let p = shard_dest_path("/mnt/shards", "gemma4-26b", 0, 14);
        assert!(p.ends_with("gemma4-26b/layers-0-14") || p.ends_with("gemma4-26b\\layers-0-14"));
    }

    #[tokio::test]
    async fn unpacks_tar_into_atomic_destination() {
        // End-to-end: serve a tar from a hyper-axum test server and verify the
        // client unpacks it into the right directory atomically.
        use axum::body::Body;
        use axum::extract::Path;
        use axum::http::{header, StatusCode};
        use axum::response::Response;
        use axum::routing::get;
        use axum::Router;

        async fn serve_tar(Path((_model, _range)): Path<(String, String)>) -> Response {
            let tar = build_tar_in_memory(&[
                ("index.json", b"{\"hello\":\"world\"}"),
                ("layer-0.bin", &[1u8, 2, 3, 4]),
            ]);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/x-tar")
                .body(Body::from(tar))
                .unwrap()
        }

        let app = Router::new().route("/v1/shard/{model_id}/{range}", get(serve_tar));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let tmp = TempDir::new().unwrap();
        let store = tmp.path().to_str().unwrap();
        let origin = format!("http://{addr}");

        download_and_load_shard(&origin, store, "", "gemma-test", 0, 5)
            .await
            .expect("download must succeed");

        let dest = shard_dest_path(store, "gemma-test", 0, 5);
        assert!(dest.is_dir(), "shard directory not created at {dest:?}");
        let manifest = std::fs::read(dest.join("index.json")).unwrap();
        assert_eq!(manifest, b"{\"hello\":\"world\"}");
        let layer = std::fs::read(dest.join("layer-0.bin")).unwrap();
        assert_eq!(layer, &[1u8, 2, 3, 4]);

        // tmp directory must have been renamed away.
        let tmp_dir = tmp.path().join("gemma-test").join(".tmp-layers-0-5");
        assert!(
            !tmp_dir.exists(),
            "stale tmp directory survived: {tmp_dir:?}"
        );

        // Idempotent re-call must not fail.
        download_and_load_shard(&origin, store, "", "gemma-test", 0, 5)
            .await
            .expect("re-download must be idempotent");

        server_handle.abort();
    }

    #[tokio::test]
    async fn rejects_hash_mismatch() {
        use axum::body::Body;
        use axum::extract::Path;
        use axum::http::{header, StatusCode};
        use axum::response::Response;
        use axum::routing::get;
        use axum::Router;

        async fn serve_tar(Path((_m, _r)): Path<(String, String)>) -> Response {
            let tar = build_tar_in_memory(&[("a.txt", b"hi")]);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/x-tar")
                .body(Body::from(tar))
                .unwrap()
        }

        let app = Router::new().route("/v1/shard/{model_id}/{range}", get(serve_tar));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let tmp = TempDir::new().unwrap();
        let store = tmp.path().to_str().unwrap();
        let origin = format!("http://{addr}");

        let err = download_and_load_shard(
            &origin,
            store,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "gemma-test",
            0,
            0,
        )
        .await
        .expect_err("expected hash mismatch error");
        assert!(format!("{err}").contains("hash mismatch"));
    }
}
