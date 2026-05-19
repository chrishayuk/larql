//! Streaming PUT to a signed LFS URL with progress callbacks.

use std::collections::HashMap;
use std::path::Path;

use crate::error::VindexError;

use super::super::protocol::{LFS_PUT_TIMEOUT, UPLOAD_PROGRESS_POLL_INTERVAL};
use super::super::PublishCallbacks;
use super::CountingReader;

/// PUT the file contents to the signed LFS URL, streaming through a
/// `CountingReader` so the worker thread can report progress.
pub(super) fn stream_put_with_progress(
    href: &str,
    extra_headers: &HashMap<String, String>,
    local_path: &Path,
    size: u64,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;

    let file = std::fs::File::open(local_path)?;
    let counter = std::sync::Arc::new(portable_atomic::AtomicU64::new(0));
    let reader = CountingReader {
        inner: file,
        counter: counter.clone(),
    };
    let body = reqwest::blocking::Body::sized(reader, size);

    let client = reqwest::blocking::Client::builder()
        .timeout(LFS_PUT_TIMEOUT)
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    // Build the request on the worker thread (reqwest's Body needs to
    // travel there). Include any signature headers the LFS server
    // requested — on AWS-backed buckets these carry the AWS sigv4 bits.
    let href_owned = href.to_string();
    let headers_owned: Vec<(String, String)> = extra_headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut req = client.put(&href_owned);
        for (k, v) in &headers_owned {
            req = req.header(k.as_str(), v.as_str());
        }
        let result = req.body(body).send();
        let _ = tx.send(result);
    });

    loop {
        match rx.try_recv() {
            Ok(resp) => {
                let _ = handle.join();
                let resp = resp.map_err(|e| VindexError::Parse(format!("LFS PUT failed: {e}")))?;
                if resp.status().is_success() {
                    callbacks.on_file_progress(remote_filename, size, size);
                    return Ok(());
                }
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(VindexError::Parse(format!(
                    "LFS PUT {remote_filename} ({status}): {body}"
                )));
            }
            Err(TryRecvError::Empty) => {
                let sent = counter.load(Ordering::Relaxed);
                callbacks.on_file_progress(remote_filename, sent, size);
                std::thread::sleep(UPLOAD_PROGRESS_POLL_INTERVAL);
            }
            Err(TryRecvError::Disconnected) => {
                let _ = handle.join();
                return Err(VindexError::Parse(
                    "upload worker terminated unexpectedly".into(),
                ));
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::{write_temp_bytes, CapturingCallbacks, EnvBaseGuard};
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn stream_put_uploads_body_and_ticks_progress_to_100() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let payload = b"streaming-body-bytes";
        let (_dir, path) = write_temp_bytes(payload);

        let mock = server
            .mock("PUT", "/lfs/upload/x")
            .match_body(payload.to_vec())
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        let href = format!("{}/lfs/upload/x", server.url());
        let extra: HashMap<String, String> = HashMap::new();
        stream_put_with_progress(
            &href,
            &extra,
            &path,
            payload.len() as u64,
            "blob.bin",
            &mut cb,
        )
        .unwrap();
        mock.assert();
        let last = cb.progress_calls.last().expect("at least one tick");
        assert_eq!(last.0, "blob.bin");
        assert_eq!(last.1, payload.len() as u64);
        assert_eq!(last.2, payload.len() as u64);
    }

    #[test]
    #[serial]
    fn stream_put_forwards_extra_headers() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let mock = server
            .mock("PUT", "/sig")
            .match_header("x-amz-signature", "test-sig")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        let href = format!("{}/sig", server.url());
        let mut extra: HashMap<String, String> = HashMap::new();
        extra.insert("X-Amz-Signature".to_string(), "test-sig".to_string());
        stream_put_with_progress(&href, &extra, &path, 1, "x", &mut cb).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn stream_put_propagates_http_error_with_filename() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let mock = server
            .mock("PUT", "/blocked")
            .with_status(403)
            .with_body("nope")
            .create();

        let mut cb = CapturingCallbacks::default();
        let href = format!("{}/blocked", server.url());
        let extra: HashMap<String, String> = HashMap::new();
        let err = stream_put_with_progress(&href, &extra, &path, 1, "blocked.bin", &mut cb)
            .expect_err("403 must error");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("403"), "{msg}");
        assert!(msg.contains("blocked.bin"), "{msg}");
    }
}
