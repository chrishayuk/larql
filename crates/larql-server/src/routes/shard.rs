//! `GET /v1/shard/{model_id}/{layer_start}-{layer_end}` — streams the donor's
//! on-disk vindex directory as a tar so a Mode B server can mirror the shard
//! after receiving an `AssignMsg` from the router.
//!
//! The layer range in the URL is the slice the receiver intends to load. The
//! donor streams its whole vindex directory regardless; the receiver re-mmaps
//! only the owned layers via its own `--layers` flag. This mirrors how Mode A
//! sharding already works (every replica has the same on-disk vindex; only
//! `--layers` controls what is touched).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::state::AppState;

/// Channel-bounded writer that turns blocking tar writes into an async
/// `Bytes` stream consumed by the axum response body.
struct MpscWriter {
    tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
}

impl std::io::Write for MpscWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // `tar::Builder` writes in small chunks; copy once to a Bytes and ship.
        let bytes = Bytes::copy_from_slice(buf);
        self.tx
            .blocking_send(Ok(bytes))
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receiver dropped"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Parse "{start}-{end}" into an inclusive layer range.
fn parse_range(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

pub async fn handle_shard(
    State(state): State<Arc<AppState>>,
    Path((model_id, range)): Path<(String, String)>,
) -> Response {
    let Some((start, end)) = parse_range(&range) else {
        return (
            StatusCode::BAD_REQUEST,
            format!("invalid layer range '{range}': expected 'START-END'"),
        )
            .into_response();
    };
    if start > end {
        return (
            StatusCode::BAD_REQUEST,
            format!("layer range '{range}': start ({start}) must be <= end ({end})"),
        )
            .into_response();
    }

    let model = match state.model(Some(&model_id)) {
        Some(m) => m.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                format!("model '{model_id}' not loaded on this server"),
            )
                .into_response();
        }
    };

    let shard_dir: PathBuf = model.path.clone();
    if !shard_dir.is_dir() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("vindex path is not a directory: {}", shard_dir.display()),
        )
            .into_response();
    }

    tracing::info!(
        model_id = %model_id,
        layers = %format!("{start}-{end}"),
        dir = %shard_dir.display(),
        "Mode B: streaming shard tar"
    );

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    tokio::task::spawn_blocking(move || {
        let writer = MpscWriter { tx: tx.clone() };
        let mut tar = tar::Builder::new(writer);
        // Follow symlinks rather than archiving them; vindex directories
        // sometimes resolve through cache-style symlinks.
        tar.follow_symlinks(true);
        if let Err(e) = tar.append_dir_all(".", &shard_dir) {
            let _ = tx.blocking_send(Err(std::io::Error::other(format!("tar build failed: {e}"))));
            return;
        }
        if let Err(e) = tar.finish() {
            let _ = tx.blocking_send(Err(std::io::Error::other(format!(
                "tar finalise failed: {e}"
            ))));
        }
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-tar")
        .body(body)
        .expect("static headers always build")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_accepts_inclusive_pairs() {
        assert_eq!(parse_range("0-14"), Some((0, 14)));
        assert_eq!(parse_range("3-3"), Some((3, 3)));
    }

    #[test]
    fn parse_range_rejects_malformed() {
        assert_eq!(parse_range(""), None);
        assert_eq!(parse_range("0"), None);
        assert_eq!(parse_range("a-b"), None);
        assert_eq!(parse_range("0-"), None);
    }

    #[tokio::test]
    async fn mpsc_writer_streams_bytes_through_channel() {
        use std::io::Write;
        let (tx, mut rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let mut writer = MpscWriter { tx };
        let payload = b"chunk-one";
        tokio::task::spawn_blocking(move || {
            writer.write_all(payload).unwrap();
            writer.flush().unwrap();
        })
        .await
        .unwrap();

        let received = rx.recv().await.expect("a chunk").expect("ok bytes");
        assert_eq!(received.as_ref(), payload);
    }

    #[tokio::test]
    async fn mpsc_writer_reports_broken_pipe_when_receiver_dropped() {
        use std::io::Write;
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        drop(rx);
        let err = tokio::task::spawn_blocking(move || {
            let mut writer = MpscWriter { tx };
            writer.write_all(b"x").unwrap_err()
        })
        .await
        .unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
    }
}
