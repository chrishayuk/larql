//! The socket: one client at a time, requests answered in order, a second client turned away.
//!
//! **One client.** While a client is connected, the service answers only it. A second connection is
//! accepted, sent one `busy` error line and closed; it is recorded in the audit log. The check runs between
//! requests, so a second client that connects during a long execution is turned away when it ends.
//!
//! **Shutdown.** The loop polls `shutdown` (set by SIGTERM or SIGINT) every [`POLL`]: between requests, and
//! while waiting for a client. A request being executed is finished and answered first. The socket file is
//! removed by [`BoundSocket::remove`] only if it is still the socket this process bound.

use std::io::{ErrorKind as IoKind, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use super::audit::{EVENT_CONNECT, EVENT_DISCONNECT, EVENT_REJECTED};
use super::protocol::{response, ErrorKind, Failure, Meta};
use super::service::Service;

/// How often the loop looks at the shutdown flag and for a second client.
pub const POLL: Duration = Duration::from_millis(50);
/// Bytes read from the socket at a time.
const READ_CHUNK: usize = 64 * 1024;

/// Why the loop ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    /// The shutdown flag was set.
    Shutdown,
    /// The audit log could not be written, so nothing more may be answered.
    AuditFailed(String),
}

impl Stopped {
    pub fn reason(&self) -> String {
        match self {
            Stopped::Shutdown => "shutdown".into(),
            Stopped::AuditFailed(why) => why.clone(),
        }
    }
}

/// The listening socket, and the identity of the file it bound.
pub struct BoundSocket {
    pub listener: UnixListener,
    path: PathBuf,
    identity: (u64, u64),
}

impl BoundSocket {
    /// Bind `path`, which must not exist: a stale socket is left for the operator to remove, never unlinked
    /// on a guess.
    pub fn bind(path: &Path) -> Result<Self, String> {
        if path.symlink_metadata().is_ok() {
            return Err(format!(
                "{} already exists; remove it if no service is using it",
                path.display()
            ));
        }
        let listener = UnixListener::bind(path)
            .map_err(|e| format!("bind the socket {}: {e}", path.display()))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("set the socket non-blocking: {e}"))?;
        let meta = path
            .symlink_metadata()
            .map_err(|e| format!("stat the socket {}: {e}", path.display()))?;
        Ok(Self {
            listener,
            path: path.to_path_buf(),
            identity: (meta.dev(), meta.ino()),
        })
    }

    /// Remove the socket file, if it is still the one bound here.
    pub fn remove(self) -> bool {
        let same = self
            .path
            .symlink_metadata()
            .is_ok_and(|m| (m.dev(), m.ino()) == self.identity);
        same && std::fs::remove_file(&self.path).is_ok()
    }
}

/// Serve `listener` until `shutdown` is set or the audit log fails.
pub fn serve(service: &mut Service, listener: &UnixListener, shutdown: &AtomicBool) -> Stopped {
    loop {
        if let Some(why) = service.fatal() {
            return Stopped::AuditFailed(why.to_string());
        }
        if shutdown.load(Ordering::SeqCst) {
            return Stopped::Shutdown;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                service.record(EVENT_CONNECT, json!({}));
                let outcome = converse(service, listener, stream, shutdown);
                service.record(EVENT_DISCONNECT, json!({"how": outcome}));
            }
            Err(e) if e.kind() == IoKind::WouldBlock => std::thread::sleep(POLL),
            Err(_) => std::thread::sleep(POLL),
        }
    }
}

/// Turn away every client waiting to connect.
fn reject_waiting(service: &mut Service, listener: &UnixListener) {
    while let Ok((mut stream, _)) = listener.accept() {
        let busy = Failure::new(
            ErrorKind::Busy,
            "another client is connected; this service answers one client at a time",
        );
        let meta = Meta {
            seq: 0,
            budget: Value::Null,
            edited: false,
            counted: false,
        };
        let mut line = response(&Value::Null, &meta, &Err(busy));
        line.push(b'\n');
        let _ = stream.set_nonblocking(false);
        let _ = stream.write_all(&line);
        service.record(EVENT_REJECTED, json!({}));
    }
}

/// Answer one client until it goes away. Returns how the conversation ended.
fn converse(
    service: &mut Service,
    listener: &UnixListener,
    mut stream: UnixStream,
    shutdown: &AtomicBool,
) -> &'static str {
    // An accepted socket can inherit the listener's non-blocking mode; reads here wait, up to POLL.
    if stream.set_nonblocking(false).is_err() || stream.set_read_timeout(Some(POLL)).is_err() {
        return "socket_error";
    }
    let max_line = service.limits.max_request_bytes as usize;
    let mut pending: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; READ_CHUNK];
    loop {
        while let Some(end) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=end).collect();
            let mut answer = service.handle_line(&line[..line.len() - 1]);
            answer.push(b'\n');
            if stream.write_all(&answer).is_err() {
                return "client_gone";
            }
            if service.fatal().is_some() {
                return "audit_failed";
            }
        }
        if pending.len() > max_line {
            // The rest of an over-long line cannot be told from the next request: answer and hang up.
            let mut answer = service.refuse_line(Failure::refused(format!(
                "the request line is longer than the policy's max_request_bytes ({max_line}); \
                 the connection is closed"
            )));
            answer.push(b'\n');
            let _ = stream.write_all(&answer);
            return "request_too_long";
        }
        if shutdown.load(Ordering::SeqCst) {
            return "shutdown";
        }
        reject_waiting(service, listener);
        match stream.read(&mut chunk) {
            Ok(0) => return "client_closed",
            Ok(n) => pending.extend_from_slice(&chunk[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    IoKind::WouldBlock | IoKind::TimedOut | IoKind::Interrupted
                ) => {}
            Err(_) => return "socket_error",
        }
    }
}
