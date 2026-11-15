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
/// The longest a response may take to write: a client that stops reading is hung up on, so it cannot hold
/// the service (or its shutdown).
pub const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
/// Bytes read from the socket at a time.
const READ_CHUNK: usize = 64 * 1024;

/// Why the loop ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    /// The shutdown flag was set.
    Shutdown,
    /// The service faulted (the audit log could not be written, or its gate missed an edited execution), so
    /// nothing more may be answered.
    Fault(String),
}

impl Stopped {
    pub fn reason(&self) -> String {
        match self {
            Stopped::Shutdown => "shutdown".into(),
            Stopped::Fault(why) => why.clone(),
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
    serve_with(service, listener, shutdown, WRITE_TIMEOUT)
}

/// [`serve`], with the write timeout given.
pub fn serve_with(
    service: &mut Service,
    listener: &UnixListener,
    shutdown: &AtomicBool,
    write_timeout: Duration,
) -> Stopped {
    loop {
        if let Some(why) = service.fatal() {
            return Stopped::Fault(why.to_string());
        }
        if shutdown.load(Ordering::SeqCst) {
            return Stopped::Shutdown;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                service.record(EVENT_CONNECT, peer(&stream));
                let outcome = converse(service, listener, stream, shutdown, write_timeout);
                service.record(EVENT_DISCONNECT, json!({"how": outcome}));
            }
            Err(e) if e.kind() == IoKind::WouldBlock => std::thread::sleep(POLL),
            Err(_) => std::thread::sleep(POLL),
        }
    }
}

/// Who connected: the peer's uid and gid, and its pid where the platform reports it.
fn peer(stream: &UnixStream) -> Value {
    use std::os::fd::AsRawFd;
    match peer_credentials(stream.as_raw_fd()) {
        Some((uid, gid, pid)) => json!({"peer": {"uid": uid, "gid": gid, "pid": pid}}),
        None => json!({"peer": Value::Null}),
    }
}

#[cfg(target_os = "linux")]
fn peer_credentials(fd: std::os::fd::RawFd) -> Option<(u32, u32, Option<i32>)> {
    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED writes one ucred into `cred`.
    let ok = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut len,
        )
    } == 0;
    ok.then_some((cred.uid, cred.gid, Some(cred.pid)))
}

#[cfg(not(target_os = "linux"))]
fn peer_credentials(fd: std::os::fd::RawFd) -> Option<(u32, u32, Option<i32>)> {
    let mut uid: libc::uid_t = 0;
    let mut gid: libc::gid_t = 0;
    // SAFETY: `fd` is a connected Unix socket and the out-pointers are valid.
    if unsafe { libc::getpeereid(fd, &mut uid, &mut gid) } != 0 {
        return None;
    }
    Some((uid, gid, peer_pid(fd)))
}

#[cfg(target_os = "macos")]
fn peer_pid(fd: std::os::fd::RawFd) -> Option<i32> {
    let mut pid: libc::pid_t = 0;
    let mut len = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
    // SAFETY: LOCAL_PEERPID writes one pid_t into `pid`.
    let ok = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_LOCAL,
            libc::LOCAL_PEERPID,
            (&mut pid as *mut libc::pid_t).cast(),
            &mut len,
        )
    } == 0;
    ok.then_some(pid)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peer_pid(_: std::os::fd::RawFd) -> Option<i32> {
    None
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
            request_uid: Value::Null,
            budget: Value::Null,
            edited: false,
            counted: false,
        };
        let mut line = response(&Value::Null, &meta, &Err(busy));
        line.push(b'\n');
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_write_timeout(Some(POLL));
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
    write_timeout: Duration,
) -> &'static str {
    // An accepted socket can inherit the listener's non-blocking mode; reads here wait, up to POLL. A peer
    // that has already hung up makes these calls fail (macOS refuses socket options on a shut-down
    // socket); its request is still read and answered, and the write then fails.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(POLL));
    let _ = stream.set_write_timeout(Some(POLL));
    let max_line = service.limits.max_request_bytes as usize;
    let mut pending: Vec<u8> = Vec::new();
    // Bytes of `pending` already scanned for a newline: each byte is scanned once.
    let mut scanned = 0;
    let mut chunk = vec![0u8; READ_CHUNK];
    loop {
        while let Some(at) = pending[scanned..].iter().position(|&b| b == b'\n') {
            let end = scanned + at;
            let line: Vec<u8> = pending.drain(..=end).collect();
            scanned = 0;
            let mut answer = service.handle_line(&line[..line.len() - 1]);
            answer.push(b'\n');
            if let Err(how) = write_response(&mut stream, &answer, write_timeout, shutdown) {
                return how;
            }
            if service.fatal().is_some() {
                return "service_fault";
            }
        }
        scanned = pending.len();
        if pending.len() > max_line {
            // The rest of an over-long line cannot be told from the next request: answer and hang up.
            let mut answer = service.refuse_line(Failure::refused(format!(
                "the request line is longer than the policy's max_request_bytes ({max_line}); \
                 the connection is closed"
            )));
            answer.push(b'\n');
            let _ = write_response(&mut stream, &answer, write_timeout, shutdown);
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

/// Write a whole response within `deadline` of starting, a chunk at a time, giving up on shutdown between
/// chunks. Each `write` waits at most [`POLL`], so neither a stalled client nor a signal waits longer.
fn write_response(
    stream: &mut UnixStream,
    bytes: &[u8],
    deadline: Duration,
    shutdown: &AtomicBool,
) -> Result<(), &'static str> {
    let started = std::time::Instant::now();
    let mut written = 0;
    while written < bytes.len() {
        if shutdown.load(Ordering::SeqCst) {
            return Err("shutdown");
        }
        if started.elapsed() >= deadline {
            return Err("client_gone");
        }
        let end = (written + READ_CHUNK).min(bytes.len());
        match stream.write(&bytes[written..end]) {
            Ok(0) => return Err("client_gone"),
            Ok(n) => written += n,
            Err(e)
                if matches!(
                    e.kind(),
                    IoKind::WouldBlock | IoKind::TimedOut | IoKind::Interrupted
                ) => {}
            Err(_) => return Err("client_gone"),
        }
    }
    Ok(())
}
