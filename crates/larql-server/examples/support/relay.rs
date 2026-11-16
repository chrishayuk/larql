//! The DEC-V3-1 link relay's core: shaping, lateness tallies, and the
//! generation-boundary mark (VERIFY-REMOTE-1 revision 2).
//!
//! Shared by `v3_link_shaper` and its tests, so the tests drive the code the
//! binary runs. The per-chunk link model is in the shaper's module docs.
//!
//! **The mark.** A control connection that sends `mark` starts one new epoch.
//! Every connection open at that moment snapshots both directions' tallies
//! at its next loop iteration and records whether it was quiescent:
//! - nothing queued on its link;
//! - nothing unread in either socket (a non-consuming peek);
//! - its last delivered chunk was a response, so no request awaits an
//!   answer.
//!
//! The relay answers `ok <epoch>` only if every such connection snapshotted
//! within the timeout, every one was quiescent, and none read a chunk between
//! its snapshot and the answer. Otherwise it answers `refused <epoch>
//! <reasons>`. It writes the snapshots and a summary to the mark file before
//! answering. A relay that is never marked behaves, and records, exactly as
//! before: the loop adds one atomic load per iteration and one atomic
//! increment per chunk read.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest slice scheduled as one unit. Serialization time is linear in
/// bytes, so the choice changes only how finely a large frame is paced.
pub const CHUNK_BYTES: usize = 16 * 1024;
const BITS_PER_BYTE: f64 = 8.0;
const NANOS_PER_SECOND: f64 = 1e9;
/// Spelling of "no rate limit" on the command line and in the stats record.
pub const UNLIMITED: &str = "inf";
/// The one control command.
pub const MARK: &str = "mark";
pub const OK: &str = "ok";
pub const REFUSED: &str = "refused";
/// How long the control thread waits for every open connection's snapshot.
pub const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(1);
const SNAPSHOT_POLL: Duration = Duration::from_micros(50);
const REQUEST: &str = "request";
const RESPONSE: &str = "response";

#[derive(Clone, Copy)]
pub struct Link {
    pub delay: Duration,
    /// Bits per second; `None` is an unlimited rate.
    pub rate: Option<f64>,
}

impl Link {
    fn serialization(&self, bytes: usize) -> Duration {
        match self.rate {
            Some(rate) => Duration::from_nanos(
                (bytes as f64 * BITS_PER_BYTE / rate * NANOS_PER_SECOND) as u64,
            ),
            None => Duration::ZERO,
        }
    }
}

/// What one direction of one connection actually did.
#[derive(Default, Clone, Copy)]
pub struct Tally {
    pub bytes: u64,
    pub chunks: u64,
    pub late_sum_ns: u64,
    pub late_max_ns: u64,
}

impl Tally {
    fn json(&self) -> String {
        format!(
            "{{\"bytes\":{},\"chunks\":{},\"late_sum_ns\":{},\"late_max_ns\":{}}}",
            self.bytes, self.chunks, self.late_sum_ns, self.late_max_ns
        )
    }
}

/// A chunk on the link: when it may be delivered, and how much already was.
struct Scheduled {
    release: Instant,
    bytes: Vec<u8>,
    written: usize,
}

/// One direction of one connection: sender socket, link, receiver socket.
struct Direction {
    from: TcpStream,
    to: TcpStream,
    link_free: Instant,
    queue: VecDeque<Scheduled>,
    /// The sender closed; deliver what is queued, then close the receiver.
    finished: bool,
    closed: bool,
    tally: Tally,
}

impl Direction {
    fn new(from: TcpStream, to: TcpStream) -> Self {
        Self {
            from,
            to,
            link_free: Instant::now(),
            queue: VecDeque::new(),
            finished: false,
            closed: false,
            tally: Tally::default(),
        }
    }

    /// Take whatever the sender has written and put it on the link.
    /// Returns whether a chunk was read.
    fn accept(&mut self, link: Link, buffer: &mut [u8]) -> std::io::Result<bool> {
        if self.finished {
            return Ok(false);
        }
        match self.from.read(buffer) {
            Ok(0) => self.finished = true,
            Ok(read) => {
                let start = Instant::now().max(self.link_free);
                self.link_free = start + link.serialization(read);
                self.queue.push_back(Scheduled {
                    release: self.link_free + link.delay,
                    bytes: buffer[..read].to_vec(),
                    written: 0,
                });
                return Ok(true);
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
        Ok(false)
    }

    /// Deliver every chunk whose release time has passed, in order.
    fn deliver(&mut self) -> std::io::Result<()> {
        while let Some(head) = self.queue.front_mut() {
            let now = Instant::now();
            if now < head.release {
                return Ok(());
            }
            if head.written == 0 {
                let late = (now - head.release).as_nanos() as u64;
                self.tally.chunks += 1;
                self.tally.late_sum_ns += late;
                self.tally.late_max_ns = self.tally.late_max_ns.max(late);
            }
            match self.to.write(&head.bytes[head.written..]) {
                Ok(written) => {
                    head.written += written;
                    self.tally.bytes += written as u64;
                    if head.written == head.bytes.len() {
                        self.queue.pop_front();
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(e) => return Err(e),
            }
        }
        if self.finished && !self.closed {
            self.closed = true;
            let _ = self.to.shutdown(Shutdown::Write);
        }
        Ok(())
    }

    /// Bytes waiting in the sender's socket that the relay has not read.
    fn unread(&self) -> bool {
        let mut probe = [0u8; 1];
        matches!(self.from.peek(&mut probe), Ok(n) if n > 0)
    }
}

/// One connection's state at a mark.
#[derive(Clone)]
pub struct Snapshot {
    pub epoch: u64,
    pub request: Tally,
    pub response: Tally,
    pub queued: usize,
    pub unread: bool,
    /// `request`, `response`, or `none` if nothing was ever delivered.
    pub last_delivered: &'static str,
    /// Chunks read from either sender up to the snapshot.
    pub accepted: u64,
}

impl Snapshot {
    /// Why this connection was not at a clean boundary; empty if it was.
    pub fn faults(&self) -> Vec<String> {
        let mut faults = Vec::new();
        if self.queued > 0 {
            faults.push(format!("{} chunks queued on the link", self.queued));
        }
        if self.unread {
            faults.push("unread bytes in a socket".to_string());
        }
        if self.last_delivered == REQUEST {
            faults.push("a request awaits its response".to_string());
        }
        faults
    }
}

/// A connection as the mark sees it.
pub struct Watched {
    pub connection: u64,
    open: AtomicBool,
    /// Chunks read from either sender, ever.
    accepted: AtomicU64,
    snapshot: Mutex<Option<Snapshot>>,
}

/// The mark's shared state: the epoch every connection loop watches, the
/// connections open now, and where mark records go.
pub struct Marks {
    epoch: AtomicU64,
    open: Mutex<Vec<Arc<Watched>>>,
    pub path: String,
    pub timeout: Duration,
}

impl Marks {
    pub fn new(path: String, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            epoch: AtomicU64::new(0),
            open: Mutex::new(Vec::new()),
            path,
            timeout,
        })
    }

    pub fn watch(&self, connection: u64) -> Arc<Watched> {
        let watched = Arc::new(Watched {
            connection,
            open: AtomicBool::new(true),
            accepted: AtomicU64::new(0),
            snapshot: Mutex::new(None),
        });
        self.open.lock().unwrap().push(watched.clone());
        watched
    }

    fn unwatch(&self, watched: &Arc<Watched>) {
        watched.open.store(false, Ordering::Release);
        self.open
            .lock()
            .unwrap()
            .retain(|w| !Arc::ptr_eq(w, watched));
    }

    /// Run one mark: snapshot every open connection, judge the boundary,
    /// record it, and return the answer line (without its newline).
    pub fn mark(&self) -> String {
        let watched: Vec<Arc<Watched>> = self.open.lock().unwrap().clone();
        let epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let deadline = Instant::now() + self.timeout;
        let mut reasons = Vec::new();
        if watched.is_empty() {
            reasons.push("no open connection".to_string());
        }
        let mut snapshots: Vec<(u64, Option<Snapshot>)> = Vec::new();
        for w in &watched {
            let snapshot = loop {
                let taken = w
                    .snapshot
                    .lock()
                    .unwrap()
                    .clone()
                    .filter(|s| s.epoch == epoch);
                if taken.is_some() || !w.open.load(Ordering::Acquire) || Instant::now() > deadline {
                    break taken;
                }
                std::thread::sleep(SNAPSHOT_POLL);
            };
            match &snapshot {
                None => reasons.push(format!("connection {} gave no snapshot", w.connection)),
                Some(s) => reasons.extend(
                    s.faults()
                        .into_iter()
                        .map(|f| format!("connection {}: {f}", w.connection)),
                ),
            }
            snapshots.push((w.connection, snapshot));
        }
        // Nothing may cross the boundary before the answer.
        let mut crossed = Vec::new();
        for (w, (_, snapshot)) in watched.iter().zip(&snapshots) {
            if let Some(s) = snapshot {
                let after = w.accepted.load(Ordering::Acquire) - s.accepted;
                crossed.push(after);
                if after > 0 {
                    reasons.push(format!(
                        "connection {}: {after} chunks read after its snapshot",
                        w.connection
                    ));
                }
            } else {
                crossed.push(0);
            }
        }
        let answer = if reasons.is_empty() {
            format!("{OK} {epoch}")
        } else {
            format!("{REFUSED} {epoch} {}", reasons.join("; "))
        };
        let mut lines = String::new();
        for ((connection, snapshot), after) in snapshots.iter().zip(&crossed) {
            if let Some(s) = snapshot {
                lines += &format!(
                    "{{\"mark\":{epoch},\"connection\":{connection},\"request\":{},\
                     \"response\":{},\"queued\":{},\"unread\":{},\"last_delivered\":\"{}\",\
                     \"read_after_snapshot\":{after}}}\n",
                    s.request.json(),
                    s.response.json(),
                    s.queued,
                    s.unread,
                    s.last_delivered,
                );
            }
        }
        let connections: Vec<String> = snapshots.iter().map(|(c, _)| c.to_string()).collect();
        let quoted: Vec<String> = reasons
            .iter()
            .map(|r| format!("\"{}\"", r.replace('"', "'")))
            .collect();
        lines += &format!(
            "{{\"mark\":{epoch},\"summary\":true,\"connections\":[{}],\"answer\":\"{}\",\
             \"reasons\":[{}]}}\n",
            connections.join(","),
            if reasons.is_empty() { OK } else { REFUSED },
            quoted.join(","),
        );
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = file.write_all(lines.as_bytes());
        }
        answer
    }
}

/// Serve one control connection: each `mark` line gets one answer line.
pub fn control(stream: TcpStream, marks: &Marks) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    for line in BufReader::new(stream).lines() {
        let Ok(line) = line else { return };
        let answer = if line.trim() == MARK {
            marks.mark()
        } else {
            format!("{REFUSED} 0 unknown command")
        };
        if writer.write_all(format!("{answer}\n").as_bytes()).is_err() {
            return;
        }
    }
}

fn record(path: &str, connection: u64, direction: &str, link: Link, tally: &Tally, class: &str) {
    let rate = link
        .rate
        .map_or(format!("\"{UNLIMITED}\""), |r| format!("{r}"));
    let line = format!(
        "{{\"connection\":{connection},\"direction\":\"{direction}\",\"qos\":\"{class}\",\"delay_us\":{},\
         \"rate_bits_per_s\":{rate},\"bytes\":{},\"chunks\":{},\"late_sum_ns\":{},\
         \"late_max_ns\":{}}}\n",
        link.delay.as_micros(),
        tally.bytes,
        tally.chunks,
        tally.late_sum_ns,
        tally.late_max_ns,
    );
    // One append per record: concurrent directions never interleave a line.
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Relay one client connection to `upstream` until both directions close,
/// then append its two closing records to `stats`.
pub fn serve(
    client: TcpStream,
    upstream: &str,
    link: Link,
    stats: &str,
    connection: u64,
    class: &str,
    marks: &Marks,
) {
    let Ok(server) = TcpStream::connect(upstream) else {
        return;
    };
    let (Ok(client_peer), Ok(server_peer)) = (client.try_clone(), server.try_clone()) else {
        return;
    };
    // No Nagle batching on either hop, and no blocking call anywhere below.
    for socket in [&client, &server] {
        if socket.set_nodelay(true).is_err() || socket.set_nonblocking(true).is_err() {
            return;
        }
    }
    let watched = marks.watch(connection);
    let mut seen = marks.epoch.load(Ordering::Acquire);
    let mut last_delivered = "none";
    let mut request = Direction::new(client, server);
    let mut response = Direction::new(server_peer, client_peer);
    let mut buffer = vec![0u8; CHUNK_BYTES];
    while !(request.closed && response.closed) {
        let before = (request.tally.chunks, response.tally.chunks);
        let step = request.accept(link, &mut buffer).and_then(|read_request| {
            request.deliver()?;
            let read_response = response.accept(link, &mut buffer)?;
            response.deliver()?;
            Ok(u64::from(read_request) + u64::from(read_response))
        });
        let Ok(read) = step else { break };
        if read > 0 {
            watched.accepted.fetch_add(read, Ordering::AcqRel);
        }
        if request.tally.chunks > before.0 {
            last_delivered = REQUEST;
        }
        if response.tally.chunks > before.1 {
            last_delivered = RESPONSE;
        }
        let epoch = marks.epoch.load(Ordering::Acquire);
        if epoch != seen {
            seen = epoch;
            *watched.snapshot.lock().unwrap() = Some(Snapshot {
                epoch,
                request: request.tally,
                response: response.tally,
                queued: request.queue.len() + response.queue.len(),
                unread: request.unread() || response.unread(),
                last_delivered,
                accepted: watched.accepted.load(Ordering::Acquire),
            });
        }
        std::hint::spin_loop();
    }
    marks.unwatch(&watched);
    record(stats, connection, REQUEST, link, &request.tally, class);
    record(stats, connection, RESPONSE, link, &response.tally, class);
}
