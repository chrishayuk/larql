//! Diagnostic copy of `v3_link_shaper` that records when each chunk moved.
//!
//! **Not the frozen relay and never an instrument of record.** It exists to
//! locate a frame-size-dependent timing error found by the LINK-2
//! block-frame diagnostic. The forwarding logic is `v3_link_shaper`'s,
//! line for line: the same chunk size, the same schedule, the same polling
//! loop. The only additions are four timestamps per chunk, kept in memory
//! and written when the connection closes:
//!
//! ```text
//! arrival   the read that took the chunk from the sender returned
//! release   the time the link model scheduled it for
//! start     the first write of it to the receiver was attempted
//! done      its last byte was accepted by the receiver's socket
//! ```
//!
//! Taking those timestamps costs a few clock reads per chunk, so this
//! build's timing is close to the frozen relay's and not identical to it.
//! Conclusions are about where time goes, not about the relay's accuracy.
//!
//! Same arguments as `v3_link_shaper`. The trace is written beside the
//! statistics file as `<stats>.trace.jsonl`, one line per chunk, times in
//! nanoseconds since this process started.

#[path = "support/qos.rs"]
mod qos;

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// Largest slice scheduled as one unit. Serialization time is linear in
/// bytes, so the choice changes only how finely a large frame is paced.
const CHUNK_BYTES: usize = 16 * 1024;
const BITS_PER_BYTE: f64 = 8.0;
const NANOS_PER_SECOND: f64 = 1e9;
/// Spelling of "no rate limit" on the command line and in the stats record.
const UNLIMITED: &str = "inf";
/// Printed once the listener is bound, so a harness can read the real port.
const READY_PREFIX: &str = "listening";
const ARGUMENTS: usize = 5;

#[derive(Clone, Copy)]
struct Link {
    delay: Duration,
    /// Bits per second; `None` is an unlimited rate.
    rate: Option<f64>,
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
#[derive(Default)]
struct Tally {
    bytes: u64,
    chunks: u64,
    late_sum_ns: u64,
    late_max_ns: u64,
}

/// A chunk on the link: when it may be delivered, and how much already was.
struct Scheduled {
    release: Instant,
    bytes: Vec<u8>,
    written: usize,
    arrival: Instant,
    start: Option<Instant>,
}

/// One chunk's four timestamps.
struct Traced {
    arrival: Instant,
    release: Instant,
    start: Instant,
    done: Instant,
    bytes: usize,
}

/// Chunks reserved for per direction, so recording never reallocates in a
/// run of the size this diagnostic makes.
const TRACE_RESERVE: usize = 1 << 18;

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
    trace: Vec<Traced>,
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
            trace: Vec::with_capacity(TRACE_RESERVE),
        }
    }

    /// Take whatever the sender has written and put it on the link.
    fn accept(&mut self, link: Link, buffer: &mut [u8]) -> std::io::Result<()> {
        if self.finished {
            return Ok(());
        }
        match self.from.read(buffer) {
            Ok(0) => self.finished = true,
            Ok(read) => {
                let arrival = Instant::now();
                let start = arrival.max(self.link_free);
                self.link_free = start + link.serialization(read);
                self.queue.push_back(Scheduled {
                    release: self.link_free + link.delay,
                    bytes: buffer[..read].to_vec(),
                    written: 0,
                    arrival,
                    start: None,
                });
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
        Ok(())
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
                head.start = Some(now);
            }
            match self.to.write(&head.bytes[head.written..]) {
                Ok(written) => {
                    head.written += written;
                    self.tally.bytes += written as u64;
                    if head.written == head.bytes.len() {
                        self.trace.push(Traced {
                            arrival: head.arrival,
                            release: head.release,
                            start: head.start.unwrap_or(now),
                            done: Instant::now(),
                            bytes: head.bytes.len(),
                        });
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

fn write_trace(path: &str, connection: u64, direction: &str, epoch: Instant, trace: &[Traced]) {
    let ns = |t: Instant| t.saturating_duration_since(epoch).as_nanos();
    let mut out = String::with_capacity(trace.len() * 96);
    for chunk in trace {
        out.push_str(&format!(
            "{{\"connection\":{connection},\"direction\":\"{direction}\",\"bytes\":{},\
             \"arrival_ns\":{},\"release_ns\":{},\"start_ns\":{},\"done_ns\":{}}}\n",
            chunk.bytes,
            ns(chunk.arrival),
            ns(chunk.release),
            ns(chunk.start),
            ns(chunk.done),
        ));
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(out.as_bytes());
    }
}

fn serve(
    client: TcpStream,
    upstream: &str,
    link: Link,
    stats: &str,
    connection: u64,
    epoch: Instant,
) {
    let Ok(class) = qos::apply() else {
        return;
    };
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
    let mut request = Direction::new(client, server);
    let mut response = Direction::new(server_peer, client_peer);
    let mut buffer = vec![0u8; CHUNK_BYTES];
    while !(request.closed && response.closed) {
        let step = request
            .accept(link, &mut buffer)
            .and_then(|()| request.deliver())
            .and_then(|()| response.accept(link, &mut buffer))
            .and_then(|()| response.deliver());
        if step.is_err() {
            break;
        }
        std::hint::spin_loop();
    }
    record(stats, connection, "request", link, &request.tally, class);
    record(stats, connection, "response", link, &response.tally, class);
    let trace = format!("{stats}.trace.jsonl");
    write_trace(&trace, connection, "request", epoch, &request.trace);
    write_trace(&trace, connection, "response", epoch, &response.trace);
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != ARGUMENTS {
        return Err(
            "usage: v3_link_shaper <listen> <upstream> <one_way_delay_us> \
             <rate_bits_per_s|inf> <stats.jsonl>"
                .into(),
        );
    }
    let delay = Duration::from_micros(args[2].parse::<u64>().map_err(|e| e.to_string())?);
    let rate = if args[3] == UNLIMITED {
        None
    } else {
        let rate = args[3].parse::<f64>().map_err(|e| e.to_string())?;
        if !rate.is_finite() || rate <= 0.0 {
            return Err("rate must be a positive number of bits per second".into());
        }
        Some(rate)
    };
    // Refuse an unknown class before listening, not silently per connection.
    qos::apply()?;
    let link = Link { delay, rate };
    let listener = TcpListener::bind(&args[0]).map_err(|e| e.to_string())?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    let epoch = Instant::now();
    println!("{READY_PREFIX} {bound}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    for (connection, client) in listener.incoming().enumerate() {
        let Ok(client) = client else { continue };
        let (upstream, stats) = (args[1].clone(), args[4].clone());
        std::thread::spawn(move || {
            serve(client, &upstream, link, &stats, connection as u64, epoch)
        });
    }
    Ok(())
}
