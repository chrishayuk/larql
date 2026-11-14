//! DEC-V3-1 diagnostic: where does the zero-delay relay baseline's
//! instance-to-instance variation come from?
//!
//! Not a calibration and not part of LINK-1's qualification. It drives the
//! frozen `v3_link_shaper` binary, unmodified, at zero delay and unlimited
//! rate, and records every exchange instead of a median.
//!
//! One process holds both ends, so one monotonic clock stamps four
//! boundaries of each exchange and splits it into three segments:
//!
//! ```text
//! outbound  client starts sending  -> peer holds the whole request
//! peer      peer holds the request -> peer has handed over the response
//! inbound   peer handed over       -> client holds the whole response
//! ```
//!
//! Four arms, interleaved round by round:
//!
//! ```text
//! direct                 no relay
//! fresh                  a new relay process for every batch (calibration's lifecycle)
//! persistent             one relay process, a new connection for every batch
//! persistent-connection  one relay process, one connection for every batch
//! ```
//!
//! Each batch also records process accounting for this process and for the
//! relay: CPU time, time spent runnable but not running, cycles (cycles per
//! CPU second separates fast from slow cores) and interrupt wake-ups.
//!
//! Usage:
//!   v3_link_trace <relay_binary> <rounds> <out.json>

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Same exchange as calibration: the one-worker GPT-OSS frame sizes.
const REQUEST_BYTES: usize = 11_576;
const RESPONSE_BYTES: usize = 46_136;
/// Same batch as calibration, so a batch here is what a baseline is there.
const EXCHANGES: usize = 200;
const WARMUP: usize = 20;
const HEADER_BYTES: usize = 8;
const LENGTH_BYTES: usize = 4;
const LOOPBACK: &str = "127.0.0.1:0";
const READY_PREFIX: &str = "listening ";
const FILL: u8 = 0xA5;
const ARGUMENTS: usize = 3;
const ARMS: [&str; 4] = ["direct", "fresh", "persistent", "persistent-connection"];

/// Peer-side boundaries of every exchange, in nanoseconds since `epoch`.
type PeerLog = Arc<Mutex<Vec<(u64, u64)>>>;

fn receive(stream: &mut TcpStream, buffer: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buffer.len() {
        match stream.read(&mut buffer[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(read) => filled += read,
            Err(e) if e.kind() == ErrorKind::WouldBlock => std::hint::spin_loop(),
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn send(stream: &mut TcpStream, mut bytes: &[u8]) -> std::io::Result<()> {
    while !bytes.is_empty() {
        match stream.write(bytes) {
            Ok(written) => bytes = &bytes[written..],
            Err(e) if e.kind() == ErrorKind::WouldBlock => std::hint::spin_loop(),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn polling(stream: &TcpStream) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_nonblocking(true)
}

fn echo(mut stream: TcpStream, epoch: Instant, log: PeerLog) -> std::io::Result<()> {
    polling(&stream)?;
    let mut request = vec![0u8; HEADER_BYTES + REQUEST_BYTES];
    let response = vec![FILL; RESPONSE_BYTES];
    while receive(&mut stream, &mut request)? {
        let held = epoch.elapsed().as_nanos() as u64;
        send(&mut stream, &response)?;
        let handed = epoch.elapsed().as_nanos() as u64;
        log.lock().expect("peer log").push((held, handed));
    }
    Ok(())
}

/// Process accounting from the kernel, in nanoseconds and raw counts.
#[derive(Clone, Copy, Default)]
struct Accounting {
    cpu_ns: u64,
    runnable_ns: u64,
    cycles: u64,
    instructions: u64,
    interrupt_wakeups: u64,
}

impl Accounting {
    fn since(self, earlier: Accounting) -> serde_json::Value {
        serde_json::json!({
            "cpu_ns": self.cpu_ns - earlier.cpu_ns,
            "runnable_not_running_ns": self.runnable_ns - earlier.runnable_ns,
            "cycles": self.cycles - earlier.cycles,
            "instructions": self.instructions - earlier.instructions,
            "interrupt_wakeups": self.interrupt_wakeups - earlier.interrupt_wakeups,
        })
    }
}

// libc marks the Mach timebase call deprecated in favour of the `mach2`
// crate; a diagnostic example does not justify a new dependency.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn accounting(pid: u32) -> Accounting {
    // SAFETY: both calls fill a caller-owned, correctly sized plain struct.
    unsafe {
        let mut timebase = libc::mach_timebase_info { numer: 0, denom: 0 };
        libc::mach_timebase_info(&mut timebase);
        let ns = |ticks: u64| ticks * u64::from(timebase.numer) / u64::from(timebase.denom);
        let mut info: libc::rusage_info_v4 = std::mem::zeroed();
        let status = libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        );
        if status != 0 {
            return Accounting::default();
        }
        Accounting {
            cpu_ns: ns(info.ri_user_time + info.ri_system_time),
            runnable_ns: ns(info.ri_runnable_time),
            cycles: info.ri_cycles,
            instructions: info.ri_instructions,
            interrupt_wakeups: info.ri_interrupt_wkups,
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn accounting(_pid: u32) -> Accounting {
    Accounting::default()
}

struct Relay {
    child: Child,
    address: String,
}

impl Relay {
    fn start(binary: &str, upstream: &str, stats: &str) -> Result<Self, String> {
        let mut child = Command::new(binary)
            .args([LOOPBACK, upstream, "0", "inf", stats])
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut line = String::new();
        BufReader::new(child.stdout.take().ok_or("relay stdout")?)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        let address = line
            .strip_prefix(READY_PREFIX)
            .ok_or_else(|| format!("relay did not start: {line:?}"))?
            .trim()
            .to_string();
        Ok(Self { child, address })
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One batch on an open connection: every exchange's three segments.
fn batch(
    stream: &mut TcpStream,
    epoch: Instant,
    log: &PeerLog,
    frame: &[u8],
    reply: &mut [u8],
) -> Result<Vec<[u64; 3]>, String> {
    log.lock().expect("peer log").clear();
    let mut client = Vec::with_capacity(EXCHANGES + WARMUP);
    for _ in 0..EXCHANGES + WARMUP {
        let started = epoch.elapsed().as_nanos() as u64;
        send(stream, frame).map_err(|e| e.to_string())?;
        if !receive(stream, reply).map_err(|e| e.to_string())? {
            return Err("peer closed".into());
        }
        client.push((started, epoch.elapsed().as_nanos() as u64));
    }
    // The peer stamps "handed over" just after the client can already hold
    // the response, so wait for its last entry instead of racing it.
    let peer = loop {
        let entries = log.lock().expect("peer log");
        if entries.len() == client.len() {
            break entries.clone();
        }
        drop(entries);
        std::hint::spin_loop();
    };
    Ok(client
        .iter()
        .zip(&peer)
        .map(|((started, finished), (held, handed))| {
            [
                held.saturating_sub(*started),
                handed.saturating_sub(*held),
                finished.saturating_sub(*handed),
            ]
        })
        .collect())
}

fn connect(address: &str) -> Result<TcpStream, String> {
    let stream = TcpStream::connect(address).map_err(|e| e.to_string())?;
    polling(&stream).map_err(|e| e.to_string())?;
    Ok(stream)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != ARGUMENTS {
        return Err("usage: v3_link_trace <relay_binary> <rounds> <out.json>".into());
    }
    let rounds: usize = args[1].parse().map_err(|_| "rounds must be a number")?;
    let stats = format!("{}.relay-stats.jsonl", args[2]);
    let epoch = Instant::now();
    let log: PeerLog = Arc::new(Mutex::new(Vec::with_capacity(EXCHANGES + WARMUP)));
    let listener = TcpListener::bind(LOOPBACK).map_err(|e| e.to_string())?;
    let peer = listener
        .local_addr()
        .map_err(|e| e.to_string())?
        .to_string();
    let peer_log = Arc::clone(&log);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let log = Arc::clone(&peer_log);
            std::thread::spawn(move || {
                let _ = echo(stream, epoch, log);
            });
        }
    });
    let mut frame = vec![FILL; HEADER_BYTES + REQUEST_BYTES];
    frame[..LENGTH_BYTES].copy_from_slice(&(REQUEST_BYTES as u32).to_le_bytes());
    frame[LENGTH_BYTES..HEADER_BYTES].copy_from_slice(&(RESPONSE_BYTES as u32).to_le_bytes());
    let mut reply = vec![0u8; RESPONSE_BYTES];

    let persistent = Relay::start(&args[0], &peer, &stats)?;
    let pinned = Relay::start(&args[0], &peer, &stats)?;
    let mut pinned_stream = connect(&pinned.address)?;
    let own = std::process::id();
    let mut batches = Vec::new();
    for round in 0..rounds {
        // Rotate the starting arm so no arm always follows the same one.
        for step in 0..ARMS.len() {
            let arm = ARMS[(round + step) % ARMS.len()];
            let fresh = match arm {
                "fresh" => Some(Relay::start(&args[0], &peer, &stats)?),
                _ => None,
            };
            let relay_pid = match arm {
                "fresh" => fresh.as_ref().map(|r| r.child.id()),
                "persistent" => Some(persistent.child.id()),
                "persistent-connection" => Some(pinned.child.id()),
                _ => None,
            };
            let before = (accounting(own), relay_pid.map(accounting));
            let started_ns = epoch.elapsed().as_nanos() as u64;
            let exchanges = match arm {
                "persistent-connection" => {
                    batch(&mut pinned_stream, epoch, &log, &frame, &mut reply)?
                }
                _ => {
                    let address = match arm {
                        "direct" => &peer,
                        "fresh" => &fresh.as_ref().expect("fresh relay").address,
                        _ => &persistent.address,
                    };
                    batch(&mut connect(address)?, epoch, &log, &frame, &mut reply)?
                }
            };
            let after = (accounting(own), relay_pid.map(accounting));
            batches.push(serde_json::json!({
                "arm": arm, "round": round, "started_ns": started_ns,
                "warmup": WARMUP, "exchanges": exchanges,
                "trace_process": after.0.since(before.0),
                "relay_process": after.1.zip(before.1).map(|(a, b)| a.since(b)),
            }));
        }
    }
    let report = serde_json::json!({
        "schema": "larql.v3.link-trace.v1",
        "purpose": "diagnostic only; not a calibration",
        "relay_binary": args[0], "request_bytes": REQUEST_BYTES,
        "response_bytes": RESPONSE_BYTES, "segments": ["outbound", "peer", "inbound"],
        "available_parallelism": std::thread::available_parallelism().map_or(0, usize::from),
        "batches": batches,
    });
    std::fs::write(
        &args[2],
        serde_json::to_vec(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
