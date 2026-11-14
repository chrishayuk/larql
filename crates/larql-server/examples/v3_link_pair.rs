//! LINK-2: the paired calibration probe for `v3_link_shaper`.
//!
//! One long-lived process holds the client and the echo peer. It starts two
//! relays of the given (frozen) build once, a baseline relay at zero delay
//! and unlimited rate and a shaped relay at the point under test, opens one
//! connection through each, and measures them as interleaved pairs:
//!
//! ```text
//! pair = baseline block, shaped block, shaped block, baseline block
//! ```
//!
//! so both paths see the same host state and drift that is linear over a
//! pair cancels. Every exchange is written out; nothing is summarised,
//! judged or excluded here. The estimate, its interval and the decision
//! rules belong to `link_pair_model.py`, which reads this file's output.
//!
//! The block size, pair count and warm-up are arguments so the frozen values
//! live in one place (the protocol and its harness), not in two.
//!
//! Frozen protocol: `bench/v3-routed-experts/LINK-2-revision-3.md`, which
//! amends `LINK-2.md`.
//!
//! Usage:
//!   v3_link_pair <relay_binary> <one_way_delay_us> <rate_bits_per_s|inf> \
//!       <request_bytes> <response_bytes> <warmup> <block> <pairs> <out.json>

#[path = "support/polling.rs"]
mod polling;

use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const LOOPBACK: &str = "127.0.0.1:0";
const READY_PREFIX: &str = "listening ";
const ARGUMENTS: usize = 9;
/// The baseline relay's link: no delay, no rate limit.
const NO_DELAY: &str = "0";
const UNLIMITED: &str = "inf";
/// A relay appends two records when its connection closes; wait this long
/// for them before stopping it, so its lateness is not lost.
const STATS_WAIT: Duration = Duration::from_secs(5);
const STATS_POLL: Duration = Duration::from_millis(10);
const STATS_RECORDS: usize = 2;
/// Blocks of one pair, in order: false is the baseline path, true the shaped.
const PAIR_ORDER: [bool; 4] = [false, true, true, false];

struct Relay {
    child: Child,
    address: String,
    stats: String,
}

impl Relay {
    fn start(
        binary: &str,
        upstream: &str,
        delay: &str,
        rate: &str,
        stats: String,
    ) -> Result<Self, String> {
        let _ = std::fs::remove_file(&stats);
        let mut child = Command::new(binary)
            .args([LOOPBACK, upstream, delay, rate, &stats])
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
        Ok(Self {
            child,
            address,
            stats,
        })
    }

    /// The relay's own lateness records, once its connection has closed.
    fn finish(mut self) -> Result<Vec<serde_json::Value>, String> {
        let deadline = Instant::now() + STATS_WAIT;
        let records = loop {
            let text = std::fs::read_to_string(&self.stats).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() >= STATS_RECORDS {
                break lines
                    .iter()
                    .map(|line| serde_json::from_str(line).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            if Instant::now() > deadline {
                return Err(format!("relay statistics incomplete: {}", self.stats));
            }
            std::thread::sleep(STATS_POLL);
        };
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(records)
    }
}

fn connect(address: &str) -> Result<TcpStream, String> {
    let stream = TcpStream::connect(address).map_err(|e| e.to_string())?;
    polling::polling(&stream).map_err(|e| e.to_string())?;
    Ok(stream)
}

/// `count` exchanges on one connection; each duration in nanoseconds.
fn block(
    stream: &mut TcpStream,
    frame: &[u8],
    reply: &mut [u8],
    count: usize,
) -> Result<Vec<u64>, String> {
    let mut durations = Vec::with_capacity(count);
    for _ in 0..count {
        let started = Instant::now();
        polling::send(stream, frame).map_err(|e| e.to_string())?;
        if !polling::receive(stream, reply).map_err(|e| e.to_string())? {
            return Err("echo peer closed".into());
        }
        durations.push(started.elapsed().as_nanos() as u64);
    }
    Ok(durations)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != ARGUMENTS {
        return Err(
            "usage: v3_link_pair <relay_binary> <one_way_delay_us> <rate_bits_per_s|inf> \
             <request_bytes> <response_bytes> <warmup> <block> <pairs> <out.json>"
                .into(),
        );
    }
    let number = |i: usize| args[i].parse::<usize>().map_err(|e| e.to_string());
    let (request_bytes, response_bytes) = (number(3)?, number(4)?);
    let (warmup, block_size, pairs) = (number(5)?, number(6)?, number(7)?);
    if block_size == 0 || pairs == 0 {
        return Err("block and pairs must be positive".into());
    }
    let out = &args[8];

    let listener = TcpListener::bind(LOOPBACK).map_err(|e| e.to_string())?;
    let peer = listener
        .local_addr()
        .map_err(|e| e.to_string())?
        .to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || {
                let _ = polling::echo(stream);
            });
        }
    });

    let baseline = Relay::start(
        &args[0],
        &peer,
        NO_DELAY,
        UNLIMITED,
        format!("{out}.baseline-relay.jsonl"),
    )?;
    let shaped = Relay::start(
        &args[0],
        &peer,
        &args[1],
        &args[2],
        format!("{out}.shaped-relay.jsonl"),
    )?;
    let mut baseline_stream = connect(&baseline.address)?;
    let mut shaped_stream = connect(&shaped.address)?;
    let frame = polling::frame(request_bytes, response_bytes);
    let mut reply = vec![0u8; response_bytes];

    let started = Instant::now();
    block(&mut baseline_stream, &frame, &mut reply, warmup)?;
    block(&mut shaped_stream, &frame, &mut reply, warmup)?;
    let mut records = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let (mut baseline_ns, mut shaped_ns) = (Vec::new(), Vec::new());
        for shaped_block in PAIR_ORDER {
            if shaped_block {
                shaped_ns.extend(block(&mut shaped_stream, &frame, &mut reply, block_size)?);
            } else {
                baseline_ns.extend(block(&mut baseline_stream, &frame, &mut reply, block_size)?);
            }
        }
        records.push(serde_json::json!({"baseline_ns": baseline_ns, "shaped_ns": shaped_ns}));
    }
    let elapsed_ns = started.elapsed().as_nanos() as u64;
    drop(baseline_stream);
    drop(shaped_stream);
    let report = serde_json::json!({
        "schema": "larql.v3.link-pair.v1",
        "relay_binary": args[0],
        "one_way_delay_us": args[1], "rate_bits_per_s": args[2],
        "request_bytes": request_bytes, "response_bytes": response_bytes,
        "warmup": warmup, "block": block_size, "pairs": records,
        "pair_order": "baseline, shaped, shaped, baseline",
        "elapsed_ns": elapsed_ns,
        "available_parallelism": std::thread::available_parallelism().map_or(0, usize::from),
        "baseline_relay": baseline.finish()?, "shaped_relay": shaped.finish()?,
    });
    std::fs::write(out, serde_json::to_vec(&report).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
