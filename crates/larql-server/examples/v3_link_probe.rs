//! DEC-V3-1: the calibration probe for `v3_link_shaper`.
//!
//! An echo peer and a client that time expert-shaped exchanges over one
//! persistent connection. Both ends poll their sockets instead of blocking,
//! so neither end's wake-up from sleep is charged to the link under test,
//! and the measured path allocates nothing.
//!
//! Polling occupies one core per end. That is an observer effect in its own
//! right: the harness records the relay's self-reported lateness under this
//! probe beside the same figure under a blocking probe, so starvation of the
//! relay by the probe would be visible rather than assumed away.
//!
//! No affinity is requested (macOS exposes no affinity control). The
//! scheduling class is the OS default unless `LARQL_LINK_QOS` asks otherwise;
//! the client reports the class it requested and the parallelism it observed.
//!
//! Frozen protocol: `bench/v3-routed-experts/LINK-1.md`.
//!
//! Usage:
//!   v3_link_probe serve
//!   v3_link_probe measure <address> <request_bytes> <response_bytes> <count> <warmup>

#[path = "support/qos.rs"]
mod qos;

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// Request frame prefix: request and response payload lengths, little-endian u32 each.
const HEADER_BYTES: usize = 8;
const LENGTH_BYTES: usize = 4;
const LOOPBACK: &str = "127.0.0.1:0";
/// Printed once the echo peer is bound, so a harness can read the real port.
const READY_PREFIX: &str = "listening";
/// Filler byte, so no page of a payload is all zero.
const FILL: u8 = 0xA5;
const MEDIAN: f64 = 0.50;
const P90: f64 = 0.90;
const MEASURE_ARGUMENTS: usize = 6;

/// Fill `buffer` by polling. `Ok(false)` is a clean close before any byte.
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

fn length(header: &[u8], field: usize) -> usize {
    let start = field * LENGTH_BYTES;
    let mut bytes = [0u8; LENGTH_BYTES];
    bytes.copy_from_slice(&header[start..start + LENGTH_BYTES]);
    u32::from_le_bytes(bytes) as usize
}

fn echo(mut stream: TcpStream) -> std::io::Result<()> {
    qos::apply().map_err(std::io::Error::other)?;
    polling(&stream)?;
    let mut header = [0u8; HEADER_BYTES];
    let (mut request, mut response) = (Vec::new(), Vec::new());
    while receive(&mut stream, &mut header)? {
        // Sized on the first exchange; later exchanges of the same shape reuse them.
        request.resize(length(&header, 0), 0);
        response.resize(length(&header, 1), FILL);
        if !receive(&mut stream, &mut request)? {
            break;
        }
        send(&mut stream, &response)?;
    }
    Ok(())
}

fn serve() -> Result<(), String> {
    qos::apply()?;
    let listener = TcpListener::bind(LOOPBACK).map_err(|e| e.to_string())?;
    println!(
        "{READY_PREFIX} {}",
        listener.local_addr().map_err(|e| e.to_string())?
    );
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || {
            let _ = echo(stream);
        });
    }
    Ok(())
}

fn quantile(sorted: &[u64], fraction: f64) -> u64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn measure(args: &[String]) -> Result<(), String> {
    let number = |i: usize| args[i].parse::<usize>().map_err(|e| e.to_string());
    let (request_bytes, response_bytes, count, warmup) =
        (number(2)?, number(3)?, number(4)?, number(5)?);
    if count == 0 {
        return Err("count must be positive".into());
    }
    let class = qos::apply()?;
    let mut stream = TcpStream::connect(&args[1]).map_err(|e| e.to_string())?;
    polling(&stream).map_err(|e| e.to_string())?;
    let mut frame = vec![FILL; HEADER_BYTES + request_bytes];
    frame[..LENGTH_BYTES].copy_from_slice(&(request_bytes as u32).to_le_bytes());
    frame[LENGTH_BYTES..HEADER_BYTES].copy_from_slice(&(response_bytes as u32).to_le_bytes());
    let mut reply = vec![0u8; response_bytes];
    let mut samples = Vec::with_capacity(count);
    for exchange in 0..count + warmup {
        let started = Instant::now();
        send(&mut stream, &frame).map_err(|e| e.to_string())?;
        if !receive(&mut stream, &mut reply).map_err(|e| e.to_string())? {
            return Err("echo peer closed".into());
        }
        if exchange >= warmup {
            samples.push(started.elapsed().as_nanos() as u64);
        }
    }
    samples.sort_unstable();
    let parallelism = std::thread::available_parallelism().map_or(0, usize::from);
    println!(
        "{{\"probe\":\"native-polling\",\"exchanges\":{},\"median_ns\":{},\"min_ns\":{},\
         \"p90_ns\":{},\"max_ns\":{},\"mean_ns\":{},\"available_parallelism\":{parallelism},\
         \"qos\":\"{class}\",\"affinity\":\"none requested\"}}",
        samples.len(),
        quantile(&samples, MEDIAN),
        samples[0],
        quantile(&samples, P90),
        samples[samples.len() - 1],
        samples.iter().sum::<u64>() / samples.len() as u64,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") if args.len() == 1 => serve(),
        Some("measure") if args.len() == MEASURE_ARGUMENTS => measure(&args),
        _ => Err(
            "usage: v3_link_probe serve | measure <address> <request_bytes> \
             <response_bytes> <count> <warmup>"
                .into(),
        ),
    }
}
