//! DEC-V3-1: a controlled link in front of one VINDEX3 worker.
//!
//! A TCP relay that imposes a one-way propagation delay and a serialization
//! rate on each direction independently. The coordinator connects to the
//! relay instead of the worker; the worker, the wire frames and every
//! numerical path are untouched, so the link is the only thing that changes
//! between a shaped arm and its unshaped control.
//!
//! Link model, per direction, per chunk read from the sender:
//!
//! ```text
//! start   = max(arrival, link_free)        the link carries one chunk at a time
//! finish  = start + bits / rate            serialization at the configured rate
//! release = finish + delay                 one-way propagation
//! ```
//!
//! Each connection is one polling loop over both sockets: there is no thread
//! hand-off and no blocking read inside the relay, because a sleeping thread's
//! wake-up (tens of microseconds, and longer after a longer idle) would be
//! charged to the link and would differ between a zero-delay control and a
//! delayed arm. The price is one busy core per open connection.
//!
//! This models delay and bandwidth only. It does not model loss, reordering,
//! jitter or congestion control, and it is not a kernel shaper: the relay's
//! own cost is measured by a zero-delay, unlimited-rate arm, never assumed.
//! Each direction reports how late its chunks were actually released, so a
//! sweep point can be rejected when the instrument missed its schedule.
//!
//! The relay also opens a loopback control listener and prints its address
//! after the ready line. A control connection that sends `mark` snapshots
//! every open connection's tallies at a quiescent boundary and answers
//! whether the boundary was clean; mark records go to `<stats.jsonl>.marks`,
//! never to the stats file (VERIFY-REMOTE-1 revision 2: relay lateness by
//! phase). Unmarked, the relay records exactly what it always did. The core
//! is `support/relay.rs`.
//!
//! Frozen protocol: `bench/v3-routed-experts/LINK-1.md`.
//!
//! Usage:
//!   cargo run --release -p larql-server --no-default-features \
//!       --example v3_link_shaper -- \
//!       <listen> <upstream> <one_way_delay_us> <rate_bits_per_s|inf> <stats.jsonl>

#[path = "support/qos.rs"]
mod qos;
#[path = "support/relay.rs"]
mod relay;

use relay::{Link, Marks, SNAPSHOT_TIMEOUT, UNLIMITED};
use std::io::Write;
use std::net::TcpListener;
use std::time::Duration;

/// Printed once the listener is bound, so a harness can read the real port.
const READY_PREFIX: &str = "listening";
/// Printed next: the control listener's address.
const CONTROL_PREFIX: &str = "control";
/// The control listener only ever binds loopback.
const CONTROL_LISTEN: &str = "127.0.0.1:0";
/// Mark records go beside the stats file, never into it.
const MARKS_SUFFIX: &str = ".marks";
const ARGUMENTS: usize = 5;

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
    let control = TcpListener::bind(CONTROL_LISTEN).map_err(|e| e.to_string())?;
    let control_bound = control.local_addr().map_err(|e| e.to_string())?;
    let marks = Marks::new(format!("{}{MARKS_SUFFIX}", args[4]), SNAPSHOT_TIMEOUT);
    println!("{READY_PREFIX} {bound}");
    println!("{CONTROL_PREFIX} {control_bound}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let control_marks = marks.clone();
    std::thread::spawn(move || {
        for stream in control.incoming().flatten() {
            relay::control(stream, &control_marks);
        }
    });
    for client in listener.incoming() {
        let Ok(client) = client else { continue };
        // Admitted on accept, so a mark either watches it or counts it after.
        let (watched, seen) = marks.admit();
        let (upstream, stats, marks) = (args[1].clone(), args[4].clone(), marks.clone());
        std::thread::spawn(move || {
            let Ok(class) = qos::apply() else {
                marks.unwatch(&watched);
                return;
            };
            relay::serve(
                client, &upstream, link, &stats, class, &marks, watched, seen,
            );
        });
    }
    Ok(())
}
