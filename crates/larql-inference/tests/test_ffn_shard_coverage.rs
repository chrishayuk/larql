//! A remote FFN shard map must give every layer exactly one owner.
//!
//! `FfnBackend::forward` cannot fail. For a layer no shard owned, the
//! sharded backend answered with zeros, so a map with a hole ran the model
//! with that layer's FFN missing and reported nothing. The map is now
//! checked when the backend connects.
//!
//! Each "shard" here is a loopback HTTP server that answers every request
//! with a fixed stats body.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use larql_inference::ffn::remote::LayerShardedBackend;

const TIMEOUT: Duration = Duration::from_secs(10);
const HIDDEN: usize = 4;
/// The layer count the shards report unless a test says otherwise.
const LAYERS: usize = 6;

/// A server whose stats report `layers` layers, or none when `None`.
fn shard(layers: Option<usize>) -> SocketAddr {
    let body = match layers {
        Some(layers) => format!("{{\"hidden_size\":{HIDDEN},\"layers\":{layers}}}"),
        None => format!("{{\"hidden_size\":{HIDDEN}}}"),
    };
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    addr
}

/// Connect a map of `(start, end)` layer ranges, one shard each.
fn connect(ranges: &[(usize, usize)], layers: Option<usize>) -> Result<(), String> {
    let spec: Vec<String> = ranges
        .iter()
        .map(|(start, end)| format!("{start}-{end}=http://{}", shard(layers)))
        .collect();
    LayerShardedBackend::connect(&spec.join(","), TIMEOUT)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn a_map_that_covers_every_layer_connects() {
    connect(&[(0, 2), (3, 5)], Some(LAYERS)).expect("a full map");
    connect(&[(3, 5), (0, 2)], Some(LAYERS)).expect("in any order");
    connect(&[(0, 99)], Some(LAYERS)).expect("a range past the last layer");
}

#[test]
fn a_hole_between_two_ranges_is_refused_by_layer() {
    let err = connect(&[(0, 2), (4, 5)], Some(LAYERS)).expect_err("layer 3 has no owner");

    assert!(err.contains("layer(s) 3-3"), "{err}");
}

#[test]
fn layers_past_the_last_range_are_refused() {
    let err = connect(&[(0, 2)], Some(LAYERS)).expect_err("layers 3-5 have no owner");

    assert!(err.contains("layer(s) 3-5"), "{err}");
}

#[test]
fn a_map_that_does_not_start_at_layer_zero_is_refused() {
    let err = connect(&[(1, 5)], Some(LAYERS)).expect_err("layer 0 has no owner");

    assert!(err.contains("layer(s) 0-0"), "{err}");
}

#[test]
fn a_layer_claimed_by_two_shards_is_refused() {
    let err = connect(&[(0, 3), (3, 5)], Some(LAYERS)).expect_err("layer 3 has two owners");

    assert!(err.contains("layer 3 is claimed by two shards"), "{err}");
}

#[test]
fn a_hole_is_refused_even_when_no_shard_reports_a_layer_count() {
    connect(&[(0, 2), (3, 5)], None).expect("no hole");

    let err = connect(&[(0, 2), (4, 5)], None).expect_err("layer 3 has no owner");

    assert!(err.contains("layer(s) 3-3"), "{err}");
}

#[test]
fn shards_that_report_different_layer_counts_are_refused() {
    let spec = format!(
        "0-2=http://{},3-5=http://{}",
        shard(Some(LAYERS)),
        shard(Some(LAYERS + 1))
    );

    let err = match LayerShardedBackend::connect(&spec, TIMEOUT) {
        Ok(_) => panic!("two models behind one map"),
        Err(err) => err.to_string(),
    };

    assert!(err.contains("different models"), "{err}");
}

#[test]
fn a_single_server_with_no_map_owns_every_layer() {
    let url = format!("http://{}", shard(Some(LAYERS)));

    LayerShardedBackend::connect(&url, TIMEOUT).expect("one server, every layer");
}
