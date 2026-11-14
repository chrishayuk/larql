//! Polling socket helpers shared by the LINK-2 paired probe: both ends of an
//! exchange poll their sockets instead of blocking, so neither end's wake-up
//! from sleep is charged to the link, and the measured path allocates nothing.

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

/// Request frame prefix: request and response payload lengths, little-endian u32 each.
pub const HEADER_BYTES: usize = 8;
pub const LENGTH_BYTES: usize = 4;
/// Filler byte, so no page of a payload is all zero.
pub const FILL: u8 = 0xA5;

/// Fill `buffer` by polling. `Ok(false)` is a clean close before any byte.
pub fn receive(stream: &mut TcpStream, buffer: &mut [u8]) -> std::io::Result<bool> {
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

pub fn send(stream: &mut TcpStream, mut bytes: &[u8]) -> std::io::Result<()> {
    while !bytes.is_empty() {
        match stream.write(bytes) {
            Ok(written) => bytes = &bytes[written..],
            Err(e) if e.kind() == ErrorKind::WouldBlock => std::hint::spin_loop(),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub fn polling(stream: &TcpStream) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_nonblocking(true)
}

/// A request frame asking for `response_bytes` back.
pub fn frame(request_bytes: usize, response_bytes: usize) -> Vec<u8> {
    let mut frame = vec![FILL; HEADER_BYTES + request_bytes];
    frame[..LENGTH_BYTES].copy_from_slice(&(request_bytes as u32).to_le_bytes());
    frame[LENGTH_BYTES..HEADER_BYTES].copy_from_slice(&(response_bytes as u32).to_le_bytes());
    frame
}

/// Answer every framed request on `stream` with the bytes it asks for.
pub fn echo(mut stream: TcpStream) -> std::io::Result<()> {
    polling(&stream)?;
    let mut header = [0u8; HEADER_BYTES];
    let (mut request, mut response) = (Vec::new(), Vec::new());
    let length = |header: &[u8], field: usize| {
        let start = field * LENGTH_BYTES;
        let mut bytes = [0u8; LENGTH_BYTES];
        bytes.copy_from_slice(&header[start..start + LENGTH_BYTES]);
        u32::from_le_bytes(bytes) as usize
    };
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
