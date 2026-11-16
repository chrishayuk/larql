//! The trial's side of the relay's generation-boundary mark
//! (VERIFY-REMOTE-1 revision 2).
//!
//! The trial marks once, after prefill has committed and before its clock
//! starts, and records exactly what came back. It sends no generation
//! exchange until the answer arrives or the wait ends. It judges nothing:
//! the driver voids a trial whose mark was not answered `ok`, or whose answer
//! disagrees with the relay's own record.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// The command the relay's control listener takes.
pub const MARK: &str = "mark";
/// How long the trial waits for the relay's answer.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);
/// The flag value for a trial run without a relay.
pub const NONE: &str = "none";

/// What one mark attempt produced, as recorded in the trial.
pub struct Outcome {
    pub control: String,
    /// The answer line, trimmed; `None` if none arrived.
    pub answer: Option<String>,
    pub error: Option<String>,
    pub waited_ns: u64,
    /// Exchanges the trial had started, all completed, when it marked.
    pub exchanges_completed: usize,
}

impl Outcome {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "control": self.control,
            "answer": self.answer,
            "error": self.error,
            "waited_ns": self.waited_ns,
            "exchanges_completed": self.exchanges_completed,
        })
    }
}

/// Mark the relay at `control` and wait at most `timeout` for its answer.
pub fn mark(control: &str, timeout: Duration, exchanges_completed: usize) -> Outcome {
    let started = Instant::now();
    let result = ask(control, timeout);
    let waited_ns = started.elapsed().as_nanos() as u64;
    let (answer, error) = match result {
        Ok(answer) => (Some(answer), None),
        Err(error) => (None, Some(error)),
    };
    Outcome {
        control: control.to_string(),
        answer,
        error,
        waited_ns,
        exchanges_completed,
    }
}

fn ask(control: &str, timeout: Duration) -> Result<String, String> {
    let address = control
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .next()
        .ok_or("the control address resolves to nothing")?;
    let mut stream = TcpStream::connect_timeout(&address, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(format!("{MARK}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    let mut answer = String::new();
    let read = BufReader::new(stream)
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    if read == 0 {
        return Err("the relay closed without answering".into());
    }
    Ok(answer.trim().to_string())
}
