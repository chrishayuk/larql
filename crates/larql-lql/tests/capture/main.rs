//! `CAPTURE`: what the session's program computed at one site, written to a NumPy file with a sidecar that names
//! the model state. Every value is held to the executor's own: logits to INFER's pass bit for bit, the down
//! projection's input to `DownInputCapture`, a patched capture to a capture of the container COMPILE writes.

mod hyper_connection;
mod live;
mod parity;
mod parsing;
mod refusals;
mod shapes;
mod sidecar;
mod support;
mod typed;
