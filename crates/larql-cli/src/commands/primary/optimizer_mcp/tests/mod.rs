//! **The transport carries; it does not derive.**
//!
//! Every test here drives the server the way a client does — frames in,
//! frames out — over the real Rung 5 record. The load-bearing one is
//! [`dispatch::every_tool_answers_with_the_views_own_words`]: each
//! tool's payload is compared byte for byte against what the view
//! itself renders, so a transport that reshaped, summarised, ordered or
//! annotated an answer fails rather than being noticed in review.

mod command;
mod dispatch;
mod protocol;
mod tools;

use larql_vindex::format::vindex3::represent::state::fixtures;
use larql_vindex::format::vindex3::represent::state::snapshot::SearchSnapshot;
use larql_vindex::format::vindex3::represent::view::OptimizerView;

use super::server::Server;

/// The Rung 5 record, stored and read back — the same round trip the
/// command performs when it opens a file.
pub(super) fn record() -> SearchSnapshot {
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_str(&json).expect("deserialize");
    back.check_schema().expect("schema");
    back
}

pub(super) fn server(record: &SearchSnapshot) -> Server<'_> {
    Server::new(OptimizerView::new(record))
}

/// Send one frame and parse the answer.
pub(super) fn ask(server: &Server<'_>, frame: serde_json::Value) -> serde_json::Value {
    let answer = server
        .answer(&frame.to_string())
        .expect("a request with an id is answered");
    serde_json::from_str(&answer).expect("the answer is JSON")
}

/// A `tools/call` frame.
pub(super) fn call(id: u64, name: &str, arguments: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": { "name": name, "arguments": arguments },
    })
}

/// The text a successful tool call carried.
pub(super) fn payload(answer: &serde_json::Value) -> &str {
    answer["result"]["content"][0]["text"]
        .as_str()
        .expect("a text content block")
}
