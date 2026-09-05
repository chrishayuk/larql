//! **Dispatch and serialisation, and nothing else.**
//!
//! The server holds an [`OptimizerView`] and never a
//! [`SearchSnapshot`], so "derive nothing in transport" is a matter of
//! what is reachable from here rather than of what a reviewer noticed.
//! Every tool body is one call and one `to_string`.

use std::io::{BufRead, Write};

use larql_vindex::format::vindex3::represent::state::RepresentationStateId;
use larql_vindex::format::vindex3::represent::view::OptimizerView;

use super::protocol::*;
use super::tools;

/// What this server calls itself when a client asks.
pub const SERVER_NAME: &str = "larql-optimizer";
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A read-only MCP server over one search record.
pub struct Server<'a> {
    view: OptimizerView<'a>,
}

impl<'a> Server<'a> {
    pub fn new(view: OptimizerView<'a>) -> Self {
        Self { view }
    }

    /// Read requests until end of input, writing one line of JSON per
    /// answered request.
    ///
    /// A blank line is skipped and an unparseable one is answered with
    /// a parse error rather than closing the connection: a client that
    /// sends one bad frame should learn that, not lose the session.
    pub fn serve(&self, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let Some(response) = self.answer(&line) else {
                continue;
            };
            writeln!(output, "{response}")?;
            output.flush()?;
        }
        Ok(())
    }

    /// One frame in, at most one frame out. `None` for a notification,
    /// which the protocol answers with silence.
    pub fn answer(&self, frame: &str) -> Option<String> {
        let response = match serde_json::from_str::<Request>(frame) {
            Ok(request) => self.handle(request)?,
            Err(e) => Response::error(
                serde_json::Value::Null,
                ERROR_PARSE,
                format!("could not parse the request: {e}"),
            ),
        };
        Some(serde_json::to_string(&response).expect("a response serializes"))
    }

    fn handle(&self, request: Request) -> Option<Response> {
        if request.is_notification() {
            return None;
        }
        let id = request.id.clone().unwrap_or(serde_json::Value::Null);
        if request.jsonrpc != JSONRPC_VERSION {
            return Some(Response::error(
                id,
                ERROR_INVALID_REQUEST,
                format!("this server speaks JSON-RPC {JSONRPC_VERSION}"),
            ));
        }
        Some(match request.method.as_str() {
            METHOD_INITIALIZE => Response::result(id, Self::initialize()),
            METHOD_PING => Response::result(id, serde_json::json!({})),
            METHOD_TOOLS_LIST => Response::result(id, serde_json::json!({ "tools": tools::all() })),
            METHOD_TOOLS_CALL => match self.call(&request.params) {
                Ok(result) => Response::result(id, result),
                Err(e) => Response::error(id, e.code, e.message),
            },
            other => Response::error(
                id,
                ERROR_METHOD_NOT_FOUND,
                format!("this server does not serve `{other}`"),
            ),
        })
    }

    fn initialize() -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
            "instructions": "A read-only view of one LARQL physical-plan search. \
                             Every answer is derived from the stored record on demand; \
                             nothing here writes, records or promotes.",
        })
    }

    fn call(&self, params: &serde_json::Value) -> Result<serde_json::Value, Refused> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Refused::params("a tools/call needs a `name`"))?;
        let arguments = params.get("arguments").unwrap_or(&serde_json::Value::Null);

        let result = match name {
            tools::DESCRIBE => ToolResult::of(&self.view.describe(), false),
            tools::CURRENT => ToolResult::of(&self.view.current(), false),
            tools::FRONTIER => ToolResult::of(&self.view.frontier(), false),
            tools::NEXT_EXPERIMENT => ToolResult::of(&self.view.next_experiment(), false),
            tools::EXPLAIN => {
                let state = required_state(arguments, tools::ARG_STATE)?;
                match self.view.explain(&state) {
                    Some(explanation) => ToolResult::of(&explanation, false),
                    None => ToolResult::of(&serde_json::json!({ "not_held": state }), true),
                }
            }
            tools::COMPARE => {
                let left = required_state(arguments, tools::ARG_LEFT)?;
                let right = required_state(arguments, tools::ARG_RIGHT)?;
                match self.view.compare(&left, &right) {
                    Ok(comparison) => ToolResult::of(&comparison, false),
                    Err(not_held) => ToolResult::of(&not_held, true),
                }
            }
            tools::EVIDENCE => {
                let state = optional_state(arguments, tools::ARG_STATE);
                ToolResult::of(&self.view.evidence(state.as_ref()), false)
            }
            other => {
                return Err(Refused::params(format!(
                    "this server does not serve a tool called `{other}`"
                )))
            }
        };
        let result = result.map_err(|e| Refused::params(format!("could not render: {e}")))?;
        serde_json::to_value(result).map_err(|e| Refused::params(format!("could not render: {e}")))
    }
}

/// A call the server understood and would not answer.
pub struct Refused {
    pub code: i64,
    pub message: String,
}

impl Refused {
    fn params(message: impl Into<String>) -> Self {
        Self {
            code: ERROR_INVALID_PARAMS,
            message: message.into(),
        }
    }
}

fn required_state(
    arguments: &serde_json::Value,
    name: &str,
) -> Result<RepresentationStateId, Refused> {
    optional_state(arguments, name)
        .ok_or_else(|| Refused::params(format!("this tool needs a `{name}` state id")))
}

fn optional_state(arguments: &serde_json::Value, name: &str) -> Option<RepresentationStateId> {
    let raw = arguments.get(name)?;
    serde_json::from_value(raw.clone()).ok()
}
