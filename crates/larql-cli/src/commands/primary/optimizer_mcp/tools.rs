//! **The seven questions, declared.**
//!
//! Every tool here reads. There is no `record`, no `apply`, no
//! `expand`, no `promote` and above all no `accept_candidate` — an
//! agent chooses which question to ask, and the optimiser and the
//! evidence system decide what is true.

use serde::Serialize;

pub const DESCRIBE: &str = "optimizer.describe";
pub const CURRENT: &str = "optimizer.current";
pub const FRONTIER: &str = "optimizer.frontier";
pub const EXPLAIN: &str = "optimizer.explain";
pub const COMPARE: &str = "optimizer.compare";
pub const EVIDENCE: &str = "optimizer.evidence";
pub const NEXT_EXPERIMENT: &str = "optimizer.next_experiment";

/// The argument every state-scoped tool takes.
pub const ARG_STATE: &str = "state";
pub const ARG_LEFT: &str = "left";
pub const ARG_RIGHT: &str = "right";

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

fn no_arguments() -> serde_json::Value {
    serde_json::json!({ "type": "object", "properties": {}, "required": [] })
}

fn state_argument(name: &str, required: bool) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            name: { "type": "string", "description": "A representation state id." }
        },
        "required": if required { vec![name] } else { Vec::new() },
    })
}

/// Every tool this server serves, in the order a reader should meet
/// them: what the search is, where it stands, then the questions about
/// particular states.
pub fn all() -> Vec<Tool> {
    vec![
        Tool {
            name: DESCRIBE,
            description: "What this search IS: the model, its tensor surface, the frozen \
                          behavioural contract, the transition policy, and every decision \
                          procedure a conclusion is drawn under. Read this first — it says \
                          what the numbers in every other answer were judged by.",
            input_schema: no_arguments(),
        },
        Tool {
            name: CURRENT,
            description: "Where the search stands: the graph's shape, the incumbent (the \
                          cheapest state carrying an authority reading that satisfies the \
                          contract), every admitted state cheapest first, and the states \
                          carrying no reading at each evidence scale.",
            input_schema: no_arguments(),
        },
        Tool {
            name: FRONTIER,
            description: "Every state the graph holds, with each observation of it \
                          adjudicated against the frozen gate: admissible, sound, the \
                          binding constraint, and every criterion it failed. Recomputed \
                          from the readings, never stored.",
            input_schema: no_arguments(),
        },
        Tool {
            name: EXPLAIN,
            description: "One state: what it IS (its footprint and the realizations that \
                          present it) kept separate from how it was REACHED (every incoming \
                          edge, each with its own action and provenance). One state can \
                          have several explanations of how it was arrived at.",
            input_schema: state_argument(ARG_STATE, true),
        },
        Tool {
            name: COMPARE,
            description: "Two states side by side, with the difference between their \
                          footprints and the edge that joins them where the graph holds \
                          one. Does not say whether their readings are comparable: two \
                          observations under different banks, scales or instruments are \
                          different experiments.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    ARG_LEFT: { "type": "string", "description": "A representation state id." },
                    ARG_RIGHT: { "type": "string", "description": "A representation state id." }
                },
                "required": [ARG_LEFT, ARG_RIGHT],
            }),
        },
        Tool {
            name: EVIDENCE,
            description: "The measurement record: each experiment's raw bank beside the \
                          verdict the contract draws from it, with the calibrations and \
                          policies that say how a statistic may be read. Optionally \
                          narrowed to one state.",
            input_schema: state_argument(ARG_STATE, false),
        },
        Tool {
            name: NEXT_EXPERIMENT,
            description: "What to measure next — which this record cannot say, and the \
                          refusal names exactly which facts are missing. Returns the move \
                          vocabulary and the unmeasured states, which need no pricing.",
            input_schema: no_arguments(),
        },
    ]
}
