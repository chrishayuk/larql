//! **The seven, and that there is no eighth.**

use larql_vindex::format::vindex3::represent::view::OptimizerView;

use super::super::protocol::METHOD_TOOLS_LIST;
use super::super::tools;
use super::{ask, record, server};

/// Every public method the facade exposes, from its own source, so the
/// declared surface is checked against the code that serves it rather
/// than against a list kept by hand in two crates.
const FACADE: &str =
    include_str!("../../../../../../larql-vindex/src/format/vindex3/represent/view.rs");

#[test]
fn the_declared_tools_are_exactly_the_facades_questions() {
    let declared: Vec<&str> = tools::all().iter().map(|t| t.name).collect();
    let questions: Vec<String> = FACADE
        .split_once("impl<'a> OptimizerView<'a> {")
        .expect("the facade impl block")
        .1
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub fn "))
        .filter_map(|rest| rest.split(['(', '<']).next())
        .filter(|name| *name != "new")
        .map(|name| format!("optimizer.{name}"))
        .collect();

    assert_eq!(
        declared, questions,
        "a question the facade answers and this server does not declare is a \
         capability nobody can reach; the reverse is a tool that cannot be served"
    );
    assert_eq!(declared.len(), 7);
}

#[test]
fn no_declared_tool_writes() {
    for tool in tools::all() {
        for forbidden in [
            "record", "apply", "expand", "promote", "accept", "measure", "search", "set", "add",
        ] {
            assert!(
                !tool.name.contains(forbidden),
                "`{}` is not a read",
                tool.name
            );
        }
    }
}

#[test]
fn tools_list_carries_a_schema_a_client_can_call_from() {
    let record = record();
    let answer = ask(
        &server(&record),
        serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": METHOD_TOOLS_LIST}),
    );
    let listed = answer["result"]["tools"].as_array().expect("an array");

    assert_eq!(listed.len(), 7);
    for tool in listed {
        assert!(!tool["name"].as_str().expect("a name").is_empty());
        assert!(
            tool["description"].as_str().expect("a description").len() > 40,
            "a one-word description tells an agent nothing about when to ask"
        );
        assert_eq!(tool["inputSchema"]["type"], "object");
    }
}

#[test]
fn the_tools_that_need_a_state_say_so_and_the_optional_one_does_not() {
    let by_name = |name: &str| {
        tools::all()
            .into_iter()
            .find(|t| t.name == name)
            .expect("declared")
    };

    let explain = by_name(tools::EXPLAIN);
    assert_eq!(
        explain.input_schema["required"],
        serde_json::json!(["state"])
    );

    let compare = by_name(tools::COMPARE);
    assert_eq!(
        compare.input_schema["required"],
        serde_json::json!(["left", "right"])
    );

    // Evidence over the whole record is a legitimate question, so its
    // state argument is optional rather than defaulted to something.
    let evidence = by_name(tools::EVIDENCE);
    assert_eq!(evidence.input_schema["required"], serde_json::json!([]));
    assert!(evidence.input_schema["properties"]["state"].is_object());

    for name in [
        tools::DESCRIBE,
        tools::CURRENT,
        tools::FRONTIER,
        tools::NEXT_EXPERIMENT,
    ] {
        assert_eq!(
            by_name(name).input_schema["required"],
            serde_json::json!([])
        );
    }
}

#[test]
fn the_facade_reachable_from_here_is_the_read_only_one() {
    // A compile-time fact worth stating once: the server is handed an
    // `OptimizerView`, and the view owns its snapshot privately. There
    // is no accessor back to the record, so no tool body can reach
    // past the seven questions even by accident.
    let record = record();
    let view = OptimizerView::new(&record);
    let _: &dyn Fn() -> _ = &|| view.describe();
}
