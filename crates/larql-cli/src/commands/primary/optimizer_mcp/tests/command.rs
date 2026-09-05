//! **Opening the record**, and refusing the ones that are not one.

use larql_vindex::format::vindex3::represent::state::fixtures;

use super::super::{declare, dispatch, load, serve, OptimizerMcpArgs};
use super::record;

/// Write the Rung 5 record to a file, the way an operator would.
fn stored(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write");
    path
}

#[test]
fn a_stored_record_loads_and_is_the_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("json");
    let path = stored(dir.path(), "snapshot.json", &json);

    let loaded = load(&path).expect("a record");
    assert_eq!(loaded.graph().len(), record().graph().len());
    assert_eq!(loaded.schema(), record().schema());
}

#[test]
fn a_missing_file_is_refused_by_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("nothing-here.json");

    let refusal = load(&missing).expect_err("there is no file").to_string();
    assert!(
        refusal.contains("nothing-here.json"),
        "an operator with several records needs to know which one: {refusal}"
    );
}

#[test]
fn a_file_that_is_not_a_record_is_refused_before_a_word_of_it_is_served() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = stored(dir.path(), "not-a-record.json", r#"{"hello": "world"}"#);

    let refusal = load(&path).expect_err("not a record").to_string();
    assert!(refusal.contains("not-a-record.json"));
}

#[test]
fn a_record_this_build_does_not_know_the_schema_of_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut value: serde_json::Value =
        serde_json::to_value(fixtures::rung5_snapshot()).expect("value");
    value["schema"] = serde_json::json!("represent-search-snapshot/v99");
    let path = stored(dir.path(), "future.json", &value.to_string());

    // A reader that does not know the schema string should not trust
    // its reading of anything under it, so nothing under it is served.
    load(&path).expect_err("an unknown schema is refused, not served partially");
}

#[test]
fn listing_the_tools_needs_no_record_at_all() {
    // The declarations are static, so an operator can see what a client
    // would see without having a search to point at.
    let mut output = Vec::new();
    declare(&mut output).expect("the tool list does not open the record");

    let declared: serde_json::Value =
        serde_json::from_slice(&output).expect("valid JSON on stdout");
    assert_eq!(declared.as_array().expect("an array").len(), 7);
}

#[test]
fn the_whole_command_answers_from_a_record_on_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("json");
    let path = stored(dir.path(), "snapshot.json", &json);

    let frames = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"optimizer.current","arguments":{}}}"#,
        "\n",
    );
    let mut output = Vec::new();
    serve(
        &path,
        std::io::BufReader::new(frames.as_bytes()),
        &mut output,
    )
    .expect("serve");

    let lines: Vec<&str> = std::str::from_utf8(&output)
        .expect("utf8")
        .lines()
        .collect();
    assert_eq!(lines.len(), 2, "the notification is not an answer");

    let current: serde_json::Value = serde_json::from_str(lines[1]).expect("JSON");
    let body: serde_json::Value = serde_json::from_str(
        current["result"]["content"][0]["text"]
            .as_str()
            .expect("text"),
    )
    .expect("the payload is structured");
    assert_eq!(body["states"], 4);
    assert_eq!(body["incumbent"]["logical_bytes"], 13_684_764_800u64);
}

#[test]
fn a_record_that_cannot_be_opened_stops_the_command_before_it_serves() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut output = Vec::new();
    let refusal = serve(
        &dir.path().join("absent.json"),
        std::io::BufReader::new(&b"{}"[..]),
        &mut output,
    )
    .expect_err("there is no record");

    assert!(refusal.to_string().contains("absent.json"));
    assert!(output.is_empty(), "nothing was served");
}

#[test]
fn the_verb_takes_either_arm_from_the_same_arguments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("json");
    let path = stored(dir.path(), "snapshot.json", &json);

    let mut listed = Vec::new();
    dispatch(
        &OptimizerMcpArgs {
            snapshot: path.clone(),
            list_tools: true,
        },
        std::io::BufReader::new(&b""[..]),
        &mut listed,
    )
    .expect("list");
    let declared: serde_json::Value = serde_json::from_slice(&listed).expect("JSON");
    assert_eq!(declared.as_array().expect("array").len(), 7);

    let mut served = Vec::new();
    dispatch(
        &OptimizerMcpArgs {
            snapshot: path,
            list_tools: false,
        },
        std::io::BufReader::new(&br#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#[..]),
        &mut served,
    )
    .expect("serve");
    assert!(std::str::from_utf8(&served)
        .expect("utf8")
        .contains("\"id\":1"));
}
