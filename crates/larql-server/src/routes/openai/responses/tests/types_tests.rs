use crate::routes::openai::responses::types::*;

#[test]
fn input_accepts_bare_string() {
    let req: ResponsesRequest =
        serde_json::from_value(serde_json::json!({"input": "hello"})).unwrap();
    assert!(matches!(req.input, ResponseInput::Text(ref s) if s == "hello"));
}

#[test]
fn input_accepts_item_list_with_part_content() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"}
        ]
    }))
    .unwrap();
    let ResponseInput::Items(items) = req.input else {
        panic!("expected items");
    };
    assert_eq!(items.len(), 3);
    assert_eq!(items[1].kind.as_deref(), Some("function_call"));
    assert_eq!(items[2].call_id.as_deref(), Some("call_1"));
}

#[test]
fn request_optional_knobs_deserialise() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "input": "hi",
        "instructions": "be brief",
        "max_output_tokens": 7,
        "temperature": 0.4,
        "top_p": 0.9,
        "stream": true,
        "store": false,
        "previous_response_id": "resp_0",
        "metadata": {"k": "v"},
        "background": false,
        "truncation": "auto",
        "user": "u1",
        "stop": ["\n"]
    }))
    .unwrap();
    assert_eq!(req.max_output_tokens, Some(7));
    assert_eq!(req.store, Some(false));
    assert_eq!(req.previous_response_id.as_deref(), Some("resp_0"));
    assert_eq!(req.truncation.as_deref(), Some("auto"));
    assert!(req.stop.is_some());
}

#[test]
fn output_item_serialises_with_type_tag() {
    let item = OutputItem::Message {
        id: "msg_1".into(),
        status: STATUS_COMPLETED.into(),
        role: "assistant",
        content: vec![OutputContent::OutputText {
            text: "hi".into(),
            annotations: Vec::new(),
        }],
    };
    let v = serde_json::to_value(&item).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["content"][0]["type"], "output_text");
    assert_eq!(v["content"][0]["text"], "hi");
    assert!(v["content"][0]["annotations"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn function_call_item_serialises_flat() {
    let item = OutputItem::FunctionCall {
        id: "fc_1".into(),
        status: STATUS_COMPLETED.into(),
        call_id: "call_1".into(),
        name: "f".into(),
        arguments: "{}".into(),
    };
    let v = serde_json::to_value(&item).unwrap();
    assert_eq!(v["type"], "function_call");
    assert_eq!(v["name"], "f");
    assert_eq!(v["call_id"], "call_1");
}

fn envelope(output: Vec<OutputItem>) -> ResponseObject {
    ResponseObject {
        id: "resp_1".into(),
        object: RESPONSE_OBJECT,
        created_at: 7,
        status: STATUS_COMPLETED.into(),
        error: None,
        incomplete_details: None,
        model: "m".into(),
        output,
        previous_response_id: None,
        instructions: None,
        max_output_tokens: None,
        temperature: None,
        top_p: None,
        metadata: None,
        usage: None,
    }
}

#[test]
fn output_text_concatenates_message_parts_and_skips_calls() {
    let resp = envelope(vec![
        OutputItem::Message {
            id: "msg_1".into(),
            status: STATUS_COMPLETED.into(),
            role: "assistant",
            content: vec![OutputContent::OutputText {
                text: "a".into(),
                annotations: Vec::new(),
            }],
        },
        OutputItem::FunctionCall {
            id: "fc_1".into(),
            status: STATUS_COMPLETED.into(),
            call_id: "call_1".into(),
            name: "f".into(),
            arguments: "{}".into(),
        },
        OutputItem::Message {
            id: "msg_2".into(),
            status: STATUS_COMPLETED.into(),
            role: "assistant",
            content: vec![OutputContent::OutputText {
                text: "b".into(),
                annotations: Vec::new(),
            }],
        },
    ]);
    assert_eq!(resp.output_text(), "ab");
}

#[test]
fn envelope_serialises_openai_field_names() {
    let mut resp = envelope(Vec::new());
    resp.status = STATUS_INCOMPLETE.into();
    resp.incomplete_details = Some(IncompleteDetails {
        reason: INCOMPLETE_MAX_OUTPUT_TOKENS,
    });
    resp.previous_response_id = Some("resp_0".into());
    resp.usage = Some(ResponseUsage {
        input_tokens_details: super::super::types::InputTokensDetails {
            cached_tokens: 0,
            continuation_refused: None,
        },
        input_tokens: 3,
        output_tokens: 5,
        total_tokens: 8,
    });
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["created_at"], 7);
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(v["usage"]["input_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], 8);
    assert_eq!(v["previous_response_id"], "resp_0");
    // `error` is always present (null), matching OpenAI envelopes.
    assert!(v.get("error").is_some());
    assert!(v["error"].is_null());
}
