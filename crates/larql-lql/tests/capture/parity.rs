//! A capture is what the executor computed: held bit for bit to INFER's own pass, to INFER GENERATE's tokens and
//! to the decode step's down-projection capture.

use larql_inference::vindex3::{encode_raw_prompt, PlaneEvent};
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationFactory;
use larql_vindex::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;

use crate::support::*;

/// How many tokens the generation tests continue by.
const NEW_TOKENS: u32 = 4;
const GENERATED_IDS: &str = "  ids:";

#[test]
fn logits_at_the_last_prompt_position_are_infers_bit_for_bit() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("logits.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &out);
    let captured = read_npy(&out);
    assert_eq!(captured.shape, vec![1, VOCAB]);
    assert_eq!(
        bits(&captured.values),
        bits(&infer_logits(&fx.container, PROMPT))
    );

    // And INFER, run through the session, ranks the same top token.
    let printed = run(&mut s, &format!("INFER \"{PROMPT}\" TOP 1;"));
    let top = captured
        .values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0;
    assert!(has(&printed, &format!("[id {top}]")), "{printed:?}");
}

#[test]
fn a_continued_position_equals_infer_on_the_concatenated_text() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let text = fx.path("text.npy");
    capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\" CONTINUE \"{CONTINUATION}\""),
        &text,
    );
    let joined = format!("{PROMPT}{CONTINUATION}");
    assert_eq!(
        bits(&read_npy(&text).values),
        bits(&infer_logits(&fx.container, &joined))
    );
    // The same sequence given by id, and given as one prompt, captures the same values at every position.
    let by_id = fx.path("ids.npy");
    capture(
        &mut s,
        &format!("residual AT LAYER 1 FOR \"{PROMPT}\" CONTINUE TOKENS ({PARIS_ID}) POSITIONS ALL"),
        &by_id,
    );
    let as_prompt = fx.path("prompt.npy");
    capture(
        &mut s,
        &format!("residual AT LAYER 1 FOR \"{joined}\" POSITIONS ALL"),
        &as_prompt,
    );
    assert_eq!(read_npy(&by_id).shape, vec![PROMPT_TOKENS + 1, HIDDEN]);
    assert_eq!(
        bits(&read_npy(&by_id).values),
        bits(&read_npy(&as_prompt).values)
    );
}

#[test]
fn the_per_layer_and_final_sites_are_infers_planes() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let runtime = runtime(&fx.container);
    let tokenizer = larql_vindex::load_vindex_tokenizer(&fx.container).unwrap();
    let ids = encode_raw_prompt(&tokenizer, PROMPT, None).unwrap();
    let continuation = runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .unwrap();
    let mut planes = Vec::new();
    let output = runtime
        .execute_streaming(&ids, &continuation, &mut |event| {
            if let PlaneEvent::Layer { trace, .. } = event {
                planes.push(trace);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(planes.len(), LAYERS);
    for (layer, trace) in planes.iter().enumerate() {
        for (site, want) in [
            ("residual", trace.post_layer.rows()),
            ("post_attention", trace.post_attention.rows()),
            ("ffn.input", &trace.ffn_input[..]),
        ] {
            let out = fx.path(&format!("{site}{layer}.npy"));
            capture(
                &mut s,
                &format!("{site} AT LAYER {layer} FOR \"{PROMPT}\" POSITIONS ALL"),
                &out,
            );
            assert_eq!(
                bits(&read_npy(&out).values),
                bits(&want.concat()),
                "{site} {layer}"
            );
        }
    }
    let out = fx.path("final.npy");
    capture(&mut s, &format!("final FOR \"{PROMPT}\""), &out);
    assert_eq!(bits(&read_npy(&out).values), bits(output.final_hidden()));
}

#[test]
fn the_down_input_is_the_decode_steps_own_capture() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("down.npy");
    capture(
        &mut s,
        &format!("ffn.down_input AT LAYERS (0, 1) FOR \"{PROMPT}\" POSITIONS (1, 4)"),
        &out,
    );
    let captured = read_npy(&out);
    assert_eq!(captured.shape, vec![2, 2, INTERMEDIATE]);

    let runtime = runtime(&fx.container);
    let tokenizer = larql_vindex::load_vindex_tokenizer(&fx.container).unwrap();
    let ids = encode_raw_prompt(&tokenizer, PROMPT, None).unwrap();
    let continuation = runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .unwrap();
    let mut session = runtime.session(&continuation).unwrap();
    let mut direct = DownInputCapture::at([(0, 1), (0, 4), (1, 1), (1, 4)]);
    for &token in &ids {
        session.step_observed(token, &mut direct).unwrap();
    }
    for (block, layer) in [0usize, 1].into_iter().enumerate() {
        let got = rows(&captured, block);
        for (row, position) in [1usize, 4].into_iter().enumerate() {
            assert_eq!(bits(&got[row]), bits(direct.get(layer, position).unwrap()));
        }
    }
    let check = &sidecar(&out)["execution"]["cross_check"];
    assert_eq!(check["bit_identical"], true, "{check}");
    assert!(sidecar(&out)["execution"]["path"]
        .as_str()
        .unwrap()
        .contains("decode steps"));
}

#[test]
fn generate_captures_along_the_tokens_infer_generate_prints() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let printed = run(
        &mut s,
        &format!("INFER \"{PROMPT}\" GENERATE {NEW_TOKENS};"),
    );
    let ids: Vec<u64> = printed
        .iter()
        .find_map(|l| l.strip_prefix(GENERATED_IDS))
        .unwrap()
        .trim()
        .split(',')
        .filter(|id| !id.is_empty())
        .map(|id| id.parse().unwrap())
        .collect();
    assert!(!ids.is_empty());

    let out = fx.path("generated.npy");
    capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\" GENERATE {NEW_TOKENS} POSITIONS REPLY"),
        &out,
    );
    let car = sidecar(&out);
    let recorded: Vec<u64> = car["sequence"]["continuation"]["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(recorded, ids);
    let sequence: Vec<u64> = car["sequence"]["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(&sequence[PROMPT_TOKENS..], &ids[..]);
    // The same capture as teacher-forcing those ids.
    let list = ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let forced = fx.path("forced.npy");
    capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\" CONTINUE TOKENS ({list}) POSITIONS REPLY"),
        &forced,
    );
    assert_eq!(read_npy(&out).shape, vec![ids.len(), VOCAB]);
    assert_eq!(
        bits(&read_npy(&out).values),
        bits(&read_npy(&forced).values)
    );
    // Greedy: each reply position's top logit is the token generated after it.
    for (row, next) in rows(&read_npy(&out), 0).iter().zip(&ids) {
        let top = row
            .iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            )
            .0;
        assert_eq!(top as u64, *next);
    }
}
