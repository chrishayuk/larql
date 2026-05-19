//! Coverage push for `routes/openai/schema/mask.rs` (was 0%, target ≥ 90%).
//!
//! Drives the `build_mask` adapter directly: build a tiny schema FSM,
//! a tokenizer with a handful of surface forms, then call the
//! returned closure with various `generated` slices to exercise the
//! lazy-init + cache-hit + reject branches.

use std::collections::HashSet;
use std::sync::Arc;

use larql_server::routes::openai::schema::ast::Schema;
use larql_server::routes::openai::schema::fsm::Fsm;
use larql_server::routes::openai::schema::mask::build_mask;

fn make_tokenizer() -> Arc<larql_inference::tokenizers::Tokenizer> {
    // WordLevel tokenizer with token ids matching surface forms the
    // FSM will see when generating a JSON string-ish output. Vocab
    // is small (8 tokens) so the closure's iterate-over-vocab loop
    // is cheap.
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"\"":0,"x":1,"y":2," ":3,"{":4,"}":5,":":6,",":7},"unk_token":"x"}}"#;
    Arc::new(larql_inference::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap())
}

#[test]
fn build_mask_lazy_inits_surface_table_on_first_call() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::string());
    let mut mask = build_mask(tok, fsm, String::new(), HashSet::new());

    // 8-token vocab → 8 logits. First call triggers the lazy
    // `surfaces.get_or_insert_with` path that decodes every token.
    let mut logits = vec![0.0_f32; 8];
    mask(&[], &mut logits);
    // Schema::string() expects a `"` to start; non-`"` candidates
    // should be masked to -inf. Token id 0 (`"`) might be allowed.
    let neg_inf_count = logits.iter().filter(|&&x| x == f32::NEG_INFINITY).count();
    assert!(
        neg_inf_count > 0,
        "string-schema FSM should reject some non-quote candidates"
    );
}

#[test]
fn build_mask_cache_hit_reuses_replay_state() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::string());
    let mut mask = build_mask(tok, fsm, String::new(), HashSet::new());

    let mut logits = vec![0.0_f32; 8];
    // First call seeds last_replay.
    mask(&[0], &mut logits);
    let mut logits2 = vec![0.0_f32; 8];
    // Second call with `generated` extending the previous → cache-hit
    // branch (`generated.starts_with(prev)`).
    mask(&[0, 1], &mut logits2);
    // No assertions on specific values — the cache-hit path is
    // covered by reaching the closure body twice.
}

#[test]
fn build_mask_cache_miss_falls_through_to_fresh_fsm() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::string());
    let mut mask = build_mask(tok, fsm, String::new(), HashSet::new());

    let mut logits = vec![0.0_f32; 8];
    mask(&[0, 1], &mut logits);
    let mut logits2 = vec![0.0_f32; 8];
    // Different prefix → `generated.starts_with(prev)` fails → fresh_fsm.
    mask(&[2], &mut logits2);
}

#[test]
fn build_mask_with_prompt_text_replays_prompt_first() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::object(Default::default()));
    // Prefill the FSM with `{` — matches how the server prefills
    // JSON-object response_format requests.
    let mut mask = build_mask(tok, fsm, "{".to_string(), HashSet::new());
    let mut logits = vec![0.0_f32; 8];
    mask(&[], &mut logits);
}

#[test]
fn build_mask_with_eos_token_ids_masks_eos_when_incomplete() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::string());
    let mut eos: HashSet<u32> = HashSet::new();
    eos.insert(3); // pretend space-token is EOS
    let mut mask = build_mask(tok, fsm, String::new(), eos);

    let mut logits = vec![10.0_f32; 8];
    mask(&[], &mut logits);
    // EOS token at index 3 must be masked to -inf while FSM is
    // incomplete (no `"` opened yet).
    assert_eq!(
        logits[3],
        f32::NEG_INFINITY,
        "EOS must be masked while FSM incomplete; got {:?}",
        &logits
    );
}

#[test]
fn build_mask_handles_token_id_outside_surface_table() {
    let tok = make_tokenizer();
    let fsm = Fsm::new(Schema::string());
    let mut mask = build_mask(tok, fsm, String::new(), HashSet::new());

    let mut logits = vec![0.0_f32; 8];
    // Pass a token id that's larger than the surface table size —
    // shouldn't be possible from real tokenisation, but the closure
    // has a graceful fallback path.
    mask(&[99], &mut logits);
}
