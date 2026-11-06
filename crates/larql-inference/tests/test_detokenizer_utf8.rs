//! The streaming detokeniser never prints part of a character: a
//! multi-byte character split across tokens is emitted whole by the token
//! that completes it, and one cut off by the end of generation is dropped
//! rather than printed as U+FFFD.

use larql_inference::Detokenizer;
use tokenizers::Tokenizer;

/// 😊 in UTF-8: four bytes, one byte-fallback token each.
const EMOJI: &str = "\u{1F60A}";
const EMOJI_BYTES: [u32; 4] = [0, 1, 2, 3];
const HI: u32 = 4;
const THERE: u32 = 5;
const REPLACEMENT: char = char::REPLACEMENT_CHARACTER;

/// A byte-fallback tokenizer: `<0xNN>` tokens decode to raw bytes (and an
/// incomplete run to U+FFFD, as SentencePiece/Llama-style decoders do);
/// word tokens decode to themselves.
fn byte_fallback_tokenizer() -> Tokenizer {
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "ByteFallback"},
            {"type": "Fuse"}
        ]},
        "model": {"type": "WordLevel", "vocab": {
            "<0xF0>": 0, "<0x9F>": 1, "<0x98>": 2, "<0x8A>": 3,
            "hi": 4, "there": 5, "[UNK]": 6
        }, "unk_token": "[UNK]"}
    });
    Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

#[test]
fn the_fixture_decodes_a_partial_character_as_a_replacement() {
    // The premise of the hold-back: the decoder itself renders an
    // incomplete character as U+FFFD.
    let tok = byte_fallback_tokenizer();
    let partial = tok.decode(&EMOJI_BYTES[..2], true).unwrap();
    assert!(partial.ends_with(REPLACEMENT), "{partial:?}");
    assert_eq!(tok.decode(&EMOJI_BYTES, true).unwrap(), EMOJI);
}

#[test]
fn a_character_split_across_tokens_is_emitted_whole_once_complete() {
    let tok = byte_fallback_tokenizer();
    let mut detok = Detokenizer::new(&tok);
    assert_eq!(detok.push(HI), "hi");
    let deltas: Vec<String> = EMOJI_BYTES.iter().map(|&id| detok.push(id)).collect();
    assert_eq!(deltas[..3], ["", "", ""]);
    assert_eq!(deltas[3], EMOJI);
    assert_eq!(detok.push(THERE), "there");
    assert_eq!(detok.cumulative(), format!("hi{EMOJI}there"));
}

#[test]
fn a_stop_partway_through_a_character_prints_no_replacement() {
    let tok = byte_fallback_tokenizer();
    let mut detok = Detokenizer::new(&tok);
    let mut streamed = detok.push(HI);
    for &id in &EMOJI_BYTES[..3] {
        streamed.push_str(&detok.push(id));
    }
    // Generation stops here, mid-character.
    assert_eq!(streamed, "hi");
    assert!(!streamed.contains(REPLACEMENT));
}

#[test]
fn a_seed_ending_mid_character_completes_in_the_first_push() {
    let tok = byte_fallback_tokenizer();
    let mut detok = Detokenizer::new(&tok);
    detok.seed(&[HI, EMOJI_BYTES[0], EMOJI_BYTES[1]]);
    assert_eq!(detok.push(EMOJI_BYTES[2]), "");
    // The completed character is new text, though half its bytes were
    // in the prompt: it was never printed, so it is printed now.
    assert_eq!(detok.push(EMOJI_BYTES[3]), EMOJI);
}
