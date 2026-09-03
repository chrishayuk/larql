//! Conformance against a FOREIGN reference, not against ourselves.
//!
//! `kquant_tests.rs` proves `decode(encode(x)) ≈ x`. That is
//! self-consistency, and it passes just as happily if the encoder and
//! the decoder share one misunderstanding of the block layout — a
//! swapped nibble order, a scale and a min the wrong way round, a
//! sub-scale sequence read backwards. Every such pair round-trips
//! perfectly and is wrong against the rest of the world.
//!
//! So the bytes here come from **llama.cpp's ggml, not from LARQL**.
//! `fixtures/ggml_kquant_golden.gen.c` links against ggml and dumps, for
//! one frozen input:
//!
//! ```text
//! input_bits          the f32 input, as bit patterns
//! blck_size/type_size ggml's OWN geometry for the type
//! quantised_hex       what ggml's quantiser produced
//! dequantised_bits    what ggml's own decoder read back from those bytes
//! ```
//!
//! Two independent claims are separated here, because they are different
//! claims and the head-to-head needs to know which one is held:
//!
//! ```text
//! LAYOUT CONFORMANCE   decode_larql(bytes_ggml) == decode_ggml(bytes_ggml)
//!                      "we read the ecosystem's bytes correctly"
//! ENCODER EQUIVALENCE  encode_larql(x) == bytes_ggml
//!                      "we reproduce llama.cpp's quantiser exactly"
//! ```
//!
//! The first is required for a compiled pack to be a GGUF byte
//! pass-through and for any interoperability claim. The second is
//! *optional*: choosing scales differently is a legitimate encoder
//! difference, but if it holds we must say "VINDEX3's Q4_K encoder using
//! the ggml Q4_K representation" rather than "llama.cpp's Q4_K".

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    input_bits: Vec<u32>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    encoding: String,
    ggml_type: u32,
    blck_size: usize,
    type_size: usize,
    elements: usize,
    quantised_hex: String,
    dequantised_bits: Vec<u32>,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("fixtures/ggml_kquant_golden.json"))
        .expect("the ggml golden fixture parses")
}

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

/// ggml names its types lowercase (`q6_K`); the map vocabulary is upper.
fn ours(name: &str) -> KQuant {
    lookup(&name.to_uppercase()).unwrap_or_else(|| panic!("no LARQL encoding for ggml `{name}`"))
}

/// The geometry table, checked against **ggml's own** `blck_size` and
/// `type_size` rather than against this workspace's decoders. Those
/// decoders are where the table's numbers were read from, so they cannot
/// independently confirm it.
#[test]
fn ggml_agrees_with_the_geometry_table() {
    let g = golden();
    assert_eq!(
        g.cases.len(),
        COMPILABLE.len(),
        "fixture covers every encoding"
    );
    for c in &g.cases {
        let k = ours(&c.encoding);
        assert_eq!(k.elements_per_block, c.blck_size, "{}: blck_size", k.name);
        assert_eq!(k.bytes_per_block, c.type_size, "{}: type_size", k.name);
        assert_eq!(
            k.encoded_len(c.elements, "t").unwrap(),
            hex_to_bytes(&c.quantised_hex).len(),
            "{}: encoded_len disagrees with ggml's row size",
            k.name
        );
    }
}

/// **A type id is a shared contract, exactly like a byte layout.**
///
/// This is the check that caught `TYPE_Q8_0 = 6` / `TYPE_Q5_0 = 8` —
/// transposed against upstream, so a GGUF carrying either decoded as the
/// other. It survived every existing test because internal callers pass
/// these constants both ways round; the id never left the workspace
/// until it crossed the FFI to ggml, which answered with 352 bytes where
/// 544 were expected (16 Q5_0 blocks, not 16 Q8_0 blocks).
///
/// The ids come from `ggml_get_type_traits` itself, not from a header
/// this workspace transcribed.
#[test]
fn ggml_agrees_with_our_type_ids() {
    for c in &golden().cases {
        let k = ours(&c.encoding);
        assert_eq!(
            k.ggml_type, c.ggml_type,
            "{}: this build says type id {}, ggml says {} — a GGUF carrying this type \
             would be decoded as whatever {} actually is",
            k.name, k.ggml_type, c.ggml_type, k.ggml_type
        );
    }
}

/// **The load-bearing test.** LARQL's decoder, on bytes it did not
/// produce, must read what ggml reads from them.
///
/// Asserted as bit-for-bit equality: both sides widen the same stored
/// f16 scales and integer quants, so any difference is a layout or
/// arithmetic disagreement, not rounding. If this ever weakens to a
/// tolerance, the reason must be written down here.
#[test]
fn larql_decodes_ggml_bytes_exactly_as_ggml_does() {
    for c in &golden().cases {
        let k = ours(&c.encoding);
        let bytes = hex_to_bytes(&c.quantised_hex);
        let theirs = bits_to_f32(&c.dequantised_bits);
        let mine = k.decode(&bytes, c.elements, "golden").expect("decodes");

        assert_eq!(mine.len(), theirs.len(), "{}: element count", k.name);
        let differing = mine
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{}: {differing}/{} values differ from ggml's own decode of the same bytes \
             — the layouts disagree",
            k.name,
            theirs.len()
        );
    }
}

/// The fixture has to be capable of failing. If the input were constant,
/// a ramp, or symmetric, a wrong nibble order or a swapped scale/min
/// could still decode to the right answer.
#[test]
fn the_golden_input_is_adversarial_enough_to_catch_a_layout_error() {
    let g = golden();
    let input = bits_to_f32(&g.input_bits);
    assert!(input.len() >= 512, "at least two super-blocks");

    // Signed on both sides, so an unsigned misreading cannot pass.
    assert!(input.iter().any(|v| *v < -1.0) && input.iter().any(|v| *v > 1.0));
    // Not monotone: a ramp hides a reversed sub-block order.
    let ascending = input.windows(2).filter(|w| w[1] > w[0]).count();
    assert!(
        ascending > input.len() / 8 && ascending < input.len() * 7 / 8,
        "the input must not be monotone; {ascending} of {} steps ascend",
        input.len() - 1
    );
    // The two super-blocks must differ, or a fixture would not detect a
    // decoder that reads block 0 twice.
    let (a, b) = (&input[..256], &input[256..512]);
    assert_ne!(a, b, "the two super-blocks must not be identical");

    // And the proof that the check has teeth: reversing the bytes of a
    // real Q6_K pack must change what comes out.
    let c = g
        .cases
        .iter()
        .find(|c| c.encoding.eq_ignore_ascii_case("q6_K"))
        .unwrap();
    let k = ours(&c.encoding);
    let bytes = hex_to_bytes(&c.quantised_hex);
    let mut scrambled = bytes.clone();
    scrambled.reverse();
    let straight = k.decode(&bytes, c.elements, "t").unwrap();
    let reversed = k.decode(&scrambled, c.elements, "t").unwrap();
    assert_ne!(straight, reversed, "the decode ignores byte order");
}

/// LARQL's encoder does **not** reproduce llama.cpp's bytes, and this
/// test records that as a standing fact rather than hiding it.
///
/// Measured against the golden fixture, on one frozen 512-element input:
///
/// ```text
/// Q8_0   10/544 bytes differ
/// Q6_K  366/420 bytes differ
/// Q4_K   94/288 bytes differ
/// ```
///
/// This is a legitimate encoder difference, not a layout error — layout
/// is proven bit-for-bit by
/// [`larql_decodes_ggml_bytes_exactly_as_ggml_does`]. The claim LARQL
/// holds is therefore:
///
/// > **VINDEX3's Q4_K encoder, using the ggml Q4_K representation.**
///
/// NOT "llama.cpp's Q4_K quantisation". Any comparison written up
/// against a llama.cpp-derived artifact must say which of those it means.
#[test]
fn larql_encodes_the_ggml_representation_with_its_own_chosen_values() {
    let g = golden();
    let input = bits_to_f32(&g.input_bits);
    for c in &g.cases {
        let k = ours(&c.encoding);
        let theirs = hex_to_bytes(&c.quantised_hex);
        let mine = k.encode(&input[..c.elements], "golden").expect("encodes");
        assert_eq!(
            mine.len(),
            theirs.len(),
            "{}: byte count must still match",
            k.name
        );
        // Recorded, not asserted equal. If this ever becomes zero for
        // every encoding, the stronger claim has been earned and this
        // test should be promoted to assert it.
        let differing = mine.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        assert!(
            differing > 0,
            "{}: bytes now match ggml exactly — the STRONGER claim is available, \
             promote this test rather than leaving it recording a weaker one",
            k.name
        );
    }
}

/// A quality test for the NATIVE encoder — not an eligibility test for
/// REPRESENT research.
///
/// It once was the latter, asserting LARQL within 5% of ggml, on the
/// reasoning that a codec deficit would contaminate a behaviour-per-byte
/// curve. It failed at Q6_K (1.1146), and the right fix was
/// architectural rather than statistical:
///
/// > **A comparative campaign does not call the LARQL encoder at all.**
///
/// With `--features reference-encoder`, K-quant payloads for comparative
/// artifacts come from ggml itself, so codec implementation quality is
/// removed from the causal graph by construction rather than by a
/// threshold that has to keep passing. A regression here can therefore
/// no longer perturb an experiment months later, because that experiment
/// never called this code.
///
/// What remains is worth keeping: the native encoders are executable
/// format documentation, a self-contained fallback, and differential-test
/// subjects. This records their standing against the ecosystem and
/// catches a real regression.
///
/// ```text
/// encoding   LARQL rms      ggml rms      ratio    (frozen input, 512 values)
/// Q8_0       8.690570e-3   8.719152e-3   0.9967
/// Q4_K       1.409723e-1   1.365251e-1   1.0326
/// Q6_K       3.476481e-2   3.119103e-2   1.1146   <- most scale search to lose
/// ```
///
/// The deficit tracks how much scale search a format admits: Q8_0 has
/// one scale per 32 values and nothing to search; Q6_K has 16 int8
/// sub-scales plus an f16 `d`. A mechanism, not noise — and the standing
/// research target if the native encoders are ever improved.
#[test]
fn the_native_encoders_hold_their_measured_standing_against_ggml() {
    let g = golden();
    let input = bits_to_f32(&g.input_bits);
    let rms = |a: &[f32], b: &[f32]| -> f64 {
        (a.iter()
            .zip(b)
            .map(|(x, y)| ((*x - *y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    };
    // Every encoding measured before anything is asserted: stopping at
    // the first offender would hide how wide a regression is.
    let mut rows = Vec::new();
    for c in &g.cases {
        let k = ours(&c.encoding);
        let src = &input[..c.elements];
        let theirs = rms(src, &bits_to_f32(&c.dequantised_bits));
        let ours_bytes = k.encode(src, "golden").expect("encodes");
        let mine = rms(
            src,
            &k.decode(&ours_bytes, c.elements, "golden")
                .expect("decodes"),
        );
        rows.push((k.name, mine / theirs));
    }
    let table = rows
        .iter()
        .map(|(n, r)| format!("{n} {r:.4}"))
        .collect::<Vec<_>>()
        .join(", ");

    // A ceiling set from the measured standing plus headroom, so an
    // actual regression fails while the known Q6_K gap does not. NOT a
    // research gate — see this test's docs.
    let regressed: Vec<_> = rows.iter().filter(|(_, r)| *r > 1.20).collect();
    assert!(
        regressed.is_empty(),
        "a native encoder regressed against the ecosystem reference — {table}"
    );
    // And the floor: if these ever reach parity, the native encoders have
    // caught up and this test should be tightened rather than left loose.
    let all_parity = rows.iter().all(|(_, r)| *r <= 1.01);
    assert!(
        !all_parity,
        "the native encoders now match ggml within 1% ({table}) — tighten this test \
         rather than leaving a loose bound recording a gap that no longer exists"
    );
}
