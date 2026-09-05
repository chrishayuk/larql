//! Does executing the stored blocks give what decoding them gives?
//!
//! `kquant_conformance_tests.rs` settles that we READ the ecosystem's
//! bytes correctly. This settles a different claim, and the difference
//! matters: **"a kernel exists" and "a kernel implements exactly the
//! comparison the authority path makes" are separate facts.** PARETO-1
//! rung A decodes every K-quant to f32 and runs an f32 GEMV precisely so
//! that kernel quality cannot move a behavioural curve. Executing the
//! blocks directly is a different program, and it is only admissible as
//! the campaign executor if it agrees with the one it replaces.
//!
//! ```text
//! the SAME stored bytes
//!     ├─ decode -> f32 -> multiply    the authority path
//!     └─ direct K-quant GEMV          the candidate
//! ```
//!
//! The bytes are llama.cpp's, from `ggml_kquant_golden.json`, so neither
//! side can be right for the wrong reason: this is not LARQL's encoder
//! checked against LARQL's decoder checked against LARQL's kernel.
//!
//! Layer 1 of the gate frozen in `~/chris-models/pareto1/V3-QUALIFICATION.md`
//! (the gate section as frozen hashed 9ddcc968…; the file has since had
//! RESULTS appended below it and its hash moves with them, the thresholds
//! do not). Layer 2 — a real projection through `PhysicalProjectionPlan`
//! — is `exec/tests/kquant_projection.rs`; layer 3, end-to-end logits, is
//! the one that carries the KL thresholds and runs against the anchors.

use larql_compute::backend::QuantMatVec;
use larql_compute::cpu::CpuBackend;
use larql_compute::QuantFormat;

use crate::format::vindex3::represent::kquant;

#[derive(serde::Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    encoding: String,
    elements: usize,
    quantised_hex: String,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("fixtures/ggml_kquant_golden.json"))
        .expect("the golden fixture parses")
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// Activations spanning sign and magnitude, deterministic so a failure
/// is reproducible.
fn activations(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 23) as f32 - 11.0) / 13.0).collect()
}

/// The authority path, written out longhand: decode the stored bytes,
/// then multiply in element order. This is deliberately NOT a call to a
/// BLAS gemv — the claim under test is about the association the
/// decoder defines, and a library's blocked accumulation would import a
/// second unknown into the comparison.
fn decode_then_multiply(bytes: &[u8], name: &str, n: usize, k: usize) -> Vec<f32> {
    let decoded = kquant::lookup(name)
        .expect("a known K-quant")
        .decode(bytes, n * k, "fixture")
        .expect("decode");
    let x = activations(k);
    (0..n)
        .map(|r| {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += decoded[r * k + c] * x[c];
            }
            acc
        })
        .collect()
}

/// The candidate: the codec's OWN direct-execution association, which is
/// the one production dispatches through. Layer 1 therefore covers the
/// dispatch table and not only the kernels behind it — a test that kept
/// its own copy of that table could pass while production routed
/// differently.
fn direct(bytes: &[u8], name: &str, n: usize, k: usize) -> Option<Vec<f32>> {
    let x = activations(k);
    kquant::lookup(name)
        .expect("a known K-quant")
        .gemv(bytes, &x, n, k)
}

/// Relative difference, reported rather than merely thresholded, so a
/// failure says how far off it was and not only that it was off.
fn worst_relative(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = (x.abs().max(y.abs()) as f64).max(1e-6);
            ((*x as f64) - (*y as f64)).abs() / denom
        })
        .fold(0.0f64, f64::max)
}

/// The fixture's canonical name for a case, since the JSON spells them
/// `q8_0` / `q6_K` / `q4_K`.
fn canonical(encoding: &str) -> &'static str {
    match encoding.to_ascii_lowercase().as_str() {
        "q8_0" => "Q8_0",
        "q6_k" => "Q6_K",
        "q4_k" => "Q4_K",
        other => panic!("unexpected encoding `{other}` in the golden fixture"),
    }
}

/// Every case in the fixture, at two shapes, direct against decoded.
///
/// The measured agreement is asserted per encoding rather than with one
/// shared bound: Q8_0's kernel folds the f16 scale per element and is
/// therefore bit-for-bit with the decoder, and holding it to a loose
/// tolerance would stop testing the property it was written to have.
#[test]
fn direct_execution_agrees_with_decode_then_multiply() {
    let g = golden();
    assert_eq!(g.cases.len(), 3, "the fixture should carry all three");
    for c in &g.cases {
        let name = canonical(&c.encoding);
        let bytes = hex_to_bytes(&c.quantised_hex);
        for (n, k) in [(1, c.elements), (2, c.elements / 2)] {
            let want = decode_then_multiply(&bytes, name, n, k);
            let Some(got) = direct(&bytes, name, n, k) else {
                panic!("{name} [{n},{k}]: no direct kernel answered");
            };
            let rel = worst_relative(&got, &want);
            // Printed, not merely thresholded. "Passed" hides whether
            // the agreement was 1e-7 or 9e-4, and layer 1 of the gate is
            // a NUMBER that belongs in the campaign record.
            println!("  layer-1  {name:<5} [{n},{k}]  worst relative {rel:e}");
            if name == "Q8_0" {
                assert_eq!(
                    got, want,
                    "{name} [{n},{k}]: the Q8_0 kernel folds the scale per element and \
                     must be bit-for-bit with the decoder; worst relative {rel:e}"
                );
            } else {
                // Calibrated to what was MEASURED, not to a round
                // number: Q6_K came in at 3.6e-7 / 6.4e-7 and Q4_K at
                // 8.6e-7 / 8.3e-7, which is f32 accumulation order
                // (sqrt(512) * f32 eps is about 2.7e-6) rather than a
                // different program. A 1e-3 bound would pass a kernel
                // that had genuinely regressed; this one leaves about
                // 6x headroom over the worst observed.
                assert!(
                    rel < 5e-6,
                    "{name} [{n},{k}]: direct execution differs from the decoder by \
                     {rel:e} relative — too large to be accumulation order"
                );
            }
        }
    }
}

/// The comparison must be able to fail.
///
/// An earlier check in this campaign reported IDENTICAL by comparing two
/// empty strings, and a gate that cannot return the other answer is not
/// evidence. Perturbing one stored byte must move the result.
#[test]
fn the_agreement_check_can_report_disagreement() {
    let g = golden();
    for c in &g.cases {
        let name = canonical(&c.encoding);
        let mut bytes = hex_to_bytes(&c.quantised_hex);
        let (n, k) = (1, c.elements);
        let clean = decode_then_multiply(&bytes, name, n, k);
        // Flip a bit in a code byte, past any block header.
        let victim = bytes.len() / 2;
        bytes[victim] ^= 0x10;
        let dirty = decode_then_multiply(&bytes, name, n, k);
        assert_ne!(
            clean, dirty,
            "{name}: perturbing a stored byte did not change the decoded product, so \
             this comparison could not have detected a wrong one"
        );
    }
}

/// The codec refuses a geometry that does not describe the blocks, before
/// any kernel sees them — the stride check is the codec's, not the
/// kernel's, so a wrong shape over the right total length is caught by
/// the one party that knows the layout.
#[test]
fn the_codec_refuses_a_geometry_that_does_not_describe_the_blocks() {
    let g = golden();
    for c in &g.cases {
        let name = canonical(&c.encoding);
        let k = kquant::lookup(name).expect("a known K-quant");
        let bytes = hex_to_bytes(&c.quantised_hex);
        let (n, width) = (1, c.elements);
        let x = activations(width);
        assert!(
            k.gemv(&bytes, &x, n, width).is_some(),
            "{name}: the control case"
        );
        // More rows than the stream holds.
        assert!(k.gemv(&bytes, &x, n + 1, width).is_none(), "{name}: rows");
        // A width off the block grid.
        assert!(
            k.gemv(&bytes, &activations(width - 1), n, width - 1)
                .is_none(),
            "{name}: ragged width"
        );
        // An activation that is not the row's width.
        assert!(
            k.gemv(&bytes, &activations(width / 2), n, width).is_none(),
            "{name}: activation width"
        );
        // A stream one byte short.
        assert!(
            k.gemv(&bytes[..bytes.len() - 1], &x, n, width).is_none(),
            "{name}: truncated stream"
        );
    }
}

/// Q8_0 must not be answered by the pre-existing `QuantFormat::Q8_0`
/// path, which is a DIFFERENT layout under the same name: int8 codes
/// with an external per-block f32 scale stream, against ggml's inline
/// f16 scale in a 34-byte block. Routing one to the other is not
/// hypothetical — it happened in this workspace and produced garbage.
#[test]
fn the_larql_q8_0_format_does_not_answer_for_ggml_blocks() {
    let g = golden();
    let c = g
        .cases
        .iter()
        .find(|c| canonical(&c.encoding) == "Q8_0")
        .expect("a Q8_0 case");
    let bytes = hex_to_bytes(&c.quantised_hex);
    let x = activations(c.elements);
    assert!(
        CpuBackend
            .quant_matvec(QuantFormat::Q8_0, &bytes, &x, 1, c.elements)
            .is_none(),
        "`QuantFormat::Q8_0` answered for ggml Q8_0 bytes; those are different layouts \
         and a silent answer is the 34-vs-18 byte stride bug returning"
    );
}
