//! Each mask contract arm, against an in-memory source.

use std::collections::BTreeMap;

use super::*;

/// A source serving named payloads from memory.
struct Memory(BTreeMap<String, Vec<u8>>);

impl TensorSource for Memory {
    fn stream_payload(
        &self,
        name: &str,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError> {
        let bytes = self
            .0
            .get(name)
            .ok_or_else(|| VindexError::Parse(format!("no `{name}`")))?;
        observe(bytes);
        write.write_all(bytes)?;
        Ok(bytes.len() as u64)
    }
}

fn policy(span: &str, window: Option<usize>) -> AttentionLayerPolicy {
    let mut json = serde_json::json!({
        "operator": "softmax", "span": span, "position": {"kind": "none"}
    });
    if let Some(w) = window {
        json["window"] = serde_json::json!(w);
    }
    serde_json::from_value(json).unwrap()
}

fn buffer(name: &str, kind: BufferKind, dtype: &str, shape: &[usize]) -> ExcludedTensor {
    ExcludedTensor {
        name: name.into(),
        dtype: dtype.into(),
        shape: shape.to_vec(),
        reason: "test".into(),
        kind,
    }
}

fn mask(n: usize, visible: impl Fn(usize, usize) -> bool) -> Vec<f32> {
    (0..n * n)
        .map(|e| f32::from(u8::from(visible(e / n, e % n))))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn run(buf: &ExcludedTensor, table: &[AttentionLayerPolicy], bytes: Vec<u8>) -> Result<(), String> {
    let source = Memory(BTreeMap::from([(buf.name.clone(), bytes)]));
    check_one(buf, Some(table), &source)
}

#[test]
fn causal_and_sliding_masks_check_against_the_declared_span() {
    let n = 5;
    let full = [policy("full", None)];
    let causal = buffer(
        "h.0.attn.bias",
        BufferKind::CausalMask,
        "F32",
        &[1, 1, n, n],
    );
    assert!(run(&causal, &full, f32_bytes(&mask(n, |i, j| j <= i))).is_ok());
    let err = run(&causal, &full, f32_bytes(&mask(n, |_, _| true))).unwrap_err();
    assert!(err.contains("query 0, key 1"), "{err}");

    let sliding = [policy("full", None), policy("sliding", Some(2))];
    let local = buffer("h.1.attn.bias", BufferKind::CausalMask, "U8", &[1, 1, n, n]);
    let bytes: Vec<u8> = mask(n, |i, j| j <= i && j + 2 > i)
        .iter()
        .map(|v| *v as u8)
        .collect();
    assert!(run(&local, &sliding, bytes).is_ok());
}

#[test]
fn a_mask_that_cannot_be_checked_refuses() {
    let n = 2;
    let full = [policy("full", None)];
    let bytes = || f32_bytes(&mask(n, |i, j| j <= i));
    // A non-square shape, a name with no layer, a layer with no span.
    let flat = buffer("h.0.attn.bias", BufferKind::CausalMask, "F32", &[n, n]);
    assert!(run(&flat, &full, bytes())
        .unwrap_err()
        .contains("[1, 1, n, n]"));
    let unlayered = buffer("attn.bias", BufferKind::CausalMask, "F32", &[1, 1, n, n]);
    assert!(run(&unlayered, &full, bytes())
        .unwrap_err()
        .contains("names no layer"));
    let past = buffer(
        "h.3.attn.bias",
        BufferKind::CausalMask,
        "F32",
        &[1, 1, n, n],
    );
    assert!(run(&past, &full, bytes())
        .unwrap_err()
        .contains("no declared attention span"));
    // A sliding layer with no window has no mask form.
    let windowless = [policy("sliding", None)];
    let zero = buffer(
        "h.0.attn.bias",
        BufferKind::CausalMask,
        "F32",
        &[1, 1, n, n],
    );
    assert!(run(&zero, &windowless, bytes())
        .unwrap_err()
        .contains("no mask form"));
    // An unreadable dtype, and a payload the source does not hold.
    let wide = buffer(
        "h.0.attn.bias",
        BufferKind::CausalMask,
        "I64",
        &[1, 1, n, n],
    );
    assert!(run(&wide, &full, vec![0; 32])
        .unwrap_err()
        .contains("not a mask dtype"));
    let source = Memory(BTreeMap::new());
    assert!(check_one(&zero, Some(&full), &source).is_err());
}

#[test]
fn a_fill_must_drive_masked_weight_to_zero() {
    let full = [policy("full", None)];
    let fill = buffer("h.0.attn.masked_bias", BufferKind::MaskFill, "F32", &[]);
    assert!(run(&fill, &full, f32_bytes(&[-1e4])).is_ok());
    assert!(run(&fill, &full, f32_bytes(&[-1.0]))
        .unwrap_err()
        .contains("would keep weight"));
    assert!(run(&fill, &full, f32_bytes(&[-1e4, -1e4]))
        .unwrap_err()
        .contains("one value"));
    // Half-precision fills decode exactly.
    let half = buffer("h.0.attn.masked_bias", BufferKind::MaskFill, "BF16", &[]);
    // A BF16 copy of -1e4 rounds to -9984, still far past the ceiling.
    assert!(run(&half, &full, (0xc61cu16).to_le_bytes().to_vec()).is_ok());
    let f16 = buffer("h.0.attn.masked_bias", BufferKind::MaskFill, "F16", &[]);
    assert!(run(&f16, &full, (0xfbffu16).to_le_bytes().to_vec()).is_ok()); // -65504
    let i8 = buffer("h.0.attn.masked_bias", BufferKind::MaskFill, "I8", &[]);
    assert!(run(&i8, &full, vec![0x80]).is_err()); // -128: too weak
}

#[test]
fn every_excluded_buffer_of_every_artifact_is_checked_or_refused() {
    use crate::format::vindex3::fixtures_gpt2::{miniature_gpt2, GPT2_LAYERS};
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt2(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let named = [("target".to_string(), inventory)];
    let plan = crate::format::vindex3::plan::plan_system(&named);
    let source = super::super::source::ArtifactSource::open(dir.path()).unwrap();
    let sources: BTreeMap<&str, &dyn TensorSource> = BTreeMap::from([("target", &source as _)]);
    assert_eq!(
        check_excluded_buffers(&plan.graph, &named, &sources).unwrap(),
        2 * GPT2_LAYERS
    );
    // An artifact with excluded buffers and no payload source refuses.
    let empty: BTreeMap<&str, &dyn TensorSource> = BTreeMap::new();
    assert!(check_excluded_buffers(&plan.graph, &named, &empty).is_err());
}
