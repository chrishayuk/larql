//! checkpoint.rs: the shared one-checkpoint pipeline (rung M2)

use super::*;

/// The full pipeline on the miniature checkpoint: encode succeeds, the
/// container detects as V3, and the capability files present in the
/// checkpoint (tokenizer.json here) are placed beside the segments.
#[test]
fn encode_checkpoint_produces_a_v3_container_with_capabilities() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    std::fs::write(checkpoint.path().join("tokenizer.json"), "{}").unwrap();
    std::fs::write(checkpoint.path().join("generation_config.json"), "{}").unwrap();
    std::fs::write(checkpoint.path().join("tokenizer.model"), b"\n\x00").unwrap();

    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(checkpoint.path(), out.path()).unwrap();

    assert!(encoded.outcome.representations > 0);
    assert_eq!(
        crate::format::generation::detect_generation(out.path()).unwrap(),
        crate::format::generation::ContainerGeneration::V3,
    );
    assert_eq!(
        encoded.capabilities,
        vec![
            "tokenizer.json".to_string(),
            "generation_config.json".to_string(),
            "tokenizer.model".to_string()
        ],
        "present capability files copy, in declaration order"
    );
    assert!(out.path().join("tokenizer.json").exists());
    assert!(out.path().join("generation_config.json").exists());
    assert!(out.path().join("tokenizer.model").exists());
    assert!(!out.path().join("chat_template.jinja").exists());
}

/// A tokenizer-less checkpoint still encodes — absence narrows
/// capability, it is not an error.
#[test]
fn encode_checkpoint_tolerates_absent_capability_files() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(checkpoint.path(), out.path()).unwrap();
    assert!(encoded.capabilities.is_empty());
    assert!(!out.path().join("tokenizer.json").exists());
}

/// A directory that is not an HF checkpoint refuses by name — the
/// message says what the encoder consumes.
#[test]
fn encode_checkpoint_refuses_a_non_checkpoint_dir() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;

    let not_a_checkpoint = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = encode_checkpoint(not_a_checkpoint.path(), out.path()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("config.json + safetensors"), "{msg}");
}

/// The artifact name is the checkpoint directory's stem — the same rule
/// `larql vindex3 encode` applies.
#[test]
fn encode_checkpoint_names_the_artifact_by_directory_stem() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let base = tempfile::tempdir().unwrap();
    let checkpoint = base.path().join("mini-glimmer");
    std::fs::create_dir(&checkpoint).unwrap();
    miniature_glimmer(&checkpoint);
    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(&checkpoint, out.path()).unwrap();
    assert_eq!(encoded.artifact, "mini-glimmer");
}

/// An inadmissible plan refuses with the blocking findings ITEMISED —
/// `encode_system`'s own gate discards them and points at `vindex3
/// plan`; the shared pipeline must not make two surfaces do that dance.
#[test]
fn encode_checkpoint_renders_blocking_findings_into_the_refusal() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;

    // A drafter checkpoint alone is inadmissible: its producer
    // interface cannot resolve without the target artifact.
    let checkpoint = tempfile::tempdir().unwrap();
    drafter_shaped(checkpoint.path());

    let out = tempfile::tempdir().unwrap();
    let err = encode_checkpoint(checkpoint.path(), out.path()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("blocking finding"), "{msg}");
    assert!(
        msg.lines().count() > 1,
        "findings must be itemised, not counted: {msg}"
    );
}

/// **Every segment the encoder writes starts its payload aligned.**
///
/// A misaligned payload puts every tensor in the segment at a
/// misaligned address, which a zero-copy backend cannot bind — and on
/// Metal binds anyway as garbage, with a command buffer that reports
/// success. Checked across many tensor-table sizes, because the header
/// is JSON and its length is whatever the table happens to serialise to.
#[test]
fn every_segment_starts_its_payload_on_the_declared_alignment() {
    use crate::format::vindex3::encode::segment::{
        realign_segment, write_segment, SEGMENT_PAYLOAD_ALIGN,
    };

    use crate::format::vindex3::encode::segment::PlannedTensor;

    let dir = std::env::temp_dir().join(format!("larql-align-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    for tensors in 1..24usize {
        // Names of VARYING length, because the header is JSON and its
        // byte count — hence the payload's start — follows the table's
        // spelling, not the tensor count.
        let planned: Vec<PlannedTensor> = (0..tensors)
            .map(|i| {
                let name = format!("t{i}{}", "x".repeat(i % 7));
                PlannedTensor {
                    relative_name: name.clone(),
                    source_name: name,
                    dtype: "BF16".into(),
                    shape: vec![16],
                    len: 32,
                }
            })
            .collect();
        let path = dir.join(format!("seg_{tensors}.bin"));
        write_segment(&path, "test@BF16", planned, |_name, w, hash| {
            let bytes = [7u8; 32];
            w.write_all(&bytes).expect("payload");
            hash(&bytes);
            Ok(32)
        })
        .expect("write");
        let (_, payload_start) =
            crate::format::vindex3::encode::segment::read_segment_header(&path).expect("header");
        assert!(
            payload_start.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64),
            "{tensors} tensors put the payload at {payload_start}, not a multiple of \
             {SEGMENT_PAYLOAD_ALIGN}"
        );
        // Realigning an already-aligned segment is a no-op in place.
        let (before, after) = realign_segment(&path, &path).expect("realign");
        assert_eq!((before, after), (payload_start, payload_start));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// **Realigning a legacy misaligned segment preserves its payload
/// exactly** — the operation that makes an already-encoded container
/// bindable without re-encoding it.
///
/// Built by writing a conforming segment and then re-writing it with a
/// header one byte SHORTER, which is precisely the shape a container
/// encoded before the padding existed has: a valid tensor table over a
/// payload that begins at an odd byte.
#[test]
fn realigning_a_misaligned_segment_preserves_the_payload_byte_for_byte() {
    use crate::format::vindex3::encode::segment::{
        read_segment_header, realign_segment, write_segment, PlannedTensor, SEGMENT_PAYLOAD_ALIGN,
    };
    use std::io::{Read, Write};

    let dir = std::env::temp_dir().join(format!("larql-realign-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let aligned = dir.join("aligned.bin");
    let payload: Vec<u8> = (0..96u16).map(|i| (i % 251) as u8).collect();
    write_segment(
        &aligned,
        "test@BF16",
        vec![PlannedTensor {
            relative_name: "w".into(),
            source_name: "w".into(),
            dtype: "BF16".into(),
            shape: vec![48],
            len: 96,
        }],
        |_n, w, hash| {
            w.write_all(&payload).expect("payload");
            hash(&payload);
            Ok(96)
        },
    )
    .expect("write");

    // Re-cut it with a one-byte-shorter header: same table, payload now
    // at an odd offset. This is a container encoded before the encoder
    // padded anything.
    let mut raw = Vec::new();
    std::fs::File::open(&aligned)
        .expect("open")
        .read_to_end(&mut raw)
        .expect("read");
    let header_len = u64::from_le_bytes(raw[..8].try_into().expect("len")) as usize;
    let header = &raw[8..8 + header_len];
    assert_eq!(
        header[header_len - 1],
        b' ',
        "the padding is trailing space"
    );
    let legacy = dir.join("legacy.bin");
    {
        let mut f = std::fs::File::create(&legacy).expect("create");
        f.write_all(&((header_len - 1) as u64).to_le_bytes())
            .expect("len");
        f.write_all(&header[..header_len - 1]).expect("header");
        f.write_all(&raw[8 + header_len..]).expect("payload");
    }
    let (_, legacy_start) = read_segment_header(&legacy).expect("legacy header parses");
    assert!(
        !legacy_start.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64),
        "the fixture must actually be misaligned (starts at {legacy_start})"
    );

    let fixed = dir.join("fixed.bin");
    let (before, after) = realign_segment(&legacy, &fixed).expect("realign");
    assert_eq!(before, legacy_start);
    assert!(after.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64));

    // The tensor table survives, and the payload is byte-identical.
    let (h_before, _) = read_segment_header(&legacy).expect("before");
    let (h_after, start_after) = read_segment_header(&fixed).expect("after");
    assert_eq!(h_before.tensors, h_after.tensors, "the table is unchanged");
    assert_eq!(h_before.representation, h_after.representation);
    assert_eq!(start_after, after);
    let mut got = Vec::new();
    let mut f = std::fs::File::open(&fixed).expect("open fixed");
    std::io::Read::read_to_end(&mut f, &mut got).expect("read fixed");
    assert_eq!(
        &got[start_after as usize..],
        &payload[..],
        "the payload bytes must be copied verbatim"
    );
    // Realigning an ALREADY-aligned segment to a new path still copies.
    let again = dir.join("again.bin");
    let (b2, a2) = realign_segment(&fixed, &again).expect("realign aligned");
    assert_eq!((b2, a2), (after, after));
    std::fs::remove_dir_all(&dir).ok();
}

/// A truncated or absurd header is refused by name rather than read.
#[test]
fn a_corrupt_segment_header_is_refused() {
    use crate::format::vindex3::encode::segment::read_segment_header;
    use std::io::Write;

    let dir = std::env::temp_dir().join(format!("larql-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");

    let absurd = dir.join("absurd.bin");
    std::fs::File::create(&absurd)
        .expect("create")
        .write_all(&u64::MAX.to_le_bytes())
        .expect("len");
    let err = read_segment_header(&absurd).expect_err("an absurd header length must be refused");
    assert!(format!("{err}").contains("corrupt"), "{err}");

    let garbage = dir.join("garbage.bin");
    {
        let body = b"not json at all";
        let mut f = std::fs::File::create(&garbage).expect("create");
        f.write_all(&(body.len() as u64).to_le_bytes())
            .expect("len");
        f.write_all(body).expect("body");
    }
    assert!(
        read_segment_header(&garbage).is_err(),
        "a non-JSON header must be refused"
    );
    std::fs::remove_dir_all(&dir).ok();
}
