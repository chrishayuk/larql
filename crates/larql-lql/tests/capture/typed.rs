//! The typed operations (`larql_lql::operations`) equal the statements they sit behind: a capture's and an
//! export's values and provenance equal the file and sidecar the statement writes, the encoders are the ones
//! CAPTURE records, predictions and generations are what INFER prints, operands handed over in memory stage the
//! op an operand file stages, and the patch lifecycle reports what its statements print.

use larql_lql::operations::{
    operand_bits_from_values, operand_origin, CaptureContinuation, CaptureLayers, CapturePositions,
    CaptureSite, CaptureSpec, NpyDescr, OperandBits, OperandValues,
};

use crate::support::*;

const F64_PROFILE: &str = "f64-seq-rne1/v1";
const F32_PROFILE: &str = "f32-seq-rne1/v1";
const COLUMNS: [usize; 3] = [8, 9, 33];

/// The sidecar with its `statement` removed, and the provenance with its null `statement` removed.
fn without_statement(mut record: serde_json::Value, expect_null: bool) -> serde_json::Value {
    let statement = record.as_object_mut().unwrap().remove("statement").unwrap();
    assert_eq!(statement.is_null(), expect_null, "{statement}");
    record
}

/// A capture as typed arguments, and the same capture as statement clauses.
type Case = (
    CaptureSite,
    Option<CaptureLayers>,
    Option<CaptureContinuation>,
    CapturePositions,
    String,
);

#[test]
fn capture_equals_the_statement_file_and_sidecar() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let cases: [Case; 4] = [
        (CaptureSite::Logits, None, None, CapturePositions::Last, format!("logits FOR \"{PROMPT}\"")),
        (
            CaptureSite::Residual,
            Some(CaptureLayers::Many(vec![1, 0])),
            Some(CaptureContinuation::Text(CONTINUATION.into())),
            CapturePositions::All,
            format!("residual AT LAYERS (1, 0) FOR \"{PROMPT}\" CONTINUE \"{CONTINUATION}\" POSITIONS ALL"),
        ),
        (
            CaptureSite::FfnDownOutput,
            Some(CaptureLayers::One(1)),
            Some(CaptureContinuation::Generate(2)),
            CapturePositions::Reply,
            format!("ffn.down_output AT LAYER 1 FOR \"{PROMPT}\" GENERATE 2 POSITIONS REPLY"),
        ),
        (
            CaptureSite::Final,
            None,
            Some(CaptureContinuation::Tokens(vec![3, 7])),
            CapturePositions::List(vec![0, 6]),
            format!("final FOR \"{PROMPT}\" CONTINUE TOKENS (3, 7) POSITIONS (0, 6)"),
        ),
    ];
    for (i, (site, layers, continuation, positions, clauses)) in cases.iter().enumerate() {
        let out = fx.path(&format!("c{i}.npy"));
        capture(&mut s, clauses, &out);
        let written = read_npy(&out);
        let captured = s
            .capture(&CaptureSpec {
                site: *site,
                layers: layers.as_ref(),
                prompt: PROMPT,
                continuation: continuation.as_ref(),
                positions,
            })
            .unwrap();
        assert_eq!(captured.shape, written.shape, "{clauses}");
        assert_eq!(bits(&captured.values()), bits(&written.values), "{clauses}");
        assert_eq!(
            without_statement(captured.provenance.clone(), true),
            without_statement(sidecar(&out), false),
            "{clauses}"
        );
        assert_eq!(
            captured.provenance["npy_sha256"],
            sha256_hex(&written.bytes)
        );
    }
}

#[test]
fn read_weight_equals_export_weight() {
    let fx = Fixture::new();
    let mut s = fx.session();
    for (i, (rows, columns, bits_flag)) in [
        (None, None, false),
        (Some(vec![1, 3]), Some(vec![0, 5]), false),
        (Some(vec![4]), None, true),
    ]
    .into_iter()
    .enumerate()
    {
        let out = fx.path(&format!("w{i}.npy"));
        let mut stmt = format!("EXPORT WEIGHT \"{TENSOR}\"");
        if let Some(r) = &rows {
            stmt.push_str(&format!(" ROWS ({})", list(r)));
        }
        if let Some(c) = &columns {
            stmt.push_str(&format!(" COLUMNS ({})", list(c)));
        }
        stmt.push_str(&format!(
            " TO \"{}\"{};",
            q(&out),
            if bits_flag { " AS BITS" } else { "" }
        ));
        run(&mut s, &stmt);
        let file = std::fs::read(&out).unwrap();
        let read = s
            .read_weight(TENSOR, rows.as_deref(), columns.as_deref(), bits_flag)
            .unwrap();
        assert_eq!(
            read.descr,
            if bits_flag {
                NpyDescr::U2
            } else {
                NpyDescr::F4
            }
        );
        assert!(file.ends_with(&read.bytes), "{stmt}");
        assert_eq!(read.provenance["npy_sha256"], sha256_hex(&file));
        assert_eq!(
            without_statement(read.provenance, true),
            without_statement(sidecar(&out), false)
        );
    }
}

fn list(items: &[usize]) -> String {
    items
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn tensors_are_what_show_tensors_lists() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let listed = s.tensors(None).unwrap();
    let lines = run(&mut s, "SHOW TENSORS;");
    assert_eq!(lines.len(), listed.len() + 3);
    for (line, t) in lines[2..].iter().zip(&listed) {
        assert!(line.starts_with(&t.qualified_name()), "{line}");
    }
    let filtered = s.tensors(Some("decoder_stack")).unwrap();
    assert!(!filtered.is_empty() && filtered.len() < listed.len());
}

#[test]
fn the_encoders_are_the_ones_capture_records() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("c.npy");
    capture(
        &mut s,
        &format!("logits FOR \"{PROMPT}\" CONTINUE \"{CONTINUATION}\" POSITIONS ALL"),
        &out,
    );
    let record = sidecar(&out);
    let ids: Vec<u32> = serde_json::from_value(record["sequence"]["token_ids"].clone()).unwrap();
    let prompt = s.tokenize(PROMPT, true).unwrap();
    assert_eq!(prompt.len(), PROMPT_TOKENS);
    assert_eq!(&ids[..PROMPT_TOKENS], &prompt[..]);
    assert_eq!(
        &ids[PROMPT_TOKENS..],
        &s.tokenize(CONTINUATION, false).unwrap()[..]
    );
    assert_eq!(ids[PROMPT_TOKENS], PARIS_ID);
    for position in record["positions"].as_array().unwrap() {
        let id = position["token_id"].as_u64().unwrap() as u32;
        assert_eq!(
            s.decode(&[id]).unwrap(),
            position["token"].as_str().unwrap()
        );
    }
}

#[test]
fn predict_and_generate_are_what_infer_prints() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let predicted = s.predict(PROMPT, 3).unwrap();
    let printed = run(&mut s, &format!("INFER \"{PROMPT}\" TOP 3;"));
    for (i, p) in predicted.top.iter().enumerate() {
        let line = format!(
            "  {:2}. {:20} ({:.2}%)  [id {}]",
            i + 1,
            p.token,
            p.probability * 100.0,
            p.token_id
        );
        assert_eq!(printed[i + 1], line);
    }
    let logits = infer_logits(&fx.container, PROMPT);
    let top = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    assert_eq!(predicted.top[0].token_id, top);

    let generated = s.generate(PROMPT, 3).unwrap();
    let printed = run(&mut s, &format!("INFER \"{PROMPT}\" GENERATE 3;"));
    let ids: Vec<String> = generated.token_ids.iter().map(u32::to_string).collect();
    assert_eq!(printed[1], format!("  ids:  {}", ids.join(",")));
    assert_eq!(printed[2], format!("  text: {}", generated.text));
}

#[test]
fn operands_from_values_are_the_bits_an_operand_file_holds() {
    let ones = [1.0f64, 0.125];
    assert_eq!(
        operand_bits_from_values(F64_PROFILE, OperandValues::F64(&ones), "a").unwrap(),
        OperandBits::F64(vec![0x3ff0_0000_0000_0000, 0x3fc0_0000_0000_0000])
    );
    assert_eq!(
        operand_bits_from_values(F64_PROFILE, OperandValues::F32(&[1.0, 0.125]), "a").unwrap(),
        OperandBits::F64(vec![0x3ff0_0000_0000_0000, 0x3fc0_0000_0000_0000])
    );
    assert_eq!(
        operand_bits_from_values(F32_PROFILE, OperandValues::F32(&[1.0]), "a").unwrap(),
        OperandBits::F32(vec![0x3f80_0000])
    );
    let narrowing = operand_bits_from_values(F32_PROFILE, OperandValues::F64(&ones), "a");
    assert!(narrowing.unwrap_err().to_string().contains("narrowing"));
}

/// Stage a low-rank op from bits in a session recording to `vlp`, and save it.
fn staged_low_rank_vlp(fx: &Fixture, vlp: &std::path::Path, scale: f64) -> String {
    let mut s = fx.session();
    let rows = s.tensors(None).unwrap();
    let rows = rows
        .iter()
        .find(|t| t.tensor == TENSOR && t.canonical)
        .unwrap()
        .shape[0];
    let a: Vec<f64> = (0..rows)
        .map(|r| if r == 2 { 0.0 } else { scale })
        .collect();
    let b = vec![1.0f64; COLUMNS.len()];
    let a_bits = operand_bits_from_values(F64_PROFILE, OperandValues::F64(&a), "a").unwrap();
    let b_bits = operand_bits_from_values(F64_PROFILE, OperandValues::F64(&b), "b").unwrap();
    let origin = operand_origin("in memory", &a_bits, [rows, 1], &b_bits, [1, COLUMNS.len()]);
    s.begin_patch(&q(vlp)).unwrap();
    let staged = s
        .stage_low_rank(
            TENSOR,
            &COLUMNS,
            F64_PROFILE,
            1,
            &a_bits,
            &b_bits,
            Some(origin.clone()),
        )
        .unwrap();
    assert!(staged.changed_elements > 0 && !staged.sham);
    assert_eq!(staged.op.origin.as_deref(), Some(origin.as_str()));
    let saved = s.save_patch().unwrap();
    assert_eq!((saved.operations, saved.weight_edits), (1, 1));
    origin
}

#[test]
fn staged_low_rank_saves_the_vlp_the_operand_file_saves() {
    let fx = Fixture::new();
    let from_memory = fx.path("memory.vlp");
    let origin = staged_low_rank_vlp(&fx, &from_memory, 0.125);
    assert!(origin.starts_with("in memory: a ") && origin.contains(" f64 sha256 "));

    let mut s = fx.session();
    let rows = s.tensors(None).unwrap();
    let rows = rows
        .iter()
        .find(|t| t.tensor == TENSOR && t.canonical)
        .unwrap()
        .shape[0];
    let hex = |v: f64| format!("{:016x}", v.to_bits());
    let doc = serde_json::json!({
        "rank": 1,
        "a": (0..rows).map(|r| hex(if r == 2 { 0.0 } else { 0.125 })).collect::<Vec<_>>(),
        "b": vec![hex(1.0); COLUMNS.len()],
        "origin": origin,
    });
    let operands = fx.path("operands.json");
    std::fs::write(&operands, doc.to_string()).unwrap();
    let from_file = fx.path("file.vlp");
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&from_file)));
    run(
        &mut s,
        &format!(
            "ADD LOW RANK \"{}\" TO \"{TENSOR}\" COLUMNS (8, 9, 33) NUMERICS \"{F64_PROFILE}\";",
            q(&operands)
        ),
    );
    run(&mut s, "SAVE PATCH;");
    assert_eq!(
        std::fs::read(&from_memory).unwrap(),
        std::fs::read(&from_file).unwrap()
    );
}

#[test]
fn the_patch_lifecycle_reports_what_its_statements_print() {
    let fx = Fixture::new();
    let vlp = fx.path("edit.vlp");
    staged_low_rank_vlp(&fx, &vlp, 4.0);
    let spec_positions = CapturePositions::All;
    let spec = CaptureSpec {
        site: CaptureSite::Logits,
        layers: None,
        prompt: PROMPT,
        continuation: None,
        positions: &spec_positions,
    };

    let mut s = fx.session();
    let base = s.capture(&spec).unwrap().values();
    let applied = s.apply_patch(&q(&vlp)).unwrap();
    assert_eq!(applied.sha256, sha256_hex(&std::fs::read(&vlp).unwrap()));
    assert_eq!((applied.operations, applied.structured_ops), (1, 1));
    let edited = s.capture(&spec).unwrap();
    assert_ne!(bits(&edited.values()), bits(&base));
    assert_eq!(edited.provenance["executes"], "patched");

    let report = s.patches().unwrap();
    assert_eq!(report.applied.len(), 1);
    assert_eq!(
        report.applied[0].sha256.as_deref(),
        Some(applied.sha256.as_str())
    );
    assert_eq!(report.weight_ops, 1);
    let shown = run(&mut s, "SHOW PATCHES;");
    let p = &report.applied[0];
    assert_eq!(
        shown[0],
        format!(
            "  1. {:<40} {} ops ({} ins, {} upd, {} del)",
            p.name.as_deref().unwrap(),
            p.operations,
            p.inserts,
            p.updates,
            p.deletes
        )
    );

    let removed = s.remove_patch(&q(&vlp)).unwrap();
    assert_eq!(removed.number, 1);
    assert!(removed.carried_weight_ops);
    assert_eq!(bits(&s.capture(&spec).unwrap().values()), bits(&base));
    assert!(s.patches().unwrap().applied.is_empty());
}

#[test]
fn staged_weight_delta_is_what_update_weights_prints() {
    let fx = Fixture::new();
    let mut typed = fx.session();
    let mut stated = fx.session();
    let staged = typed.stage_weight_delta(TENSOR, 5, &[0, 1], 2.0).unwrap();
    let printed = run(
        &mut stated,
        &format!(
            "UPDATE WEIGHTS SET DELTA = 2.0 WHERE TENSOR = \"{TENSOR}\" AND ROW = 5 AND COL IN (0, 1);"
        ),
    );
    assert_eq!(staged.elements.len(), 2);
    assert!(printed.contains(&format!("  op digest {}", &staged.digest[..16])));
    assert!(printed.contains(&format!(
        "  {} weight element(s) pending in this session",
        staged.pending_elements
    )));
    let bf16 = |b: u16| f32::from_bits(u32::from(b) << 16);
    for (e, line) in staged.elements.iter().zip(&printed[1..]) {
        assert_eq!(
            line,
            &format!(
                "  [5, {}]: {:.7} -> {:.7}",
                e.col,
                bf16(e.old_bits),
                bf16(e.new_bits)
            )
        );
    }
}

#[test]
fn the_typed_operations_refuse_without_a_binding() {
    let mut s = larql_lql::Session::new();
    assert!(s.tensors(None).is_err());
    assert!(s.read_weight(TENSOR, None, None, false).is_err());
    assert!(s.tokenize(PROMPT, true).is_err());
    assert!(s.decode(&[1]).is_err());
    assert!(s.predict(PROMPT, 1).is_err());
    assert!(s.generate(PROMPT, 1).is_err());
    assert!(s.patches().is_err());
    assert!(s.save_patch().is_err());
    assert!(s
        .stage_low_rank(
            TENSOR,
            &COLUMNS,
            F64_PROFILE,
            1,
            &OperandBits::F64(vec![]),
            &OperandBits::F64(vec![]),
            None
        )
        .is_err());
    assert!(s.stage_weight_delta(TENSOR, 0, &[0], 1.0).is_err());
}
