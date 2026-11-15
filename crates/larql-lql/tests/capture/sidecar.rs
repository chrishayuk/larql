//! The sidecar describes the file beside it and the sequence it was captured over.

use crate::support::*;

#[test]
fn the_sidecar_names_the_model_the_sequence_and_the_file() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("residual.npy");
    let reply = capture(
        &mut s,
        &format!("residual AT LAYERS (0, 1) FOR \"{PROMPT}\" CONTINUE \"{CONTINUATION}\" POSITIONS REPLY"),
        &out,
    );
    assert!(has(&reply, "Captured residual [2, 1, 64]"), "{reply:?}");
    let car = sidecar(&out);
    let npy = read_npy(&out);
    assert_eq!(car["schema"], "larql-capture/v1");
    assert_eq!(
        car["statement"],
        format!(
            "CAPTURE residual AT LAYERS (0, 1) FOR \"{PROMPT}\" CONTINUE \"{CONTINUATION}\" POSITIONS REPLY TO \"{}\";",
            q(&out)
        )
    );
    assert_eq!(car["npy_sha256"], sha256_hex(&npy.bytes));
    assert_eq!(car["shape"], serde_json::json!([2, 1, HIDDEN]));
    assert_eq!(car["dtype"], "<f4");
    assert_eq!(car["site"], "residual");
    assert_eq!(car["layers"], serde_json::json!([0, 1]));
    let index = std::fs::read(fx.container.join("index.json")).unwrap();
    assert_eq!(car["index_sha256"], sha256_hex(&index));
    let model: serde_json::Value = serde_json::from_slice(&index).unwrap();
    assert_eq!(car["model"], model["model"]);
    assert_eq!(
        car["container"],
        std::fs::canonicalize(&fx.container)
            .unwrap()
            .display()
            .to_string()
    );
    assert_eq!(car["larql_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(car["executes"], "stored");
    assert_eq!(
        car["execution"]["binding"]["representation_source"],
        "stored"
    );

    let sequence = &car["sequence"];
    assert_eq!(sequence["prompt_tokens"], PROMPT_TOKENS);
    let ids = sequence["token_ids"].as_array().unwrap();
    assert_eq!(ids.len(), PROMPT_TOKENS + 1);
    assert_eq!(ids[PROMPT_TOKENS], PARIS_ID);
    assert_eq!(sequence["continuation"]["kind"], "text");
    assert_eq!(
        sequence["continuation"]["token_ids"],
        serde_json::json!([PARIS_ID])
    );

    let position = &car["positions"][0];
    assert_eq!(position["position"], PROMPT_TOKENS - 1);
    assert_eq!(position["part"], "prompt");
    assert_eq!(position["token"], "is");
    assert_eq!(position["predicts_token_id"], PARIS_ID);
}

#[test]
fn the_sidecar_of_a_site_after_the_last_layer_has_no_layers() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let out = fx.path("final.npy");
    capture(&mut s, &format!("final FOR \"{PROMPT}\""), &out);
    let car = sidecar(&out);
    assert_eq!(car["layers"], serde_json::Value::Null);
    assert_eq!(
        car["positions"][0]["predicts_token_id"],
        serde_json::Value::Null
    );
    assert_eq!(car["execution"]["cross_check"], serde_json::Value::Null);
}

/// The longest file name the filesystems the suite runs on accept (`NAME_MAX` on APFS, ext4 and tmpfs).
#[cfg(unix)]
const NAME_MAX: usize = 255;

#[cfg(unix)]
#[test]
fn a_sidecar_that_cannot_be_written_takes_the_written_array_with_it() {
    let fx = Fixture::new();
    let mut s = fx.session();
    let before = std::fs::read_dir(fx.dir.path()).unwrap().count();
    // The array's name is as long as a name may be, so the sidecar's, five bytes longer, is refused.
    let stem = "a".repeat(NAME_MAX - ".npy".len());
    let out = fx.path(&format!("{stem}.npy"));
    let err = try_capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &out).unwrap_err();
    let sidecar_path = format!("{}.json", out.display());
    assert!(err.contains(&format!("write {sidecar_path}")), "{err}");
    assert!(
        !err.contains("could not remove"),
        "the array was removable: {err}"
    );
    assert!(
        !out.exists(),
        "the array written before the sidecar must be removed"
    );
    let after = std::fs::read_dir(fx.dir.path()).unwrap().count();
    assert_eq!(after, before, "the failed capture left something behind");
}
