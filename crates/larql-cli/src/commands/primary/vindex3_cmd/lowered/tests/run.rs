//! `--bank` on lowered backends must refuse, not silently ignore the
//! manifest. Device-free: the check runs before any Metal session is
//! touched, so this covers the same code path a real run takes without
//! needing a GPU.

use std::path::Path;

use super::super::run::refuse_bank_on_lowered;

#[test]
fn no_bank_path_is_fine() {
    assert!(refuse_bank_on_lowered(None).is_ok());
}

#[test]
fn a_bank_path_is_refused_with_a_clear_message() {
    let err = refuse_bank_on_lowered(Some(Path::new("/tmp/some-manifest.jsonl")))
        .expect_err("--bank must be refused on lowered backends");
    let message = err.to_string();
    assert!(
        message.contains("--bank"),
        "error should name the offending flag: {message}"
    );
    assert!(
        message.contains("lowered"),
        "error should say why (lowered backends specifically): {message}"
    );
}
