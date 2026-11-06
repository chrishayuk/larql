//! The tokenizer a GGUF conversion reads from beside the GGUF file.

use larql_vindex::format::filenames::TOKENIZER_JSON;
use larql_vindex::tokenizers::Tokenizer;
use std::path::{Path, PathBuf};

/// Load `tokenizer.json` from the directory holding `gguf`.
///
/// A missing file and a file that fails to parse are different errors:
/// reporting a parse failure as "not found" sent users looking for a file
/// that was already there (#365). The error names the path either way.
pub(super) fn load_sibling_tokenizer(gguf: &Path) -> Result<Tokenizer, String> {
    let path = sibling_tokenizer_path(gguf);
    if !path.is_file() {
        return Err(format!(
            "{TOKENIZER_JSON} not found next to GGUF file (looked for {}). \
             Place it in the same directory.",
            path.display()
        ));
    }
    Tokenizer::from_file(&path).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

fn sibling_tokenizer_path(gguf: &Path) -> PathBuf {
    gguf.parent()
        .unwrap_or_else(|| Path::new(""))
        .join(TOKENIZER_JSON)
}
