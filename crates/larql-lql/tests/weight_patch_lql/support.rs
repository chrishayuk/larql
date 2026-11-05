//! Shared helpers: a source container, a session, and ways to look at results.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;

/// Layer 1's down projection, `[64, 256]`.
pub const TENSOR: &str = "1.mlp.down_proj.weight";
pub const ROW: usize = 5;
pub const DELTA: &str = "0.0625";
pub const CANONICAL: &str = "target.decoder_stack@BF16";
pub const DERIVED: &str = "target.decoder_stack@Q8_0";
pub const COLS: [usize; 4] = [32, 33, 34, 40];

pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub container: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = bf16_container_with_q8_pack(dir.path());
        Self { dir, container }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

pub fn q(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

pub fn session_on(container: &Path) -> Session {
    let mut s = Session::new();
    run(&mut s, &format!("USE \"{}\";", q(container)));
    s
}

pub fn run(s: &mut Session, stmt: &str) -> Vec<String> {
    try_run(s, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

pub fn try_run(s: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse: {e}"))?;
    s.execute(&parsed).map_err(|e| e.to_string())
}

pub fn update(cols: &[usize], delta: &str) -> String {
    let list = cols
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("UPDATE WEIGHTS SET DELTA = {delta} WHERE TENSOR = \"{TENSOR}\" AND ROW = {ROW} AND COL IN ({list});")
}

/// Representation id -> (payload digest, segment digest).
pub fn digests(container: &Path) -> BTreeMap<String, (String, String)> {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(container.join("index.json")).unwrap())
            .unwrap();
    index["representations"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, r)| {
            (
                id.clone(),
                (
                    r["payload_sha256"].as_str().unwrap().into(),
                    r["segment_sha256"].as_str().unwrap().into(),
                ),
            )
        })
        .collect()
}

/// A cheap content fingerprint of a whole directory tree (equality only).
pub fn fingerprint(root: &Path) -> Vec<(String, u64)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                walk(root, &e.path(), out);
            } else {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                std::fs::read(e.path()).unwrap().hash(&mut h);
                out.push((
                    e.path()
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                    h.finish(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

pub fn ledger(container: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(container.join("weight_patches.json")).unwrap())
        .unwrap()
}

/// Write a `.vlp` carrying `cols` of the standard edit, from a fresh session.
pub fn save_patch(fx: &Fixture, name: &str, cols: &[usize], delta: &str) -> PathBuf {
    let vlp = fx.path(name);
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(&mut s, &update(cols, delta));
    run(&mut s, "SAVE PATCH;");
    vlp
}

pub fn is_coherent(container: &Path) -> bool {
    larql_vindex::format::vindex3::inspect::inspect_container(container, true)
        .unwrap()
        .is_coherent()
}

/// The raw bytes of one tensor in one representation's segment.
pub fn tensor_bytes(container: &Path, representation: &str, tensor: &str) -> Vec<u8> {
    use larql_vindex::format::vindex3::encode::segment::read_segment_header;
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(container.join("index.json")).unwrap())
            .unwrap();
    let segment = container.join(
        index["representations"][representation]["segment"]
            .as_str()
            .unwrap(),
    );
    let (header, start) = read_segment_header(&segment).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    let raw = std::fs::read(&segment).unwrap();
    raw[(start + t.offset) as usize..(start + t.offset + t.len) as usize].to_vec()
}

/// How many 16-bit elements differ between two tensors' bytes.
pub fn elements_differing(a: &[u8], b: &[u8]) -> usize {
    (0..a.len() / 2)
        .filter(|&i| a[2 * i..2 * i + 2] != b[2 * i..2 * i + 2])
        .count()
}

/// Rewrite a `.vlp` through `change`, to a new file.
pub fn mutated(
    fx: &Fixture,
    vlp: &Path,
    name: &str,
    change: impl FnOnce(&mut serde_json::Value),
) -> PathBuf {
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vlp).unwrap()).unwrap();
    change(&mut v);
    let out = fx.path(name);
    std::fs::write(&out, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    out
}

pub fn pending_line(s: &mut larql_lql::Session) -> Option<String> {
    run(s, "SHOW PATCHES;")
        .into_iter()
        .find(|l| l.contains("Stored-weight edits"))
}
