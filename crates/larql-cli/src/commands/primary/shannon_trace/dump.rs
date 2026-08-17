//! Write side of the Gate-B layer diff: one forward pass, one plane per layer.
//!
//! Plane `000` is the embedding output; plane `i + 1` is the residual after
//! layer `i`. That is exactly `forward_hidden_all_layers`' capture list, and
//! the reference side must hook the same points — see
//! `scripts/dump_layers_hf.py`.

use std::fs;
use std::io::Write;
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::LayerDumpArgs;
use crate::commands::primary::shannon_cmd;

/// Manifest filename inside a dump directory.
pub const MANIFEST_NAME: &str = "layer_dump.json";

/// Engine tag this side writes. The reference scorer writes its own.
pub const ENGINE_LARQL_F32: &str = "larql-rust-f32";

/// Plane encoding: little-endian `f32[seq_len * hidden_size]`, row-major.
/// Recorded in the manifest so a future dtype cannot be inferred wrongly by a
/// reader that only counts bytes.
pub const PLANE_DTYPE: &str = "f32-le";

/// What a dump directory contains, and the fixture it was taken over.
///
/// `token_ids` is the load-bearing field. The reference side reads it instead
/// of tokenising the corpus itself, so the two engines cannot silently score
/// different token sequences — [`docs/k3-funnel.md`] §4.7.7 recorded a whole
/// verdict withdrawn to that class of mismatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerDumpManifest {
    /// Which forward produced these planes.
    pub engine: String,
    /// Model path or HF id, as given on the command line.
    pub model: String,
    /// Transformer blocks. `planes.len() == num_layers + 1`.
    pub num_layers: usize,
    /// Rows in every plane.
    pub seq_len: usize,
    /// Columns in every plane.
    pub hidden_size: usize,
    /// The exact token window the pass ran over.
    pub token_ids: Vec<u32>,
    /// Plane filenames, in capture order.
    pub planes: Vec<String>,
    /// Encoding of every plane file.
    pub dtype: String,
}

/// Bytes one `[seq_len, hidden_size]` plane file must be. The single
/// definition of the geometry→size rule: `read_plane` validates against it,
/// so a manifest and the file it describes cannot disagree about it.
pub fn plane_bytes(seq_len: usize, hidden_size: usize) -> usize {
    seq_len * hidden_size * std::mem::size_of::<f32>()
}

/// Plane filename for capture `idx`. Zero-padded so a directory listing sorts
/// into capture order.
pub fn plane_name(idx: usize) -> String {
    format!("layer_{idx:03}.f32")
}

/// Write one `[seq, hidden]` plane as little-endian f32.
pub fn write_plane(path: &Path, plane: &Array2<f32>) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(fs::File::create(path)?);
    for v in plane.iter() {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

/// Read one plane back, validating its length against the declared geometry.
pub fn read_plane(
    path: &Path,
    seq_len: usize,
    hidden_size: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let expected = plane_bytes(seq_len, hidden_size);
    if bytes.len() != expected {
        return Err(format!(
            "{}: {} bytes, expected {expected} for [{seq_len}, {hidden_size}] {PLANE_DTYPE}",
            path.display(),
            bytes.len(),
        )
        .into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Array2::from_shape_vec((seq_len, hidden_size), values)?)
}

/// Load a dump directory's manifest.
pub fn read_manifest(dir: &Path) -> Result<LayerDumpManifest, Box<dyn std::error::Error>> {
    let path = dir.join(MANIFEST_NAME);
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("{}: {e} (is this a layer-dump directory?)", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn run_layer_dump(args: LayerDumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = shannon_cmd::load_model(&args.model)?;
    let mut ids = match (&args.tokens, &args.corpus) {
        (Some(tokens), _) => tokens
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<u32>, _>>()
            .map_err(|e| format!("--tokens: {e}"))?,
        (None, Some(corpus)) => {
            let text = shannon_cmd::read_text(corpus, args.bytes)?;
            larql_inference::encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?
        }
        (None, None) => return Err("one of --corpus or --tokens is required".into()),
    };
    if ids.is_empty() {
        return Err("token window is empty".into());
    }
    ids.truncate(args.context);

    let weights = model.weights();
    eprintln!(
        "dumping {} layers over {} tokens...",
        weights.num_layers,
        ids.len(),
    );
    let captures = shannon_cmd::forward_hidden_all_layers(weights, &ids)?;

    fs::create_dir_all(&args.out)?;
    let mut planes = Vec::with_capacity(captures.len());
    for (idx, plane) in captures.iter().enumerate() {
        let name = plane_name(idx);
        write_plane(&args.out.join(&name), plane)?;
        planes.push(name);
    }

    let hidden_size = captures[0].shape()[1];
    let manifest = LayerDumpManifest {
        engine: ENGINE_LARQL_F32.to_string(),
        model: args.model.clone(),
        num_layers: weights.num_layers,
        seq_len: ids.len(),
        hidden_size,
        token_ids: ids,
        planes,
        dtype: PLANE_DTYPE.to_string(),
    };
    fs::write(
        args.out.join(MANIFEST_NAME),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    println!(
        "wrote {} planes of [{}, {}] to {}",
        manifest.planes.len(),
        manifest.seq_len,
        manifest.hidden_size,
        args.out.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_for(seq_len: usize, hidden_size: usize) -> LayerDumpManifest {
        LayerDumpManifest {
            engine: ENGINE_LARQL_F32.to_string(),
            model: "test/model".to_string(),
            num_layers: 1,
            seq_len,
            hidden_size,
            token_ids: vec![1, 2],
            planes: vec![plane_name(0), plane_name(1)],
            dtype: PLANE_DTYPE.to_string(),
        }
    }

    /// Zero padding is what makes a lexical directory listing agree with
    /// capture order; without it `layer_10` sorts before `layer_2`.
    #[test]
    fn plane_names_sort_into_capture_order() {
        let mut names: Vec<String> = [10, 2, 0, 100].iter().map(|i| plane_name(*i)).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                plane_name(0),
                plane_name(2),
                plane_name(10),
                plane_name(100)
            ]
        );
    }

    #[test]
    fn plane_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(plane_name(0));
        let plane =
            Array2::from_shape_vec((2, 3), vec![1.0f32, -2.5, 3.25, 0.0, 1e-8, -1e8]).unwrap();
        write_plane(&path, &plane).unwrap();
        assert_eq!(read_plane(&path, 2, 3).unwrap(), plane);
    }

    /// A plane whose byte count disagrees with the manifest is the signature
    /// of two engines dumping different geometries — it must be an error, not
    /// a reshape.
    #[test]
    fn plane_with_wrong_length_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(plane_name(0));
        write_plane(
            &path,
            &Array2::from_shape_vec((2, 3), vec![0.0f32; 6]).unwrap(),
        )
        .unwrap();
        let err = read_plane(&path, 2, 4).unwrap_err().to_string();
        assert!(err.contains("expected"), "unhelpful error: {err}");
    }

    #[test]
    fn plane_bytes_follows_declared_geometry() {
        let m = manifest_for(2, 3);
        assert_eq!(plane_bytes(m.seq_len, m.hidden_size), 24);
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = manifest_for(2, 3);
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(MANIFEST_NAME),
            serde_json::to_string_pretty(&m).unwrap(),
        )
        .unwrap();
        assert_eq!(read_manifest(dir.path()).unwrap(), m);
    }

    /// The error has to say what kind of directory was expected — this fires
    /// whenever someone points `layer-diff` at a vindex or a model dir.
    #[test]
    fn missing_manifest_names_what_was_expected() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_manifest(dir.path()).unwrap_err().to_string();
        assert!(err.contains("layer-dump directory"), "unhelpful: {err}");
    }
}
