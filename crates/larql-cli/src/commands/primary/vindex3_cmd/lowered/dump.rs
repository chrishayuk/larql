//! Teacher-force a token window through the lowering, capturing every
//! layer's output per position into planes a `shannon layer-diff` reads.

use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use super::LoweredSession;

/// Teacher-force `tokens` through the lowering, capturing every layer's
/// output per position, and write the planes + manifest a `shannon
/// layer-diff` consumes — byte-compatible with `vindex3 exec --dump-layers`
/// on the interpreter backends, so the two can be diffed directly.
pub(super) fn dump_lowered(
    session: &mut LoweredSession<'_>,
    tokens: &[u32],
    plan: &ComponentOpPlan,
    container: &std::path::Path,
    label: &str,
    dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use super::super::super::shannon_trace::dump::{
        plane_name, LayerDumpManifest, MANIFEST_NAME, PLANE_DTYPE,
    };
    let hidden = session.hidden;
    let num_layers = plan.layers.len();
    let seq = tokens.len();
    std::fs::create_dir_all(dir)?;
    // Row-major [seq, hidden] planes: plane 0 = embeddings, plane i+1 =
    // layer i's post-FFN-residual hidden.
    let mut planes: Vec<Vec<f32>> = vec![Vec::with_capacity(seq * hidden); num_layers + 1];
    for (pos, &token) in tokens.iter().enumerate() {
        let (_logits, embedding, layers_out) = session.step_capturing(token)?;
        planes[0].extend_from_slice(&embedding);
        for (i, row) in layers_out.into_iter().enumerate() {
            planes[i + 1].extend_from_slice(&row);
        }
        eprintln!("captured position {}/{seq}", pos + 1);
    }
    let plane_names: Vec<String> = (0..=num_layers).map(plane_name).collect();
    for (name, data) in plane_names.iter().zip(&planes) {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join(name))?);
        for v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
    }
    let manifest = LayerDumpManifest {
        engine: format!("vindex3-metal-lowered-{label}"),
        model: container.display().to_string(),
        num_layers,
        seq_len: seq,
        hidden_size: hidden,
        token_ids: tokens.to_vec(),
        planes: plane_names,
        dtype: PLANE_DTYPE.to_string(),
    };
    std::fs::write(
        dir.join(MANIFEST_NAME),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    eprintln!(
        "wrote {} planes + manifest to {}",
        num_layers + 1,
        dir.display()
    );
    Ok(())
}
