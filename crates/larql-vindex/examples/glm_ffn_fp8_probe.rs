//! Run GLM-5.3-Flash's dense FFN through LARQL's fine-grained FP8 kernel,
//! over the CHECKPOINT's own bytes, and dump each projection's output.
//!
//! The point is the arithmetic, not the plumbing: it reads the
//! safetensors shard directly and drives
//! [`FusedFp8Block`](larql_vindex::format::vindex3::opplan::exec::cpu::kernels)
//! with no dequantised copy of any weight. `scripts/glm_ffn_fp8_gate.py`
//! compares the result against the same layer's boundaries as produced by
//! the pinned upstream reference.
//!
//! ```text
//! cargo run --release -p larql-vindex --example glm_ffn_fp8_probe -- \
//!     <shard.safetensors> <layer-prefix> <input.f32> <out-dir>
//! ```
use larql_models::quant::fp8_finegrained::{scale_sibling_name, Fp8Grid};
use larql_vindex::format::vindex3::opplan::exec::cpu::kernels::FusedFp8Block;
use larql_vindex::format::vindex3::opplan::exec::cpu::projector::{DenseProjector, WeightRows};
use std::io::{Read, Seek, SeekFrom, Write};

/// One tensor as the shard stores it: dtype, shape, raw bytes.
type Raw = (String, Vec<usize>, Vec<u8>);

struct Shard {
    file: std::fs::File,
    header: serde_json::Value,
    base: u64,
}

impl Shard {
    fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len) as usize;
        let mut buf = vec![0u8; n];
        file.read_exact(&mut buf)?;
        Ok(Self {
            file,
            header: serde_json::from_slice(&buf)?,
            base: 8 + n as u64,
        })
    }

    fn raw(&mut self, name: &str) -> Result<Raw, Box<dyn std::error::Error>> {
        let e = self
            .header
            .get(name)
            .ok_or_else(|| format!("tensor `{name}` not in this shard"))?;
        let dtype = e["dtype"].as_str().ok_or("no dtype")?.to_string();
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .ok_or("no shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = e["data_offsets"].as_array().ok_or("no offsets")?;
        let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
        let mut bytes = vec![0u8; (b - a) as usize];
        self.file.seek(SeekFrom::Start(self.base + a))?;
        self.file.read_exact(&mut bytes)?;
        Ok((dtype, shape, bytes))
    }
}

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_f32(path: &str, v: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

/// One FP8 projection, straight off the checkpoint's bytes.
fn project(
    shard: &mut Shard,
    tensor: &str,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (wd, wshape, codes) = shard.raw(tensor)?;
    assert_eq!(wd, "F8_E4M3", "`{tensor}` is {wd}");
    let (sd, sshape, sbytes) = shard.raw(&scale_sibling_name(tensor))?;
    assert_eq!(sd, "F32", "scale sibling of `{tensor}` is {sd}");
    let scales: Vec<f32> = sbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let grid = Fp8Grid {
        rows: wshape[0],
        cols: wshape[1],
        scale_rows: sshape[0],
        scale_cols: sshape[1],
    };
    let (block_rows, block_cols) = grid.tile()?;
    assert_eq!(
        grid.cols,
        x.len(),
        "`{tensor}` expects {} inputs",
        grid.cols
    );

    let rows = WeightRows::Fp8Block {
        codes: &codes,
        scales: &scales,
        block_rows,
        block_cols,
        scale_cols: grid.scale_cols,
        row_in_tile: 0,
    };
    eprintln!(
        "  {tensor}: [{}, {}] / [{}, {}] -> {block_rows}x{block_cols}, slab {} bytes",
        grid.rows,
        grid.cols,
        grid.scale_rows,
        grid.scale_cols,
        rows.bytes()
    );
    let mut out = vec![0.0f32; grid.rows];
    FusedFp8Block.project_rows(rows, x, &mut out);
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: glm_ffn_fp8_probe <shard> <layer-prefix> <input.f32> <out-dir>");
        std::process::exit(2);
    }
    let (path, prefix, input, out_dir) = (&a[1], &a[2], &a[3], &a[4]);
    let mut shard = Shard::open(path)?;
    let x = read_f32(input)?;
    eprintln!("input: {} values", x.len());

    let gate = project(&mut shard, &format!("{prefix}.mlp.gate_proj.weight"), &x)?;
    let up = project(&mut shard, &format!("{prefix}.mlp.up_proj.weight"), &x)?;
    write_f32(&format!("{out_dir}/gate_proj.f32"), &gate)?;
    write_f32(&format!("{out_dir}/up_proj.f32"), &up)?;

    // `swiglu_limit`, applied exactly as `Glm5NextTextMLP.forward` applies
    // it — and note the ASYMMETRY: the gate is clamped ABOVE only, the up
    // branch on BOTH sides. A symmetric clamp on the gate would be a
    // different function, and would agree with this one on every input
    // whose gate never goes below -limit.
    let limit: f32 = std::env::var("GLM_SWIGLU_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);
    let clamped_gate: Vec<f32> = gate.iter().map(|&g| g.min(limit)).collect();
    let clamped_up: Vec<f32> = up.iter().map(|&u| u.clamp(-limit, limit)).collect();

    // How many values the clamp actually bit. Reported so a run cannot
    // silently qualify the clamp on a fixture where it never fires.
    let bit_gate = gate.iter().filter(|&&g| g > limit).count();
    let bit_up = up.iter().filter(|&&u| u.abs() > limit).count();
    eprintln!("  swiglu_limit {limit}: clamped {bit_gate} gate and {bit_up} up values");

    // `act_fn` is `silu(gate)` ALONE — the reference applies the module
    // to the gate and multiplies by `up` afterwards, so a capture of that
    // module's output is not the GLU product.
    let act_fn: Vec<f32> = clamped_gate
        .iter()
        .map(|&g| g / (1.0 + (-g).exp()))
        .collect();
    write_f32(&format!("{out_dir}/act_fn.f32"), &act_fn)?;

    let glu: Vec<f32> = act_fn
        .iter()
        .zip(&clamped_up)
        .map(|(&a, &u)| a * u)
        .collect();
    write_f32(&format!("{out_dir}/glu.f32"), &glu)?;

    let down = project(&mut shard, &format!("{prefix}.mlp.down_proj.weight"), &glu)?;
    write_f32(&format!("{out_dir}/down_proj.f32"), &down)?;
    eprintln!("wrote 5 boundaries to {out_dir}");
    Ok(())
}
