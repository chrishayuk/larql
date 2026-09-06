//! Run GLM-5.3-Flash's MLA-NoPE attention through LARQL's MLA executor,
//! over the CHECKPOINT's own bytes, and dump every boundary.
//!
//! The q-LoRA path is the point: GLM compresses its query
//! (`q_a_proj → q_a_layernorm → q_b_proj`) where Kimi Linear ships one
//! flat `q_proj`, and the form is chosen by the DECLARED `q_lora_rank`.
//! Mixed representation on purpose — `q_a_proj`, `q_b_proj`,
//! `kv_a_proj_with_mqa` and `o_proj` are fine-grained FP8 while
//! `kv_b_proj` and both norms are BF16 — so one layer exercises both
//! binding paths.
//!
//! ```text
//! cargo run --release -p larql-vindex --example glm_mla_probe -- \
//!     <shard.safetensors> <layer-prefix> <input.f32> <positions> <out-dir>
//! ```
use larql_models::config::MlaGeometry;
use larql_models::quant::fp8_finegrained::{scale_sibling_name, Fp8Grid};
use larql_vindex::format::vindex3::opplan::exec::continuation::LatentKvRows;
use larql_vindex::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use larql_vindex::format::vindex3::opplan::exec::mla::{
    mla_forward, MlaQueryWeights, MlaWeights, Mutation,
};
use std::io::{Read, Seek, SeekFrom, Write};

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

/// A matrix as LARQL will hold it: FP8 codes + scales, or BF16 units.
enum Held {
    Fp8 {
        codes: Vec<u8>,
        scales: Vec<f32>,
        block_rows: usize,
        block_cols: usize,
        scale_cols: usize,
    },
    Bf16(Vec<u16>),
}

impl Held {
    fn rows(&self) -> WeightRows<'_> {
        match self {
            Held::Fp8 {
                codes,
                scales,
                block_rows,
                block_cols,
                scale_cols,
            } => WeightRows::Fp8Block {
                codes,
                scales,
                block_rows: *block_rows,
                block_cols: *block_cols,
                scale_cols: *scale_cols,
                row_in_tile: 0,
            },
            Held::Bf16(w) => WeightRows::Bf16(w),
        }
    }
}

/// Bind one matrix in whatever form the checkpoint stores it — nothing
/// widened.
fn hold(shard: &mut Shard, tensor: &str) -> Result<Held, Box<dyn std::error::Error>> {
    let (dtype, shape, bytes) = shard.raw(tensor)?;
    match dtype.as_str() {
        "F8_E4M3" => {
            let (sd, sshape, sbytes) = shard.raw(&scale_sibling_name(tensor))?;
            assert_eq!(sd, "F32", "scale sibling of `{tensor}` is {sd}");
            let scales: Vec<f32> = sbytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let grid = Fp8Grid {
                rows: shape[0],
                cols: shape[1],
                scale_rows: sshape[0],
                scale_cols: sshape[1],
            };
            let (block_rows, block_cols) = grid.tile()?;
            eprintln!(
                "  {tensor}: FP8 [{}, {}] {block_rows}x{block_cols}",
                shape[0], shape[1]
            );
            Ok(Held::Fp8 {
                codes: bytes,
                scales,
                block_rows,
                block_cols,
                scale_cols: grid.scale_cols,
            })
        }
        "BF16" => {
            eprintln!("  {tensor}: BF16 {shape:?}");
            Ok(Held::Bf16(
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ))
        }
        other => Err(format!("`{tensor}` is {other}, which this probe does not bind").into()),
    }
}

fn bf16_f32(shard: &mut Shard, tensor: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (dtype, _, bytes) = shard.raw(tensor)?;
    assert_eq!(dtype, "BF16", "`{tensor}` is {dtype}");
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: glm_mla_probe <shard> <layer-prefix> <input.f32> <positions> <out-dir>");
        std::process::exit(2);
    }
    let (path, prefix, input, out_dir) = (&a[1], &a[2], &a[3], &a[5]);
    let positions: usize = a[4].parse()?;

    let mut shard = Shard::open(path)?;
    let all = std::fs::read(input)?;
    let xs: Vec<f32> = all
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let hidden = xs.len() / positions;
    eprintln!("input: {positions} positions x {hidden}");

    let q_a = hold(&mut shard, &format!("{prefix}.self_attn.q_a_proj.weight"))?;
    let q_b = hold(&mut shard, &format!("{prefix}.self_attn.q_b_proj.weight"))?;
    let kv_a = hold(
        &mut shard,
        &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
    )?;
    let kv_b = hold(&mut shard, &format!("{prefix}.self_attn.kv_b_proj.weight"))?;
    let o = hold(&mut shard, &format!("{prefix}.self_attn.o_proj.weight"))?;
    let q_a_norm = bf16_f32(
        &mut shard,
        &format!("{prefix}.self_attn.q_a_layernorm.weight"),
    )?;
    let kv_a_norm = bf16_f32(
        &mut shard,
        &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
    )?;

    // Geometry from the checkpoint's own shapes, so a wrong reading of
    // the config cannot pass unnoticed.
    let (_, q_b_shape, _) = shard.raw(&format!("{prefix}.self_attn.q_b_proj.weight"))?;
    let (_, kv_b_shape, _) = shard.raw(&format!("{prefix}.self_attn.kv_b_proj.weight"))?;
    let kv_lora_rank = kv_a_norm.len();
    let num_heads: usize = std::env::var("GLM_MLA_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let q_head_dim = q_b_shape[0] / num_heads;
    let per_head_kv = kv_b_shape[0] / num_heads;
    // NoPE: `qk_rope_head_dim` is 0, so the query head width is all nope
    // and `kv_a_proj_with_mqa` emits the latent alone.
    let geometry = MlaGeometry {
        num_heads,
        kv_lora_rank,
        qk_nope_head_dim: q_head_dim,
        qk_rope_head_dim: 0,
        v_head_dim: per_head_kv - q_head_dim,
    };
    eprintln!(
        "  geometry: {num_heads} heads, qk_nope {q_head_dim}, v {}, latent {kv_lora_rank}",
        geometry.v_head_dim
    );

    let eps: f64 = std::env::var("GLM_RMS_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    let weights = MlaWeights {
        query: MlaQueryWeights::LowRank {
            q_a_proj: q_a.rows(),
            q_a_norm: &q_a_norm,
            q_a_norm_eps: eps,
            q_b_proj: q_b.rows(),
        },
        kv_a_proj: kv_a.rows(),
        kv_a_norm: &kv_a_norm,
        kv_b_proj: kv_b.rows(),
        o_proj: o.rows(),
        kv_a_norm_eps: eps,
        output_gate: None,
    };

    // A control arm, so the gate can be shown to fail. `QbFedPreNorm`
    // feeds `q_b_proj` the UNNORMALISED compression — the q-LoRA path's
    // own norm site, distinct from the KV latent's.
    let mutation = match std::env::var("GLM_MLA_MUTATION").as_deref() {
        Ok("omit-q-a-norm") => Mutation::QbFedPreNorm,
        Ok("omit-kv-a-norm") => Mutation::OmitKvANorm,
        Ok(other) => return Err(format!("unknown mutation `{other}`").into()),
        Err(_) => Mutation::None,
    };
    if mutation != Mutation::None {
        eprintln!("  MUTATION: {mutation:?}");
    }

    let mut state = LatentKvRows::default();
    let mut outputs = Vec::new();
    let mut last = None;
    for p in 0..positions {
        let trace = mla_forward(
            &xs[p * hidden..(p + 1) * hidden],
            hidden,
            weights,
            geometry,
            &mut state,
            mutation,
        );
        outputs.extend_from_slice(&trace.output);
        last = Some(trace);
    }
    let last = last.expect("at least one position");
    // `q_a_normed` is the normalised query latent — GLM's DSA indexer
    // reads exactly this value (`q_resid`). Named by the executor, not
    // recomputed here.
    write_f32(
        &format!("{out_dir}/q_latent.f32"),
        last.q_a_normed.as_deref().unwrap_or(&[]),
    )?;
    write_f32(&format!("{out_dir}/q_proj.f32"), &last.q_states)?;
    write_f32(&format!("{out_dir}/compressed_kv.f32"), &last.compressed_kv)?;
    write_f32(&format!("{out_dir}/kv_a_normed.f32"), &last.kv_a_normed)?;
    write_f32(&format!("{out_dir}/kv_b.f32"), &last.kv_b)?;
    write_f32(&format!("{out_dir}/attn_value.f32"), &last.attn_value)?;
    write_f32(&format!("{out_dir}/output.f32"), &outputs)?;
    eprintln!("wrote {positions}-position boundaries to {out_dir}");
    Ok(())
}
