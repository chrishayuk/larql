//! PREC-DIAG-1 E1b: write the RoPE cos/sin tables LARQL computes, under
//! each `RopeAngles` construction, for an outside comparison against
//! transformers (`scripts/prec_diag_rope_tables.py`).
//!
//! usage: cargo run -p larql-vindex --example rope_angle_tables -- \
//!            <head_dim> <theta> <positions> <out_dir> [linear_factor]
//!
//! Writes, little-endian f32, `[positions, head_dim/2]` row-major:
//!   `larql.cos.f32`, `larql.sin.f32`   — the shipped construction, read
//!       back out of `kernels::rope_rotate` itself (a unit vector rotated
//!       at each pair), so the table IS what execution applies;
//!   `hf-f32.cos.f32`, `hf-f32.sin.f32` — `rope_angles::hf_f32_cos_sin`
//!       over `hf_f32_inv_freq`, the functions execution calls;
//!   `hf-f32.inv_freq.f32` — `[head_dim/2]`.
//! Arithmetic only: no container, no weights, no forward pass.

use std::io::Write;
use std::path::PathBuf;

use larql_vindex::format::vindex3::opplan::exec::kernels::linear_frequencies;
use larql_vindex::format::vindex3::opplan::exec::kernels::{rope_rotate, rope_rotate_scaled};
use larql_vindex::format::vindex3::opplan::exec::rope_angles::{hf_f32_cos_sin, hf_f32_inv_freq};

fn write(path: PathBuf, values: &[f32]) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: <head_dim> <theta> <positions> <out_dir> [linear_factor]".into());
    }
    let head_dim: usize = args[1].parse()?;
    let theta: f64 = args[2].parse()?;
    let positions: usize = args[3].parse()?;
    let out = PathBuf::from(&args[4]);
    let factor: Option<f64> = args.get(5).map(|f| f.parse()).transpose()?;
    std::fs::create_dir_all(&out)?;
    let half = head_dim / 2;

    let shipped_linear = factor.map(|f| linear_frequencies(head_dim, theta, f));
    let mut larql_cos = Vec::with_capacity(positions * half);
    let mut larql_sin = Vec::with_capacity(positions * half);
    for position in 0..positions {
        for i in 0..half {
            let mut unit = vec![0.0f32; head_dim];
            unit[i] = 1.0;
            match &shipped_linear {
                None => rope_rotate(&mut unit, position, theta),
                Some(inv_freq) => rope_rotate_scaled(&mut unit, position, inv_freq, 1.0),
            }
            larql_cos.push(unit[i]);
            larql_sin.push(unit[half + i]);
        }
    }

    let inv_freq = hf_f32_inv_freq(head_dim, theta, factor);
    let mut hf_cos = Vec::with_capacity(positions * half);
    let mut hf_sin = Vec::with_capacity(positions * half);
    for position in 0..positions {
        for f in &inv_freq {
            let (c, s) = hf_f32_cos_sin(position, *f);
            hf_cos.push(c);
            hf_sin.push(s);
        }
    }

    write(out.join("larql.cos.f32"), &larql_cos)?;
    write(out.join("larql.sin.f32"), &larql_sin)?;
    write(out.join("hf-f32.cos.f32"), &hf_cos)?;
    write(out.join("hf-f32.sin.f32"), &hf_sin)?;
    write(out.join("hf-f32.inv_freq.f32"), &inv_freq)?;
    println!(
        "wrote [{positions}, {half}] tables for head_dim {head_dim}, theta {theta}, \
         linear factor {factor:?} to {}",
        out.display()
    );
    Ok(())
}
