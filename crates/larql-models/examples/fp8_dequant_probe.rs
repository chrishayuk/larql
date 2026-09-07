//! Dequantise one fine-grained FP8 tensor from a real checkpoint and dump
//! the f32 values, so an independent implementation can be compared
//! against this one byte for byte.
//!
//! Reads the safetensors shard directly — no container, no plan — because
//! the property under test is the CODEC, and routing it through the
//! encode path would make a codec failure and a placement failure look
//! the same.
//!
//! ```text
//! cargo run --release -p larql-models --example fp8_dequant_probe -- \
//!     <shard.safetensors> <tensor-name> <out.f32>
//! ```
use larql_models::inventory::representation::read_stored_representation;
use larql_models::quant::fp8_finegrained::{dequantize, scale_sibling_name, Fp8Grid};
use std::io::{Read, Seek, SeekFrom, Write};

type Tensor = (String, Vec<usize>, Vec<u8>);

fn read_tensor(
    f: &mut std::fs::File,
    header: &serde_json::Value,
    base: u64,
    name: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let e = header
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
    let mut buf = vec![0u8; (b - a) as usize];
    f.seek(SeekFrom::Start(base + a))?;
    f.read_exact(&mut buf)?;
    Ok((dtype, shape, buf))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: fp8_dequant_probe <shard.safetensors> <tensor> <out.f32>");
        std::process::exit(2);
    }
    let (path, tensor, out_path) = (&args[1], &args[2], &args[3]);

    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len) as usize;
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let base = 8 + header_len as u64;

    let (wd, wshape, wbytes) = read_tensor(&mut f, &header, base, tensor)?;
    assert_eq!(wd, "F8_E4M3", "tensor `{tensor}` is {wd}, not F8_E4M3");
    let sib = scale_sibling_name(tensor);
    let (sd, sshape, sbytes) = read_tensor(&mut f, &header, base, &sib)?;
    assert_eq!(sd, "F32", "scale sibling `{sib}` is {sd}, not F32");

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
    let (bm, bn) = grid.tile()?;
    eprintln!(
        "{tensor}: [{}, {}] E4M3 / [{}, {}] F32 scales -> {bm}x{bn} tiles",
        grid.rows, grid.cols, grid.scale_rows, grid.scale_cols
    );

    // Cross-check the checkpoint's declared tile against this tensor's
    // own grid. The derived tile is what gets applied either way; a
    // disagreement is reported because on a checkpoint where the two are
    // meant to agree it is the first sign of a mis-read shape.
    let ckpt_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let cfg_path = ckpt_dir.join("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg_path) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(r) = read_stored_representation(&cfg) {
                let rep = &r.representation;
                eprintln!(
                    "  quant_method={} fmt={:?} finegrained_e4m3={}",
                    rep.method,
                    rep.fmt,
                    rep.is_finegrained_fp8_e4m3()
                );
                if let Some(declared) = rep.declared_tile() {
                    match grid.check_declared_tile(declared)? {
                        Ok(()) => eprintln!("  declared tile {declared:?} agrees"),
                        Err(d) => eprintln!("  TILE DISAGREEMENT: {d}"),
                    }
                }
            }
        }
    }

    let values = dequantize(&wbytes, &scales, grid)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    for v in &values {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()?;
    eprintln!("wrote {} f32 values to {out_path}", values.len());
    Ok(())
}
