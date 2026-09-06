//! Run GLM-5.3-Flash's sparse MoE branch through LARQL's routed FFN, over
//! the CHECKPOINT's own bytes, and dump every boundary.
//!
//! The whole 288-expert bank is bound — 6.78 GiB of it — because the
//! router selects inside the call: handing it only the experts the
//! reference chose would test the branch while assuming the selection.
//! Nothing is widened; the experts and the shared expert are
//! fine-grained FP8 and the router is BF16.
//!
//! ```text
//! cargo run --release -p larql-vindex --example glm_moe_probe -- \
//!     <checkpoint-dir> <layer> <input.f32> <out-dir>
//! ```
use larql_models::config::Activation;
use larql_models::quant::fp8_finegrained::{scale_sibling_name, Fp8Grid};
use larql_models::{ExpertGatePolicy, ExpertRoutingPolicy, MoeRouterKind};
use larql_vindex::format::vindex3::opplan::exec::backend::{
    ExpertSlices, PlanBackend, RoutedFfnCall, WeightSlice,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::MappedAccess;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};

/// One tensor as the shard stores it: dtype, shape, raw bytes.
type Raw = (String, Vec<usize>, Vec<u8>);

/// One matrix as the checkpoint stores it — nothing widened.
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
    fn slice(&self) -> WeightSlice<'_> {
        match self {
            Held::Fp8 {
                codes,
                scales,
                block_rows,
                block_cols,
                scale_cols,
            } => WeightSlice::Fp8Block {
                codes,
                scales,
                block_rows: *block_rows,
                block_cols: *block_cols,
                scale_cols: *scale_cols,
            },
            Held::Bf16(w) => WeightSlice::Bf16(w),
        }
    }
}

struct Checkpoint {
    dir: std::path::PathBuf,
    map: BTreeMap<String, String>,
    open: BTreeMap<String, (std::fs::File, serde_json::Value, u64)>,
}

impl Checkpoint {
    fn open(dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            std::path::Path::new(dir).join("model.safetensors.index.json"),
        )?)?;
        let map = idx["weight_map"]
            .as_object()
            .ok_or("no weight_map")?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect();
        Ok(Self {
            dir: dir.into(),
            map,
            open: BTreeMap::new(),
        })
    }

    fn raw(&mut self, name: &str) -> Result<Raw, Box<dyn std::error::Error>> {
        let shard = self
            .map
            .get(name)
            .ok_or_else(|| format!("`{name}` is not in the index"))?
            .clone();
        if !self.open.contains_key(&shard) {
            let mut f = std::fs::File::open(self.dir.join(&shard))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let n = u64::from_le_bytes(len) as usize;
            let mut buf = vec![0u8; n];
            f.read_exact(&mut buf)?;
            self.open.insert(
                shard.clone(),
                (f, serde_json::from_slice(&buf)?, 8 + n as u64),
            );
        }
        let (f, header, base) = self.open.get_mut(&shard).expect("just inserted");
        let e = &header[name];
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
        f.seek(SeekFrom::Start(*base + a))?;
        f.read_exact(&mut bytes)?;
        Ok((dtype, shape, bytes))
    }

    fn hold(&mut self, tensor: &str) -> Result<Held, Box<dyn std::error::Error>> {
        let (dtype, shape, bytes) = self.raw(tensor)?;
        match dtype.as_str() {
            "F8_E4M3" => {
                let (sd, sshape, sbytes) = self.raw(&scale_sibling_name(tensor))?;
                assert_eq!(sd, "F32");
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
                Ok(Held::Fp8 {
                    codes: bytes,
                    scales,
                    block_rows,
                    block_cols,
                    scale_cols: grid.scale_cols,
                })
            }
            "BF16" => Ok(Held::Bf16(
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            )),
            other => Err(format!("`{tensor}` is {other}").into()),
        }
    }

    fn f32(&mut self, tensor: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (dtype, _, bytes) = self.raw(tensor)?;
        Ok(match dtype.as_str() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => return Err(format!("`{tensor}` is {other}").into()),
        })
    }
}

fn write_f32(path: &str, v: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for x in v {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: glm_moe_probe <checkpoint-dir> <layer> <input.f32> <out-dir>");
        std::process::exit(2);
    }
    let (dir, layer, input, out_dir) = (&a[1], &a[2], &a[3], &a[4]);
    let prefix = format!("model.language_model.layers.{layer}");
    let mut ck = Checkpoint::open(dir)?;

    let experts = env_usize("GLM_EXPERTS", 288);
    let top_k = env_usize("GLM_TOP_K", 8);
    let intermediate = env_usize("GLM_MOE_INTERMEDIATE", 2048);
    let branch_scale: f32 = std::env::var("GLM_ROUTED_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.5);
    let limit: f32 = std::env::var("GLM_SWIGLU_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);

    let router = ck.f32(&format!("{prefix}.mlp.gate.weight"))?;
    let router_bias = ck.f32(&format!("{prefix}.mlp.gate.e_score_correction_bias"))?;
    let hidden = router.len() / experts;
    eprintln!("router: [{experts}, {hidden}], bias {}", router_bias.len());

    eprintln!("binding {experts} experts (nothing widened) …");
    let mut gate_h = Vec::with_capacity(experts);
    let mut up_h = Vec::with_capacity(experts);
    let mut down_h = Vec::with_capacity(experts);
    for e in 0..experts {
        gate_h.push(ck.hold(&format!("{prefix}.mlp.experts.{e}.gate_proj.weight"))?);
        up_h.push(ck.hold(&format!("{prefix}.mlp.experts.{e}.up_proj.weight"))?);
        down_h.push(ck.hold(&format!("{prefix}.mlp.experts.{e}.down_proj.weight"))?);
    }
    let gate: Vec<WeightSlice<'_>> = gate_h.iter().map(Held::slice).collect();
    let up: Vec<WeightSlice<'_>> = up_h.iter().map(Held::slice).collect();
    let down: Vec<WeightSlice<'_>> = down_h.iter().map(Held::slice).collect();

    let xs: Vec<f32> = std::fs::read(input)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let positions = xs.len() / hidden;
    eprintln!("input: {positions} positions x {hidden}");

    // Controls. Each perturbs a DECLARED routing or gating fact and
    // leaves everything else alone, so a firing control names which fact
    // the comparison is actually sensitive to.
    //
    // `no-renorm` and `no-scale` are the ones that matter most here:
    // both leave the SELECTED EXPERTS untouched and change only their
    // weights, so a gate that merely agreed on top-k could not pass
    // them.
    let control = std::env::var("GLM_MOE_CONTROL").unwrap_or_default();
    let (routing_policy, scale, policy) = match control.as_str() {
        "" => (
            ExpertRoutingPolicy::NormalisedOverSelected,
            branch_scale,
            ExpertGatePolicy::ClampedGated { limit },
        ),
        "no-renorm" => (
            ExpertRoutingPolicy::SoftmaxThenSelect,
            branch_scale,
            ExpertGatePolicy::ClampedGated { limit },
        ),
        "no-scale" => (
            ExpertRoutingPolicy::NormalisedOverSelected,
            1.0,
            ExpertGatePolicy::ClampedGated { limit },
        ),
        // The defect this rung actually found: GPT-OSS's clamped GLU
        // served for GLM's clamped SwiGLU.
        "gpt-oss-glu" => (
            ExpertRoutingPolicy::NormalisedOverSelected,
            branch_scale,
            ExpertGatePolicy::ClampedGlu { limit, alpha: 1.0 },
        ),
        "no-clamp" => (
            ExpertRoutingPolicy::NormalisedOverSelected,
            branch_scale,
            ExpertGatePolicy::Gated,
        ),
        other => return Err(format!("unknown control `{other}`").into()),
    };
    if !control.is_empty() {
        eprintln!("  CONTROL: {control}");
    }

    let backend = ProductionBackend;
    let mut routed_out = Vec::new();
    for p in 0..positions {
        let x = &xs[p * hidden..(p + 1) * hidden];
        let y = backend.routed_ffn(RoutedFfnCall {
            x,
            hidden,
            intermediate,
            experts,
            top_k,
            router_kind: MoeRouterKind::Sigmoid,
            routing_policy,
            branch_scale: scale,
            activation: Activation::Silu,
            gate_policy: policy,
            router: &router,
            router_bias: Some(&router_bias),
            weights: ExpertSlices::Separate {
                gate: &gate,
                up: &up,
                down: &down,
                access: MappedAccess::Demand,
            },
            gate_up_bias: None,
            down_bias: None,
            router_input: None,
            router_scale: None,
            router_per_expert_scale: None,
            router_norm_eps: None,
        })?;
        routed_out.extend_from_slice(&y);
    }
    write_f32(&format!("{out_dir}/routed.f32"), &routed_out)?;
    eprintln!("wrote the routed branch for {positions} positions to {out_dir}");
    Ok(())
}
