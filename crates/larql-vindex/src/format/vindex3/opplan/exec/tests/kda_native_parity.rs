//! **NATIVE-KDA-1's gate: stored Q8 versus transient Q8, BIT-EQUAL.**
//!
//! The transient arm re-encodes the source's bf16 projections at load;
//! the native arm reads bytes a compiler sealed to disk. Same intended
//! bytes, same kernels — so the tolerance is ZERO: any difference in
//! the banks or in one logit means the representation pipeline is
//! lying somewhere (encoder drift between the arena and the loader's
//! requant, a placement error, a truncated read), never "quantisation
//! error". This is the same no-tolerance posture the REPRESENT verb
//! established for stored-vs-transient compilation.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.s6.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=~/chris-models/qbanks/kimi-quality-bank-256x32 \
//! LARQL_KIMI_KDA_CANDIDATE=/tmp/kimi-kda-l20-25q80.vindex3 \
//!   cargo test -p larql-vindex --features gpu --release --lib \
//!   kda_native_parity -- --nocapture
//! ```

use std::path::PathBuf;

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use super::kda_q8_real::build_layers;
use super::q2a_teacher_forced::{env_dir, run_sequence, sequence_embeddings, BANK_ENV, SOURCE_ENV};
use crate::format::vindex3::opplan::exec::kimi_source::{KdaOverlay, KimiSourceModel};
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceAttn, DeviceLayer, HybridStack};

const KDA_CANDIDATE_ENV: &str = "LARQL_KIMI_KDA_CANDIDATE";
/// Sequences for the end-to-end logit comparison. Two is enough: the
/// claim is bitwise identity of an already-proven computation, not a
/// statistical bound.
const SEQUENCES: usize = 2;

#[test]
fn native_kda_bytes_and_logits_are_bit_equal_to_the_transient_arm() {
    let (Some(source_dir), Some(bank_dir), Some(kda_dir)) = (
        env_dir(SOURCE_ENV),
        env_dir(BANK_ENV),
        std::env::var_os(KDA_CANDIDATE_ENV).map(PathBuf::from),
    ) else {
        eprintln!("skipped: set {SOURCE_ENV}, {BANK_ENV} and {KDA_CANDIDATE_ENV}");
        return;
    };
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bank_dir.join("manifest.json")).expect("manifest"))
            .expect("bank manifest parses");
    let positions = manifest["positions"].as_u64().unwrap() as usize;

    let model = KimiSourceModel::open(&source_dir).expect("source container opens");
    let g = model.geometry.clone();
    let overlay = KdaOverlay::open(&kda_dir, &source_dir, &g).expect("kda overlay opens");
    let layers = overlay.compiled_layers();
    assert!(!layers.is_empty());
    eprintln!(
        "[kda-native] candidate `{}` holds layers {layers:?}",
        overlay.index.map.name
    );
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    model
        .register_stores(&metal, &moe_layers)
        .expect("stores register");

    // Transient arm: the probe's own requant-at-load.
    let targets: Vec<usize> = layers.iter().map(|&l| l as usize).collect();
    let (transient, swapped) = build_layers(&metal, &model, &targets, None);
    assert_eq!(swapped.len(), targets.len());
    // Native arm: bytes a compiler sealed to disk, verified against the
    // ledger at open and at every binding read.
    let native: Vec<Option<DeviceLayer>> = (0..g.num_layers)
        .map(|i| {
            Some(
                model
                    .device_layer_with_kda(&metal, i, None, Some(&overlay))
                    .unwrap_or_else(|e| panic!("layer {i} must load: {e}")),
            )
        })
        .collect();

    // ── The core claim, checked at the BYTES before any kernel runs:
    //    every compiled layer's bank, offsets and encoding identical
    //    between the arms. This is where an arena-vs-loader encoder
    //    drift would surface, named by layer. ──
    for (i, (t, n)) in transient.iter().zip(&native).enumerate() {
        let (t, n) = (t.as_ref().unwrap(), n.as_ref().unwrap());
        if let (
            DeviceAttn::Kda {
                qkv_bank: tq,
                qkv_offsets: to,
                o_proj: tp,
                encoding: te,
                ..
            },
            DeviceAttn::Kda {
                qkv_bank: nq,
                qkv_offsets: no,
                o_proj: np,
                encoding: ne,
                ..
            },
        ) = (&t.attn, &n.attn)
        {
            assert_eq!(te, ne, "layer {i}: encodings differ");
            assert_eq!(to, no, "layer {i}: qkv offsets differ");
            assert!(tq == nq, "layer {i}: qkv banks are not byte-equal");
            assert!(tp == np, "layer {i}: o_proj banks are not byte-equal");
        }
    }
    eprintln!(
        "[kda-native] every compiled layer's banks BYTE-EQUAL between transient and \
         native arms"
    );

    // ── And the end-to-end seal: identical logits, bitwise. ──
    let assemble = |layers: Vec<Option<DeviceLayer>>| -> HybridStack<'_> {
        for d in layers.iter().flatten() {
            for bank in d.attention_banks() {
                metal.register_weight_region(bank);
            }
        }
        let host = (0..layers.len()).map(|_| None).collect();
        let mut stack = HybridStack::new(layers, host);
        assert!(stack.attach_head(model.head().expect("head loads")));
        stack
    };
    let mut a = assemble(transient);
    let mut b = assemble(native);
    metal.seal_weight_regions();
    for seq in 0..SEQUENCES {
        let rows = sequence_embeddings(&bank_dir, seq, positions, g.hidden);
        let la = run_sequence(&metal, &mut a, &rows, g.hidden);
        let lb = run_sequence(&metal, &mut b, &rows, g.hidden);
        for (pos, ((va, _), (vb, _))) in la.into_iter().zip(lb).enumerate() {
            assert!(
                va.iter().zip(&vb).all(|(x, y)| x.to_bits() == y.to_bits()),
                "seq {seq} pos {pos}: logits differ bitwise — the pipeline is lying \
                 somewhere between the compiler and the loader"
            );
        }
    }
    eprintln!(
        "[kda-native] {SEQUENCES} sequences x {positions} positions: logits BIT-EQUAL. \
         Native KDA storage serves exactly the bytes the transient experiment earned \
         its evidence with."
    );
}
