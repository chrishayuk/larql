//! `larql shannon` — next-token bit measurements for scriptable demos.
//!
//! These commands put the existing dense transformer forward pass behind a
//! Shannon-style surface: score the true next token, report `-log2(p)`, and
//! optionally drive a real arithmetic coder from the model distribution.

use std::fs;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::attention::SharedKV;
use larql_inference::forward::{apply_norm, dot_proj};
use larql_inference::{
    encode_prompt, ExpertWeightFfn, FfnBackend, InferenceModel, ModelWeights,
    PackedExpertWeightFfn, WeightFfn,
};
use ndarray::{s, Array2};

mod args;
mod arith;
mod layers;
mod scoring;
mod verify;
mod vindex;
pub use args::ShannonCommand;
use args::*;
use arith::*;
use layers::*;
use scoring::*;
pub(crate) use scoring::{forward_hidden_all_layers, load_model, read_text};
use verify::*;
use vindex::*;

const LN_2: f64 = std::f64::consts::LN_2;
pub(crate) const DEFAULT_CONTEXT: usize = 512;
const DEFAULT_STRIDE: usize = 256;

// ── Engine identifiers used across `shannon verify` ─────────────────────
// Engines name themselves in the comparison table, in the --engines arg
// parser, and in the `RESULT {...}` JSON line each Python scorer emits.
// Keeping the literals here means a typo can't drift them apart.
const ENGINE_RUST: &str = "rust";
const ENGINE_MLX: &str = "mlx";
const ENGINE_HF: &str = "hf";

/// Prefix the Python reference scorers emit on their final JSON line when
/// invoked with `--json`. The verify subprocess parser greps for this. If
/// you change it, also update `scripts/shannon_score_{mlx,hf}.py` and the
/// `--json` flag's help text there.
const RESULT_PREFIX: &str = "RESULT ";
// Arithmetic coding must rebuild the exact same integer frequency table when
// decoding. The vindex/Metal path is fast but can produce tiny cross-run float
// drift, so keep this comfortably above Gemma's 262K vocab without making the
// table hypersensitive to low-order logit differences.
const FREQ_TOTAL: u32 = 1 << 19;
const CODE_BITS: u32 = 32;
const TOP_VALUE: u64 = (1u64 << CODE_BITS) - 1;
const FIRST_QTR: u64 = TOP_VALUE / 4 + 1;
const HALF: u64 = FIRST_QTR * 2;
const THIRD_QTR: u64 = FIRST_QTR * 3;
const VINDEX_BLOCK_TARGET_TOKENS: usize = 512;

pub fn run(cmd: ShannonCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        ShannonCommand::Score(args) => run_score(args),
        ShannonCommand::Slot(args) => run_slot(args),
        ShannonCommand::Repeat(args) => run_repeat(args),
        ShannonCommand::Layers(args) => run_layers(args),
        ShannonCommand::Encode(args) => run_encode(args),
        ShannonCommand::Decode(args) => run_decode(args),
        ShannonCommand::Verify(args) => run_verify(args),
        ShannonCommand::LayerDump(args) => {
            crate::commands::primary::shannon_trace::dump::run_layer_dump(args)
        }
        ShannonCommand::BlockNorms(args) => {
            crate::commands::primary::shannon_trace::block_norms::run_block_norms(args)
        }
        ShannonCommand::LayerDiff(args) => {
            crate::commands::primary::shannon_trace::compare::run_layer_diff(args)
        }
        ShannonCommand::DecodeDiff(args) => {
            crate::commands::primary::shannon_trace::decode_diff::run_decode_diff(args)
        }
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_idx: 0,
            bit_idx: 0,
        }
    }

    pub(super) fn read(&mut self) -> bool {
        if self.byte_idx >= self.bytes.len() {
            return false;
        }
        let bit = (self.bytes[self.byte_idx] & (0x80 >> self.bit_idx)) != 0;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }
        bit
    }
}

impl<'a> ArithmeticDecoder<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        let mut bits = BitReader::new(bytes);
        let mut code = 0u64;
        for _ in 0..CODE_BITS {
            code = code * 2 + u64::from(bits.read());
        }
        Self {
            low: 0,
            high: TOP_VALUE,
            code,
            bits,
        }
    }

    pub(super) fn scaled_value(&self, total: u32) -> u32 {
        let range = self.high - self.low + 1;
        ((((self.code - self.low + 1) * total as u64 - 1) / range) as u32).min(total - 1)
    }

    pub(super) fn decode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.code -= FIRST_QTR;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
            self.code = self.code * 2 + u64::from(self.bits.read());
        }
    }
}

pub(super) fn progress_bar(len: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(label.to_string());
    pb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub(super) fn arithmetic_round_trip_fixed_counts() {
        let counts = vec![3, 1, 4, 2];
        let total: u32 = counts.iter().sum();
        let symbols = [0u32, 2, 2, 3, 1, 0, 2];

        let mut enc = ArithmeticEncoder::new();
        for &sym in &symbols {
            let (low, high) = interval_for_symbol(&counts, sym).unwrap();
            enc.encode(low, high, total);
        }
        let payload = enc.finish();
        let mut dec = ArithmeticDecoder::new(&payload);
        let mut out = Vec::new();
        for _ in 0..symbols.len() {
            let value = dec.scaled_value(total);
            let (sym, low, high) = symbol_for_value(&counts, value).unwrap();
            dec.decode(low, high, total);
            out.push(sym);
        }

        assert_eq!(out, symbols);
    }

    #[test]
    pub(super) fn shannon_file_round_trip() {
        let file = ShannonFile {
            context: 128,
            first_token: 2,
            target_tokens: 42,
            original_bytes: 100,
            payload: vec![1, 2, 3, 4],
        };
        let parsed = ShannonFile::from_bytes(&file.to_bytes()).unwrap();
        assert_eq!(parsed.context, 128);
        assert_eq!(parsed.first_token, 2);
        assert_eq!(parsed.target_tokens, 42);
        assert_eq!(parsed.original_bytes, 100);
        assert_eq!(parsed.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    pub(super) fn vindex_blocks_round_trip() {
        let blocks = vec![
            VindexShannonBlock {
                first_token: 2,
                target_tokens: 3,
                payload: vec![1, 2, 3],
            },
            VindexShannonBlock {
                first_token: 5,
                target_tokens: 1,
                payload: vec![8, 13],
            },
        ];

        let encoded = encode_vindex_blocks(&blocks);
        let parsed = parse_vindex_blocks(&encoded).unwrap().unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].first_token, 2);
        assert_eq!(parsed[0].target_tokens, 3);
        assert_eq!(parsed[0].payload, vec![1, 2, 3]);
        assert_eq!(parsed[1].first_token, 5);
        assert_eq!(parsed[1].target_tokens, 1);
        assert_eq!(parsed[1].payload, vec![8, 13]);
        assert!(parse_vindex_blocks(&[1, 2, 3]).unwrap().is_none());
    }
}
