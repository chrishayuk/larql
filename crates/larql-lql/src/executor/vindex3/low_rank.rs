//! `ADD LOW RANK`: stage a structured `add_low_rank_columns` (VLP numerics v2) on the bound VINDEX3 container.

use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, Profile};
use larql_vindex::format::vindex3::weight_patch::LowRankColumnsOp;
use serde::Deserialize;

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// Hex digits of one operand element in an f64-working profile, and in the f32 one.
const F64_HEX_DIGITS: usize = 16;
const F32_HEX_DIGITS: usize = 8;

/// The operand file an `ADD LOW RANK` reads: the rank, both factors as stored bits in the profile's working
/// width, and optionally where they came from. Bits, never decimals: the file fixes the operands exactly.
/// `a` is `rows × rank` and `b` is `rank × |columns|`, both row-major.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowRankFile {
    rank: usize,
    a: Vec<String>,
    b: Vec<String>,
    #[serde(default)]
    origin: Option<String>,
}

fn refused(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("ADD LOW RANK refused: {e}"))
}

/// Fixed-width lowercase hex, as the profile's working format.
fn operand_bits(texts: &[String], f64_width: bool, which: &str) -> Result<OperandBits, LqlError> {
    let width = if f64_width {
        F64_HEX_DIGITS
    } else {
        F32_HEX_DIGITS
    };
    let bits = texts
        .iter()
        .map(|h| {
            (h.len() == width && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
                .then(|| u64::from_str_radix(h, 16).ok())
                .flatten()
                .ok_or_else(|| {
                    refused(format!(
                        "`{h}` in `{which}` is not {width} lowercase hex digits"
                    ))
                })
        })
        .collect::<Result<Vec<u64>, _>>()?;
    Ok(if f64_width {
        OperandBits::F64(bits)
    } else {
        OperandBits::F32(bits.into_iter().map(|b| b as u32).collect())
    })
}

/// Operand values handed over in memory rather than in an operand file.
#[derive(Debug, Clone, Copy)]
pub enum OperandValues<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

/// The operand bits `values` are in the working width of the profile `numerics`: f64 values for an f64-working
/// profile (f32 values are widened, exactly), f32 values for the f32 one (f64 values are refused, since
/// narrowing rounds). The same [`OperandBits`] an operand file's hex parses to.
pub fn operand_bits_from_values(
    numerics: &str,
    values: OperandValues<'_>,
    which: &str,
) -> Result<OperandBits, LqlError> {
    let profile = Profile::parse(Some(numerics)).map_err(refused)?;
    match (profile.works_in_f64(), values) {
        (true, OperandValues::F64(v)) => Ok(OperandBits::F64(v.iter().map(|x| x.to_bits()).collect())),
        (true, OperandValues::F32(v)) => Ok(OperandBits::F64(
            v.iter().map(|&x| f64::from(x).to_bits()).collect(),
        )),
        (false, OperandValues::F32(v)) => Ok(OperandBits::F32(v.iter().map(|x| x.to_bits()).collect())),
        (false, OperandValues::F64(_)) => Err(refused(format!(
            "`{which}` holds f64 values and `{numerics}` works in f32; narrowing would round them, so pass f32 \
             values"
        ))),
    }
}

/// The origin an op records for operands handed over in memory: where they came from, each factor's shape and
/// working width, and the SHA-256 of its bits as little-endian bytes in that width (for f64 operands, the bytes
/// of a C-order `<f8` array of the values).
pub fn operand_origin(
    source: &str,
    a: &OperandBits,
    a_shape: [usize; 2],
    b: &OperandBits,
    b_shape: [usize; 2],
) -> String {
    use sha2::{Digest, Sha256};
    let describe = |bits: &OperandBits, [rows, cols]: [usize; 2]| {
        let (width, bytes): (&str, Vec<u8>) = match bits {
            OperandBits::F64(v) => ("f64", v.iter().flat_map(|b| b.to_le_bytes()).collect()),
            OperandBits::F32(v) => ("f32", v.iter().flat_map(|b| b.to_le_bytes()).collect()),
        };
        format!("{rows}x{cols} {width} sha256 {:x}", Sha256::digest(&bytes))
    };
    format!(
        "{source}: a {}; b {}",
        describe(a, a_shape),
        describe(b, b_shape)
    )
}

/// What staging one `add_low_rank_columns` did.
#[derive(Debug, Clone)]
pub struct StagedLowRank {
    /// The op as staged, sealed: what a recording saves into a `.vlp`.
    pub op: LowRankColumnsOp,
    /// How many elements of the tensor the op changes, resolved by the frozen interpreter.
    pub changed_elements: usize,
    /// Every term has a zero factor: the op compiles to the base's exact bytes.
    pub sham: bool,
    /// Structured ops pending COMPILE in this session after staging.
    pub pending_structured_ops: usize,
    /// The live-weights report the statement prints after staging.
    live: Vec<String>,
}

fn vindex3_path(session: &Session) -> Result<std::path::PathBuf, LqlError> {
    match &session.backend {
        Backend::Vindex3 { path, .. } => Ok(path.clone()),
        Backend::Vindex { .. } => Err(LqlError::Execution(
            "ADD LOW RANK edits the stored weights of a VINDEX3 container; \
             this session is bound to a VINDEX2 vindex"
                .into(),
        )),
        _ => Err(LqlError::NoBackend),
    }
}

impl Session {
    /// The typed operation behind `ADD LOW RANK`: stage `a · b` (`a` is `rows × rank`, `b` is
    /// `rank × |columns|`, both row-major, in the profile's working width) on `columns` of `tensor`. Everything
    /// checkable is checked before anything is staged: the profile, the tensor, the columns, and that the op,
    /// resolved by the frozen interpreter, edits at least one element. A `BEGIN PATCH` recording records it.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_low_rank(
        &mut self,
        tensor: &str,
        columns: &[usize],
        numerics: &str,
        rank: usize,
        a: &OperandBits,
        b: &OperandBits,
        origin: Option<String>,
    ) -> Result<StagedLowRank, LqlError> {
        let path = vindex3_path(self)?;
        Profile::parse(Some(numerics)).map_err(refused)?;
        let mut op = LowRankColumnsOp::bind(&path, tensor, numerics, rank, a, b, columns.to_vec())
            .map_err(refused)?;
        if let Some(origin) = origin {
            op = op.with_origin(origin);
        }
        // Resolve now, so a bad op is refused when it is written down and not when it is compiled.
        let resolved = op.resolve_tensor(&path).map_err(refused)?;
        let sham = resolved.is_sham().map_err(refused)?;
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay
            .stage_low_rank_columns_op(op.clone())
            .map_err(refused)?;
        let pending_structured_ops = overlay.structured_op_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::LowRankColumnsWeight(op.clone()));
        }
        let live = self.refresh_live_weights();
        Ok(StagedLowRank {
            op,
            changed_elements: resolved.changed_elements,
            sham,
            pending_structured_ops,
            live,
        })
    }

    /// `ADD LOW RANK`: the operand file is read and parsed, then staged by [`Self::stage_low_rank`].
    pub(crate) fn exec_add_low_rank(
        &mut self,
        operands: &str,
        tensor: &str,
        columns: &[usize],
        numerics: &str,
    ) -> Result<Vec<String>, LqlError> {
        vindex3_path(self)?;
        let profile = Profile::parse(Some(numerics)).map_err(refused)?;
        let text = std::fs::read_to_string(operands)
            .map_err(|e| refused(format!("cannot read operand file {operands:?}: {e}")))?;
        let file: LowRankFile = serde_json::from_str(&text)
            .map_err(|e| refused(format!("operand file {operands:?}: {e}")))?;
        let a = operand_bits(&file.a, profile.works_in_f64(), "a")?;
        let b = operand_bits(&file.b, profile.works_in_f64(), "b")?;
        let staged =
            self.stage_low_rank(tensor, columns, numerics, file.rank, &a, &b, file.origin)?;
        let mut out = vec![
            format!(
                "Staged: add rank-{} update to {tensor}, {} column(s) {}..={} under {numerics}; \
                 {} element(s) would change",
                file.rank,
                columns.len(),
                columns.first().copied().unwrap_or_default(),
                columns.last().copied().unwrap_or_default(),
                staged.changed_elements
            ),
            format!("  op digest {}", &staged.op.digest[..16]),
            if staged.sham {
                "  zero-valued (every term has a zero factor): a declared sham, compiled to the base's exact bytes".into()
            } else {
                "  results are BF16 as the profile defines them; an element whose terms all have a zero factor keeps its bits".into()
            },
            format!(
                "  {} structured op(s) pending COMPILE in this session",
                staged.pending_structured_ops
            ),
        ];
        out.extend(staged.live);
        Ok(out)
    }
}
