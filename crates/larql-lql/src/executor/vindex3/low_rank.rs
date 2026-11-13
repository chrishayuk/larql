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

impl Session {
    /// `ADD LOW RANK`: everything checkable is checked before anything is staged: the operand file, the
    /// profile, the tensor, the columns, and that the op, resolved by the frozen interpreter, edits at least
    /// one element.
    pub(crate) fn exec_add_low_rank(
        &mut self,
        operands: &str,
        tensor: &str,
        columns: &[usize],
        numerics: &str,
    ) -> Result<Vec<String>, LqlError> {
        let path = match &self.backend {
            Backend::Vindex3 { path, .. } => path.clone(),
            Backend::Vindex { .. } => {
                return Err(LqlError::Execution(
                    "ADD LOW RANK edits the stored weights of a VINDEX3 container; \
                     this session is bound to a VINDEX2 vindex"
                        .into(),
                ))
            }
            _ => return Err(LqlError::NoBackend),
        };
        let profile = Profile::parse(Some(numerics)).map_err(refused)?;
        let text = std::fs::read_to_string(operands)
            .map_err(|e| refused(format!("cannot read operand file {operands:?}: {e}")))?;
        let file: LowRankFile = serde_json::from_str(&text)
            .map_err(|e| refused(format!("operand file {operands:?}: {e}")))?;
        let a = operand_bits(&file.a, profile.works_in_f64(), "a")?;
        let b = operand_bits(&file.b, profile.works_in_f64(), "b")?;
        let mut op =
            LowRankColumnsOp::bind(&path, tensor, numerics, file.rank, &a, &b, columns.to_vec())
                .map_err(refused)?;
        if let Some(origin) = file.origin {
            op = op.with_origin(origin);
        }
        // Resolve now, so a bad op is refused when it is written down and not when it is compiled.
        let resolved = op.resolve(&path).map_err(refused)?;
        let digest = op.digest.clone();
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay
            .stage_low_rank_columns_op(op.clone())
            .map_err(refused)?;
        let pending = overlay.structured_op_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::LowRankColumnsWeight(op));
        }
        Ok(vec![
            format!(
                "Staged: add rank-{} update to {tensor}, {} column(s) {}..={} under {numerics}; \
                 {} element(s) would change",
                file.rank,
                columns.len(),
                columns.first().copied().unwrap_or_default(),
                columns.last().copied().unwrap_or_default(),
                resolved.edits.len()
            ),
            format!("  op digest {}", &digest[..16]),
            "  results are BF16 as the profile defines them; an element whose terms all have a zero factor keeps its bits".into(),
            format!(
                "  {pending} structured op(s) pending in this session; takes effect at COMPILE CURRENT INTO VINDEX"
            ),
        ])
    }
}
