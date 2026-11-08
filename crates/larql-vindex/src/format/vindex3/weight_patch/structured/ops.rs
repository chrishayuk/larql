//! The three ops, with the contract's check phases (SPEC §4) in order: parse, header, tensor bytes, compute,
//! fixups. An op either returns every output element or refuses with one code.

use super::arith::{exact_product_to_f32, Arith, Work};
use super::profile::Profile;
use super::refusal::{Refusal, RefusalCode};
use super::{
    BaseTensor, Fixup, Mutation, OpKind, OperandBits, ScalarBits, StructuredOp, TensorRole,
};
use crate::format::vindex3::weight_patch::bf16::is_finite as bf16_is_finite;

/// BF16 positive zero: what `zero_columns` writes.
const BF16_POSITIVE_ZERO: u16 = 0x0000;
const BF16_SIGN: u16 = 0x8000;

/// Run `op` on `base` under the contract's semantics.
pub fn run(op: &StructuredOp, base: &BaseTensor<'_>) -> Result<Vec<u16>, Refusal> {
    run_with(op, base, Mutation::None)
}

/// Run `op` on `base` under `mutation`. Anything but [`Mutation::None`] is a deliberately wrong semantics, for
/// tests that show the contract's choices change bytes.
pub fn run_with(
    op: &StructuredOp,
    base: &BaseTensor<'_>,
    mutation: Mutation,
) -> Result<Vec<u16>, Refusal> {
    let profile = match op.kind {
        OpKind::ZeroColumns { .. } => {
            if op.numerics.is_some() {
                return Err(refuse(
                    RefusalCode::NumericsForbidden,
                    "zero_columns has no arithmetic and no profile",
                ));
            }
            None
        }
        _ => Some(Profile::parse(op.numerics.as_deref())?),
    };
    let [rows, cols] = op.shape;
    if rows == 0 || cols == 0 {
        return Err(refuse(
            RefusalCode::ShapeInvalid,
            format!("declared shape {:?}", op.shape),
        ));
    }
    check_fixups_sorted(&op.fixups)?;
    let ar = Arith { mutation };
    match (&op.kind, profile) {
        (OpKind::ZeroColumns { columns }, _) => zero_columns(op, base, columns, mutation),
        (OpKind::RemoveDirection { u, alpha, columns }, Some(p)) => {
            check_columns_sorted(columns)?;
            if u.len() != rows {
                return Err(refuse(
                    RefusalCode::ShapeInvalid,
                    format!("u has {} values for {rows} rows", u.len()),
                ));
            }
            if p.works_in_f64() {
                remove_direction::<f64>(
                    op,
                    base,
                    p,
                    ar,
                    decode(u, p)?,
                    decode_scalar(*alpha, p)?,
                    columns,
                )
            } else {
                remove_direction::<f32>(
                    op,
                    base,
                    p,
                    ar,
                    decode(u, p)?,
                    decode_scalar(*alpha, p)?,
                    columns,
                )
            }
        }
        (OpKind::AddLowRank { rank, a, b }, Some(p)) => {
            let k = *rank;
            if k < 1 {
                return Err(refuse(RefusalCode::RankInvalid, "rank must be at least 1"));
            }
            if a.len() != rows * k || b.len() != k * cols {
                return Err(refuse(
                    RefusalCode::ShapeInvalid,
                    format!(
                        "a has {}, b has {}; rank {k} needs {} and {}",
                        a.len(),
                        b.len(),
                        rows * k,
                        k * cols
                    ),
                ));
            }
            if p.works_in_f64() {
                add_low_rank::<f64>(op, base, p, ar, k, decode(a, p)?, decode(b, p)?)
            } else {
                add_low_rank::<f32>(op, base, p, ar, k, decode(a, p)?, decode(b, p)?)
            }
        }
        (_, None) => unreachable!("only zero_columns has no profile"),
    }
}

fn refuse(code: RefusalCode, detail: impl Into<String>) -> Refusal {
    Refusal::new(code, detail)
}

/// Parse phase: fixups strictly increasing by (row, col), with finite bits.
fn check_fixups_sorted(fixups: &[Fixup]) -> Result<(), Refusal> {
    if fixups
        .windows(2)
        .any(|w| (w[0].row, w[0].col) >= (w[1].row, w[1].col))
    {
        return Err(refuse(
            RefusalCode::FixupsInvalid,
            "fixups must be strictly increasing by (row, col)",
        ));
    }
    if let Some(f) = fixups.iter().find(|f| !bf16_is_finite(f.bits)) {
        return Err(refuse(
            RefusalCode::NonFiniteOperand,
            format!("fixup ({}, {}) has bits {:#06x}", f.row, f.col, f.bits),
        ));
    }
    Ok(())
}

/// Parse phase: a non-empty, strictly increasing column set.
fn check_columns_sorted(columns: &[usize]) -> Result<(), Refusal> {
    if columns.is_empty() || columns.windows(2).any(|w| w[0] >= w[1]) {
        return Err(refuse(
            RefusalCode::ColumnsInvalid,
            "columns must be non-empty and strictly increasing",
        ));
    }
    Ok(())
}

/// Operands in the working format, all finite.
fn decode<W: Work>(bits: &OperandBits, p: Profile) -> Result<Vec<W>, Refusal> {
    let values: Vec<f64> = match (bits, W::IS_F64) {
        (OperandBits::F64(v), true) => v.iter().map(|&b| f64::from_bits(b)).collect(),
        (OperandBits::F32(v), false) => v.iter().map(|&b| f32::from_bits(b) as f64).collect(),
        _ => return Err(width_mismatch(p)),
    };
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(refuse(
            RefusalCode::NonFiniteOperand,
            format!("operand {i} is {}", values[i]),
        ));
    }
    Ok(values.into_iter().map(W::from_f64_exact).collect())
}

fn decode_scalar<W: Work>(bits: ScalarBits, p: Profile) -> Result<W, Refusal> {
    let v = match bits {
        ScalarBits::F64(b) => decode::<W>(&OperandBits::F64(vec![b]), p)?,
        ScalarBits::F32(b) => decode::<W>(&OperandBits::F32(vec![b]), p)?,
    };
    Ok(v[0])
}

fn width_mismatch(p: Profile) -> Refusal {
    let want = if p.works_in_f64() { "f64" } else { "f32" };
    refuse(
        RefusalCode::NumericsInvalid,
        format!("`{}` takes {want} operands", p.id()),
    )
}

/// Header phase: the base matches the declared shape, the tensor may be edited, the columns are in range, and
/// every fixup is inside the touched set. Returns `cols`.
fn check_header(
    op: &StructuredOp,
    base: &BaseTensor<'_>,
    columns: Option<&[usize]>,
    touched: impl Fn(usize, usize) -> bool,
) -> Result<usize, Refusal> {
    let [rows, cols] = op.shape;
    if base.shape != op.shape || base.bits.len() != rows * cols {
        return Err(refuse(
            RefusalCode::ShapeInvalid,
            format!(
                "declared {:?}; the base is {:?} with {} elements",
                op.shape,
                base.shape,
                base.bits.len()
            ),
        ));
    }
    if base.role == TensorRole::TokenEmbedding {
        return Err(refuse(
            RefusalCode::TensorRefused,
            "the token embedding doubles as the tied output head",
        ));
    }
    if let Some(&last) = columns.and_then(|c| c.last()) {
        if last >= cols {
            return Err(refuse(
                RefusalCode::ColumnsInvalid,
                format!("column {last} of a {cols}-column tensor"),
            ));
        }
    }
    if let Some(f) = op
        .fixups
        .iter()
        .find(|f| f.row >= rows || f.col >= cols || !touched(f.row, f.col))
    {
        return Err(refuse(
            RefusalCode::FixupsInvalid,
            format!("fixup ({}, {}) is outside the touched set", f.row, f.col),
        ));
    }
    Ok(cols)
}

/// Tensor-bytes phase: every base element the op reads is finite.
fn check_read_finite(
    base: &BaseTensor<'_>,
    read: impl Iterator<Item = usize>,
) -> Result<(), Refusal> {
    for i in read {
        if !bf16_is_finite(base.bits[i]) {
            return Err(refuse(
                RefusalCode::NonFiniteOperand,
                format!("base element {i} is {:#06x}", base.bits[i]),
            ));
        }
    }
    Ok(())
}

/// Fixups phase: refuse a fixup that changes nothing, then apply.
fn apply_fixups(out: &mut [u16], fixups: &[Fixup], cols: usize) -> Result<(), Refusal> {
    if let Some(f) = fixups.iter().find(|f| out[f.row * cols + f.col] == f.bits) {
        return Err(refuse(
            RefusalCode::FixupNoOp,
            format!("fixup ({}, {}) equals the computed bits", f.row, f.col),
        ));
    }
    for f in fixups {
        out[f.row * cols + f.col] = f.bits;
    }
    Ok(())
}

fn through_f32(p: Profile, mutation: Mutation) -> bool {
    match mutation {
        Mutation::SingleRounding => false,
        Mutation::DoubleRounding => true,
        _ => p.rounds_through_f32(),
    }
}

fn zero_columns(
    op: &StructuredOp,
    base: &BaseTensor<'_>,
    columns: &[usize],
    mutation: Mutation,
) -> Result<Vec<u16>, Refusal> {
    if !op.fixups.is_empty() {
        return Err(refuse(
            RefusalCode::FixupsForbidden,
            "zero_columns carries no fixups",
        ));
    }
    check_columns_sorted(columns)?;
    let cols = check_header(op, base, Some(columns), |_, _| false)?;
    let mut out = base.bits.to_vec();
    for row in out.chunks_exact_mut(cols) {
        for &j in columns {
            row[j] = if mutation == Mutation::ZeroKeepsSign {
                row[j] & BF16_SIGN
            } else {
                BF16_POSITIVE_ZERO
            };
        }
    }
    Ok(out)
}

/// `c[j] = Σ_r u[r]·w[r,j]`, from +0 — sequential in `r` unless a mutation says otherwise.
fn reduce<W: Work>(ar: Arith, u: &[W], w: &[W]) -> Result<W, Refusal> {
    match ar.mutation {
        Mutation::PairwiseReduction => {
            let products = u
                .iter()
                .zip(w)
                .map(|(&a, &b)| ar.mul(a, b))
                .collect::<Result<Vec<W>, _>>()?;
            pairwise(ar, &products)
        }
        Mutation::F32Accumulation if W::IS_F64 => {
            let mut acc = 0.0f32;
            for (&a, &b) in u.iter().zip(w) {
                let p = exact_product_to_f32(a.to_f64(), b.to_f64());
                acc = ar.add(acc, ar.result(p)?)?;
            }
            Ok(W::from_f64_exact(acc as f64))
        }
        _ => {
            let mut acc = W::zero();
            let terms: Box<dyn Iterator<Item = (&W, &W)>> =
                if ar.mutation == Mutation::ReversedReduction {
                    Box::new(u.iter().zip(w).rev())
                } else {
                    Box::new(u.iter().zip(w))
                };
            for (&a, &b) in terms {
                acc = if ar.mutation == Mutation::FmaInReduction {
                    ar.fused(a, b, acc)?
                } else {
                    ar.add(acc, ar.mul(a, b)?)?
                };
            }
            Ok(acc)
        }
    }
}

fn pairwise<W: Work>(ar: Arith, xs: &[W]) -> Result<W, Refusal> {
    if xs.len() == 1 {
        return ar.add(W::zero(), xs[0]);
    }
    let h = xs.len() / 2;
    ar.add(pairwise(ar, &xs[..h])?, pairwise(ar, &xs[h..])?)
}

#[allow(clippy::too_many_arguments)]
fn remove_direction<W: Work>(
    op: &StructuredOp,
    base: &BaseTensor<'_>,
    p: Profile,
    ar: Arith,
    u: Vec<W>,
    alpha: W,
    columns: &[usize],
) -> Result<Vec<u16>, Refusal> {
    let rows = op.shape[0];
    let cols = check_header(op, base, Some(columns), |_, c| {
        columns.binary_search(&c).is_ok()
    })?;
    check_read_finite(
        base,
        columns
            .iter()
            .flat_map(|&j| (0..rows).map(move |r| r * cols + j)),
    )?;
    let chain = through_f32(p, ar.mutation);
    let mut out = base.bits.to_vec();
    for &j in columns {
        let w: Vec<W> = (0..rows)
            .map(|r| W::widen_bf16(base.bits[r * cols + j]))
            .collect();
        let c = reduce(ar, &u, &w)?;
        for r in 0..rows {
            let t = if ar.mutation == Mutation::AlphaTimesUc {
                ar.mul(alpha, ar.mul(u[r], c)?)?
            } else {
                ar.mul(ar.mul(alpha, u[r])?, c)?
            };
            out[r * cols + j] = ar.to_bf16(ar.sub(w[r], t)?, chain)?;
        }
    }
    apply_fixups(&mut out, &op.fixups, cols)?;
    Ok(out)
}

fn add_low_rank<W: Work>(
    op: &StructuredOp,
    base: &BaseTensor<'_>,
    p: Profile,
    ar: Arith,
    k: usize,
    a: Vec<W>,
    b: Vec<W>,
) -> Result<Vec<u16>, Refusal> {
    let rows = op.shape[0];
    let cols = check_header(op, base, None, |_, _| true)?;
    check_read_finite(base, 0..rows * cols)?;
    let chain = through_f32(p, ar.mutation);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for j in 0..cols {
            let mut w = W::widen_bf16(base.bits[r * cols + j]);
            if ar.mutation == Mutation::LowRankFromZero {
                let mut s = W::zero();
                for kk in 0..k {
                    s = ar.add(s, ar.mul(a[r * k + kk], b[kk * cols + j])?)?;
                }
                w = ar.add(w, s)?;
            } else {
                for kk in 0..k {
                    w = ar.add(w, ar.mul(a[r * k + kk], b[kk * cols + j])?)?;
                }
            }
            out.push(ar.to_bf16(w, chain)?);
        }
    }
    apply_fixups(&mut out, &op.fixups, cols)?;
    Ok(out)
}
