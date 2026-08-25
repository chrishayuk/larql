//! What a projection actually COMPUTES, as a value.
//!
//! Until CPU-5 a projection was described by its weight format alone, and
//! that was sufficient because it was true: bf16 bytes meant a widening
//! f32 kernel, Q8 bytes meant a widening f32 kernel, and the residency
//! determined the arithmetic. Integer activations end that. The same Q8
//! bytes are consumed by
//!
//! ```text
//! Q8 -> widen -> F32 GEMV                    (FusedQ8)
//! Q8 x Q8[64] -> I32 -> scale -> F32         (Q8xQ8)
//! ```
//!
//! which are different physical operators over identical stored weights,
//! with different cost (83.4 vs 118.0 GB/s) and different numerics. So
//! the invariant this module exists to state is:
//!
//! > **A resident representation CONSTRAINS which plans are possible. It
//! > does not DETERMINE which plan executes.**
//!
//! Anything that infers arithmetic from residency is reintroducing the
//! assumption this file was written to remove.
//!
//! ## Why the activation carries geometry
//!
//! `ActivationRep::Q8` alone would be a lie by omission. CPU-5's A1
//! control measured exact bf16 weights against a per-TENSOR int8
//! activation at rel_rms 4.8e-01 — a destroyed activation, not a
//! quantisation cost — because the residual stream's peak is ~30x its RMS
//! at depth and one scale over that vector leaves a typical element about
//! two bits. Blocking the same int8 activation on the weights' own
//! boundaries took it to 4.7e-02. Those are not two settings of one
//! representation; they are two representations, and only one of them is
//! usable. The span is therefore part of the type.

use super::physical::PhysicalProjectionPlan;
use std::fmt;

/// What a projection multiplies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightRep {
    /// Widened f32 — the constitutional representation.
    F32,
    /// Stored bf16 code units, EXACT: bf16 is the top half of the f32 it
    /// denotes, so widening rounds nothing.
    Bf16,
    /// Symmetric int8, one f32 scale per `block` elements. Step
    /// `peak / 127`.
    Q8 { block: usize },
    /// Symmetric int4, two codes per byte, one f32 scale per `block`
    /// elements. Step `peak / 7` — **18.1x Q8's at the same block**, and
    /// that ratio is the whole numerical story of the format.
    Q4 { block: usize },
}

/// Over how many elements ONE activation scale applies.
///
/// The distinction is not a tuning knob: see the module note.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleSpan {
    /// One scale for the whole vector. Adequate only where the activation
    /// has no heavy tail, which a transformer residual stream at depth
    /// emphatically does.
    Tensor,
    /// One scale per `n` elements along the input axis, on the weights'
    /// own block boundaries so the two fold into a single multiply.
    Block(usize),
}

/// What a projection multiplies BY.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActivationRep {
    /// The activation reaches the kernel unquantised.
    F32,
    /// Symmetric int8 at a stated scale geometry.
    Q8 { span: ScaleSpan },
}

/// Where the products land before they are scaled back to f32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccumulatorRep {
    /// Float accumulation, rounding at every add.
    F32,
    /// **Exact** integer accumulation within a block; the only rounding
    /// is the one multiply-add per block that scales it back.
    I32,
}

/// The complete arithmetic of one projection.
///
/// Produced BY a plan and never chosen independently of one — the whole
/// point of [`PhysicalProjectionPlan`] being an enum of realisable
/// combinations is that a caller cannot assemble a triple no kernel
/// implements.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Arithmetic {
    pub weight: WeightRep,
    pub activation: ActivationRep,
    pub accumulator: AccumulatorRep,
}

impl fmt::Display for WeightRep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::Bf16 => write!(f, "BF16"),
            Self::Q8 { block } => write!(f, "Q8[{block}]"),
            Self::Q4 { block } => write!(f, "Q4[{block}]"),
        }
    }
}

impl fmt::Display for ActivationRep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::Q8 {
                span: ScaleSpan::Tensor,
            } => write!(f, "Q8[tensor]"),
            Self::Q8 {
                span: ScaleSpan::Block(n),
            } => write!(f, "Q8[{n}]"),
        }
    }
}

impl fmt::Display for AccumulatorRep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::I32 => write!(f, "I32"),
        }
    }
}

impl fmt::Display for Arithmetic {
    /// `Q4[64] x Q8[64] -> I32 -> F32` — the form a claim should quote,
    /// because every term in it changes the answer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} x {} -> {} -> F32",
            self.weight, self.activation, self.accumulator
        )
    }
}

/// Which plans a given resident representation MAKES POSSIBLE.
///
/// Not which one runs — that is [`PhysicalProjectionPlan::for_resident`],
/// and it needs the process's arithmetic arm to answer. This is the
/// constraint, and the gap between the two is the invariant: where this
/// returns more than one plan, residency has genuinely not determined
/// execution.
pub fn plans_possible_for(rep: WeightRep) -> &'static [PhysicalProjectionPlan] {
    match rep {
        // The oracle is reachable only by a backend that declares it,
        // never by observation — a reference path that could be selected
        // off the bytes would stop being a reference.
        WeightRep::F32 => &[PhysicalProjectionPlan::BlasF32],
        WeightRep::Bf16 => &[
            PhysicalProjectionPlan::FusedBf16,
            PhysicalProjectionPlan::Bf16xQ8,
        ],
        WeightRep::Q8 { .. } => &[
            PhysicalProjectionPlan::FusedQ8,
            PhysicalProjectionPlan::Q8xQ8,
        ],
        WeightRep::Q4 { .. } => &[
            PhysicalProjectionPlan::FusedQ4,
            PhysicalProjectionPlan::Q4xQ8,
        ],
    }
}
