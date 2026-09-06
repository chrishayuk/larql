//! Checking a certificate instead of believing it.
//!
//! An [`ExtentCertificate`](super::ExtentCertificate) is a declaration: a
//! codec states what its reconstruction costs and what it is worth. Nothing
//! made the second half true. A codec that declares a radius it does not
//! meet is indistinguishable, to a planner selecting on fidelity, from one
//! that does — and selection under a quality requirement is exactly what
//! extents exist for, so the declaration has to be checkable.
//!
//! This measures a codec's extents against a SOURCE — the values the
//! encoding was made from, not another decode of the same bytes — and
//! refuses three ways:
//!
//! ```text
//! CertificateViolated    measured error exceeds the declared radius
//! CertificateNotMonotone a deeper extent reconstructs worse than a shallower one
//! TerminalNotExact       the deepest extent is not the source, bit for bit
//! ```
//!
//! The measurement is over the CERTIFIED DOMAIN: finite normal values. A
//! relative bound says nothing about a subnormal that truncates to zero, an
//! infinity, or a NaN, and averaging those in would produce a number that
//! is neither the declaration's subject nor anything else. What happens to
//! them is a property each codec states for itself, and
//! [`Report::uncertified_elements`] says how many of the source's values
//! were outside the domain so a caller can see the measurement's reach.

use super::error::CodecError;
use super::extent::RepresentationExtent;
use super::fidelity::FidelityCertificate;
use super::streams::CodecOperands;
use super::RepresentationCodec;

/// What one extent decoded to, measured against the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtentReading {
    pub extent: RepresentationExtent,
    /// Declared by the certificate, where it declares one.
    pub declared_relative_rms: Option<f64>,
    /// Measured over finite normal source values; `0.0` when there are
    /// none to measure.
    pub measured_relative_rms: f64,
    /// Whether every element — the uncertified ones included — came back
    /// with the source's exact bit pattern.
    pub bit_exact: bool,
}

/// Every extent's reading, in declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub readings: Vec<ExtentReading>,
    /// Source values outside the certified domain: subnormals, zeroes,
    /// infinities and NaNs.
    pub uncertified_elements: usize,
    pub elements: usize,
}

impl Report {
    /// The reading at `depth`, if the codec declared that extent.
    pub fn at_depth(&self, depth: u32) -> Option<ExtentReading> {
        self.readings
            .iter()
            .find(|r| r.extent.depth == depth)
            .copied()
    }
}

/// Whether a source value is what a relative radius can describe.
fn is_certified_domain(value: f32) -> bool {
    value.is_normal()
}

/// Decode `operands` at every extent `codec` declares and hold the result
/// against `source`, refusing a certificate the decode does not honour.
///
/// `source` is the image the operands were encoded from — a foreign
/// reference, not another decode of the same bytes, because a codec
/// checked against itself agrees with itself.
pub fn certify(
    codec: &dyn RepresentationCodec,
    operands: &CodecOperands<'_>,
    shape: &[usize],
    source: &[f32],
    tensor: &str,
) -> Result<Report, CodecError> {
    let label = codec.encoding_label();
    let certified: Vec<usize> = (0..source.len())
        .filter(|i| is_certified_domain(source[*i]))
        .collect();
    let terminal = codec.terminal_extent();
    let mut readings: Vec<ExtentReading> = Vec::new();
    for certificate in codec.extents() {
        let decoded = codec.decode_all(operands, shape, certificate.extent, tensor)?;
        if decoded.len() != source.len() {
            return Err(CodecError::Destination {
                tensor: tensor.into(),
                need: source.len(),
                have: decoded.len(),
            });
        }
        let measured = relative_rms(source, &decoded, &certified);
        let bit_exact = source
            .iter()
            .zip(&decoded)
            .all(|(s, d)| s.to_bits() == d.to_bits());
        if let Some(radius) = &certificate.radius {
            if measured > radius.radius() {
                return Err(CodecError::CertificateViolated {
                    tensor: tensor.into(),
                    label: label.into(),
                    depth: certificate.extent.depth,
                    declared: radius.radius(),
                    measured,
                });
            }
        }
        if let Some(before) = readings.last() {
            if measured > before.measured_relative_rms {
                return Err(CodecError::CertificateNotMonotone {
                    tensor: tensor.into(),
                    label: label.into(),
                    depth: certificate.extent.depth,
                    shallower: before.extent.depth,
                    measured,
                    before: before.measured_relative_rms,
                });
            }
        }
        if certificate.extent == terminal && !bit_exact {
            let differing = source
                .iter()
                .zip(&decoded)
                .filter(|(s, d)| s.to_bits() != d.to_bits())
                .count();
            return Err(CodecError::TerminalNotExact {
                tensor: tensor.into(),
                label: label.into(),
                depth: certificate.extent.depth,
                differing,
                elements: source.len(),
            });
        }
        readings.push(ExtentReading {
            extent: certificate.extent,
            declared_relative_rms: certificate.radius.as_ref().map(FidelityCertificate::radius),
            measured_relative_rms: measured,
            bit_exact,
        });
    }
    Ok(Report {
        uncertified_elements: source.len() - certified.len(),
        elements: source.len(),
        readings,
    })
}

/// RMS of `|decoded - source| / |source|` over `indices`.
///
/// Normalised per element, because a tensor's values span decades and an
/// absolute residual would read as a statement about the largest of them
/// rather than about the encoding. Accumulated in f64, so the measurement
/// is finer than what it measures.
fn relative_rms(source: &[f32], decoded: &[f32], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f64 = indices
        .iter()
        .map(|&i| {
            let s = f64::from(source[i]);
            let residual = (f64::from(decoded[i]) - s) / s;
            residual * residual
        })
        .sum();
    (sum / indices.len() as f64).sqrt()
}
