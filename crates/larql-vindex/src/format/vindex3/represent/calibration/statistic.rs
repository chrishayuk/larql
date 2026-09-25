use super::{refused, StatisticKind};
use crate::error::VindexError;

pub(super) struct Accumulator {
    kind: StatisticKind,
    width: usize,
    pub values: Vec<f64>,
    pub samples: u64,
}

impl Accumulator {
    pub fn new(kind: StatisticKind, width: usize) -> Result<Self, VindexError> {
        let n = kind.elements(width)?;
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(refused)?;
        values.resize(n, 0.0);
        Ok(Self {
            kind,
            width,
            values,
            samples: 0,
        })
    }

    pub fn push(&mut self, x: &[f32]) -> Result<(), VindexError> {
        if x.len() != self.width || x.iter().any(|v| !v.is_finite()) {
            return Err(refused(
                "captured input has wrong width or non-finite values",
            ));
        }
        match self.kind {
            StatisticKind::DiagonalSecondMoment => {
                for (h, &x) in self.values.iter_mut().zip(x) {
                    *h += f64::from(x) * f64::from(x);
                }
            }
            StatisticKind::DenseGram => {
                for (i, &a) in x.iter().enumerate() {
                    for (j, &b) in x.iter().enumerate().take(i + 1) {
                        let next = self.values[i * self.width + j] + f64::from(a) * f64::from(b);
                        self.values[i * self.width + j] = next;
                        self.values[j * self.width + i] = next;
                    }
                }
            }
        }
        self.samples = self
            .samples
            .checked_add(1)
            .ok_or_else(|| refused("sample count overflow"))?;
        Ok(())
    }
}

pub(super) fn validate_values(
    kind: StatisticKind,
    width: usize,
    values: &[f64],
) -> Result<(), VindexError> {
    if values.len() != kind.elements(width)? || values.iter().any(|v| !v.is_finite()) {
        return Err(refused("statistic shape or finite f64 validation failed"));
    }
    match kind {
        StatisticKind::DiagonalSecondMoment if values.iter().any(|v| *v < 0.0) => {
            return Err(refused("negative second moment"));
        }
        StatisticKind::DenseGram => {
            for i in 0..width {
                if values[i * width + i] < 0.0 {
                    return Err(refused("negative Gram diagonal"));
                }
                for j in 0..i {
                    if values[i * width + j] != values[j * width + i] {
                        return Err(refused("asymmetric Gram"));
                    }
                    if (values[i * width + i] == 0.0 || values[j * width + j] == 0.0)
                        && values[i * width + j] != 0.0
                    {
                        return Err(refused("nonzero cross term for dead coordinate"));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
