use thiserror::Error;

#[derive(Debug, Error)]
pub enum RotorQuantError {
    #[error("head_dim {head_dim} not divisible by block size {block_size} for {format:?}")]
    HeadDimNotDivisible {
        format: super::KvFormat,
        head_dim: usize,
        block_size: usize,
    },
    #[error("input length {got} != n_rows ({n_rows}) * head_dim ({head_dim}) = {expected}")]
    InputLengthMismatch {
        got: usize,
        n_rows: usize,
        head_dim: usize,
        expected: usize,
    },
    #[error("invalid quantised buffer: {0}")]
    InvalidBuffer(String),
}
