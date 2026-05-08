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
    #[error("invalid scratch dimensions: {0}")]
    InvalidScratch(String),
    #[error("scratch format {scratch:?} does not match requested format {requested:?}")]
    ScratchFormatMismatch {
        requested: super::KvFormat,
        scratch: super::KvFormat,
    },
    #[error("scratch head_dim {scratch} does not match requested head_dim {requested}")]
    ScratchHeadDimMismatch { requested: usize, scratch: usize },
    #[error("requested {requested} rows exceeds scratch capacity {capacity}")]
    ScratchCapacityExceeded { requested: usize, capacity: usize },
}
