//! Model architecture implementations.
//!
//! Each file corresponds to a HuggingFace `model_type` value.
//! Every architecture implements [`ModelArchitecture`](crate::config::ModelArchitecture)
//! and returns its own `model_type` from `family()`.

pub mod deepseek;
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
pub mod generic;
pub mod gpt_oss;
pub mod granite;
pub mod llama;
pub mod mistral;
pub mod mixtral;
pub mod qwen;
pub mod qwen35;
pub mod starcoder2;
pub mod tinymodel;
