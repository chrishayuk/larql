//! Model architecture implementations.
//!
//! Each file corresponds to a HuggingFace `model_type` value.
//! Every architecture implements [`ModelArchitecture`](crate::config::ModelArchitecture)
//! and returns its own `model_type` from `family()`.

pub mod bitnet;
pub mod deepseek;
pub mod deepseek_v4;
pub mod exaone4;
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
pub mod generic;
pub mod gpt2;
pub mod gpt_oss;
pub mod granite;
pub mod kimi;
pub mod llama;
pub mod mamba2;
pub mod mistral;
pub mod mixtral;
pub mod moss_tts_realtime;
pub mod muse_glimmer;
pub mod olmo2;
pub mod olmoe;
pub mod qwen;
pub mod starcoder2;
pub mod tinymodel;
