pub mod architectures;
pub mod config;
pub mod connectors;
pub mod defaults;
pub mod detect;
pub mod encoders;
pub mod inventory;
pub mod loading;
pub mod multimodal;
pub mod quant;
pub mod speech;
pub(crate) mod tensor_keys;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_fixtures;
pub mod validation;
pub mod vectors;
pub mod weights;

pub use config::{
    architecture_prelude, ArchitectureCore, Attention, Embeddings, FeedForward, LatentAttention,
    Norms, Position, TensorKeys,
};
pub use config::{
    Activation, ExpertFormat, ExpertGatePolicy, ExpertRoutingPolicy, FfnType, GateUpBranch,
    GateUpLayout, Llama3RopeScaling, ModelArchitecture, ModelConfig, MoeRouterKind, NormType,
    QkNormScope, RopeScaling, YarnRopeScaling, LAYER_TYPE_FULL_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION, LAYER_TYPE_WINDOW_ATTENTION, ROPE_TYPE_DEFAULT, ROPE_TYPE_LINEAR,
    ROPE_TYPE_LLAMA3, ROPE_TYPE_YARN,
};
pub use detect::{
    detect_architecture, detect_architecture_validated, detect_from_json,
    detect_from_json_validated, ModelError,
};
pub use multimodal::{
    Connector as MmConnector, ModalEncoder, ModalInput, Modality, MultiModalProtocol,
    PlaceholderProtocol, PrecomputedScaling, TokenBudget,
};
pub use validation::{ConfigValidationError, ConfigValidationResult};

pub use architectures::deepseek::DeepSeekArch;
pub use architectures::gemma2::Gemma2Arch;
pub use architectures::gemma3::Gemma3Arch;
pub use architectures::gemma4::Gemma4Arch;
pub use architectures::generic::GenericArch;
pub use architectures::gpt2::Gpt2Arch;
pub use architectures::gpt_neo::GptNeoArch;
pub use architectures::gpt_oss::GptOssArch;
pub use architectures::granite::GraniteArch;
pub use architectures::llama::LlamaArch;
pub use architectures::mistral::MistralArch;
pub use architectures::mixtral::MixtralArch;
pub use architectures::moss_tts_realtime::MossTtsRealtimeArch;
pub use architectures::qwen::QwenArch;
pub use architectures::starcoder2::StarCoder2Arch;
pub use architectures::tinymodel::TinyModelArch;

pub use vectors::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};
pub use weights::{DequantScratch, ModelWeights, WeightArray, WeightsView};

pub use loading::{
    is_ffn_tensor, load_gguf, load_gguf_validated, load_model_dir, load_model_dir_filtered,
    load_model_dir_filtered_validated, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, resolve_model_path, I2S_SCALE_SUFFIX,
};
