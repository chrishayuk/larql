pub mod proto {
    tonic::include_proto!("larql.grid.v1");
}

pub mod expert_proto {
    tonic::include_proto!("larql.expert.v1");
}

pub mod shard_proto {
    tonic::include_proto!("larql.shard.v1");
}

#[cfg(feature = "quic")]
pub mod transport;

pub use expert_proto::expert_service_client::ExpertServiceClient;
pub use expert_proto::expert_service_server::{ExpertService, ExpertServiceServer};
pub use expert_proto::{
    ExpertBatchItem, ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertLayerInput,
    ExpertLayerOutput,
};
pub use proto::grid_service_client::GridServiceClient;
pub use proto::grid_service_server::{GridService, GridServiceServer};
pub use proto::router_message::Payload as RouterPayload;
pub use proto::server_message::Payload as ServerPayload;
pub use proto::*;
pub use shard_proto::shard_service_client::ShardServiceClient;
pub use shard_proto::shard_service_server::{ShardService, ShardServiceServer};
pub use shard_proto::{ShardQuery, ShardResult};
