//! larql-server library — shared between the binary and integration tests.

// tonic::Status is a fat error type (176 bytes). It's our external contract
// for all gRPC handlers, so flipping to Box<Status> is not worth the churn.
#![allow(clippy::result_large_err)]

pub mod announce;
pub mod attention_session;
pub mod auth;
pub mod band_utils;
pub mod bootstrap;
pub mod cache;
pub mod embed_store;
pub mod env_flags;
pub mod error;
pub mod etag;
pub mod ffn_l2_cache;
pub mod grpc;
pub mod grpc_attention;
pub mod grpc_expert;
pub mod http;
pub mod kv_snapshot;
pub mod ratelimit;
pub mod routes;
pub mod session;
pub mod state;
pub mod tokenizer_cache;
pub mod wire;
