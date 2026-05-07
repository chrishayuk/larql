pub mod proto {
    tonic::include_proto!("larql.grid.v1");
}

pub mod expert_proto {
    tonic::include_proto!("larql.expert.v1");
}

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

// ── PrefixBloom (wire-format helper) ───────────────────────────────────────
//
// 256-bit / 4-hash-position Bloom filter shared between
// `larql-server` (producer of `HeartbeatMsg.cached_prefixes`) and
// `larql-router` (consumer; mirror in `larql_router::grid::PrefixBloom`).
// We keep the canonical encoder here so both crates serialise to the
// exact same byte layout without depending on each other.

/// Wire-format bloom: 256 bits / 4 hash positions.
#[derive(Clone, Debug, Default)]
pub struct PrefixBloomEncoded {
    bits: [u64; 4],
}

impl PrefixBloomEncoded {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a prefix hash. Mixing matches `larql_router::grid`'s
    /// `PrefixBloom::positions` byte-for-byte.
    pub fn insert(&mut self, hash: u64) {
        for &pos in &positions(hash) {
            let word = (pos >> 6) as usize;
            let bit = (pos & 63) as u8;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// 32-byte raw payload (four little-endian u64s).
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, w) in self.bits.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

fn positions(h: u64) -> [u32; 4] {
    const M1: u64 = 0x9E37_79B9_7F4A_7C15;
    const M2: u64 = 0xBF58_476D_1CE4_E5B9;
    const M3: u64 = 0x94D0_49BB_1331_11EB;
    const M4: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mix = |seed: u64| -> u32 {
        let mut x = h.wrapping_mul(seed);
        x ^= x >> 33;
        x = x.wrapping_mul(M2);
        x ^= x >> 33;
        (x & 0xff) as u32
    };
    [mix(M1), mix(M2), mix(M3), mix(M4)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bloom_serialises_to_32_zero_bytes() {
        let b = PrefixBloomEncoded::new();
        assert_eq!(b.to_bytes(), [0u8; 32]);
    }

    #[test]
    fn insert_then_serialise_round_trips() {
        let mut a = PrefixBloomEncoded::new();
        a.insert(0xCAFE);
        a.insert(0xBEEF);
        let bytes = a.to_bytes();
        // Set bits aren't all zero.
        assert!(bytes.iter().any(|&b| b != 0));
        // Determinism: same inserts ⇒ same bytes.
        let mut b = PrefixBloomEncoded::new();
        b.insert(0xCAFE);
        b.insert(0xBEEF);
        assert_eq!(bytes, b.to_bytes());
    }
}
