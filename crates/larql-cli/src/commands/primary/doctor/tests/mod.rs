//! Gates for `larql doctor` — every check's decision and the report's
//! rendering, from facts built here rather than read off this machine.

mod auth;
mod build_info;
mod caches;
mod disk;
mod evaluate;
mod network;
mod platform;
mod report;
mod server;
mod units;

use std::path::PathBuf;

use super::build_info::BuildFacts;
use super::caches::{CacheDir, CacheFacts, DirFacts};
use super::network::{HubReply, Probe};
use super::platform::{CpuFeature, MetalProbe, PlatformFacts};
use super::report::{Check, Status};
use super::server::{ServerFacts, ServerLocation};
use super::Facts;

/// A gigabyte, decimal, as the requirements are stated.
const GB: u64 = 1_000_000_000;

pub(super) fn home() -> PathBuf {
    PathBuf::from("/home/ada")
}

pub(super) fn dir(path: &str, free: Option<u64>) -> DirFacts {
    DirFacts {
        path: home().join(path),
        exists: true,
        writable: true,
        size_bytes: Some(2 * GB),
        free_bytes: free,
    }
}

pub(super) fn healthy_caches() -> CacheFacts {
    CacheFacts {
        hub: CacheDir {
            facts: Ok(dir(".cache/huggingface/hub", Some(100 * GB))),
            env_override: None,
        },
        encoded: CacheDir {
            facts: Ok(dir(".cache/larql/encoded", Some(100 * GB))),
            env_override: None,
        },
        home: Some(home()),
    }
}

/// A machine on which every check passes.
pub(super) fn healthy() -> Facts {
    Facts {
        build: BuildFacts {
            version: "0.2.1".into(),
            target: "aarch64-apple-darwin".into(),
            metal: true,
            research: false,
        },
        platform: PlatformFacts {
            os: "macos".into(),
            arch: "aarch64".into(),
            cpu_threads: Some(8),
            cpu_features: vec![
                CpuFeature {
                    name: "neon",
                    present: true,
                },
                CpuFeature {
                    name: "dotprod",
                    present: true,
                },
            ],
            kernels: "q4k_matvec=neon".into(),
            ram_bytes: Some(16 * GB),
            metal: MetalProbe::Available("metal (GPU)".into()),
        },
        server: ServerFacts {
            location: ServerLocation::Sibling(home().join("bin/larql-server")),
            version: Some(Ok("0.2.1".into())),
            expected_version: "0.2.1".into(),
            home: Some(home()),
        },
        caches: healthy_caches(),
        token_source: Some("$HF_TOKEN".into()),
        hub_endpoint: "https://huggingface.co".into(),
        hub: Probe::Done(HubReply {
            status: 200,
            millis: 42,
        }),
        release: Probe::Done("v0.2.1".into()),
    }
}

/// The single check a group function returned.
pub(super) fn only(mut checks: Vec<Check>) -> Check {
    assert_eq!(checks.len(), 1, "{checks:?}");
    checks.remove(0)
}

pub(super) fn statuses(checks: &[Check]) -> Vec<Status> {
    checks.iter().map(|c| c.status).collect()
}
