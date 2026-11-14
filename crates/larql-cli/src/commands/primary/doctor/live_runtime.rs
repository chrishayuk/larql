//! The production wiring: gather [`Facts`] from this machine, then print
//! the checklist.
//!
//! Thin by design — every decision lives in the check modules, which take
//! these facts as parameters and are tested without the machine. What is
//! here is the OS, the network and a child process.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use larql_compute::BackendKind;

use super::super::cache;
use super::super::hf_auto::EncodedStore;
use super::build_info::BuildFacts;
use super::caches::{self, CacheDir, CacheFacts};
use super::network::{self, HubReply, Probe};
use super::platform::{self, MetalProbe, PlatformFacts};
use super::server::{self, ServerFacts};
use super::{evaluate, DoctorArgs, Facts};

/// How long `larql-server --version` may take before it is abandoned.
const PROCESS_TIMEOUT: Duration = Duration::from_secs(5);
/// How often a running `--version` is polled.
const POLL_INTERVAL: Duration = Duration::from_millis(20);
/// Env var moving the HF cache root, as `cache::hf_hub_dir` reads it.
const HF_HOME_ENV: &str = "HF_HOME";
/// GitHub's API refuses requests without a user agent.
const USER_AGENT: &str = concat!("larql-doctor/", env!("CARGO_PKG_VERSION"));

pub fn run(args: DoctorArgs) -> Result<(), Box<dyn std::error::Error>> {
    let facts = gather(args.offline);
    let report = evaluate(&facts);
    if args.json {
        let json = report.to_json(&facts.build.version);
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        print!("{}", report.render_text());
    }
    Ok(report.outcome()?)
}

fn gather(offline: bool) -> Facts {
    let build = BuildFacts::current();
    let home = home_dir();
    let hub_endpoint = network::hub_endpoint(std::env::var(network::HF_ENDPOINT_ENV).ok());
    let (hub, release) = if offline {
        (Probe::Skipped, Probe::Skipped)
    } else {
        (probe_hub(&hub_endpoint), probe_release())
    };
    Facts {
        platform: platform_facts(),
        server: server_facts(&build.version, home.clone()),
        caches: cache_facts(home.clone()),
        token_source: larql_vindex::format::huggingface::token_source()
            .map(|s| super::auth::describe_source(&s, home.as_deref())),
        hub_endpoint,
        hub,
        release,
        build,
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn platform_facts() -> PlatformFacts {
    PlatformFacts {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_threads: std::thread::available_parallelism().ok().map(|n| n.get()),
        cpu_features: platform::detect_cpu_features(),
        kernels: larql_compute::cpu::ops::q4k_q8k_dot::kernel_class_summary(),
        ram_bytes: larql_vindex::format::vindex3::opplan::exec::accounting::physical_memory_bytes(),
        metal: metal_probe(),
    }
}

/// Ask the backend registry `run --backend metal` selects from.
fn metal_probe() -> MetalProbe {
    if !cfg!(target_os = "macos") {
        return MetalProbe::NotApplicable;
    }
    if !cfg!(feature = "gpu") {
        return MetalProbe::NotCompiled;
    }
    match crate::backend_select::backend_for_kind(BackendKind::Metal) {
        Ok(backend) => MetalProbe::Available(backend.name().to_string()),
        Err(e) => MetalProbe::Unavailable(e.to_string()),
    }
}

fn server_facts(expected_version: &str, home: Option<PathBuf>) -> ServerFacts {
    let exe = std::env::current_exe().ok();
    let path_var = std::env::var_os("PATH");
    let location = server::locate(exe.as_deref(), path_var.as_deref());
    let version = match &location {
        server::ServerLocation::Sibling(p) | server::ServerLocation::OnPath(p) => {
            Some(server_version(p))
        }
        server::ServerLocation::Missing => None,
    };
    ServerFacts {
        location,
        version,
        expected_version: expected_version.to_string(),
        home,
    }
}

/// Run `<server> --version`, abandoning it after [`PROCESS_TIMEOUT`].
fn server_version(path: &Path) -> Result<String, String> {
    let mut child = Command::new(path)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + PROCESS_TIMEOUT;
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("no answer within {}s", PROCESS_TIMEOUT.as_secs()));
            }
            None => std::thread::sleep(POLL_INTERVAL),
        }
    };
    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        out.read_to_string(&mut stdout).map_err(|e| e.to_string())?;
    }
    if !status.success() {
        return Err(format!("exited with {status}"));
    }
    server::parse_version(&stdout).ok_or_else(|| format!("unrecognised answer {stdout:?}"))
}

fn cache_facts(home: Option<PathBuf>) -> CacheFacts {
    let probe = |path: PathBuf| caches::probe_dir(&path, super::disk::free_bytes);
    CacheFacts {
        hub: CacheDir {
            facts: cache::hf_hub_dir().map(probe).map_err(|e| e.to_string()),
            env_override: std::env::var_os(HF_HOME_ENV).map(|_| HF_HOME_ENV),
        },
        // The store `run hf://…` encodes into and `list` scans.
        encoded: CacheDir {
            facts: EncodedStore::open_default()
                .map(|store| probe(store.root().to_path_buf()))
                .map_err(|e| e.to_string()),
            env_override: std::env::var_os(cache::LARQL_HOME_ENV).map(|_| cache::LARQL_HOME_ENV),
        },
        home,
    }
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(network::NETWORK_TIMEOUT)
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| e.to_string())
}

fn probe_hub(endpoint: &str) -> Probe<HubReply> {
    let started = Instant::now();
    let reply = client().and_then(|c| {
        c.get(network::hub_probe_url(endpoint))
            .send()
            .map_err(|e| e.to_string())
    });
    match reply {
        Ok(r) => Probe::Done(HubReply {
            status: r.status().as_u16(),
            millis: started.elapsed().as_millis(),
        }),
        Err(why) => Probe::Failed(why),
    }
}

fn probe_release() -> Probe<String> {
    let tag = client()
        .and_then(|c| {
            c.get(network::LATEST_RELEASE_API)
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.json::<serde_json::Value>())
                .map_err(|e| e.to_string())
        })
        .and_then(|body| {
            body["tag_name"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| "no tag_name in the reply".to_string())
        });
    match tag {
        Ok(tag) => Probe::Done(tag),
        Err(why) => Probe::Failed(why),
    }
}
