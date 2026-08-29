//! The PUBLIC_EXPLORER LQL bridge — one session, one thread, one gate.
//!
//! `POST /v1/query` executes real LQL against the served container, so
//! the statement path must end at the same seam every other transport
//! ends at: `larql_lql::Session::execute`, where the capability
//! profile is judged after parsing and before execution. This module
//! owns that session.
//!
//! Why a dedicated thread and not shared state: an LQL `Session` is a
//! deliberately single-threaded object (interior caches, a bound
//! runtime), and wrapping it in a lock would put a `!Sync` question
//! and a poisoning story inside every handler. One OS thread owns the
//! session; handlers send `(statement, reply)` jobs over a bounded
//! channel and await the oneshot. Queries serialize — correct for a
//! public endpoint that is rate-limited anyway, and the bounded queue
//! is itself back-pressure.
//!
//! What is deliberately not here: any second filter over statements.
//! The bridge does not inspect what it forwards. If a statement must
//! not run, the profile refuses it inside the session — a bridge-side
//! allowlist would be a divergent copy of that judgement.

use std::path::Path;
use std::time::Duration;

use larql_lql::{parse, CapabilityProfile, LqlError, Session};
use tracing::info;

/// How a query failed — carries the HTTP mapping the handler needs
/// without the handler matching on `LqlError` itself.
#[derive(Debug)]
pub enum QueryFailure {
    /// The statement did not parse. 400.
    Parse(String),
    /// The capability profile declined it — nothing executed. 403.
    Refused(String),
    /// Execution failed (bad address, missing capability, …). 422.
    Execution(String),
    /// The bridge thread is gone or the reply never came. 503.
    Bridge(String),
}

struct Job {
    statement: String,
    reply: tokio::sync::oneshot::Sender<Result<Vec<String>, QueryFailure>>,
}

/// Handle to the session-owning thread. Cheap to clone into handlers.
pub struct LqlBridge {
    tx: tokio::sync::mpsc::Sender<Job>,
    timeout: Duration,
    /// What the banner and every response name.
    pub profile: &'static str,
}

impl LqlBridge {
    /// Execute one statement on the bridged session.
    pub async fn query(&self, statement: String) -> Result<Vec<String>, QueryFailure> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Job { statement, reply })
            .await
            .map_err(|_| QueryFailure::Bridge("the query session is gone".into()))?;
        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(QueryFailure::Bridge(
                "the query session dropped the job".into(),
            )),
            Err(_) => Err(QueryFailure::Bridge(format!(
                "query timed out after {}s",
                self.timeout.as_secs()
            ))),
        }
    }
}

/// Spawn the session thread: bind `container` under the FULL profile
/// (`USE` is a lifecycle statement `PUBLIC_EXPLORER` itself refuses),
/// tighten to `PUBLIC_EXPLORER`, then serve jobs until the sender side
/// drops. Returns once the bind has succeeded — a container that does
/// not bind fails the boot, not the first request.
pub fn spawn(
    container: &Path,
    timeout: Duration,
) -> Result<LqlBridge, Box<dyn std::error::Error + Send + Sync>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Job>(64);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    // Backslashes double so a Windows path survives the lexer's escape pass.
    let use_stmt = format!(
        "USE \"{}\";",
        container.display().to_string().replace('\\', "\\\\")
    );

    std::thread::Builder::new()
        .name("lql-public-explorer".into())
        .spawn(move || {
            let mut session = Session::new();
            let bound = parse(&use_stmt)
                .map_err(|e| e.to_string())
                .and_then(|stmt| session.execute(&stmt).map_err(|e| e.to_string()));
            match bound {
                Ok(_) => {
                    session.set_profile(CapabilityProfile::PublicExplorer);
                    let _ = ready_tx.send(Ok(()));
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            }
            while let Some(job) = rx.blocking_recv() {
                let result = match parse(&job.statement) {
                    Err(e) => Err(QueryFailure::Parse(e.to_string())),
                    Ok(stmt) => session.execute(&stmt).map_err(|e| match e {
                        LqlError::Refused { .. } => QueryFailure::Refused(e.to_string()),
                        other => QueryFailure::Execution(other.to_string()),
                    }),
                };
                let _ = job.reply.send(result);
            }
        })?;

    ready_rx
        .recv()
        .map_err(|_| "the query session thread died during bind")??;
    info!("LQL bridge: session bound, profile PUBLIC_EXPLORER");
    Ok(LqlBridge {
        tx,
        timeout,
        profile: "PUBLIC_EXPLORER",
    })
}
