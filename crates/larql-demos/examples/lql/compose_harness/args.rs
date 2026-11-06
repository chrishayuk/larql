//! Command line, parsed by hand: the harness takes a handful of flags.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Set {
    Controls,
    Development,
    Registered,
}

pub struct Args {
    pub container: PathBuf,
    pub tag: String,
    pub arm: String,
    pub set: Set,
    pub bank: PathBuf,
    pub controls: PathBuf,
    pub out: PathBuf,
    pub alpha: Option<f32>,
    pub unrelated: usize,
    pub check_determinism: bool,
    pub command: String,
}

const USAGE: &str = "usage: compose_harness --container DIR --tag NAME --arm A0|A1 --set controls|development|registered \\
    --bank bank.json --controls controls.json --out DIR [--alpha F] [--unrelated N] [--check-determinism]";

pub fn parse() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let take = |name: &str| -> Option<String> {
        raw.iter()
            .position(|a| a == name)
            .and_then(|i| raw.get(i + 1))
            .cloned()
    };
    let need = |v: Option<String>, name: &str| v.ok_or_else(|| format!("missing {name}\n{USAGE}"));
    let set = match need(take("--set"), "--set")?.as_str() {
        "controls" => Set::Controls,
        "development" => Set::Development,
        "registered" => Set::Registered,
        other => return Err(format!("unknown set {other:?}\n{USAGE}")),
    };
    Ok(Args {
        container: PathBuf::from(need(take("--container"), "--container")?),
        tag: need(take("--tag"), "--tag")?,
        arm: need(take("--arm"), "--arm")?,
        set,
        bank: PathBuf::from(need(take("--bank"), "--bank")?),
        controls: PathBuf::from(need(take("--controls"), "--controls")?),
        out: PathBuf::from(need(take("--out"), "--out")?),
        alpha: take("--alpha")
            .map(|v| v.parse().map_err(|_| format!("bad --alpha {v:?}")))
            .transpose()?,
        unrelated: take("--unrelated").map_or(Ok(10), |v| {
            v.parse().map_err(|_| format!("bad --unrelated {v:?}"))
        })?,
        check_determinism: raw.iter().any(|a| a == "--check-determinism"),
        command: raw.join(" "),
    })
}
