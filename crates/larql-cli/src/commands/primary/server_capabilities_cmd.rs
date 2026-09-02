//! `larql server-capabilities <URL>` — ask a running LARQL server
//! what it will and will not do.
//!
//! Deliberately *not* folded into `larql capabilities`, which answers
//! a different question about a different object: what architectures
//! **this release** recognises, a build-time fact. This asks a
//! specific running process about its own route surface. Same English
//! word, two objects — so two verbs.
//!
//! The client half of the Explorer contract's step 3. The rule it
//! exists to enforce is that nobody infers a server's powers from its
//! address: this prints what the server says, and refuses a report
//! whose schema it does not recognise rather than reading the keys it
//! happens to know.

use std::time::Duration;

use clap::Args;

/// The report schema this client understands. A server speaking a
/// different one is refused, not partially read — the same discipline
/// `SystemPlan::parse` applies to plan schema 4.
const SUPPORTED_SCHEMA: u64 = 1;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Args, Debug)]
pub struct ServerCapabilitiesArgs {
    /// Base URL of a running LARQL server, e.g. `http://localhost:8080`.
    #[arg(value_name = "SERVER_URL")]
    pub url: String,

    /// Print the server's report verbatim instead of a summary.
    #[arg(long)]
    pub json: bool,

    /// Also list every route the server reports mounting.
    #[arg(long)]
    pub routes: bool,
}

pub fn run(args: ServerCapabilitiesArgs) -> Result<(), Box<dyn std::error::Error>> {
    let base = args.url.trim_end_matches('/');
    let endpoint = format!("{base}/v1/capabilities");

    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;
    let response = client.get(&endpoint).send()?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(format!(
            "{endpoint} → 404. This server does not serve the capabilities contract: either it \
             predates it, or {base} is not a LARQL server. Nothing can be inferred about what it \
             supports from the fact that it answered."
        )
        .into());
    }
    if !response.status().is_success() {
        return Err(format!("{endpoint} → {}", response.status()).into());
    }

    let report: serde_json::Value = response.json()?;
    check_schema(&report, &endpoint)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    print!("{}", render_summary(base, &report, args.routes));
    Ok(())
}

/// Refuse a report this client cannot read, rather than picking out
/// the keys it recognises and hoping the rest meant what it expects.
fn check_schema(
    report: &serde_json::Value,
    endpoint: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = report["schema"].as_u64().ok_or_else(|| {
        format!("{endpoint} answered without a schema — refusing to read it as capabilities")
    })?;
    if schema != SUPPORTED_SCHEMA {
        return Err(format!(
            "{endpoint} speaks capabilities schema {schema}; this build understands \
             {SUPPORTED_SCHEMA}. Refusing to read a document whose shape it does not know — \
             upgrade the client, or read it with --json."
        )
        .into());
    }
    Ok(())
}

/// The human summary. Returns a `String` rather than printing so the
/// rendering is testable without a running server — the transport
/// above is the only part that needs one.
fn render_summary(base: &str, report: &serde_json::Value, list_routes: bool) -> String {
    use std::fmt::Write as _;
    const UNKNOWN: &str = "unknown";
    let mut out = String::new();

    let _ = writeln!(out, "{base}");
    let _ = writeln!(
        out,
        "  server     {} {}",
        report["server"]["name"].as_str().unwrap_or(UNKNOWN),
        report["server"]["version"].as_str().unwrap_or(UNKNOWN),
    );
    let _ = writeln!(
        out,
        "  profile    {}",
        report["profile"].as_str().unwrap_or(UNKNOWN)
    );

    let _ = writeln!(out, "\nSOURCES         LOCAL   HF");
    for verb in ["load", "plan", "encode"] {
        let _ = writeln!(
            out,
            "  {verb:<16}{:<8}{}",
            mark(&report["sources"][verb]["local"]),
            mark(&report["sources"][verb]["hf"]),
        );
    }
    // The distinction the server draws, restated where someone reading
    // "load: hf yes" might otherwise hand it a raw checkpoint.
    let _ = writeln!(
        out,
        "  load takes an encoded container; plan and encode take a checkpoint."
    );

    render_block(&mut out, "EXPLORER", &report["explorer"]);
    render_block(&mut out, "RUNTIME", &report["runtime"]);

    if let Some(backends) = report["runtime"]["backends"].as_array() {
        let names: Vec<&str> = backends.iter().filter_map(|b| b.as_str()).collect();
        let _ = writeln!(out, "  {:<16}{}", "backends", names.join(", "));
    }

    if list_routes {
        let _ = writeln!(out, "\nROUTES");
        for route in report["routes"].as_array().into_iter().flatten() {
            let _ = writeln!(out, "  {}", route.as_str().unwrap_or(UNKNOWN));
        }
    }
    out
}

/// Render every boolean in `block`, sorted by key. Driven by what the
/// server sent rather than by a list of keys this client expects — a
/// server that gains a capability shows it here with no client
/// release, and one that drops a key simply stops printing it.
fn render_block(out: &mut String, title: &str, block: &serde_json::Value) {
    use std::fmt::Write as _;
    let Some(map) = block.as_object() else { return };
    let _ = writeln!(out, "\n{title}");
    for (key, value) in map {
        if value.is_boolean() {
            let _ = writeln!(out, "  {key:<16}{}", mark(value));
        }
    }
}

fn mark(value: &serde_json::Value) -> &'static str {
    match value.as_bool() {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(schema: u64) -> serde_json::Value {
        serde_json::json!({
            "object": "capabilities",
            "schema": schema,
            "profile": "single_model",
            "server": { "name": "larql-server", "version": "0.2.0" },
            "sources": {
                "load": { "local": true, "hf": true },
                "plan": { "local": false, "hf": false },
                "encode": { "local": false, "hf": false },
            },
            "explorer": { "components": false, "walk": true },
            "runtime": { "execute": true, "backends": ["cpu"] },
            "routes": ["/v1/capabilities", "/v1/walk"],
        })
    }

    #[test]
    fn accepts_the_schema_it_understands() {
        assert!(check_schema(&report(SUPPORTED_SCHEMA), "u").is_ok());
    }

    #[test]
    fn refuses_a_schema_it_does_not_understand() {
        let err = check_schema(&report(SUPPORTED_SCHEMA + 1), "u")
            .unwrap_err()
            .to_string();
        assert!(err.contains("schema 2"), "{err}");
        assert!(err.contains("Refusing"), "{err}");
    }

    #[test]
    fn refuses_a_document_with_no_schema_at_all() {
        let err = check_schema(&serde_json::json!({"profile": "x"}), "u")
            .unwrap_err()
            .to_string();
        assert!(err.contains("without a schema"), "{err}");
    }

    #[test]
    fn summary_names_the_server_and_its_profile() {
        let out = render_summary("http://localhost:8080", &report(1), false);
        assert!(out.contains("http://localhost:8080"), "{out}");
        assert!(out.contains("larql-server 0.2.0"), "{out}");
        assert!(out.contains("single_model"), "{out}");
    }

    #[test]
    fn summary_separates_what_load_takes_from_what_encode_takes() {
        let out = render_summary("u", &report(1), false);
        let load = out.lines().find(|l| l.contains("load ")).unwrap();
        let encode = out.lines().find(|l| l.contains("encode")).unwrap();
        assert!(load.contains("yes"), "{load}");
        assert!(!encode.contains("yes"), "{encode}");
        assert!(
            out.contains("load takes an encoded container"),
            "the container/checkpoint distinction must survive into the summary: {out}"
        );
    }

    /// The renderer prints whatever booleans the server sent. A server
    /// that gains a capability shows it without a client release —
    /// this is what stops the CLI becoming a second, staler list of
    /// what servers can do.
    #[test]
    fn summary_renders_capabilities_this_build_has_never_heard_of() {
        let mut r = report(1);
        r["explorer"]["time_travel"] = serde_json::json!(true);
        let out = render_summary("u", &r, false);
        assert!(out.contains("time_travel"), "{out}");
    }

    #[test]
    fn backends_are_named() {
        let out = render_summary("u", &report(1), false);
        assert!(out.contains("backends"), "{out}");
        assert!(out.contains("cpu"), "{out}");
    }

    #[test]
    fn routes_are_listed_only_when_asked() {
        assert!(!render_summary("u", &report(1), false).contains("/v1/walk"));
        let listed = render_summary("u", &report(1), true);
        assert!(listed.contains("ROUTES"), "{listed}");
        assert!(listed.contains("/v1/walk"), "{listed}");
    }

    #[test]
    fn a_non_object_block_renders_nothing() {
        let mut out = String::new();
        render_block(&mut out, "EXPLORER", &serde_json::json!("not an object"));
        assert!(out.is_empty(), "{out}");
    }

    #[test]
    fn marks_cover_true_false_and_absent() {
        assert_eq!(mark(&serde_json::json!(true)), "yes");
        assert_eq!(mark(&serde_json::json!(false)), "no");
        assert_eq!(mark(&serde_json::Value::Null), "-");
    }

    #[test]
    fn missing_identity_fields_render_as_unknown_rather_than_panicking() {
        let out = render_summary("u", &serde_json::json!({"schema": 1}), true);
        assert!(out.contains("unknown"), "{out}");
    }
}
