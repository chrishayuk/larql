//! COMPOSE-CORRECTION-1 harness (development stage).
//!
//! Reads only the public `larql_lql` API; no product code is involved. Three sets:
//! `controls` replays recorded cases, `development` runs the bank's development facts, and
//! `registered` is unreachable until the pre-registration is frozen (see `guard`). Registered facts
//! are never read on any other path.

mod args;
mod bank;
mod consts;
mod controls;
mod fact;
mod faults;
mod guard;
mod infer;
mod observer;
mod provenance;

use args::Set;
use serde_json::json;
use tokenizers::Tokenizer;

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<bool, String> {
    let args = args::parse()?;
    let loaded = bank::load(&args.bank)?;
    let bank_dir = args
        .bank
        .parent()
        .ok_or("bank has no directory")?
        .to_path_buf();
    if args.set == Set::Registered {
        guard::admit_registered(&bank_dir, &loaded.sha256, &args.command)?;
    }
    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let provenance = provenance::collect(
        &args.container,
        &args.bank,
        &loaded.sha256,
        &args.arm,
        &args.tag,
        &args.command,
    )?;
    let file = args
        .out
        .join(format!("{}-{}-{:?}.json", args.tag, args.arm, args.set).to_lowercase());

    let (ok, body) = match args.set {
        Set::Controls => {
            let (ok, results) =
                controls::run(&args.controls, &args.container, &args.tag, &args.arm)?;
            (ok, json!({"controls": results, "all_reproduce": ok}))
        }
        Set::Faults => {
            let matrix = faults::run(&args.container, &loaded.bank)?;
            (true, json!({"fault_matrix": matrix}))
        }
        Set::Development | Set::Registered => {
            let tokenizer = Tokenizer::from_file(args.container.join("tokenizer.json"))
                .map_err(|e| e.to_string())?;
            let settings = fact::Settings {
                container: args.container.clone(),
                alpha: args.alpha,
                unrelated: args.unrelated,
                check_determinism: args.check_determinism,
                out_dir: args.out.clone(),
            };
            let facts = if args.set == Set::Development {
                &loaded.bank.development
            } else {
                &loaded.bank.facts
            };
            let mut results = Vec::new();
            for f in facts {
                eprintln!("[{}] {} {} -> {}", args.tag, f.relation, f.entity, f.target);
                results.push(fact::run(f, &loaded.bank.unrelated, &settings, &tokenizer)?);
            }
            (true, json!({"facts": results}))
        }
    };
    let document = json!({"provenance": provenance, "result": body});
    std::fs::write(
        &file,
        serde_json::to_string_pretty(&document).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!("wrote {}", file.display());
    Ok(ok)
}
