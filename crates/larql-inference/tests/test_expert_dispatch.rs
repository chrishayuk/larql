#![cfg(not(target_arch = "wasm32"))]

/// Option 3 integration test: cascade trie → expert dispatch pipeline
///
/// Each case documents what the cascade would commit to (route label) and the
/// hardwired (op, args) that extraction will produce once Option 1 is wired.
/// The test validates only the dispatch half — registry.call(op, args) → result.
///
/// Requires larql-experts to be pre-built:
///   cd crates/larql-experts && cargo build --target wasm32-wasip1 --release
use std::path::PathBuf;

use larql_inference::experts::ExpertRegistry;
use serde_json::{json, Value};

// ── Infrastructure ────────────────────────────────────────────────────────────

fn wasm_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larql-experts/target/wasm32-wasip1/release")
}

fn registry() -> Option<ExpertRegistry> {
    let dir = wasm_dir();
    if !dir.exists() {
        eprintln!("skip: wasm dir missing — run `cargo build --target wasm32-wasip1 --release` in larql-experts");
        return None;
    }
    Some(ExpertRegistry::load_dir(&dir).expect("load_dir"))
}

/// Route labels — what the cascade trie commits to at L5.
/// Stored as strings for now; becomes a proper enum when the trie probe is wired.
#[allow(dead_code)]
const ARITHMETIC: &str = "arithmetic";
#[allow(dead_code)]
const LOGICAL: &str = "logical";
#[allow(dead_code)]
const DATE: &str = "date";
#[allow(dead_code)]
const CODE: &str = "code";
#[allow(dead_code)]
const FACTUAL: &str = "factual";

struct DispatchCase {
    /// The natural language prompt (documents what option 1 extraction will parse)
    prompt: &'static str,
    /// What the cascade trie commits to
    route: &'static str,
    /// Expert op name
    op: &'static str,
    /// Hardwired args (option 3) — will come from structured generation (option 1)
    args: Value,
    /// Expected result
    expected: Expected,
}

enum Expected {
    Exact(Value),
    Approx(f64, f64), // (value, tolerance)
    Field(&'static str, Value),
}

#[track_caller]
fn run_case(reg: &mut ExpertRegistry, case: &DispatchCase) {
    let result = reg.call(case.op, &case.args);
    let got = result.map(|r| r.value).unwrap_or(Value::Null);

    match &case.expected {
        Expected::Exact(expected) => {
            assert_eq!(
                got, *expected,
                "\nprompt:  {}\nroute:   {}\nop:      {}\nargs:    {}",
                case.prompt, case.route, case.op, case.args
            );
        }
        Expected::Approx(expected, tol) => {
            let got_f = got.as_f64().unwrap_or(f64::NAN);
            assert!(
                (got_f - expected).abs() <= *tol,
                "\nprompt:  {}\nroute:   {}\nop:      {}\ngot {got_f}, expected {expected} ± {tol}",
                case.prompt,
                case.route,
                case.op
            );
        }
        Expected::Field(key, expected) => {
            let field = got.get(key).unwrap_or(&Value::Null);
            assert_eq!(
                field, expected,
                "\nprompt:  {}\nroute:   {}\nop:      {}\nmissing field '{key}' in {got}",
                case.prompt, case.route, case.op
            );
        }
    }
}

// ── Test cases ────────────────────────────────────────────────────────────────

fn cases() -> Vec<DispatchCase> {
    vec![
        // ── arithmetic ──────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is the GCD of 144 and 60?",
            route: ARITHMETIC,
            op: "gcd",
            args: json!({"a": 144, "b": 60}),
            expected: Expected::Exact(json!(12)),
        },
        DispatchCase {
            prompt: "Is 97 prime?",
            route: ARITHMETIC,
            op: "is_prime",
            args: json!({"n": 97}),
            expected: Expected::Exact(json!(true)),
        },
        DispatchCase {
            prompt: "What is 2 to the power of 16?",
            route: ARITHMETIC,
            op: "pow",
            args: json!({"a": 2.0, "b": 16.0}),
            expected: Expected::Exact(json!(65536.0)),
        },
        DispatchCase {
            prompt: "What is 10 factorial?",
            route: ARITHMETIC,
            op: "factorial",
            args: json!({"n": 10}),
            expected: Expected::Exact(json!(3628800)),
        },
        DispatchCase {
            prompt: "Convert 255 to binary",
            route: ARITHMETIC,
            op: "to_base",
            args: json!({"n": 255, "base": 2}),
            expected: Expected::Exact(json!("11111111")),
        },
        DispatchCase {
            prompt: "Write 2024 as a Roman numeral",
            route: ARITHMETIC,
            op: "to_roman",
            args: json!({"n": 2024}),
            expected: Expected::Exact(json!("MMXXIV")),
        },
        // ── date ────────────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "How many days between 1st January and 1st March 2026?",
            route: DATE,
            op: "days_between",
            args: json!({"from": {"year": 2026, "month": 1, "day": 1},
                     "to":   {"year": 2026, "month": 3, "day": 1}}),
            expected: Expected::Exact(json!(59)), // Jan 31 + Feb 28
        },
        DispatchCase {
            prompt: "Is 2024 a leap year?",
            route: DATE,
            op: "is_leap_year",
            args: json!({"year": 2024}),
            expected: Expected::Exact(json!(true)),
        },
        DispatchCase {
            prompt: "How many days are in February 2026?",
            route: DATE,
            op: "days_in_month",
            args: json!({"year": 2026, "month": 2}),
            expected: Expected::Exact(json!(28)),
        },
        DispatchCase {
            prompt: "What date is 30 days after 2026-03-01?",
            route: DATE,
            op: "add_days",
            args: json!({"date": {"year": 2026, "month": 3, "day": 1}, "days": 30}),
            expected: Expected::Exact(json!({"year": 2026, "month": 3, "day": 31})),
        },
        // ── logic ───────────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "Simplify NOT NOT A",
            route: LOGICAL,
            op: "simplify",
            args: json!({"expr": "NOT NOT A"}),
            expected: Expected::Exact(json!("A")),
        },
        DispatchCase {
            prompt: "Simplify A AND TRUE",
            route: LOGICAL,
            op: "simplify",
            args: json!({"expr": "A AND TRUE"}),
            expected: Expected::Exact(json!("A")),
        },
        DispatchCase {
            prompt: "Is A OR NOT A a tautology?",
            route: LOGICAL,
            op: "classify",
            args: json!({"expr": "A OR NOT A"}),
            expected: Expected::Exact(json!("tautology")),
        },
        DispatchCase {
            prompt: "Evaluate A AND B when A=true and B=false",
            route: LOGICAL,
            op: "eval",
            args: json!({"expr": "A AND B", "assignments": {"A": true, "B": false}}),
            expected: Expected::Exact(json!(false)),
        },
        // ── unit conversion ─────────────────────────────────────────────────────
        DispatchCase {
            prompt: "Convert 100 kilometres to miles",
            route: FACTUAL, // unit queries land in factual route at L5
            op: "convert",
            args: json!({"value": 100.0, "from": "km", "to": "mi"}),
            expected: Expected::Approx(62.137, 0.001),
        },
        DispatchCase {
            prompt: "Convert 37 degrees Celsius to Fahrenheit",
            route: FACTUAL,
            op: "convert",
            args: json!({"value": 37.0, "from": "C", "to": "F"}),
            expected: Expected::Approx(98.6, 1e-6),
        },
        DispatchCase {
            prompt: "Convert 100 kilograms to pounds",
            route: FACTUAL,
            op: "convert",
            args: json!({"value": 100.0, "from": "kg", "to": "lb"}),
            expected: Expected::Approx(220.462, 0.001),
        },
        // ── statistics ──────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is the mean of 2, 4, 6, 8, 10?",
            route: ARITHMETIC,
            op: "mean",
            args: json!({"values": [2, 4, 6, 8, 10]}),
            expected: Expected::Approx(6.0, 1e-12),
        },
        DispatchCase {
            prompt: "What is the standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?",
            route: ARITHMETIC,
            op: "stddev",
            args: json!({"values": [2, 4, 4, 4, 5, 5, 7, 9]}),
            expected: Expected::Approx(2.0, 1e-9),
        },
        // ── geometry ────────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is the area of a circle with radius 5?",
            route: ARITHMETIC,
            op: "circle_area",
            args: json!({"r": 5.0}),
            expected: Expected::Approx(std::f64::consts::PI * 25.0, 1e-9),
        },
        DispatchCase {
            prompt: "What is the hypotenuse of a right triangle with sides 3 and 4?",
            route: ARITHMETIC,
            op: "hypotenuse",
            args: json!({"a": 3.0, "b": 4.0}),
            expected: Expected::Approx(5.0, 1e-9),
        },
        // ── trigonometry ────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is sin(π/6)?",
            route: ARITHMETIC,
            op: "sin",
            args: json!({"x": std::f64::consts::FRAC_PI_6}),
            expected: Expected::Approx(0.5, 1e-12),
        },
        DispatchCase {
            prompt: "What is cos(π/3)?",
            route: ARITHMETIC,
            op: "cos",
            args: json!({"x": std::f64::consts::FRAC_PI_3}),
            expected: Expected::Approx(0.5, 1e-12),
        },
        // ── SQL ─────────────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "SELECT COUNT(*) FROM users WHERE age > 25",
            route: CODE,
            op: "execute",
            args: json!({"sql": "CREATE TABLE users (name TEXT, age INT); \
                             INSERT INTO users VALUES ('Alice', 30); \
                             INSERT INTO users VALUES ('Bob', 20); \
                             INSERT INTO users VALUES ('Carol', 35); \
                             SELECT COUNT(*) FROM users WHERE age > 25"}),
            expected: Expected::Exact(json!(2)),
        },
        DispatchCase {
            prompt: "SELECT the name of the user with id 2",
            route: CODE,
            op: "execute",
            args: json!({"sql": "CREATE TABLE u (id INT, name TEXT); \
                             INSERT INTO u VALUES (1, 'Alice'); \
                             INSERT INTO u VALUES (2, 'Bob'); \
                             SELECT name FROM u WHERE id = 2"}),
            expected: Expected::Exact(json!("Bob")),
        },
        // ── string ops ──────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "Reverse the string 'hello world'",
            route: CODE,
            op: "reverse",
            args: json!({"s": "hello world"}),
            expected: Expected::Exact(json!("dlrow olleh")),
        },
        DispatchCase {
            prompt: "Is 'racecar' a palindrome?",
            route: CODE,
            op: "is_palindrome",
            args: json!({"s": "racecar"}),
            expected: Expected::Exact(json!(true)),
        },
        DispatchCase {
            prompt: "Apply a Caesar cipher with shift 13 to 'attack'",
            route: CODE,
            op: "caesar",
            args: json!({"s": "attack", "shift": 13}),
            expected: Expected::Exact(json!("nggnpx")),
        },
        // ── hash / encoding ─────────────────────────────────────────────────────
        DispatchCase {
            prompt: "Base64 encode 'hello world'",
            route: CODE,
            op: "base64_encode",
            args: json!({"s": "hello world"}),
            expected: Expected::Exact(json!("aGVsbG8gd29ybGQ=")),
        },
        // ── element lookup ──────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is the atomic mass of gold?",
            route: FACTUAL,
            op: "by_name",
            args: json!({"name": "gold"}),
            expected: Expected::Field("symbol", json!("Au")),
        },
        DispatchCase {
            prompt: "What element has atomic number 26?",
            route: FACTUAL,
            op: "by_number",
            args: json!({"z": 26}),
            expected: Expected::Field("name", json!("iron")),
        },
        // ── HTTP status ─────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What does HTTP 404 mean?",
            route: FACTUAL,
            op: "lookup",
            args: json!({"code": 404}),
            expected: Expected::Field("reason", json!("Not Found")),
        },
        DispatchCase {
            prompt: "What category is HTTP 503?",
            route: FACTUAL,
            op: "lookup",
            args: json!({"code": 503}),
            expected: Expected::Field("category", json!("5xx")),
        },
        // ── finance ─────────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "What is the future value of £1000 at 5% for 10 years?",
            route: ARITHMETIC,
            op: "future_value",
            args: json!({"pv": 1000.0, "rate_pct": 5.0, "years": 10}),
            expected: Expected::Approx(1628.89, 0.01),
        },
        // ── Luhn / ISBN ─────────────────────────────────────────────────────────
        DispatchCase {
            prompt: "Is the card number 4532015112830366 valid?",
            route: FACTUAL,
            op: "check",
            args: json!({"number": "4532015112830366"}),
            expected: Expected::Exact(json!(true)),
        },
        DispatchCase {
            prompt: "What card network is 378282246310005?",
            route: FACTUAL,
            op: "card_type",
            args: json!({"number": "378282246310005"}),
            expected: Expected::Exact(json!("amex")),
        },
    ]
}

// ── Single test function ──────────────────────────────────────────────────────

#[test]
fn expert_dispatch_pipeline() {
    let Some(mut reg) = registry() else { return };
    let cases = cases();

    let mut passed = 0usize;
    let failed = 0usize;

    for case in &cases {
        // Panic on the first failure so the message is readable
        run_case(&mut reg, case);
        passed += 1;
        eprintln!("  ok  [{}] {} → {} ", case.route, case.prompt, case.op);
    }

    eprintln!("\n{}/{} dispatch cases passed", passed, cases.len());
    assert_eq!(failed, 0);
}
