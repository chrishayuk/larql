#![cfg(not(target_arch = "wasm32"))]

/// Integration tests for the WASM expert registry.
///
/// Requires the larql-experts workspace to be pre-built:
///   cd crates/larql-experts && cargo build --target wasm32-wasip1 --release
///
/// Each test loads the expert under test from the release WASM directory and
/// invokes ops with structured args, asserting on typed JSON values.
use std::path::PathBuf;

use larql_inference::experts::ExpertRegistry;
use serde_json::{json, Value};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn wasm_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../larql-experts/target/wasm32-wasip1/release")
}

fn wasm(name: &str) -> PathBuf {
    wasm_dir().join(format!("larql_expert_{}.wasm", name))
}

/// Load a single expert and invoke `op` with `args`.
/// Returns None if the expert binary is missing (skip) or the expert declined.
fn call(expert: &str, op: &str, args: Value) -> Option<Value> {
    let path = wasm(expert);
    if !path.exists() {
        eprintln!("skip (missing wasm): {}", expert);
        return None;
    }
    let mut reg = ExpertRegistry::default();
    reg.load_file(&path).expect("load");
    reg.call(op, &args).map(|r| r.value)
}

/// Assert the expert's value equals `expected`.
#[track_caller]
fn assert_eq_expert(expert: &str, op: &str, args: Value, expected: Value) {
    if let Some(v) = call(expert, op, args.clone()) {
        assert_eq!(v, expected, "expert={expert} op={op} args={args}");
    }
}

/// Assert approximate equality for f64 results (relative or absolute).
#[track_caller]
fn assert_approx(expert: &str, op: &str, args: Value, expected: f64, tol: f64) {
    if let Some(v) = call(expert, op, args.clone()) {
        let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {}", v));
        assert!(
            (got - expected).abs() <= tol,
            "expert={expert} op={op} args={args}: expected ~{}, got {}",
            expected,
            got
        );
    }
}

/// Assert the value has `field` equal to `expected`.
#[track_caller]
fn assert_field(expert: &str, op: &str, args: Value, field: &str, expected: Value) {
    if let Some(v) = call(expert, op, args.clone()) {
        let f = v
            .get(field)
            .unwrap_or_else(|| panic!("missing field {field} in {v}"));
        assert_eq!(
            f, &expected,
            "expert={expert} op={op} args={args}: field {field}"
        );
    }
}

// ── arithmetic ────────────────────────────────────────────────────────────────

#[test]
fn arithmetic_add() {
    assert_eq_expert("arithmetic", "add", json!({"a": 12, "b": 34}), json!(46.0));
}

#[test]
fn arithmetic_subtract() {
    assert_eq_expert("arithmetic", "sub", json!({"a": 100, "b": 37}), json!(63.0));
}

#[test]
fn arithmetic_multiply() {
    assert_eq_expert("arithmetic", "mul", json!({"a": 7, "b": 8}), json!(56.0));
}

#[test]
fn arithmetic_divide() {
    assert_eq_expert("arithmetic", "div", json!({"a": 144, "b": 12}), json!(12.0));
}

#[test]
fn arithmetic_divide_by_zero() {
    assert_eq_expert("arithmetic", "div", json!({"a": 1, "b": 0}), Value::Null);
}

#[test]
fn arithmetic_power() {
    assert_eq_expert("arithmetic", "pow", json!({"a": 2, "b": 10}), json!(1024.0));
}

#[test]
fn arithmetic_mod() {
    assert_eq_expert("arithmetic", "mod", json!({"a": 17, "b": 5}), json!(2));
}

#[test]
fn arithmetic_prime_true() {
    assert_eq_expert("arithmetic", "is_prime", json!({"n": 17}), json!(true));
}

#[test]
fn arithmetic_prime_false() {
    assert_eq_expert("arithmetic", "is_prime", json!({"n": 15}), json!(false));
}

#[test]
fn arithmetic_gcd() {
    assert_eq_expert("arithmetic", "gcd", json!({"a": 48, "b": 18}), json!(6));
}

#[test]
fn arithmetic_lcm() {
    assert_eq_expert("arithmetic", "lcm", json!({"a": 4, "b": 6}), json!(12));
}

#[test]
fn arithmetic_factorial() {
    assert_eq_expert("arithmetic", "factorial", json!({"n": 5}), json!(120));
}

#[test]
fn arithmetic_binary() {
    assert_eq_expert(
        "arithmetic",
        "to_base",
        json!({"n": 255, "base": 2}),
        json!("11111111"),
    );
}

#[test]
fn arithmetic_hex() {
    assert_eq_expert(
        "arithmetic",
        "to_base",
        json!({"n": 255, "base": 16}),
        json!("FF"),
    );
}

#[test]
fn arithmetic_roman_from() {
    assert_eq_expert("arithmetic", "from_roman", json!({"s": "XIV"}), json!(14));
}

#[test]
fn arithmetic_roman_to() {
    assert_eq_expert("arithmetic", "to_roman", json!({"n": 42}), json!("XLII"));
}

#[test]
fn arithmetic_percent_of() {
    assert_eq_expert(
        "arithmetic",
        "percent_of",
        json!({"pct": 20, "n": 150}),
        json!(30.0),
    );
}

#[test]
fn arithmetic_unknown_op() {
    // Unknown op should return None (expert declines).
    assert!(call("arithmetic", "flibbertigibbet", json!({})).is_none());
}

// ── date ──────────────────────────────────────────────────────────────────────

#[test]
fn date_days_between() {
    assert_eq_expert(
        "date",
        "days_between",
        json!({"from": {"year": 2023, "month": 3, "day": 15}, "to": {"year": 2023, "month": 3, "day": 20}}),
        json!(5),
    );
}

#[test]
fn date_days_between_year() {
    assert_eq_expert(
        "date",
        "days_between",
        json!({"from": {"year": 2023, "month": 1, "day": 1}, "to": {"year": 2024, "month": 1, "day": 1}}),
        json!(365),
    );
}

#[test]
fn date_day_of_week_wednesday() {
    // 25 December 2024 was a Wednesday (ISO index 3).
    assert_eq_expert(
        "date",
        "day_of_week",
        json!({"date": {"year": 2024, "month": 12, "day": 25}}),
        json!(3),
    );
}

#[test]
fn date_add_days() {
    assert_eq_expert(
        "date",
        "add_days",
        json!({"date": {"year": 2025, "month": 1, "day": 1}, "days": 10}),
        json!({"year": 2025, "month": 1, "day": 11}),
    );
}

#[test]
fn date_subtract_days() {
    assert_eq_expert(
        "date",
        "add_days",
        json!({"date": {"year": 2023, "month": 3, "day": 10}, "days": -5}),
        json!({"year": 2023, "month": 3, "day": 5}),
    );
}

#[test]
fn date_leap_year_true() {
    assert_eq_expert("date", "is_leap_year", json!({"year": 2024}), json!(true));
}

#[test]
fn date_leap_year_false() {
    assert_eq_expert("date", "is_leap_year", json!({"year": 2023}), json!(false));
}

#[test]
fn date_days_in_feb_leap() {
    assert_eq_expert(
        "date",
        "days_in_month",
        json!({"year": 2024, "month": 2}),
        json!(29),
    );
}

#[test]
fn date_days_in_feb_normal() {
    assert_eq_expert(
        "date",
        "days_in_month",
        json!({"year": 2023, "month": 2}),
        json!(28),
    );
}

#[test]
fn date_weeks_between() {
    assert_eq_expert(
        "date",
        "weeks_between",
        json!({"from": {"year": 2024, "month": 1, "day": 1}, "to": {"year": 2025, "month": 1, "day": 1}}),
        json!(52),
    );
}

// ── unit ─────────────────────────────────────────────────────────────────────

#[test]
fn unit_km_to_m() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 5, "from": "km", "to": "m"}),
        5000.0,
        1e-6,
    );
}

#[test]
fn unit_miles_to_km() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 10, "from": "mi", "to": "km"}),
        16.0934,
        1e-3,
    );
}

#[test]
fn unit_kg_to_lbs() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 70, "from": "kg", "to": "lb"}),
        154.32,
        0.5,
    );
}

#[test]
fn unit_celsius_to_fahrenheit() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 100, "from": "C", "to": "F"}),
        212.0,
        1e-6,
    );
}

#[test]
fn unit_inches_to_cm() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 12, "from": "in", "to": "cm"}),
        30.48,
        1e-6,
    );
}

#[test]
fn unit_incompatible_groups() {
    // length to mass => explicit null, not None (expert does handle the op).
    assert_eq_expert(
        "unit",
        "convert",
        json!({"value": 1, "from": "km", "to": "kg"}),
        Value::Null,
    );
}

// ── statistics ────────────────────────────────────────────────────────────────

#[test]
fn statistics_mean() {
    assert_approx(
        "statistics",
        "mean",
        json!({"values": [1,2,3,4,5]}),
        3.0,
        1e-12,
    );
}

#[test]
fn statistics_median_odd() {
    assert_approx(
        "statistics",
        "median",
        json!({"values": [1,3,5,7,9]}),
        5.0,
        1e-12,
    );
}

#[test]
fn statistics_median_even() {
    assert_approx(
        "statistics",
        "median",
        json!({"values": [1,2,3,4]}),
        2.5,
        1e-12,
    );
}

#[test]
fn statistics_mode() {
    assert_eq_expert(
        "statistics",
        "mode",
        json!({"values": [1,2,2,3,3,3]}),
        json!([3.0]),
    );
}

#[test]
fn statistics_min() {
    assert_approx(
        "statistics",
        "min",
        json!({"values": [4,2,9,1,7]}),
        1.0,
        1e-12,
    );
}

#[test]
fn statistics_max() {
    assert_approx(
        "statistics",
        "max",
        json!({"values": [4,2,9,1,7]}),
        9.0,
        1e-12,
    );
}

#[test]
fn statistics_sort() {
    assert_eq_expert(
        "statistics",
        "sort",
        json!({"values": [5,2,8,1]}),
        json!([1.0, 2.0, 5.0, 8.0]),
    );
}

#[test]
fn statistics_count() {
    assert_eq_expert(
        "statistics",
        "count",
        json!({"values": [1,2,3,4,5]}),
        json!(5),
    );
}

#[test]
fn statistics_stddev() {
    // Population stddev of [2,4,4,4,5,5,7,9] is exactly 2.
    assert_approx(
        "statistics",
        "stddev",
        json!({"values": [2,4,4,4,5,5,7,9]}),
        2.0,
        1e-12,
    );
}

// ── geometry ─────────────────────────────────────────────────────────────────

#[test]
fn geometry_circle_area() {
    assert_approx(
        "geometry",
        "circle_area",
        json!({"r": 10}),
        std::f64::consts::PI * 100.0,
        1e-9,
    );
}

#[test]
fn geometry_sphere_volume() {
    assert_approx(
        "geometry",
        "sphere_volume",
        json!({"r": 5}),
        4.0 / 3.0 * std::f64::consts::PI * 125.0,
        1e-9,
    );
}

#[test]
fn geometry_triangle_area() {
    assert_approx(
        "geometry",
        "triangle_area_bh",
        json!({"base": 10, "height": 6}),
        30.0,
        1e-12,
    );
}

#[test]
fn geometry_rectangle_perimeter() {
    assert_approx(
        "geometry",
        "rectangle_perimeter",
        json!({"l": 5, "w": 8}),
        26.0,
        1e-12,
    );
}

#[test]
fn geometry_hypotenuse() {
    assert_approx(
        "geometry",
        "hypotenuse",
        json!({"a": 3, "b": 4}),
        5.0,
        1e-12,
    );
}

// ── trig (radians) ────────────────────────────────────────────────────────────

#[test]
fn trig_sin_pi_6() {
    assert_approx(
        "trig",
        "sin",
        json!({"x": std::f64::consts::FRAC_PI_6}),
        0.5,
        1e-12,
    );
}

#[test]
fn trig_cos_zero() {
    assert_approx("trig", "cos", json!({"x": 0}), 1.0, 1e-12);
}

#[test]
fn trig_tan_pi_4() {
    assert_approx(
        "trig",
        "tan",
        json!({"x": std::f64::consts::FRAC_PI_4}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_asin_half() {
    assert_approx(
        "trig",
        "asin",
        json!({"x": 0.5}),
        std::f64::consts::FRAC_PI_6,
        1e-12,
    );
}

#[test]
fn trig_deg_to_rad() {
    assert_approx(
        "trig",
        "deg_to_rad",
        json!({"deg": 90}),
        std::f64::consts::FRAC_PI_2,
        1e-12,
    );
}

// ── string_ops ────────────────────────────────────────────────────────────────

#[test]
fn string_ops_reverse() {
    assert_eq_expert(
        "string_ops",
        "reverse",
        json!({"s": "hello"}),
        json!("olleh"),
    );
}

#[test]
fn string_ops_palindrome_true() {
    assert_eq_expert(
        "string_ops",
        "is_palindrome",
        json!({"s": "racecar"}),
        json!(true),
    );
}

#[test]
fn string_ops_palindrome_false() {
    assert_eq_expert(
        "string_ops",
        "is_palindrome",
        json!({"s": "hello"}),
        json!(false),
    );
}

#[test]
fn string_ops_anagram_true() {
    assert_eq_expert(
        "string_ops",
        "is_anagram",
        json!({"a": "listen", "b": "silent"}),
        json!(true),
    );
}

#[test]
fn string_ops_anagram_false() {
    assert_eq_expert(
        "string_ops",
        "is_anagram",
        json!({"a": "hello", "b": "world"}),
        json!(false),
    );
}

#[test]
fn string_ops_caesar() {
    assert_eq_expert(
        "string_ops",
        "caesar",
        json!({"s": "abc", "shift": 1}),
        json!("bcd"),
    );
}

#[test]
fn string_ops_uppercase() {
    assert_eq_expert(
        "string_ops",
        "uppercase",
        json!({"s": "hello"}),
        json!("HELLO"),
    );
}

// ── hash ──────────────────────────────────────────────────────────────────────

#[test]
fn hash_base64_encode() {
    assert_eq_expert(
        "hash",
        "base64_encode",
        json!({"s": "hello"}),
        json!("aGVsbG8="),
    );
}

#[test]
fn hash_base64_decode() {
    assert_eq_expert(
        "hash",
        "base64_decode",
        json!({"s": "aGVsbG8="}),
        json!("hello"),
    );
}

#[test]
fn hash_hex_encode() {
    assert_eq_expert(
        "hash",
        "hex_encode",
        json!({"s": "test"}),
        json!("74657374"),
    );
}

#[test]
fn hash_url_encode() {
    assert_eq_expert(
        "hash",
        "url_encode",
        json!({"s": "hello world"}),
        json!("hello%20world"),
    );
}

#[test]
fn hash_fnv() {
    if let Some(v) = call("hash", "fnv1a_32", json!({"s": "key"})) {
        assert!(v.as_str().unwrap_or("").starts_with("0x"));
    }
}

// ── logic ─────────────────────────────────────────────────────────────────────

#[test]
fn logic_eval_and() {
    assert_eq_expert(
        "logic",
        "eval",
        json!({"expr": "A AND B", "assignments": {"A": true, "B": false}}),
        json!(false),
    );
}

#[test]
fn logic_tautology() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A OR NOT A"}),
        json!("tautology"),
    );
}

#[test]
fn logic_contradiction() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A AND NOT A"}),
        json!("contradiction"),
    );
}

#[test]
fn logic_contingent() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A OR B"}),
        json!("contingent"),
    );
}

#[test]
fn logic_truth_table_rows() {
    if let Some(v) = call("logic", "truth_table", json!({"expr": "A AND B"})) {
        let rows = v
            .get("rows")
            .and_then(|r| r.as_array())
            .expect("rows array");
        assert_eq!(rows.len(), 4);
    }
}

// ── finance ───────────────────────────────────────────────────────────────────

#[test]
fn finance_future_value() {
    assert_approx(
        "finance",
        "future_value",
        json!({"pv": 1000, "rate_pct": 5, "years": 10}),
        1628.89,
        1.0,
    );
}

#[test]
fn finance_compound_interest() {
    assert_approx(
        "finance",
        "compound_interest",
        json!({"principal": 1000, "rate_pct": 10, "years": 1}),
        100.0,
        1e-9,
    );
}

#[test]
fn finance_kelly() {
    assert_approx("finance", "kelly", json!({"p": 0.6, "b": 2}), 0.4, 1e-9);
}

#[test]
fn finance_roi() {
    assert_approx(
        "finance",
        "roi",
        json!({"gain": 120, "cost": 100}),
        0.20,
        1e-9,
    );
}

#[test]
fn finance_npv() {
    // -1000 + 400/1.1 + 400/1.1² + 400/1.1³ ≈ -5.26
    if let Some(v) = call(
        "finance",
        "npv",
        json!({"cash_flows": [-1000, 400, 400, 400], "discount_pct": 10}),
    ) {
        let got = v.as_f64().expect("number");
        assert!((got + 5.26).abs() < 1.0, "got {got}");
    }
}

// ── element ───────────────────────────────────────────────────────────────────

#[test]
fn element_atomic_number() {
    assert_field(
        "element",
        "by_name",
        json!({"name": "oxygen"}),
        "z",
        json!(8),
    );
}

#[test]
fn element_symbol() {
    assert_field(
        "element",
        "by_name",
        json!({"name": "carbon"}),
        "symbol",
        json!("C"),
    );
}

#[test]
fn element_name_by_number() {
    assert_field(
        "element",
        "by_number",
        json!({"z": 79}),
        "name",
        json!("gold"),
    );
}

#[test]
fn element_mass() {
    if let Some(v) = call("element", "by_name", json!({"name": "hydrogen"})) {
        let mass = v.get("mass").and_then(|m| m.as_f64()).expect("mass");
        assert!((mass - 1.008).abs() < 1e-3);
    }
}

// ── http_status ───────────────────────────────────────────────────────────────

#[test]
fn http_status_404() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 404}),
        "reason",
        json!("Not Found"),
    );
}

#[test]
fn http_status_200() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 200}),
        "reason",
        json!("OK"),
    );
}

#[test]
fn http_status_500() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 500}),
        "reason",
        json!("Internal Server Error"),
    );
}

#[test]
fn http_status_301() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 301}),
        "reason",
        json!("Moved Permanently"),
    );
}

#[test]
fn http_status_403_category() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 403}),
        "category",
        json!("4xx"),
    );
}

#[test]
fn http_status_unknown() {
    // 999 is not a real code — expert declines.
    assert!(call("http_status", "lookup", json!({"code": 999})).is_none());
}

// ── isbn ──────────────────────────────────────────────────────────────────────

#[test]
fn isbn_valid_13() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "978-0-596-52068-7"}),
        "valid",
        json!(true),
    );
}

#[test]
fn isbn_valid_10() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "0-306-40615-2"}),
        "valid",
        json!(true),
    );
}

#[test]
fn isbn_invalid() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "978-0-000-00000-0"}),
        "valid",
        json!(false),
    );
}

// ── luhn ──────────────────────────────────────────────────────────────────────

#[test]
fn luhn_visa_valid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "4532015112830366"}),
        json!(true),
    );
}

#[test]
fn luhn_amex_valid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "378282246310005"}),
        json!(true),
    );
}

#[test]
fn luhn_invalid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "1234567890123456"}),
        json!(false),
    );
}

#[test]
fn luhn_check_digit() {
    assert_eq_expert(
        "luhn",
        "generate_check_digit",
        json!({"number": "453201511283036"}),
        json!(6),
    );
}

#[test]
fn luhn_card_type_amex() {
    assert_eq_expert(
        "luhn",
        "card_type",
        json!({"number": "378282246310005"}),
        json!("amex"),
    );
}

// ── markov ────────────────────────────────────────────────────────────────────

#[test]
fn markov_expected_value() {
    assert_approx(
        "markov",
        "expected_value",
        json!({"outcomes": [1, 2, 3], "probabilities": [0.2, 0.5, 0.3]}),
        2.1,
        1e-9,
    );
}

#[test]
fn markov_steady_state() {
    // Symmetric-ish test: equal columns of the transpose fixed point.
    if let Some(v) = call(
        "markov",
        "steady_state",
        json!({"matrix": [[0.5, 0.5], [0.3, 0.7]]}),
    ) {
        let arr = v.as_array().expect("array");
        let sum: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "probabilities must sum to 1, got {sum}"
        );
    }
}

// ── conway ────────────────────────────────────────────────────────────────────

#[test]
fn conway_blinker_one_gen() {
    if let Some(v) = call(
        "conway",
        "simulate",
        json!({
            "grid": [[0,0,0],[1,1,1],[0,0,0]],
            "generations": 1
        }),
    ) {
        assert_eq!(v.get("live").and_then(|x| x.as_i64()), Some(3));
    }
}

#[test]
fn conway_still_block() {
    // A 2×2 block is a still life — stays at 4 live cells.
    if let Some(v) = call(
        "conway",
        "simulate",
        json!({
            "grid": [[1,1],[1,1]],
            "generations": 1
        }),
    ) {
        assert_eq!(v.get("live").and_then(|x| x.as_i64()), Some(4));
    }
}

// ── dijkstra ──────────────────────────────────────────────────────────────────

#[test]
fn dijkstra_shortest_path() {
    assert_field(
        "dijkstra",
        "shortest_path",
        json!({"edges": [["A","C",2],["C","B",1],["A","B",5]], "from": "A", "to": "B"}),
        "distance",
        json!(3),
    );
}

#[test]
fn dijkstra_reachable() {
    assert_field(
        "dijkstra",
        "reachable",
        json!({"edges": [["A","B"],["B","C"]], "from": "A", "to": "C"}),
        "reachable",
        json!(true),
    );
}

#[test]
fn dijkstra_mst() {
    assert_field(
        "dijkstra",
        "mst",
        json!({"edges": [["A","B",4],["B","C",2],["A","C",5]]}),
        "weight",
        json!(6),
    );
}

// ── graph ─────────────────────────────────────────────────────────────────────

#[test]
fn graph_most_central() {
    assert_field(
        "graph",
        "most_central",
        json!({"edges": [["A","B"],["B","C"],["B","D"],["B","E"]]}),
        "node",
        json!("B"),
    );
}

#[test]
fn graph_cycle_detected() {
    assert_eq_expert(
        "graph",
        "has_cycle",
        json!({"edges": [["A","B"],["B","C"],["C","A"]]}),
        json!(true),
    );
}

#[test]
fn graph_connected_components() {
    assert_eq_expert(
        "graph",
        "connected_components",
        json!({"edges": [["A","B"],["C","D"]]}),
        json!(2),
    );
}

#[test]
fn graph_bipartite_yes() {
    assert_eq_expert(
        "graph",
        "is_bipartite",
        json!({"edges": [["A","B"],["B","C"],["C","D"]]}),
        json!(true),
    );
}

// ── sql ───────────────────────────────────────────────────────────────────────

#[test]
fn sql_count() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE t (x int); INSERT INTO t VALUES (1); INSERT INTO t VALUES (2); SELECT COUNT(*) FROM t"}),
        json!(2),
    );
}

#[test]
fn sql_sum() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE s (v int); INSERT INTO s VALUES (10); INSERT INTO s VALUES (20); INSERT INTO s VALUES (30); SELECT SUM(v) FROM s"}),
        json!(60),
    );
}

#[test]
fn sql_select_with_where() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE u (id int, name text); INSERT INTO u VALUES (1, 'Alice'); INSERT INTO u VALUES (2, 'Bob'); SELECT name FROM u WHERE id = 2"}),
        json!("Bob"),
    );
}

#[test]
fn sql_avg() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE a (n int); INSERT INTO a VALUES (10); INSERT INTO a VALUES (20); SELECT AVG(n) FROM a"}),
        json!(15),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional op coverage — at least one test per advertised op.
// ─────────────────────────────────────────────────────────────────────────────

// arithmetic — remaining ops

#[test]
fn arithmetic_is_perfect_square_true() {
    assert_eq_expert(
        "arithmetic",
        "is_perfect_square",
        json!({"n": 49}),
        json!(true),
    );
}

#[test]
fn arithmetic_is_perfect_square_false() {
    assert_eq_expert(
        "arithmetic",
        "is_perfect_square",
        json!({"n": 50}),
        json!(false),
    );
}

#[test]
fn arithmetic_from_base_hex() {
    assert_eq_expert(
        "arithmetic",
        "from_base",
        json!({"s": "ff", "base": 16}),
        json!(255),
    );
}

#[test]
fn arithmetic_from_base_binary() {
    assert_eq_expert(
        "arithmetic",
        "from_base",
        json!({"s": "1010", "base": 2}),
        json!(10),
    );
}

#[test]
fn arithmetic_percent_increase() {
    assert_approx(
        "arithmetic",
        "percent_increase",
        json!({"n": 100, "pct": 20}),
        120.0,
        1e-9,
    );
}

#[test]
fn arithmetic_percent_decrease() {
    assert_approx(
        "arithmetic",
        "percent_decrease",
        json!({"n": 100, "pct": 25}),
        75.0,
        1e-9,
    );
}

// unit — remaining ops

#[test]
fn unit_info_km() {
    if let Some(v) = call("unit", "unit_info", json!({"unit": "km"})) {
        assert_eq!(v.get("group").and_then(|g| g.as_str()), Some("length"));
        assert_eq!(v.get("to_si").and_then(|x| x.as_f64()), Some(1000.0));
    }
}

#[test]
fn unit_list_length_group() {
    if let Some(v) = call("unit", "list_units", json!({"group": "length"})) {
        let ids: Vec<&str> = v
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        assert!(ids.contains(&"km"));
        assert!(ids.contains(&"mi"));
        assert!(
            !ids.contains(&"kg"),
            "length group must not contain mass unit"
        );
    }
}

#[test]
fn unit_list_all() {
    if let Some(v) = call("unit", "list_units", json!({})) {
        let arr = v.as_array().expect("array");
        assert!(arr.len() > 20, "expected >20 units, got {}", arr.len());
    }
}

// statistics — remaining ops

#[test]
fn statistics_variance() {
    assert_approx(
        "statistics",
        "variance",
        json!({"values": [2,4,4,4,5,5,7,9]}),
        4.0,
        1e-12,
    );
}

#[test]
fn statistics_sum() {
    assert_approx(
        "statistics",
        "sum",
        json!({"values": [1,2,3,4,5]}),
        15.0,
        1e-12,
    );
}

#[test]
fn statistics_range() {
    assert_approx(
        "statistics",
        "range",
        json!({"values": [1,2,3,4,10]}),
        9.0,
        1e-12,
    );
}

// geometry — remaining ops

#[test]
fn geometry_circle_circumference() {
    assert_approx(
        "geometry",
        "circle_circumference",
        json!({"r": 10}),
        std::f64::consts::TAU * 10.0,
        1e-9,
    );
}

#[test]
fn geometry_circle_diameter() {
    assert_approx("geometry", "circle_diameter", json!({"r": 5}), 10.0, 1e-12);
}

#[test]
fn geometry_sphere_surface_area() {
    assert_approx(
        "geometry",
        "sphere_surface_area",
        json!({"r": 3}),
        4.0 * std::f64::consts::PI * 9.0,
        1e-9,
    );
}

#[test]
fn geometry_cylinder_volume() {
    assert_approx(
        "geometry",
        "cylinder_volume",
        json!({"r": 2, "h": 5}),
        std::f64::consts::PI * 4.0 * 5.0,
        1e-9,
    );
}

#[test]
fn geometry_cone_volume() {
    assert_approx(
        "geometry",
        "cone_volume",
        json!({"r": 3, "h": 4}),
        std::f64::consts::PI * 9.0 * 4.0 / 3.0,
        1e-9,
    );
}

#[test]
fn geometry_cube_volume() {
    assert_approx("geometry", "cube_volume", json!({"s": 3}), 27.0, 1e-12);
}

#[test]
fn geometry_box_volume() {
    assert_approx(
        "geometry",
        "box_volume",
        json!({"l": 2, "w": 3, "h": 4}),
        24.0,
        1e-12,
    );
}

#[test]
fn geometry_square_area() {
    assert_approx("geometry", "square_area", json!({"s": 5}), 25.0, 1e-12);
}

#[test]
fn geometry_square_perimeter() {
    assert_approx("geometry", "square_perimeter", json!({"s": 5}), 20.0, 1e-12);
}

#[test]
fn geometry_rectangle_area() {
    assert_approx(
        "geometry",
        "rectangle_area",
        json!({"l": 4, "w": 5}),
        20.0,
        1e-12,
    );
}

#[test]
fn geometry_triangle_area_heron() {
    // 3-4-5 right triangle has area 6.
    assert_approx(
        "geometry",
        "triangle_area_heron",
        json!({"a": 3, "b": 4, "c": 5}),
        6.0,
        1e-9,
    );
}

#[test]
fn geometry_trapezoid_area() {
    assert_approx(
        "geometry",
        "trapezoid_area",
        json!({"a": 3, "b": 5, "h": 4}),
        16.0,
        1e-12,
    );
}

#[test]
fn geometry_ellipse_area() {
    assert_approx(
        "geometry",
        "ellipse_area",
        json!({"a": 3, "b": 5}),
        std::f64::consts::PI * 15.0,
        1e-9,
    );
}

// trig — remaining ops

#[test]
fn trig_acos_one() {
    assert_approx("trig", "acos", json!({"x": 1}), 0.0, 1e-12);
}

#[test]
fn trig_atan_one() {
    assert_approx(
        "trig",
        "atan",
        json!({"x": 1}),
        std::f64::consts::FRAC_PI_4,
        1e-12,
    );
}

#[test]
fn trig_sec_zero() {
    assert_approx("trig", "sec", json!({"x": 0}), 1.0, 1e-12);
}

#[test]
fn trig_csc_pi_half() {
    assert_approx(
        "trig",
        "csc",
        json!({"x": std::f64::consts::FRAC_PI_2}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_cot_pi_quarter() {
    assert_approx(
        "trig",
        "cot",
        json!({"x": std::f64::consts::FRAC_PI_4}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_rad_to_deg() {
    assert_approx(
        "trig",
        "rad_to_deg",
        json!({"rad": std::f64::consts::PI}),
        180.0,
        1e-9,
    );
}

#[test]
fn trig_asin_out_of_range() {
    // asin only defined on [-1, 1]; expert returns explicit null.
    assert_eq_expert("trig", "asin", json!({"x": 2}), Value::Null);
}

// string_ops — remaining ops

#[test]
fn string_ops_rot13() {
    assert_eq_expert("string_ops", "rot13", json!({"s": "hello"}), json!("uryyb"));
}

#[test]
fn string_ops_lowercase() {
    assert_eq_expert(
        "string_ops",
        "lowercase",
        json!({"s": "HELLO"}),
        json!("hello"),
    );
}

#[test]
fn string_ops_length() {
    assert_eq_expert("string_ops", "length", json!({"s": "hello"}), json!(5));
}

#[test]
fn string_ops_length_unicode() {
    // `é` is one character but multiple bytes.
    assert_eq_expert("string_ops", "length", json!({"s": "café"}), json!(4));
}

#[test]
fn string_ops_count_char() {
    assert_eq_expert(
        "string_ops",
        "count_char",
        json!({"s": "banana", "ch": "a"}),
        json!(3),
    );
}

#[test]
fn string_ops_count_substring() {
    // `matches()` counts non-overlapping occurrences: "aaaa" contains "aa" twice.
    assert_eq_expert(
        "string_ops",
        "count_substring",
        json!({"s": "aaaa", "needle": "aa"}),
        json!(2),
    );
}

#[test]
fn string_ops_count_words() {
    assert_eq_expert(
        "string_ops",
        "count_words",
        json!({"s": "hello world foo bar"}),
        json!(4),
    );
}

#[test]
fn string_ops_contains_true() {
    assert_eq_expert(
        "string_ops",
        "contains",
        json!({"s": "hello", "needle": "ell"}),
        json!(true),
    );
}

#[test]
fn string_ops_contains_false() {
    assert_eq_expert(
        "string_ops",
        "contains",
        json!({"s": "hello", "needle": "xyz"}),
        json!(false),
    );
}

#[test]
fn string_ops_starts_with() {
    assert_eq_expert(
        "string_ops",
        "starts_with",
        json!({"s": "hello", "prefix": "hel"}),
        json!(true),
    );
}

#[test]
fn string_ops_ends_with() {
    assert_eq_expert(
        "string_ops",
        "ends_with",
        json!({"s": "hello", "suffix": "llo"}),
        json!(true),
    );
}

// hash — remaining ops

#[test]
fn hash_hex_decode() {
    assert_eq_expert("hash", "hex_decode", json!({"s": "616263"}), json!("abc"));
}

#[test]
fn hash_hex_decode_with_prefix() {
    assert_eq_expert("hash", "hex_decode", json!({"s": "0x616263"}), json!("abc"));
}

#[test]
fn hash_url_decode() {
    assert_eq_expert(
        "hash",
        "url_decode",
        json!({"s": "hello%20world"}),
        json!("hello world"),
    );
}

// logic — direct simplify check

#[test]
fn logic_simplify_double_negation() {
    assert_eq_expert(
        "logic",
        "simplify",
        json!({"expr": "NOT NOT A"}),
        json!("A"),
    );
}

// finance — remaining ops

#[test]
fn finance_present_value() {
    // PV of 1100 at 10% for 1 year = 1000.
    assert_approx(
        "finance",
        "present_value",
        json!({"fv": 1100, "rate_pct": 10, "years": 1}),
        1000.0,
        1e-9,
    );
}

#[test]
fn finance_simple_interest() {
    assert_approx(
        "finance",
        "simple_interest",
        json!({"principal": 1000, "rate_pct": 5, "years": 3}),
        150.0,
        1e-9,
    );
}

#[test]
fn finance_mortgage_payment() {
    // 100k at 6% over 30 years ≈ $599.55/mo.
    assert_approx(
        "finance",
        "mortgage_payment",
        json!({"principal": 100000, "annual_rate_pct": 6, "years": 30}),
        599.55,
        1.0,
    );
}

#[test]
fn finance_bayes() {
    // P(B|A)=0.9, P(A)=0.01, P(B)=0.1 → P(A|B) = 0.09.
    assert_approx(
        "finance",
        "bayes",
        json!({"p_b_given_a": 0.9, "p_a": 0.01, "p_b": 0.1}),
        0.09,
        1e-9,
    );
}

#[test]
fn finance_bayes_p_b_zero() {
    assert_eq_expert(
        "finance",
        "bayes",
        json!({"p_b_given_a": 0.9, "p_a": 0.1, "p_b": 0}),
        Value::Null,
    );
}

// element — remaining ops

#[test]
fn element_by_symbol() {
    assert_field(
        "element",
        "by_symbol",
        json!({"symbol": "Au"}),
        "name",
        json!("gold"),
    );
}

#[test]
fn element_by_symbol_case_insensitive() {
    assert_field(
        "element",
        "by_symbol",
        json!({"symbol": "fe"}),
        "name",
        json!("iron"),
    );
}

#[test]
fn element_list() {
    if let Some(v) = call("element", "list", json!({})) {
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 118, "expected 118 elements");
    }
}

// isbn — conversion ops

#[test]
fn isbn_isbn10_to_isbn13() {
    assert_eq_expert(
        "isbn",
        "isbn10_to_isbn13",
        json!({"isbn": "0-306-40615-2"}),
        json!("9780306406157"),
    );
}

#[test]
fn isbn_isbn13_to_isbn10() {
    assert_eq_expert(
        "isbn",
        "isbn13_to_isbn10",
        json!({"isbn": "978-0-596-52068-7"}),
        json!("0596520689"),
    );
}

// conway — step op

#[test]
fn conway_step_blinker() {
    // A horizontal blinker → vertical blinker after one step.
    if let Some(v) = call("conway", "step", json!({"grid": [[0,0,0],[1,1,1],[0,0,0]]})) {
        assert_eq!(v, json!([[0, 1, 0], [0, 1, 0], [0, 1, 0]]));
    }
}

// graph — remaining ops

#[test]
fn graph_topological_sort_dag() {
    if let Some(v) = call(
        "graph",
        "topological_sort",
        json!({
            "edges": [["A","B"],["B","C"],["A","C"]],
            "directed": true
        }),
    ) {
        let order: Vec<&str> = v
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        // Any valid topo order places A before B and B before C.
        let ai = order.iter().position(|&n| n == "A").expect("A present");
        let bi = order.iter().position(|&n| n == "B").expect("B present");
        let ci = order.iter().position(|&n| n == "C").expect("C present");
        assert!(ai < bi && bi < ci, "invalid topo order: {:?}", order);
    }
}

#[test]
fn graph_topological_sort_cycle_returns_null() {
    assert_eq_expert(
        "graph",
        "topological_sort",
        json!({"edges": [["A","B"],["B","C"],["C","A"]], "directed": true}),
        Value::Null,
    );
}

#[test]
fn graph_degrees() {
    if let Some(v) = call(
        "graph",
        "degrees",
        json!({"edges": [["A","B"],["B","C"],["B","D"]]}),
    ) {
        let arr = v.as_array().expect("array");
        let b_degree = arr
            .iter()
            .find(|x| x.get("node").and_then(|n| n.as_str()) == Some("B"))
            .and_then(|x| x.get("degree").and_then(|d| d.as_i64()));
        assert_eq!(b_degree, Some(3));
    }
}

#[test]
fn graph_bipartite_no() {
    // Odd cycle is not bipartite.
    assert_eq_expert(
        "graph",
        "is_bipartite",
        json!({"edges": [["A","B"],["B","C"],["C","A"]]}),
        json!(false),
    );
}

// ── registry-level tests ──────────────────────────────────────────────────────

#[test]
fn registry_load_dir_tier_order() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    if reg.len() < 2 {
        return;
    }
    let tiers: Vec<u8> = reg.list().iter().map(|m| m.tier).collect();
    let mut sorted = tiers.clone();
    sorted.sort();
    assert_eq!(tiers, sorted, "experts must be sorted by tier ascending");
}

#[test]
fn registry_dispatches_by_op() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    let result = reg.call("mul", &json!({"a": 6, "b": 7}));
    assert!(result.is_some(), "arithmetic.mul should dispatch");
    assert_eq!(result.unwrap().value, json!(42.0));
}

#[test]
fn registry_unknown_op_returns_none() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    assert!(reg.call("nonexistent_op_abc_xyz", &json!({})).is_none());
}

#[test]
fn registry_all_experts_have_metadata() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    for meta in reg.list() {
        assert!(!meta.id.is_empty(), "id must not be empty");
        assert!(
            !meta.description.is_empty(),
            "description must not be empty"
        );
        assert!(!meta.version.is_empty(), "version must not be empty");
        assert!(meta.tier >= 1, "tier must be >= 1");
        assert!(!meta.ops.is_empty(), "expert {} advertises no ops", meta.id);
    }
}

#[test]
fn registry_memory_stable_across_many_calls() {
    // Without the larql_dealloc pairing in caller.rs, arithmetic's linear
    // memory grew by ~140 bytes per call (op + args + result strings leaked).
    // This test locks that regression down.
    let path = wasm("arithmetic");
    if !path.exists() {
        return;
    }
    let mut reg = ExpertRegistry::default();
    reg.load_file(&path).expect("load arithmetic");

    // Warm up so the expert is instantiated and its allocator has reached
    // steady state.
    for _ in 0..32 {
        let _ = reg.call("gcd", &json!({"a": 144, "b": 60}));
    }
    let pages_before = reg
        .wasm_info_for("arithmetic")
        .expect("present")
        .memory_pages;

    // 2000 calls was empirically enough pre-fix to grow memory by 3+ pages.
    for _ in 0..2000 {
        let _ = reg.call("gcd", &json!({"a": 144, "b": 60}));
    }
    let pages_after = reg
        .wasm_info_for("arithmetic")
        .expect("present")
        .memory_pages;

    assert_eq!(
        pages_before, pages_after,
        "arithmetic linear memory grew from {} to {} pages across 2000 calls — \
         dealloc is probably not paired in caller.rs",
        pages_before, pages_after
    );
}

#[test]
fn module_cache_file_is_written_and_reused() {
    // Exercise the .cwasm precompile cache against a private copy. Other
    // expert tests load the same fixture in parallel, and the cache lives next
    // to the .wasm, so using the shared fixture makes this mtime assertion
    // race-prone.
    let wasm_path = wasm("arithmetic");
    if !wasm_path.exists() {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let private_wasm_path = tmp.path().join("larql_expert_arithmetic.wasm");
    std::fs::copy(&wasm_path, &private_wasm_path).expect("copy wasm fixture");
    let cwasm_path = private_wasm_path.with_extension("cwasm");

    // First load compiles and writes the cache.
    {
        let mut reg = ExpertRegistry::default();
        reg.load_file(&private_wasm_path).expect("first load");
    }
    assert!(
        cwasm_path.exists(),
        "expected cache file {:?} to be created on first load",
        cwasm_path
    );

    // Second load should succeed against the cached artifact. We can't
    // reliably assert a speedup in a unit test, but we can at least confirm
    // the cached file is still present and the registry still functions.
    let cwasm_mtime_before = std::fs::metadata(&cwasm_path).unwrap().modified().unwrap();
    {
        let mut reg = ExpertRegistry::default();
        reg.load_file(&private_wasm_path).expect("second load");
        let result = reg
            .call("gcd", &json!({"a": 12, "b": 8}))
            .expect("gcd dispatches");
        assert_eq!(result.value, json!(4));
    }
    let cwasm_mtime_after = std::fs::metadata(&cwasm_path).unwrap().modified().unwrap();
    assert_eq!(
        cwasm_mtime_before, cwasm_mtime_after,
        "cache file should be reused on second load, not rewritten"
    );
}

#[test]
fn registry_experts_are_lazy_instantiated() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");

    // Freshly loaded: nothing instantiated yet, zero linear memory pages.
    for info in reg.wasm_infos() {
        assert!(
            !info.instantiated,
            "expert {:?} should not be instantiated at load",
            info.path
        );
        assert_eq!(info.memory_pages, 0);
    }

    // One call to arithmetic.gcd instantiates only arithmetic.
    let _ = reg
        .call("gcd", &json!({"a": 12, "b": 8}))
        .expect("gcd dispatches");
    let arith = reg.wasm_info_for("arithmetic").expect("arithmetic present");
    assert!(arith.instantiated);
    assert!(arith.memory_pages > 0);

    // No other expert should be instantiated yet.
    let still_cold: Vec<String> = reg
        .wasm_infos()
        .into_iter()
        .filter(|i| !i.instantiated)
        .map(|i| i.path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(
        still_cold.len() >= 17,
        "expected ≥17 cold experts, got {}",
        still_cold.len()
    );

    // evict_all drops every live instance.
    reg.evict_all();
    for info in reg.wasm_infos() {
        assert!(!info.instantiated, "evict_all should drop {:?}", info.path);
    }

    // Calls work again after eviction — recompilation is not required.
    let r = reg
        .call("gcd", &json!({"a": 12, "b": 8}))
        .expect("gcd still dispatches");
    assert_eq!(r.value, json!(4));
}

#[test]
fn registry_ops_are_discoverable() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    let ops = reg.ops();
    // A few specific ops we expect to be present somewhere.
    for expected in &[
        "add",
        "gcd",
        "base64_encode",
        "convert",
        "lookup",
        "execute",
    ] {
        assert!(
            ops.contains(expected),
            "op {:?} missing from registry ops",
            expected
        );
    }
}
