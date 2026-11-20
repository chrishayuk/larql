//! Datetime kernel — date/time arithmetic via `time`.
//!
//! # Syntax
//!
//! | Expression | Result |
//! |---|---|
//! | `days_between(YYYY-MM-DD, YYYY-MM-DD)` | integer days (second - first) |
//! | `add_days(YYYY-MM-DD, N)` | YYYY-MM-DD + N days |
//! | `weekday(YYYY-MM-DD)` | `Mon`..`Sun` |
//! | `parse_date(YYYY-MM-DD)` | canonical `YYYY-MM-DD` echo / validate |
//!
//! # Bounded cost
//!
//! Each operation is O(1). Dates must be in the gregorian range chrono
//! supports (±262k years). Outside that → `KernelError::OutOfRange`.
//!
//! # Examples
//!
//! ```
//! use model_compute::native::{DateTimeKernel, Kernel};
//! let k = DateTimeKernel;
//! assert_eq!(k.invoke("days_between(2026-01-01, 2026-04-16)").unwrap(), "105");
//! assert_eq!(k.invoke("weekday(2026-04-16)").unwrap(), "Thu");
//! ```

use time::{Date, Duration, Month};

use super::{Kernel, KernelError};

pub struct DateTimeKernel;

impl Kernel for DateTimeKernel {
    fn name(&self) -> &'static str {
        "datetime"
    }

    fn invoke(&self, expr: &str) -> Result<String, KernelError> {
        let (head, rest) = split_call(expr)?;
        let args: Vec<&str> = split_args(rest);

        match head {
            "days_between" => {
                expect_args(head, &args, 2)?;
                let a = parse_date(args[0])?;
                let b = parse_date(args[1])?;
                Ok((b - a).whole_days().to_string())
            }
            "add_days" => {
                expect_args(head, &args, 2)?;
                let d = parse_date(args[0])?;
                let n: i64 = args[1].trim().parse().map_err(|_| {
                    KernelError::Parse(format!("add_days: expected integer, got {:?}", args[1]))
                })?;
                let overflow =
                    || KernelError::OutOfRange(format!("add_days({}, {}) overflow", args[0], n));
                let span = n
                    .checked_mul(SECONDS_PER_DAY)
                    .map(Duration::seconds)
                    .ok_or_else(overflow)?;
                let result = d.checked_add(span).ok_or_else(overflow)?;
                Ok(format_date(result))
            }
            "weekday" => {
                expect_args(head, &args, 1)?;
                let d = parse_date(args[0])?;
                Ok(short_weekday(d))
            }
            "parse_date" => {
                expect_args(head, &args, 1)?;
                let d = parse_date(args[0])?;
                Ok(format_date(d))
            }
            _ => Err(KernelError::Unsupported(format!(
                "datetime: unknown function {:?}",
                head
            ))),
        }
    }
}

fn split_call(expr: &str) -> Result<(&str, &str), KernelError> {
    let expr = expr.trim();
    let open = expr.find('(').ok_or_else(|| {
        KernelError::Parse(format!("datetime: expected `name(args)`, got {:?}", expr))
    })?;
    if !expr.ends_with(')') {
        return Err(KernelError::Parse(format!(
            "datetime: missing closing paren in {:?}",
            expr
        )));
    }
    Ok((&expr[..open], &expr[open + 1..expr.len() - 1]))
}

fn split_args(s: &str) -> Vec<&str> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    s.split(',').map(|a| a.trim()).collect()
}

fn expect_args(name: &str, args: &[&str], expected: usize) -> Result<(), KernelError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(KernelError::Parse(format!(
            "{}: expected {} args, got {}",
            name,
            expected,
            args.len()
        )))
    }
}

const SECONDS_PER_DAY: i64 = 86_400;

/// `YYYY-MM-DD`: a year of one or more digits with an optional leading `-`,
/// then a month and a day of one or two digits each. The calendar check
/// (month range, day in month, leap years) is `Date::from_calendar_date`'s.
fn parse_date(s: &str) -> Result<Date, KernelError> {
    let invalid = |why: &str| KernelError::Parse(format!("invalid date {:?}: {}", s, why));
    let text = s.trim();
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let mut parts = body.split('-');
    let (Some(y), Some(m), Some(d), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(invalid("expected YYYY-MM-DD"));
    };
    let digits = |field: &str, max_len: usize| -> Result<u32, KernelError> {
        if field.is_empty() || field.len() > max_len || !field.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("expected YYYY-MM-DD"));
        }
        field.parse().map_err(|_| invalid("expected YYYY-MM-DD"))
    };
    let year = digits(y, 6)? as i32;
    let year = if negative { -year } else { year };
    let month = Month::try_from(digits(m, 2)? as u8).map_err(|e| invalid(&e.to_string()))?;
    let day = digits(d, 2)? as u8;
    Date::from_calendar_date(year, month, day).map_err(|e| invalid(&e.to_string()))
}

/// `YYYY-MM-DD`, the year zero-padded to four digits.
fn format_date(d: Date) -> String {
    let year = d.year();
    let sign = if year < 0 { "-" } else { "" };
    format!(
        "{sign}{:04}-{:02}-{:02}",
        year.unsigned_abs(),
        u8::from(d.month()),
        d.day()
    )
}

/// The three-letter weekday (`Mon`..`Sun`), the kernel's output since it
/// was first written.
fn short_weekday(d: Date) -> String {
    d.weekday().to_string()[..3].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_between_forward() {
        let k = DateTimeKernel;
        assert_eq!(
            k.invoke("days_between(2026-01-01, 2026-04-16)").unwrap(),
            "105"
        );
    }

    #[test]
    fn days_between_negative_when_reversed() {
        let k = DateTimeKernel;
        assert_eq!(
            k.invoke("days_between(2026-04-16, 2026-01-01)").unwrap(),
            "-105"
        );
    }

    #[test]
    fn add_days_positive_and_negative() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("add_days(2026-04-16, 7)").unwrap(), "2026-04-23");
        assert_eq!(k.invoke("add_days(2026-04-16, -16)").unwrap(), "2026-03-31");
    }

    #[test]
    fn weekday_known() {
        let k = DateTimeKernel;
        // 2026-04-16 is a Thursday
        assert_eq!(k.invoke("weekday(2026-04-16)").unwrap(), "Thu");
    }

    #[test]
    fn invalid_date_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("add_days(2026-02-30, 1)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn unknown_function_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("nonexistent(2026-04-16)").unwrap_err();
        assert!(matches!(err, KernelError::Unsupported(_)));
    }

    #[test]
    fn leap_year_day_count() {
        let k = DateTimeKernel;
        // 2024 is a leap year — Feb 29 exists
        assert_eq!(k.invoke("weekday(2024-02-29)").unwrap(), "Thu");
        // 2025 is not — Feb 29 must reject
        let err = k.invoke("weekday(2025-02-29)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
        // 365 days across non-leap 2025; 366 across leap 2024
        assert_eq!(
            k.invoke("days_between(2025-01-01, 2026-01-01)").unwrap(),
            "365"
        );
        assert_eq!(
            k.invoke("days_between(2024-01-01, 2025-01-01)").unwrap(),
            "366"
        );
    }

    #[test]
    fn year_boundary_add_days() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("add_days(2025-12-31, 1)").unwrap(), "2026-01-01");
        assert_eq!(k.invoke("add_days(2026-01-01, -1)").unwrap(), "2025-12-31");
    }

    #[test]
    fn wrong_arg_count_parse_error() {
        let k = DateTimeKernel;
        let err = k.invoke("days_between(2026-01-01)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn missing_closing_paren_errors() {
        let k = DateTimeKernel;
        let err = k.invoke("weekday(2026-04-16").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn parse_date_roundtrips() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("parse_date(2026-04-16)").unwrap(), "2026-04-16");
    }

    #[test]
    fn weekday_names_are_three_letters_for_every_day() {
        let k = DateTimeKernel;
        let week: Vec<String> = (13..=19)
            .map(|d| k.invoke(&format!("weekday(2026-04-{d})")).unwrap())
            .collect();
        assert_eq!(week, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]);
    }

    #[test]
    fn malformed_dates_are_parse_errors() {
        let k = DateTimeKernel;
        for bad in [
            "2026/04/16",
            "2026-04",
            "2026-04-16-01",
            "2026-13-01",
            "2026-00-10",
            "2026-04-x",
            "2026-004-16",
            "",
        ] {
            let err = k.invoke(&format!("parse_date({bad})")).unwrap_err();
            assert!(matches!(err, KernelError::Parse(_)), "{bad:?}: {err:?}");
        }
    }

    #[test]
    fn unpadded_month_and_day_are_accepted_and_formatted_padded() {
        let k = DateTimeKernel;
        assert_eq!(k.invoke("parse_date(2026-4-6)").unwrap(), "2026-04-06");
    }

    #[test]
    fn add_days_past_the_calendar_range_is_out_of_range() {
        let k = DateTimeKernel;
        for n in ["100000000", "-100000000", "9223372036854775807"] {
            let err = k.invoke(&format!("add_days(2026-04-16, {n})")).unwrap_err();
            assert!(matches!(err, KernelError::OutOfRange(_)), "{n}: {err:?}");
        }
    }

    #[test]
    fn add_days_rejects_a_non_integer_count() {
        let k = DateTimeKernel;
        let err = k.invoke("add_days(2026-04-16, 1.5)").unwrap_err();
        assert!(matches!(err, KernelError::Parse(_)));
    }

    #[test]
    fn years_are_zero_padded_and_signed() {
        assert_eq!(
            format_date(Date::from_calendar_date(33, Month::March, 7).unwrap()),
            "0033-03-07"
        );
        assert_eq!(
            format_date(Date::from_calendar_date(-44, Month::March, 15).unwrap()),
            "-0044-03-15"
        );
        assert_eq!(
            format_date(parse_date("-0044-03-15").unwrap()),
            "-0044-03-15"
        );
    }
}
