## ADDED Requirements

### Requirement: Kernel trait and registry contract

The `model_compute::native::Kernel` trait SHALL define the single
abstraction for in-process bounded-cost compute: every kernel exposes a
`name() -> &'static str` identifier and an `invoke(&self, expr: &str) ->
Result<String, KernelError>` entry point. `KernelRegistry` SHALL provide
name-keyed dispatch and MUST allow callers to register, override, and
look up kernels by name. `KernelRegistry::with_defaults()` MUST preload
exactly the V1 kernels (`arithmetic`, `datetime`); no other kernels MAY
be implicitly registered. `KernelError` SHALL distinguish the variants
`NotFound`, `Parse`, `Eval`, `OutOfRange`, and `Unsupported`.

#### Scenario: Default registry exposes arithmetic and datetime by name
- **WHEN** `KernelRegistry::with_defaults()` is constructed and `names()` is read
- **THEN** the result SHALL contain exactly `"arithmetic"` and `"datetime"`
<!-- test: model_compute::native::registry::tests::defaults_have_arithmetic_and_datetime -->

#### Scenario: Unknown kernel name produces NotFound
- **WHEN** `invoke("nonexistent", ...)` is called on the default registry
- **THEN** the call SHALL return `Err(KernelError::NotFound("nonexistent"))`
<!-- test: model_compute::native::registry::tests::not_found_errors_clearly -->

#### Scenario: Registry dispatches to the named kernel
- **WHEN** `invoke("arithmetic", "2 + 3")` is called on the default registry
- **THEN** the call SHALL return `Ok("5")`
<!-- test: model_compute::native::registry::tests::dispatches_to_arithmetic -->

#### Scenario: Custom kernel registers and dispatches
- **WHEN** a caller-provided `Kernel` is registered into a fresh registry
- **THEN** subsequent `invoke()` calls keyed on the kernel's name SHALL route to that implementation
<!-- test: model_compute::native::registry::tests::custom_kernel_registers_and_dispatches -->

#### Scenario: Registering a duplicate name overrides the prior kernel
- **WHEN** a kernel sharing the `"arithmetic"` name is registered after `with_defaults()`
- **THEN** subsequent dispatches to `"arithmetic"` SHALL invoke the override, not the default
<!-- test: model_compute::native::registry::tests::custom_kernel_overrides_default -->

### Requirement: Arithmetic kernel semantics

The `ArithmeticKernel` SHALL evaluate expression strings consisting of
basic operators (`+ - * / %`), `evalexpr` math built-ins (`math::pow`,
`math::sqrt`, `math::ln`, `math::log`, `math::abs`, `min`, `max`,
`floor`, `ceil`, `round`), and the bounded aggregates `sum(lo..hi)`,
`product(lo..hi)`, and `factorial(n)`. Range aggregates SHALL be
half-open in the Rust sense and MUST iterate at most 10⁸ values;
`factorial` SHALL be capped at 20. Evaluation MUST be deterministic and
MUST NOT panic on adversarial input — every error MUST surface as a
`KernelError` variant.

#### Scenario: Basic operators evaluate to integer strings
- **WHEN** `invoke("2 + 3")`, `invoke("100 * 101 / 2")`, and `invoke("(1 + 2) * 4")` are called
- **THEN** the kernel SHALL return `"5"`, `"5050"`, and `"12"` respectively
<!-- test: model_compute::native::arithmetic::tests::basic_ops -->

#### Scenario: Gauss sum over bounded range
- **WHEN** `invoke("sum(1..101)")` is called
- **THEN** the kernel SHALL return `"5050"`
<!-- test: model_compute::native::arithmetic::tests::gauss_sum -->

#### Scenario: Factorial across boundary cases
- **WHEN** `invoke("factorial(0)")`, `invoke("factorial(5)")`, and `invoke("factorial(20)")` are called
- **THEN** the kernel SHALL return `"1"`, `"120"`, and `"2432902008176640000"` respectively
<!-- test: model_compute::native::arithmetic::tests::factorial_cases -->

#### Scenario: Factorial above the cap is rejected
- **WHEN** `invoke("factorial(21)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::OutOfRange(_))`
<!-- test: model_compute::native::arithmetic::tests::factorial_out_of_range -->

#### Scenario: Aggregates compose with the surrounding expression
- **WHEN** `invoke("sum(1..11) * 2")` is called
- **THEN** the kernel SHALL return `"110"`
<!-- test: model_compute::native::arithmetic::tests::aggregate_composes_with_arithmetic -->

#### Scenario: Floating-point math built-ins return formatted output
- **WHEN** `invoke("math::pow(2.0, 10.0)")` and `invoke("math::sqrt(16.0)")` are called
- **THEN** the kernel SHALL return `"1024"` and `"4"` (integer-typed output for whole-number floats)
<!-- test: model_compute::native::arithmetic::tests::float_ops -->

#### Scenario: Range exceeding the iteration cap is rejected
- **WHEN** `invoke("sum(0..200000000)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::OutOfRange(_))`
<!-- test: model_compute::native::arithmetic::tests::sum_range_too_large -->

#### Scenario: Reversed range is rejected
- **WHEN** `invoke("sum(10..5)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::OutOfRange(_))`
<!-- test: model_compute::native::arithmetic::tests::reversed_range_rejected -->

#### Scenario: Empty range yields the additive / multiplicative identity
- **WHEN** `invoke("sum(5..5)")` and `invoke("product(5..5)")` are called
- **THEN** the kernel SHALL return `"0"` and `"1"`
<!-- test: model_compute::native::arithmetic::tests::empty_range_sum_is_zero -->

#### Scenario: Aggregate arguments may themselves be aggregates
- **WHEN** `invoke("sum(factorial(3)..factorial(4))")` is called
- **THEN** inner aggregates SHALL expand to integers before the outer range is parsed and the result SHALL be `"261"`
<!-- test: model_compute::native::arithmetic::tests::nested_aggregates -->

#### Scenario: Negative factorial argument is rejected
- **WHEN** `invoke("factorial(-1)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::OutOfRange(_))`
<!-- test: model_compute::native::arithmetic::tests::factorial_negative_rejected -->

#### Scenario: Malformed range surfaces as Parse error
- **WHEN** `invoke("sum(abc..xyz)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::Parse(_))`
<!-- test: model_compute::native::arithmetic::tests::malformed_range_reports_parse_error -->

#### Scenario: Identifier prefix does not collide with aggregate names
- **WHEN** `invoke("summary(1..10)")` is called
- **THEN** the aggregate handler SHALL NOT match `summary` and the error SHALL be `KernelError::Eval(_)` from the underlying expression engine
<!-- test: model_compute::native::arithmetic::tests::identifier_prefix_does_not_match -->

### Requirement: Datetime kernel semantics

The `DateTimeKernel` SHALL evaluate calls of the form
`name(YYYY-MM-DD, …)` against the chrono Gregorian calendar, supporting
exactly four operations: `days_between(a, b)` (signed integer days,
`b - a`), `add_days(date, n)` (returns `YYYY-MM-DD`), `weekday(date)`
(returns `Mon`..`Sun`), and `parse_date(date)` (canonical echo /
validation). Each call MUST be O(1). Invalid dates, missing parens,
wrong arity, unknown function names, and date arithmetic that overflows
chrono's range MUST surface as `KernelError::Parse`,
`KernelError::Unsupported`, or `KernelError::OutOfRange` rather than
panic.

#### Scenario: Forward day count between two dates
- **WHEN** `invoke("days_between(2026-01-01, 2026-04-16)")` is called
- **THEN** the kernel SHALL return `"105"`
<!-- test: model_compute::native::datetime::tests::days_between_forward -->

#### Scenario: Reversed arguments yield a signed negative count
- **WHEN** `invoke("days_between(2026-04-16, 2026-01-01)")` is called
- **THEN** the kernel SHALL return `"-105"`
<!-- test: model_compute::native::datetime::tests::days_between_negative_when_reversed -->

#### Scenario: add_days handles positive and negative offsets
- **WHEN** `invoke("add_days(2026-04-16, 7)")` and `invoke("add_days(2026-04-16, -16)")` are called
- **THEN** the kernel SHALL return `"2026-04-23"` and `"2026-03-31"`
<!-- test: model_compute::native::datetime::tests::add_days_positive_and_negative -->

#### Scenario: weekday returns the chrono short name
- **WHEN** `invoke("weekday(2026-04-16)")` is called
- **THEN** the kernel SHALL return `"Thu"`
<!-- test: model_compute::native::datetime::tests::weekday_known -->

#### Scenario: Invalid calendar date is rejected
- **WHEN** `invoke("add_days(2026-02-30, 1)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::Parse(_))`
<!-- test: model_compute::native::datetime::tests::invalid_date_errors -->

#### Scenario: Unknown datetime function is rejected
- **WHEN** `invoke("nonexistent(2026-04-16)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::Unsupported(_))`
<!-- test: model_compute::native::datetime::tests::unknown_function_errors -->

#### Scenario: Leap-year handling matches the Gregorian calendar
- **WHEN** `invoke("weekday(2024-02-29)")`, `invoke("weekday(2025-02-29)")`, `invoke("days_between(2025-01-01, 2026-01-01)")`, and `invoke("days_between(2024-01-01, 2025-01-01)")` are called
- **THEN** the kernel SHALL return `"Thu"`, `Err(KernelError::Parse(_))`, `"365"`, and `"366"` respectively
<!-- test: model_compute::native::datetime::tests::leap_year_day_count -->

#### Scenario: add_days crosses year boundaries cleanly
- **WHEN** `invoke("add_days(2025-12-31, 1)")` and `invoke("add_days(2026-01-01, -1)")` are called
- **THEN** the kernel SHALL return `"2026-01-01"` and `"2025-12-31"`
<!-- test: model_compute::native::datetime::tests::year_boundary_add_days -->

#### Scenario: Wrong argument count surfaces as Parse error
- **WHEN** `invoke("days_between(2026-01-01)")` is called
- **THEN** the kernel SHALL return `Err(KernelError::Parse(_))`
<!-- test: model_compute::native::datetime::tests::wrong_arg_count_parse_error -->

#### Scenario: Missing closing paren surfaces as Parse error
- **WHEN** `invoke("weekday(2026-04-16")` is called
- **THEN** the kernel SHALL return `Err(KernelError::Parse(_))`
<!-- test: model_compute::native::datetime::tests::missing_closing_paren_errors -->

#### Scenario: parse_date echoes a canonical YYYY-MM-DD
- **WHEN** `invoke("parse_date(2026-04-16)")` is called
- **THEN** the kernel SHALL return `"2026-04-16"`
<!-- test: model_compute::native::datetime::tests::parse_date_roundtrips -->

### Requirement: Bounded-cost guarantee for native kernels

Every native kernel SHALL execute in deterministic, panic-free,
hard-capped cost. `ArithmeticKernel` MUST cap range aggregates at 10⁸
iterations and factorial at 20. `DateTimeKernel` operations MUST be
O(1). Out-of-range, malformed, or otherwise rejectable inputs MUST be
returned as `KernelError` rather than panicking, so adversarial input
cannot wedge a host process.

#### Scenario: Out-of-range arithmetic input does not panic
- **WHEN** `invoke("sum(0..200000000)")` is called on the arithmetic kernel
- **THEN** the call SHALL return `Err(KernelError::OutOfRange(_))` and the host process SHALL remain live
<!-- test: model_compute::native::arithmetic::tests::sum_range_too_large -->

#### Scenario: Factorial overflow is reported, not panicked
- **WHEN** `invoke("factorial(21)")` or `invoke("factorial(-1)")` is called
- **THEN** the call SHALL return `Err(KernelError::OutOfRange(_))`
<!-- test: model_compute::native::arithmetic::tests::factorial_out_of_range -->
<!-- test: model_compute::native::arithmetic::tests::factorial_negative_rejected -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: model_compute::native::registry::tests::**::* -->
<!-- test: model_compute::native::arithmetic::tests::**::* -->
<!-- test: model_compute::native::datetime::tests::**::* -->
