## ADDED Requirements

### Requirement: Arithmetic expert (tier 1, 18 ops)

The `arithmetic` expert SHALL be a tier-1 WASM module advertising 18
ops covering basic arithmetic (`add`, `sub`, `mul`, `div`, `pow`,
`mod`), number theory (`gcd`, `lcm`, `factorial`, `is_prime`,
`is_perfect_square`), base conversion (`to_base`, `from_base`,
`to_roman`, `from_roman`), and percentages (`percent_of`,
`percent_increase`, `percent_decrease`). Results MUST be typed JSON
values (numbers or booleans), not formatted strings. Division by zero
SHALL surface as the expert declining (registry returns `None`),
never as a panic.

#### Scenario: Basic arithmetic ops return typed numeric values
- **WHEN** `add`, `sub`, `mul`, `div`, `pow`, and `mod` are called with their canonical numeric args
- **THEN** each call SHALL return the arithmetic result as a JSON number
<!-- test: larql_inference::test_experts::arithmetic_add -->
<!-- test: larql_inference::test_experts::arithmetic_subtract -->
<!-- test: larql_inference::test_experts::arithmetic_multiply -->
<!-- test: larql_inference::test_experts::arithmetic_divide -->
<!-- test: larql_inference::test_experts::arithmetic_power -->
<!-- test: larql_inference::test_experts::arithmetic_mod -->

#### Scenario: Division by zero is contained by the sandbox
- **WHEN** `arithmetic` is called with `div` and a zero divisor
- **THEN** the registry SHALL return `None` and the host process SHALL remain live
<!-- test: larql_inference::test_experts::arithmetic_divide_by_zero -->

#### Scenario: Number-theory ops return typed booleans and integers
- **WHEN** `is_prime`, `gcd`, `lcm`, `factorial`, and `is_perfect_square` are called with their canonical args
- **THEN** each call SHALL return a typed JSON boolean or integer (e.g. `is_prime(7) == true`, `gcd(144, 60) == 12`)
<!-- test: larql_inference::test_experts::arithmetic_prime_true -->
<!-- test: larql_inference::test_experts::arithmetic_prime_false -->
<!-- test: larql_inference::test_experts::arithmetic_gcd -->
<!-- test: larql_inference::test_experts::arithmetic_lcm -->
<!-- test: larql_inference::test_experts::arithmetic_factorial -->
<!-- test: larql_inference::test_experts::arithmetic_is_perfect_square_true -->
<!-- test: larql_inference::test_experts::arithmetic_is_perfect_square_false -->

#### Scenario: Base and Roman conversions are deterministic
- **WHEN** `to_base`, `from_base`, `to_roman`, and `from_roman` are called with valid inputs
- **THEN** each call SHALL return the canonical representation as typed JSON (string for symbolic forms, integer for `from_*`)
<!-- test: larql_inference::test_experts::arithmetic_binary -->
<!-- test: larql_inference::test_experts::arithmetic_hex -->
<!-- test: larql_inference::test_experts::arithmetic_from_base_hex -->
<!-- test: larql_inference::test_experts::arithmetic_from_base_binary -->
<!-- test: larql_inference::test_experts::arithmetic_roman_from -->
<!-- test: larql_inference::test_experts::arithmetic_roman_to -->

#### Scenario: Percentage ops compute as JSON numbers
- **WHEN** `percent_of`, `percent_increase`, and `percent_decrease` are called with their canonical args
- **THEN** each call SHALL return the computed percentage as a JSON number
<!-- test: larql_inference::test_experts::arithmetic_percent_of -->
<!-- test: larql_inference::test_experts::arithmetic_percent_increase -->
<!-- test: larql_inference::test_experts::arithmetic_percent_decrease -->

#### Scenario: Op name not advertised by the expert returns None
- **WHEN** an op name not in `arithmetic`'s advertised set reaches the expert via dispatch
- **THEN** the expert SHALL decline by returning `0` from `larql_call` and the registry SHALL surface `None`
<!-- test: larql_inference::test_experts::arithmetic_unknown_op -->

### Requirement: Geometry, trig, and unit experts

The `geometry` expert (tier 1, 18 ops) SHALL provide deterministic
2-D and 3-D figure formulas (areas, perimeters, volumes, surface
areas, hypotenuse). The `trig` expert (tier 1, 11 ops) SHALL provide
`sin/cos/tan/sec/csc/cot` plus their inverses with all angles in
radians, and degree↔radian conversions. The `unit` expert (tier 1,
3 ops) SHALL provide unit conversion across length, mass,
temperature, volume, speed, and energy groups. Conversions across
incompatible groups SHALL return `None`.

#### Scenario: Geometry primitives compute classical formulas
- **WHEN** `circle_area`, `sphere_volume`, `triangle_area`, `rectangle_perimeter`, and `hypotenuse` are called
- **THEN** each call SHALL return the classical formula's value as a JSON number
<!-- test: larql_inference::test_experts::geometry_circle_area -->
<!-- test: larql_inference::test_experts::geometry_sphere_volume -->
<!-- test: larql_inference::test_experts::geometry_triangle_area -->
<!-- test: larql_inference::test_experts::geometry_rectangle_perimeter -->
<!-- test: larql_inference::test_experts::geometry_hypotenuse -->

#### Scenario: Geometry covers the rest of its 18-op surface
- **WHEN** the remaining geometry ops are called (`circle_circumference`, `circle_diameter`, `sphere_surface_area`, `cylinder_volume`, `cone_volume`, `cube_volume`, `box_volume`, `square_area`, `square_perimeter`, `rectangle_area`, `triangle_area_heron`, `trapezoid_area`, `ellipse_area`)
- **THEN** each call SHALL return the formula's value as a JSON number
<!-- test: larql_inference::test_experts::geometry_circle_circumference -->
<!-- test: larql_inference::test_experts::geometry_circle_diameter -->
<!-- test: larql_inference::test_experts::geometry_sphere_surface_area -->
<!-- test: larql_inference::test_experts::geometry_cylinder_volume -->
<!-- test: larql_inference::test_experts::geometry_cone_volume -->
<!-- test: larql_inference::test_experts::geometry_cube_volume -->
<!-- test: larql_inference::test_experts::geometry_box_volume -->
<!-- test: larql_inference::test_experts::geometry_square_area -->
<!-- test: larql_inference::test_experts::geometry_square_perimeter -->
<!-- test: larql_inference::test_experts::geometry_rectangle_area -->
<!-- test: larql_inference::test_experts::geometry_triangle_area_heron -->
<!-- test: larql_inference::test_experts::geometry_trapezoid_area -->
<!-- test: larql_inference::test_experts::geometry_ellipse_area -->

#### Scenario: Trig functions evaluate at canonical angles
- **WHEN** `sin(pi/6)`, `cos(0)`, `tan(pi/4)`, `asin(0.5)`, and `acos(1.0)` are called with angles in radians
- **THEN** each call SHALL return the textbook value as a JSON number
<!-- test: larql_inference::test_experts::trig_sin_pi_6 -->
<!-- test: larql_inference::test_experts::trig_cos_zero -->
<!-- test: larql_inference::test_experts::trig_tan_pi_4 -->
<!-- test: larql_inference::test_experts::trig_asin_half -->
<!-- test: larql_inference::test_experts::trig_acos_one -->

#### Scenario: Trig covers reciprocal functions and degree conversion
- **WHEN** `atan`, `sec`, `csc`, `cot`, `deg_to_rad`, and `rad_to_deg` are called
- **THEN** each call SHALL return the textbook value as a JSON number
<!-- test: larql_inference::test_experts::trig_atan_one -->
<!-- test: larql_inference::test_experts::trig_sec_zero -->
<!-- test: larql_inference::test_experts::trig_csc_pi_half -->
<!-- test: larql_inference::test_experts::trig_cot_pi_quarter -->
<!-- test: larql_inference::test_experts::trig_deg_to_rad -->
<!-- test: larql_inference::test_experts::trig_rad_to_deg -->

#### Scenario: Trig inverse out-of-range argument is rejected
- **WHEN** `asin` is called with an argument outside `[-1, 1]`
- **THEN** the expert SHALL decline by returning `None`
<!-- test: larql_inference::test_experts::trig_asin_out_of_range -->

#### Scenario: Unit conversions span all advertised groups
- **WHEN** `convert` is called between units of length, mass, temperature, and speed (`km→m`, `miles→km`, `kg→lbs`, `°C→°F`, `inches→cm`)
- **THEN** each call SHALL return the converted value as a JSON number
<!-- test: larql_inference::test_experts::unit_km_to_m -->
<!-- test: larql_inference::test_experts::unit_miles_to_km -->
<!-- test: larql_inference::test_experts::unit_kg_to_lbs -->
<!-- test: larql_inference::test_experts::unit_celsius_to_fahrenheit -->
<!-- test: larql_inference::test_experts::unit_inches_to_cm -->

#### Scenario: Unit conversion across incompatible groups is rejected
- **WHEN** `convert` is invoked between units in different groups (e.g. mass to length)
- **THEN** the expert SHALL decline by returning `None`
<!-- test: larql_inference::test_experts::unit_incompatible_groups -->

#### Scenario: Unit info and listing ops are introspectable
- **WHEN** `info` is called for a known unit and `list` is called with and without a group filter
- **THEN** each call SHALL return the requested unit metadata or unit list as typed JSON
<!-- test: larql_inference::test_experts::unit_info_km -->
<!-- test: larql_inference::test_experts::unit_list_length_group -->
<!-- test: larql_inference::test_experts::unit_list_all -->

### Requirement: Statistics, finance, graph, dijkstra, markov, and conway experts

The `statistics` expert (tier 1, 11 ops) SHALL provide `mean`,
`median`, `mode`, `stddev`, `variance`, `sort`, `min`, `max`, `sum`,
`range`, `count` over arrays of numbers. The `finance` expert (tier
1, 9 ops) SHALL provide future/present value, simple/compound
interest, mortgage payment, NPV, Bayes, Kelly criterion, and ROI.
The `graph` expert (tier 1, 6 ops) SHALL provide centrality, cycle
detection, connected components, topological sort, bipartite check,
and degrees. The `dijkstra` expert (tier 1, 3 ops) SHALL provide
shortest path, reachability, and minimum spanning tree. The `markov`
expert (tier 1, 2 ops) SHALL provide expected value and steady-state
distribution. The `conway` expert (tier 1, 2 ops) SHALL provide
single-step Game of Life and N-generation simulation.

#### Scenario: Statistics aggregate ops compute correct values
- **WHEN** `mean`, `median` (odd and even N), `mode`, `min`, `max`, `sort`, `count`, and `stddev` are called over a known array
- **THEN** each call SHALL return the textbook value as typed JSON
<!-- test: larql_inference::test_experts::statistics_mean -->
<!-- test: larql_inference::test_experts::statistics_median_odd -->
<!-- test: larql_inference::test_experts::statistics_median_even -->
<!-- test: larql_inference::test_experts::statistics_mode -->
<!-- test: larql_inference::test_experts::statistics_min -->
<!-- test: larql_inference::test_experts::statistics_max -->
<!-- test: larql_inference::test_experts::statistics_sort -->
<!-- test: larql_inference::test_experts::statistics_count -->
<!-- test: larql_inference::test_experts::statistics_stddev -->

#### Scenario: Statistics covers variance, sum, and range
- **WHEN** `variance`, `sum`, and `range` are called over a known array
- **THEN** each call SHALL return the textbook value as a JSON number
<!-- test: larql_inference::test_experts::statistics_variance -->
<!-- test: larql_inference::test_experts::statistics_sum -->
<!-- test: larql_inference::test_experts::statistics_range -->

#### Scenario: Finance time-value-of-money ops are deterministic
- **WHEN** `future_value`, `present_value`, `compound_interest`, `simple_interest`, and `mortgage_payment` are called
- **THEN** each call SHALL return the textbook value as a JSON number
<!-- test: larql_inference::test_experts::finance_future_value -->
<!-- test: larql_inference::test_experts::finance_present_value -->
<!-- test: larql_inference::test_experts::finance_compound_interest -->
<!-- test: larql_inference::test_experts::finance_simple_interest -->
<!-- test: larql_inference::test_experts::finance_mortgage_payment -->

#### Scenario: Finance decision-theory ops handle edge cases
- **WHEN** `kelly`, `roi`, `npv`, `bayes` (with non-zero P(B)), and `bayes` (with P(B)=0) are called
- **THEN** the standard cases SHALL return numeric values and the P(B)=0 case SHALL be rejected by the expert
<!-- test: larql_inference::test_experts::finance_kelly -->
<!-- test: larql_inference::test_experts::finance_roi -->
<!-- test: larql_inference::test_experts::finance_npv -->
<!-- test: larql_inference::test_experts::finance_bayes -->
<!-- test: larql_inference::test_experts::finance_bayes_p_b_zero -->

#### Scenario: Graph algorithms cover all six advertised ops
- **WHEN** `most_central`, `cycle_detected`, `connected_components`, `topological_sort` (DAG and cyclic), `bipartite` (yes and no), and `degrees` are called on representative graphs
- **THEN** each call SHALL return the algorithm's output as typed JSON; topological sort over a cyclic graph SHALL be rejected by returning null/none
<!-- test: larql_inference::test_experts::graph_most_central -->
<!-- test: larql_inference::test_experts::graph_cycle_detected -->
<!-- test: larql_inference::test_experts::graph_connected_components -->
<!-- test: larql_inference::test_experts::graph_topological_sort_dag -->
<!-- test: larql_inference::test_experts::graph_topological_sort_cycle_returns_null -->
<!-- test: larql_inference::test_experts::graph_bipartite_yes -->
<!-- test: larql_inference::test_experts::graph_bipartite_no -->
<!-- test: larql_inference::test_experts::graph_degrees -->

#### Scenario: Dijkstra covers shortest path, reachability, and MST
- **WHEN** `shortest_path`, `reachable`, and `minimum_spanning_tree` are called on a representative weighted graph
- **THEN** each call SHALL return the algorithm's output as typed JSON
<!-- test: larql_inference::test_experts::dijkstra_shortest_path -->
<!-- test: larql_inference::test_experts::dijkstra_reachable -->
<!-- test: larql_inference::test_experts::dijkstra_mst -->

#### Scenario: Markov chain ops compute expectation and steady state
- **WHEN** `expected_value` and `steady_state` are called on a representative chain
- **THEN** each call SHALL return the textbook value as typed JSON; steady-state SHALL converge via the power method
<!-- test: larql_inference::test_experts::markov_expected_value -->
<!-- test: larql_inference::test_experts::markov_steady_state -->

#### Scenario: Conway evolves a known pattern
- **WHEN** `step` is called on a blinker and `simulate` is called for one generation on a blinker / for any generation on a still-life block
- **THEN** the blinker SHALL alternate between its two states and the still-life block SHALL remain unchanged
<!-- test: larql_inference::test_experts::conway_step_blinker -->
<!-- test: larql_inference::test_experts::conway_blinker_one_gen -->
<!-- test: larql_inference::test_experts::conway_still_block -->

### Requirement: String, hash, ISBN, Luhn, element, http_status, logic, sql experts

The `string_ops` expert (tier 1, 14 ops) SHALL provide string
manipulation, classification, and search helpers. The `hash` expert
(tier 1, 7 ops) SHALL provide Base64, hex, URL percent encoding /
decoding, and FNV-1a 32-bit. The `isbn` expert (tier 1, 3 ops) SHALL
validate and convert ISBN-10/13. The `luhn` expert (tier 1, 3 ops)
SHALL compute Luhn checksums and detect card networks. The
`element` expert (tier 1, 4 ops) SHALL look up periodic-table data
by atomic number, symbol, or IUPAC name. The `http_status` expert
(tier 1, 1 op) SHALL look up IANA HTTP status codes with category.
The `logic` expert (tier 1, 4 ops) SHALL evaluate, simplify,
classify, and tabulate propositional formulas. The `sql` expert
(tier 1, 1 op) SHALL execute in-memory SQL over CREATE/INSERT/SELECT
plus aggregates. Results MUST be typed JSON, never English prose.

#### Scenario: String ops cover transformation and identity helpers
- **WHEN** `reverse`, `palindrome`, `anagram`, `caesar`, `rot13`, `uppercase`, and `lowercase` are called
- **THEN** each call SHALL return the transformed string or boolean as typed JSON
<!-- test: larql_inference::test_experts::string_ops_reverse -->
<!-- test: larql_inference::test_experts::string_ops_palindrome_true -->
<!-- test: larql_inference::test_experts::string_ops_palindrome_false -->
<!-- test: larql_inference::test_experts::string_ops_anagram_true -->
<!-- test: larql_inference::test_experts::string_ops_anagram_false -->
<!-- test: larql_inference::test_experts::string_ops_caesar -->
<!-- test: larql_inference::test_experts::string_ops_rot13 -->
<!-- test: larql_inference::test_experts::string_ops_uppercase -->
<!-- test: larql_inference::test_experts::string_ops_lowercase -->

#### Scenario: String ops cover length, counting, and matching
- **WHEN** `length` (ASCII and Unicode), `count_char`, `count_substring`, `count_words`, `contains`, `starts_with`, and `ends_with` are called
- **THEN** each call SHALL return the count or boolean as typed JSON; `length` SHALL count Unicode scalar values, not bytes
<!-- test: larql_inference::test_experts::string_ops_length -->
<!-- test: larql_inference::test_experts::string_ops_length_unicode -->
<!-- test: larql_inference::test_experts::string_ops_count_char -->
<!-- test: larql_inference::test_experts::string_ops_count_substring -->
<!-- test: larql_inference::test_experts::string_ops_count_words -->
<!-- test: larql_inference::test_experts::string_ops_contains_true -->
<!-- test: larql_inference::test_experts::string_ops_contains_false -->
<!-- test: larql_inference::test_experts::string_ops_starts_with -->
<!-- test: larql_inference::test_experts::string_ops_ends_with -->

#### Scenario: Hash ops encode and decode the canonical formats
- **WHEN** `base64_encode`, `base64_decode`, `hex_encode`, `hex_decode` (with and without `0x` prefix), `url_encode`, `url_decode`, and `fnv` are called on representative inputs
- **THEN** each call SHALL return the canonical encoded / decoded result as typed JSON
<!-- test: larql_inference::test_experts::hash_base64_encode -->
<!-- test: larql_inference::test_experts::hash_base64_decode -->
<!-- test: larql_inference::test_experts::hash_hex_encode -->
<!-- test: larql_inference::test_experts::hash_hex_decode -->
<!-- test: larql_inference::test_experts::hash_hex_decode_with_prefix -->
<!-- test: larql_inference::test_experts::hash_url_encode -->
<!-- test: larql_inference::test_experts::hash_url_decode -->
<!-- test: larql_inference::test_experts::hash_fnv -->

#### Scenario: ISBN validation and conversion are deterministic
- **WHEN** `validate` is called on valid ISBN-13, valid ISBN-10, and an invalid input, and `to_isbn13` / `to_isbn10` are called
- **THEN** each call SHALL return the boolean / converted value as typed JSON
<!-- test: larql_inference::test_experts::isbn_valid_13 -->
<!-- test: larql_inference::test_experts::isbn_valid_10 -->
<!-- test: larql_inference::test_experts::isbn_invalid -->
<!-- test: larql_inference::test_experts::isbn_isbn10_to_isbn13 -->
<!-- test: larql_inference::test_experts::isbn_isbn13_to_isbn10 -->

#### Scenario: Luhn covers validation, checksum, and card-network detection
- **WHEN** `validate` is called on a valid Visa, valid Amex, and invalid number; `check_digit` is called; and `card_type` is called on an Amex
- **THEN** each call SHALL return the boolean / digit / network as typed JSON
<!-- test: larql_inference::test_experts::luhn_visa_valid -->
<!-- test: larql_inference::test_experts::luhn_amex_valid -->
<!-- test: larql_inference::test_experts::luhn_invalid -->
<!-- test: larql_inference::test_experts::luhn_check_digit -->
<!-- test: larql_inference::test_experts::luhn_card_type_amex -->

#### Scenario: Element lookup covers all four advertised ops
- **WHEN** the periodic table is queried by atomic number, by symbol (case-insensitive), by IUPAC name, by mass, and listed
- **THEN** each call SHALL return the requested element record or list as typed JSON
<!-- test: larql_inference::test_experts::element_atomic_number -->
<!-- test: larql_inference::test_experts::element_symbol -->
<!-- test: larql_inference::test_experts::element_by_symbol -->
<!-- test: larql_inference::test_experts::element_by_symbol_case_insensitive -->
<!-- test: larql_inference::test_experts::element_name_by_number -->
<!-- test: larql_inference::test_experts::element_mass -->
<!-- test: larql_inference::test_experts::element_list -->

#### Scenario: HTTP status codes are looked up by code with categories
- **WHEN** `lookup` is called with `404`, `200`, `500`, `301`, `403`, and an unknown code
- **THEN** each known code SHALL return `{code, reason, category}` as typed JSON; the unknown code SHALL be rejected by the expert
<!-- test: larql_inference::test_experts::http_status_404 -->
<!-- test: larql_inference::test_experts::http_status_200 -->
<!-- test: larql_inference::test_experts::http_status_500 -->
<!-- test: larql_inference::test_experts::http_status_301 -->
<!-- test: larql_inference::test_experts::http_status_403_category -->
<!-- test: larql_inference::test_experts::http_status_unknown -->

#### Scenario: Logic propositional ops classify and simplify
- **WHEN** `eval` (AND), `classify` (tautology, contradiction, contingent), `truth_table`, and `simplify` (double negation) are called
- **THEN** each call SHALL return the boolean / classification / table / simplified expression as typed JSON
<!-- test: larql_inference::test_experts::logic_eval_and -->
<!-- test: larql_inference::test_experts::logic_tautology -->
<!-- test: larql_inference::test_experts::logic_contradiction -->
<!-- test: larql_inference::test_experts::logic_contingent -->
<!-- test: larql_inference::test_experts::logic_truth_table_rows -->
<!-- test: larql_inference::test_experts::logic_simplify_double_negation -->

#### Scenario: SQL expert runs CREATE/INSERT/SELECT and aggregates
- **WHEN** `sql.execute` is called with COUNT, SUM, AVG, and SELECT-WHERE statements over an in-memory schema
- **THEN** each call SHALL return the result rows or aggregate value as typed JSON
<!-- test: larql_inference::test_experts::sql_count -->
<!-- test: larql_inference::test_experts::sql_sum -->
<!-- test: larql_inference::test_experts::sql_avg -->
<!-- test: larql_inference::test_experts::sql_select_with_where -->

### Requirement: Date expert (tier 2, Julian-day arithmetic)

The `date` expert SHALL be the only tier-2 module. It SHALL implement
Gregorian date arithmetic via Julian day number conversions and SHALL
advertise the ops `days_between`, `add_days`, `subtract_days`,
`day_of_week`, `weeks_between`, `is_leap_year`, and `days_in_month`.
Because it is tier 2, when another expert advertises the same op
name (e.g. a hypothetical tier-1 `date_lite`), the tier-1 expert
SHALL win and `date` SHALL be shadowed for that op. Results MUST be
typed JSON values (integers, booleans, ISO date strings).

#### Scenario: Date arithmetic covers forward and backward day deltas
- **WHEN** `days_between` is called within a year and across years, and `add_days` / `subtract_days` are called
- **THEN** each call SHALL return the signed integer delta or the resulting `YYYY-MM-DD` string as typed JSON
<!-- test: larql_inference::test_experts::date_days_between -->
<!-- test: larql_inference::test_experts::date_days_between_year -->
<!-- test: larql_inference::test_experts::date_add_days -->
<!-- test: larql_inference::test_experts::date_subtract_days -->

#### Scenario: Day-of-week and week deltas are computed from JDN
- **WHEN** `day_of_week` and `weeks_between` are called on representative dates
- **THEN** each call SHALL return the weekday name and the integer week delta as typed JSON
<!-- test: larql_inference::test_experts::date_day_of_week_wednesday -->
<!-- test: larql_inference::test_experts::date_weeks_between -->

#### Scenario: Leap-year and month-length lookups respect the Gregorian calendar
- **WHEN** `is_leap_year` is called on `2024` and `2025`, and `days_in_month` is called for February in a leap and non-leap year
- **THEN** the calls SHALL return `true`/`false` and `29`/`28` respectively as typed JSON
<!-- test: larql_inference::test_experts::date_leap_year_true -->
<!-- test: larql_inference::test_experts::date_leap_year_false -->
<!-- test: larql_inference::test_experts::date_days_in_feb_leap -->
<!-- test: larql_inference::test_experts::date_days_in_feb_normal -->

### Requirement: Cross-module language-neutral op contract

Every expert in the registry SHALL satisfy the cross-module contract:
op names MUST be language-neutral identifiers (snake_case, no
English prose), argument keys MUST be JSON object keys, and result
values MUST be typed JSON (numbers, booleans, strings, arrays,
objects) — not formatted natural-language sentences. Experts MUST
return `0` from `larql_call` (i.e. surface as `None` to the host)
rather than panicking when an op is not advertised, when arguments
fail to validate, or when a domain-specific error occurs.

#### Scenario: Op not advertised by an expert surfaces as None
- **WHEN** the registry is asked to call an op that is not in any expert's advertised set
- **THEN** the registry SHALL return `None` and SHALL NOT invoke any guest's `larql_call`
<!-- test: larql_inference::test_experts::registry_unknown_op_returns_none -->

#### Scenario: Domain-specific failures surface as None, not panic
- **WHEN** an expert detects a domain-specific failure such as `arithmetic.div` with a zero divisor or `trig.asin` with an out-of-range argument
- **THEN** the expert SHALL return `None` and the host SHALL remain live for subsequent calls
<!-- test: larql_inference::test_experts::arithmetic_divide_by_zero -->
<!-- test: larql_inference::test_experts::trig_asin_out_of_range -->

#### Scenario: Tier ordering shadows duplicate op names
- **WHEN** a tier-1 expert and a tier-2 expert both advertise the same op name
- **THEN** the tier-1 expert SHALL win for that op and the tier-2 expert SHALL remain reachable for ops it uniquely advertises
<!-- test: larql_inference::test_experts::registry_load_dir_tier_order -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_experts::**::* -->
