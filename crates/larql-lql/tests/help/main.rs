//! `HELP;` and `HELP <topic>;`: the in-language help.
//!
//! The claim under test is that the help cannot fall behind the grammar:
//! every statement kind has a page, every example on a page is that
//! statement according to the real parser, every page is listed by `HELP;`
//! and found by `HELP <topic>;`, and what a page says a statement needs is
//! what the executor refuses without.

mod banner;
mod catalogue;
mod live_effect;
mod lookup;
mod parse_errors;
mod statement;
mod support;
