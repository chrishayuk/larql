//! Parse errors that say what would have been accepted.
//!
//! The valid statement keywords and the words that may follow a keyword
//! such as `REMOVE` are read from the help catalogue
//! ([`crate::help::grammar`]), which is checked against this parser by
//! `tests/help/catalogue.rs`. The messages therefore list every statement
//! `HELP;` documents, and name pages `HELP <topic>;` finds.

use super::{ParseError, Parser};
use crate::help::{grammar, lookup};
use crate::lexer::{Keyword, Token};

/// `A`, `A or B`, `A, B or C`.
fn one_of(options: &[&str]) -> String {
    match options.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => options.concat(),
    }
}

impl Parser {
    /// The token the parser stopped at, as the user wrote it.
    pub(crate) fn found(&self) -> String {
        match self.peek() {
            Token::Ident(word) => word,
            Token::Keyword(keyword) => keyword.as_field_name().to_uppercase(),
            Token::StringLit(text) => format!("the string \"{text}\""),
            Token::IntegerLit(number) => format!("the number {number}"),
            Token::NumberLit(number) => format!("the number {number}"),
            Token::Semicolon | Token::Eof => "the end of the statement".to_string(),
            other => format!("{other:?}"),
        }
    }

    /// The error for text that begins no statement: the keywords that do,
    /// the nearest one when the word looks like a typo, and where the
    /// syntax is.
    pub(crate) fn unknown_statement(&self) -> ParseError {
        let found = self.found();
        let keywords = grammar::leading_keywords();
        let guess = lookup::closest(&found, &keywords)
            .map(|keyword| format!(" Did you mean {keyword}?"))
            .unwrap_or_default();
        ParseError(format!(
            "expected statement keyword, got {found}.{guess} Statements start with: {}. \
             Run HELP; for the syntax of each.",
            keywords.join(", ")
        ))
    }

    /// The error for a wrong word after `leading`, a keyword that begins
    /// several statements (`REMOVE`) or a several-word name (`ADD LOW
    /// RANK`). It names every continuation the catalogue documents.
    pub(crate) fn expected_after(&self, leading: &str) -> ParseError {
        let options = grammar::continuations(leading);
        let found = self.found();
        let first_words: Vec<&str> = options
            .iter()
            .filter_map(|option| option.split(' ').next())
            .collect();
        let hint = match (options.as_slice(), lookup::closest(&found, &first_words)) {
            ([only], _) => format!("Run HELP {leading} {only}; for the syntax."),
            (_, Some(first)) => {
                let meant = options
                    .iter()
                    .find(|option| option.split(' ').next() == Some(first))
                    .copied()
                    .unwrap_or(first);
                format!(
                    "Did you mean {leading} {meant}? Run HELP {leading} {meant}; for the syntax."
                )
            }
            _ => "Run HELP; for the syntax of each.".to_string(),
        };
        ParseError(format!(
            "after {leading} expected {}, got {found}. {hint}",
            one_of(&options)
        ))
    }

    /// The error for a wrong word at a point where the parser itself
    /// branches on a fixed set of words. `topic` is the help page that
    /// documents the statement.
    pub(crate) fn expected_one_of(&self, after: &str, options: &[&str], topic: &str) -> ParseError {
        ParseError(format!(
            "after {after} expected {}, got {}. Run HELP {topic}; for the syntax.",
            one_of(options),
            self.found()
        ))
    }

    /// Consume `keyword` as what follows `leading`, or say what may.
    pub(crate) fn expect_keyword_after(
        &mut self,
        leading: &str,
        keyword: Keyword,
    ) -> Result<(), ParseError> {
        if self.check_keyword(keyword) {
            self.advance();
            Ok(())
        } else {
            Err(self.expected_after(leading))
        }
    }

    /// Consume the contextual `words` that follow `leading`, or say what
    /// may follow it.
    pub(crate) fn expect_words_after(
        &mut self,
        leading: &str,
        words: &[&str],
    ) -> Result<(), ParseError> {
        for word in words {
            if self.expect_ident_eq(word).is_err() {
                return Err(self.expected_after(leading));
            }
        }
        Ok(())
    }
}
