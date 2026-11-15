//! `EXPORT WEIGHT "<tensor>" [ROWS (<i>, …)] [COLUMNS (<j>, …)] TO "<file>.npy" [AS BITS]`.
//!
//! `EXPORT`, `WEIGHT`, `ROWS`, `COLUMNS`, `AS` and `BITS` are read as words in this position, not reserved, so no
//! existing identifier changes meaning.

use super::{ParseError, Parser};
use crate::ast::Statement;
use crate::lexer::{Keyword, Token};

/// Whether the next token is the word `word`.
fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

impl Parser {
    pub(crate) fn parse_export_weight(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("EXPORT")?;
        self.expect_words_after("EXPORT", &["WEIGHT"])?;
        let tensor = self.expect_string()?;
        let rows = if is_word(&self.peek(), "ROWS") {
            self.advance();
            Some(self.parse_column_list()?)
        } else {
            None
        };
        let columns = if is_word(&self.peek(), "COLUMNS") {
            self.advance();
            Some(self.parse_column_list()?)
        } else {
            None
        };
        if !self.check_keyword(Keyword::To) {
            return Err(self.expected_one_of(
                "EXPORT WEIGHT \"<tensor>\"",
                &["ROWS", "COLUMNS", "TO"],
                "EXPORT WEIGHT",
            ));
        }
        self.advance();
        let output = self.expect_string()?;
        let bits = if is_word(&self.peek(), "AS") {
            self.advance();
            self.expect_ident_eq("BITS")?;
            true
        } else {
            false
        };
        self.eat_semicolon();
        Ok(Statement::ExportWeight {
            tensor,
            rows,
            columns,
            output,
            bits,
        })
    }
}
