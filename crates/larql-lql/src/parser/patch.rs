//! Patch statement parsers: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    /// Parse a statement starting with BEGIN (BEGIN PATCH "file.vlp").
    pub(crate) fn parse_begin(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Begin)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::BeginPatch { path })
    }

    /// Parse SAVE PATCH.
    pub(crate) fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Save)?;
        self.expect_keyword(Keyword::Patch)?;
        self.eat_semicolon();
        Ok(Statement::SavePatch)
    }

    /// Parse APPLY PATCH "file.vlp".
    pub(crate) fn parse_apply(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Apply)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::ApplyPatch { path })
    }

    /// Parse REMOVE PATCH "file.vlp", or REMOVE DIRECTION (a structured weight edit).
    pub(crate) fn parse_remove(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Remove)?;
        if matches!(self.peek(), Token::Ident(ref s) if s.eq_ignore_ascii_case("DIRECTION")) {
            return self.parse_remove_direction();
        }
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemovePatch { path })
    }
}

/// The largest integer magnitude an f64 holds exactly.
const MAX_EXACT_INTEGER_SCALE: u64 = 1 << 53;

impl Parser {
    /// `REMOVE DIRECTION "<operand file>" FROM "<tensor>" COLUMNS (<n>, …) SCALE <f> NUMERICS "<profile>"`;
    /// `REMOVE` is already consumed. `DIRECTION`, `COLUMNS`, `SCALE` and `NUMERICS` are read as words in
    /// this position, not reserved. The scale is the literal's f64 value, exactly as lexed.
    fn parse_remove_direction(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("DIRECTION")?;
        let operand = self.expect_string()?;
        self.expect_keyword(Keyword::From)?;
        let tensor = self.expect_string()?;
        self.expect_ident_eq("COLUMNS")?;
        let columns = self.parse_column_list()?;
        self.expect_ident_eq("SCALE")?;
        let negative = matches!(self.peek(), Token::Dash);
        if negative {
            self.advance();
        }
        let magnitude = match self.peek() {
            Token::NumberLit(v) => {
                self.advance();
                v
            }
            // `SCALE 1` lexes as an integer; it is the f64 it names.
            Token::IntegerLit(v) => {
                // Beyond 2^53 an integer is not exactly an f64; the scale is never rounded silently.
                if v.unsigned_abs() > MAX_EXACT_INTEGER_SCALE {
                    return Err(ParseError(format!(
                        "SCALE {v} is not exactly representable as an f64"
                    )));
                }
                self.advance();
                v as f64
            }
            other => {
                return Err(ParseError(format!(
                    "expected a number for SCALE, got {other:?}"
                )))
            }
        };
        let scale = if negative { -magnitude } else { magnitude };
        self.expect_ident_eq("NUMERICS")?;
        let numerics = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemoveDirection {
            operand,
            tensor,
            columns,
            scale,
            numerics,
        })
    }
}

impl Parser {
    /// `(<n>, <n>, …)`: a parenthesised, non-empty list of column indices.
    fn parse_column_list(&mut self) -> Result<Vec<usize>, ParseError> {
        self.expect_token(&Token::LParen)?;
        let mut columns = vec![self.expect_u32()? as usize];
        while matches!(self.peek(), Token::Comma) {
            self.advance();
            columns.push(self.expect_u32()? as usize);
        }
        self.expect_token(&Token::RParen)?;
        Ok(columns)
    }

    /// `ADD LOW RANK "<operand file>" TO "<tensor>" COLUMNS (<n>, …) NUMERICS "<profile>"`. `ADD`, `LOW`,
    /// `RANK`, `COLUMNS` and `NUMERICS` are read as words in this position, not reserved. The rank and both
    /// operands are in the file, as exact bits.
    pub(crate) fn parse_add_low_rank(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("ADD")?;
        self.expect_ident_eq("LOW")?;
        self.expect_ident_eq("RANK")?;
        let operands = self.expect_string()?;
        self.expect_keyword(Keyword::To)?;
        let tensor = self.expect_string()?;
        self.expect_ident_eq("COLUMNS")?;
        let columns = self.parse_column_list()?;
        self.expect_ident_eq("NUMERICS")?;
        let numerics = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::AddLowRank {
            operands,
            tensor,
            columns,
            numerics,
        })
    }
}
