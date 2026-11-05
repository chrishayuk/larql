//! Mutation statement parsers: INSERT, DELETE, UPDATE, MERGE

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    pub(crate) fn parse_insert(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;
        self.expect_keyword(Keyword::Edges)?;

        // (entity, relation, target)
        self.expect_token(&Token::LParen)?;
        self.expect_ident_eq("entity")?;
        self.expect_token(&Token::Comma)?;
        self.expect_ident_eq("relation")?;
        self.expect_token(&Token::Comma)?;
        self.expect_ident_eq("target")?;
        self.expect_token(&Token::RParen)?;

        // VALUES (e, r, t)
        self.expect_keyword(Keyword::Values)?;
        self.expect_token(&Token::LParen)?;
        let entity = self.expect_string()?;
        self.expect_token(&Token::Comma)?;
        let relation = self.expect_string()?;
        self.expect_token(&Token::Comma)?;
        let target = self.expect_string()?;
        self.expect_token(&Token::RParen)?;

        let mut layer = None;
        let mut confidence = None;
        let mut alpha = None;
        let mut mode = InsertMode::default(); // Knn

        loop {
            match self.peek() {
                Token::Keyword(Keyword::At) => {
                    self.advance();
                    self.expect_keyword(Keyword::Layer)?;
                    layer = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Confidence) => {
                    self.advance();
                    confidence = Some(self.expect_f32()?);
                }
                Token::Keyword(Keyword::Alpha) => {
                    self.advance();
                    alpha = Some(self.expect_f32()?);
                }
                Token::Keyword(Keyword::Mode) => {
                    self.advance();
                    // Optional `=` for readability: `MODE = knn`
                    if matches!(self.peek(), Token::Eq) {
                        self.advance();
                    }
                    match self.peek() {
                        Token::Keyword(Keyword::Knn) => {
                            self.advance();
                            mode = InsertMode::Knn;
                        }
                        Token::Keyword(Keyword::Compose) => {
                            self.advance();
                            mode = InsertMode::Compose;
                        }
                        other => {
                            return Err(ParseError(format!(
                                "expected KNN or COMPOSE after MODE, got {other:?}"
                            )))
                        }
                    }
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Insert {
            entity,
            relation,
            target,
            layer,
            confidence,
            alpha,
            mode,
        })
    }

    pub(crate) fn parse_delete(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        self.expect_keyword(Keyword::Edges)?;
        self.expect_keyword(Keyword::Where)?;
        let conditions = self.parse_conditions()?;
        self.eat_semicolon();
        Ok(Statement::Delete { conditions })
    }

    pub(crate) fn parse_update(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Update)?;
        if matches!(self.peek(), Token::Keyword(Keyword::Weights)) {
            return self.parse_update_weights();
        }
        self.expect_keyword(Keyword::Edges)?;
        self.expect_keyword(Keyword::Set)?;

        let set = self.parse_assignments()?;

        self.expect_keyword(Keyword::Where)?;
        let conditions = self.parse_conditions()?;
        self.eat_semicolon();
        Ok(Statement::Update { set, conditions })
    }

    pub(crate) fn parse_rebalance(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Rebalance)?;
        let mut max_iters = None;
        let mut floor = None;
        let mut ceiling = None;
        loop {
            match self.peek() {
                Token::Keyword(Keyword::Until) => {
                    self.advance();
                    self.expect_keyword(Keyword::Converged)?;
                }
                Token::Keyword(Keyword::Max) => {
                    self.advance();
                    max_iters = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Floor) => {
                    self.advance();
                    if matches!(self.peek(), Token::Eq) {
                        self.advance();
                    }
                    floor = Some(self.expect_f32()?);
                }
                Token::Keyword(Keyword::Ceiling) => {
                    self.advance();
                    if matches!(self.peek(), Token::Eq) {
                        self.advance();
                    }
                    ceiling = Some(self.expect_f32()?);
                }
                _ => break,
            }
        }
        self.eat_semicolon();
        Ok(Statement::Rebalance {
            max_iters,
            floor,
            ceiling,
        })
    }

    pub(crate) fn parse_merge(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Merge)?;
        let source = self.expect_string()?;

        let mut target = None;
        let mut conflict = None;

        if self.check_keyword(Keyword::Into) {
            self.advance();
            target = Some(self.expect_string()?);
        }

        if self.check_keyword(Keyword::On) {
            self.advance();
            self.expect_keyword(Keyword::Conflict)?;
            conflict = Some(self.parse_conflict_strategy()?);
        }

        self.eat_semicolon();
        Ok(Statement::Merge {
            source,
            target,
            conflict,
        })
    }
}

impl Parser {
    /// `UPDATE WEIGHTS SET DELTA = <f> WHERE TENSOR = "<name>" AND ROW = <n>
    /// AND COL = <n> | COL IN (<n>, …)`; the `UPDATE` is already consumed.
    ///
    /// `DELTA`, `TENSOR`, `ROW` and `COL` are read as words in this position,
    /// not reserved, so they stay usable as names everywhere else.
    fn parse_update_weights(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Weights)?;
        self.expect_keyword(Keyword::Set)?;
        self.expect_ident_eq("DELTA")?;
        self.expect_token(&Token::Eq)?;
        let delta = self.expect_signed_f32()?;
        self.expect_keyword(Keyword::Where)?;
        self.expect_ident_eq("TENSOR")?;
        self.expect_token(&Token::Eq)?;
        let tensor = self.expect_string()?;
        self.expect_keyword(Keyword::And)?;
        self.expect_ident_eq("ROW")?;
        self.expect_token(&Token::Eq)?;
        let row = self.expect_u32()? as usize;
        self.expect_keyword(Keyword::And)?;
        self.expect_ident_eq("COL")?;
        let cols = if matches!(self.peek(), Token::Keyword(Keyword::In)) {
            self.advance();
            self.expect_token(&Token::LParen)?;
            let mut cols = vec![self.expect_u32()? as usize];
            while matches!(self.peek(), Token::Comma) {
                self.advance();
                cols.push(self.expect_u32()? as usize);
            }
            self.expect_token(&Token::RParen)?;
            cols
        } else {
            self.expect_token(&Token::Eq)?;
            vec![self.expect_u32()? as usize]
        };
        self.eat_semicolon();
        Ok(Statement::UpdateWeights {
            tensor,
            row,
            cols,
            delta,
        })
    }
}
