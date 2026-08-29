//! Introspection statement parsers: SHOW (RELATIONS, LAYERS, FEATURES, MODELS), STATS.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    pub(crate) fn parse_show(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Show)?;

        match self.peek() {
            Token::Keyword(Keyword::Relations) => {
                self.advance();
                let mut layer = None;
                let mut with_examples = false;
                let mut mode = DescribeMode::default();

                loop {
                    match self.peek() {
                        Token::Keyword(Keyword::At) => {
                            self.advance();
                            self.expect_keyword(Keyword::Layer)?;
                            layer = Some(self.expect_u32()?);
                        }
                        Token::Keyword(Keyword::With) => {
                            self.advance();
                            self.expect_keyword(Keyword::Examples)?;
                            with_examples = true;
                        }
                        Token::Keyword(Keyword::Verbose) => {
                            self.advance();
                            mode = DescribeMode::Verbose;
                        }
                        Token::Keyword(Keyword::Brief) => {
                            self.advance();
                            mode = DescribeMode::Brief;
                        }
                        Token::Keyword(Keyword::Raw) => {
                            self.advance();
                            mode = DescribeMode::Raw;
                        }
                        _ => break,
                    }
                }
                self.eat_semicolon();
                Ok(Statement::ShowRelations {
                    layer,
                    with_examples,
                    mode,
                })
            }
            Token::Keyword(Keyword::Layers) => {
                self.advance();
                let range = if self.check_keyword(Keyword::Range) {
                    self.advance();
                    Some(self.parse_range()?)
                } else if matches!(self.peek(), Token::IntegerLit(_)) {
                    Some(self.parse_range()?)
                } else {
                    None
                };
                self.eat_semicolon();
                Ok(Statement::ShowLayers { range })
            }
            Token::Keyword(Keyword::Features) => {
                self.advance();
                let layer = self.expect_u32()?;

                let conditions = if self.check_keyword(Keyword::Where) {
                    self.advance();
                    self.parse_conditions()?
                } else {
                    vec![]
                };

                let limit = if self.check_keyword(Keyword::Limit) {
                    self.advance();
                    Some(self.expect_u32()?)
                } else {
                    None
                };

                self.eat_semicolon();
                Ok(Statement::ShowFeatures {
                    layer,
                    conditions,
                    limit,
                })
            }
            Token::Keyword(Keyword::Entities) => {
                self.advance();
                let layer = if self.check_keyword(Keyword::At) {
                    self.advance();
                    self.expect_keyword(Keyword::Layer)?;
                    Some(self.expect_u32()?)
                } else if matches!(self.peek(), Token::IntegerLit(_)) {
                    Some(self.expect_u32()?)
                } else {
                    None
                };
                let limit = if self.check_keyword(Keyword::Limit) {
                    self.advance();
                    Some(self.expect_u32()?)
                } else {
                    None
                };
                self.eat_semicolon();
                Ok(Statement::ShowEntities { layer, limit })
            }
            Token::Keyword(Keyword::Models) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowModels)
            }
            Token::Keyword(Keyword::Patches) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowPatches)
            }
            Token::Keyword(Keyword::Compact) => {
                self.advance();
                self.expect_keyword(Keyword::Status)?;
                self.eat_semicolon();
                Ok(Statement::ShowCompactStatus)
            }
            Token::Keyword(Keyword::Components) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowComponents)
            }
            Token::Keyword(Keyword::Representations) => {
                self.advance();
                let object = self.maybe_object_filter()?;
                self.eat_semicolon();
                Ok(Statement::ShowRepresentations { object })
            }
            Token::Keyword(Keyword::Provenance) => {
                self.advance();
                let object = self.maybe_object_filter()?;
                self.eat_semicolon();
                Ok(Statement::ShowProvenance { object })
            }
            Token::Keyword(Keyword::Authority) => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::ShowAuthority)
            }
            _ => Err(ParseError(format!(
                "expected RELATIONS, LAYERS, FEATURES, ENTITIES, MODELS, PATCHES, COMPACT, \
                 COMPONENTS, REPRESENTATIONS, PROVENANCE, or AUTHORITY after SHOW, got {:?}",
                self.peek()
            ))),
        }
    }

    /// An optional object filter after SHOW REPRESENTATIONS / SHOW
    /// PROVENANCE — a quoted string, since logical object ids carry
    /// dots and digits the identifier lexer would split.
    fn maybe_object_filter(&mut self) -> Result<Option<String>, ParseError> {
        if let Token::StringLit(_) = self.peek() {
            Ok(Some(self.expect_string()?))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn parse_stats(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Stats)?;
        let vindex = if let Token::StringLit(_) = self.peek() {
            Some(self.expect_string()?)
        } else {
            None
        };
        self.eat_semicolon();
        Ok(Statement::Stats { vindex })
    }
}
