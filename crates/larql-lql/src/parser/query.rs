//! Query statement parsers: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    pub(crate) fn parse_walk(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Walk)?;
        let prompt = self.expect_string()?;

        let mut top = None;
        let mut layers = None;
        let mut mode = None;
        let mut compare = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Top) => {
                    self.advance();
                    top = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    if self.check_keyword(Keyword::All) {
                        self.advance();
                        // ALL means no range filter
                    } else {
                        layers = Some(self.parse_range()?);
                    }
                }
                Token::Keyword(Keyword::Mode) => {
                    self.advance();
                    mode = Some(self.parse_walk_mode()?);
                }
                Token::Keyword(Keyword::Compare) => {
                    self.advance();
                    compare = true;
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Walk {
            prompt,
            top,
            layers,
            mode,
            compare,
        })
    }

    pub(crate) fn parse_infer(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Infer)?;
        let prompt = self.expect_string()?;

        let mut top = None;
        let mut compare = false;
        let mut route = None;
        let mut generate = None;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Top) => {
                    self.advance();
                    top = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Generate) => {
                    self.advance();
                    generate = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Compare) => {
                    self.advance();
                    compare = true;
                }
                Token::Keyword(Keyword::Route) => {
                    self.advance();
                    // ROUTE VERIFY [FALLBACK] [TOPK n] [EXIT]
                    self.expect_keyword(Keyword::Verify)?;
                    let mut r = crate::ast::InferRoute::default();
                    loop {
                        match self.peek() {
                            Token::Keyword(Keyword::Fallback) => {
                                self.advance();
                                r.fallback = true;
                            }
                            Token::Keyword(Keyword::Topk) => {
                                self.advance();
                                r.topk = Some(self.expect_u32()?);
                            }
                            Token::Keyword(Keyword::Exit) => {
                                self.advance();
                                r.exit = true;
                            }
                            _ => break,
                        }
                    }
                    route = Some(r);
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Infer {
            prompt,
            top,
            compare,
            route,
            generate,
        })
    }

    pub(crate) fn parse_select(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Select)?;

        let fields = self.parse_field_list()?;

        self.expect_keyword(Keyword::From)?;
        let source = match self.peek() {
            Token::Keyword(Keyword::Edges) => {
                self.advance();
                SelectSource::Edges
            }
            Token::Keyword(Keyword::Features) => {
                self.advance();
                SelectSource::Features
            }
            Token::Keyword(Keyword::Entities) => {
                self.advance();
                SelectSource::Entities
            }
            _ => {
                // Default to EDGES for backwards compatibility.
                self.expect_keyword(Keyword::Edges)?;
                SelectSource::Edges
            }
        };

        let mut nearest = None;
        if self.check_keyword(Keyword::Nearest) {
            self.advance();
            self.expect_keyword(Keyword::To)?;
            let entity = self.expect_string()?;
            self.expect_keyword(Keyword::At)?;
            self.expect_keyword(Keyword::Layer)?;
            let layer = self.expect_u32()?;
            nearest = Some(NearestClause { entity, layer });
        }

        let conditions = if self.check_keyword(Keyword::Where) {
            self.advance();
            self.parse_conditions()?
        } else {
            vec![]
        };

        let order = if self.check_keyword(Keyword::Order) {
            self.advance();
            self.expect_keyword(Keyword::By)?;
            Some(self.parse_order_by()?)
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
        Ok(Statement::Select {
            source,
            fields,
            conditions,
            nearest,
            order,
            limit,
        })
    }

    pub(crate) fn parse_describe(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Describe)?;
        let entity = self.expect_string()?;

        let mut band = None;
        let mut layer = None;
        let mut relations_only = false;
        let mut mode = DescribeMode::default();

        loop {
            match self.peek() {
                Token::Keyword(Keyword::At) => {
                    self.advance();
                    self.expect_keyword(Keyword::Layer)?;
                    layer = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Relations) => {
                    self.advance();
                    self.expect_keyword(Keyword::Only)?;
                    relations_only = true;
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
                _ => {
                    if let Some(b) = self.try_parse_layer_band() {
                        band = Some(b);
                    } else {
                        break;
                    }
                }
            }
        }

        self.eat_semicolon();
        Ok(Statement::Describe {
            entity,
            band,
            layer,
            relations_only,
            mode,
        })
    }

    pub(crate) fn parse_explain(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Explain)?;

        // Determine mode: EXPLAIN WALK or EXPLAIN INFER
        let mode = if self.check_keyword(Keyword::Infer) {
            self.advance();
            ExplainMode::Infer
        } else if self.check_keyword(Keyword::Walk) {
            self.advance();
            ExplainMode::Walk
        } else {
            return Err(self.expected_one_of("EXPLAIN", &["WALK", "INFER"], "EXPLAIN"));
        };

        let prompt = self.expect_string()?;

        let mut layers = None;
        let mut band = None;
        let mut verbose = false;
        let mut top = None;
        let mut relations_only = false;
        let mut with_attention = false;

        loop {
            match self.peek() {
                Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                Token::Keyword(Keyword::Verbose) => {
                    self.advance();
                    verbose = true;
                }
                Token::Keyword(Keyword::Top) => {
                    self.advance();
                    top = Some(self.expect_u32()?);
                }
                Token::Keyword(Keyword::Relations) => {
                    self.advance();
                    self.expect_keyword(Keyword::Only)?;
                    relations_only = true;
                }
                Token::Keyword(Keyword::With) => {
                    self.advance();
                    self.expect_keyword(Keyword::Attention)?;
                    with_attention = true;
                }
                _ => {
                    if let Some(b) = self.try_parse_layer_band() {
                        band = Some(b);
                    } else {
                        break;
                    }
                }
            }
        }

        self.eat_semicolon();
        Ok(Statement::Explain {
            prompt,
            mode,
            layers,
            band,
            verbose,
            top,
            relations_only,
            with_attention,
        })
    }
}
