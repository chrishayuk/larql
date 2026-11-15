//! Lifecycle statement parsers: EXTRACT, COMPILE, DIFF, USE, COMPACT

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::Keyword;

/// The help page a COMPILE parse error points at.
const COMPILE_TOPIC: &str = "COMPILE";

impl Parser {
    pub(crate) fn parse_extract(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Extract)?;
        self.expect_keyword(Keyword::Model)?;
        let model = self.expect_string()?;
        self.expect_keyword(Keyword::Into)?;
        let output = self.expect_string()?;

        let mut components = None;
        let mut layers = None;
        let mut extract_level = ExtractLevel::Browse;
        let mut format = None;

        loop {
            match self.peek() {
                crate::lexer::Token::Keyword(Keyword::Components) => {
                    self.advance();
                    components = Some(self.parse_component_list()?);
                }
                crate::lexer::Token::Keyword(Keyword::Layers) => {
                    self.advance();
                    layers = Some(self.parse_range()?);
                }
                crate::lexer::Token::Keyword(Keyword::Format) => {
                    self.advance();
                    // VINDEX2 / VINDEX3 are bare identifiers, as with
                    // COMPILE INTO VINDEX.
                    format = Some(match self.peek() {
                        crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("vindex2") => {
                            self.advance();
                            ExtractFormat::Vindex2
                        }
                        crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("vindex3") => {
                            self.advance();
                            ExtractFormat::Vindex3
                        }
                        other => {
                            return Err(ParseError(format!(
                                "EXTRACT FORMAT expects VINDEX2 or VINDEX3, found {other:?}"
                            )))
                        }
                    });
                }
                crate::lexer::Token::Keyword(Keyword::With) => {
                    self.advance();
                    // WITH INFERENCE | WITH ALL | WITH WEIGHTS (legacy)
                    if self.check_keyword(Keyword::Inference) {
                        self.advance();
                        extract_level = ExtractLevel::Inference;
                    } else if self.check_keyword(Keyword::All) {
                        self.advance();
                        extract_level = ExtractLevel::All;
                    } else {
                        // WITH WEIGHTS is legacy — maps to Inference
                        self.expect_keyword(Keyword::Weights)?;
                        extract_level = ExtractLevel::Inference;
                    }
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Extract {
            model,
            output,
            components,
            layers,
            extract_level,
            format,
        })
    }

    pub(crate) fn parse_compile(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Compile)?;
        if !self.check_keyword(Keyword::Current)
            && !matches!(self.peek(), crate::lexer::Token::StringLit(_))
        {
            return Err(self.expected_one_of(
                "COMPILE",
                &["CURRENT", "a quoted vindex path"],
                COMPILE_TOPIC,
            ));
        }
        let vindex = self.parse_vindex_ref()?;
        if !self.check_keyword(Keyword::Into) {
            return Err(self.expected_one_of("the COMPILE source", &["INTO"], COMPILE_TOPIC));
        }
        self.advance();

        // COMPILE ... INTO MODEL or COMPILE ... INTO VINDEX
        let target = if self.check_keyword(Keyword::Model) {
            self.advance();
            CompileTarget::Model
        } else {
            // Accept "VINDEX" as an identifier (not a keyword)
            match self.peek() {
                crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("vindex") => {
                    self.advance();
                    CompileTarget::Vindex
                }
                _ => {
                    return Err(self.expected_one_of(
                        "COMPILE ... INTO",
                        &["VINDEX", "MODEL"],
                        COMPILE_TOPIC,
                    ))
                }
            }
        };

        let output = self.expect_string()?;

        let format = if self.check_keyword(Keyword::Format) {
            self.advance();
            Some(self.parse_output_format()?)
        } else {
            None
        };

        // Optional trailing clause for COMPILE INTO VINDEX:
        //   ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}
        // Restricted to COMPILE INTO VINDEX — applying it to COMPILE INTO MODEL
        // is a parse error so users get a clear message instead of silent acceptance.
        let mut on_conflict = None;

        while let crate::lexer::Token::Keyword(Keyword::On) = self.peek() {
            self.advance();
            self.expect_keyword(Keyword::Conflict)?;
            let strat = match self.peek() {
                crate::lexer::Token::Keyword(Keyword::LastWins) => {
                    self.advance();
                    CompileConflict::LastWins
                }
                crate::lexer::Token::Keyword(Keyword::HighestConfidence) => {
                    self.advance();
                    CompileConflict::HighestConfidence
                }
                crate::lexer::Token::Keyword(Keyword::Fail) => {
                    self.advance();
                    CompileConflict::Fail
                }
                t => {
                    return Err(ParseError(format!(
                    "expected LAST_WINS | HIGHEST_CONFIDENCE | FAIL after ON CONFLICT, got {:?}",
                    t
                )))
                }
            };
            if target != CompileTarget::Vindex {
                return Err(ParseError(
                    "ON CONFLICT is only valid for COMPILE INTO VINDEX".into(),
                ));
            }
            on_conflict = Some(strat);
        }

        self.eat_semicolon();
        Ok(Statement::Compile {
            vindex,
            output,
            format,
            target,
            on_conflict,
        })
    }

    pub(crate) fn parse_diff(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Diff)?;
        let a = self.parse_vindex_ref()?;
        let b = self.parse_vindex_ref()?;

        let mut layer = None;
        let mut relation = None;
        let mut limit = None;
        let mut physical = false;

        loop {
            match self.peek() {
                crate::lexer::Token::Keyword(Keyword::Physical) => {
                    self.advance();
                    physical = true;
                }
                crate::lexer::Token::Keyword(Keyword::Layer) => {
                    self.advance();
                    layer = Some(self.expect_u32()?);
                }
                crate::lexer::Token::Keyword(Keyword::Relation)
                | crate::lexer::Token::Keyword(Keyword::Relations) => {
                    self.advance();
                    relation = Some(self.expect_string()?);
                }
                crate::lexer::Token::Keyword(Keyword::Limit) => {
                    self.advance();
                    limit = Some(self.expect_u32()?);
                }
                crate::lexer::Token::Keyword(Keyword::Into) => {
                    self.advance();
                    self.expect_keyword(Keyword::Patch)?;
                    let path = self.expect_string()?;
                    self.eat_semicolon();
                    return Ok(Statement::Diff {
                        a,
                        b,
                        layer,
                        relation,
                        limit,
                        into_patch: Some(path),
                        physical,
                    });
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::Diff {
            a,
            b,
            layer,
            relation,
            limit,
            into_patch: None,
            physical,
        })
    }

    pub(crate) fn parse_use(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Use)?;

        let target = if self.check_keyword(Keyword::Model) {
            self.advance();
            let id = self.expect_string()?;
            let auto_extract = self.check_keyword(Keyword::AutoExtract);
            if auto_extract {
                self.advance();
            }
            UseTarget::Model { id, auto_extract }
        } else if self.check_keyword(Keyword::Remote) {
            self.advance();
            let url = self.expect_string()?;
            UseTarget::Remote(url)
        } else {
            let path = self.expect_string()?;
            UseTarget::Vindex(path)
        };

        self.eat_semicolon();
        Ok(Statement::Use { target })
    }

    /// `COMPACT MINOR;`
    /// `COMPACT MAJOR [FULL] [WITH LAMBDA = <f>];`
    pub(crate) fn parse_compact(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Compact)?;
        match self.peek() {
            crate::lexer::Token::Keyword(Keyword::Into) => {
                self.advance();
                // "VINDEX" is an identifier, as in COMPILE INTO VINDEX.
                match self.peek() {
                    crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("vindex") => {
                        self.advance();
                    }
                    other => {
                        return Err(ParseError(format!(
                            "COMPACT INTO expects VINDEX, found {other:?}"
                        )))
                    }
                }
                let output = self.expect_string()?;
                self.eat_semicolon();
                Ok(Statement::CompactInto { output })
            }
            crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("MINOR") => {
                self.advance();
                self.eat_semicolon();
                Ok(Statement::CompactMinor)
            }
            crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("MAJOR") => {
                self.advance();
                let full = match self.peek() {
                    crate::lexer::Token::Keyword(Keyword::All) => {
                        // COMPACT MAJOR FULL — we reuse ALL since FULL isn't a keyword yet
                        self.advance();
                        true
                    }
                    crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("FULL") => {
                        self.advance();
                        true
                    }
                    _ => false,
                };
                let lambda = if self.check_keyword(Keyword::With) {
                    self.advance();
                    // WITH LAMBDA = <f> or WITH lambda = <f>
                    match self.peek() {
                        crate::lexer::Token::Ident(ref s) if s.eq_ignore_ascii_case("LAMBDA") => {
                            self.advance();
                            if !matches!(self.peek(), crate::lexer::Token::Eq) {
                                return Err(ParseError("expected '=' after LAMBDA".into()));
                            }
                            self.advance();
                            Some(self.expect_f32()?)
                        }
                        _ => {
                            return Err(ParseError(
                                "expected LAMBDA after WITH in COMPACT MAJOR".into(),
                            ));
                        }
                    }
                } else {
                    None
                };
                self.eat_semicolon();
                Ok(Statement::CompactMajor { full, lambda })
            }
            _ => Err(self.expected_after("COMPACT")),
        }
    }
}
