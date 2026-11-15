//! `CAPTURE <site> [AT LAYER <n> | AT LAYERS (<n>, …)] FOR "<prompt>" [CONTINUE "<text>" | CONTINUE TOKENS
//! (<id>, …) | GENERATE <n>] [POSITIONS PROMPT_END | REPLY | ALL | LAST | (<p>, …)] TO "<file>.npy"`.
//!
//! `CAPTURE`, `CONTINUE`, `TOKENS`, `PROMPT_END`, `REPLY`, `LAST` and the site names are read as words in this
//! position, not reserved, so no existing identifier changes meaning. A site with a dot (`ffn.input`) is the
//! words on either side of the dot.

use super::{ParseError, Parser};
use crate::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite, Statement};
use crate::lexer::{Keyword, Token};

/// The help page every message points at.
const TOPIC: &str = "CAPTURE";

/// Whether `token` is the word `word`.
fn is_word(token: &Token, word: &str) -> bool {
    matches!(token, Token::Ident(s) if s.eq_ignore_ascii_case(word))
}

impl Parser {
    pub(crate) fn parse_capture(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq(TOPIC)?;
        let site = self.parse_capture_site()?;
        let layers = self.parse_capture_layers()?;
        if !self.check_keyword(Keyword::For) {
            let options: &[&str] = if layers.is_some() {
                &["FOR"]
            } else {
                &["AT", "FOR"]
            };
            return Err(self.expected_one_of(&format!("CAPTURE {}", site.name()), options, TOPIC));
        }
        self.advance();
        let prompt = self.expect_string()?;
        let continuation = self.parse_capture_continuation()?;
        let positions = self.parse_capture_positions()?;
        if !self.check_keyword(Keyword::To) {
            let mut options = Vec::new();
            if continuation.is_none() {
                options.extend(["CONTINUE", "GENERATE"]);
            }
            options.extend(["POSITIONS", "TO"]);
            return Err(self.expected_one_of("FOR \"<prompt>\"", &options, TOPIC));
        }
        self.advance();
        let output = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::Capture {
            site,
            layers,
            prompt,
            continuation,
            positions: positions.unwrap_or(CapturePositions::Last),
            output,
        })
    }

    /// A site name: one word, or two joined by a dot.
    fn parse_capture_site(&mut self) -> Result<CaptureSite, ParseError> {
        let Token::Ident(first) = self.peek() else {
            return Err(self.unknown_site(&self.found()));
        };
        self.advance();
        let mut name = first;
        if matches!(self.peek(), Token::Dot) {
            self.advance();
            match self.peek() {
                Token::Ident(second) => {
                    self.advance();
                    name = format!("{name}.{second}");
                }
                _ => return Err(self.unknown_site(&format!("{name}."))),
            }
        }
        CaptureSite::from_name(&name).ok_or_else(|| self.unknown_site(&name))
    }

    fn unknown_site(&self, found: &str) -> ParseError {
        let sites: Vec<&str> = CaptureSite::ALL.iter().map(|s| s.name()).collect();
        ParseError(format!(
            "CAPTURE expected a site ({}), got {found}. Run HELP {TOPIC}; for the syntax.",
            sites.join(", ")
        ))
    }

    fn parse_capture_layers(&mut self) -> Result<Option<CaptureLayers>, ParseError> {
        if !self.check_keyword(Keyword::At) {
            return Ok(None);
        }
        self.advance();
        if self.check_keyword(Keyword::Layer) {
            self.advance();
            return Ok(Some(CaptureLayers::One(self.expect_u32()? as usize)));
        }
        if self.check_keyword(Keyword::Layers) {
            self.advance();
            return Ok(Some(CaptureLayers::Many(self.parse_column_list()?)));
        }
        Err(self.expected_one_of("CAPTURE <site> AT", &["LAYER", "LAYERS"], TOPIC))
    }

    fn parse_capture_continuation(&mut self) -> Result<Option<CaptureContinuation>, ParseError> {
        if self.check_keyword(Keyword::Generate) {
            self.advance();
            return Ok(Some(CaptureContinuation::Generate(self.expect_u32()?)));
        }
        if !is_word(&self.peek(), "CONTINUE") {
            return Ok(None);
        }
        self.advance();
        if let Token::StringLit(text) = self.peek() {
            self.advance();
            return Ok(Some(CaptureContinuation::Text(text)));
        }
        if is_word(&self.peek(), "TOKENS") {
            self.advance();
            let ids = self
                .parse_column_list()?
                .into_iter()
                .map(|id| id as u32)
                .collect();
            return Ok(Some(CaptureContinuation::Tokens(ids)));
        }
        Err(self.expected_one_of("CONTINUE", &["\"<text>\"", "TOKENS"], TOPIC))
    }

    fn parse_capture_positions(&mut self) -> Result<Option<CapturePositions>, ParseError> {
        if !self.check_keyword(Keyword::Positions) {
            return Ok(None);
        }
        self.advance();
        let token = self.peek();
        let positions = if is_word(&token, "PROMPT_END") {
            CapturePositions::PromptEnd
        } else if is_word(&token, "REPLY") {
            CapturePositions::Reply
        } else if is_word(&token, "LAST") {
            CapturePositions::Last
        } else if self.check_keyword(Keyword::All) {
            CapturePositions::All
        } else if matches!(token, Token::LParen) {
            return Ok(Some(CapturePositions::List(self.parse_column_list()?)));
        } else {
            return Err(self.expected_one_of(
                "POSITIONS",
                &["PROMPT_END", "REPLY", "ALL", "LAST", "(<p>, ...)"],
                TOPIC,
            ));
        };
        self.advance();
        Ok(Some(positions))
    }
}
