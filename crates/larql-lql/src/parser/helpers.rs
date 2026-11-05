//! Shared parsing helpers: token utilities, value/field/condition parsers.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    // ── Composite parsers ──

    pub(crate) fn parse_vindex_ref(&mut self) -> Result<VindexRef, ParseError> {
        if self.check_keyword(Keyword::Current) {
            self.advance();
            Ok(VindexRef::Current)
        } else {
            Ok(VindexRef::Path(self.expect_string()?))
        }
    }

    pub(crate) fn parse_range(&mut self) -> Result<Range, ParseError> {
        let start = self.expect_u32()?;
        self.expect_token(&Token::Dash)?;
        let end = self.expect_u32()?;
        if start > end {
            return Err(ParseError(format!(
                "invalid range: start ({start}) > end ({end})"
            )));
        }
        Ok(Range { start, end })
    }

    /// Try to parse a layer band keyword (ALL LAYERS, SYNTAX, KNOWLEDGE, OUTPUT).
    /// Returns None if the current token is not a layer band keyword.
    pub(crate) fn try_parse_layer_band(&mut self) -> Option<LayerBand> {
        match self.peek() {
            Token::Keyword(Keyword::All) => {
                let saved = self.pos;
                self.advance();
                if self.check_keyword(Keyword::Layers) {
                    self.advance();
                    Some(LayerBand::All)
                } else {
                    self.pos = saved;
                    None
                }
            }
            Token::Keyword(Keyword::Syntax) => {
                self.advance();
                Some(LayerBand::Syntax)
            }
            Token::Keyword(Keyword::Knowledge) => {
                self.advance();
                Some(LayerBand::Knowledge)
            }
            Token::Keyword(Keyword::Output) => {
                self.advance();
                Some(LayerBand::Output)
            }
            _ => None,
        }
    }

    pub(crate) fn parse_walk_mode(&mut self) -> Result<WalkMode, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Hybrid) => {
                self.advance();
                Ok(WalkMode::Hybrid)
            }
            Token::Keyword(Keyword::Pure) => {
                self.advance();
                Ok(WalkMode::Pure)
            }
            Token::Keyword(Keyword::Dense) => {
                self.advance();
                Ok(WalkMode::Dense)
            }
            _ => Err(ParseError(format!(
                "expected HYBRID, PURE, or DENSE, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_output_format(&mut self) -> Result<OutputFormat, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Safetensors) => {
                self.advance();
                Ok(OutputFormat::Safetensors)
            }
            Token::Keyword(Keyword::Gguf) => {
                self.advance();
                Ok(OutputFormat::Gguf)
            }
            _ => Err(ParseError(format!(
                "expected SAFETENSORS or GGUF, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_conflict_strategy(&mut self) -> Result<ConflictStrategy, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::KeepSource) => {
                self.advance();
                Ok(ConflictStrategy::KeepSource)
            }
            Token::Keyword(Keyword::KeepTarget) => {
                self.advance();
                Ok(ConflictStrategy::KeepTarget)
            }
            Token::Keyword(Keyword::HighestConfidence) => {
                self.advance();
                Ok(ConflictStrategy::HighestConfidence)
            }
            _ => Err(ParseError(format!(
                "expected KEEP_SOURCE, KEEP_TARGET, or HIGHEST_CONFIDENCE, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_component_list(&mut self) -> Result<Vec<Component>, ParseError> {
        let mut components = vec![self.parse_component()?];
        while self.check_comma() {
            self.advance();
            components.push(self.parse_component()?);
        }
        Ok(components)
    }

    fn parse_component(&mut self) -> Result<Component, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::FfnGate) => {
                self.advance();
                Ok(Component::FfnGate)
            }
            Token::Keyword(Keyword::FfnDown) => {
                self.advance();
                Ok(Component::FfnDown)
            }
            Token::Keyword(Keyword::FfnUp) => {
                self.advance();
                Ok(Component::FfnUp)
            }
            Token::Keyword(Keyword::Embeddings) => {
                self.advance();
                Ok(Component::Embeddings)
            }
            Token::Keyword(Keyword::AttnOv) => {
                self.advance();
                Ok(Component::AttnOv)
            }
            Token::Keyword(Keyword::AttnQk) => {
                self.advance();
                Ok(Component::AttnQk)
            }
            // Also accept unquoted identifiers for convenience
            Token::Ident(ref s) => {
                let c = match s.to_lowercase().as_str() {
                    "ffn_gate" => Component::FfnGate,
                    "ffn_down" => Component::FfnDown,
                    "ffn_up" => Component::FfnUp,
                    "embeddings" => Component::Embeddings,
                    "attn_ov" => Component::AttnOv,
                    "attn_qk" => Component::AttnQk,
                    _ => return Err(ParseError(format!("unknown component: {s}"))),
                };
                self.advance();
                Ok(c)
            }
            _ => Err(ParseError(format!(
                "expected component name, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_field_list(&mut self) -> Result<Vec<Field>, ParseError> {
        if matches!(self.peek(), Token::Star) {
            self.advance();
            return Ok(vec![Field::Star]);
        }

        let mut fields = vec![self.parse_field()?];
        while self.check_comma() {
            self.advance();
            fields.push(self.parse_field()?);
        }
        Ok(fields)
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        match self.peek() {
            Token::Star => {
                self.advance();
                Ok(Field::Star)
            }
            Token::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(Field::Named(name))
            }
            // Some field names collide with keywords (e.g. "layer", "confidence")
            Token::Keyword(kw) => {
                let name = kw.as_field_name().to_string();
                self.advance();
                Ok(Field::Named(name))
            }
            _ => Err(ParseError(format!(
                "expected field name, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_conditions(&mut self) -> Result<Vec<Condition>, ParseError> {
        let mut conditions = vec![self.parse_condition()?];
        while self.check_keyword(Keyword::And) {
            self.advance();
            conditions.push(self.parse_condition()?);
        }
        Ok(conditions)
    }

    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let field = self.expect_field_name()?;
        let op = self.parse_compare_op()?;
        let value = self.parse_value()?;
        Ok(Condition { field, op, value })
    }

    fn parse_compare_op(&mut self) -> Result<CompareOp, ParseError> {
        match self.peek() {
            Token::Eq => {
                self.advance();
                Ok(CompareOp::Eq)
            }
            Token::Neq => {
                self.advance();
                Ok(CompareOp::Neq)
            }
            Token::Gt => {
                self.advance();
                Ok(CompareOp::Gt)
            }
            Token::Lt => {
                self.advance();
                Ok(CompareOp::Lt)
            }
            Token::Gte => {
                self.advance();
                Ok(CompareOp::Gte)
            }
            Token::Lte => {
                self.advance();
                Ok(CompareOp::Lte)
            }
            Token::Keyword(Keyword::Like) => {
                self.advance();
                Ok(CompareOp::Like)
            }
            Token::Keyword(Keyword::In) => {
                self.advance();
                Ok(CompareOp::In)
            }
            _ => Err(ParseError(format!(
                "expected comparison operator, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Token::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Value::String(s))
            }
            Token::NumberLit(n) => {
                self.advance();
                Ok(Value::Number(n))
            }
            Token::IntegerLit(n) => {
                self.advance();
                Ok(Value::Integer(n))
            }
            Token::Dash => {
                // Negative number: - followed by number
                self.advance();
                match self.peek() {
                    Token::NumberLit(n) => {
                        self.advance();
                        Ok(Value::Number(-n))
                    }
                    Token::IntegerLit(n) => {
                        self.advance();
                        Ok(Value::Integer(-n))
                    }
                    _ => Err(ParseError(format!(
                        "expected number after '-', got {:?}",
                        self.peek()
                    ))),
                }
            }
            Token::LParen => {
                self.advance();
                let mut items = Vec::new();
                if !matches!(self.peek(), Token::RParen) {
                    items.push(self.parse_value()?);
                    while self.check_comma() {
                        self.advance();
                        items.push(self.parse_value()?);
                    }
                }
                self.expect_token(&Token::RParen)?;
                Ok(Value::List(items))
            }
            _ => Err(ParseError(format!("expected value, got {:?}", self.peek()))),
        }
    }

    pub(crate) fn parse_order_by(&mut self) -> Result<OrderBy, ParseError> {
        let field = self.expect_field_name()?;
        let descending = if self.check_keyword(Keyword::Desc) {
            self.advance();
            true
        } else if self.check_keyword(Keyword::Asc) {
            self.advance();
            false
        } else {
            false // default ascending
        };
        Ok(OrderBy { field, descending })
    }

    pub(crate) fn parse_assignments(&mut self) -> Result<Vec<Assignment>, ParseError> {
        let mut assignments = vec![self.parse_assignment()?];
        while self.check_comma() {
            self.advance();
            assignments.push(self.parse_assignment()?);
        }
        Ok(assignments)
    }

    fn parse_assignment(&mut self) -> Result<Assignment, ParseError> {
        let field = self.expect_field_name()?;
        self.expect_token(&Token::Eq)?;
        let value = self.parse_value()?;
        Ok(Assignment { field, value })
    }

    // ── Token helpers ──

    pub(crate) fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof)
    }

    pub(crate) fn advance(&mut self) {
        self.pos += 1;
    }

    pub(crate) fn check_keyword(&self, kw: Keyword) -> bool {
        matches!(self.peek(), Token::Keyword(k) if k == kw)
    }

    pub(crate) fn check_comma(&self) -> bool {
        matches!(self.peek(), Token::Comma)
    }

    pub(crate) fn check_pipe(&self) -> bool {
        matches!(self.peek(), Token::Pipe)
    }

    pub(crate) fn eat_semicolon(&mut self) {
        if matches!(self.peek(), Token::Semicolon) {
            self.advance();
        }
    }

    pub(crate) fn expect_keyword(&mut self, kw: Keyword) -> Result<(), ParseError> {
        if self.check_keyword(kw) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError(format!(
                "expected {:?}, got {:?}",
                kw,
                self.peek()
            )))
        }
    }

    pub(crate) fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Token::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError(format!(
                "expected string literal, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn expect_u32(&mut self) -> Result<u32, ParseError> {
        match self.peek() {
            Token::IntegerLit(n) if n >= 0 => {
                let value = u32::try_from(n).map_err(|_| {
                    ParseError(format!("integer {n} is out of range (max {})", u32::MAX))
                })?;
                self.advance();
                Ok(value)
            }
            _ => Err(ParseError(format!(
                "expected positive integer, got {:?}",
                self.peek()
            ))),
        }
    }

    pub(crate) fn expect_f32(&mut self) -> Result<f32, ParseError> {
        match self.peek() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(n as f32)
            }
            Token::IntegerLit(n) => {
                self.advance();
                Ok(n as f32)
            }
            _ => Err(ParseError(format!(
                "expected number, got {:?}",
                self.peek()
            ))),
        }
    }

    /// A number that may be negative. The lexer emits `-` as its own token
    /// (ranges like `0-33` are far more common than negative literals), so a
    /// leading `-` is read here, where a sign is meant.
    pub(crate) fn expect_signed_f32(&mut self) -> Result<f32, ParseError> {
        if matches!(self.peek(), Token::Dash) {
            self.advance();
            Ok(-self.expect_f32()?)
        } else {
            self.expect_f32()
        }
    }

    pub(crate) fn expect_token(&mut self, expected: &Token) -> Result<(), ParseError> {
        let tok = self.peek();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError(format!(
                "expected {:?}, got {:?}",
                expected, tok
            )))
        }
    }

    pub(crate) fn expect_ident_eq(&mut self, name: &str) -> Result<(), ParseError> {
        match self.peek() {
            Token::Ident(ref s) if s.eq_ignore_ascii_case(name) => {
                self.advance();
                Ok(())
            }
            // Also accept keywords that match field names
            Token::Keyword(kw) if kw.as_field_name().eq_ignore_ascii_case(name) => {
                self.advance();
                Ok(())
            }
            _ => Err(ParseError(format!(
                "expected '{}', got {:?}",
                name,
                self.peek()
            ))),
        }
    }

    pub(crate) fn expect_field_name(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Token::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            Token::Keyword(kw) => {
                // Allow keywords as field names (e.g., "layer", "confidence", "relation")
                let name = kw.as_field_name().to_string();
                self.advance();
                Ok(name)
            }
            _ => Err(ParseError(format!(
                "expected field name, got {:?}",
                self.peek()
            ))),
        }
    }
}
