//! LQL Parser — recursive descent from token stream to AST.

mod helpers;
mod introspection;
mod lifecycle;
mod mutation;
mod patch;
mod query;
mod trace;

#[cfg(test)]
mod tests;

use crate::ast::*;
use crate::lexer::{Keyword, Token};

pub struct Parser {
    pub(crate) tokens: Vec<Token>,
    pub(crate) pos: usize,
}

#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Statement, ParseError> {
        let stmt = self.parse_statement()?;
        let stmt = if self.check_pipe() {
            self.advance();
            let right = self.parse_statement()?;
            Statement::Pipe {
                left: Box::new(stmt),
                right: Box::new(right),
            }
        } else {
            stmt
        };

        if !matches!(self.peek(), Token::Eof) {
            return Err(ParseError(format!(
                "unexpected trailing token: {:?}",
                self.peek()
            )));
        }

        Ok(stmt)
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek() {
            Token::Keyword(Keyword::Extract) => self.parse_extract(),
            Token::Keyword(Keyword::Compile) => self.parse_compile(),
            Token::Keyword(Keyword::Diff) => self.parse_diff(),
            Token::Keyword(Keyword::Use) => self.parse_use(),
            Token::Keyword(Keyword::Walk) => self.parse_walk(),
            Token::Keyword(Keyword::Infer) => self.parse_infer(),
            Token::Keyword(Keyword::Select) => self.parse_select(),
            Token::Keyword(Keyword::Describe) => self.parse_describe(),
            Token::Keyword(Keyword::Explain) => self.parse_explain(),
            Token::Keyword(Keyword::Insert) => self.parse_insert(),
            Token::Keyword(Keyword::Delete) => self.parse_delete(),
            Token::Keyword(Keyword::Update) => self.parse_update(),
            Token::Keyword(Keyword::Merge) => self.parse_merge(),
            Token::Keyword(Keyword::Rebalance) => self.parse_rebalance(),
            Token::Keyword(Keyword::Show) => self.parse_show(),
            Token::Keyword(Keyword::Stats) => self.parse_stats(),
            Token::Keyword(Keyword::Verify) => self.parse_verify(),
            Token::Keyword(Keyword::Begin) => self.parse_begin(),
            Token::Keyword(Keyword::Save) => self.parse_save(),
            Token::Keyword(Keyword::Apply) => self.parse_apply(),
            Token::Keyword(Keyword::Remove) => self.parse_remove(),
            Token::Keyword(Keyword::Trace) => self.parse_trace(),
            Token::Keyword(Keyword::Compact) => self.parse_compact(),
            // `ADD` is a word in this position, not a reserved keyword.
            Token::Ident(ref s) if s.eq_ignore_ascii_case("ADD") => self.parse_add_low_rank(),
            _ => Err(ParseError(format!(
                "expected statement keyword, got {:?}",
                self.peek()
            ))),
        }
    }
}

/// Convenience: parse a string directly into a Statement.
pub fn parse(input: &str) -> Result<Statement, Box<dyn std::error::Error>> {
    let mut lexer = crate::lexer::Lexer::new(input);
    let tokens = lexer.tokenise()?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse()?)
}
