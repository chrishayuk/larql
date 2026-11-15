//! Parser for `HELP [<topic>];`.

use super::{ParseError, Parser};
use crate::ast::Statement;
use crate::lexer::Token;

impl Parser {
    /// `HELP;` or `HELP <topic>;`. `HELP` is read as a word in this
    /// position, not reserved. The topic is every word up to the end of the
    /// statement, keywords included (`HELP REMOVE DIRECTION;`), or one
    /// quoted string (`HELP "exit codes";`). Whether it names a page is the
    /// executor's question: the parser only collects it.
    pub(crate) fn parse_help(&mut self) -> Result<Statement, ParseError> {
        self.expect_ident_eq("HELP")?;
        let mut words: Vec<String> = Vec::new();
        loop {
            match self.peek() {
                Token::Semicolon | Token::Eof | Token::Pipe => break,
                Token::Ident(word) | Token::StringLit(word) => words.push(word),
                Token::Keyword(keyword) => words.push(keyword.as_field_name().to_uppercase()),
                _ => {
                    return Err(ParseError(format!(
                        "HELP takes a topic in words, such as HELP REMOVE DIRECTION; got {}. \
                         Run HELP; for the topics.",
                        self.found()
                    )))
                }
            }
            self.advance();
        }
        self.eat_semicolon();
        let topic = (!words.is_empty()).then(|| words.join(" "));
        Ok(Statement::Help { topic })
    }
}
