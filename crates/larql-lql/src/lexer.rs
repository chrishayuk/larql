/// LQL Lexer — tokenises an input string into a stream of `Token`s

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Keyword(Keyword),

    // Literals
    StringLit(String),
    NumberLit(f64),
    IntegerLit(i64),

    // Punctuation
    Star,      // *
    Comma,     // ,
    Semicolon, // ;
    LParen,    // (
    RParen,    // )
    Dot,       // .
    Pipe,      // |>
    Eq,        // =
    Neq,       // !=
    Gt,        // >
    Lt,        // <
    Gte,       // >=
    Lte,       // <=
    Dash,      // -  (inside ranges like 0-33)

    // Identifiers (column names, unquoted entity names, etc.)
    Ident(String),

    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Extract,
    Compile,
    Diff,
    Use,
    Walk,
    Select,
    Describe,
    Explain,
    Insert,
    Delete,
    Update,
    Merge,
    Show,
    Stats,
    From,
    Into,
    Where,
    And,
    Or,
    Not,
    In,
    Like,
    Between,
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Top,
    Generate,
    Layers,
    Mode,
    Compare,
    At,
    Layer,
    Confidence,
    Model,
    Edges,
    Entities,
    Relation,
    Relations,
    Features,
    Models,
    Format,
    Components,
    Representations,
    Provenance,
    Authority,
    On,
    Conflict,
    KeepSource,
    KeepTarget,
    HighestConfidence,
    LastWins,
    Fail,
    Set,
    Values,
    Current,
    With,
    Examples,
    Only,
    Verbose,
    Range,
    All,
    Nearest,
    To,
    Pure,
    Hybrid,
    Dense,
    Safetensors,
    Gguf,
    AutoExtract,
    FfnGate,
    FfnDown,
    FfnUp,
    Embeddings,
    AttnOv,
    AttnQk,
    Infer,
    Syntax,
    Knowledge,
    Output,
    Weights,
    Inference,
    Begin,
    Save,
    Apply,
    Remove,
    Patch,
    Patches,
    Remote,
    Trace,
    For,
    Decompose,
    Positions,
    Brief,
    Raw,
    Attention,
    Alpha,
    Knn,
    Compose,
    Rebalance,
    Floor,
    Ceiling,
    Max,
    Until,
    Converged,
    Compact,
    Physical,
    Status,
    Route,
    Verify,
    Fallback,
    Topk,
    Exit,
}

impl Keyword {
    /// Map a keyword to its lowercase field name for use in conditions/assignments.
    /// Single exhaustive match — the compiler refuses to compile if a `Keyword`
    /// variant is added without a string here, so this can't panic at runtime.
    pub(crate) fn as_field_name(&self) -> &'static str {
        match self {
            Self::Extract => "extract",
            Self::Compile => "compile",
            Self::Diff => "diff",
            Self::Use => "use",
            Self::Walk => "walk",
            Self::Select => "select",
            Self::Describe => "describe",
            Self::Explain => "explain",
            Self::Insert => "insert",
            Self::Delete => "delete",
            Self::Update => "update",
            Self::Merge => "merge",
            Self::Show => "show",
            Self::Stats => "stats",
            Self::From => "from",
            Self::Into => "into",
            Self::Where => "where",
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::In => "in",
            Self::Like => "like",
            Self::Between => "between",
            Self::Order => "order",
            Self::By => "by",
            Self::Asc => "asc",
            Self::Desc => "desc",
            Self::Limit => "limit",
            Self::Top => "top",
            Self::Generate => "generate",
            Self::Layers => "layers",
            Self::Mode => "mode",
            Self::Compare => "compare",
            Self::At => "at",
            Self::Layer => "layer",
            Self::Confidence => "confidence",
            Self::Model => "model",
            Self::Edges => "edges",
            Self::Entities => "entities",
            Self::Relation => "relation",
            Self::Relations => "relations",
            Self::Features => "features",
            Self::Models => "models",
            Self::Format => "format",
            Self::Components => "components",
            Self::Representations => "representations",
            Self::Provenance => "provenance",
            Self::Authority => "authority",
            Self::On => "on",
            Self::Conflict => "conflict",
            Self::KeepSource => "keepsource",
            Self::KeepTarget => "keeptarget",
            Self::HighestConfidence => "highestconfidence",
            Self::LastWins => "lastwins",
            Self::Fail => "fail",
            Self::Set => "set",
            Self::Values => "values",
            Self::Current => "current",
            Self::With => "with",
            Self::Examples => "examples",
            Self::Only => "only",
            Self::Verbose => "verbose",
            Self::Range => "range",
            Self::All => "all",
            Self::Nearest => "nearest",
            Self::To => "to",
            Self::Pure => "pure",
            Self::Hybrid => "hybrid",
            Self::Dense => "dense",
            Self::Safetensors => "safetensors",
            Self::Gguf => "gguf",
            Self::AutoExtract => "auto_extract",
            Self::FfnGate => "ffn_gate",
            Self::FfnDown => "ffn_down",
            Self::FfnUp => "ffn_up",
            Self::Embeddings => "embeddings",
            Self::AttnOv => "attn_ov",
            Self::AttnQk => "attn_qk",
            Self::Infer => "infer",
            Self::Syntax => "syntax",
            Self::Knowledge => "knowledge",
            Self::Output => "output",
            Self::Weights => "weights",
            Self::Inference => "inference",
            Self::Begin => "begin",
            Self::Save => "save",
            Self::Apply => "apply",
            Self::Remove => "remove",
            Self::Patch => "patch",
            Self::Patches => "patches",
            Self::Remote => "remote",
            Self::Trace => "trace",
            Self::For => "for",
            Self::Decompose => "decompose",
            Self::Positions => "positions",
            Self::Brief => "brief",
            Self::Raw => "raw",
            Self::Attention => "attention",
            Self::Alpha => "alpha",
            Self::Knn => "knn",
            Self::Compose => "compose",
            Self::Rebalance => "rebalance",
            Self::Floor => "floor",
            Self::Ceiling => "ceiling",
            Self::Max => "max",
            Self::Until => "until",
            Self::Converged => "converged",
            Self::Compact => "compact",
            Self::Physical => "physical",
            Self::Status => "status",
            Self::Route => "route",
            Self::Verify => "verify",
            Self::Fallback => "fallback",
            Self::Topk => "topk",
            Self::Exit => "exit",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "EXTRACT" => Some(Self::Extract),
            "COMPILE" => Some(Self::Compile),
            "DIFF" => Some(Self::Diff),
            "USE" => Some(Self::Use),
            "WALK" => Some(Self::Walk),
            "SELECT" => Some(Self::Select),
            "DESCRIBE" => Some(Self::Describe),
            "EXPLAIN" => Some(Self::Explain),
            "INSERT" => Some(Self::Insert),
            "DELETE" => Some(Self::Delete),
            "UPDATE" => Some(Self::Update),
            "MERGE" => Some(Self::Merge),
            "SHOW" => Some(Self::Show),
            "STATS" => Some(Self::Stats),
            "FROM" => Some(Self::From),
            "INTO" => Some(Self::Into),
            "WHERE" => Some(Self::Where),
            "AND" => Some(Self::And),
            "OR" => Some(Self::Or),
            "NOT" => Some(Self::Not),
            "IN" => Some(Self::In),
            "LIKE" => Some(Self::Like),
            "BETWEEN" => Some(Self::Between),
            "ORDER" => Some(Self::Order),
            "BY" => Some(Self::By),
            "ASC" => Some(Self::Asc),
            "DESC" => Some(Self::Desc),
            "LIMIT" => Some(Self::Limit),
            "TOP" => Some(Self::Top),
            "GENERATE" => Some(Self::Generate),
            "LAYERS" => Some(Self::Layers),
            "MODE" => Some(Self::Mode),
            "COMPARE" => Some(Self::Compare),
            "AT" => Some(Self::At),
            "LAYER" => Some(Self::Layer),
            "CONFIDENCE" => Some(Self::Confidence),
            "MODEL" => Some(Self::Model),
            "EDGES" => Some(Self::Edges),
            "RELATION" => Some(Self::Relation),
            "RELATIONS" => Some(Self::Relations),
            "ENTITIES" => Some(Self::Entities),
            "FEATURES" => Some(Self::Features),
            "MODELS" => Some(Self::Models),
            "FORMAT" => Some(Self::Format),
            "COMPONENTS" => Some(Self::Components),
            "REPRESENTATIONS" => Some(Self::Representations),
            "PROVENANCE" => Some(Self::Provenance),
            "AUTHORITY" => Some(Self::Authority),
            "ON" => Some(Self::On),
            "CONFLICT" => Some(Self::Conflict),
            "KEEP_SOURCE" => Some(Self::KeepSource),
            "KEEP_TARGET" => Some(Self::KeepTarget),
            "HIGHEST_CONFIDENCE" => Some(Self::HighestConfidence),
            "LAST_WINS" => Some(Self::LastWins),
            "FAIL" => Some(Self::Fail),
            "FOR" => Some(Self::For),
            "SET" => Some(Self::Set),
            "VALUES" => Some(Self::Values),
            "CURRENT" => Some(Self::Current),
            "WITH" => Some(Self::With),
            "EXAMPLES" => Some(Self::Examples),
            "ONLY" => Some(Self::Only),
            "VERBOSE" => Some(Self::Verbose),
            "RANGE" => Some(Self::Range),
            "ALL" => Some(Self::All),
            "NEAREST" => Some(Self::Nearest),
            "TO" => Some(Self::To),
            "PURE" => Some(Self::Pure),
            "HYBRID" => Some(Self::Hybrid),
            "DENSE" => Some(Self::Dense),
            "SAFETENSORS" => Some(Self::Safetensors),
            "GGUF" => Some(Self::Gguf),
            "AUTO_EXTRACT" => Some(Self::AutoExtract),
            "FFN_GATE" => Some(Self::FfnGate),
            "FFN_DOWN" => Some(Self::FfnDown),
            "FFN_UP" => Some(Self::FfnUp),
            "EMBEDDINGS" => Some(Self::Embeddings),
            "ATTN_OV" => Some(Self::AttnOv),
            "ATTN_QK" => Some(Self::AttnQk),
            "INFER" => Some(Self::Infer),
            "SYNTAX" => Some(Self::Syntax),
            "KNOWLEDGE" => Some(Self::Knowledge),
            "OUTPUT" => Some(Self::Output),
            "WEIGHTS" => Some(Self::Weights),
            "INFERENCE" => Some(Self::Inference),
            "BEGIN" => Some(Self::Begin),
            "SAVE" => Some(Self::Save),
            "APPLY" => Some(Self::Apply),
            "REMOVE" => Some(Self::Remove),
            "PATCH" => Some(Self::Patch),
            "PATCHES" => Some(Self::Patches),
            "REMOTE" => Some(Self::Remote),
            "TRACE" => Some(Self::Trace),
            "DECOMPOSE" => Some(Self::Decompose),
            "POSITIONS" => Some(Self::Positions),
            "BRIEF" => Some(Self::Brief),
            "RAW" => Some(Self::Raw),
            "ATTENTION" => Some(Self::Attention),
            "ALPHA" => Some(Self::Alpha),
            "KNN" => Some(Self::Knn),
            "COMPOSE" => Some(Self::Compose),
            "REBALANCE" => Some(Self::Rebalance),
            "FLOOR" => Some(Self::Floor),
            "CEILING" => Some(Self::Ceiling),
            "MAX" => Some(Self::Max),
            "UNTIL" => Some(Self::Until),
            "CONVERGED" => Some(Self::Converged),
            "COMPACT" => Some(Self::Compact),
            "PHYSICAL" => Some(Self::Physical),
            "STATUS" => Some(Self::Status),
            "ROUTE" => Some(Self::Route),
            "VERIFY" => Some(Self::Verify),
            "FALLBACK" => Some(Self::Fallback),
            "TOPK" => Some(Self::Topk),
            "EXIT" => Some(Self::Exit),
            _ => None,
        }
    }
}

pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    pub fn tokenise(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            if tok == Token::Eof {
                tokens.push(Token::Eof);
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Ok(Token::Eof);
        }

        let ch = self.input[self.pos] as char;

        match ch {
            '*' => {
                self.pos += 1;
                Ok(Token::Star)
            }
            ',' => {
                self.pos += 1;
                Ok(Token::Comma)
            }
            ';' => {
                self.pos += 1;
                Ok(Token::Semicolon)
            }
            '(' => {
                self.pos += 1;
                Ok(Token::LParen)
            }
            ')' => {
                self.pos += 1;
                Ok(Token::RParen)
            }
            '.' => {
                self.pos += 1;
                Ok(Token::Dot)
            }
            '|' => {
                self.pos += 1;
                if self.pos < self.input.len() && self.input[self.pos] == b'>' {
                    self.pos += 1;
                    Ok(Token::Pipe)
                } else {
                    Err(LexError(format!(
                        "expected '>' after '|' at position {}",
                        self.pos
                    )))
                }
            }
            '=' => {
                self.pos += 1;
                Ok(Token::Eq)
            }
            '!' => {
                self.pos += 1;
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    Ok(Token::Neq)
                } else {
                    Err(LexError(format!(
                        "expected '=' after '!' at position {}",
                        self.pos
                    )))
                }
            }
            '>' => {
                self.pos += 1;
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    Ok(Token::Gte)
                } else {
                    Ok(Token::Gt)
                }
            }
            '<' => {
                self.pos += 1;
                if self.pos < self.input.len() && self.input[self.pos] == b'=' {
                    self.pos += 1;
                    Ok(Token::Lte)
                } else {
                    Ok(Token::Lt)
                }
            }
            '"' => self.read_string(),
            '\'' => self.read_string_single(),
            _ if ch.is_ascii_digit() => self.read_number(),
            _ if ch == '-' => {
                // Always emit Dash — ranges (0-33) are far more common than
                // negative literals in LQL. The parser handles Dash + Number
                // as a negative value where needed.
                self.pos += 1;
                Ok(Token::Dash)
            }
            _ if ch.is_ascii_alphabetic() || ch == '_' => self.read_word(),
            _ => Err(LexError(format!(
                "unexpected character '{}' at position {}",
                ch, self.pos
            ))),
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.input.len() {
            let ch = self.input[self.pos] as char;
            if ch.is_ascii_whitespace() {
                self.pos += 1;
            } else if ch == '-'
                && self.pos + 1 < self.input.len()
                && self.input[self.pos + 1] == b'-'
            {
                // Line comment: -- ...
                self.pos += 2;
                while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<Token, LexError> {
        self.read_quoted(b'"')
    }

    fn read_string_single(&mut self) -> Result<Token, LexError> {
        self.read_quoted(b'\'')
    }

    /// Scan a quoted string literal, decoding standard escape sequences
    /// (\\, \", \', \n, \t, \r). Unknown escapes pass through as the literal
    /// escapee (e.g. `\x` → `x`).
    fn read_quoted(&mut self, quote: u8) -> Result<Token, LexError> {
        self.pos += 1; // skip opening quote
        let mut out = String::new();
        // Track segments of unescaped bytes so we can copy them in bulk.
        let mut segment_start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] != quote {
            if self.input[self.pos] == b'\\' {
                // Flush the unescaped segment, then decode the escape.
                if segment_start < self.pos {
                    out.push_str(&String::from_utf8_lossy(
                        &self.input[segment_start..self.pos],
                    ));
                }
                self.pos += 1;
                if self.pos >= self.input.len() {
                    return Err(LexError("unterminated string literal".into()));
                }
                let decoded = match self.input[self.pos] {
                    b'\\' => '\\',
                    b'"' => '"',
                    b'\'' => '\'',
                    b'n' => '\n',
                    b't' => '\t',
                    b'r' => '\r',
                    b'0' => '\0',
                    other => other as char,
                };
                out.push(decoded);
                self.pos += 1;
                segment_start = self.pos;
            } else {
                self.pos += 1;
            }
        }
        if self.pos >= self.input.len() {
            return Err(LexError("unterminated string literal".into()));
        }
        if segment_start < self.pos {
            out.push_str(&String::from_utf8_lossy(
                &self.input[segment_start..self.pos],
            ));
        }
        self.pos += 1; // skip closing quote
        Ok(Token::StringLit(out))
    }

    fn read_number(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        while self.pos < self.input.len() && (self.input[self.pos] as char).is_ascii_digit() {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.pos < self.input.len() && self.input[self.pos] == b'.' {
            // Peek: if next char is a digit, it's a float. Otherwise it's an int followed by dot.
            if self.pos + 1 < self.input.len()
                && (self.input[self.pos + 1] as char).is_ascii_digit()
            {
                is_float = true;
                self.pos += 1;
                while self.pos < self.input.len() && (self.input[self.pos] as char).is_ascii_digit()
                {
                    self.pos += 1;
                }
            }
        }
        // Safe in practice: read_number only advances over ASCII digits
        // and a single '.', so the slice is always valid UTF-8. Returning
        // an error here keeps the code panic-free in case the byte filter
        // is ever loosened.
        let text = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| LexError(format!("invalid UTF-8 in numeric literal: {e}")))?;
        if is_float {
            let val: f64 = text
                .parse()
                .map_err(|_| LexError(format!("invalid number: {text}")))?;
            Ok(Token::NumberLit(val))
        } else {
            let val: i64 = text
                .parse()
                .map_err(|_| LexError(format!("invalid integer: {text}")))?;
            Ok(Token::IntegerLit(val))
        }
    }

    fn read_word(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let ch = self.input[self.pos] as char;
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        // Safe in practice: read_word only advances over ASCII alphanumerics
        // and underscore, so the slice is always valid UTF-8. Returning an
        // error here keeps the code panic-free in case the byte filter is
        // ever loosened.
        let word = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| LexError(format!("invalid UTF-8 in identifier: {e}")))?;
        if let Some(kw) = Keyword::from_str(word) {
            Ok(Token::Keyword(kw))
        } else {
            Ok(Token::Ident(word.to_string()))
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexError(pub String);

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lex error: {}", self.0)
    }
}

impl std::error::Error for LexError {}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic tokenisation ──

    #[test]
    fn walk_simple() {
        let mut lex = Lexer::new(r#"WALK "The capital of France is" TOP 5;"#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Walk)));
        assert!(matches!(tokens[1], Token::StringLit(ref s) if s == "The capital of France is"));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::Top)));
        assert!(matches!(tokens[3], Token::IntegerLit(5)));
        assert!(matches!(tokens[4], Token::Semicolon));
    }

    #[test]
    fn use_vindex() {
        let mut lex = Lexer::new(r#"USE "gemma3-4b.vindex";"#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Use)));
        assert!(matches!(tokens[1], Token::StringLit(ref s) if s == "gemma3-4b.vindex"));
    }

    #[test]
    fn select_with_conditions() {
        let mut lex =
            Lexer::new(r#"SELECT entity, relation FROM EDGES WHERE entity = "France" LIMIT 10;"#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Select)));
        assert!(matches!(tokens[1], Token::Ident(ref s) if s == "entity"));
        assert!(matches!(tokens[2], Token::Comma));
        assert!(matches!(tokens[3], Token::Keyword(Keyword::Relation)));
    }

    // ── Comments ──

    #[test]
    fn comment_skipping() {
        let mut lex = Lexer::new("-- this is a comment\nSTATS;");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Stats)));
    }

    #[test]
    fn multiple_comments() {
        let input = "-- first comment\n-- second comment\nSTATS;\n-- trailing";
        let mut lex = Lexer::new(input);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Stats)));
        assert!(matches!(tokens[1], Token::Semicolon));
        assert!(matches!(tokens[2], Token::Eof));
    }

    #[test]
    fn inline_comment_after_statement() {
        let mut lex = Lexer::new("STATS; -- inline comment");
        let tokens = lex.tokenise().unwrap();
        assert_eq!(tokens.len(), 3); // STATS, ;, EOF
    }

    // ── Numbers ──

    #[test]
    fn integer_literal() {
        let mut lex = Lexer::new("42");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::IntegerLit(42)));
    }

    #[test]
    fn float_literal() {
        let mut lex = Lexer::new("0.89");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::NumberLit(n) if (n - 0.89).abs() < 0.001));
    }

    #[test]
    fn negative_number_is_dash_plus_int() {
        let mut lex = Lexer::new("-5");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Dash));
        assert!(matches!(tokens[1], Token::IntegerLit(5)));
    }

    #[test]
    fn range_with_dash() {
        let mut lex = Lexer::new("0-33");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::IntegerLit(0)));
        assert!(matches!(tokens[1], Token::Dash));
        assert!(matches!(tokens[2], Token::IntegerLit(33)));
    }

    // ── Strings ──

    #[test]
    fn double_quoted_string() {
        let mut lex = Lexer::new(r#""hello world""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "hello world"));
    }

    #[test]
    fn single_quoted_string() {
        let mut lex = Lexer::new("'hello world'");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "hello world"));
    }

    #[test]
    fn string_with_escaped_quote_decodes() {
        let mut lex = Lexer::new(r#""hello \"world\"""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == r#"hello "world""#));
    }

    #[test]
    fn string_with_escaped_backslash_decodes() {
        let mut lex = Lexer::new(r#""C:\\path""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == r"C:\path"));
    }

    #[test]
    fn string_with_control_escapes_decodes() {
        let mut lex = Lexer::new(r#""a\nb\tc\rd\0e""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "a\nb\tc\rd\0e"));
    }

    #[test]
    fn single_quoted_string_with_escape() {
        let mut lex = Lexer::new(r"'it\'s'");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "it's"));
    }

    #[test]
    fn unknown_escape_passes_through() {
        let mut lex = Lexer::new(r#""\x""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s == "x"));
    }

    #[test]
    fn unterminated_string_after_backslash() {
        let mut lex = Lexer::new(r#""abc\"#);
        assert!(lex.tokenise().is_err());
    }

    #[test]
    fn keyword_as_field_name_covers_compact_and_status() {
        // Regression: prior implementation routed Compact / Status through an
        // `unreachable!()` and panicked when they appeared as identifiers in
        // WHERE / SET / ORDER BY clauses.
        assert_eq!(Keyword::Compact.as_field_name(), "compact");
        assert_eq!(Keyword::Status.as_field_name(), "status");
    }

    #[test]
    fn unterminated_string_error() {
        let mut lex = Lexer::new(r#""unterminated"#);
        assert!(lex.tokenise().is_err());
    }

    #[test]
    fn empty_string() {
        let mut lex = Lexer::new(r#""""#);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::StringLit(ref s) if s.is_empty()));
    }

    // ── Operators ──

    #[test]
    fn comparison_operators() {
        let mut lex = Lexer::new("= != > < >= <=");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Eq));
        assert!(matches!(tokens[1], Token::Neq));
        assert!(matches!(tokens[2], Token::Gt));
        assert!(matches!(tokens[3], Token::Lt));
        assert!(matches!(tokens[4], Token::Gte));
        assert!(matches!(tokens[5], Token::Lte));
    }

    #[test]
    fn pipe_operator() {
        let mut lex = Lexer::new("|>");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Pipe));
    }

    #[test]
    fn all_punctuation() {
        let mut lex = Lexer::new("* , ; ( ) .");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Star));
        assert!(matches!(tokens[1], Token::Comma));
        assert!(matches!(tokens[2], Token::Semicolon));
        assert!(matches!(tokens[3], Token::LParen));
        assert!(matches!(tokens[4], Token::RParen));
        assert!(matches!(tokens[5], Token::Dot));
    }

    // ── Keywords ──

    #[test]
    fn case_insensitive_keywords() {
        let mut lex = Lexer::new("walk WALK Walk wAlK");
        let tokens = lex.tokenise().unwrap();
        for (i, tok) in tokens.iter().take(4).enumerate() {
            assert!(
                matches!(tok, Token::Keyword(Keyword::Walk)),
                "token {i} should be Walk keyword"
            );
        }
    }

    #[test]
    fn all_lifecycle_keywords() {
        let mut lex = Lexer::new("EXTRACT COMPILE DIFF USE");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Extract)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::Compile)));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::Diff)));
        assert!(matches!(tokens[3], Token::Keyword(Keyword::Use)));
    }

    #[test]
    fn all_query_keywords() {
        let mut lex = Lexer::new("WALK SELECT DESCRIBE EXPLAIN");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Walk)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::Select)));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::Describe)));
        assert!(matches!(tokens[3], Token::Keyword(Keyword::Explain)));
    }

    #[test]
    fn all_mutation_keywords() {
        let mut lex = Lexer::new("INSERT DELETE UPDATE MERGE");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Insert)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::Delete)));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::Update)));
        assert!(matches!(tokens[3], Token::Keyword(Keyword::Merge)));
    }

    #[test]
    fn component_keywords() {
        let mut lex = Lexer::new("FFN_GATE FFN_DOWN FFN_UP EMBEDDINGS ATTN_OV ATTN_QK");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::FfnGate)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::FfnDown)));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::FfnUp)));
        assert!(matches!(tokens[3], Token::Keyword(Keyword::Embeddings)));
        assert!(matches!(tokens[4], Token::Keyword(Keyword::AttnOv)));
        assert!(matches!(tokens[5], Token::Keyword(Keyword::AttnQk)));
    }

    #[test]
    fn mode_keywords() {
        let mut lex = Lexer::new("HYBRID PURE DENSE");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Hybrid)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::Pure)));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::Dense)));
    }

    #[test]
    fn conflict_strategy_keywords() {
        let mut lex = Lexer::new("KEEP_SOURCE KEEP_TARGET HIGHEST_CONFIDENCE");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::KeepSource)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::KeepTarget)));
        assert!(matches!(
            tokens[2],
            Token::Keyword(Keyword::HighestConfidence)
        ));
    }

    #[test]
    fn format_keywords() {
        let mut lex = Lexer::new("SAFETENSORS GGUF");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Safetensors)));
        assert!(matches!(tokens[1], Token::Keyword(Keyword::Gguf)));
    }

    // ── Identifiers ──

    #[test]
    fn unknown_word_is_ident() {
        let mut lex = Lexer::new("my_column foobar");
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Ident(ref s) if s == "my_column"));
        assert!(matches!(tokens[1], Token::Ident(ref s) if s == "foobar"));
    }

    // ── Error cases ──

    #[test]
    fn unexpected_character_error() {
        let mut lex = Lexer::new("@");
        assert!(lex.tokenise().is_err());
    }

    #[test]
    fn incomplete_pipe_error() {
        let mut lex = Lexer::new("|x");
        assert!(lex.tokenise().is_err());
    }

    #[test]
    fn incomplete_bang_error() {
        let mut lex = Lexer::new("!x");
        assert!(lex.tokenise().is_err());
    }

    // ── Empty / whitespace ──

    #[test]
    fn empty_input() {
        let mut lex = Lexer::new("");
        let tokens = lex.tokenise().unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0], Token::Eof));
    }

    #[test]
    fn whitespace_only() {
        let mut lex = Lexer::new("   \n\t  \n  ");
        let tokens = lex.tokenise().unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0], Token::Eof));
    }

    // ── Full statement tokenisation ──

    #[test]
    fn extract_statement_tokens() {
        let input = r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN LAYERS 0-33;"#;
        let mut lex = Lexer::new(input);
        let tokens = lex.tokenise().unwrap();
        // Count non-EOF tokens
        let count = tokens.iter().filter(|t| !matches!(t, Token::Eof)).count();
        assert!(count >= 12, "expected at least 12 tokens, got {count}");
    }

    #[test]
    fn insert_statement_tokens() {
        let input = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");"#;
        let mut lex = Lexer::new(input);
        let tokens = lex.tokenise().unwrap();
        // INSERT INTO EDGES ( entity , relation , target ) VALUES ( "John" , "lives-in" , "London" ) ;
        // =19 tokens: INSERT INTO EDGES ( entity , relation , target ) VALUES ( str , str , str ) ;
        let count = tokens.iter().filter(|t| !matches!(t, Token::Eof)).count();
        assert_eq!(count, 19);
    }

    #[test]
    fn multiline_statement_tokens() {
        let input = "SELECT *\n  FROM EDGES\n  WHERE layer = 26\n  LIMIT 5;";
        let mut lex = Lexer::new(input);
        let tokens = lex.tokenise().unwrap();
        assert!(matches!(tokens[0], Token::Keyword(Keyword::Select)));
        assert!(matches!(tokens[1], Token::Star));
        assert!(matches!(tokens[2], Token::Keyword(Keyword::From)));
    }
}
