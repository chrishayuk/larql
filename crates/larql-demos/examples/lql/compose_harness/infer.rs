//! Forward passes through the public session API, scored on token ids.

use larql_lql::ast::Statement;
use larql_lql::Session;

use crate::consts::TOP_K;

/// One candidate of a next-token prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Pred {
    pub id: u32,
    /// Probability as a fraction.
    pub p: f64,
    /// The token as the session prints it (leading space not preserved).
    pub text: String,
}

/// The top candidates for `prompt`. The statement is built as an AST, so quotes and newlines in a
/// prompt never meet the lexer.
pub fn predict(session: &mut Session, prompt: &str) -> Result<Vec<Pred>, String> {
    let statement = Statement::Infer {
        prompt: prompt.to_string(),
        top: Some(TOP_K),
        compare: false,
        route: None,
        generate: None,
    };
    let lines = session
        .execute(&statement)
        .map_err(|e| format!("INFER {prompt:?}: {e}"))?;
    let preds: Vec<Pred> = lines.iter().filter_map(|l| parse_line(l)).collect();
    if preds.is_empty() {
        return Err(format!("INFER {prompt:?}: no predictions in {lines:?}"));
    }
    Ok(preds)
}

/// `"   1.  Paris               (80.05%)  [id 9079]"`.
fn parse_line(line: &str) -> Option<Pred> {
    let (head, rest) = line.split_once('(')?;
    let (percent, tail) = rest.split_once("%)")?;
    let id = tail
        .trim()
        .strip_prefix("[id ")?
        .strip_suffix(']')?
        .trim()
        .parse()
        .ok()?;
    let (_, text) = head.trim().split_once('.')?;
    Some(Pred {
        id,
        p: percent.trim().parse::<f64>().ok()? / 100.0,
        text: text.trim().to_string(),
    })
}

pub fn top1(preds: &[Pred]) -> &Pred {
    &preds[0]
}

/// Probability of token `id` in a prediction list, 0 when it is not among the candidates.
pub fn prob_of(preds: &[Pred], id: u32) -> f64 {
    preds.iter().find(|p| p.id == id).map_or(0.0, |p| p.p)
}
