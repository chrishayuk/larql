//! What a help page holds. Every field is static text, kept beside the
//! statement it describes in [`super::catalogue`].

use super::topic::HelpGroup;

/// Something a statement needs before it can run. Stated once here so every
/// page words it the same way, and so the facts the session banner is built
/// from ([`super::render::vindex3_supported`]) are data, not prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// No vindex or container has to be bound.
    Nothing,
    /// Any bound vindex or container.
    Backend,
    /// A bound VINDEX3 container; a VINDEX2 vindex is refused.
    Vindex3,
    /// A bound VINDEX2 vindex; a VINDEX3 container is refused.
    Vindex2,
    /// A `tokenizer.json` in the bound vindex or container.
    Tokenizer,
    /// An external verifier found through configuration.
    Verifier,
    /// A requirement of this statement alone.
    Other(&'static str),
}

impl Need {
    /// The sentence a page prints for this need.
    pub fn text(&self) -> &'static str {
        match self {
            Need::Nothing => "No vindex or container needs to be bound.",
            Need::Backend => "A bound vindex or container: run USE first.",
            Need::Vindex3 => {
                "A bound VINDEX3 container: run USE \"<container>\" first. A VINDEX2 vindex is refused."
            }
            Need::Vindex2 => "A bound VINDEX2 vindex. A VINDEX3 container refuses the statement.",
            Need::Tokenizer => {
                "A tokenizer: tokenizer.json in the bound vindex or container. Without one the statement is refused."
            }
            Need::Verifier => {
                "An external verifier, found through configuration (HELP VERIFIER;). With none the statement is refused."
            }
            Need::Other(text) => text,
        }
    }
}

/// The help page of one statement kind.
#[derive(Debug)]
pub struct StatementHelp {
    /// The statement's verb, as [`crate::ast::Statement::verb`] reports it:
    /// what `HELP <name>;` is asked with.
    pub name: &'static str,
    /// Other words that find this page.
    pub aliases: &'static [&'static str],
    pub group: HelpGroup,
    /// Every form of the statement. The first is the skeleton the overview
    /// lists.
    pub syntax: &'static [&'static str],
    /// What it does, in one or two sentences.
    pub does: &'static [&'static str],
    /// Statements that parse, checked by the catalogue's tests.
    pub examples: &'static [&'static str],
    pub needs: &'static [Need],
    /// When its effect becomes visible, and to what.
    pub effect: &'static [&'static str],
    /// The format of a file the statement reads, when it reads one.
    pub operands: &'static [&'static str],
    /// The refusals a caller is likely to hit.
    pub refusals: &'static [&'static str],
    /// Names of related pages; each resolves through [`super::lookup`].
    pub related: &'static [&'static str],
}

impl StatementHelp {
    /// The one-line form the overview lists.
    pub fn skeleton(&self) -> &'static str {
        self.syntax.first().copied().unwrap_or(self.name)
    }

    pub fn needs(&self, need: Need) -> bool {
        self.needs.contains(&need)
    }
}

/// A page about how statements are run, not about one statement.
#[derive(Debug)]
pub struct ConceptHelp {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub group: HelpGroup,
    /// The line the overview lists beside the name.
    pub summary: &'static str,
    pub body: &'static [&'static str],
    pub related: &'static [&'static str],
}

/// One group of the overview.
#[derive(Debug)]
pub struct GroupHelp {
    /// The word `HELP <name>;` is asked with.
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// The heading the overview prints.
    pub title: &'static str,
    pub intro: &'static [&'static str],
}
