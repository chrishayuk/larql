//! The grouped top-level `larql --help`.
//!
//! clap 4 lists a command's subcommands as one flat block in declaration
//! order. `next_help_heading` on a subcommand variant only titles *that
//! subcommand's own* options; it does not group subcommands in the parent's
//! help. So the root help is rendered here instead: everyday verbs first,
//! then each family of tooling under its own heading.
//!
//! The groups name commands; they do not define them. Every command still
//! comes from the `Commands` enum, a name compiled out of this build
//! (`research` off) is skipped, and a command no group names is still
//! listed, under [`UNGROUPED_HEADING`] — so this table can make a command
//! harder to find, never invisible. The tests pin that every command the
//! enum defines is grouped exactly once.

use clap::Command;

/// The one-line description `larql --help` opens with.
pub(crate) const TAGLINE: &str = "LARQL — run, serve, query and edit AI models";

/// One help heading and the commands listed under it, in display order.
pub(crate) struct HelpGroup {
    pub heading: &'static str,
    pub commands: &'static [&'static str],
}

/// Heading for any command the table below does not place.
pub(crate) const UNGROUPED_HEADING: &str = "Other";

/// The root help's layout. The first group is the everyday surface: what a
/// newcomer runs to get a model, run it, serve it and query it.
pub(crate) const HELP_GROUPS: &[HelpGroup] = &[
    HelpGroup {
        heading: "Commands",
        commands: &[
            "run", "chat", "pull", "list", "show", "rm", "link", "serve", "repl", "lql", "vindex3",
            "bench", "doctor", "help",
        ],
    },
    HelpGroup {
        heading: "Models and cache",
        commands: &["model", "slice", "publish", "registry"],
    },
    HelpGroup {
        heading: "Build",
        commands: &[
            "extract",
            "extract-index",
            "build",
            "compile",
            "convert",
            "hf",
            "verify",
        ],
    },
    HelpGroup {
        heading: "Server",
        commands: &["server-capabilities"],
    },
    HelpGroup {
        heading: "Factory",
        commands: &["recipe", "capabilities", "inspect-hf", "card"],
    },
    HelpGroup {
        heading: "Query (graph files)",
        commands: &["query", "describe", "stats", "validate", "merge", "filter"],
    },
    HelpGroup {
        heading: "Research and diagnostics",
        commands: &[
            "accuracy",
            "shannon",
            "diag",
            "dec-bench",
            "k3-ledger",
            "parity",
            "moe-locality",
            "optimizer-mcp",
            "dev",
        ],
    },
];

/// Gap between a command's name column and its description, as clap pads.
const NAME_GAP: usize = 2;
/// Indent of a command line under its heading, as clap indents.
const INDENT: &str = "  ";

/// Install the grouped listing as the root command's help.
///
/// Builds the command first so clap's generated `help` subcommand exists
/// and is listed with the rest.
pub(crate) fn apply(mut cmd: Command) -> Command {
    cmd.build();
    let header = *cmd.get_styles().get_header();
    let groups = render_groups(&cmd);
    let template = format!(
        "{{about-with-newline}}\n{{usage-heading}} {{usage}}\n\n{{before-help}}\
         {}Options:{}\n{{options}}{{after-help}}",
        header.render(),
        header.render_reset()
    );
    cmd.before_help(groups).help_template(template)
}

/// The visible subcommands, grouped under their headings, styled like
/// clap's own headings. Group listing text goes in `before_help`, which
/// clap writes verbatim: a description containing `{…}` cannot be read
/// as a template tag.
pub(crate) fn render_groups(cmd: &Command) -> String {
    let header = *cmd.get_styles().get_header();
    let literal = *cmd.get_styles().get_literal();
    let visible: Vec<&Command> = cmd.get_subcommands().filter(|c| !c.is_hide_set()).collect();
    let width = visible
        .iter()
        .map(|c| c.get_name().len())
        .max()
        .unwrap_or(0);

    let mut sections: Vec<(&str, Vec<&Command>)> = HELP_GROUPS
        .iter()
        .map(|g| {
            let members = g
                .commands
                .iter()
                .filter_map(|name| visible.iter().copied().find(|c| c.get_name() == *name))
                .collect();
            (g.heading, members)
        })
        .collect();
    let ungrouped: Vec<&Command> = visible
        .iter()
        .copied()
        .filter(|c| group_of(c.get_name()).is_none())
        .collect();
    sections.push((UNGROUPED_HEADING, ungrouped));

    let mut blocks = Vec::new();
    for (heading, members) in sections.into_iter().filter(|(_, m)| !m.is_empty()) {
        let mut block = format!("{}{heading}:{}", header.render(), header.render_reset());
        for c in members {
            let about = c.get_about().map(|a| a.to_string()).unwrap_or_default();
            let name = c.get_name();
            let pad = " ".repeat(width - name.len() + NAME_GAP);
            block.push_str(&format!(
                "\n{INDENT}{}{name}{}{pad}{about}",
                literal.render(),
                literal.render_reset()
            ));
        }
        blocks.push(block);
    }
    blocks.join("\n\n")
}

/// The heading `name` is listed under, if the table places it.
pub(crate) fn group_of(name: &str) -> Option<&'static str> {
    HELP_GROUPS
        .iter()
        .find(|g| g.commands.contains(&name))
        .map(|g| g.heading)
}

#[cfg(test)]
mod tests;
