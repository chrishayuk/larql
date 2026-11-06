//! Tests for the grouped root help.

use std::collections::BTreeSet;

use clap::{Command, CommandFactory};

use super::{apply, group_of, render_groups, HELP_GROUPS, TAGLINE, UNGROUPED_HEADING};

fn root() -> Command {
    apply(crate::Cli::command())
}

fn help_text() -> String {
    root().render_help().to_string()
}

/// Every command the enum defines is placed by the table, so none falls
/// into the catch-all heading.
#[test]
fn every_command_is_grouped() {
    let cmd = root();
    let ungrouped: Vec<_> = cmd
        .get_subcommands()
        .map(|c| c.get_name())
        .filter(|n| group_of(n).is_none())
        .collect();
    assert!(ungrouped.is_empty(), "add to HELP_GROUPS: {ungrouped:?}");
}

/// No command is listed twice.
#[test]
fn no_command_is_grouped_twice() {
    let mut seen = BTreeSet::new();
    for name in HELP_GROUPS.iter().flat_map(|g| g.commands) {
        assert!(seen.insert(*name), "{name} is in two groups");
    }
}

/// In a research build every name the table lists is a real command — a
/// renamed command must not leave a stale entry behind.
#[cfg(feature = "research")]
#[test]
fn every_grouped_name_is_a_real_command() {
    let cmd = root();
    for name in HELP_GROUPS.iter().flat_map(|g| g.commands) {
        assert!(
            cmd.find_subcommand(name).is_some(),
            "HELP_GROUPS names `{name}`, which is not a command"
        );
    }
}

#[test]
fn help_opens_with_the_tagline_and_lists_everyday_verbs_first() {
    let help = help_text();
    assert!(help.starts_with(TAGLINE), "{help}");
    let first_heading = format!("{}:", HELP_GROUPS[0].heading);
    let everyday = help.find(&first_heading).expect("everyday heading");
    let research = help
        .find("Research and diagnostics:")
        .expect("research heading");
    assert!(everyday < research);
    for verb in HELP_GROUPS[0].commands {
        let at = help
            .find(&format!("\n  {verb} "))
            .unwrap_or_else(|| panic!("{verb} missing from:\n{help}"));
        assert!(at < research, "{verb} is listed after research tooling");
    }
    assert!(help.contains("Options:"), "{help}");
    assert!(help.contains("--help"), "{help}");
    assert!(!help.contains(UNGROUPED_HEADING), "{help}");
}

/// A command the table does not name is still listed, under the
/// catch-all heading; a hidden one is not.
#[test]
fn an_unplaced_command_is_listed_under_other_and_hidden_ones_are_not() {
    let cmd = Command::new("t")
        .subcommand(Command::new("run").about("everyday"))
        .subcommand(Command::new("brand-new").about("not in any group"))
        .subcommand(Command::new("secret").hide(true));
    let text = render_groups(&cmd);
    let plain = clap::builder::StyledStr::from(text).to_string();
    assert!(plain.contains("Commands:\n  run"), "{plain}");
    assert!(
        plain.contains(&format!(
            "{UNGROUPED_HEADING}:\n  brand-new  not in any group"
        )),
        "{plain}"
    );
    assert!(!plain.contains("secret"), "{plain}");
}
