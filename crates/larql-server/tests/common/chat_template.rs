//! A chat template for VINDEX3 test fixtures.
//!
//! The server renders VINDEX3 conversations through the container's own
//! template and refuses a multi-message conversation on a container that
//! has none, so a fixture that is served conversations must carry one.
//! [`TRANSCRIPT_TEMPLATE`] is a role-labelled transcript
//! (`User: …\nAssistant: `): any word-level fixture tokenizer splits it
//! on whitespace, so the tests keep their token streams.

use std::path::Path;

/// The template file name a container's own template is read from.
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// `System:` / `User:` / `Assistant:`-labelled turns, one per line, then
/// the open assistant label when a generation prompt is asked for.
pub const TRANSCRIPT_TEMPLATE: &str = "{%- for m in messages -%}\
{%- if m.role == 'system' %}System{% elif m.role == 'user' %}User\
{%- elif m.role == 'assistant' %}Assistant{% else %}{{ m.role }}{% endif -%}\
: {{ m.content }}\n{% endfor -%}\
{%- if add_generation_prompt %}Assistant: {% endif -%}";

/// Give the container at `dir` the transcript template.
pub fn write_transcript_template(dir: &Path) {
    std::fs::write(dir.join(CHAT_TEMPLATE_FILE), TRANSCRIPT_TEMPLATE)
        .expect("write the fixture's chat template");
}
