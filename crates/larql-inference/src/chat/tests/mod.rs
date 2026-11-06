//! The conversation path: [`render_conversation`] against containers
//! written into temp directories, and [`PromptOptions`] — no model needed.

mod conversation;
mod options;

use std::path::Path;

use super::options::{ENV_NO_DEFAULT_SYSTEM, ENV_RAW_PROMPT, ENV_SYSTEM, ENV_THINKING};
use larql_vindex::format::filenames::TOKENIZER_CONFIG_JSON;

/// Clears every per-thread env override on drop (panic-safe, no lock).
struct EnvGuard;

impl Drop for EnvGuard {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
    }
}

/// Pin the prompt-affecting env vars on this thread — all unset, then
/// `vars` — for the life of the returned guard.
fn env(vars: &[(&'static str, &str)]) -> EnvGuard {
    for k in [
        ENV_RAW_PROMPT,
        ENV_THINKING,
        ENV_SYSTEM,
        ENV_NO_DEFAULT_SYSTEM,
    ] {
        larql_compute::options::set_env_override(k, None);
    }
    for &(name, value) in vars {
        larql_compute::options::set_env_override(name, Some(value));
    }
    EnvGuard
}

/// A template that shows every input the renderer gives it.
const SHOW_ALL: &str =
    "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\
{% if enable_thinking is defined %}[think={{ enable_thinking }}]{% endif %}\
{% if add_generation_prompt %}<assistant>{% endif %}";

fn write_jinja(dir: &Path, template: &str) {
    std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
}

fn write_tokenizer_config(dir: &Path, json: &str) {
    std::fs::write(dir.join(TOKENIZER_CONFIG_JSON), json).unwrap();
}
