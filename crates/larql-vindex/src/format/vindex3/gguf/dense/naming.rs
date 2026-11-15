//! `general.name` and friends, spelled the way llama.cpp's converter
//! spells them.
//!
//! The reference converter has no model card to read for a bare
//! checkpoint directory, so it falls back to a heuristic over the
//! directory's name (`gguf-py/gguf/metadata.py`,
//! `get_model_id_components`) and a size label over the parameter count
//! (`gguf/utility.py`, `size_label`). These are display metadata — the
//! runtime reads none of them — but they are part of the file, so this is
//! a port of both, applied to the name the container records for its
//! source artifact.

/// Naming components, as `get_model_id_components` returns them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelNaming {
    pub name: Option<String>,
    pub organization: Option<String>,
    pub basename: Option<String>,
    pub finetune: Option<String>,
    pub version: Option<String>,
    pub size_label: Option<String>,
}

/// Size suffixes in ascending powers of a thousand, `" KMBT".find(c)`.
const SIZE_SUFFIXES: &str = " KMBT";
const THOUSAND: f64 = 1000.0;
/// Word-like size labels the heuristic recognises.
const WORD_SIZES: &[&str] = &["small", "mini", "medium", "large", "xl", "xxl"];
const FINETUNE_WORDS: &[&str] = &["chat", "instruct", "vision", "lora"];
/// A label further than 7/8 of the model size away is a context length.
const SIZE_LABEL_SLACK_EIGHTHS: i64 = 7;
const EIGHTHS: i64 = 8;

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `(v|iter)?\d+([.]\d+)*`, case-insensitive.
fn is_version(part: &str) -> bool {
    let lower = part.to_ascii_lowercase();
    let rest = lower
        .strip_prefix("iter")
        .or_else(|| lower.strip_prefix('v'))
        .unwrap_or(&lower);
    rest.split('.').all(all_digits)
}

/// `i?q\d(_\w)*|b?fp?(16|32)`, case-insensitive.
fn is_quant_type(part: &str) -> bool {
    let lower = part.to_ascii_lowercase();
    let q = lower.strip_prefix('i').unwrap_or(&lower);
    if let Some(after_q) = q.strip_prefix('q') {
        let mut chars = after_q.chars();
        if chars.next().is_some_and(|c| c.is_ascii_digit()) {
            // `(_\w)*`: underscore-and-one-word-character pairs.
            let tail: Vec<char> = chars.collect();
            return tail.len().is_multiple_of(2)
                && tail
                    .chunks(2)
                    .all(|p| p[0] == '_' && (p[1].is_alphanumeric() || p[1] == '_'));
        }
    }
    let f = lower.strip_prefix('b').unwrap_or(&lower);
    let Some(after_f) = f.strip_prefix('f') else {
        return false;
    };
    let bits = after_f.strip_prefix('p').unwrap_or(after_f);
    matches!(bits, "16" | "32")
}

/// `(([A]|\d+[x])?\d+([._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)`.
fn is_size(part: &str) -> bool {
    let lower = part.to_ascii_lowercase();
    if WORD_SIZES.contains(&lower.as_str()) {
        return true;
    }
    let b = lower.as_bytes();
    let mut i = 0;
    let digits = |i: &mut usize| {
        let start = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        *i > start
    };
    // Optional `a` or `\d+x` prefix, then the mandatory digits.
    if b.first() == Some(&b'a') {
        i = 1;
    } else {
        let save = i;
        if !(digits(&mut i) && b.get(i) == Some(&b'x')) {
            i = save;
        } else {
            i += 1;
        }
    }
    if !digits(&mut i) {
        return false;
    }
    if matches!(b.get(i), Some(b'.') | Some(b'_')) {
        i += 1;
        if !digits(&mut i) {
            return false;
        }
    }
    if !matches!(b.get(i), Some(b'k' | b'm' | b'b' | b't')) {
        return false;
    }
    i += 1;
    if b.get(i).is_some_and(|c| c.is_ascii_digit()) {
        i += 1;
    }
    i == b.len()
}

/// Python's `str.islower()`: some cased character, none upper-case.
fn py_islower(s: &str) -> bool {
    s.chars().any(char::is_lowercase) && !s.chars().any(char::is_uppercase)
}

/// Python's `str.title()`.
fn py_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_cased = false;
    for c in s.chars() {
        if prev_cased {
            out.extend(c.to_lowercase());
        } else {
            out.extend(c.to_uppercase());
        }
        prev_cased = c.is_alphabetic();
    }
    out
}

/// `Metadata.id_to_title`.
pub fn id_to_title(s: &str) -> String {
    s.trim()
        .replace('-', " ")
        .split_whitespace()
        .map(|w| {
            let numbered = w.starts_with(|c: char| c.is_ascii_digit())
                || (w.starts_with('v') && is_version(w));
            if py_islower(w) && !numbered {
                py_title(w)
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default)]
struct Tags {
    basename: bool,
    size_label: bool,
    finetune: bool,
    version: bool,
    kind: bool,
}

impl Tags {
    fn is_empty(&self) -> bool {
        !(self.basename || self.size_label || self.finetune || self.version || self.kind)
    }
}

/// Normalise a size part the way the heuristic does before tagging it.
fn normalise_size(part: &str) -> String {
    let mut p = part.replace('_', ".");
    let chars: Vec<char> = p.chars().collect();
    if chars.last().is_some_and(char::is_ascii_digit) && chars.len() >= 2 {
        // `7b1` → `7.1b`.
        let n = chars.len();
        let mut fixed: String = chars[..n - 2].iter().collect();
        fixed.push('.');
        fixed.push(chars[n - 1]);
        fixed.push(chars[n - 2]);
        p = fixed;
    }
    let chars: Vec<char> = p.chars().collect();
    let n = chars.len();
    if n > 1 && chars[n - 2].is_ascii_digit() && "kmbt".contains(chars[n - 1]) {
        p = format!("{}{}", &p[..p.len() - 1], chars[n - 1].to_ascii_uppercase());
    }
    p
}

/// Whether a size label is far enough from the real size to be a context
/// length instead (`label_params` vs `total_params`).
fn looks_like_context(part: &str, total_params: i64) -> Option<bool> {
    let (num, suffix) = part.split_at(part.len() - 1);
    // `" KMBT".find(c)` is -1 for a suffix it does not hold.
    let power = SIZE_SUFFIXES
        .find(suffix.chars().next()?)
        .map_or(-1, |p| p as i32);
    let label = num.parse::<f64>().ok()? * THOUSAND.powi(power);
    Some(if total_params < 0 {
        label < (total_params.abs() / EIGHTHS) as f64
    } else {
        (label - total_params as f64).abs()
            > (SIZE_LABEL_SLACK_EIGHTHS * total_params / EIGHTHS) as f64
    })
}

/// `Metadata.get_model_id_components`, then `id_to_title` on the name
/// and organisation as `apply_metadata_heuristic` does.
pub fn naming_from_id(model_id: &str, total_params: i64) -> ModelNaming {
    if model_id.contains(' ') {
        return ModelNaming {
            name: Some(id_to_title(model_id)),
            ..Default::default()
        };
    }
    let (org, full) = match model_id.split_once('/') {
        Some((o, f)) => ((!o.starts_with('.')).then(|| o.to_string()), f.to_string()),
        None => (None, model_id.to_string()),
    };
    let mut parts: Vec<String> = full
        .split('-')
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect();
    let mut tags: Vec<Tags> = parts.iter().map(|_| Tags::default()).collect();

    for (i, part) in parts.iter_mut().enumerate() {
        let lower = part.to_ascii_lowercase();
        if is_version(part) {
            tags[i].version = true;
        } else if is_quant_type(part) {
            tags[i].kind = true;
            *part = part.to_uppercase();
        } else if i > 0 && is_size(part) {
            let mut p = normalise_size(part);
            if total_params != 0 && !WORD_SIZES.contains(&lower.as_str()) {
                if let Some(true) = looks_like_context(&p, total_params) {
                    tags[i].finetune = true;
                    let (head, last) = p.split_at(p.len() - 1);
                    p = format!("{head}{}", last.to_lowercase());
                }
            }
            if tags[i].is_empty() {
                tags[i].size_label = true;
            }
            *part = p;
        } else if i > 0 && FINETUNE_WORDS.contains(&lower.as_str()) {
            if total_params < 0 && lower == "lora" {
                tags[i].kind = true;
            } else {
                tags[i].finetune = true;
            }
        }
    }

    // Word sizes yield to numeric ones.
    let numeric_size = parts
        .iter()
        .zip(&tags)
        .any(|(p, t)| t.size_label && p.chars().any(|c| c.is_ascii_digit()));
    if numeric_size {
        for (p, t) in parts.iter().zip(tags.iter_mut()) {
            if t.size_label && p.chars().all(char::is_alphabetic) {
                t.size_label = false;
            }
        }
    }

    let mut at_start = true;
    for (p, t) in parts.iter().zip(tags.iter_mut()) {
        let alpha_start = p.chars().next().is_some_and(char::is_alphabetic);
        if at_start && ((t.is_empty() && alpha_start) || t.version) {
            t.basename = true;
        } else {
            at_start = false;
            if t.is_empty() {
                t.finetune = true;
            }
        }
    }
    for t in tags.iter_mut().rev() {
        let others = t.size_label || t.finetune || t.version || t.kind;
        if t.basename && others {
            t.basename = false;
        } else {
            break;
        }
    }

    let join = |pick: &dyn Fn(&Tags) -> bool| -> Option<String> {
        let mut seen: Vec<&str> = Vec::new();
        for (p, t) in parts.iter().zip(&tags) {
            if pick(t) && !seen.contains(&p.as_str()) {
                seen.push(p);
            }
        }
        (!seen.is_empty()).then(|| seen.join("-"))
    };
    let joined_all = |pick: &dyn Fn(&Tags) -> bool| -> Option<String> {
        let v: Vec<&str> = parts
            .iter()
            .zip(&tags)
            .filter(|(_, t)| pick(t))
            .map(|(p, _)| p.as_str())
            .collect();
        (!v.is_empty()).then(|| v.join("-"))
    };
    let mut basename = joined_all(&|t| t.basename);
    let size_label = join(&|t| t.size_label);
    let finetune = joined_all(&|t| t.finetune);
    let version = joined_all(&|t| t.version && !t.basename);
    if size_label.is_none() && finetune.is_none() && version.is_none() {
        basename = None;
    }
    ModelNaming {
        name: Some(id_to_title(&full)),
        organization: org.map(|o| id_to_title(&o)),
        basename,
        finetune,
        version,
        size_label,
    }
}

/// `model_weight_count_rounded_notation(n, min_digits=2)`.
pub fn size_label(total_params: u64) -> String {
    let n = total_params as f64;
    let (scaled, suffix) = if n > 1e12 {
        (n * 1e-12, 'T')
    } else if n > 1e9 {
        (n * 1e-9, 'B')
    } else if n > 1e6 {
        (n * 1e-6, 'M')
    } else {
        (n * 1e-3, 'K')
    };
    const MIN_DIGITS: usize = 2;
    let whole = format!("{}", scaled.round_ties_even() as u64);
    let significant = whole.trim_start_matches('0').len();
    let fix = MIN_DIGITS.saturating_sub(significant);
    format!("{scaled:.fix$}{suffix}")
}
