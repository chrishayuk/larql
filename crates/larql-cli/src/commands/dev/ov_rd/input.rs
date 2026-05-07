use std::collections::HashMap;
use std::path::PathBuf;

use super::types::{HeadId, PqConfig, PromptRecord};

pub(super) fn load_prompts(
    path: &PathBuf,
    max_prompts: Option<usize>,
) -> Result<Vec<PromptRecord>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut prompts = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        prompts.push(serde_json::from_str::<PromptRecord>(line)?);
        if max_prompts.is_some_and(|n| prompts.len() >= n) {
            break;
        }
    }
    Ok(prompts)
}

pub(super) fn limit_prompts_per_stratum(
    prompts: Vec<PromptRecord>,
    max_per_stratum: usize,
) -> Vec<PromptRecord> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut selected = Vec::new();
    for prompt in prompts {
        let key = prompt
            .stratum
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let count = counts.entry(key).or_default();
        if *count < max_per_stratum {
            *count += 1;
            selected.push(prompt);
        }
    }
    selected
}

pub(super) fn split_prompt_records(
    prompts: &[PromptRecord],
    eval_mod: usize,
    eval_offset: usize,
) -> Result<(Vec<PromptRecord>, Vec<PromptRecord>), Box<dyn std::error::Error>> {
    if eval_mod == 0 {
        return Err("--eval-mod must be greater than zero".into());
    }
    if eval_offset >= eval_mod {
        return Err("--eval-offset must be smaller than --eval-mod".into());
    }
    let mut fit = Vec::new();
    let mut eval = Vec::new();
    for (idx, prompt) in prompts.iter().cloned().enumerate() {
        if idx % eval_mod == eval_offset {
            eval.push(prompt);
        } else {
            fit.push(prompt);
        }
    }
    if fit.is_empty() || eval.is_empty() {
        return Err("held-out split produced an empty fit or eval set".into());
    }
    eprintln!(
        "Held-out split: fit_prompts={}, eval_prompts={} (idx % {} == {})",
        fit.len(),
        eval.len(),
        eval_mod,
        eval_offset
    );
    Ok((fit, eval))
}

pub(super) fn parse_head_spec(spec: &str) -> Result<Vec<HeadId>, Box<dyn std::error::Error>> {
    let mut heads = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (layer, head) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid head spec '{part}', expected layer:head"))?;
        heads.push(HeadId {
            layer: layer.parse()?,
            head: head.parse()?,
        });
    }
    Ok(heads)
}

pub(super) fn parse_usize_list(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        values.push(part.parse()?);
    }
    Ok(values)
}

pub(super) fn parse_pq_configs(spec: &str) -> Result<Vec<PqConfig>, Box<dyn std::error::Error>> {
    let mut configs = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let fields = part.split(':').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!("invalid PQ config '{part}', expected K:groups:bits").into());
        }
        let config = PqConfig {
            k: fields[0].parse()?,
            groups: fields[1].parse()?,
            bits_per_group: fields[2].parse()?,
        };
        if config.k == 0 || config.groups == 0 || config.bits_per_group == 0 {
            return Err(format!("invalid zero value in PQ config '{part}'").into());
        }
        if !config.k.is_multiple_of(config.groups) {
            return Err(format!("PQ config '{part}' requires K divisible by groups").into());
        }
        if config.bits_per_group > 12 {
            return Err(format!("PQ config '{part}' has too many bits/group for smoke run").into());
        }
        configs.push(config);
    }
    configs.sort_by_key(|c| (c.k, c.groups, c.bits_per_group));
    configs.dedup();
    Ok(configs)
}

pub(super) fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else if !part.is_empty() {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
