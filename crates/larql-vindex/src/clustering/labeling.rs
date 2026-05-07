//! Cluster labeling — auto-generate human-readable labels for discovered clusters.
//!
//! Two approaches:
//! 1. TF-IDF: distinctive tokens per cluster (fallback)
//! 2. Member-based: average member embeddings → nearest category word

use ndarray::Array1;
use std::collections::HashMap;

use super::categories::{category_words, is_stop_word};

/// TF-IDF labeling: find distinctive tokens per cluster.
pub fn auto_label_clusters(
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    let mut cluster_tokens: Vec<HashMap<String, usize>> = vec![HashMap::new(); k];
    let mut global_tokens: HashMap<String, usize> = HashMap::new();

    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < top_tokens.len() {
            for tok in top_tokens[i].split('|') {
                let tok = tok.trim().to_lowercase();
                if tok.is_empty() || tok.len() < 3 {
                    continue;
                }
                let ascii_count = tok.chars().filter(|c| c.is_ascii_alphanumeric()).count();
                if ascii_count * 2 < tok.chars().count() {
                    continue;
                }
                if is_stop_word(&tok) {
                    continue;
                }
                *cluster_tokens[cluster].entry(tok.clone()).or_default() += 1;
                *global_tokens.entry(tok).or_default() += 1;
            }
        }
    }

    let total_features = assignments.len().max(1) as f64;
    let mut labels = Vec::with_capacity(k);
    let mut top_lists = Vec::with_capacity(k);

    for (c, cluster_tok) in cluster_tokens.iter().enumerate().take(k) {
        let cluster_size = cluster_tok.values().sum::<usize>().max(1) as f64;

        let mut scored: Vec<(String, f64)> = cluster_tok
            .iter()
            .filter(|(_, &count)| count >= 1)
            .map(|(tok, &count)| {
                let tf = count as f64 / cluster_size;
                let global = *global_tokens.get(tok).unwrap_or(&1) as f64;
                let idf = (total_features / global).ln();
                (tok.clone(), tf * idf)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = scored
            .iter()
            .filter(|(t, _)| t.len() >= 3)
            .take(5)
            .map(|(t, _)| t.clone())
            .collect();

        let label = if top.is_empty() {
            let mut freq: Vec<(String, usize)> =
                cluster_tok.iter().map(|(t, &c)| (t.clone(), c)).collect();
            freq.sort_by_key(|x| std::cmp::Reverse(x.1));
            let fallback: Vec<String> = freq
                .iter()
                .filter(|(t, _)| t.len() >= 3)
                .take(3)
                .map(|(t, _)| t.clone())
                .collect();
            if fallback.is_empty() {
                format!("cluster-{c}")
            } else {
                fallback.join("/")
            }
        } else {
            top.iter().take(3).cloned().collect::<Vec<_>>().join("/")
        };

        labels.push(label);
        top_lists.push(top);
    }

    (labels, top_lists)
}

/// Member-based labeling: for each cluster, average the embeddings of its
/// top member tokens, then find the nearest category word.
///
/// Duplicates allowed — two clusters both labeled "country" means two
/// country-related feature groups.
pub fn auto_label_clusters_from_embeddings(
    _centres: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    tokenizer: &tokenizers::Tokenizer,
    assignments: &[usize],
    top_tokens: &[String],
    k: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    let (tfidf_labels, top_lists) = auto_label_clusters(assignments, top_tokens, k);

    let categories = category_words();
    let hidden = embed.shape()[1];

    // Encode category words → embeddings
    let mut cat_embeds: Vec<(String, Array1<f32>)> = Vec::new();
    for word in &categories {
        if let Some(emb) = encode_token_with_tokenizer(word, embed, hidden, tokenizer) {
            cat_embeds.push((word.clone(), emb));
        }
    }

    // Collect member embeddings per cluster
    let mut cluster_member_embeds: Vec<Vec<Array1<f32>>> = vec![Vec::new(); k];
    for (c, member_embeds) in cluster_member_embeds.iter_mut().enumerate().take(k) {
        if let Some(members) = top_lists.get(c) {
            for tok in members.iter().take(10) {
                if let Some(emb) = encode_token_with_tokenizer(tok, embed, hidden, tokenizer) {
                    member_embeds.push(emb);
                }
            }
        }
    }

    // Score each candidate category by mean similarity to cluster members
    let mut labels = Vec::with_capacity(k);

    for (c, members) in cluster_member_embeds.iter().enumerate().take(k) {
        if members.is_empty() {
            labels.push(
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}")),
            );
            continue;
        }

        let mut best_word = String::new();
        let mut best_sim = f32::NEG_INFINITY;

        for (word, cat_embed) in &cat_embeds {
            let mean_sim: f32 = members
                .iter()
                .map(|m| larql_compute::dot(&m.view(), &cat_embed.view()))
                .sum::<f32>()
                / members.len() as f32;
            if mean_sim > best_sim {
                best_sim = mean_sim;
                best_word = word.clone();
            }
        }

        if best_sim >= 0.25 {
            labels.push(best_word);
        } else if let Some(members) = top_lists.get(c) {
            // Fallback: check if members match known entity patterns
            let pattern_label = detect_entity_pattern(members);
            labels.push(pattern_label.unwrap_or_else(|| {
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}"))
            }));
        } else {
            labels.push(
                tfidf_labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("cluster-{c}")),
            );
        }
    }

    (labels, top_lists)
}

/// Detect entity patterns from cluster member tokens.
/// If 60%+ of members match a known pattern, return the pattern label.
pub fn detect_entity_pattern(members: &[String]) -> Option<String> {
    if members.is_empty() {
        return None;
    }

    static COUNTRIES: &[&str] = &[
        "australia",
        "china",
        "chinese",
        "japan",
        "japanese",
        "germany",
        "german",
        "france",
        "french",
        "italy",
        "italian",
        "spain",
        "spanish",
        "russia",
        "russian",
        "brazil",
        "brazil",
        "india",
        "indian",
        "canada",
        "canadian",
        "mexico",
        "mexican",
        "britain",
        "british",
        "korea",
        "korean",
        "turkey",
        "turkish",
        "poland",
        "polish",
        "sweden",
        "swedish",
        "norway",
        "norwegian",
        "portugal",
        "portuguese",
        "netherlands",
        "dutch",
        "greece",
        "greek",
        "egypt",
        "egyptian",
        "argentina",
        "iran",
        "iranian",
        "thailand",
        "thai",
        "vietnam",
        "vietnamese",
        "indonesia",
        "indonesian",
        "malaysia",
        "malaysian",
        "philippines",
        "filipino",
    ];

    static LANGUAGES: &[&str] = &[
        "english",
        "french",
        "german",
        "spanish",
        "italian",
        "portuguese",
        "russian",
        "chinese",
        "japanese",
        "korean",
        "arabic",
        "hindi",
        "bengali",
        "turkish",
        "dutch",
        "polish",
        "swedish",
        "norwegian",
        "danish",
        "finnish",
        "greek",
        "czech",
        "romanian",
        "hungarian",
        "thai",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
        "swahili",
        "hebrew",
        "persian",
        "urdu",
    ];

    static MONTHS: &[&str] = &[
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];

    static NUMBERS: &[&str] = &[
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first",
        "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    ];

    let lower_members: Vec<String> = members.iter().map(|m| m.to_lowercase()).collect();
    let n = lower_members.len();
    let threshold = (n as f64 * 0.5).ceil() as usize; // 50% match

    // Check languages BEFORE countries — many language names overlap
    // (french, german, spanish are both language and country-related)
    let lang_hits = lower_members
        .iter()
        .filter(|m| LANGUAGES.contains(&m.as_str()))
        .count();
    if lang_hits >= threshold {
        return Some("language".into());
    }

    let country_hits = lower_members
        .iter()
        .filter(|m| COUNTRIES.contains(&m.as_str()))
        .count();
    if country_hits >= threshold {
        return Some("country".into());
    }

    let month_hits = lower_members
        .iter()
        .filter(|m| MONTHS.contains(&m.as_str()))
        .count();
    if month_hits >= threshold {
        return Some("month".into());
    }

    let num_hits = lower_members
        .iter()
        .filter(|m| NUMBERS.contains(&m.as_str()))
        .count();
    if num_hits >= threshold {
        return Some("number".into());
    }

    // Morphological: if most members are short suffixes/prefixes
    let suffix_hits = lower_members
        .iter()
        .filter(|m| m.len() <= 4 && m.chars().all(|c| c.is_ascii_alphabetic()))
        .count();
    if suffix_hits >= threshold {
        return Some("morphological".into());
    }

    None
}

/// Encode a token using the tokenizer and embedding matrix.
pub fn encode_token_with_tokenizer(
    tok: &str,
    embed: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    hidden: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(tok, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }
    let mut avg = Array1::<f32>::zeros(hidden);
    let mut n = 0;
    for &id in ids {
        // Skip BOS (0, 1, 2) and EOS tokens — they dilute the embedding
        if id <= 2 {
            continue;
        }
        if (id as usize) < embed.shape()[0] {
            avg += &embed.row(id as usize);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    avg /= n as f32;
    let norm = larql_compute::norm(&avg.view());
    if norm > 1e-8 {
        avg /= norm;
        Some(avg)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfidf_labels_basic() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let tokens = vec![
            "Paris".into(),
            "Berlin".into(),
            "Tokyo".into(),
            "French".into(),
            "German".into(),
            "Japanese".into(),
        ];
        let (labels, tops) = auto_label_clusters(&assignments, &tokens, 2);
        assert_eq!(labels.len(), 2);
        assert_eq!(tops.len(), 2);
    }

    #[test]
    fn tfidf_with_pipe_separated() {
        let assignments = vec![0, 0];
        let tokens = vec![
            "Paris|PARIS|Parisian".into(),
            "Berlin|BERLIN|Berliner".into(),
        ];
        let (labels, tops) = auto_label_clusters(&assignments, &tokens, 1);
        assert_eq!(labels.len(), 1);
        // Should find distinctive tokens from all pipe-separated entries
        assert!(!tops[0].is_empty());
    }

    #[test]
    fn tfidf_filters_stop_words() {
        let assignments = vec![0, 0, 0];
        let tokens = vec!["the".into(), "and".into(), "for".into()];
        let (labels, _) = auto_label_clusters(&assignments, &tokens, 1);
        // Should not contain stop words in label
        assert!(!labels[0].contains("the"));
    }

    #[test]
    fn detect_country_pattern() {
        let members = vec![
            "australia".into(),
            "italy".into(),
            "germany".into(),
            "france".into(),
            "japan".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("country".into()));
    }

    #[test]
    fn detect_language_pattern() {
        let members = vec![
            "english".into(),
            "french".into(),
            "german".into(),
            "spanish".into(),
            "italian".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("language".into()));
    }

    #[test]
    fn detect_month_pattern() {
        let members = vec![
            "january".into(),
            "february".into(),
            "march".into(),
            "october".into(),
            "november".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("month".into()));
    }

    #[test]
    fn detect_number_pattern() {
        let members = vec![
            "one".into(),
            "two".into(),
            "three".into(),
            "four".into(),
            "five".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), Some("number".into()));
    }

    #[test]
    fn detect_morphological_pattern() {
        let members = vec![
            "ing".into(),
            "tion".into(),
            "ness".into(),
            "ment".into(),
            "ity".into(),
        ];
        assert_eq!(
            detect_entity_pattern(&members),
            Some("morphological".into())
        );
    }

    #[test]
    fn detect_no_pattern() {
        let members = vec![
            "Paris".into(),
            "music".into(),
            "running".into(),
            "table".into(),
            "happy".into(),
        ];
        assert_eq!(detect_entity_pattern(&members), None);
    }

    #[test]
    fn detect_empty_members() {
        assert_eq!(detect_entity_pattern(&[]), None);
    }
}
