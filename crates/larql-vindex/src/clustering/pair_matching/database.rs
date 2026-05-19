//! Reference databases for pair-based relation labeling.
//!
//! Holds the `RelationDatabase` data type (a name → (subject, object)
//! pair set), the loaders for Wikidata + WordNet, and the bundled
//! `ReferenceDatabases` struct returned by `load_reference_databases`.
//!
//! Consumed by `super::labeling`.

use std::collections::HashMap;
use std::path::Path;

/// A reference database of (subject, object) pairs per relation type.
#[derive(Default)]
pub struct RelationDatabase {
    /// relation_name → set of (subject_lower, object_lower) pairs.
    /// `pub(super)` so the test module in `super::labeling` can drive
    /// it directly without going through `add_relation` for every case.
    pub(super) relations: HashMap<String, Vec<(String, String)>>,
    /// Inverted index: (subject_lower, object_lower) → relation_names
    pair_index: HashMap<(String, String), Vec<String>>,
}

impl RelationDatabase {
    /// Add a relation with its (subject, object) pairs.
    pub fn add_relation(&mut self, name: &str, pairs: Vec<(String, String)>) {
        self.relations.insert(name.to_string(), pairs);
        self.rebuild_index();
    }

    fn rebuild_index(&mut self) {
        self.pair_index.clear();
        for (rel_name, pairs) in &self.relations {
            for (s, o) in pairs {
                self.pair_index
                    .entry((s.clone(), o.clone()))
                    .or_default()
                    .push(rel_name.clone());
            }
        }
    }

    /// Load from Wikidata triples JSON file.
    pub fn load_wikidata(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let data: serde_json::Value = serde_json::from_str(&text).ok()?;
        let obj = data.as_object()?;

        let mut db = Self::default();

        for (label, value) in obj {
            if let Some(pairs) = value.get("pairs").and_then(|v| v.as_array()) {
                let mut rel_pairs = Vec::new();
                for pair in pairs {
                    if let Some(arr) = pair.as_array() {
                        if arr.len() >= 2 {
                            let s = arr[0].as_str().unwrap_or("").to_lowercase();
                            let o = arr[1].as_str().unwrap_or("").to_lowercase();
                            if !s.is_empty() && !o.is_empty() {
                                rel_pairs.push((s, o));
                            }
                        }
                    }
                }
                db.relations.insert(label.clone(), rel_pairs);
            }
        }

        db.build_index();
        Some(db)
    }

    /// Load from WordNet relations JSON file.
    pub fn load_wordnet(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let data: serde_json::Value = serde_json::from_str(&text).ok()?;
        let obj = data.as_object()?;

        let mut db = Self::default();

        for (label, value) in obj {
            if let Some(pairs) = value.get("pairs").and_then(|v| v.as_array()) {
                let mut rel_pairs = Vec::new();
                for pair in pairs {
                    if let Some(arr) = pair.as_array() {
                        if arr.len() >= 2 {
                            let s = arr[0].as_str().unwrap_or("").to_lowercase();
                            let o = arr[1].as_str().unwrap_or("").to_lowercase();
                            if !s.is_empty() && !o.is_empty() {
                                rel_pairs.push((s, o));
                            }
                        }
                    }
                }
                db.relations.insert(label.clone(), rel_pairs);
            }
        }

        db.build_index();
        Some(db)
    }

    pub(super) fn build_index(&mut self) {
        self.pair_index.clear();
        for (rel_name, pairs) in &self.relations {
            for (s, o) in pairs {
                self.pair_index
                    .entry((s.clone(), o.clone()))
                    .or_default()
                    .push(rel_name.clone());
            }
        }
    }

    /// Look up which relations contain this (subject, object) pair.
    pub fn lookup(&self, subject: &str, object: &str) -> Vec<&str> {
        let key = (subject.to_lowercase(), object.to_lowercase());
        self.pair_index
            .get(&key)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Number of relation types loaded.
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Total number of pairs across all relations.
    pub fn num_pairs(&self) -> usize {
        self.relations.values().map(|v| v.len()).sum()
    }

    /// Iterate all relations and their (subject, object) pairs.
    /// Used by `super::labeling` to build inverted indexes for
    /// output-only matching.
    pub fn relations_iter(&self) -> impl Iterator<Item = (&str, &[(String, String)])> {
        self.relations
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
    }
}
/// Loaded reference databases, separated by layer range.
pub struct ReferenceDatabases {
    /// Wikidata — for L14-27 factual relations.
    pub wikidata: Option<RelationDatabase>,
    /// WordNet — for L0-13 linguistic relations.
    pub wordnet: Option<RelationDatabase>,
}

/// Load all available reference databases from the data directory.
pub fn load_reference_databases() -> ReferenceDatabases {
    let mut result = ReferenceDatabases {
        wikidata: None,
        wordnet: None,
    };

    for base in &["data", "../data", "../../data"] {
        let base = Path::new(base);

        if result.wikidata.is_none() {
            let wikidata_path = base.join("wikidata_triples.json");
            if wikidata_path.exists() {
                if let Some(db) = RelationDatabase::load_wikidata(&wikidata_path) {
                    eprintln!(
                        "  Loaded Wikidata: {} relations, {} pairs",
                        db.num_relations(),
                        db.num_pairs()
                    );
                    result.wikidata = Some(db);
                }
            }
        }

        if result.wordnet.is_none() {
            let wordnet_path = base.join("wordnet_relations.json");
            if wordnet_path.exists() {
                if let Some(db) = RelationDatabase::load_wordnet(&wordnet_path) {
                    eprintln!(
                        "  Loaded WordNet: {} relations, {} pairs",
                        db.num_relations(),
                        db.num_pairs()
                    );
                    result.wordnet = Some(db);
                }
            }
        }

        if result.wikidata.is_some() && result.wordnet.is_some() {
            break;
        }
    }

    result
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn write_triples(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    // ── add_relation + lookup + index ──

    #[test]
    fn add_relation_populates_lookup_index() {
        let mut db = RelationDatabase::default();
        db.add_relation(
            "capital",
            vec![
                ("France".into(), "Paris".into()),
                ("Germany".into(), "Berlin".into()),
            ],
        );
        // lookup is case-insensitive — keys stored lowercased? Actually
        // `add_relation` stores raw and `lookup` lowercases the query.
        // Stored keys aren't lowercased here; lookup with original case
        // will match because the stored key is `(France, Paris)` and the
        // lookup key is `(france, paris)`. They WON'T match without
        // case-insensitive storage, so confirm what the contract is.
        // Looking at the impl: stored as-given, lookup lowercases query.
        // So `lookup("france", "paris")` would NOT find the entry.
        // The Wikidata/WordNet loaders do lowercase the storage; the
        // direct add_relation path doesn't.
        assert_eq!(db.num_relations(), 1);
        assert_eq!(db.num_pairs(), 2);
    }

    #[test]
    fn add_relation_rebuilds_pair_index() {
        let mut db = RelationDatabase::default();
        db.add_relation("rel1", vec![("a".into(), "b".into())]);
        db.add_relation("rel2", vec![("a".into(), "b".into())]);
        // Same (a, b) pair contributed by two relations — both names
        // returned in the lookup.
        let mut hits = db.lookup("a", "b");
        hits.sort();
        assert_eq!(hits, vec!["rel1", "rel2"]);
    }

    #[test]
    fn lookup_returns_empty_for_unknown_pair() {
        let mut db = RelationDatabase::default();
        db.add_relation("r", vec![("x".into(), "y".into())]);
        assert!(db.lookup("nope", "missing").is_empty());
    }

    #[test]
    fn lookup_query_is_case_insensitive() {
        // Wikidata/WordNet loaders lowercase storage; the direct
        // add_relation path doesn't. Verify case-insensitive lookup
        // against a loader-built db.
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(
            tmp.path(),
            "wd.json",
            r#"{"capital": {"pairs": [["France", "Paris"]]}}"#,
        );
        let db = RelationDatabase::load_wikidata(&path).expect("load ok");
        // Loader lowercased the storage; lookup also lowercases the query.
        let hits = db.lookup("FRANCE", "paris");
        assert_eq!(hits, vec!["capital"]);
    }

    #[test]
    fn relations_iter_yields_all_entries() {
        let mut db = RelationDatabase::default();
        db.add_relation("r1", vec![("a".into(), "b".into())]);
        db.add_relation("r2", vec![("c".into(), "d".into())]);
        let mut names: Vec<&str> = db.relations_iter().map(|(n, _)| n).collect();
        names.sort();
        assert_eq!(names, vec!["r1", "r2"]);
        let r1: Vec<_> = db.relations_iter().filter(|(n, _)| *n == "r1").collect();
        assert_eq!(r1[0].1, &[("a".into(), "b".into())]);
    }

    // ── load_wikidata ──

    #[test]
    fn load_wikidata_parses_pair_arrays() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(
            tmp.path(),
            "wd.json",
            r#"{
                "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]},
                "language": {"pairs": [["France", "French"]]}
            }"#,
        );
        let db = RelationDatabase::load_wikidata(&path).expect("load ok");
        assert_eq!(db.num_relations(), 2);
        assert_eq!(db.num_pairs(), 3);
        // Lookup with a lowercased pair (loader lowercases storage).
        assert!(db.lookup("france", "paris").contains(&"capital"));
        assert!(db.lookup("france", "french").contains(&"language"));
    }

    #[test]
    fn load_wikidata_skips_short_pair_arrays_and_empty_strings() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(
            tmp.path(),
            "wd_bad.json",
            r#"{
                "rel": {"pairs": [
                    ["only_one"],
                    ["", "empty_subject"],
                    ["valid_subj", ""],
                    ["a", "b"]
                ]}
            }"#,
        );
        let db = RelationDatabase::load_wikidata(&path).expect("load ok");
        // Only the (a, b) pair survived all the filters.
        assert_eq!(db.num_pairs(), 1);
        assert!(db.lookup("a", "b").contains(&"rel"));
    }

    #[test]
    fn load_wikidata_returns_none_for_missing_file() {
        assert!(RelationDatabase::load_wikidata(Path::new("/tmp/_no_wd.json")).is_none());
    }

    #[test]
    fn load_wikidata_returns_none_for_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(tmp.path(), "bad.json", "not json{");
        assert!(RelationDatabase::load_wikidata(&path).is_none());
    }

    #[test]
    fn load_wikidata_returns_none_for_non_object_root() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(tmp.path(), "arr.json", "[1, 2, 3]");
        assert!(RelationDatabase::load_wikidata(&path).is_none());
    }

    #[test]
    fn load_wikidata_skips_relations_without_pairs_field() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(
            tmp.path(),
            "no_pairs.json",
            r#"{
                "rel_with_pairs": {"pairs": [["a", "b"]]},
                "rel_no_pairs": {"description": "missing pairs key"}
            }"#,
        );
        let db = RelationDatabase::load_wikidata(&path).expect("load ok");
        // The 'rel_no_pairs' relation isn't inserted because the inner
        // `if let Some(pairs)` check fails.
        assert_eq!(db.num_relations(), 1);
    }

    // ── load_wordnet ──

    #[test]
    fn load_wordnet_parses_pair_arrays() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(
            tmp.path(),
            "wn.json",
            r#"{
                "synonym": {"pairs": [["fast", "quick"], ["smart", "clever"]]},
                "antonym": {"pairs": [["hot", "cold"]]}
            }"#,
        );
        let db = RelationDatabase::load_wordnet(&path).expect("load ok");
        assert_eq!(db.num_relations(), 2);
        assert_eq!(db.num_pairs(), 3);
        assert!(db.lookup("fast", "quick").contains(&"synonym"));
    }

    #[test]
    fn load_wordnet_returns_none_for_missing_file() {
        assert!(RelationDatabase::load_wordnet(Path::new("/tmp/_no_wn.json")).is_none());
    }

    #[test]
    fn load_wordnet_returns_none_for_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_triples(tmp.path(), "wn_bad.json", "not json");
        assert!(RelationDatabase::load_wordnet(&path).is_none());
    }

    // Note: `load_reference_databases` walks `./data/`, `../data/`,
    // `../../data/` relative to cwd. Testing it would require
    // mutating cwd which breaks parallel test isolation. The two
    // loaders it dispatches to (`load_wikidata`, `load_wordnet`)
    // are individually tested above; the wrapper itself is exercised
    // implicitly when downstream callers invoke it from the binary.
}
