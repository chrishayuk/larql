//! Whole-estate write closure, with a non-vacuous positive control.
//! Rust visibility and the non-deserializable AcceptedMeasurement token prevent
//! raw writes; this scan also pins every recording call to its exact owner.
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Site {
    file: String,
    owner: String,
    call: String,
}

fn production_sources(root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("source directory must be readable") {
            let path = entry.expect("source entry").path();
            let name = path.file_name().unwrap().to_string_lossy();
            if path.is_dir() {
                if !matches!(
                    name.as_ref(),
                    "target" | "tests" | "benches" | "examples" | ".git"
                ) {
                    dirs.push(path);
                }
            } else if path.extension().is_some_and(|e| e == "rs")
                && !name.ends_with("_tests.rs")
                && name != "tests.rs"
                && path.components().any(|c| c.as_os_str() == "src")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("test")
            || (attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Meta>()
                    .is_ok_and(|m| !can_be_production(&m)))
    })
}

// Conservatively retain unknown platform/feature predicates. Only test and
// test-utils are known false in the production configuration being scanned.
fn can_be_production(meta: &syn::Meta) -> bool {
    use syn::punctuated::Punctuated;
    match meta {
        syn::Meta::Path(p) => !p.is_ident("test"),
        syn::Meta::NameValue(n) if n.path.is_ident("feature") => {
            !matches!(&n.value, syn::Expr::Lit(l) if matches!(&l.lit, syn::Lit::Str(s) if s.value() == "test-utils"))
        }
        syn::Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            let items = list
                .parse_args_with(Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
                .expect("cfg syntax");
            if list.path.is_ident("all") {
                items.iter().all(can_be_production)
            } else {
                items.iter().any(can_be_production)
            }
        }
        _ => true,
    }
}

struct Scanner {
    file: String,
    owners: Vec<String>,
    sites: Vec<Site>,
}
impl Scanner {
    fn found(&mut self, call: &str) {
        self.sites.push(Site {
            file: self.file.clone(),
            owner: self.owners.join("::"),
            call: call.into(),
        });
    }
}
impl<'ast> Visit<'ast> for Scanner {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if let syn::Item::Mod(m) = item {
            if test_only(&m.attrs) {
                return;
            }
            self.owners.push(m.ident.to_string());
            visit::visit_item(self, item);
            self.owners.pop();
        } else {
            visit::visit_item(self, item);
        }
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if test_only(&item.attrs) {
            return;
        }
        self.owners.push(item.sig.ident.to_string());
        visit::visit_item_fn(self, item);
        self.owners.pop();
    }
    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        if test_only(&item.attrs) {
            return;
        }
        let owner = match item.self_ty.as_ref() {
            syn::Type::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
            _ => "impl".into(),
        };
        self.owners.push(owner);
        visit::visit_item_impl(self, item);
        self.owners.pop();
    }
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if test_only(&item.attrs) {
            return;
        }
        self.owners.push(item.sig.ident.to_string());
        visit::visit_impl_item_fn(self, item);
        self.owners.pop();
    }
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "record" || call.method == "record_validated" {
            self.found(&call.method.to_string());
        }
        visit::visit_expr_method_call(self, call);
    }
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        let name = path.path.segments.last().unwrap().ident.to_string();
        if path.path.segments.len() > 1 && (name == "record" || name == "record_validated") {
            self.found(&name);
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if macro_records(mac.tokens.clone()) {
            self.found("record-in-macro");
        }
    }
}

fn macro_records(stream: proc_macro2::TokenStream) -> bool {
    use proc_macro2::{Delimiter, TokenTree};
    let tokens: Vec<_> = stream.into_iter().collect();
    let punct = |at: usize, value| matches!(tokens.get(at), Some(TokenTree::Punct(p)) if p.as_char() == value);
    for (i, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Group(g) if macro_records(g.stream()) => return true,
            TokenTree::Ident(id) if id == "record" || id == "record_validated" => {
                let associated = i >= 2 && punct(i - 1, ':') && punct(i - 2, ':');
                let method = i >= 1
                    && punct(i - 1, '.')
                    && (matches!(tokens.get(i+1), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
                        || punct(i + 1, ':'));
                if associated || method {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn sites(file: &str, text: &str) -> Vec<Site> {
    let parsed = syn::parse_file(text).unwrap_or_else(|e| panic!("{file}: {e}"));
    let mut scanner = Scanner {
        file: file.into(),
        owners: Vec::new(),
        sites: Vec::new(),
    };
    scanner.visit_file(&parsed);
    scanner.sites
}

#[test]
fn every_production_recording_call_has_an_exact_registered_owner() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let sources = production_sources(root);
    assert!(
        sources.len() > 1000,
        "source scan is unexpectedly empty/narrow"
    );
    let mut observed = Vec::new();
    for path in sources {
        let name = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        observed.extend(sites(
            &name,
            &std::fs::read_to_string(path).expect("source readable"),
        ));
    }
    observed.sort();
    let expected: Vec<Site> = serde_json::from_str::<Vec<(String, String, String)>>(include_str!(
        "ingestion_record_sites.json"
    ))
    .unwrap()
    .into_iter()
    .map(|(file, owner, call)| Site { file, owner, call })
    .collect();
    let mut expected = expected;
    expected.sort();
    assert_eq!(
        observed, expected,
        "recording call moved or a new route appeared; review its exact owner"
    );
    assert!(
        observed.iter().any(
            |s| s.file == "larql-vindex/src/format/vindex3/represent/ingest.rs"
                && s.owner == "ingest"
                && s.call == "record"
        ),
        "positive control: actual ingestion write must be found"
    );
}

#[test]
fn scanner_detects_new_owners_aliases_and_macro_calls_but_excludes_test_only_code() {
    let found = sites(
        "stray.rs",
        r#"
        fn stray(registry: &mut Registry, accepted: &Accepted) { registry.record(accepted); }
        fn alias(registry: &mut Registry, accepted: &Accepted) { R::record(registry, accepted); }
        fn reference(){ let writer = Registry::record; }
        fn wrapped(registry: &mut Registry, accepted: &Accepted) { assert!(registry.record(accepted).is_ok()); }
        #[cfg(test)] mod tests {fn fixture(){ r.record(k, v); }}
        #[cfg(any(test, feature="test-utils"))] fn fixture(){ r.record(k,v); }
    "#,
    );
    assert_eq!(
        found.iter().map(|s| s.owner.as_str()).collect::<Vec<_>>(),
        ["stray", "alias", "reference", "wrapped"]
    );
    assert_eq!(
        found.iter().map(|s| s.call.as_str()).collect::<Vec<_>>(),
        ["record", "record", "record", "record-in-macro"]
    );
}
