//! ADR-0027 closure criterion 1: the legacy bank shape has exactly one
//! production producer.
//!
//! The graph container is the sole normative VINDEX3 3.0 shape. The bank
//! shape (`moe_manifest.json` + `.lyrw` segments, no system graph) is
//! written by two entry points — `write_container` and `ContainerBuilder`
//! — and survives in production only as the legacy LYRW facility behind
//! `extract-index --expert-banks native|auto --expert-banks-out`, which
//! feeds `run`/`bench --routed-from`. This scan pins every production
//! reference to either entry point to its exact owner, so a new
//! bank-shape producer is a failing test, not a review finding. Test,
//! bench and example code is out of scope: fixtures and kernel bring-up
//! are the bank writer's sanctioned remaining uses.
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

/// The bank-shape writer entry points. `write_container` is matched as
/// the last segment of any path; `ContainerBuilder` wherever it appears in
/// an expression path, so `ContainerBuilder::create` and a bare function
/// reference to it are both caught.
const WRITE_CONTAINER: &str = "write_container";
const CONTAINER_BUILDER: &str = "ContainerBuilder";

/// The one sanctioned production producer: the legacy LYRW facility.
const LEGACY_FACILITY_FILE: &str = "larql-vindex/src/extract/orchestrate.rs";
const LEGACY_FACILITY_OWNER: &str = "write_native_container";

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
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        let segments = &path.path.segments;
        if segments.last().unwrap().ident == WRITE_CONTAINER {
            self.found(WRITE_CONTAINER);
        } else if segments.iter().any(|s| s.ident == CONTAINER_BUILDER) {
            self.found(CONTAINER_BUILDER);
        }
        visit::visit_expr_path(self, path);
    }
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if let Some(call) = macro_names_a_writer(mac.tokens.clone()) {
            self.found(&format!("{call}-in-macro"));
        }
    }
}

fn macro_names_a_writer(stream: proc_macro2::TokenStream) -> Option<&'static str> {
    use proc_macro2::TokenTree;
    for token in stream {
        match token {
            TokenTree::Group(g) => {
                if let Some(call) = macro_names_a_writer(g.stream()) {
                    return Some(call);
                }
            }
            TokenTree::Ident(id) if id == WRITE_CONTAINER => return Some(WRITE_CONTAINER),
            TokenTree::Ident(id) if id == CONTAINER_BUILDER => return Some(CONTAINER_BUILDER),
            _ => {}
        }
    }
    None
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

fn site(file: &str, owner: &str, call: &str) -> Site {
    Site {
        file: file.into(),
        owner: owner.into(),
        call: call.into(),
    }
}

#[test]
fn the_legacy_lyrw_facility_is_the_only_production_bank_shape_producer() {
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
    let expected = vec![site(
        LEGACY_FACILITY_FILE,
        LEGACY_FACILITY_OWNER,
        CONTAINER_BUILDER,
    )];
    assert_eq!(
        observed, expected,
        "a production path reaches the bank-shape writer. ADR-0027: the graph \
         container is the sole normative VINDEX3 3.0 shape — produce it through \
         the graph encoder, or, if this is deliberate legacy LYRW work, amend \
         the ADR and this pin together"
    );
}

#[test]
fn scanner_detects_calls_references_aliases_and_macros_but_excludes_test_only_code() {
    let found = sites(
        "stray.rs",
        r#"
        fn direct(root: &Path, spec: &ContainerSpec) { write_container(root, spec).unwrap(); }
        fn qualified(root: &Path, spec: &ContainerSpec) { vindex3::write_container(root, spec).unwrap(); }
        fn builder(dest: &Path) { let b = ContainerBuilder::create(dest); }
        fn reference() { let make = ContainerBuilder::create; }
        fn wrapped(root: &Path, spec: &ContainerSpec) { assert!(write_container(root, spec).is_ok()); }
        #[cfg(test)] mod tests { fn fixture(){ write_container(d, s); } }
        #[cfg(any(test, feature="test-utils"))] fn fixture(){ ContainerBuilder::create(d); }
        fn graph(root: &Path) { encode_checkpoint(root); }
    "#,
    );
    assert_eq!(
        found,
        vec![
            site("stray.rs", "direct", WRITE_CONTAINER),
            site("stray.rs", "qualified", WRITE_CONTAINER),
            site("stray.rs", "builder", CONTAINER_BUILDER),
            site("stray.rs", "reference", CONTAINER_BUILDER),
            site("stray.rs", "wrapped", "write_container-in-macro"),
        ]
    );
}
