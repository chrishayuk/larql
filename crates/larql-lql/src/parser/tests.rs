use super::parse;
use crate::ast::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

// ══════════════════════════════════════════════════════════════
// LIFECYCLE STATEMENTS
// ══════════════════════════════════════════════════════════════

// ── EXTRACT ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_minimal() {
    let stmt = parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#).unwrap();
    match stmt {
        Statement::Extract {
            model,
            output,
            components,
            layers,
            extract_level,
        } => {
            assert_eq!(model, "google/gemma-3-4b-it");
            assert_eq!(output, "gemma3-4b.vindex");
            assert!(components.is_none());
            assert!(layers.is_none());
            assert_eq!(extract_level, ExtractLevel::Browse);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_with_components_and_layers() {
    let stmt = parse(
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33;"#,
    )
    .unwrap();
    match stmt {
        Statement::Extract {
            components,
            layers,
            extract_level,
            ..
        } => {
            let c = components.unwrap();
            assert_eq!(c.len(), 4);
            assert_eq!(c[0], Component::FfnGate);
            assert_eq!(c[1], Component::FfnDown);
            assert_eq!(c[2], Component::FfnUp);
            assert_eq!(c[3], Component::Embeddings);
            let l = layers.unwrap();
            assert_eq!(l.start, 0);
            assert_eq!(l.end, 33);
            assert_eq!(extract_level, ExtractLevel::Browse);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_attn_components() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" COMPONENTS ATTN_OV, ATTN_QK;"#).unwrap();
    match stmt {
        Statement::Extract { components, .. } => {
            let c = components.unwrap();
            assert_eq!(c.len(), 2);
            assert_eq!(c[0], Component::AttnOv);
            assert_eq!(c[1], Component::AttnQk);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_with_inference() {
    let stmt =
        parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE;"#)
            .unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::Inference);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_with_all() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" WITH ALL;"#).unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::All);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_with_weights_legacy() {
    // WITH WEIGHTS is legacy syntax, maps to Inference
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" WITH WEIGHTS;"#).unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::Inference);
        }
        _ => panic!("expected Extract"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_extract_with_all_and_components() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" COMPONENTS FFN_GATE WITH ALL;"#).unwrap();
    match stmt {
        Statement::Extract {
            components,
            extract_level,
            ..
        } => {
            assert_eq!(extract_level, ExtractLevel::All);
            assert_eq!(components.unwrap().len(), 1);
        }
        _ => panic!("expected Extract"),
    }
}

// ── COMPILE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_current_safetensors() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;"#).unwrap();
    match stmt {
        Statement::Compile {
            vindex,
            output,
            format,
            ..
        } => {
            assert!(matches!(vindex, VindexRef::Current));
            assert_eq!(output, "edited/");
            assert_eq!(format, Some(OutputFormat::Safetensors));
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_path_gguf() {
    let stmt = parse(r#"COMPILE "gemma3.vindex" INTO MODEL "out/" FORMAT gguf;"#).unwrap();
    match stmt {
        Statement::Compile {
            vindex,
            output,
            format,
            ..
        } => {
            assert!(matches!(vindex, VindexRef::Path(ref p) if p == "gemma3.vindex"));
            assert_eq!(output, "out/");
            assert_eq!(format, Some(OutputFormat::Gguf));
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_no_format() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    match stmt {
        Statement::Compile { format, .. } => assert!(format.is_none()),
        _ => panic!("expected Compile"),
    }
}

// ── DIFF ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_two_paths() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex";"#).unwrap();
    match stmt {
        Statement::Diff { a, b, .. } => {
            assert!(matches!(a, VindexRef::Path(ref p) if p == "a.vindex"));
            assert!(matches!(b, VindexRef::Path(ref p) if p == "b.vindex"));
        }
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_current() {
    let stmt = parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
    match stmt {
        Statement::Diff {
            a: VindexRef::Path(p),
            b: VindexRef::Current,
            ..
        } => assert_eq!(p, "gemma3-4b.vindex"),
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_limit() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LIMIT 20;"#).unwrap();
    match stmt {
        Statement::Diff { limit, .. } => assert_eq!(limit, Some(20)),
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_layer() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LAYER 26;"#).unwrap();
    match stmt {
        Statement::Diff { layer, .. } => assert_eq!(layer, Some(26)),
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_relation_singular() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" RELATION "lives-in";"#).unwrap();
    match stmt {
        Statement::Diff { relation, .. } => assert_eq!(relation.as_deref(), Some("lives-in")),
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_relations_plural() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" RELATIONS "capital-of";"#).unwrap();
    match stmt {
        Statement::Diff { relation, .. } => assert_eq!(relation.as_deref(), Some("capital-of")),
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_with_relation_and_limit() {
    let stmt =
        parse(r#"DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex" RELATION "capital" LIMIT 20;"#)
            .unwrap();
    match stmt {
        Statement::Diff {
            relation, limit, ..
        } => {
            assert_eq!(relation.as_deref(), Some("capital"));
            assert_eq!(limit, Some(20));
        }
        _ => panic!("expected Diff"),
    }
}

// ── USE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_use_vindex() {
    let stmt = parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Vindex(path),
        } => assert_eq!(path, "gemma3-4b.vindex"),
        _ => panic!("expected Use Vindex"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_use_model() {
    let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it";"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Model { id, auto_extract },
        } => {
            assert_eq!(id, "google/gemma-3-4b-it");
            assert!(!auto_extract);
        }
        _ => panic!("expected Use Model"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_use_model_auto_extract() {
    let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Model { auto_extract, .. },
        } => assert!(auto_extract),
        _ => panic!("expected Use Model AUTO_EXTRACT"),
    }
}

// ══════════════════════════════════════════════════════════════
// QUERY STATEMENTS
// ══════════════════════════════════════════════════════════════

// ── WALK ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_minimal() {
    let stmt = parse(r#"WALK "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Walk {
            prompt,
            top,
            layers,
            mode,
            compare,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert!(top.is_none());
            assert!(layers.is_none());
            assert!(mode.is_none());
            assert!(!compare);
        }
        _ => panic!("expected Walk"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_with_top() {
    let stmt = parse(r#"WALK "The capital of France is" TOP 5;"#).unwrap();
    match stmt {
        Statement::Walk { top, .. } => assert_eq!(top, Some(5)),
        _ => panic!("expected Walk"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_full_options() {
    let stmt = parse(r#"WALK "prompt" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#).unwrap();
    match stmt {
        Statement::Walk {
            top,
            layers,
            mode,
            compare,
            ..
        } => {
            assert_eq!(top, Some(5));
            let l = layers.unwrap();
            assert_eq!(l.start, 25);
            assert_eq!(l.end, 33);
            assert_eq!(mode, Some(WalkMode::Hybrid));
            assert!(compare);
        }
        _ => panic!("expected Walk"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_mode_pure() {
    let stmt = parse(r#"WALK "x" MODE pure;"#).unwrap();
    match stmt {
        Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Pure)),
        _ => panic!("expected Walk"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_mode_dense() {
    let stmt = parse(r#"WALK "x" MODE dense;"#).unwrap();
    match stmt {
        Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Dense)),
        _ => panic!("expected Walk"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk_layers_all() {
    let stmt = parse(r#"WALK "x" LAYERS ALL;"#).unwrap();
    match stmt {
        Statement::Walk { layers, .. } => assert!(layers.is_none()),
        _ => panic!("expected Walk"),
    }
}

// ── SELECT ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_star() {
    let stmt = parse("SELECT * FROM EDGES;").unwrap();
    match stmt {
        Statement::Select { fields, .. } => {
            assert_eq!(fields.len(), 1);
            assert!(matches!(fields[0], Field::Star));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_named_fields() {
    let stmt = parse(
        r#"SELECT entity, relation, target, confidence FROM EDGES WHERE entity = "France" ORDER BY confidence DESC LIMIT 10;"#,
    ).unwrap();
    match stmt {
        Statement::Select {
            fields,
            conditions,
            order,
            limit,
            ..
        } => {
            assert_eq!(fields.len(), 4);
            assert_eq!(conditions.len(), 1);
            let ord = order.unwrap();
            assert!(ord.descending);
            assert_eq!(limit, Some(10));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_multiple_conditions() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE relation = "capital-of" AND confidence > 0.5;"#)
        .unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0].op, CompareOp::Eq));
            assert!(matches!(conditions[1].op, CompareOp::Gt));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_by_layer_and_feature() {
    let stmt = parse("SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;").unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0].value, Value::Integer(26)));
            assert!(matches!(conditions[1].value, Value::Integer(9515)));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_nearest() {
    let stmt = parse(
        r#"SELECT entity, target, distance FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
    )
    .unwrap();
    match stmt {
        Statement::Select { nearest, limit, .. } => {
            let n = nearest.unwrap();
            assert_eq!(n.entity, "Mozart");
            assert_eq!(n.layer, 26);
            assert_eq!(limit, Some(20));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_no_where() {
    let stmt = parse("SELECT * FROM EDGES LIMIT 5;").unwrap();
    match stmt {
        Statement::Select {
            conditions, limit, ..
        } => {
            assert!(conditions.is_empty());
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_order_asc() {
    let stmt = parse("SELECT * FROM EDGES ORDER BY layer ASC;").unwrap();
    match stmt {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_order_default_asc() {
    let stmt = parse("SELECT * FROM EDGES ORDER BY layer;").unwrap();
    match stmt {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        _ => panic!("expected Select"),
    }
}

// ── DESCRIBE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_minimal() {
    let stmt = parse(r#"DESCRIBE "France";"#).unwrap();
    match stmt {
        Statement::Describe {
            entity,
            band,
            layer,
            relations_only,
            mode,
        } => {
            assert_eq!(entity, "France");
            assert!(band.is_none());
            assert!(layer.is_none());
            assert!(!relations_only);
            assert_eq!(mode, DescribeMode::Brief); // brief is the default (clean output)
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_at_layer() {
    let stmt = parse(r#"DESCRIBE "Mozart" AT LAYER 26;"#).unwrap();
    match stmt {
        Statement::Describe {
            entity,
            band,
            layer,
            ..
        } => {
            assert_eq!(entity, "Mozart");
            assert!(band.is_none());
            assert_eq!(layer, Some(26));
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe { relations_only, .. } => assert!(relations_only),
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_layer_and_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" AT LAYER 26 RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe {
            layer,
            relations_only,
            ..
        } => {
            assert_eq!(layer, Some(26));
            assert!(relations_only);
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_syntax() {
    let stmt = parse(r#"DESCRIBE "def" SYNTAX;"#).unwrap();
    match stmt {
        Statement::Describe { entity, band, .. } => {
            assert_eq!(entity, "def");
            assert_eq!(band, Some(LayerBand::Syntax));
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_knowledge() {
    let stmt = parse(r#"DESCRIBE "France" KNOWLEDGE;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::Knowledge));
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_output() {
    let stmt = parse(r#"DESCRIBE "France" OUTPUT;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::Output));
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_all_layers() {
    let stmt = parse(r#"DESCRIBE "France" ALL LAYERS;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::All));
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_band_with_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" KNOWLEDGE RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe {
            band,
            relations_only,
            ..
        } => {
            assert_eq!(band, Some(LayerBand::Knowledge));
            assert!(relations_only);
        }
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_verbose() {
    let stmt = parse(r#"DESCRIBE "France" VERBOSE;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Verbose),
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_brief() {
    let stmt = parse(r#"DESCRIBE "France" BRIEF;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Brief),
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_raw() {
    let stmt = parse(r#"DESCRIBE "France" RAW;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Raw),
        _ => panic!("expected Describe"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_describe_band_verbose() {
    let stmt = parse(r#"DESCRIBE "France" ALL LAYERS VERBOSE;"#).unwrap();
    match stmt {
        Statement::Describe { band, mode, .. } => {
            assert_eq!(band, Some(LayerBand::All));
            assert_eq!(mode, DescribeMode::Verbose);
        }
        _ => panic!("expected Describe"),
    }
}

// ── EXPLAIN ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_walk_minimal() {
    let stmt = parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Explain {
            prompt,
            mode,
            layers,
            verbose,
            ..
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(mode, ExplainMode::Walk);
            assert!(layers.is_none());
            assert!(!verbose);
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_walk_with_layers_and_verbose() {
    let stmt = parse(r#"EXPLAIN WALK "prompt" LAYERS 24-33 VERBOSE;"#).unwrap();
    match stmt {
        Statement::Explain {
            layers, verbose, ..
        } => {
            let l = layers.unwrap();
            assert_eq!(l.start, 24);
            assert_eq!(l.end, 33);
            assert!(verbose);
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_minimal() {
    let stmt = parse(r#"EXPLAIN INFER "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Explain {
            prompt,
            mode,
            layers,
            band,
            verbose,
            top,
            relations_only,
            with_attention,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(mode, ExplainMode::Infer);
            assert!(layers.is_none());
            assert!(band.is_none());
            assert!(!verbose);
            assert!(top.is_none());
            assert!(!relations_only);
            assert!(!with_attention);
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_with_options() {
    let stmt = parse(r#"EXPLAIN INFER "test prompt" LAYERS 20-30 VERBOSE TOP 10;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            layers,
            verbose,
            top,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            let l = layers.unwrap();
            assert_eq!(l.start, 20);
            assert_eq!(l.end, 30);
            assert!(verbose);
            assert_eq!(top, Some(10));
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_walk_with_top() {
    let stmt = parse(r#"EXPLAIN WALK "test" TOP 5;"#).unwrap();
    match stmt {
        Statement::Explain { mode, top, .. } => {
            assert_eq!(mode, ExplainMode::Walk);
            assert_eq!(top, Some(5));
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_with_band() {
    let stmt = parse(r#"EXPLAIN INFER "test" KNOWLEDGE;"#).unwrap();
    match stmt {
        Statement::Explain { mode, band, .. } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert_eq!(band, Some(LayerBand::Knowledge));
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_relations_only() {
    let stmt = parse(r#"EXPLAIN INFER "test" RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            relations_only,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert!(relations_only);
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_with_attention() {
    let stmt = parse(r#"EXPLAIN INFER "test" WITH ATTENTION;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            with_attention,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert!(with_attention);
        }
        _ => panic!("expected Explain"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_explain_infer_all_options() {
    let stmt =
        parse(r#"EXPLAIN INFER "test" KNOWLEDGE TOP 1 RELATIONS ONLY WITH ATTENTION;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            band,
            top,
            relations_only,
            with_attention,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert_eq!(band, Some(LayerBand::Knowledge));
            assert_eq!(top, Some(1));
            assert!(relations_only);
            assert!(with_attention);
        }
        _ => panic!("expected Explain"),
    }
}

// ══════════════════════════════════════════════════════════════
// MUTATION STATEMENTS
// ══════════════════════════════════════════════════════════════

// ── INSERT ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_insert_minimal() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            entity,
            relation,
            target,
            layer,
            confidence,
            alpha,
            mode,
        } => {
            assert_eq!(entity, "John Coyle");
            assert_eq!(relation, "lives-in");
            assert_eq!(target, "Colchester");
            assert!(layer.is_none());
            assert!(confidence.is_none());
            assert!(alpha.is_none());
            assert_eq!(mode, InsertMode::Knn);
        }
        _ => panic!("expected Insert"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_insert_with_layer_and_confidence() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "occupation", "engineer") AT LAYER 26 CONFIDENCE 0.8;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            layer,
            confidence,
            alpha,
            ..
        } => {
            assert_eq!(layer, Some(26));
            assert!((confidence.unwrap() - 0.8).abs() < 0.01);
            assert!(alpha.is_none());
        }
        _ => panic!("expected Insert"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_insert_with_alpha() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") ALPHA 0.5;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            alpha,
            layer,
            confidence,
            ..
        } => {
            assert!((alpha.unwrap() - 0.5).abs() < 1e-6);
            assert!(layer.is_none());
            assert!(confidence.is_none());
        }
        _ => panic!("expected Insert"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_insert_with_layer_confidence_alpha() {
    // All three optional clauses can coexist in any order encountered.
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") AT LAYER 24 CONFIDENCE 0.95 ALPHA 0.3;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            layer,
            confidence,
            alpha,
            ..
        } => {
            assert_eq!(layer, Some(24));
            assert!((confidence.unwrap() - 0.95).abs() < 1e-6);
            assert!((alpha.unwrap() - 0.3).abs() < 1e-6);
        }
        _ => panic!("expected Insert"),
    }
}

// ── DELETE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_delete_single_condition() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "outdated_fact";"#).unwrap();
    match stmt {
        Statement::Delete { conditions } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(conditions[0].field, "entity");
        }
        _ => panic!("expected Delete"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_delete_multiple_conditions() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";"#)
        .unwrap();
    match stmt {
        Statement::Delete { conditions } => assert_eq!(conditions.len(), 2),
        _ => panic!("expected Delete"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_delete_by_layer() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "outdated" AND layer = 26;"#).unwrap();
    match stmt {
        Statement::Delete { conditions } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[1].value, Value::Integer(26)));
        }
        _ => panic!("expected Delete"),
    }
}

// ── UPDATE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_update_single_set() {
    let stmt = parse(
        r#"UPDATE EDGES SET target = "London" WHERE entity = "John Coyle" AND relation = "lives-in";"#,
    ).unwrap();
    match stmt {
        Statement::Update { set, conditions } => {
            assert_eq!(set.len(), 1);
            assert_eq!(set[0].field, "target");
            assert_eq!(conditions.len(), 2);
        }
        _ => panic!("expected Update"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_update_multiple_assignments() {
    let stmt = parse(
        r#"UPDATE EDGES SET target = "London", confidence = 0.9 WHERE entity = "John Coyle";"#,
    )
    .unwrap();
    match stmt {
        Statement::Update { set, conditions } => {
            assert_eq!(set.len(), 2);
            assert_eq!(conditions.len(), 1);
        }
        _ => panic!("expected Update"),
    }
}

// ── MERGE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_merge_minimal() {
    let stmt = parse(r#"MERGE "source.vindex";"#).unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "source.vindex");
            assert!(target.is_none());
            assert!(conflict.is_none());
        }
        _ => panic!("expected Merge"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_merge_into_no_conflict() {
    let stmt = parse(r#"MERGE "source.vindex" INTO "target.vindex";"#).unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "source.vindex");
            assert_eq!(target.as_deref(), Some("target.vindex"));
            assert!(conflict.is_none());
        }
        _ => panic!("expected Merge"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_merge_into_with_conflict() {
    let stmt =
        parse(r#"MERGE "medical.vindex" INTO "gemma3.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#)
            .unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "medical.vindex");
            assert_eq!(target.as_deref(), Some("gemma3.vindex"));
            assert_eq!(conflict, Some(ConflictStrategy::HighestConfidence));
        }
        _ => panic!("expected Merge"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_merge_keep_source() {
    let stmt = parse(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_SOURCE;"#).unwrap();
    match stmt {
        Statement::Merge { conflict, .. } => {
            assert_eq!(conflict, Some(ConflictStrategy::KeepSource))
        }
        _ => panic!("expected Merge"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_merge_keep_target() {
    let stmt = parse(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_TARGET;"#).unwrap();
    match stmt {
        Statement::Merge { conflict, .. } => {
            assert_eq!(conflict, Some(ConflictStrategy::KeepTarget))
        }
        _ => panic!("expected Merge"),
    }
}

// ══════════════════════════════════════════════════════════════
// INTROSPECTION STATEMENTS
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_minimal() {
    let stmt = parse("SHOW RELATIONS;").unwrap();
    match stmt {
        Statement::ShowRelations {
            layer,
            with_examples,
            mode,
        } => {
            assert!(layer.is_none());
            assert!(!with_examples);
            assert_eq!(mode, DescribeMode::Brief); // Brief is the default
        }
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_with_examples() {
    let stmt = parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    match stmt {
        Statement::ShowRelations { with_examples, .. } => assert!(with_examples),
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_at_layer() {
    let stmt = parse("SHOW RELATIONS AT LAYER 26;").unwrap();
    match stmt {
        Statement::ShowRelations { layer, .. } => assert_eq!(layer, Some(26)),
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_verbose() {
    let stmt = parse("SHOW RELATIONS VERBOSE;").unwrap();
    match stmt {
        Statement::ShowRelations { mode, .. } => assert_eq!(mode, DescribeMode::Verbose),
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_raw() {
    let stmt = parse("SHOW RELATIONS RAW;").unwrap();
    match stmt {
        Statement::ShowRelations { mode, .. } => assert_eq!(mode, DescribeMode::Raw),
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_relations_verbose_with_examples() {
    let stmt = parse("SHOW RELATIONS VERBOSE WITH EXAMPLES;").unwrap();
    match stmt {
        Statement::ShowRelations {
            mode,
            with_examples,
            ..
        } => {
            assert_eq!(mode, DescribeMode::Verbose);
            assert!(with_examples);
        }
        _ => panic!("expected ShowRelations"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_layers_minimal() {
    let stmt = parse("SHOW LAYERS;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => assert!(range.is_none()),
        _ => panic!("expected ShowLayers"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_layers_with_range() {
    let stmt = parse("SHOW LAYERS RANGE 0-10;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 10);
        }
        _ => panic!("expected ShowLayers"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_layers_bare_range() {
    let stmt = parse("SHOW LAYERS 0-10;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 10);
        }
        _ => panic!("expected ShowLayers"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_features_minimal() {
    let stmt = parse("SHOW FEATURES 26;").unwrap();
    match stmt {
        Statement::ShowFeatures {
            layer,
            conditions,
            limit,
        } => {
            assert_eq!(layer, 26);
            assert!(conditions.is_empty());
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowFeatures"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_features_with_where_and_limit() {
    let stmt = parse(r#"SHOW FEATURES 26 WHERE relation = "capital-of" LIMIT 5;"#).unwrap();
    match stmt {
        Statement::ShowFeatures {
            layer,
            conditions,
            limit,
        } => {
            assert_eq!(layer, 26);
            assert_eq!(conditions.len(), 1);
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected ShowFeatures"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_models() {
    let stmt = parse("SHOW MODELS;").unwrap();
    assert!(matches!(stmt, Statement::ShowModels));
}

// ── SHOW ENTITIES ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_entities_minimal() {
    let stmt = parse("SHOW ENTITIES;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert!(layer.is_none());
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_entities_bare_layer() {
    let stmt = parse("SHOW ENTITIES 26;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert_eq!(layer, Some(26));
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_entities_at_layer_with_limit() {
    let stmt = parse("SHOW ENTITIES AT LAYER 26 LIMIT 50;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert_eq!(layer, Some(26));
            assert_eq!(limit, Some(50));
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_entities_limit_only() {
    let stmt = parse("SHOW ENTITIES LIMIT 100;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert!(layer.is_none());
            assert_eq!(limit, Some(100));
        }
        _ => panic!("expected ShowEntities"),
    }
}

// ── REBALANCE ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_rebalance_minimal() {
    let stmt = parse("REBALANCE;").unwrap();
    match stmt {
        Statement::Rebalance {
            max_iters,
            floor,
            ceiling,
        } => {
            assert!(max_iters.is_none());
            assert!(floor.is_none());
            assert!(ceiling.is_none());
        }
        _ => panic!("expected Rebalance"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_rebalance_until_converged() {
    let stmt = parse("REBALANCE UNTIL CONVERGED;").unwrap();
    assert!(matches!(stmt, Statement::Rebalance { .. }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_rebalance_max_iters() {
    let stmt = parse("REBALANCE MAX 32;").unwrap();
    match stmt {
        Statement::Rebalance { max_iters, .. } => assert_eq!(max_iters, Some(32)),
        _ => panic!("expected Rebalance"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_rebalance_floor_ceiling() {
    let stmt = parse("REBALANCE FLOOR 0.3 CEILING 0.9;").unwrap();
    match stmt {
        Statement::Rebalance { floor, ceiling, .. } => {
            assert!((floor.unwrap() - 0.3).abs() < 1e-6);
            assert!((ceiling.unwrap() - 0.9).abs() < 1e-6);
        }
        _ => panic!("expected Rebalance"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_rebalance_all_clauses() {
    let stmt = parse("REBALANCE UNTIL CONVERGED MAX 16 FLOOR = 0.25 CEILING = 0.95;").unwrap();
    match stmt {
        Statement::Rebalance {
            max_iters,
            floor,
            ceiling,
        } => {
            assert_eq!(max_iters, Some(16));
            assert!((floor.unwrap() - 0.25).abs() < 1e-6);
            assert!((ceiling.unwrap() - 0.95).abs() < 1e-6);
        }
        _ => panic!("expected Rebalance"),
    }
}

// ── SHOW COMPACT STATUS ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_compact_status() {
    let stmt = parse("SHOW COMPACT STATUS;").unwrap();
    assert!(matches!(stmt, Statement::ShowCompactStatus));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_compact_status_no_semicolon() {
    let stmt = parse("SHOW COMPACT STATUS").unwrap();
    assert!(matches!(stmt, Statement::ShowCompactStatus));
}

// ── COMPACT ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compact_minor() {
    let stmt = parse("COMPACT MINOR;").unwrap();
    assert!(matches!(stmt, Statement::CompactMinor));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compact_major() {
    let stmt = parse("COMPACT MAJOR;").unwrap();
    assert!(matches!(
        stmt,
        Statement::CompactMajor {
            full: false,
            lambda: None
        }
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compact_major_full() {
    let stmt = parse("COMPACT MAJOR FULL;").unwrap();
    assert!(matches!(
        stmt,
        Statement::CompactMajor {
            full: true,
            lambda: None
        }
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compact_major_with_lambda() {
    let stmt = parse("COMPACT MAJOR WITH LAMBDA = 0.001;").unwrap();
    match stmt {
        Statement::CompactMajor { full, lambda } => {
            assert!(!full);
            assert!((lambda.unwrap() - 0.001).abs() < 1e-6);
        }
        _ => panic!("expected CompactMajor"),
    }
}

// ── STATS ──

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_stats_no_path() {
    let stmt = parse("STATS;").unwrap();
    assert!(matches!(stmt, Statement::Stats { vindex: None }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_stats_with_path() {
    let stmt = parse(r#"STATS "gemma3.vindex";"#).unwrap();
    match stmt {
        Statement::Stats { vindex } => assert_eq!(vindex.as_deref(), Some("gemma3.vindex")),
        _ => panic!("expected Stats"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_stats_no_semicolon() {
    let stmt = parse("STATS").unwrap();
    assert!(matches!(stmt, Statement::Stats { vindex: None }));
}

// ══════════════════════════════════════════════════════════════
// PIPE OPERATOR
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_pipe_walk_to_explain() {
    let stmt = parse(
        r#"WALK "The capital of France is" TOP 5 |> EXPLAIN WALK "The capital of France is";"#,
    )
    .unwrap();
    match stmt {
        Statement::Pipe { left, right } => {
            assert!(matches!(*left, Statement::Walk { .. }));
            assert!(matches!(*right, Statement::Explain { .. }));
        }
        _ => panic!("expected Pipe"),
    }
}

// ══════════════════════════════════════════════════════════════
// COMPARISON OPERATORS
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_neq() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE relation != "morphological";"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => assert!(matches!(conditions[0].op, CompareOp::Neq)),
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_gte_lte() {
    let stmt = parse("SELECT * FROM EDGES WHERE layer >= 20 AND layer <= 30;").unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::Gte));
            assert!(matches!(conditions[1].op, CompareOp::Lte));
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_like() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE entity LIKE "Fran%";"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::Like))
        }
        _ => panic!("expected Select"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_select_in() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE entity IN ("France", "Germany");"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::In));
            if let Value::List(items) = &conditions[0].value {
                assert_eq!(items.len(), 2);
            } else {
                panic!("expected list value");
            }
        }
        _ => panic!("expected Select"),
    }
}

// ══════════════════════════════════════════════════════════════
// COMMENTS AND WHITESPACE
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_with_leading_comment() {
    let stmt = parse("-- This is a comment\nSTATS;").unwrap();
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_with_trailing_comment() {
    let stmt = parse("STATS; -- trailing comment").unwrap();
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_multiline_statement() {
    let stmt = parse("SELECT *\n  FROM EDGES\n  WHERE layer = 26\n  LIMIT 5;").unwrap();
    match stmt {
        Statement::Select {
            conditions, limit, ..
        } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected Select"),
    }
}

// ══════════════════════════════════════════════════════════════
// ERROR CASES
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_unknown_statement() {
    assert!(parse("FOOBAR;").is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_walk_missing_prompt() {
    assert!(parse("WALK TOP 5;").is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_select_missing_from() {
    assert!(parse(r#"SELECT * WHERE entity = "x";"#).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_insert_missing_values() {
    assert!(parse("INSERT INTO EDGES (entity, relation, target);").is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_show_invalid_noun() {
    assert!(parse("SHOW FOOBAR;").is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_empty_input() {
    assert!(parse("").is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_error_comment_only() {
    assert!(parse("-- just a comment").is_err());
}

// ══════════════════════════════════════════════════════════════
// FULL DEMO SCRIPT FROM SPEC v0.3 — every statement parses
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_demo_script_act1() {
    parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;"#).unwrap();
    parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
    parse("STATS;").unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_demo_script_act2() {
    parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    parse(r#"DESCRIBE "France";"#).unwrap();
    parse(r#"DESCRIBE "Einstein";"#).unwrap();
    parse(r#"DESCRIBE "def" SYNTAX;"#).unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_demo_script_act3() {
    parse(r#"WALK "France" TOP 10;"#).unwrap();
    parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
    parse(r#"INFER "The capital of France is" TOP 5 COMPARE;"#).unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_demo_script_act4() {
    parse(r#"DESCRIBE "John Coyle";"#).unwrap();
    parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    ).unwrap();
    parse(r#"DESCRIBE "John Coyle";"#).unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_demo_script_act5() {
    parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
    parse(r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#).unwrap();
}

// ══════════════════════════════════════════════════════════════
// INFER STATEMENT
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_infer_minimal() {
    let stmt = parse(r#"INFER "The capital of France is" TOP 5;"#).unwrap();
    match stmt {
        Statement::Infer {
            prompt,
            top,
            compare,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(top, Some(5));
            assert!(!compare);
        }
        _ => panic!("expected Infer"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_infer_with_compare() {
    let stmt = parse(r#"INFER "test prompt" TOP 3 COMPARE;"#).unwrap();
    match stmt {
        Statement::Infer {
            prompt,
            top,
            compare,
        } => {
            assert_eq!(prompt, "test prompt");
            assert_eq!(top, Some(3));
            assert!(compare);
        }
        _ => panic!("expected Infer"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_infer_no_top() {
    let stmt = parse(r#"INFER "test";"#).unwrap();
    match stmt {
        Statement::Infer { top, .. } => assert!(top.is_none()),
        _ => panic!("expected Infer"),
    }
}

// ══════════════════════════════════════════════════════════════
// PATCH STATEMENTS
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_begin_patch() {
    let stmt = parse(r#"BEGIN PATCH "medical-knowledge.vlp";"#).unwrap();
    match stmt {
        Statement::BeginPatch { path } => assert_eq!(path, "medical-knowledge.vlp"),
        _ => panic!("expected BeginPatch"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_save_patch() {
    let stmt = parse("SAVE PATCH;").unwrap();
    assert!(matches!(stmt, Statement::SavePatch));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_apply_patch() {
    let stmt = parse(r#"APPLY PATCH "medical-knowledge.vlp";"#).unwrap();
    match stmt {
        Statement::ApplyPatch { path } => assert_eq!(path, "medical-knowledge.vlp"),
        _ => panic!("expected ApplyPatch"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_show_patches() {
    let stmt = parse("SHOW PATCHES;").unwrap();
    assert!(matches!(stmt, Statement::ShowPatches));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_remove_patch() {
    let stmt = parse(r#"REMOVE PATCH "fix-hallucinations.vlp";"#).unwrap();
    match stmt {
        Statement::RemovePatch { path } => assert_eq!(path, "fix-hallucinations.vlp"),
        _ => panic!("expected RemovePatch"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_patch_workflow() {
    // Full patch workflow from spec
    parse(r#"BEGIN PATCH "medical-knowledge.vlp";"#).unwrap();
    parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "side_effect", "bleeding");"#).unwrap();
    parse("SAVE PATCH;").unwrap();
    parse(r#"APPLY PATCH "medical-knowledge.vlp";"#).unwrap();
    parse("SHOW PATCHES;").unwrap();
    parse(r#"REMOVE PATCH "medical-knowledge.vlp";"#).unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_into_patch() {
    let stmt = parse(
        r#"DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex" INTO PATCH "medical-changes.vlp";"#,
    )
    .unwrap();
    match stmt {
        Statement::Diff {
            a, b, into_patch, ..
        } => {
            assert!(matches!(a, VindexRef::Path(ref p) if p == "gemma3-4b.vindex"));
            assert!(matches!(b, VindexRef::Path(ref p) if p == "gemma3-4b-medical.vindex"));
            assert_eq!(into_patch.as_deref(), Some("medical-changes.vlp"));
        }
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_diff_without_into_patch() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LIMIT 10;"#).unwrap();
    match stmt {
        Statement::Diff {
            into_patch, limit, ..
        } => {
            assert!(into_patch.is_none());
            assert_eq!(limit, Some(10));
        }
        _ => panic!("expected Diff"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_vindex() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "output.vindex";"#).unwrap();
    match stmt {
        Statement::Compile { target, output, .. } => {
            assert_eq!(target, CompileTarget::Vindex);
            assert_eq!(output, "output.vindex");
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_vindex_on_conflict_last_wins() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT LAST_WINS;"#).unwrap();
    match stmt {
        Statement::Compile {
            target,
            on_conflict,
            ..
        } => {
            assert_eq!(target, CompileTarget::Vindex);
            assert_eq!(on_conflict, Some(CompileConflict::LastWins));
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_vindex_on_conflict_highest_confidence() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#)
        .unwrap();
    match stmt {
        Statement::Compile { on_conflict, .. } => {
            assert_eq!(on_conflict, Some(CompileConflict::HighestConfidence));
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_vindex_on_conflict_fail() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT FAIL;"#).unwrap();
    match stmt {
        Statement::Compile { on_conflict, .. } => {
            assert_eq!(on_conflict, Some(CompileConflict::Fail));
        }
        _ => panic!("expected Compile"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_model_with_on_conflict_errors() {
    let result = parse(r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors ON CONFLICT FAIL;"#);
    assert!(
        result.is_err(),
        "ON CONFLICT must reject COMPILE INTO MODEL"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_compile_into_model_explicit() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#).unwrap();
    match stmt {
        Statement::Compile { target, .. } => {
            assert_eq!(target, CompileTarget::Model);
        }
        _ => panic!("expected Compile"),
    }
}

// ══════════════════════════════════════════════════════════════
// TRACE STATEMENTS
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_minimal() {
    let stmt = parse(r#"TRACE "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Trace {
            prompt,
            answer,
            decompose,
            layers,
            positions,
            save,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert!(answer.is_none());
            assert!(!decompose);
            assert!(layers.is_none());
            assert!(positions.is_none());
            assert!(save.is_none());
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_with_for_token() {
    let stmt = parse(r#"TRACE "The capital of France is" FOR "Paris";"#).unwrap();
    match stmt {
        Statement::Trace { prompt, answer, .. } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(answer.unwrap(), "Paris");
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_decompose_with_layers() {
    let stmt = parse(r#"TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;"#).unwrap();
    match stmt {
        Statement::Trace {
            decompose, layers, ..
        } => {
            assert!(decompose);
            let r = layers.unwrap();
            assert_eq!(r.start, 22);
            assert_eq!(r.end, 27);
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_save() {
    let stmt = parse(r#"TRACE "The capital of France is" SAVE "france.trace";"#).unwrap();
    match stmt {
        Statement::Trace { save, .. } => {
            assert_eq!(save.unwrap(), "france.trace");
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_positions_all() {
    let stmt = parse(r#"TRACE "The capital of France is" POSITIONS ALL;"#).unwrap();
    match stmt {
        Statement::Trace { positions, .. } => {
            assert_eq!(positions.unwrap(), TracePositionMode::All);
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_positions_last() {
    let stmt = parse(r#"TRACE "The capital of France is" POSITIONS LAST;"#).unwrap();
    match stmt {
        Statement::Trace { positions, .. } => {
            assert_eq!(positions.unwrap(), TracePositionMode::Last);
        }
        _ => panic!("expected Trace"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_trace_full() {
    let stmt = parse(
        r#"TRACE "The capital of France is" FOR "Paris" DECOMPOSE LAYERS 22-27 SAVE "out.trace";"#,
    )
    .unwrap();
    match stmt {
        Statement::Trace {
            prompt,
            answer,
            decompose,
            layers,
            save,
            ..
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(answer.unwrap(), "Paris");
            assert!(decompose);
            assert_eq!(layers.as_ref().unwrap().start, 22);
            assert_eq!(save.unwrap(), "out.trace");
        }
        _ => panic!("expected Trace"),
    }
}

// ══════════════════════════════════════════════════════════════
// Range validation
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn range_invalid_start_greater_than_end() {
    let result = parse("SHOW LAYERS 10-5;");
    assert!(result.is_err(), "range 10-5 should fail");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn range_valid_same_start_end() {
    let stmt = parse("SHOW LAYERS 5-5;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 5);
            assert_eq!(r.end, 5);
        }
        _ => panic!("expected ShowLayers"),
    }
}

// ══════════════════════════════════════════════════════════════
// Keyword field name mapping
// ══════════════════════════════════════════════════════════════

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn keyword_field_names_consistent() {
    use crate::lexer::Keyword;
    // Key field-name keywords that must map correctly
    assert_eq!(Keyword::Layer.as_field_name(), "layer");
    assert_eq!(Keyword::Confidence.as_field_name(), "confidence");
    assert_eq!(Keyword::Relation.as_field_name(), "relation");
    assert_eq!(Keyword::FfnGate.as_field_name(), "ffn_gate");
    assert_eq!(Keyword::FfnDown.as_field_name(), "ffn_down");
    assert_eq!(Keyword::AttnOv.as_field_name(), "attn_ov");
    assert_eq!(Keyword::AutoExtract.as_field_name(), "auto_extract");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parser_rejects_trailing_tokens_after_semicolon() {
    let result = parse(r#"STATS; SELECT * FROM EDGES;"#);
    assert!(result.is_err(), "single-statement parser must reject tails");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parser_rejects_trailing_identifier_without_semicolon() {
    let result = parse(r#"STATS unexpected"#);
    assert!(result.is_err(), "single-statement parser must consume EOF");
}
