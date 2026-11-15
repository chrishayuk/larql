//! The help topics: one per statement kind, the concept topics, and the
//! groups the overview lists them under.
//!
//! Each enum is declared once, through [`topic_enum`], which also writes its
//! `ALL` list. A variant therefore cannot exist without being listed, and
//! the catalogue's matches over these enums have no wildcard arm, so a
//! variant cannot exist without an entry either.

/// Declare a fieldless enum together with `ALL`, every variant in
/// declaration order.
macro_rules! topic_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];
        }
    };
}

topic_enum! {
    /// The statement groups of the overview, in the order an editing loop
    /// uses them.
    HelpGroup {
        Bind,
        Inspect,
        Run,
        Weights,
        Patches,
        Compile,
        Verify,
        Abandon,
        Knowledge,
        Process,
    }
}

topic_enum! {
    /// One topic per [`crate::ast::Statement`] variant; the mapping is
    /// [`crate::ast::Statement::help_topic`]. Declared in overview order.
    StatementTopic {
        // Bind
        Use,
        Extract,
        // Inspect
        Stats,
        ShowComponents,
        ShowRepresentations,
        ShowTensors,
        ExportWeight,
        ShowProvenance,
        ShowAuthority,
        ShowLayers,
        ShowModels,
        ShowRelations,
        ShowFeatures,
        ShowEntities,
        Describe,
        Select,
        Diff,
        // Run
        Infer,
        Walk,
        Explain,
        Trace,
        Capture,
        // Edit weights
        UpdateWeights,
        RemoveDirection,
        AddLowRank,
        // Patches
        BeginPatch,
        SavePatch,
        ApplyPatch,
        ShowPatches,
        // Compile
        Compile,
        CompactInto,
        // Verify
        VerifyPatch,
        VerifyLedger,
        VerifyRepresentation,
        VerifyOperator,
        // Abandon
        DiscardWeightEdits,
        RemovePatch,
        // Knowledge edits
        Insert,
        Delete,
        Update,
        Merge,
        Rebalance,
        CompactMinor,
        CompactMajor,
        ShowCompactStatus,
        // Process model
        Help,
        Pipe,
    }
}

topic_enum! {
    /// Topics that describe how `larql lql` runs statements; none is a
    /// statement.
    ConceptTopic {
        Batch,
        ExitCodes,
        Verifier,
    }
}

/// What `HELP <topic>;` resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HelpPage {
    Statement(StatementTopic),
    Concept(ConceptTopic),
    Group(HelpGroup),
}
