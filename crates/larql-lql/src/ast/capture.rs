//! The parts of a `CAPTURE` statement: where the values are read, at which layers, over which sequence and at
//! which of its positions.

/// Where `CAPTURE` reads the values the executor computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSite {
    /// The residual after a layer: what the next layer reads.
    Residual,
    /// The residual after a layer's attention sublayer has written, before its FFN.
    PostAttention,
    /// The FFN's normed input: what the gate and up projections multiply.
    FfnInput,
    /// The input of the FFN's down projection: the activated intermediate.
    FfnDownInput,
    /// The output of the FFN's down projection, before any post-FFN norm or residual scale.
    FfnDownOutput,
    /// The final-normed hidden state: what the output head reads.
    Final,
    /// The output head's logits over the whole vocabulary.
    Logits,
}

impl CaptureSite {
    /// Every site, in forward order.
    pub const ALL: [CaptureSite; 7] = [
        CaptureSite::Residual,
        CaptureSite::PostAttention,
        CaptureSite::FfnInput,
        CaptureSite::FfnDownInput,
        CaptureSite::FfnDownOutput,
        CaptureSite::Final,
        CaptureSite::Logits,
    ];

    /// The site as a statement spells it.
    pub fn name(self) -> &'static str {
        match self {
            CaptureSite::Residual => "residual",
            CaptureSite::PostAttention => "post_attention",
            CaptureSite::FfnInput => "ffn.input",
            CaptureSite::FfnDownInput => "ffn.down_input",
            CaptureSite::FfnDownOutput => "ffn.down_output",
            CaptureSite::Final => "final",
            CaptureSite::Logits => "logits",
        }
    }

    /// The site a spelling names, ignoring case.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|site| site.name().eq_ignore_ascii_case(name))
    }

    /// Whether the site is inside a layer, so a capture of it names the layer.
    pub fn per_layer(self) -> bool {
        !matches!(self, CaptureSite::Final | CaptureSite::Logits)
    }
}

/// `AT LAYER <n>` or `AT LAYERS (<n>, …)`. One layer gives a `[positions, width]` array, a list a
/// `[layers, positions, width]` array, even when it holds one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureLayers {
    One(usize),
    Many(Vec<usize>),
}

impl CaptureLayers {
    /// The layers, in the order given.
    pub fn layers(&self) -> Vec<usize> {
        match self {
            CaptureLayers::One(layer) => vec![*layer],
            CaptureLayers::Many(layers) => layers.clone(),
        }
    }
}

/// What follows the prompt in the captured sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureContinuation {
    /// `CONTINUE "<text>"`: the text's tokens, with no special tokens added, after the prompt's.
    Text(String),
    /// `CONTINUE TOKENS (<id>, …)`: these token ids after the prompt's.
    Tokens(Vec<u32>),
    /// `GENERATE <n>`: the tokens `INFER … GENERATE <n>` produces for the prompt.
    Generate(u32),
}

/// Which positions of the sequence (prompt, then continuation; 0-based) a capture keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturePositions {
    /// The prompt's last position: the one whose logits predict the first token after the prompt.
    PromptEnd,
    /// Every position that predicts a continuation token: the prompt's last through the second-to-last.
    Reply,
    /// Every position.
    All,
    /// The sequence's last position. The default.
    Last,
    /// These positions, in the order given.
    List(Vec<usize>),
}

impl CapturePositions {
    /// The clause as a statement spells it.
    pub fn spelled(&self) -> String {
        match self {
            CapturePositions::PromptEnd => "PROMPT_END".into(),
            CapturePositions::Reply => "REPLY".into(),
            CapturePositions::All => "ALL".into(),
            CapturePositions::Last => "LAST".into(),
            CapturePositions::List(list) => format!(
                "({})",
                list.iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
