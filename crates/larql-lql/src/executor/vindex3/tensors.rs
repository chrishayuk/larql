//! `SHOW TENSORS ["filter"]`: every tensor of every representation by its qualified name.

use larql_vindex::format::vindex3::stored_tensor::{list_tensors, BareName};

use crate::error::LqlError;
use crate::executor::Session;

const STATEMENT: &str = "SHOW TENSORS";
const HEADING_NAME: &str = "Tensor";
const HEADING_DTYPE: &str = "Dtype";
const HEADING_SHAPE: &str = "Shape";
const HEADING_BARE: &str = "Bare";
/// Spaces between columns.
const GUTTER: usize = 2;

impl Session {
    pub(crate) fn exec_show_tensors(&self, filter: Option<&str>) -> Result<Vec<String>, LqlError> {
        let root = self.v3_container_root(STATEMENT)?;
        let listings =
            list_tensors(root).map_err(|e| LqlError::exec("read the tensor tables", e))?;
        let kept: Vec<_> = listings
            .into_iter()
            .filter(|l| filter.is_none_or(|f| l.representation.contains(f) || l.object.contains(f)))
            .collect();
        if kept.is_empty() {
            return Ok(vec![match filter {
                Some(f) => format!("  (no representations match \"{f}\")"),
                None => "  (the directory is empty)".into(),
            }]);
        }
        let rows: Vec<[String; 4]> = kept
            .iter()
            .map(|l| {
                [
                    l.qualified_name(),
                    l.dtype.clone(),
                    format!("{:?}", l.shape),
                    l.bare_name.label().to_string(),
                ]
            })
            .collect();
        let headings = [HEADING_NAME, HEADING_DTYPE, HEADING_SHAPE, HEADING_BARE];
        let widths: Vec<usize> = (0..headings.len())
            .map(|i| {
                rows.iter()
                    .map(|r| r[i].len())
                    .chain(std::iter::once(headings[i].len()))
                    .max()
                    .unwrap_or_default()
            })
            .collect();
        let line = |cells: [&str; 4]| {
            let [name, dtype, shape, bare] = cells;
            format!(
                "{name:<w0$}{gap}{dtype:<w1$}{gap}{shape:<w2$}{gap}{bare}",
                w0 = widths[0],
                w1 = widths[1],
                w2 = widths[2],
                gap = " ".repeat(GUTTER),
            )
        };
        let mut out = vec![line(headings)];
        out.push("-".repeat(widths.iter().sum::<usize>() + GUTTER * (headings.len() - 1)));
        for r in &rows {
            out.push(line([&r[0], &r[1], &r[2], &r[3]]));
        }
        let representations = kept
            .iter()
            .map(|l| l.representation.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let ambiguous = kept
            .iter()
            .filter(|l| l.bare_name == BareName::Ambiguous)
            .count();
        out.push(format!(
            "{} tensor(s) in {representations} representation(s); {ambiguous} bare name(s) ambiguous. \
             The qualified name always works; a bare name works where Bare says yes.",
            kept.len()
        ));
        Ok(out)
    }
}
