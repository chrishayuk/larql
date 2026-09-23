//! Opt-in per-expert intermediate-activation trace for BW12-0 (static
//! sub-expert repackability). `LARQL_MOE_BLOCK_CONTRIB_TRACE=<path>` makes
//! every [`super::ExpertWeightFfn::run_expert_observed`] call append its
//! pre-down activation (`[n_tokens_routed_this_call, intermediate]`, raw
//! signed f32 — the analysis side takes `abs()` and multiplies by the
//! static down-column norm; this file only captures, it does not score) as
//! one binary chunk, plus one manifest line naming that chunk's
//! `(layer, expert, n_tokens, intermediate)` shape.
//!
//! Two files, sharing `<path>`'s stem: `<path>` holds the raw little-endian
//! f32 data (chunks concatenated in call order — a reader walks the
//! manifest to know where each chunk starts and how long it is, exactly
//! `layer-dump`'s plane-plus-manifest convention); `<path>.manifest.jsonl`
//! holds one JSON object per chunk.
//!
//! Attached at the same boundary as the data it captures — inside
//! `run_expert_observed` is where gate/up/activation are already computed
//! for real, so there is no "wrong executor" question here the way
//! [`super::trace`]'s header describes for routing (that trace has two
//! independent route implementations to worry about missing; this one has
//! exactly one place in the codebase that computes a GPT-OSS expert's real
//! pre-down activation on CPU).
//!
//! Default off = byte-identical: the cheap `sink().is_none()` check in
//! [`record`] is the only cost an untraced run pays.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

use ndarray::Array2;

use crate::options;

/// One captured chunk's shape, in call order — the manifest line for one
/// `record()` call. No `serde` derive: `larql-compute` does not take a
/// production dependency for the sake of a diagnostic (same reasoning as
/// [`super::trace::TraceWriter`], which hand-writes its JSON for the same
/// reason) — every field here is a non-negative integer, so there is
/// nothing to escape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkMeta {
    pub layer: usize,
    pub expert: usize,
    pub n_tokens: usize,
    pub intermediate: usize,
}

/// Appends `(meta, data)` pairs: one manifest JSON line, one raw f32 chunk.
///
/// Generic over both sinks so the bookkeeping is testable without touching
/// the filesystem or process env — mirrors [`super::trace::TraceWriter`].
pub struct BlockContribWriter<M: Write, D: Write> {
    manifest: M,
    data: D,
}

impl<M: Write, D: Write> BlockContribWriter<M, D> {
    pub fn new(manifest: M, data: D) -> Self {
        Self { manifest, data }
    }

    /// `activation` is `[n_tokens, intermediate]`, row-major, already the
    /// real post-gate value for whatever policy this architecture uses.
    pub fn record(
        &mut self,
        layer: usize,
        expert: usize,
        activation: &Array2<f32>,
    ) -> std::io::Result<()> {
        let (n_tokens, intermediate) = activation.dim();
        writeln!(
            self.manifest,
            "{{\"layer\":{layer},\"expert\":{expert},\"n_tokens\":{n_tokens},\"intermediate\":{intermediate}}}"
        )?;
        self.manifest.flush()?;
        for &v in activation.iter() {
            self.data.write_all(&v.to_le_bytes())?;
        }
        self.data.flush()
    }
}

/// The process-wide sink's concrete type — factored out purely to keep
/// [`open_sink`]/[`sink`]'s signatures readable.
type FileSink = Mutex<BlockContribWriter<BufWriter<File>, BufWriter<File>>>;

fn open_sink(path: Option<String>) -> Option<FileSink> {
    let path = path?;
    let manifest_path = format!("{path}.manifest.jsonl");
    let open = |p: &str| OpenOptions::new().create(true).append(true).open(p);
    match (open(&manifest_path), open(&path)) {
        (Ok(m), Ok(d)) => Some(Mutex::new(BlockContribWriter::new(
            BufWriter::new(m),
            BufWriter::new(d),
        ))),
        (m, d) => {
            let err = m.err().or(d.err()).expect("one side failed");
            eprintln!(
                "[{}] cannot open {path}: {err} — block-contribution trace disabled",
                options::ENV_MOE_BLOCK_CONTRIB_TRACE
            );
            None
        }
    }
}

fn sink() -> Option<&'static FileSink> {
    static SINK: OnceLock<Option<FileSink>> = OnceLock::new();
    SINK.get_or_init(|| {
        open_sink(options::env_nonempty_value(
            options::ENV_MOE_BLOCK_CONTRIB_TRACE,
        ))
    })
    .as_ref()
}

/// Append one expert call's activation. No-op when tracing is off — the
/// caller still pays for computing `activation` itself (that value already
/// exists as `run_expert_observed`'s return, not an extra cost this
/// function introduces), but nothing is copied or written.
pub fn record(layer: usize, expert: usize, activation: &Array2<f32>) {
    if let Some(sink) = sink() {
        record_into(sink, layer, expert, activation);
    }
}

/// Split from [`record`] for the same reason as [`open_sink`]: the process-
/// global sink cannot be reconfigured in-process, and the two failure arms
/// (writer error, poisoned lock) deserve tests.
fn record_into<M: Write, D: Write>(
    sink: &Mutex<BlockContribWriter<M, D>>,
    layer: usize,
    expert: usize,
    activation: &Array2<f32>,
) {
    match sink.lock() {
        Ok(mut writer) => {
            if let Err(err) = writer.record(layer, expert, activation) {
                eprintln!(
                    "[{}] write failed at layer {layer} expert {expert}: {err}",
                    options::ENV_MOE_BLOCK_CONTRIB_TRACE
                );
            }
        }
        Err(_) => eprintln!(
            "[{}] sink poisoned; trace is incomplete",
            options::ENV_MOE_BLOCK_CONTRIB_TRACE
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use std::io::Cursor;

    // serde_json is a dev-dependency only (see `ChunkMeta`'s doc comment),
    // so tests parse with the untyped `Value` rather than deriving
    // `Deserialize` on a production type.
    fn read_manifest_lines(bytes: &[u8]) -> Vec<ChunkMeta> {
        std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).unwrap();
                ChunkMeta {
                    layer: v["layer"].as_u64().unwrap() as usize,
                    expert: v["expert"].as_u64().unwrap() as usize,
                    n_tokens: v["n_tokens"].as_u64().unwrap() as usize,
                    intermediate: v["intermediate"].as_u64().unwrap() as usize,
                }
            })
            .collect()
    }

    fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn records_one_chunk_with_matching_manifest_and_data() {
        let mut manifest = Vec::new();
        let mut data = Vec::new();
        {
            let mut w = BlockContribWriter::new(Cursor::new(&mut manifest), Cursor::new(&mut data));
            let activation = arr2(&[[1.0f32, -2.0, 3.0], [4.0, -5.0, 6.0]]);
            w.record(3, 7, &activation).unwrap();
        }
        let metas = read_manifest_lines(&manifest);
        assert_eq!(
            metas,
            vec![ChunkMeta {
                layer: 3,
                expert: 7,
                n_tokens: 2,
                intermediate: 3
            }]
        );
        assert_eq!(
            read_f32_le(&data),
            vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            "row-major, sign preserved"
        );
    }

    #[test]
    fn records_multiple_chunks_in_call_order() {
        let mut manifest = Vec::new();
        let mut data = Vec::new();
        {
            let mut w = BlockContribWriter::new(Cursor::new(&mut manifest), Cursor::new(&mut data));
            w.record(0, 1, &arr2(&[[1.0f32, 2.0]])).unwrap();
            w.record(0, 2, &arr2(&[[3.0f32, 4.0], [5.0, 6.0]])).unwrap();
        }
        let metas = read_manifest_lines(&manifest);
        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0].expert, 1);
        assert_eq!(metas[0].n_tokens, 1);
        assert_eq!(metas[1].expert, 2);
        assert_eq!(metas[1].n_tokens, 2);
        assert_eq!(
            read_f32_le(&data),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "second chunk appended after the first, not overwriting it"
        );
    }

    #[test]
    fn open_sink_is_none_for_an_unset_path() {
        assert!(open_sink(None).is_none());
    }

    #[test]
    fn open_sink_reports_and_returns_none_for_an_unwritable_path() {
        // A directory that does not exist and cannot be created (a file
        // component in the middle of the path is not a directory).
        let bogus = "/dev/null/does-not-exist/trace.f32".to_string();
        assert!(open_sink(Some(bogus)).is_none());
    }

    /// A diagnostic must not take down a scoring run: both failure arms
    /// (writer error, poisoned lock) report and return rather than panic.
    #[test]
    fn record_into_swallows_writer_errors_and_poison() {
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("no space"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let failing = Mutex::new(BlockContribWriter::new(Vec::new(), Failing));
        record_into(&failing, 0, 1, &arr2(&[[1.0f32]]));

        let poisoned = Mutex::new(BlockContribWriter::new(Vec::new(), Vec::new()));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the lock");
        }));
        record_into(&poisoned, 0, 1, &arr2(&[[1.0f32]]));
    }

    #[test]
    fn open_sink_writable_path_round_trips_a_record() {
        let stem = std::env::temp_dir().join(format!(
            "larql-block-contrib-test-{}.f32",
            std::process::id()
        ));
        let path = stem.display().to_string();
        let sink = open_sink(Some(path.clone())).expect("temp file is creatable");

        record_into(&sink, 2, 9, &arr2(&[[1.5f32, -2.5]]));
        drop(sink.into_inner().expect("unpoisoned")); // flush both BufWriters

        let manifest_path = format!("{path}.manifest.jsonl");
        let metas = read_manifest_lines(&std::fs::read(&manifest_path).unwrap());
        assert_eq!(metas[0].layer, 2);
        assert_eq!(metas[0].expert, 9);
        assert_eq!(read_f32_le(&std::fs::read(&path).unwrap()), vec![1.5, -2.5]);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&manifest_path);
    }
}
