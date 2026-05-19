//! Mmap'd trace store — append-only file for growing context graphs.
//!
//! File layout:
//!   Header (64 bytes): magic, version, hidden_size, n_layers, n_tokens, ...
//!   Token chains:      contiguous, fixed-size per token
//!
//! Each token chain = (n_layers + 1) × 3 vectors × hidden_size × 4 bytes
//!   - 3 vectors per layer-waypoint: residual, attn_delta, ffn_delta
//!   - (n_layers + 1) because layer -1 (embedding) is included
//!
//! For gemma3-4b (34 layers, 2560 hidden):
//!   Chain size = 35 × 3 × 2560 × 4 = 1,075,200 bytes (~1.05 MB per token)
//!
//! Append-only: new tokens extend the file. Old chains are frozen.
//! Mmap'd: OS pages in/out on demand. RSS ≈ active chain only.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use memmap2::Mmap;

use super::types::TraceNode;

const MAGIC: [u8; 4] = *b"TRAC";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

/// Fixed header at the start of the trace file.
#[repr(C)]
#[derive(Clone, Copy)]
struct TraceHeader {
    magic: [u8; 4],
    version: u32,
    hidden_size: u32,
    n_layers: u32, // transformer layers (not counting embedding)
    n_tokens: u32, // number of complete token chains
    _reserved: [u8; 44],
}

impl TraceHeader {
    fn chain_size(&self) -> usize {
        let waypoints = self.n_layers as usize + 1; // +1 for embedding
        let vectors_per_waypoint = 3; // residual, attn_delta, ffn_delta
        waypoints * vectors_per_waypoint * self.hidden_size as usize * 4
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }

    fn expected_file_len(&self) -> usize {
        HEADER_SIZE + self.n_tokens as usize * self.chain_size()
    }
}

/// Read-only mmap'd trace store.
pub struct TraceStore {
    mmap: Mmap,
    header: TraceHeader,
}

impl TraceStore {
    /// Open an existing trace file for reading.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let mut header_bytes = [0u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&mmap[..HEADER_SIZE]);
        let header = TraceHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported version",
            ));
        }
        let expected_len = header.expected_file_len();
        if mmap.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "trace file length mismatch: expected {} bytes from header, got {} bytes",
                    expected_len,
                    mmap.len()
                ),
            ));
        }

        // Advise OS: random access (attention reads arbitrary token chains)
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_RANDOM,
            );
        }

        Ok(Self { mmap, header })
    }

    pub fn n_tokens(&self) -> usize {
        self.header.n_tokens as usize
    }
    pub fn n_layers(&self) -> usize {
        self.header.n_layers as usize
    }
    pub fn hidden_size(&self) -> usize {
        self.header.hidden_size as usize
    }

    /// Read a specific vector from the store.
    /// Returns a slice into mmap'd memory — zero-copy.
    ///
    /// `token`: token position (0-indexed)
    /// `layer`: layer index (0 = embedding, 1..=n_layers = transformer layers)
    /// `component`: 0 = residual, 1 = attn_delta, 2 = ffn_delta
    pub fn read_vector(&self, token: usize, layer: usize, component: usize) -> Option<&[f32]> {
        if token >= self.header.n_tokens as usize {
            return None;
        }
        let n_waypoints = self.header.n_layers as usize + 1;
        if layer >= n_waypoints {
            return None;
        }
        if component >= 3 {
            return None;
        }

        let hidden = self.header.hidden_size as usize;
        let chain_offset = HEADER_SIZE + token * self.header.chain_size();
        let waypoint_offset = layer * 3 * hidden * 4;
        let vec_offset = component * hidden * 4;
        let start = chain_offset + waypoint_offset + vec_offset;
        let end = start + hidden * 4;

        if end > self.mmap.len() {
            return None;
        }

        let slice = &self.mmap[start..end];
        let floats = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, hidden) };
        Some(floats)
    }

    /// Read the residual at (token, layer). Layer 0 = embedding.
    pub fn residual(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 0)
    }

    /// Read the attention delta at (token, layer).
    pub fn attn_delta(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 1)
    }

    /// Read the FFN delta at (token, layer).
    pub fn ffn_delta(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 2)
    }

    /// Read all 3 vectors for a waypoint as a TraceNode.
    pub fn node(&self, token: usize, layer: usize) -> Option<TraceNode> {
        let residual = self.read_vector(token, layer, 0)?.to_vec();
        let attn_delta = self.read_vector(token, layer, 1)?.to_vec();
        let ffn_delta = self.read_vector(token, layer, 2)?.to_vec();
        Some(TraceNode {
            layer: layer as i32 - 1, // convert: store layer 0 = embedding = layer -1
            position: token,
            residual,
            attn_delta,
            ffn_delta,
        })
    }
}

/// Writable trace store — append-only.
pub struct TraceWriter {
    file: File,
    header: TraceHeader,
    path: std::path::PathBuf,
}

impl TraceWriter {
    /// Create a new trace file.
    pub fn create(path: &Path, hidden_size: usize, n_layers: usize) -> io::Result<Self> {
        let header = TraceHeader {
            magic: MAGIC,
            version: VERSION,
            hidden_size: hidden_size as u32,
            n_layers: n_layers as u32,
            n_tokens: 0,
            _reserved: [0; 44],
        };

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        Ok(Self {
            file,
            header,
            path: path.to_path_buf(),
        })
    }

    /// Open an existing trace file for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = TraceHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported version",
            ));
        }

        let expected_len = header.expected_file_len() as u64;
        let actual_len = file.metadata()?.len();
        if actual_len != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "trace file length mismatch: expected {} bytes from header, got {} bytes",
                    expected_len, actual_len
                ),
            ));
        }

        file.seek(io::SeekFrom::Start(expected_len))?;

        Ok(Self {
            file,
            header,
            path: path.to_path_buf(),
        })
    }

    /// Append a complete token chain (all layers) to the store.
    ///
    /// `nodes` must contain (n_layers + 1) nodes for this token, ordered by layer
    /// (-1, 0, 1, ..., n_layers-1).
    pub fn append_chain(&mut self, nodes: &[TraceNode]) -> io::Result<()> {
        let expected = self.header.n_layers as usize + 1;
        if nodes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} nodes, got {}", expected, nodes.len()),
            ));
        }

        let hidden = self.header.hidden_size as usize;
        validate_chain(nodes, hidden)?;

        // Write vectors in order: for each waypoint, [residual, attn_delta, ffn_delta]
        for node in nodes {
            let r_bytes = unsafe {
                std::slice::from_raw_parts(node.residual.as_ptr() as *const u8, hidden * 4)
            };
            let a_bytes = unsafe {
                std::slice::from_raw_parts(node.attn_delta.as_ptr() as *const u8, hidden * 4)
            };
            let f_bytes = unsafe {
                std::slice::from_raw_parts(node.ffn_delta.as_ptr() as *const u8, hidden * 4)
            };
            self.file.write_all(r_bytes)?;
            self.file.write_all(a_bytes)?;
            self.file.write_all(f_bytes)?;
        }

        // Update token count in header
        self.header.n_tokens += 1;
        self.file.seek(io::SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.seek(io::SeekFrom::End(0))?;
        self.file.flush()?;

        Ok(())
    }

    /// Write a full ResidualTrace (all positions) to the store.
    pub fn write_trace(&mut self, trace: &super::types::ResidualTrace) -> io::Result<usize> {
        let n_positions = trace.tokens.len();
        let n_waypoints = self.header.n_layers as usize + 1;

        let mut chains = Vec::with_capacity(n_positions);
        for pos in 0..n_positions {
            // Collect nodes for this position, ordered by layer
            let mut chain: Vec<&TraceNode> =
                trace.nodes.iter().filter(|n| n.position == pos).collect();
            chain.sort_by_key(|n| n.layer);

            if chain.len() != n_waypoints {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "incomplete trace chain for position {}: expected {} nodes, got {}",
                        pos,
                        n_waypoints,
                        chain.len()
                    ),
                ));
            }

            let owned: Vec<TraceNode> = chain.into_iter().cloned().collect();
            validate_chain(&owned, self.header.hidden_size as usize)?;
            chains.push(owned);
        }

        for chain in &chains {
            self.append_chain(chain)?;
        }

        Ok(chains.len())
    }

    /// Finish writing — flush and return the path.
    pub fn finish(mut self) -> io::Result<std::path::PathBuf> {
        self.file.flush()?;
        Ok(self.path)
    }

    pub fn n_tokens(&self) -> usize {
        self.header.n_tokens as usize
    }
}

fn validate_chain(nodes: &[TraceNode], hidden: usize) -> io::Result<()> {
    let Some(first) = nodes.first() else {
        return Ok(());
    };
    let position = first.position;

    for (i, node) in nodes.iter().enumerate() {
        let expected_layer = i as i32 - 1;
        if node.layer != expected_layer {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "trace chain layer mismatch at waypoint {}: expected {}, got {}",
                    i, expected_layer, node.layer
                ),
            ));
        }
        if node.position != position {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "trace chain position mismatch: expected {}, got {}",
                    position, node.position
                ),
            ));
        }
        if node.residual.len() != hidden
            || node.attn_delta.len() != hidden
            || node.ffn_delta.len() != hidden
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector size mismatch: expected {}", hidden),
            ));
        }
    }

    Ok(())
}

// Need Seek for TraceWriter
use std::io::Seek;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::types::{ResidualTrace, TraceNode};
    use super::*;

    fn zero_node(layer: i32, position: usize, hidden: usize) -> TraceNode {
        TraceNode {
            layer,
            position,
            residual: vec![layer as f32; hidden],
            attn_delta: vec![0.0; hidden],
            ffn_delta: vec![position as f32; hidden],
        }
    }

    fn make_chain(n_layers: usize, position: usize, hidden: usize) -> Vec<TraceNode> {
        // (n_layers + 1) nodes: embedding at layer -1, then 0..n_layers-1
        let mut chain = vec![zero_node(-1, position, hidden)];
        for l in 0..n_layers as i32 {
            chain.push(zero_node(l, position, hidden));
        }
        chain
    }

    // ── TraceWriter + TraceStore roundtrip ────────────────────────────────────

    #[test]
    fn create_write_read_roundtrip() {
        let path = std::env::temp_dir().join("larql_trace_test_roundtrip.trac");
        let hidden = 4;
        let n_layers = 2;

        // Write one chain
        let mut writer = TraceWriter::create(&path, hidden, n_layers).expect("create");
        let chain = make_chain(n_layers, 0, hidden);
        writer.append_chain(&chain).expect("append");
        assert_eq!(writer.n_tokens(), 1);
        writer.finish().expect("finish");

        // Read back
        let store = TraceStore::open(&path).expect("open");
        assert_eq!(store.n_tokens(), 1);
        assert_eq!(store.n_layers(), n_layers);
        assert_eq!(store.hidden_size(), hidden);

        // Residual at token=0, layer=0 (embedding) should be [-1.0, -1.0, -1.0, -1.0]
        let residual = store.residual(0, 0).expect("residual");
        assert_eq!(residual.len(), hidden);
        assert!(
            (residual[0] - (-1.0_f32)).abs() < 1e-6,
            "embedding residual = layer -1"
        );

        // FFN delta at token=0, layer=1 (first transformer layer) should be position=0
        let ffn = store.ffn_delta(0, 1).expect("ffn_delta");
        assert!((ffn[0] - 0.0_f32).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let path = std::env::temp_dir().join("larql_trace_test_bounds.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        writer.append_chain(&make_chain(2, 0, 4)).expect("append");
        writer.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open");
        assert!(store.residual(99, 0).is_none(), "out-of-range token → None");
        assert!(store.residual(0, 99).is_none(), "out-of-range layer → None");
        assert!(
            store.read_vector(0, 0, 99).is_none(),
            "out-of-range component → None"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn multiple_tokens_roundtrip() {
        let path = std::env::temp_dir().join("larql_trace_test_multi.trac");
        let hidden = 4;
        let n_layers = 2;
        let mut writer = TraceWriter::create(&path, hidden, n_layers).expect("create");
        for pos in 0..3 {
            writer
                .append_chain(&make_chain(n_layers, pos, hidden))
                .expect("append");
        }
        assert_eq!(writer.n_tokens(), 3);
        writer.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open");
        assert_eq!(store.n_tokens(), 3);
        // Last token (pos=2) FFN delta at embedding layer should reflect position=2
        let ffn = store.ffn_delta(2, 0).expect("ffn_delta for token 2");
        assert!(
            (ffn[0] - 2.0_f32).abs() < 1e-6,
            "ffn_delta should encode position 2"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wrong_chain_length_returns_error() {
        let path = std::env::temp_dir().join("larql_trace_test_bad_len.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        // n_layers=2 requires n_layers+1=3 nodes; pass only 1 → error
        let short = vec![zero_node(-1, 0, 4)];
        let result = writer.append_chain(&short);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_order_chain_returns_error() {
        let path = std::env::temp_dir().join("larql_trace_test_bad_order.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        let mut chain = make_chain(2, 0, 4);
        chain.swap(1, 2);

        let result = writer.append_chain(&chain);
        assert!(
            result.is_err(),
            "layer order should be part of the contract"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_trace_rejects_incomplete_position_without_partial_write() {
        let path = std::env::temp_dir().join("larql_trace_test_incomplete_trace.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        let mut nodes = make_chain(2, 0, 4);
        nodes.push(zero_node(-1, 1, 4));
        let trace = ResidualTrace {
            prompt: "test".into(),
            tokens: vec!["a".into(), "b".into()],
            token_ids: vec![1, 2],
            n_layers: 2,
            hidden_size: 4,
            nodes,
            attention: Vec::new(),
        };

        let result = writer.write_trace(&trace);
        assert!(result.is_err(), "incomplete chains should fail loudly");
        assert_eq!(writer.n_tokens(), 0, "failed write should not append");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn node_accessor_reconstructs_trace_node() {
        let path = std::env::temp_dir().join("larql_trace_test_node.trac");
        let hidden = 4;
        let n_layers = 2;
        let mut writer = TraceWriter::create(&path, hidden, n_layers).expect("create");
        writer
            .append_chain(&make_chain(n_layers, 0, hidden))
            .expect("append");
        writer.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open");
        let node = store.node(0, 1).expect("node at token=0, store_layer=1");
        // store layer 1 = transformer layer 0 (store layer 0 = embedding = trace layer -1)
        assert_eq!(node.layer, 0);
        assert_eq!(node.position, 0);
        assert_eq!(node.residual.len(), hidden);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_bad_magic_returns_error() {
        let path = std::env::temp_dir().join("larql_trace_test_bad_magic.trac");
        let mut bytes = [0u8; 64];
        bytes[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, bytes).expect("write");
        let result = TraceStore::open(&path);
        assert!(result.is_err(), "bad magic should return error");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_truncated_trace_returns_error() {
        let path = std::env::temp_dir().join("larql_trace_test_truncated.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        writer.append_chain(&make_chain(2, 0, 4)).expect("append");
        writer.finish().expect("finish");

        let expected_len = std::fs::metadata(&path).expect("metadata").len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open")
            .set_len(expected_len - 4)
            .expect("truncate");

        let result = TraceStore::open(&path);
        assert!(result.is_err(), "truncated trace should not open");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_file_smaller_than_header_returns_error() {
        // mmap.len() < HEADER_SIZE — early return before parsing magic.
        let path = std::env::temp_dir().join("larql_trace_test_too_small.trac");
        std::fs::write(&path, [0u8; 16]).expect("write tiny file");
        let result = TraceStore::open(&path);
        assert!(result.is_err(), "file smaller than header must error");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_unsupported_version_returns_error() {
        let path = std::env::temp_dir().join("larql_trace_test_bad_version.trac");
        // Forge a header with the right magic but wrong version. Header
        // is exactly HEADER_SIZE bytes, which matches expected_file_len
        // when n_tokens=0, so the length check passes first.
        let header = TraceHeader {
            magic: MAGIC,
            version: VERSION + 99,
            hidden_size: 4,
            n_layers: 2,
            n_tokens: 0,
            _reserved: [0; 44],
        };
        std::fs::write(&path, header.to_bytes()).expect("write");
        let result = TraceStore::open(&path);
        assert!(result.is_err(), "unsupported version must error");
        let err = match result {
            Ok(_) => unreachable!("unsupported version must error"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("version"),
            "error should mention version: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn attn_delta_accessor_reads_correct_component() {
        // attn_delta is exercised by component-2 tests above only via
        // residual/ffn_delta. Lock the accessor explicitly.
        let path = std::env::temp_dir().join("larql_trace_test_attn_delta.trac");
        let hidden = 4;
        let mut writer = TraceWriter::create(&path, hidden, 2).expect("create");
        let mut chain = make_chain(2, 0, hidden);
        // Stamp attn_delta with a recognizable signature so we can verify
        // the offset math hits component=1 (not residual or ffn_delta).
        for n in &mut chain {
            n.attn_delta = vec![42.0; hidden];
        }
        writer.append_chain(&chain).expect("append");
        writer.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open");
        let attn = store.attn_delta(0, 0).expect("attn_delta");
        assert_eq!(attn.len(), hidden);
        for &v in attn {
            assert!((v - 42.0).abs() < 1e-6, "attn_delta should be 42.0");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_trace_appends_all_positions_in_order() {
        // Drives `write_trace` happy path: collects per-position chains,
        // sorts by layer, validates, appends.
        let path = std::env::temp_dir().join("larql_trace_test_write_trace.trac");
        let hidden = 4;
        let n_layers = 2;
        let mut writer = TraceWriter::create(&path, hidden, n_layers).expect("create");

        // Build a 3-position trace where ffn_delta encodes position so we
        // can verify ordering survived the round trip.
        let mut nodes = Vec::new();
        for pos in 0..3 {
            for layer in -1..(n_layers as i32) {
                nodes.push(zero_node(layer, pos, hidden));
            }
        }
        let trace = ResidualTrace {
            prompt: String::new(),
            tokens: vec!["a".into(), "b".into(), "c".into()],
            token_ids: vec![1, 2, 3],
            n_layers,
            hidden_size: hidden,
            nodes,
            attention: Vec::new(),
        };
        let written = writer.write_trace(&trace).expect("write_trace");
        assert_eq!(written, 3);
        writer.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open");
        assert_eq!(store.n_tokens(), 3);
        for pos in 0..3 {
            let ffn = store.ffn_delta(pos, 0).expect("ffn_delta");
            assert!(
                (ffn[0] - pos as f32).abs() < 1e-6,
                "position {pos} round-trip failed"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_open_appends_to_existing_file() {
        // TraceWriter::open exercises the "open existing for append" path,
        // including header parsing + length validation + seek-to-end.
        let path = std::env::temp_dir().join("larql_trace_test_writer_open.trac");
        {
            let mut w = TraceWriter::create(&path, 4, 2).expect("create");
            w.append_chain(&make_chain(2, 0, 4)).expect("append");
            w.finish().expect("finish");
        }
        // Reopen and append a second chain.
        let mut w2 = TraceWriter::open(&path).expect("open existing");
        assert_eq!(w2.n_tokens(), 1);
        w2.append_chain(&make_chain(2, 1, 4)).expect("append #2");
        assert_eq!(w2.n_tokens(), 2);
        w2.finish().expect("finish");

        let store = TraceStore::open(&path).expect("open store");
        assert_eq!(store.n_tokens(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_open_rejects_bad_magic() {
        let path = std::env::temp_dir().join("larql_trace_test_writer_open_bad_magic.trac");
        std::fs::write(&path, [0u8; HEADER_SIZE]).expect("write");
        let result = TraceWriter::open(&path);
        assert!(result.is_err(), "bad magic must error on writer open");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_open_rejects_unsupported_version() {
        let path = std::env::temp_dir().join("larql_trace_test_writer_open_bad_version.trac");
        let header = TraceHeader {
            magic: MAGIC,
            version: VERSION + 99,
            hidden_size: 4,
            n_layers: 2,
            n_tokens: 0,
            _reserved: [0; 44],
        };
        std::fs::write(&path, header.to_bytes()).expect("write");
        let result = TraceWriter::open(&path);
        assert!(result.is_err(), "writer open must reject unknown version");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_open_rejects_length_mismatch() {
        let path = std::env::temp_dir().join("larql_trace_test_writer_open_bad_len.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        writer.append_chain(&make_chain(2, 0, 4)).expect("append");
        writer.finish().expect("finish");
        // Pad the file with extra bytes so length doesn't match the header.
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("append-open")
            .write_all(&[0u8; 16])
            .expect("pad");
        let result = TraceWriter::open(&path);
        assert!(result.is_err(), "writer open must reject length mismatch");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn validate_chain_rejects_position_mismatch() {
        let path = std::env::temp_dir().join("larql_trace_test_pos_mismatch.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        let mut chain = make_chain(2, 0, 4);
        chain[1].position = 7; // diverge from chain[0]
        let result = writer.append_chain(&chain);
        assert!(result.is_err(), "position mismatch must error");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn validate_chain_rejects_vector_size_mismatch() {
        let path = std::env::temp_dir().join("larql_trace_test_vec_size.trac");
        let mut writer = TraceWriter::create(&path, 4, 2).expect("create");
        let mut chain = make_chain(2, 0, 4);
        chain[0].residual = vec![0.0; 99]; // wrong hidden size
        let result = writer.append_chain(&chain);
        assert!(result.is_err(), "vector size mismatch must error");
        let _ = std::fs::remove_file(&path);
    }
}
