//! Boundary residual store — compressed context for infinite sequences.
//!
//! Stores one residual vector per window boundary. The Markov property
//! guarantees each boundary residual is the complete compressed state
//! of all preceding tokens.
//!
//! File layout:
//!   Header (64 bytes):  magic, version, hidden_size, window_size, n_boundaries, ...
//!   Boundary index:     n_boundaries × BoundaryEntry (16 bytes each)
//!   Residual data:      n_boundaries × hidden_size × 4 bytes
//!
//! For gemma3-4b (2560 hidden), 200-token windows:
//!   Per boundary:  2560 × 4 = 10,240 bytes (~10 KB)
//!   370K tokens:   1,850 boundaries × 10 KB = ~18.5 MB
//!   vs KV cache:   56 GB
//!   Compression:   ~3,000×
//!
//! The store is append-only. New boundaries extend the file.
//! Mmap'd for zero-copy reads. RSS ≈ one boundary at a time.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

use memmap2::Mmap;

const MAGIC: [u8; 4] = *b"BNDX";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const ENTRY_SIZE: usize = 16;

/// Fixed header.
#[repr(C)]
#[derive(Clone, Copy)]
struct BoundaryHeader {
    magic: [u8; 4],
    version: u32,
    hidden_size: u32,
    window_size: u32,  // tokens per window
    n_boundaries: u32, // number of stored boundaries
    total_tokens: u32, // total tokens processed
    _reserved: [u8; 40],
}

impl BoundaryHeader {
    fn residual_bytes(&self) -> usize {
        self.hidden_size as usize * 4
    }

    fn index_offset(&self) -> usize {
        HEADER_SIZE
    }

    #[allow(dead_code)]
    fn index_size(&self) -> usize {
        self.n_boundaries as usize * ENTRY_SIZE
    }

    #[allow(dead_code)]
    fn data_offset(&self, max_boundaries: usize) -> usize {
        // Reserve index space for up to max_boundaries entries
        HEADER_SIZE + max_boundaries * ENTRY_SIZE
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }
}

/// Index entry for one boundary.
#[repr(C)]
#[derive(Clone, Copy)]
struct BoundaryEntry {
    /// Token offset where this boundary starts.
    token_offset: u32,
    /// Number of tokens in this window.
    window_tokens: u32,
    /// Byte offset into the data section where the residual lives.
    data_offset: u32,
    _reserved: u32,
}

impl BoundaryEntry {
    fn to_bytes(self) -> [u8; ENTRY_SIZE] {
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(bytes: &[u8; ENTRY_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }
}

/// Read-only mmap'd boundary store.
pub struct BoundaryStore {
    mmap: Mmap,
    header: BoundaryHeader,
}

impl BoundaryStore {
    /// Open an existing boundary file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let mut header_bytes = [0u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&mmap[..HEADER_SIZE]);
        let header = BoundaryHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

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

    pub fn n_boundaries(&self) -> usize {
        self.header.n_boundaries as usize
    }
    pub fn total_tokens(&self) -> usize {
        self.header.total_tokens as usize
    }
    pub fn hidden_size(&self) -> usize {
        self.header.hidden_size as usize
    }
    pub fn window_size(&self) -> usize {
        self.header.window_size as usize
    }

    /// Get the index entry for boundary i.
    fn entry(&self, i: usize) -> Option<BoundaryEntry> {
        if i >= self.header.n_boundaries as usize {
            return None;
        }
        let offset = self.header.index_offset() + i * ENTRY_SIZE;
        if offset + ENTRY_SIZE > self.mmap.len() {
            return None;
        }
        let mut bytes = [0u8; ENTRY_SIZE];
        bytes.copy_from_slice(&self.mmap[offset..offset + ENTRY_SIZE]);
        Some(BoundaryEntry::from_bytes(&bytes))
    }

    /// Read boundary residual i — zero-copy from mmap.
    pub fn residual(&self, i: usize) -> Option<&[f32]> {
        let entry = self.entry(i)?;
        let hidden = self.header.hidden_size as usize;
        let start = entry.data_offset as usize;
        let end = start + hidden * 4;
        if end > self.mmap.len() {
            return None;
        }
        let slice = &self.mmap[start..end];
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, hidden) })
    }

    /// Find the boundary that contains a given token offset.
    pub fn boundary_for_token(&self, token: usize) -> Option<usize> {
        for i in 0..self.header.n_boundaries as usize {
            let entry = self.entry(i)?;
            let start = entry.token_offset as usize;
            let end = start + entry.window_tokens as usize;
            if token >= start && token < end {
                return Some(i);
            }
        }
        None
    }

    /// Get the token range for boundary i.
    pub fn token_range(&self, i: usize) -> Option<(usize, usize)> {
        let entry = self.entry(i)?;
        Some((
            entry.token_offset as usize,
            entry.token_offset as usize + entry.window_tokens as usize,
        ))
    }

    /// File size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Data size (just the residuals, no header/index overhead).
    pub fn data_size(&self) -> usize {
        self.header.n_boundaries as usize * self.header.residual_bytes()
    }
}

/// Writable boundary store — append-only.
pub struct BoundaryWriter {
    file: File,
    header: BoundaryHeader,
    path: std::path::PathBuf,
    max_boundaries: usize,
}

impl BoundaryWriter {
    /// Create a new boundary store.
    ///
    /// `max_boundaries` pre-allocates index space. Can be extended later
    /// but pre-allocation avoids rewriting. Default: 10,000 (enough for 2M tokens
    /// at window_size=200).
    pub fn create(
        path: &Path,
        hidden_size: usize,
        window_size: usize,
        max_boundaries: usize,
    ) -> io::Result<Self> {
        let header = BoundaryHeader {
            magic: MAGIC,
            version: VERSION,
            hidden_size: hidden_size as u32,
            window_size: window_size as u32,
            n_boundaries: 0,
            total_tokens: 0,
            _reserved: [0; 40],
        };

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Write header
        file.write_all(&header.to_bytes())?;

        // Pre-allocate index space (zeroed)
        let index_bytes = vec![0u8; max_boundaries * ENTRY_SIZE];
        file.write_all(&index_bytes)?;
        file.flush()?;

        Ok(Self {
            file,
            header,
            path: path.to_path_buf(),
            max_boundaries,
        })
    }

    /// Append a boundary residual.
    ///
    /// `token_offset`: the token position where this window starts
    /// `window_tokens`: number of tokens in this window
    /// `residual`: the boundary residual vector (hidden_size floats)
    pub fn append(
        &mut self,
        token_offset: usize,
        window_tokens: usize,
        residual: &[f32],
    ) -> io::Result<()> {
        let hidden = self.header.hidden_size as usize;
        if residual.len() != hidden {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} floats, got {}", hidden, residual.len()),
            ));
        }

        let boundary_idx = self.header.n_boundaries as usize;
        if boundary_idx >= self.max_boundaries {
            return Err(io::Error::other(
                "boundary index full — increase max_boundaries",
            ));
        }

        // Write residual data at end of file
        self.file.seek(SeekFrom::End(0))?;
        let data_pos = self.file.stream_position()? as u32;
        let r_bytes =
            unsafe { std::slice::from_raw_parts(residual.as_ptr() as *const u8, hidden * 4) };
        self.file.write_all(r_bytes)?;

        // Write index entry
        let entry = BoundaryEntry {
            token_offset: token_offset as u32,
            window_tokens: window_tokens as u32,
            data_offset: data_pos,
            _reserved: 0,
        };
        let entry_offset = HEADER_SIZE + boundary_idx * ENTRY_SIZE;
        self.file.seek(SeekFrom::Start(entry_offset as u64))?;
        self.file.write_all(&entry.to_bytes())?;

        // Update header
        self.header.n_boundaries += 1;
        self.header.total_tokens = (token_offset + window_tokens) as u32;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;

        self.file.flush()?;
        Ok(())
    }

    pub fn n_boundaries(&self) -> usize {
        self.header.n_boundaries as usize
    }
    pub fn total_tokens(&self) -> usize {
        self.header.total_tokens as usize
    }

    pub fn finish(mut self) -> io::Result<std::path::PathBuf> {
        self.file.flush()?;
        Ok(self.path)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ── BoundaryWriter + BoundaryStore ────────────────────────────────────────

    #[test]
    fn create_append_open_roundtrip() {
        let path = std::env::temp_dir().join("larql_boundary_test_roundtrip.bndx");
        let hidden = 4;
        let residual: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut writer = BoundaryWriter::create(&path, hidden, 100, 50).expect("create");
        writer.append(0, 100, &residual).expect("append");
        assert_eq!(writer.n_boundaries(), 1);
        assert_eq!(writer.total_tokens(), 100);
        writer.finish().expect("finish");

        let store = BoundaryStore::open(&path).expect("open");
        assert_eq!(store.n_boundaries(), 1);
        assert_eq!(store.hidden_size(), hidden);
        assert_eq!(store.window_size(), 100);
        assert_eq!(store.total_tokens(), 100);

        let r = store.residual(0).expect("residual 0");
        assert_eq!(r.len(), hidden);
        for (i, &v) in r.iter().enumerate() {
            assert!((v - residual[i]).abs() < 1e-6, "residual[{i}] mismatch");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn multiple_boundaries_indexed_correctly() {
        let path = std::env::temp_dir().join("larql_boundary_test_multi.bndx");
        let hidden = 4;
        let mut writer = BoundaryWriter::create(&path, hidden, 200, 10).expect("create");
        for i in 0..3 {
            writer
                .append(i * 200, 200, &vec![i as f32; hidden])
                .expect("append");
        }
        writer.finish().expect("finish");

        let store = BoundaryStore::open(&path).expect("open");
        assert_eq!(store.n_boundaries(), 3);

        // Each residual should reflect the index used to write it
        for i in 0..3 {
            let r = store.residual(i).expect("residual");
            assert!(
                (r[0] - i as f32).abs() < 1e-6,
                "boundary {i} residual mismatch"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn out_of_range_residual_returns_none() {
        let path = std::env::temp_dir().join("larql_boundary_test_oob.bndx");
        let mut writer = BoundaryWriter::create(&path, 4, 100, 10).expect("create");
        writer.append(0, 100, &[1.0f32; 4]).expect("append");
        writer.finish().expect("finish");

        let store = BoundaryStore::open(&path).expect("open");
        assert!(store.residual(99).is_none(), "out-of-range boundary → None");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn boundary_for_token_finds_correct_window() {
        let path = std::env::temp_dir().join("larql_boundary_test_tok.bndx");
        let mut writer = BoundaryWriter::create(&path, 4, 100, 10).expect("create");
        writer.append(0, 100, &[0.0f32; 4]).expect("append 0");
        writer.append(100, 100, &[1.0f32; 4]).expect("append 1");
        writer.finish().expect("finish");

        let store = BoundaryStore::open(&path).expect("open");
        assert_eq!(
            store.boundary_for_token(50),
            Some(0),
            "token 50 in window 0"
        );
        assert_eq!(
            store.boundary_for_token(150),
            Some(1),
            "token 150 in window 1"
        );
        assert!(
            store.boundary_for_token(999).is_none(),
            "out-of-range token"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn token_range_returns_correct_bounds() {
        let path = std::env::temp_dir().join("larql_boundary_test_range.bndx");
        let mut writer = BoundaryWriter::create(&path, 4, 200, 5).expect("create");
        writer.append(0, 200, &[0.0f32; 4]).expect("append");
        writer.finish().expect("finish");

        let store = BoundaryStore::open(&path).expect("open");
        let (start, end) = store.token_range(0).expect("token range");
        assert_eq!(start, 0);
        assert_eq!(end, 200);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wrong_residual_size_returns_error() {
        let path = std::env::temp_dir().join("larql_boundary_test_bad_size.bndx");
        let mut writer = BoundaryWriter::create(&path, 4, 100, 10).expect("create");
        let result = writer.append(0, 100, &[1.0f32; 8]); // wrong size
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }
}
