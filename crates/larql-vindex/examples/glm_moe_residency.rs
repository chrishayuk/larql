//! **What does one correctly-executing GLM-5.3-Flash sparse layer cost in
//! physical memory?**
//!
//! The bank is 288 experts, 6.78 GiB of native FP8, and top-8 routing
//! selects a small fraction of it per token. This measures what actually
//! enters RAM — against what the plan predicts — with the routed output
//! held fixed.
//!
//! # The invariant
//!
//! **Same token, same selected experts, same routed output; only the
//! physical access policy changes.** Every arm's output is compared
//! byte-for-byte against the first arm's, and a mismatch aborts: a
//! residency result over a changed computation is not a residency result.
//!
//! # Method
//!
//! `mincore(2)`, not fault counting, for the same reason
//! `vindex3_residency_probe` gives: Darwin's `MADV_DONTNEED` is lazy and
//! re-eviction unreliable, which makes `getrusage` deltas awkward here.
//! Faults are reported too, as a secondary signal, and the two are kept
//! visibly separate rather than blended.
//!
//! `MADV_RANDOM` is set on the mapping: kernel readahead would page in
//! the very sparsity being measured.
//!
//! # Scope
//!
//! This maps the CHECKPOINT's own safetensors shards, because GLM has no
//! VINDEX3 container yet (its plan is not admissible). So the layout
//! measured is the checkpoint's, in which layer 3's experts are 83–98 %
//! dense in their span with other layers' tensors interleaved. A
//! container that carves the expert bank into its own object would have a
//! different — very likely better — layout, and this number is the
//! baseline that claim will have to beat.
//!
//! ```text
//! cargo run --release -p larql-vindex --example glm_moe_residency -- \
//!     <checkpoint-dir> <layer> <input.f32> [--arms demand,advise,warm]
//! ```
// POSIX-only, like `vindex3_residency_probe` beside it: the measurement
// IS the page-fault behaviour, so there is no portable shape for it.
#[cfg(unix)]
mod probe {
    use larql_models::config::Activation;
    use larql_models::quant::fp8_finegrained::{scale_sibling_name, Fp8Grid};
    use larql_models::{ExpertGatePolicy, ExpertRoutingPolicy, MoeRouterKind};
    use larql_vindex::format::vindex3::opplan::exec::backend::{
        ExpertSlices, PlanBackend, RoutedFfnCall, WeightSlice,
    };
    use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
    use larql_vindex::format::vindex3::opplan::exec::realization::MappedAccess;
    use larql_vindex::runtime::residency::{account, ExpertRegion};
    use memmap2::{Mmap, MmapOptions};
    use std::collections::BTreeMap;
    use std::ops::Range;

    /// Where one tensor lives inside a mapped shard.
    #[derive(Clone)]
    struct Extent {
        shard: usize,
        dtype: String,
        shape: Vec<usize>,
        bytes: Range<usize>,
    }

    struct Mapped {
        maps: Vec<Mmap>,
        extents: BTreeMap<String, Extent>,
        /// Which shard index each shard path took.
        paths: Vec<String>,
    }

    impl Mapped {
        /// Map every shard that holds a tensor under `prefix`, readahead off.
        fn open(dir: &str, prefix: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
                std::path::Path::new(dir).join("model.safetensors.index.json"),
            )?)?;
            let wm = idx["weight_map"].as_object().ok_or("no weight_map")?;
            let mut want: Vec<String> = wm
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(_, v)| v.as_str().unwrap_or_default().to_string())
                .collect();
            want.sort();
            want.dedup();

            let mut maps = Vec::new();
            let mut extents = BTreeMap::new();
            for (i, shard) in want.iter().enumerate() {
                let path = std::path::Path::new(dir).join(shard);
                let file = std::fs::File::open(&path)?;
                // SAFETY: the file is read-only for this process's lifetime.
                let map = unsafe { MmapOptions::new().map(&file)? };
                // Readahead OFF. Without this the kernel pages in neighbours
                // of every touched page and the sparsity under measurement
                // disappears into the measurement instrument.
                advise(&map, libc::MADV_RANDOM, "MADV_RANDOM");

                let n = u64::from_le_bytes(map[..8].try_into()?) as usize;
                let header: serde_json::Value = serde_json::from_slice(&map[8..8 + n])?;
                let base = 8 + n;
                for (name, v) in header.as_object().ok_or("bad header")? {
                    if name == "__metadata__" {
                        continue;
                    }
                    let off = v["data_offsets"].as_array().ok_or("no offsets")?;
                    let (a, b) = (
                        off[0].as_u64().unwrap_or(0) as usize,
                        off[1].as_u64().unwrap_or(0) as usize,
                    );
                    extents.insert(
                        name.clone(),
                        Extent {
                            shard: i,
                            dtype: v["dtype"].as_str().unwrap_or_default().to_string(),
                            shape: v["shape"]
                                .as_array()
                                .ok_or("no shape")?
                                .iter()
                                .map(|s| s.as_u64().unwrap_or(0) as usize)
                                .collect(),
                            bytes: base + a..base + b,
                        },
                    );
                }
                maps.push(map);
            }
            Ok(Self {
                maps,
                extents,
                paths: want,
            })
        }

        fn ext(&self, name: &str) -> Result<&Extent, Box<dyn std::error::Error>> {
            self.extents
                .get(name)
                .ok_or_else(|| format!("`{name}` is in no mapped shard").into())
        }

        /// A tensor's bytes, borrowed from the mapping — never copied, so a
        /// read of them is a page fault on the real file.
        fn raw(&self, name: &str) -> Result<&[u8], Box<dyn std::error::Error>> {
            let e = self.ext(name)?;
            Ok(&self.maps[e.shard][e.bytes.clone()])
        }
    }

    /// Darwin's own "you may reclaim these" advice. `MADV_DONTNEED` is
    /// documented in this workspace's incumbent probe as LAZY here — it
    /// returns success and leaves the pages resident — so eviction tries the
    /// Darwin-specific reuse advice as well, and then CHECKS.
    #[cfg(target_os = "macos")]
    const MADV_FREE_REUSABLE: i32 = 7;

    /// Try to make a mapping cold. Returns nothing on purpose: whether it
    /// worked is a question for `mincore`, not for the return code.
    fn evict(map: &Mmap) {
        // SAFETY: the pointer and length describe a live mapping.
        unsafe {
            libc::msync(
                map.as_ptr() as *mut libc::c_void,
                map.len(),
                libc::MS_INVALIDATE,
            );
        }
        #[cfg(target_os = "macos")]
        advise(map, MADV_FREE_REUSABLE, "MADV_FREE_REUSABLE");
        advise(map, libc::MADV_DONTNEED, "MADV_DONTNEED");
    }

    fn advise(map: &Mmap, flag: i32, what: &str) {
        // SAFETY: the pointer and length describe a live mapping.
        let rc = unsafe { libc::madvise(map.as_ptr() as *mut libc::c_void, map.len(), flag) };
        if rc != 0 {
            eprintln!("  {what} failed: {}", std::io::Error::last_os_error());
        }
    }

    /// One fetched extent: which shard, where in it, and the bytes.
    type Fetched = (usize, Range<usize>, Vec<u8>);

    /// One extent, read explicitly. `pread` rather than seek+read so the
    /// parallel arm's handles do not share a file offset.
    fn pread(file: &std::fs::File, r: &Range<usize>) -> Vec<u8> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; r.len()];
        file.read_exact_at(&mut buf, r.start as u64)
            .expect("explicit fetch");
        buf
    }

    fn page_size() -> usize {
        // SAFETY: a plain sysconf query.
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    /// Which pages of a mapping are resident right now.
    fn resident(map: &Mmap) -> Vec<bool> {
        let pages = map.len().div_ceil(page_size());
        let mut v = vec![0u8; pages];
        // SAFETY: `v` has one byte per page of the live mapping.
        let rc = unsafe {
            libc::mincore(
                map.as_ptr() as *mut libc::c_void,
                map.len(),
                v.as_mut_ptr() as *mut _,
            )
        };
        if rc != 0 {
            eprintln!("  mincore failed: {}", std::io::Error::last_os_error());
            return vec![false; pages];
        }
        v.into_iter().map(|b| b & 1 == 1).collect()
    }

    fn faults() -> (i64, i64) {
        let mut u: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: `u` is a valid, zeroed rusage.
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
        (u.ru_majflt, u.ru_minflt)
    }

    /// One matrix — borrowed from the mapping, or owned when an arm fetched
    /// it explicitly instead of faulting it.
    enum Held<'a> {
        Fp8 {
            codes: std::borrow::Cow<'a, [u8]>,
            scales: Vec<f32>,
            block_rows: usize,
            block_cols: usize,
            scale_cols: usize,
        },
        Bf16(Vec<u16>),
    }

    impl Held<'_> {
        fn slice(&self) -> WeightSlice<'_> {
            match self {
                Held::Fp8 {
                    codes,
                    scales,
                    block_rows,
                    block_cols,
                    scale_cols,
                } => WeightSlice::Fp8Block {
                    codes,
                    scales,
                    block_rows: *block_rows,
                    block_cols: *block_cols,
                    scale_cols: *scale_cols,
                },
                Held::Bf16(w) => WeightSlice::Bf16(w),
            }
        }
    }

    /// Bind one FP8 matrix with its CODES BORROWED FROM THE MAPPING.
    ///
    /// The scales are copied (they are 0.02 % of the bytes and must be f32),
    /// so the residency measured is the residency of the expert CODES — which
    /// is the question. The scale pages are attributed to the expert too, and
    /// counted in the prediction, so nothing is hidden by the copy.
    fn hold<'a>(m: &'a Mapped, tensor: &str) -> Result<Held<'a>, Box<dyn std::error::Error>> {
        let e = m.ext(tensor)?.clone();
        match e.dtype.as_str() {
            "F8_E4M3" => {
                let sib = scale_sibling_name(tensor);
                let se = m.ext(&sib)?.clone();
                let scales: Vec<f32> = m
                    .raw(&sib)?
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let grid = Fp8Grid {
                    rows: e.shape[0],
                    cols: e.shape[1],
                    scale_rows: se.shape[0],
                    scale_cols: se.shape[1],
                };
                let (block_rows, block_cols) = grid.tile()?;
                Ok(Held::Fp8 {
                    codes: std::borrow::Cow::Borrowed(m.raw(tensor)?),
                    scales,
                    block_rows,
                    block_cols,
                    scale_cols: grid.scale_cols,
                })
            }
            "BF16" => Ok(Held::Bf16(
                m.raw(tensor)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            )),
            other => Err(format!("`{tensor}` is {other}").into()),
        }
    }

    fn f32_of(m: &Mapped, tensor: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let e = m.ext(tensor)?;
        let b = m.raw(tensor)?;
        Ok(match e.dtype.as_str() {
            "BF16" => b
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            "F32" => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => return Err(format!("`{tensor}` is {other}").into()),
        })
    }

    fn env_usize(k: &str, d: usize) -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }

    /// Above this fraction of the selected working set still resident, a
    /// "cold" arm is not cold and is refused rather than reported.
    const COLD_CEILING: f64 = 0.05;

    fn gib(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn mib(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let a: Vec<String> = std::env::args().collect();
        if a.len() < 4 {
            eprintln!("usage: glm_moe_residency <checkpoint-dir> <layer> <input.f32> [arms]");
            std::process::exit(2);
        }
        let (dir, layer, input) = (&a[1], &a[2], &a[3]);
        let arms: Vec<String> = a
            .get(4)
            .map(|s| s.split(',').map(str::to_string).collect())
            .unwrap_or_else(|| {
                vec![
                    "demand".into(),
                    "advise".into(),
                    "touch".into(),
                    "warm".into(),
                ]
            });

        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let experts = env_usize("GLM_EXPERTS", 288);
        let top_k = env_usize("GLM_TOP_K", 8);
        let intermediate = env_usize("GLM_MOE_INTERMEDIATE", 2048);
        let branch_scale: f32 = std::env::var("GLM_ROUTED_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.5);
        let limit: f32 = std::env::var("GLM_SWIGLU_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0);

        eprintln!("mapping shards for `{prefix}` (readahead off) …");
        let m = Mapped::open(dir, &prefix)?;
        eprintln!(
            "  {} shard(s), {:.2} GiB of address space",
            m.paths.len(),
            gib(m.maps.iter().map(|x| x.len()).sum::<usize>())
        );

        // ── Regions: every byte each expert owns, per shard ──
        //
        // Three matrices and three scale siblings per expert, and they are
        // NOT adjacent in the checkpoint — other layers' tensors sit between
        // them. Emitting one region per tensor (all sharing the expert's id)
        // is what lets the attribution follow the real layout instead of a
        // convenient bounding box.
        let mut regions: Vec<Vec<ExpertRegion>> = vec![Vec::new(); m.maps.len()];
        let mut bank_bytes = 0usize;
        for e in 0..experts {
            for leaf in ["gate_proj", "up_proj", "down_proj"] {
                for t in [
                    format!("{prefix}.experts.{e}.{leaf}.weight"),
                    format!("{prefix}.experts.{e}.{leaf}.weight_scale_inv"),
                ] {
                    if let Ok(x) = m.ext(&t) {
                        bank_bytes += x.bytes.len();
                        regions[x.shard].push(ExpertRegion {
                            expert_id: e as u32,
                            bytes: x.bytes.clone(),
                        });
                    }
                }
            }
        }
        eprintln!("  bank: {experts} experts, {:.3} GiB", gib(bank_bytes));

        let router_ext = m.ext(&format!("{prefix}.gate.weight"))?.clone();
        let router = f32_of(&m, &format!("{prefix}.gate.weight"))?;
        let router_bias = f32_of(&m, &format!("{prefix}.gate.e_score_correction_bias"))?;
        let hidden = router.len() / experts;

        eprintln!("binding {experts} experts (codes borrowed from the mapping) …");
        let mut gate_h = Vec::with_capacity(experts);
        let mut up_h = Vec::with_capacity(experts);
        let mut down_h = Vec::with_capacity(experts);
        for e in 0..experts {
            gate_h.push(hold(&m, &format!("{prefix}.experts.{e}.gate_proj.weight"))?);
            up_h.push(hold(&m, &format!("{prefix}.experts.{e}.up_proj.weight"))?);
            down_h.push(hold(&m, &format!("{prefix}.experts.{e}.down_proj.weight"))?);
        }
        let gate: Vec<WeightSlice<'_>> = gate_h.iter().map(Held::slice).collect();
        let up: Vec<WeightSlice<'_>> = up_h.iter().map(Held::slice).collect();
        let down: Vec<WeightSlice<'_>> = down_h.iter().map(Held::slice).collect();

        let xs: Vec<f32> = std::fs::read(input)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let positions = xs.len() / hidden;
        eprintln!("  input: {positions} position(s) x {hidden}\n");

        let backend = ProductionBackend;
        // Which experts THIS token selects — read once, from a run whose
        // residency is not being measured, so the predicted set is known
        // before any arm executes.
        let selected = larql_vindex::format::vindex3::opplan::exec::kimi_router::route(
            &xs[..hidden],
            &router,
            &router_bias,
            experts,
            top_k,
            true,
            branch_scale as f64,
            larql_vindex::format::vindex3::opplan::exec::kimi_router::Mutation::None,
        );
        let ids: Vec<u32> = selected.selected_ids.iter().map(|&i| i as u32).collect();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();

        // What the plan PREDICTS this token must bring in: every byte of the
        // selected experts' six tensors, plus the router.
        // Split, because only one half is ever faulted from the mapping: the
        // router is read once at bind and lives on the heap thereafter, so
        // folding it into the prediction would depress coverage by exactly
        // its own size and look like a 1 % shortfall that is not one.
        let predicted_experts: usize = regions
            .iter()
            .flatten()
            .filter(|r| ids.contains(&r.expert_id))
            .map(|r| r.bytes.len())
            .sum();
        let predicted = predicted_experts + router_ext.bytes.len();

        println!("selected experts {sorted_ids:?}");
        println!(
            "predicted selected stored bytes: {:.2} MiB experts + {:.2} MiB router\n\
             \x20 = {:.4} of the {:.3} GiB bank, at top-{top_k} of {experts}\n",
            mib(predicted_experts),
            mib(router_ext.bytes.len()),
            predicted_experts as f64 / bank_bytes as f64,
            gib(bank_bytes)
        );

        let page = page_size();
        let mut reference_out: Option<Vec<f32>> = None;

        // Every extent the selected experts occupy, as (shard, byte range),
        // for the arms that fetch explicitly instead of faulting.
        let mut selected_extents: Vec<(usize, Range<usize>)> = Vec::new();
        for &e in &ids {
            for leaf in ["gate_proj", "up_proj", "down_proj"] {
                for t in [
                    format!("{prefix}.experts.{e}.{leaf}.weight"),
                    format!("{prefix}.experts.{e}.{leaf}.weight_scale_inv"),
                ] {
                    if let Ok(x) = m.ext(&t) {
                        selected_extents.push((x.shard, x.bytes.clone()));
                    }
                }
            }
        }
        // Address order: a fetch that walked the selection in expert order
        // would seek backwards across the file for no reason.
        selected_extents.sort_by_key(|(sh, r)| (*sh, r.start));

        // One handle per shard for the explicit-fetch arms. Opened once, so
        // those arms measure reading and not opening.
        let files: Vec<std::fs::File> = m
            .paths
            .iter()
            .map(|p| std::fs::File::open(std::path::Path::new(dir).join(p)))
            .collect::<Result<_, _>>()?;

        for arm in &arms {
            // The explicit-fetch arms replace the FAULT path for the selected
            // experts with a small number of large reads, and change nothing
            // else: same experts, same bytes, same kernel. Whether the routed
            // output survives is asserted below, not assumed.
            let explicit = arm.starts_with("pread");
            let access = match arm.as_str() {
                "demand" => MappedAccess::Demand,
                "advise" => MappedAccess::Advise,
                // Faults the selected pages CONCURRENTLY, ordered by
                // address, before the projection loop. The arm that matters
                // if paging is latency-bound rather than bandwidth-bound.
                "touch" => MappedAccess::Touch,
                "warm" => MappedAccess::Demand,
                // The bytes never reach the fault path in these arms, so the
                // mapped-access policy is irrelevant to them.
                "pread-serial" | "pread-parallel" => MappedAccess::Demand,
                other => return Err(format!("unknown arm `{other}`").into()),
            };
            let cold = arm != "warm";
            if cold {
                for map in &m.maps {
                    evict(map);
                }
            } else {
                // Warm: bring the selected regions in first, so the timed run
                // finds them resident. Touching one byte per page is enough.
                for (s, rs) in regions.iter().enumerate() {
                    for r in rs.iter().filter(|r| ids.contains(&r.expert_id)) {
                        let mut p = r.bytes.start;
                        while p < r.bytes.end {
                            std::hint::black_box(m.maps[s][p]);
                            p += page;
                        }
                    }
                }
            }

            let before: Vec<Vec<bool>> = m.maps.iter().map(resident).collect();
            let before_pages: usize = before.iter().flatten().filter(|b| **b).count();

            // **Spine check, not a result — and it must run BEFORE any
            // explicit fetch.** On Darwin a `pread` populates the same
            // unified buffer cache the mapping reads, so a check placed
            // after the fetch sees the fetch's own bytes and reports 99.2 %
            // resident on a virgin layer. Measured, and it cost a round of
            // confused arms.
            //
            // A cold arm whose SELECTED regions are still resident measures
            // nothing about paging in,
            // and reporting its latency as a cold number would be a fiction.
            // Checked on the selected regions specifically: the rest of the
            // bank being warm does not invalidate a cold read of the pages
            // this token actually needs.
            if cold {
                let mut sel_before = 0usize;
                for (s_i, rs) in regions.iter().enumerate() {
                    for r in rs.iter().filter(|r| ids.contains(&r.expert_id)) {
                        for p in (r.bytes.start / page)..=(r.bytes.end.saturating_sub(1) / page) {
                            if before[s_i].get(p).copied().unwrap_or(false) {
                                sel_before += 1;
                            }
                        }
                    }
                }
                let warm_fraction = (sel_before * page) as f64 / predicted as f64;
                if warm_fraction > COLD_CEILING {
                    println!(
                        "── arm `{arm}` · SKIPPED: eviction did not take. {:.1}% of the selected \
                         {:.1} MiB is still resident, so nothing here would be a cold measurement.\n                     \x20  Darwin's MADV_DONTNEED is lazy and MADV_FREE_REUSABLE did not \
                         reclaim these pages. Run against a bank the page cache has not already \
                         seen, or reboot between arms.\n",
                        100.0 * warm_fraction,
                        mib(predicted)
                    );
                    continue;
                }
            }
            // ── Explicit fetch: N large reads instead of 12,295 faults ──
            //
            // The selected experts' extents are read with `pread` into owned
            // buffers and the bank's slices are re-pointed at them. Every
            // other expert stays mapped and untouched. Nothing about the
            // selection, the byte set or the kernel changes — only the SHAPE
            // of the requests.
            let mut fetched: Vec<Fetched> = Vec::new();
            let mut fetch_time = std::time::Duration::ZERO;
            if explicit {
                let parallel = arm == "pread-parallel";
                let t0 = std::time::Instant::now();
                if parallel {
                    use std::sync::Mutex;
                    let out: Mutex<Vec<Fetched>> = Mutex::new(Vec::new());
                    std::thread::scope(|sc| {
                        let workers = env_usize("GLM_FETCH_THREADS", 8);
                        let chunks: Vec<_> = selected_extents
                            .chunks(selected_extents.len().div_ceil(workers).max(1))
                            .collect();
                        for chunk in chunks {
                            let out = &out;
                            let files = &files;
                            sc.spawn(move || {
                                let mut local = Vec::new();
                                for (sh, r) in chunk {
                                    local.push((*sh, r.clone(), pread(&files[*sh], r)));
                                }
                                out.lock().expect("fetch lock").extend(local);
                            });
                        }
                    });
                    fetched = out.into_inner().expect("fetch lock");
                } else {
                    for (sh, r) in &selected_extents {
                        fetched.push((*sh, r.clone(), pread(&files[*sh], r)));
                    }
                }
                fetch_time = t0.elapsed();
            }
            // Re-point the selected experts' slices at the fetched buffers.
            let mut gate = gate.clone();
            let mut up = up.clone();
            let mut down = down.clone();
            if explicit {
                let by_range: BTreeMap<(usize, usize), &[u8]> = fetched
                    .iter()
                    .map(|(sh, r, b)| ((*sh, r.start), b.as_slice()))
                    .collect();
                for &e in &ids {
                    for (leaf, bank) in [
                        ("gate_proj", &mut gate),
                        ("up_proj", &mut up),
                        ("down_proj", &mut down),
                    ] {
                        let t = format!("{prefix}.experts.{e}.{leaf}.weight");
                        let x = m.ext(&t)?;
                        if let Some(b) = by_range.get(&(x.shard, x.bytes.start)) {
                            if let WeightSlice::Fp8Block {
                                scales,
                                block_rows,
                                block_cols,
                                scale_cols,
                                ..
                            } = bank[e as usize]
                            {
                                bank[e as usize] = WeightSlice::Fp8Block {
                                    codes: b,
                                    scales,
                                    block_rows,
                                    block_cols,
                                    scale_cols,
                                };
                            }
                        }
                    }
                }
            }

            let (maj0, min0) = faults();
            // A cold arm is measured ONCE — the second run of a cold arm is a
            // warm run. A warm arm is repeated and reported by its MEDIAN,
            // because the single-shot spread here is large (19 ms to 65 ms
            // observed on the same state) and one sample of it is not a
            // latency.
            let repeats = if cold {
                1
            } else {
                env_usize("GLM_WARM_REPEATS", 9)
            };
            let mut samples = Vec::with_capacity(repeats);
            let mut out = Vec::new();
            for _ in 0..repeats {
                let t0 = std::time::Instant::now();
                out = backend.routed_ffn(RoutedFfnCall {
                    x: &xs[..hidden],
                    hidden,
                    intermediate,
                    experts,
                    top_k,
                    router_kind: MoeRouterKind::Sigmoid,
                    routing_policy: ExpertRoutingPolicy::NormalisedOverSelected,
                    branch_scale,
                    activation: Activation::Silu,
                    gate_policy: ExpertGatePolicy::ClampedGated { limit },
                    router: &router,
                    router_bias: Some(&router_bias),
                    weights: ExpertSlices::Separate {
                        gate: &gate,
                        up: &up,
                        down: &down,
                        access,
                    },
                    gate_up_bias: None,
                    down_bias: None,
                    router_input: None,
                    router_scale: None,
                    router_per_expert_scale: None,
                    router_norm_eps: None,
                })?;
                samples.push(t0.elapsed());
            }
            samples.sort();
            let elapsed = samples[samples.len() / 2];
            let (maj1, min1) = faults();
            let after: Vec<Vec<bool>> = m.maps.iter().map(resident).collect();
            let after_pages: usize = after.iter().flatten().filter(|b| **b).count();

            // The invariant: only physical policy changed.
            match &reference_out {
                None => reference_out = Some(out.clone()),
                Some(r) => {
                    let differing = r
                        .iter()
                        .zip(&out)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    if differing != 0 {
                        return Err(format!(
                            "arm `{arm}` changed the routed output in {differing} of {} values — a \
                             residency result over a changed computation is not a residency result",
                            out.len()
                        )
                        .into());
                    }
                }
            }

            // Attribution, per shard, summed.
            let (mut sel, mut unsel, mut rtr) = (0usize, 0usize, 0usize);
            for (s, res) in after.iter().enumerate() {
                let router_range = if router_ext.shard == s {
                    router_ext.bytes.clone()
                } else {
                    0..0
                };
                let acct = account(res, page, &router_range, &regions[s], &ids, predicted);
                sel += acct.resident_selected;
                unsel += acct.resident_unselected;
                rtr += acct.resident_router;
            }

            let state = if cold { "cold (verified)" } else { "warm" };
            println!("── arm `{arm}` · {access:?} · {state}");
            println!(
                "   resident before {:>9.1} MiB   after {:>9.1} MiB   delta {:>+9.1} MiB",
                mib(before_pages * page),
                mib(after_pages * page),
                (after_pages as f64 - before_pages as f64) * page as f64 / (1024.0 * 1024.0)
            );
            println!(
                "   attribution after: selected {:.1} MiB · router {:.1} MiB · UNSELECTED {:.1} MiB",
                mib(sel * page),
                mib(rtr * page),
                mib(unsel * page)
            );
            println!(
                "   predicted-experts {:.1} MiB → resident-selected {:.1} MiB   coverage {:.4}",
                mib(predicted_experts),
                mib(sel * page),
                (sel * page) as f64 / predicted_experts as f64
            );
            // **Logical bytes requested vs physical bytes paged in**, kept
            // apart on purpose. If the OS reads ahead or the extents are
            // scattered across pages the model does not need, these diverge
            // and the next thing to fix is layout. If they agree, the next
            // thing to fix is hit rate.
            let paged = (maj1 - maj0) as usize * page;
            println!(
                "   LOGICAL selected {:.1} MiB  vs  PHYSICAL paged-in {:.1} MiB              ({} major faults x {} KiB)  ratio {:.3}",
                mib(predicted_experts),
                mib(paged),
                maj1 - maj0,
                page / 1024,
                paged as f64 / predicted_experts.max(1) as f64
            );
            if explicit {
                let total: usize = selected_extents.iter().map(|(_, r)| r.len()).sum();
                println!(
                    "   EXPLICIT FETCH: {} reads, mean {:.1} MiB  =  {:.1} MiB in {:.2} ms ({:.0} MiB/s)",
                    selected_extents.len(),
                    mib(total) / selected_extents.len() as f64,
                    mib(total),
                    fetch_time.as_secs_f64() * 1e3,
                    mib(total) / fetch_time.as_secs_f64()
                );
                println!(
                    "   fetch {:.2} + compute {:.2} = {:.2} ms total routed stage",
                    fetch_time.as_secs_f64() * 1e3,
                    elapsed.as_secs_f64() * 1e3,
                    (fetch_time + elapsed).as_secs_f64() * 1e3
                );
            }
            println!(
                "   minor faults {:>8}      routed-FFN latency {:>8.2} ms               (median of {}, spread {:.2}-{:.2})\n",
                min1 - min0,
                elapsed.as_secs_f64() * 1e3,
                samples.len(),
                samples[0].as_secs_f64() * 1e3,
                samples[samples.len() - 1].as_secs_f64() * 1e3
            );
        }
        Ok(())
    }
}

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::run()
}

#[cfg(not(unix))]
fn main() {
    // mmap / madvise / msync / mincore / getrusage are POSIX. Windows would
    // need QueryWorkingSetEx and PrefetchVirtualMemory.
    eprintln!("glm_moe_residency is unix-only.");
}
