//! CUDA compute backend (Linux/NVIDIA).
//!
//! This backend keeps the first CUDA slice build-light: it uses cuBLAS for
//! dense f32 GEMV, dequantises Q4_0/Q4_K/F16 weights into temporary f32
//! views for cuBLAS-backed execution, and falls back to the CPU backend for
//! surfaces that still need dedicated kernels. It uses the CUDA Driver API plus
//! dynamically loaded cuBLAS, so building the crate does not require `nvcc` or
//! the CUDA SDK.

use crate::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, MatMulOp, QuantMatVec};
use crate::cpu::ops::q4_common::{dequantize_q4_k, f16_to_f32, quantize_to_q8};
use crate::cpu::CpuBackend;
use ndarray::{Array2, ArrayView2};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUdeviceptr = u64;
type CUresult = c_int;
type CublasHandle = *mut c_void;
type CublasStatus = c_int;

const CUDA_SUCCESS: CUresult = 0;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_T: c_int = 1;
const RTLD_LAZY: c_int = 0x00001;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytesize: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytesize: usize) -> CUresult;
}

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

type CublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type CublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type CublasSgemv = unsafe extern "C" fn(
    CublasHandle,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> CublasStatus;

struct CublasLib {
    handle: *mut c_void,
    create: CublasCreate,
    destroy: CublasDestroy,
    sgemv: CublasSgemv,
}

impl CublasLib {
    fn load() -> Option<Self> {
        let candidates = [
            "libcublas.so",
            "libcublas.so.13",
            "libcublas.so.12",
            "/usr/local/lib/ollama/cuda_v13/libcublas.so",
            "/usr/local/lib/ollama/mlx_cuda_v13/libcublas.so",
            "/usr/local/lib/ollama/cuda_v12/libcublas.so.12",
        ];
        for candidate in candidates {
            let c_path = CString::new(candidate).ok()?;
            let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_LAZY) };
            if handle.is_null() {
                continue;
            }

            let create = unsafe { symbol::<CublasCreate>(handle, "cublasCreate_v2") }?;
            let destroy = unsafe { symbol::<CublasDestroy>(handle, "cublasDestroy_v2") }?;
            let sgemv = unsafe { symbol::<CublasSgemv>(handle, "cublasSgemv_v2") }?;
            return Some(Self {
                handle,
                create,
                destroy,
                sgemv,
            });
        }
        None
    }
}

impl Drop for CublasLib {
    fn drop(&mut self) {
        unsafe {
            let _ = dlclose(self.handle);
        }
    }
}

unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
    let c_name = CString::new(name).ok()?;
    let ptr = dlsym(handle, c_name.as_ptr());
    if ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&ptr))
    }
}

struct CublasState {
    lib: CublasLib,
    handle: CublasHandle,
}

impl Drop for CublasState {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.lib.destroy)(self.handle);
        }
    }
}

struct DeviceAlloc(CUdeviceptr);

impl DeviceAlloc {
    fn new(bytes: usize) -> Option<Self> {
        let mut ptr = 0;
        let rc = unsafe { cuMemAlloc_v2(&mut ptr, bytes) };
        (rc == CUDA_SUCCESS).then_some(Self(ptr))
    }

    fn copy_from<T>(&self, src: &[T]) -> Option<()> {
        let bytes = std::mem::size_of_val(src);
        let rc = unsafe { cuMemcpyHtoD_v2(self.0, src.as_ptr().cast(), bytes) };
        (rc == CUDA_SUCCESS).then_some(())
    }

    fn copy_to<T>(&self, dst: &mut [T]) -> Option<()> {
        let bytes = std::mem::size_of_val(dst);
        let rc = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr().cast(), self.0, bytes) };
        (rc == CUDA_SUCCESS).then_some(())
    }
}

impl Drop for DeviceAlloc {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFree_v2(self.0);
        }
    }
}

/// Linux/NVIDIA CUDA backend.
pub struct CudaBackend {
    device: CUdevice,
    context: CUcontext,
    device_name: String,
    cublas: Mutex<CublasState>,
    cpu: CpuBackend,
}

/// Row-major f32 matrix copied once into CUDA device memory.
///
/// This is the first persistent-residency primitive for dense attention:
/// Q/K/V/O projection weights can live on GPU while each token/sequence input
/// is copied for GEMV. Large MoE expert/vindex tensors should not use this path.
pub struct CudaResidentF32Matrix {
    data: DeviceAlloc,
    rows: usize,
    cols: usize,
}

impl CudaResidentF32Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Compute `self @ x` using the already-resident matrix.
    pub fn gemv(&self, backend: &CudaBackend, x: &[f32]) -> Option<Vec<f32>> {
        backend.f32_gemv_resident(self, x)
    }
}

unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Try to initialise CUDA device 0 and cuBLAS.
    pub fn new() -> Option<Self> {
        if unsafe { cuInit(0) } != CUDA_SUCCESS {
            return None;
        }

        let mut device = 0;
        if unsafe { cuDeviceGet(&mut device, 0) } != CUDA_SUCCESS {
            return None;
        }

        let mut context = ptr::null_mut();
        if unsafe { cuDevicePrimaryCtxRetain(&mut context, device) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { cuCtxSetCurrent(context) } != CUDA_SUCCESS {
            unsafe {
                let _ = cuDevicePrimaryCtxRelease(device);
            }
            return None;
        }

        let lib = match CublasLib::load() {
            Some(lib) => lib,
            None => {
                unsafe {
                    let _ = cuDevicePrimaryCtxRelease(device);
                }
                return None;
            }
        };

        let mut handle = ptr::null_mut();
        let status = unsafe { (lib.create)(&mut handle) };
        if status != CUBLAS_STATUS_SUCCESS || handle.is_null() {
            unsafe {
                let _ = cuDevicePrimaryCtxRelease(device);
            }
            return None;
        }

        Some(Self {
            device,
            context,
            device_name: device_name(device),
            cublas: Mutex::new(CublasState { lib, handle }),
            cpu: CpuBackend,
        })
    }

    fn lock(&self) -> Option<MutexGuard<'_, CublasState>> {
        let rc = unsafe { cuCtxSetCurrent(self.context) };
        if rc != CUDA_SUCCESS {
            return None;
        }
        self.cublas.lock().ok()
    }

    /// Copy a row-major f32 matrix to device memory once for repeated GEMV.
    pub fn resident_f32_matrix(&self, w: ArrayView2<f32>) -> Option<CudaResidentF32Matrix> {
        let (rows, cols) = (w.shape()[0], w.shape()[1]);
        if rows == 0 || cols == 0 {
            return None;
        }
        unsafe {
            if cuCtxSetCurrent(self.context) != CUDA_SUCCESS {
                return None;
            }
        }
        let w_owned;
        let w_slice = match w.as_slice() {
            Some(slice) => slice,
            None => {
                w_owned = w.as_standard_layout().into_owned();
                w_owned.as_slice().expect("standard layout")
            }
        };
        let data = DeviceAlloc::new(std::mem::size_of_val(w_slice))?;
        data.copy_from(w_slice)?;
        Some(CudaResidentF32Matrix { data, rows, cols })
    }

    fn f32_gemv_resident(&self, w: &CudaResidentF32Matrix, x: &[f32]) -> Option<Vec<f32>> {
        if x.len() != w.cols || w.rows == 0 || w.cols == 0 {
            return None;
        }
        let d_x = DeviceAlloc::new(std::mem::size_of_val(x))?;
        let d_y = DeviceAlloc::new(w.rows * std::mem::size_of::<f32>())?;
        d_x.copy_from(x)?;

        let alpha = 1.0f32;
        let beta = 0.0f32;
        {
            let cublas = self.lock()?;
            let status = unsafe {
                (cublas.lib.sgemv)(
                    cublas.handle,
                    CUBLAS_OP_T,
                    w.cols as c_int,
                    w.rows as c_int,
                    &alpha,
                    w.data.0 as *const f32,
                    w.cols as c_int,
                    d_x.0 as *const f32,
                    1,
                    &beta,
                    d_y.0 as *mut f32,
                    1,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
        }

        let mut out = vec![0.0f32; w.rows];
        d_y.copy_to(&mut out)?;
        Some(out)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

fn dequantise_q8(q8_x: &[i8], q8_scales: &[f32]) -> Vec<f32> {
    let n_blocks = q8_x.len() / 32;
    let mut out = Vec::with_capacity(q8_x.len());
    for (block, scale) in q8_scales.iter().copied().take(n_blocks).enumerate() {
        let off = block * 32;
        for &q in &q8_x[off..off + 32] {
            out.push(q as f32 * scale);
        }
    }
    for &q in q8_x[n_blocks * 32..].iter() {
        out.push(q as f32);
    }
    out
}

fn dequantise_q4_0(q4_data: &[u8], num_rows: usize, hidden: usize) -> Option<Vec<f32>> {
    if hidden == 0 || num_rows == 0 || !hidden.is_multiple_of(32) {
        return None;
    }
    let blocks_per_row = hidden / 32;
    let row_bytes = blocks_per_row * 18;
    if q4_data.len() < num_rows.checked_mul(row_bytes)? {
        return None;
    }

    let mut out = vec![0.0f32; num_rows * hidden];
    for row in 0..num_rows {
        let row_base = row * row_bytes;
        for block in 0..blocks_per_row {
            let off = row_base + block * 18;
            let scale = f16_to_f32(u16::from_le_bytes([q4_data[off], q4_data[off + 1]]));
            let packed = &q4_data[off + 2..off + 18];
            let out_base = row * hidden + block * 32;
            for (i, &byte) in packed.iter().enumerate() {
                let lo = (byte & 0x0f) as i8 - 8;
                let hi = ((byte >> 4) & 0x0f) as i8 - 8;
                out[out_base + i * 2] = lo as f32 * scale;
                out[out_base + i * 2 + 1] = hi as f32 * scale;
            }
        }
    }
    Some(out)
}

fn top_k_from_scores(scores: &[f32], top_k: usize) -> Option<Vec<(u32, f32)>> {
    if top_k == 0 || scores.is_empty() {
        return None;
    }
    let mut pairs: Vec<(u32, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .map(|(idx, score)| (idx as u32, score))
        .collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(top_k.min(pairs.len()));
    (!pairs.is_empty()).then_some(pairs)
}

fn device_name(device: CUdevice) -> String {
    let mut buf = [0i8; 256];
    let rc = unsafe { cuDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, device) };
    if rc != CUDA_SUCCESS {
        return "CUDA device".to_string();
    }
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl MatMul for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.cpu.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.cpu.matmul_transb(a, b)
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        self.cpu.matmul_batch(ops)
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        // Avoid tiny GPU dispatches; the first CUDA slice is for large LM-head/gate GEMVs.
        if 2 * n * k < 1_000_000 {
            return None;
        }
        self.f32_gemv_force(w, x)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 || k == 0 {
            return None;
        }
        let w_owned;
        let w_slice = match w.as_slice() {
            Some(slice) => slice,
            None => {
                w_owned = w.as_standard_layout().into_owned();
                w_owned.as_slice().expect("standard layout")
            }
        };

        let d_w = DeviceAlloc::new(std::mem::size_of_val(w_slice))?;
        let d_x = DeviceAlloc::new(std::mem::size_of_val(x))?;
        let d_y = DeviceAlloc::new(n * std::mem::size_of::<f32>())?;
        d_w.copy_from(w_slice)?;
        d_x.copy_from(x)?;

        let alpha = 1.0f32;
        let beta = 0.0f32;
        {
            let cublas = self.lock()?;
            let status = unsafe {
                (cublas.lib.sgemv)(
                    cublas.handle,
                    CUBLAS_OP_T,
                    k as c_int,
                    n as c_int,
                    &alpha,
                    d_w.0 as *const f32,
                    k as c_int,
                    d_x.0 as *const f32,
                    1,
                    &beta,
                    d_y.0 as *mut f32,
                    1,
                )
            };
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
        }

        let mut out = vec![0.0f32; n];
        d_y.copy_to(&mut out)?;
        Some(out)
    }

    fn f32_gemv_topk1(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<(u32, f32)> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k || n == 0 || k == 0 {
            return None;
        }
        // Safe fallback: top-k is correctness-sensitive and cheap to reduce on CPU.
        // The full-score CUDA GEMV path remains available via f32_gemv_force.
        let x_row = ArrayView2::from_shape((1, k), x).ok()?;
        let out = self.cpu.matmul_transb(x_row, w);
        let scores = out.row(0).to_vec();
        top_k_from_scores(&scores, 1)?.into_iter().next()
    }

    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n.checked_mul(k)?.checked_mul(2)? || x.len() != k {
            return None;
        }
        let mut w = Vec::with_capacity(n * k);
        for chunk in w_f16[..n * k * 2].chunks_exact(2) {
            w.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
        }
        let view = ArrayView2::from_shape((n, k), &w).ok()?;
        self.f32_gemv(view, x)
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n.checked_mul(k)?.checked_mul(2)? || x.len() != k {
            return None;
        }
        let mut w = Vec::with_capacity(n * k);
        for chunk in w_f16[..n * k * 2].chunks_exact(2) {
            w.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
        }
        let view = ArrayView2::from_shape((n, k), &w).ok()?;
        self.f32_gemv_force(view, x)
    }

    fn f16_gemv_topk1(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<(u32, f32)> {
        let scores = self.f16_gemv_force(w_f16, x, n, k)?;
        top_k_from_scores(&scores, 1)?.into_iter().next()
    }

    fn f16_gemv_topk(
        &self,
        w_f16: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        let scores = self.f16_gemv_force(w_f16, x, n, k)?;
        top_k_from_scores(&scores, top_k)
    }
}

impl QuantMatVec for CudaBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let weights = dequantise_q4_0(q4_data, num_rows, hidden)?;
        let x = dequantise_q8(q8_x, q8_scales);
        let view = ArrayView2::from_shape((num_rows, hidden), &weights).ok()?;
        self.f32_gemv_force(view, &x)
    }

    fn q4_matvec_topk1(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<(u32, f32)> {
        let scores = self.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)?;
        top_k_from_scores(&scores, 1)?.into_iter().next()
    }

    fn q4_matvec_topk(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        let scores = self.q4_matvec(q4_data, q8_x, q8_scales, num_rows, hidden)?;
        top_k_from_scores(&scores, top_k)
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.cpu
            .q4_vecmat(activation, q4_data, intermediate, hidden)
    }

    fn q4_matvec_pair_batch(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if x_matrix.len() < seq_len.checked_mul(hidden)? {
            return None;
        }
        let mut gates = Vec::with_capacity(seq_len);
        let mut ups = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let x = &x_matrix[pos * hidden..(pos + 1) * hidden];
            let (q8_x, q8_scales) = quantize_to_q8(x);
            gates.push(self.q4_matvec(gate_q4, &q8_x, &q8_scales, num_rows, hidden)?);
            ups.push(self.q4_matvec(up_q4, &q8_x, &q8_scales, num_rows, hidden)?);
        }
        Some((gates, ups))
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        let weights = dequantize_q4_k(q4k_data, num_rows.checked_mul(hidden)?);
        if weights.is_empty() {
            return None;
        }
        let view = ArrayView2::from_shape((num_rows, hidden), &weights).ok()?;
        self.f32_gemv_force(view, x)
    }

    fn q4k_matvec_stride32(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.q4k_matvec(q4k_data, x, num_rows, hidden)
    }

    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if x.len() < seq_len.checked_mul(hidden)? {
            return None;
        }
        let mut out = Vec::with_capacity(seq_len * num_rows);
        for pos in 0..seq_len {
            let row = &x[pos * hidden..(pos + 1) * hidden];
            out.extend(self.q4k_matvec(q4k_data, row, num_rows, hidden)?);
        }
        Some(out)
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        self.cpu.q6k_matvec(q6k_data, x, num_rows, hidden)
    }

    fn has_q4(&self) -> bool {
        self.cpu.has_q4()
    }
}

impl DecodeBackend for CudaBackend {}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda (NVIDIA GPU + CPU fallback)"
    }

    fn device_info(&self) -> String {
        format!("CUDA GPU: {}", self.device_name)
    }

    fn supports(&self, cap: Capability) -> bool {
        matches!(
            cap,
            Capability::F32Gemv
                | Capability::F16Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::Q4PairBatch
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
