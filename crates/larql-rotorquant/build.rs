use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/ARCHS.txt");
    println!("cargo:rerun-if-changed=cuda/larql_rotorquant_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/shim/common.cuh");
    println!("cargo:rerun-if-changed=cuda/shim/ggml-common.h");
    println!("cargo:rerun-if-changed=cuda/upstream/cpy-planar-iso.cu");
    println!("cargo:rerun-if-changed=cuda/upstream/cpy-planar-iso.cuh");
    println!("cargo:rerun-if-changed=cuda/upstream/planar-iso-constants.cuh");
    println!("cargo:rerun-if-changed=cuda/upstream/set-rows-planar-iso.cuh");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LARQL_CUDA_ARCH");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!("larql-rotorquant cuda feature currently requires Linux");
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "larql-rotorquant cuda feature requires nvcc. Set NVCC, CUDA_HOME, CUDA_PATH, \
             or install a CUDA toolkit under /usr/local/cuda-*."
        )
    });
    let cuda_home = cuda_home_from_nvcc(&nvcc);
    let archs = cuda_arches(&manifest_dir);

    let src = manifest_dir.join("cuda/larql_rotorquant_kernels.cu");
    let obj = out_dir.join("larql_rotorquant_kernels.o");
    let mut cmd = Command::new(&nvcc);
    cmd.arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-I")
        .arg(manifest_dir.join("cuda/shim"))
        .arg("-I")
        .arg(manifest_dir.join("cuda/upstream"));

    for arch in &archs {
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{arch},code=sm_{arch}"));
    }

    run(&mut cmd, "nvcc");

    let lib = out_dir.join("liblarql_rotorquant_cuda.a");
    let mut ar = Command::new(env::var_os("AR").unwrap_or_else(|| OsString::from("ar")));
    ar.arg("crs").arg(&lib).arg(&obj);
    run(&mut ar, "ar");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=larql_rotorquant_cuda");
    if let Some(cuda_home) = cuda_home {
        for rel in ["lib64", "targets/x86_64-linux/lib"] {
            let dir = cuda_home.join(rel);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn run(cmd: &mut Command, name: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to launch {name}: {err}"));
    if !status.success() {
        panic!("{name} failed with status {status}");
    }
}

fn find_nvcc() -> Option<PathBuf> {
    for var in ["NVCC"] {
        if let Some(path) = env::var_os(var).map(PathBuf::from) {
            if path.is_file() {
                return Some(path);
            }
        }
    }

    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(var).map(PathBuf::from) {
            let path = root.join("bin/nvcc");
            if path.is_file() {
                return Some(path);
            }
        }
    }

    for dir in env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let path = dir.join("nvcc");
        if path.is_file() {
            return Some(path);
        }
    }

    let default = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if default.is_file() {
        return Some(default);
    }

    let mut candidates = Vec::new();
    if let Ok(entries) = fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("cuda-") {
                let nvcc = path.join("bin/nvcc");
                if nvcc.is_file() {
                    candidates.push(nvcc);
                }
            }
        }
    }
    candidates.sort();
    candidates.pop()
}

fn cuda_home_from_nvcc(nvcc: &Path) -> Option<PathBuf> {
    nvcc.parent().and_then(Path::parent).map(Path::to_path_buf)
}

fn cuda_arches(manifest_dir: &Path) -> Vec<String> {
    if let Ok(raw) = env::var("LARQL_CUDA_ARCH") {
        let archs = raw
            .split([',', ';', ' ', '\t', '\n'])
            .filter_map(normalize_arch)
            .collect::<Vec<_>>();
        if archs.is_empty() {
            panic!("LARQL_CUDA_ARCH did not contain any valid CUDA arch values");
        }
        return archs;
    }

    fs::read_to_string(manifest_dir.join("cuda/ARCHS.txt"))
        .unwrap_or_else(|err| panic!("failed to read cuda/ARCHS.txt: {err}"))
        .lines()
        .filter_map(normalize_arch)
        .collect()
}

fn normalize_arch(raw: &str) -> Option<String> {
    let arch = raw
        .trim()
        .strip_prefix("sm_")
        .or_else(|| raw.trim().strip_prefix("compute_"))
        .unwrap_or(raw.trim());
    if arch.is_empty() || !arch.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(arch.to_owned())
}
