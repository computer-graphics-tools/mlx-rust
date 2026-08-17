use std::{
    collections::BTreeSet,
    env,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

pub fn abs_path<P: AsRef<Path>>(path: P) -> PathBuf {
    if path.as_ref().is_absolute() {
        path.as_ref().to_path_buf()
    } else {
        env::current_dir().expect("current_dir failed").join(path)
    }
}

pub fn is_truthy_env(name: &str) -> bool {
    let Ok(value) = env::var(name) else {
        return false;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn looks_like_mlx_c_repo_root(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists() && dir.join("mlx/c/ops.h").exists()
}

pub fn looks_like_mlx_repo_root(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists()
        && dir.join("mlx/version.h").exists()
        && dir.join("mlx/backend").exists()
}

/// Every `lib*.a` under `root`, deduplicated by link name.
///
/// Walked rather than hardcoded: `libmlxc.a` sits at `<build>/`, `libmlx.a` at
/// `<build>/_deps/mlx-build/`, and `libjaccl.a` at `<build>/jaccl/`.
pub fn find_static_libs(root: &Path) -> Vec<(String, PathBuf)> {
    let mut seen = BTreeSet::new();
    let mut found = Vec::new();
    for entry in
        WalkDir::new(root).max_depth(8).into_iter().filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str())
            != Some("a")
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let name = stem.strip_prefix("lib").unwrap_or(stem).to_string();
        let Some(directory) = path.parent().map(Path::to_path_buf) else {
            continue;
        };
        if seen.insert(name.clone()) {
            found.push((name, directory));
        }
    }
    found
}

/// FNV-1a, for naming a build directory.
///
/// Not `DefaultHasher`, whose output is unspecified across Rust releases and
/// would rename the CMake tree on a toolchain bump.
pub fn stable_hash(parts: &[&str]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        // Separator, so ["ab", "c"] and ["a", "bc"] do not collide.
        hash ^= 0xff;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub fn num_jobs() -> String {
    env::var("NUM_JOBS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|count| count.get().to_string())
            .unwrap_or_else(|_| "4".into())
    })
}

pub fn have_program(name: &str) -> bool {
    let Ok(search_path) = env::var("PATH") else {
        return false;
    };
    env::split_paths(&search_path).any(|dir| dir.join(name).is_file())
}
