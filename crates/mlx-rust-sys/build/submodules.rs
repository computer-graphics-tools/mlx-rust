use std::{
    env,
    path::{Path, PathBuf},
};

use super::{
    BuildContext,
    common::{
        abs_path, is_truthy_env, looks_like_mlx_c_repo_root,
        looks_like_mlx_repo_root, stable_hash,
    },
    macos,
};

/// Whether the Apple-specific CMake knobs apply to this target.
pub fn is_apple_target(target: &str) -> bool {
    target.contains("apple-darwin") || target.contains("apple-ios")
}

fn resolve_source_tree(
    env_var: &str,
    submodule: &Path,
    validate: fn(&Path) -> bool,
    what: &str,
    repo_root: &Path,
) -> PathBuf {
    println!("cargo::rerun-if-env-changed={env_var}");
    if let Ok(override_path) = env::var(env_var) {
        let override_path = abs_path(override_path);
        if !validate(&override_path) {
            panic!(
                "{env_var}={} does not look like a {what} repo root",
                override_path.display()
            );
        }
        return override_path;
    }
    if !validate(submodule) {
        panic!(
            "{}",
            missing_submodule_message(env_var, what, submodule, repo_root)
        );
    }
    submodule.to_path_buf()
}

/// Why `what` is missing, and what to do about it. Told apart by what is on disk,
/// since each case needs different advice.
fn missing_submodule_message(
    env_var: &str,
    what: &str,
    submodule: &Path,
    repo_root: &Path,
) -> String {
    // No .gitmodules: not a checkout of this repo, so a registry or vendored
    // build. MLX is too large to ship in the published crate.
    if !repo_root.join(".gitmodules").exists() {
        return format!(
            "Cannot find the {what} sources, and {} is not an mlx-rust \
             checkout.\n\
             Set {env_var} to a {what} checkout, or depend on the repository, \
             which does fetch the submodules:\n\n    \
             mlx-rust = {{ git = \
             \"https://github.com/computer-graphics-tools/mlx-rust\" }}\n",
            repo_root.display(),
        );
    }
    if repo_root.join(".git").exists() {
        return format!(
            "The {what} submodule is not initialized at {}.\n\
             Run `git submodule update --init --recursive`, or set {env_var} to \
             a checked-out {what} repo root.",
            submodule.display(),
        );
    }
    // .gitmodules but no .git: a source archive. Those omit submodules, and the
    // pinned revisions live in git's index, so there is nothing to repair.
    format!(
        "{} is an extracted archive, not a clone, so {what} is empty at {}.\n\
         GitHub archives omit submodules -- clone instead:\n\n    \
         git clone --recurse-submodules \
         https://github.com/computer-graphics-tools/mlx-rust\n",
        repo_root.display(),
        submodule.display(),
    )
}

/// Stable CMake root, deliberately outside OUT_DIR, which cargo invalidates on
/// any feature or profile change -- and an MLX rebuild costs 5-15 minutes.
/// `MLX_RUST_BUILD_IN_OUT_DIR=1` restores hermetic semantics for CI.
///
/// The key includes the feature set because `metal-jit` changes which kernels
/// land in the metallib; sharing a tree with `metal` would let one build strip
/// the other's kernels.
fn resolve_cmake_root(
    out_dir: &Path,
    target: &str,
    profile: &str,
    deployment_target: &str,
    mlx_src_dir: &Path,
    mlx_c_src_dir: &Path,
) -> PathBuf {
    println!("cargo::rerun-if-env-changed=MLX_RUST_CMAKE_BUILD_DIR");
    println!("cargo::rerun-if-env-changed=MLX_RUST_BUILD_IN_OUT_DIR");

    if let Ok(override_dir) = env::var("MLX_RUST_CMAKE_BUILD_DIR") {
        return abs_path(override_dir);
    }
    if is_truthy_env("MLX_RUST_BUILD_IN_OUT_DIR") {
        return out_dir.join("mlx-cmake");
    }

    // OUT_DIR is <target-dir>/<profile>/build/<pkg>-<hash>/out
    let target_dir = out_dir
        .ancestors()
        .nth(4)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.to_path_buf());

    let features = match (cfg!(feature = "metal"), cfg!(feature = "metal-jit"))
    {
        (true, true) => "metal-jit",
        (true, false) => "metal",
        (false, _) => "nometal",
    };
    let src_hash = stable_hash(&[
        &mlx_src_dir.to_string_lossy(),
        &mlx_c_src_dir.to_string_lossy(),
    ]) as u32;

    target_dir.join("mlx-rust-build").join(format!(
        "{target}-{profile}-{features}-{deployment_target}-{src_hash:08x}"
    ))
}

pub fn collect_build_context() -> BuildContext {
    let manifest_dir = abs_path(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let repo_root = manifest_dir
        .ancestors()
        .nth(2)
        .expect("crates/mlx-rust-sys should be two levels below the repo root")
        .to_path_buf();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    println!(
        "cargo::rerun-if-changed={}",
        repo_root.join(".gitmodules").display()
    );

    let mlx_c_src_dir = resolve_source_tree(
        "MLX_C_SRC_DIR",
        &repo_root.join("mlx-c"),
        looks_like_mlx_c_repo_root,
        "mlx-c",
        &repo_root,
    );
    let mlx_src_dir = resolve_source_tree(
        "MLX_SRC_DIR",
        &repo_root.join("mlx"),
        looks_like_mlx_repo_root,
        "mlx",
        &repo_root,
    );

    // Rebuild when either upstream changes, but do not watch the whole tree --
    // MLX is ~4k files and cargo would stat all of them on every invocation.
    for watched in [
        mlx_c_src_dir.join("CMakeLists.txt"),
        mlx_c_src_dir.join("mlx/c"),
        mlx_src_dir.join("CMakeLists.txt"),
        mlx_src_dir.join("mlx/version.h"),
    ] {
        println!("cargo::rerun-if-changed={}", watched.display());
    }

    let target = env::var("TARGET").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".into());

    // Keyed off the target, not `cfg!(target_os)`: a build script is compiled for
    // the host, so that would answer for the wrong machine when cross-compiling.
    let deployment_target = if is_apple_target(&target) {
        macos::deployment_target()
    } else {
        String::from("0.0")
    };
    println!("cargo::rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    let cmake_root = resolve_cmake_root(
        &out_dir,
        &target,
        &profile,
        &deployment_target,
        &mlx_src_dir,
        &mlx_c_src_dir,
    );
    let metallib_dir = cmake_root.join("metallib");

    BuildContext {
        manifest_dir,
        out_dir,
        mlx_c_src_dir,
        mlx_src_dir,
        cmake_root,
        metallib_dir,
        target,
        deployment_target,
    }
}
