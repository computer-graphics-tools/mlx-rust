use std::{
    env,
    fs::{self, create_dir_all},
    path::Path,
};

use cmake::Config as CMakeConfig;

use super::{
    BuildContext,
    common::{find_static_libs, have_program, is_truthy_env, num_jobs},
};

/// Drop a build tree whose CMake cache points elsewhere.
///
/// The tree outlives OUT_DIR, so it can be inherited by a build whose source path
/// moved. CMake fails outright on that rather than reconfiguring.
fn maybe_clear_cmake_build_dir(
    build_dir: &Path,
    source_dir: &Path,
) {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(contents) = fs::read_to_string(&cache) else {
        return;
    };
    let expected_source =
        source_dir.canonicalize().unwrap_or_else(|_| source_dir.to_path_buf());
    let expected_build =
        build_dir.canonicalize().unwrap_or_else(|_| build_dir.to_path_buf());

    for line in contents.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let expected = if key.starts_with("CMAKE_HOME_DIRECTORY:") {
            &expected_source
        } else if key.starts_with("CMAKE_CACHEFILE_DIR:") {
            &expected_build
        } else {
            continue;
        };
        let stale = fs::canonicalize(value)
            .ok()
            .map(|actual| actual.as_path() != expected.as_path())
            .unwrap_or(true);
        if stale {
            println!(
                "cargo::warning=mlx-rust-sys: clearing stale CMake tree at {}",
                build_dir.display()
            );
            let _ = fs::remove_dir_all(build_dir);
            break;
        }
    }
}

fn cmake_bool(on: bool) -> &'static str {
    if on {
        "ON"
    } else {
        "OFF"
    }
}

fn cmake_profile() -> &'static str {
    match env::var("PROFILE").unwrap_or_else(|_| "release".into()).as_str() {
        // A -O0 MLX is ~10x slower, and CMake's Debug config adds
        // -fmetal-enable-logging, which perturbs kernel codegen.
        "debug" if !is_truthy_env("MLX_RUST_DEBUG_BUILD") => "Release",
        "debug" => "Debug",
        _ => "Release",
    }
}

pub fn build(ctx: &BuildContext) {
    let build_dir = ctx.cmake_root.join("build");
    maybe_clear_cmake_build_dir(&build_dir, &ctx.mlx_c_src_dir);
    create_dir_all(&build_dir).ok();
    create_dir_all(&ctx.metallib_dir).ok();

    let metal = cfg!(feature = "metal");
    let metal_jit = cfg!(feature = "metal-jit");

    if metal_jit {
        println!(
            "cargo::warning=mlx-rust-sys: MLX_METAL_JIT=ON moves the quantized, \
             fp_quantized and NAX kernels out of the AOT metallib and into \
             runtime compilation. That is the code under test -- use this for \
             iteration only, never for published numbers."
        );
    }

    let mut cmake = CMakeConfig::new(&ctx.mlx_c_src_dir);
    // The cmake crate builds in <out_dir>/build; point it at our stable root.
    cmake.out_dir(&ctx.cmake_root);
    cmake.env("CMAKE_BUILD_PARALLEL_LEVEL", num_jobs());

    // FetchContent redirect: builds against the pinned submodule with no network
    // fetch. MLX_C_USE_SYSTEM_MLX would instead need an installed MLXConfig.cmake.
    cmake.define("FETCHCONTENT_SOURCE_DIR_MLX", &ctx.mlx_src_dir);
    cmake.define("MLX_C_USE_SYSTEM_MLX", "OFF");
    cmake.define("MLX_C_BUILD_EXAMPLES", "OFF"); // defaults ON upstream
    cmake.define("BUILD_SHARED_LIBS", "OFF");

    cmake.define("MLX_BUILD_TESTS", "OFF");
    cmake.define("MLX_BUILD_EXAMPLES", "OFF");
    cmake.define("MLX_BUILD_BENCHMARKS", "OFF");
    cmake.define("MLX_BUILD_PYTHON_BINDINGS", "OFF");
    cmake.define("MLX_BUILD_CUDA", "OFF");
    cmake.define("MLX_BUILD_CPU", "ON");
    cmake.define("MLX_BUILD_METAL", cmake_bool(metal));
    cmake.define("MLX_METAL_JIT", cmake_bool(metal_jit));
    cmake.define("MLX_METAL_PATH", &ctx.metallib_dir);
    cmake.define("MLX_USE_CCACHE", cmake_bool(have_program("ccache")));

    cmake.define("CMAKE_CXX_STANDARD", "20");
    cmake.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");
    cmake.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    cmake.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF");
    cmake.cflag("-fno-lto");
    cmake.cxxflag("-fno-lto");

    if ctx.target.contains("apple-darwin") {
        let arch = if ctx.target.contains("aarch64") {
            "arm64"
        } else {
            "x86_64"
        };
        cmake.define("CMAKE_OSX_ARCHITECTURES", arch);
        cmake.define("CMAKE_OSX_DEPLOYMENT_TARGET", &ctx.deployment_target);
    }

    cmake.profile(cmake_profile());
    // Not `install`: mlxc pulls in mlx and mlx-metallib on its own.
    cmake.build_target("mlxc");
    cmake.build();
}

pub fn link(ctx: &BuildContext) {
    let build_dir = ctx.cmake_root.join("build");
    let libs = find_static_libs(&build_dir);

    if libs.is_empty() {
        panic!(
            "mlx-rust-sys: no static libraries found under {}. The CMake build \
             produced nothing linkable.",
            build_dir.display()
        );
    }

    for (_, directory) in &libs {
        println!("cargo::rustc-link-search=native={}", directory.display());
    }

    // Dependents before dependencies; the tail is sorted so the link line is
    // reproducible rather than WalkDir-ordered.
    const ORDERED: [&str; 3] = ["mlxc", "mlx", "jaccl"];
    let names: Vec<&str> = libs.iter().map(|(name, _)| name.as_str()).collect();
    for name in ORDERED {
        if names.contains(&name) {
            println!("cargo::rustc-link-lib=static={name}");
        }
    }
    let mut remaining: Vec<&str> =
        names.iter().copied().filter(|name| !ORDERED.contains(name)).collect();
    remaining.sort_unstable();
    for name in remaining {
        println!("cargo::rustc-link-lib=static={name}");
    }

    println!("cargo::rustc-link-lib=c++");
    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-lib=framework=Accelerate");
    if cfg!(feature = "metal") {
        println!("cargo::rustc-link-lib=framework=Metal");
        println!("cargo::rustc-link-lib=framework=QuartzCore");
    }
}

/// Record the metallib path and fail the build if it is missing.
///
/// MLX bakes the path in as `METAL_PATH`, so nothing needs staging at runtime --
/// but a missing metallib would otherwise surface at the first GPU op.
pub fn export_metallib(ctx: &BuildContext) {
    if !cfg!(feature = "metal") {
        return;
    }
    let metallib = ctx.metallib_dir.join("mlx.metallib");
    if !metallib.exists() {
        panic!(
            "mlx-rust-sys: mlx.metallib was not produced at {}",
            metallib.display()
        );
    }
    println!("cargo::rustc-env=MLX_RUST_METALLIB_PATH={}", metallib.display());
    // Visible to downstream build scripts as DEP_MLXC_METALLIB.
    println!("cargo::metadata=metallib={}", metallib.display());
}
