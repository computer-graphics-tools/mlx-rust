pub mod bindings;
pub mod common;
pub mod macos;
pub mod mlx_cmake;
pub mod submodules;

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BuildContext {
    pub manifest_dir: PathBuf,
    pub out_dir: PathBuf,

    /// The mlx-c submodule (or MLX_C_SRC_DIR override).
    pub mlx_c_src_dir: PathBuf,
    /// The mlx submodule (or MLX_SRC_DIR override). Fed to CMake via
    /// FETCHCONTENT_SOURCE_DIR_MLX so mlx-c builds against it unmodified.
    pub mlx_src_dir: PathBuf,

    /// Stable CMake root, deliberately outside OUT_DIR. See mlx_cmake.rs.
    pub cmake_root: PathBuf,
    /// Where MLX writes mlx.metallib (baked into the C++ as METAL_PATH).
    pub metallib_dir: PathBuf,

    pub target: String,
    pub deployment_target: String,
}
