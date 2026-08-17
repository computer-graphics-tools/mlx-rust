#[path = "build/mod.rs"]
mod build;

fn main() {
    let ctx = build::submodules::collect_build_context();

    use build::{macos, submodules::is_apple_target};
    if cfg!(feature = "metal") && is_apple_target(&ctx.target) {
        if !macos::supports_nax(&ctx.deployment_target) {
            println!(
                "cargo::warning=mlx-rust-sys: MACOSX_DEPLOYMENT_TARGET={} is below \
                 {}.{} -- MLX will compile with MLX_METAL_NO_NAX and the NAX \
                 quantized kernels will be absent from the metallib.",
                ctx.deployment_target,
                macos::NAX_MIN_DEPLOYMENT_TARGET.0,
                macos::NAX_MIN_DEPLOYMENT_TARGET.1,
            );
        }
        // Host-gated, unlike the deployment target above: `xcrun` only exists on
        // a macOS host.
        if cfg!(target_os = "macos") && macos::sdk_supports_nax() == Some(false)
        {
            println!(
                "cargo::warning=mlx-rust-sys: macOS SDK {} is below {}.{} -- the NAX \
                 quantized kernels will be absent regardless of the deployment \
                 target.",
                macos::sdk_version().unwrap_or_else(|| "?".into()),
                macos::NAX_MIN_DEPLOYMENT_TARGET.0,
                macos::NAX_MIN_DEPLOYMENT_TARGET.1,
            );
        }
    }

    build::mlx_cmake::build(&ctx);
    build::mlx_cmake::link(&ctx);
    build::mlx_cmake::export_metallib(&ctx);
    build::bindings::generate(&ctx);
}
