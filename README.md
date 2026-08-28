# mlx-rust

Safe Rust bindings to [MLX](https://github.com/ml-explore/mlx).
macOS on Apple Silicon; needs CMake.

## Use

Cargo fetches submodules for git dependencies, so this works as-is:

```toml
mlx-rust = { git = "https://github.com/computer-graphics-tools/mlx-rust" }
```

`cargo add mlx-rust` does not work: cargo never runs git for a registry crate, and
MLX is too large to vendor into one. A registry build has to be pointed at your
own checkouts with `MLX_C_SRC_DIR` and `MLX_SRC_DIR`.

## Build

```sh
git clone --recurse-submodules https://github.com/computer-graphics-tools/mlx-rust
cd mlx-rust && cargo test --release
```

MLX and mlx-c are submodules compiled from source, so GitHub's "Download ZIP" will
not build — archives omit submodules. Already cloned without them? Run
`git submodule update --init --recursive`.

The first build compiles MLX's Metal kernels and takes 5–15 minutes. Later builds
reuse a CMake tree under `target/mlx-rust-build/`, keyed by target, profile,
feature set and deployment target.

The `metal-jit` feature moves the quantized and NAX kernels out of the
ahead-of-time metallib into runtime compilation. That is the code under test, so
use it for iteration only, never for published numbers.

## Docs

`cargo doc -p mlx-rust --open`. Parameter names follow MLX's Python API, and
`tests/quantized.rs` is a port of MLX's own `python/tests/test_quantized.py`.

## License

MIT
