use super::BuildContext;

pub fn generate(ctx: &BuildContext) {
    let wrapper = ctx.manifest_dir.join("wrapper.h");
    println!("cargo::rerun-if-changed={}", wrapper.display());

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", ctx.mlx_c_src_dir.display()))
        // Parsed as C: a few headers pull in MLX's C++ headers under __cplusplus,
        // which would drag in the MLX include dir for no benefit.
        .clang_arg("-std=c11")
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        // NewType, not rustified_enum: an out-of-range dtype discriminant must
        // stay representable rather than become an invalid enum.
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate mlx-c bindings");

    bindings
        .write_to_file(ctx.out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
