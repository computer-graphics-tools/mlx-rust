//! This crate's own additions, which MLX's Python suite has no counterpart for:
//! the macro-generated stream-defaulted twins, the named-argument call form, and
//! the borrowed host read.

use mlx::{
    Array, Dtype, Stream,
    ops::{self, QuantConfig, QuantMode},
};

/// Weights that vary across the buffer, so a wrong offset or group index shows
/// up as a value mismatch rather than passing on identical elements.
fn sample_weights() -> Array {
    let host: Vec<f32> =
        (0..128 * 128).map(|index| (index % 17) as f32 * 0.01 - 0.08).collect();
    Array::from_slice(&host, &[128, 128]).unwrap()
}

/// The generated twin must do what the explicit form does. Covers every op plus
/// both `Array` methods, which the macro handles on separate code paths.
#[test]
fn defaulted_twins_agree_with_the_explicit_form() {
    let stream = Stream::gpu();
    let config = QuantConfig::affine(64, 4);
    let weights = sample_weights();
    let activations = Array::from_slice(&vec![0.5f32; 128], &[1, 128]).unwrap();

    let (quantized_weights, scales, biases) =
        ops::quantize(&weights, config).unwrap();
    let (device_weights, device_scales, device_biases) =
        ops::quantize_device(&weights, config, &stream).unwrap();
    let biases = biases.unwrap();
    let device_biases = device_biases.unwrap();
    assert_eq!(
        quantized_weights.to_vec::<u32>().unwrap(),
        device_weights.to_vec::<u32>().unwrap()
    );
    assert_eq!(
        scales.to_vec_f32().unwrap(),
        device_scales.to_vec_f32().unwrap()
    );
    assert_eq!(
        biases.to_vec_f32().unwrap(),
        device_biases.to_vec_f32().unwrap()
    );

    let pairs: [(&str, Vec<f32>, Vec<f32>); 4] = [
        (
            "quantized_matmul",
            ops::quantized_matmul(
                &activations,
                &quantized_weights,
                &scales,
                &biases,
                true,
                config,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap(),
            ops::quantized_matmul_device(
                &activations,
                &quantized_weights,
                &scales,
                &biases,
                true,
                config,
                &stream,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        ),
        (
            "dequantize",
            ops::dequantize(&quantized_weights, &scales, &biases, config, None)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            ops::dequantize_device(
                &quantized_weights,
                &scales,
                &biases,
                config,
                None,
                &stream,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        ),
        (
            "slice",
            ops::slice(&weights, &[0, 0], &[4, 128], &[1, 1])
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            ops::slice_device(&weights, &[0, 0], &[4, 128], &[1, 1], &stream)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        ),
        (
            "astype",
            weights.astype(Dtype::Bfloat16).unwrap().to_vec_f32().unwrap(),
            weights
                .astype_device(Dtype::Bfloat16, &stream)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
        ),
    ];
    for (name, defaulted, explicit) in &pairs {
        assert_eq!(defaulted, explicit, "{name} twin disagreed");
    }

    // gather_qmm needs indices, so it does not fit the table above.
    let lhs = Array::from_slice(&[0u32], &[1]).unwrap();
    let rhs = Array::from_slice(&[0u32], &[1]).unwrap();
    let gathered = ops::gather_qmm(
        &activations,
        &quantized_weights,
        &scales,
        &biases,
        &lhs,
        &rhs,
        true,
        config,
        false,
    );
    let gathered_device = ops::gather_qmm_device(
        &activations,
        &quantized_weights,
        &scales,
        &biases,
        &lhs,
        &rhs,
        true,
        config,
        false,
        &stream,
    );
    assert_eq!(
        gathered.unwrap().to_vec_f32().unwrap(),
        gathered_device.unwrap().to_vec_f32().unwrap(),
    );

    // The receiver path, on a view.
    let strided = ops::slice(&weights, &[0, 0], &[8, 8], &[2, 2]).unwrap();
    assert_eq!(
        strided.contiguous().unwrap().to_vec_f32().unwrap(),
        strided.contiguous_device(&stream).unwrap().to_vec_f32().unwrap(),
    );
}

/// Omitting an optional argument must mean `transpose = true` and
/// `affine(64, 4)`.
#[test]
fn omitted_arguments_use_documented_defaults() {
    let weights = sample_weights();
    let activations = Array::from_slice(&vec![0.5f32; 128], &[1, 128]).unwrap();

    let (quantized_weights, scales, biases) =
        ops::quantize(&weights, None).unwrap();
    let biases = biases.expect("affine quantize produces biases");

    let explicit =
        // An owned stream passes straight through, thanks to `impl AsRef<Stream>`.
        ops::quantize_device(&weights, QuantConfig::affine(64, 4), Stream::gpu())
            .unwrap();
    assert_eq!(
        quantized_weights.to_vec::<u32>().unwrap(),
        explicit.0.to_vec::<u32>().unwrap()
    );

    assert_eq!(QuantConfig::default(), QuantConfig::affine(64, 4));
    assert_eq!(QuantConfig::default().mode, QuantMode::Affine);

    let defaulted = ops::quantized_matmul(
        &activations,
        &quantized_weights,
        &scales,
        &biases,
        None,
        None,
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    let spelled_out = ops::quantized_matmul(
        &activations,
        &quantized_weights,
        &scales,
        &biases,
        true,
        QuantConfig::affine(64, 4),
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    assert_eq!(defaulted, spelled_out, "wrong defaults");
}

/// The zero-copy read must agree with `to_vec` and keep both guards.
#[test]
fn as_slice_matches_to_vec_and_keeps_its_guards() {
    let weights = sample_weights();
    let (quantized_weights, ..) = ops::quantize(&weights, None).unwrap();

    assert_eq!(
        quantized_weights.as_slice::<u32>().unwrap(),
        quantized_weights.to_vec::<u32>().unwrap()
    );
    assert_eq!(
        weights.as_slice::<f32>().unwrap(),
        weights.to_vec::<f32>().unwrap()
    );

    let dtype_err =
        quantized_weights.as_slice::<f32>().unwrap_err().to_string();
    assert!(dtype_err.contains("dtype is Uint32"), "got: {dtype_err}");

    let strided = ops::slice(&weights, &[0, 0], &[8, 8], &[2, 2]).unwrap();
    let layout_err = strided.as_slice::<f32>().unwrap_err().to_string();
    assert!(layout_err.contains("not row-contiguous"), "got: {layout_err}");

    let empty = Array::from_slice(&[] as &[f32], &[0]).unwrap();
    assert!(empty.as_slice::<f32>().unwrap().is_empty());
}

/// Two immutable borrows are legal, so the borrow checker will not stop an op
/// from consuming an array while a slice into it is live. MLX can donate an
/// input's buffer to an output; this asserts it does not do so to an array we
/// still hold.
///
/// Run under `MallocScribble=1 MallocGuardEdges=1` so a freed-buffer read faults
/// instead of passing.
#[test]
fn borrowed_slice_survives_ops_consuming_the_array() {
    let weights = sample_weights();
    let snapshot = weights.to_vec::<f32>().unwrap();

    let borrowed = weights.as_slice::<f32>().unwrap();
    assert_eq!(borrowed, snapshot.as_slice());

    // The ops most likely to want the buffer.
    for _ in 0..4 {
        let (quantized_weights, scales, biases) =
            ops::quantize(&weights, None).unwrap();
        let biases = biases.unwrap();
        let dequantized =
            ops::dequantize(&quantized_weights, &scales, &biases, None, None)
                .unwrap();
        dequantized.eval().unwrap();
        let converted = weights.astype(Dtype::Bfloat16).unwrap();
        converted.eval().unwrap();
    }

    assert_eq!(
        borrowed, snapshot,
        "MLX moved the buffer out from under a live as_slice borrow"
    );
}

/// The named-argument form must equal the positional call, including with the
/// arguments out of order and partially omitted.
#[test]
fn named_argument_macro_matches_the_positional_call() {
    let weights = sample_weights();
    let activations = Array::from_slice(&vec![0.5f32; 128], &[1, 128]).unwrap();

    let (quantized_weights, scales, biases) = mlx::quantize!(&weights).unwrap();
    let biases = biases.expect("affine quantize produces biases");
    assert_eq!(
        quantized_weights.to_vec::<u32>().unwrap(),
        ops::quantize(&weights, None).unwrap().0.to_vec::<u32>().unwrap(),
    );

    let positional = ops::quantized_matmul(
        &activations,
        &quantized_weights,
        &scales,
        &biases,
        true,
        QuantConfig::default(),
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();

    // Reversed relative to the signature order, and `config` omitted entirely.
    let named = mlx::quantized_matmul!(
        &activations,
        &quantized_weights,
        &scales,
        transpose = true,
        biases = &biases,
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    assert_eq!(named, positional);

    // All optionals omitted.
    let bare = mlx::dequantize!(&quantized_weights, &scales, biases = &biases)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(
        bare,
        ops::dequantize(&quantized_weights, &scales, &biases, None, None)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
    );
}
