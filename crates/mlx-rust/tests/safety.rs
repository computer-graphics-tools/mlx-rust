//! Rejection cases. The first two port MLX's `test_mode_error_cases` and
//! `test_throw` from `python/tests/test_quantized.py`; the rest cover
//! preconditions this binding adds, which Python has no equivalent for because it
//! never hands out a raw buffer.

use mlx::{
    Array, Dtype,
    ops::{self, QuantConfig, QuantMode},
};

fn affine_triple() -> (Array, Array, Array, Array) {
    let config = QuantConfig::affine(64, 4);
    let weights =
        Array::from_slice(&vec![0.1f32; 256 * 256], &[256, 256]).unwrap();
    let (quantized, scales, biases) = ops::quantize(&weights, config).unwrap();
    (weights, quantized, scales, biases.expect("affine yields biases"))
}

/// Ports `test_mode_error_cases`.
///
/// Upstream's first assertions pass `mode="xyz"` and expect a `ValueError`. That
/// case cannot be written here: [`QuantMode`] is an enum, so an invalid mode is a
/// compile error rather than a runtime one. The rest port directly.
#[test]
fn mode_error_cases() {
    let affine = QuantConfig::affine(64, 4);
    let (weights, quantized, scales, biases) = affine_triple();
    let activations = Array::from_slice(&vec![0.5f32; 256], &[1, 256]).unwrap();

    // Only floating point types can be quantized.
    let integers =
        Array::from_slice(&vec![0i32; 128 * 128], &[128, 128]).unwrap();
    assert!(
        ops::quantize(&integers, affine).is_err(),
        "quantizing int32 should fail"
    );
    assert!(
        ops::quantize(&integers, QuantConfig::with_defaults(QuantMode::Mxfp4))
            .is_err(),
        "quantizing int32 as mxfp4 should fail"
    );

    // Affine must be given biases, on every op that consumes them.
    for error in [
        ops::dequantize(&quantized, &scales, None, affine, None).unwrap_err(),
        ops::quantized_matmul(
            &activations,
            &quantized,
            &scales,
            None,
            true,
            affine,
        )
        .unwrap_err(),
        ops::gather_qmm(
            &activations,
            &quantized,
            &scales,
            None,
            None,
            None,
            true,
            affine,
            false,
        )
        .unwrap_err(),
    ] {
        let message = error.to_string();
        assert!(
            message.contains("requires biases"),
            "unexpected message: {message}"
        );
    }

    // The fp modes must not be given biases. MLX enforces this too, but only
    // after the FFI hop.
    let mxfp4 = QuantConfig::with_defaults(QuantMode::Mxfp4);
    for error in [
        ops::dequantize(&quantized, &scales, Some(&biases), mxfp4, None)
            .unwrap_err(),
        ops::quantized_matmul(
            &activations,
            &quantized,
            &scales,
            Some(&biases),
            true,
            mxfp4,
        )
        .unwrap_err(),
    ] {
        let message = error.to_string();
        assert!(
            message.contains("must not be given biases"),
            "unexpected message: {message}"
        );
    }

    // Not upstream: MLX silently honours a wrong fp layout rather than rejecting
    // it, which would quantize with the wrong group size.
    let wrong_layout = QuantConfig {
        group_size: 64,
        bits: 4,
        mode: QuantMode::Mxfp4,
    };
    for (expected, config) in [
        ("fixed at group_size 32", wrong_layout),
        ("positive group_size and bits", QuantConfig::affine(0, 4)),
    ] {
        let message = ops::quantize(&weights, config).unwrap_err().to_string();
        assert!(
            message.contains(expected),
            "expected {expected:?}, got: {message}"
        );
    }
}

/// Ports `test_throw`: shapes that do not line up must fail rather than compute
/// something.
///
/// Upstream builds its mismatches with `.T`, which this crate does not wrap, so
/// they are built here by quantizing a differently shaped matrix instead.
#[test]
fn throw_on_mismatched_shapes() {
    let config = QuantConfig::affine(64, 4);
    let x = Array::from_slice(&vec![0.1f32; 10 * 512], &[10, 512]).unwrap();
    let weights =
        Array::from_slice(&vec![0.1f32; 32 * 512], &[32, 512]).unwrap();
    let (quantized, scales, biases) = ops::quantize(&weights, config).unwrap();
    let biases = biases.unwrap();

    // transpose=false expects w to be [K, N] = [512, N]; it is [32, 512].
    assert!(
        ops::quantized_matmul(
            &x,
            &quantized,
            &scales,
            Some(&biases),
            false,
            config,
        )
        .is_err(),
        "transpose=false should reject a [32, 512] weight matrix"
    );

    // Scales and biases from a different matrix do not describe this one.
    let other = Array::from_slice(&vec![0.1f32; 64 * 256], &[64, 256]).unwrap();
    let (_, other_scales, other_biases) =
        ops::quantize(&other, config).unwrap();
    assert!(
        ops::quantized_matmul(
            &x,
            &quantized,
            &other_scales,
            other_biases.as_ref(),
            true,
            config,
        )
        .is_err(),
        "mismatched scales should be rejected"
    );

    // The shape that does line up must still work.
    let output = ops::quantized_matmul(
        &x,
        &quantized,
        &scales,
        Some(&biases),
        true,
        config,
    )
    .unwrap();
    assert_eq!(output.shape(), &[10, 32]);
}

/// Not upstream: every layout MLX can produce as a view must be refused or
/// materialized before a host read.
#[test]
fn host_reads_require_row_contiguous_layout() {
    let host: Vec<f32> = (0..64).map(|index| index as f32).collect();
    let grid = Array::from_slice(&host, &[8, 8]).unwrap();
    let line = Array::from_slice(&host[..16], &[16]).unwrap();

    // Shares the parent's buffer, so a flat read returns other columns' data.
    let strided = ops::slice(&grid, &[0, 0], &[8, 8], &[2, 2]).unwrap();
    assert_eq!(strided.shape(), &[4, 4]);

    // Points near the end of the buffer, so reading forward runs off it.
    let reversed = ops::slice(&line, &[15], &[-17], &[-1]).unwrap();
    assert_eq!(reversed.shape(), &[16]);

    let mut gathered = Vec::new();
    for row in (0..8).step_by(2) {
        for column in (0..8).step_by(2) {
            gathered.push(host[row * 8 + column]);
        }
    }
    let mut backwards = host[..16].to_vec();
    backwards.reverse();

    for (what, view, expected) in
        [("strided", &strided, gathered), ("reversed", &reversed, backwards)]
    {
        let error = view.to_vec_f32().unwrap_err().to_string();
        assert!(
            error.contains("not row-contiguous"),
            "{what} view was not refused: {error}"
        );
        assert_eq!(
            view.contiguous().unwrap().to_vec_f32().unwrap(),
            expected,
            "{what} view read back wrong after contiguous()"
        );
    }

    // A nonzero offset alone is fine: the pointer already includes it.
    let tail = ops::slice(&line, &[4], &[12], &[1]).unwrap();
    assert_eq!(tail.to_vec_f32().unwrap(), host[4..12].to_vec());
}

/// Not upstream: a host read must preserve the bits asked for, and must not fail
/// on an empty array.
#[test]
fn host_reads_preserve_dtype_and_size() {
    let (_, quantized, ..) = affine_triple();

    // f32 cannot represent a uint32 above 2^24, so the wrong dtype is refused
    // rather than reinterpreted.
    assert_eq!(quantized.dtype().unwrap(), Dtype::Uint32);
    assert_eq!(quantized.to_vec::<u32>().unwrap().len(), 256 * 256 / 8);
    let error = quantized.to_vec::<f32>().unwrap_err().to_string();
    assert!(error.contains("dtype is Uint32"), "unexpected message: {error}");

    // MLX returns a null data pointer here, which is not an error.
    let empty = Array::from_slice(&[] as &[f32], &[0]).unwrap();
    assert_eq!(empty.size(), 0);
    assert!(empty.to_vec::<f32>().unwrap().is_empty());
    assert!(empty.to_vec_f32().unwrap().is_empty());
}

/// Not upstream: none of these reach MLX, so none may be reported as an MLX
/// error.
#[test]
fn invalid_arguments_are_rejected_without_blaming_mlx() {
    let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let cases = [
        // Rejected, not clamped: clamping would make both sides zero.
        (
            "negative dimension",
            Array::from_slice(&[] as &[f32], &[-1]).unwrap_err(),
        ),
        // Release builds do not trap overflow, so the product is checked.
        (
            "overflows usize",
            Array::from_slice(&[1.0f32], &[i32::MAX, i32::MAX, i32::MAX])
                .unwrap_err(),
        ),
        (
            "implies 4 elements but 3 were given",
            Array::from_slice(&[1.0f32, 2.0, 3.0], &[2, 2]).unwrap_err(),
        ),
        (
            "equal length",
            ops::slice(&array, &[0, 0], &[2], &[1, 1]).unwrap_err(),
        ),
    ];

    for (expected, error) in cases {
        let message = error.to_string();
        assert!(
            message.contains(expected),
            "expected {expected:?}, got: {message}"
        );
        assert!(
            !message.starts_with("mlx:"),
            "our own precondition blames MLX: {message}"
        );
    }
}

/// Not upstream: the fp formats fix their own layouts, and
/// [`QuantMode::default_params`] is the only place that records them. Each mode's
/// defaults must also pass [`QuantConfig::validate`], or the table and the
/// validator disagree.
#[test]
fn the_mode_table_and_the_validator_agree() {
    assert_eq!(QuantMode::Affine.default_params(), (64, 4));
    assert_eq!(QuantMode::Mxfp4.default_params(), (32, 4));
    assert_eq!(QuantMode::Mxfp8.default_params(), (32, 8));
    assert_eq!(QuantMode::Nvfp4.default_params(), (16, 4));

    let (_, _, _, biases) = affine_triple();
    for mode in [
        QuantMode::Affine,
        QuantMode::Mxfp4,
        QuantMode::Mxfp8,
        QuantMode::Nvfp4,
    ] {
        assert_eq!(
            mode.uses_biases(),
            mode == QuantMode::Affine,
            "{mode:?} disagrees about taking biases"
        );
        let config = QuantConfig::with_defaults(mode);
        let stand_in = mode.uses_biases().then_some(&biases);
        config.validate(stand_in).unwrap_or_else(|error| {
            panic!("{mode:?} rejects its own defaults: {error}")
        });
    }
}
