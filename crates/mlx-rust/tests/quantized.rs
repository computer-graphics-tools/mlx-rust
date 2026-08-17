//! Line-by-line port of the quantized tests from MLX's own
//! `python/tests/test_quantized.py`, restricted to the ops this crate wraps.
//!
//! Each test keeps its upstream name, parameter lists (in upstream's order),
//! shapes, statement order and tolerances. Three MLX facilities this crate does
//! not wrap are replaced by host-side stand-ins, named after what they stand in
//! for so the correspondence stays readable:
//!
//! | upstream                    | here                |
//! |-----------------------------|---------------------|
//! | `mx.random.normal(shape=s)` | [`normal`]          |
//! | `x @ w` / `x @ w.T`         | [`matmul`]          |
//! | `(a - b).abs().max()`       | [`abs_max_diff`]    |
//!
//! `matmul` accumulates in `f64`, so it is a stricter oracle than upstream's,
//! which compares two `f32` results. Where that matters a test says so.

use half::{bf16, f16};
use mlx::{
    Array, Dtype, Element,
    ops::{self, QuantConfig},
};

/// Stands in for `mx.random.normal(shape=...)`: deterministic values in
/// `[-1, 1)`. Upstream draws from a normal distribution; only the scale matters
/// for these assertions.
fn normal(
    seed: u64,
    count: usize,
) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn array(
    values: &[f32],
    shape: &[i32],
) -> Array {
    Array::from_slice(values, shape).unwrap()
}

/// `values` re-encoded as `Value`, then uploaded. Upstream writes `.astype(dtype)`.
fn astype<Value: Element>(
    values: &[f32],
    shape: &[i32],
    encode: impl Fn(f32) -> Value,
) -> Array {
    let encoded: Vec<Value> = values.iter().copied().map(encode).collect();
    Array::from_slice(&encoded, shape).unwrap()
}

/// Stands in for `(left - right).abs().max()`.
fn abs_max_diff(
    left: &[f32],
    right: &[f32],
) -> f32 {
    assert_eq!(left.len(), right.len(), "shape mismatch");
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0f32, f32::max)
}

/// Stands in for `x @ w` (`transpose == false`) and `x @ w.T` (`true`),
/// accumulated in `f64`, over `batch` independent matrices.
///
/// `x` is `[batch, m, k]`. `w` is `[batch, n, k]` when transposed, `[batch, k,
/// n]` otherwise. A `batch` of 1 covers the unbatched case.
fn matmul(
    x: &[f32],
    w: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; batch * m * n];
    for index in 0..batch {
        let (x_base, w_base, out_base) =
            (index * m * k, index * n * k, index * m * n);
        for row in 0..m {
            for column in 0..n {
                let mut acc = 0f64;
                for inner in 0..k {
                    let w_value = if transpose {
                        w[w_base + column * k + inner]
                    } else {
                        w[w_base + inner * n + column]
                    };
                    acc += x[x_base + row * k + inner] as f64 * w_value as f64;
                }
                out[out_base + row * n + column] = acc as f32;
            }
        }
    }
    out
}

/// `mx.quantize(w, group_size, bits)`, affine.
fn quantize(
    w: &Array,
    group_size: i32,
    bits: i32,
) -> (Array, Array, Array) {
    let (w_q, scales, biases) =
        ops::quantize(w, QuantConfig::affine(group_size, bits)).unwrap();
    (w_q, scales, biases.expect("affine quantize produces biases"))
}

/// `mx.dequantize(w_q, scales, biases, group_size, bits)`, as `f32` on the host.
fn dequantize(
    w_q: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Vec<f32> {
    ops::dequantize(
        w_q,
        scales,
        Some(biases),
        QuantConfig::affine(group_size, bits),
        Some(Dtype::Float32),
    )
    .unwrap()
    .to_vec_f32()
    .unwrap()
}

#[test]
fn test_quantize_dequantize() {
    let w_values = normal(0, 128 * 512);
    let w = array(&w_values, &[128, 512]);
    for gs in [32, 64, 128] {
        for b in [2, 3, 5, 6, 4, 8] {
            let (w_q, scales, biases) = quantize(&w, gs, b);
            let w_hat = dequantize(&w_q, &scales, &biases, gs, b);

            // errors = (w - w_hat).abs().reshape(*scales.shape, -1)
            // assertTrue((errors <= (scales[..., None] + eps).abs()).all())
            let scales_host = scales.to_vec_f32().unwrap();
            let groups_per_row = 512 / gs as usize;
            let eps = 1e-6;
            for row in 0..128usize {
                for column in 0..512usize {
                    let index = row * 512 + column;
                    let group = row * groups_per_row + column / gs as usize;
                    let error = (w_values[index] - w_hat[index]).abs();
                    assert!(
                        error <= (scales_host[group] + eps).abs(),
                        "gs={gs} b={b}: [{row}][{column}] error {error:.3e} \
                         exceeds scale {:.3e}",
                        scales_host[group]
                    );
                }
            }
        }
    }

    // test quantize/dequantize 0s
    let a = array(&vec![0f32; 256 * 512], &[256, 512]);
    for gs in [32, 64, 128] {
        for b in [2, 3, 4, 5, 6, 8] {
            let (w_q, scales, biases) = quantize(&a, gs, b);
            let a_hat = dequantize(&w_q, &scales, &biases, gs, b);
            assert!(
                a_hat.iter().all(|&value| value == 0.0),
                "gs={gs} b={b}: zeros did not round-trip"
            );
        }
    }
}

#[test]
fn test_qmm() {
    // Upstream uses float16 on the GPU and float32 on the CPU.
    for group_size in [128, 64, 32] {
        for bits in [2, 4, 8] {
            for m in [8usize, 32, 33, 64] {
                for n in [128usize, 256] {
                    for k in [128usize, 256] {
                        for transposed in [true, false] {
                            let scale = 1.0 / (k as f32).sqrt();
                            let x_values: Vec<f32> = normal(1, m * k)
                                .iter()
                                .map(|value| value * scale)
                                .collect();
                            let (w_rows, w_columns) = if transposed {
                                (n, k)
                            } else {
                                (k, n)
                            };
                            let w_values: Vec<f32> =
                                normal(2, w_rows * w_columns)
                                    .iter()
                                    .map(|value| value * scale)
                                    .collect();

                            let x = astype(
                                &x_values,
                                &[m as i32, k as i32],
                                f16::from_f32,
                            );
                            let w = astype(
                                &w_values,
                                &[w_rows as i32, w_columns as i32],
                                f16::from_f32,
                            );
                            let (w_q, scales, biases) =
                                quantize(&w, group_size, bits);
                            let w_hat = dequantize(
                                &w_q, &scales, &biases, group_size, bits,
                            );
                            let y_q = ops::quantized_matmul(
                                &x,
                                &w_q,
                                &scales,
                                Some(&biases),
                                transposed,
                                QuantConfig::affine(group_size, bits),
                            )
                            .unwrap();
                            let y_hat = matmul(
                                &x.to_vec_f32().unwrap(),
                                &w_hat,
                                1,
                                m,
                                k,
                                n,
                                transposed,
                            );

                            assert_eq!(
                                y_q.shape(),
                                &[m as i32, n as i32],
                                "gs={group_size} b={bits} M={m} N={n} K={k} t={transposed}"
                            );
                            let error = abs_max_diff(
                                &y_q.to_vec_f32().unwrap(),
                                &y_hat,
                            );
                            assert!(
                                error < 1.5e-3,
                                "gs={group_size} b={bits} M={m} N={n} K={k} t={transposed}: \
             {error:.3e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_qmm_shapes() {
    let group_size = 64;
    let bits = 4;
    let w = array(&normal(2, 32 * 256), &[32, 256]);
    let (w_q, scales, biases) = quantize(&w, group_size, bits);
    let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits);
    for shape in [vec![3, 256], vec![2, 1, 7, 256]] {
        let rows: usize =
            shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let x = array(&normal(1, rows * 256), &shape);
        let y_q = ops::quantized_matmul(
            &x,
            &w_q,
            &scales,
            Some(&biases),
            true,
            QuantConfig::affine(group_size, bits),
        )
        .unwrap();
        let y_hat =
            matmul(&x.to_vec_f32().unwrap(), &w_hat, 1, rows, 256, 32, true);

        let mut expected_shape = shape.clone();
        *expected_shape.last_mut().unwrap() = 32;
        assert_eq!(y_q.shape(), expected_shape.as_slice());
        assert!(abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat) < 1e-3);
    }

    let w = array(&normal(2, 256 * 256), &[256, 256]);
    let (w_q, scales, biases) = quantize(&w, group_size, bits);
    let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits);
    for shape in [vec![3, 256], vec![2, 1, 7, 256]] {
        let rows: usize =
            shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let x = array(&normal(1, rows * 256), &shape);
        let y_q = ops::quantized_matmul(
            &x,
            &w_q,
            &scales,
            Some(&biases),
            false,
            QuantConfig::affine(group_size, bits),
        )
        .unwrap();
        let y_hat =
            matmul(&x.to_vec_f32().unwrap(), &w_hat, 1, rows, 256, 256, false);

        let mut expected_shape = shape.clone();
        *expected_shape.last_mut().unwrap() = 256;
        assert_eq!(y_q.shape(), expected_shape.as_slice());
        assert!(abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat) < 1e-3);
    }
}

#[test]
fn test_qmv() {
    for group_size in [128, 64, 32] {
        for bits in [2, 3, 4, 5, 6, 8] {
            for m in [256usize, 512, 67] {
                for n in [64usize, 256] {
                    for b in [0usize, 1, 3, 8] {
                        if group_size > n as i32 {
                            continue;
                        }
                        let batch = if b == 0 {
                            3
                        } else {
                            b
                        };
                        let w_batch = if b == 0 {
                            1
                        } else {
                            b
                        };
                        let x_shape = if b == 0 {
                            vec![3, 1, n as i32]
                        } else {
                            vec![b as i32, 1, n as i32]
                        };
                        let w_shape = if b == 0 {
                            vec![m as i32, n as i32]
                        } else {
                            vec![b as i32, m as i32, n as i32]
                        };

                        let scale = 1.0 / (n as f32).sqrt();
                        let x_values: Vec<f32> = normal(1, batch * n)
                            .iter()
                            .map(|value| value * scale)
                            .collect();
                        let w_values: Vec<f32> = normal(2, w_batch * m * n)
                            .iter()
                            .map(|value| value * scale)
                            .collect();
                        let x = array(&x_values, &x_shape);
                        let w = array(&w_values, &w_shape);

                        let (w_q, scales, biases) =
                            quantize(&w, group_size, bits);
                        let w_hat = dequantize(
                            &w_q, &scales, &biases, group_size, bits,
                        );
                        let y_q = ops::quantized_matmul(
                            &x,
                            &w_q,
                            &scales,
                            Some(&biases),
                            true,
                            QuantConfig::affine(group_size, bits),
                        )
                        .unwrap();

                        // y_hat = x @ swapaxes(w_hat, -1, -2); w_hat is broadcast
                        // across x's batch when b == 0.
                        let w_repeated = if b == 0 {
                            w_hat.repeat(batch)
                        } else {
                            w_hat.clone()
                        };
                        let y_hat = matmul(
                            &x_values,
                            &w_repeated,
                            batch,
                            1,
                            n,
                            m,
                            true,
                        );

                        assert_eq!(
                            y_q.shape(),
                            &[batch as i32, 1, m as i32],
                            "gs={group_size} b={bits} M={m} N={n} B={b}"
                        );
                        let error =
                            abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat);
                        assert!(
                            error < 1e-3,
                            "gs={group_size} b={bits} M={m} N={n} B={b}: \
                             {error:.3e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_qvm() {
    for group_size in [128, 64, 32] {
        for bits in [2, 3, 4, 5, 6, 8] {
            for m in [32usize, 128, 256] {
                for n in [128usize, 256, 67] {
                    for b in [0usize, 1, 3, 8] {
                        if (m as i32) < group_size {
                            continue;
                        }
                        let batch = if b == 0 {
                            1
                        } else {
                            b
                        };
                        let x_shape = if b == 0 {
                            vec![1, n as i32]
                        } else {
                            vec![b as i32, 1, n as i32]
                        };
                        let w_shape = if b == 0 {
                            vec![n as i32, m as i32]
                        } else {
                            vec![b as i32, n as i32, m as i32]
                        };

                        let x_values = normal(1, batch * n);
                        let w_values = normal(2, batch * n * m);
                        let x = array(&x_values, &x_shape);
                        let w = array(&w_values, &w_shape);

                        let (w_q, scales, biases) =
                            quantize(&w, group_size, bits);
                        let w_hat = dequantize(
                            &w_q, &scales, &biases, group_size, bits,
                        );
                        let y_q = ops::quantized_matmul(
                            &x,
                            &w_q,
                            &scales,
                            Some(&biases),
                            false,
                            QuantConfig::affine(group_size, bits),
                        )
                        .unwrap();
                        let y_hat =
                            matmul(&x_values, &w_hat, batch, 1, n, m, false);

                        let expected_shape = if b == 0 {
                            vec![1, m as i32]
                        } else {
                            vec![b as i32, 1, m as i32]
                        };
                        assert_eq!(
                            y_q.shape(),
                            expected_shape.as_slice(),
                            "gs={group_size} b={bits} M={m} N={n} B={b}"
                        );
                        let error =
                            abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat);
                        assert!(
                            error < 1e-3,
                            "gs={group_size} b={bits} M={m} N={n} B={b}: \
                             {error:.3e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_non_multiples() {
    // Upstream repeats the same four checks for w of 33, 3 and 99 rows.
    for (w_rows, x_rows_transposed, x_rows_plain) in
        [(33usize, 256usize, 33usize), (3, 256, 3), (99, 256, 99)]
    {
        let w = array(&normal(2, w_rows * 256), &[w_rows as i32, 256]);
        let (w_q, scales, biases) = quantize(&w, 64, 4);
        let w_hat = dequantize(&w_q, &scales, &biases, 64, 4);

        // Test qmv, then qmm_t: transpose=True, x is [rows, 256].
        for rows in [1usize, 10, 129] {
            let x = array(
                &normal(1, rows * x_rows_transposed),
                &[rows as i32, x_rows_transposed as i32],
            );
            let y_q = ops::quantized_matmul(
                &x,
                &w_q,
                &scales,
                Some(&biases),
                true,
                QuantConfig::affine(64, 4),
            )
            .unwrap();
            let y_hat = matmul(
                &x.to_vec_f32().unwrap(),
                &w_hat,
                1,
                rows,
                256,
                w_rows,
                true,
            );
            assert_eq!(y_q.shape(), &[rows as i32, w_rows as i32]);
            let error = abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat);
            assert!(
                error < 1e-3,
                "w_rows={w_rows} rows={rows} t=true: {error:.3e}"
            );
        }

        // Test qvm, then qmm: transpose=False, x is [rows, w_rows].
        for rows in [1usize, 10] {
            let x = array(
                &normal(1, rows * x_rows_plain),
                &[rows as i32, x_rows_plain as i32],
            );
            let y_q = ops::quantized_matmul(
                &x,
                &w_q,
                &scales,
                Some(&biases),
                false,
                QuantConfig::affine(64, 4),
            )
            .unwrap();
            let y_hat = matmul(
                &x.to_vec_f32().unwrap(),
                &w_hat,
                1,
                rows,
                w_rows,
                256,
                false,
            );
            assert_eq!(y_q.shape(), &[rows as i32, 256]);
            let error = abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat);
            assert!(
                error < 1e-3,
                "w_rows={w_rows} rows={rows} t=false: {error:.3e}"
            );
        }
    }
}

/// One entry of upstream's `inputs` table.
struct GatherCase {
    batch_a: usize,
    lhs_indices: Option<&'static [u32]>,
    batch_b: usize,
    rhs_indices: &'static [u32],
}

#[test]
fn test_gather_qmm() {
    // Upstream's `inputs` table, restricted to the entries whose batch dims are
    // one-dimensional. The two multi-dimensional entries and the fp-mode entries
    // need ops this crate does not wrap.
    let inputs = [
        GatherCase {
            batch_a: 1,
            lhs_indices: Some(&[0]),
            batch_b: 3,
            rhs_indices: &[2, 1],
        },
        GatherCase {
            batch_a: 1,
            lhs_indices: None,
            batch_b: 3,
            rhs_indices: &[2, 1],
        },
        GatherCase {
            batch_a: 2,
            lhs_indices: None,
            batch_b: 3,
            rhs_indices: &[2, 1],
        },
        GatherCase {
            batch_a: 3,
            lhs_indices: Some(&[0, 2]),
            batch_b: 1,
            rhs_indices: &[0],
        },
        GatherCase {
            batch_a: 5,
            lhs_indices: Some(&[0, 2]),
            batch_b: 3,
            rhs_indices: &[2, 1],
        },
    ];

    for GatherCase {
        batch_a,
        lhs_indices,
        batch_b,
        rhs_indices,
    } in inputs
    {
        for (m, n, k, transpose) in [
            (1usize, 32usize, 128usize, true),
            (32, 32, 256, true),
            (1, 32, 256, true),
            (32, 256, 32, false),
            (1, 256, 32, false),
            (32, 32, 512, true),
            (1, 32, 512, true),
            (32, 512, 32, false),
            (1, 512, 32, false),
        ] {
            check_gather_qmm(
                m,
                n,
                k,
                batch_a,
                lhs_indices,
                batch_b,
                rhs_indices,
                transpose,
            );
        }
    }
}

/// Upstream's inner `test_shape` helper.
#[expect(clippy::too_many_arguments, reason = "mirrors upstream's test_shape")]
fn check_gather_qmm(
    m: usize,
    n: usize,
    k: usize,
    batch_a: usize,
    lhs_indices: Option<&[u32]>,
    batch_b: usize,
    rhs_indices: &[u32],
    transpose: bool,
) {
    let group_size = 64;
    let bits = 4;
    let label = format!(
        "M={m} N={n} K={k} batch_A={batch_a} batch_B={batch_b} \
         transpose={transpose}"
    );

    // Upstream leaves these unscaled because it compares against `gather_mm`,
    // which carries the same f32 accumulation error. This oracle is f64, so the
    // inputs are normalized by 1/sqrt(K) -- as upstream's other tests do -- to
    // keep its `atol=1e-4` meaningful rather than relaxing the tolerance.
    let scale = 1.0 / (k as f32).sqrt();
    let x_values: Vec<f32> =
        normal(1, batch_a * m * k).iter().map(|value| value * scale).collect();
    let x = array(&x_values, &[batch_a as i32, m as i32, k as i32]);

    let (w_rows, w_columns) = if transpose {
        (n, k)
    } else {
        (k, n)
    };
    let w_values: Vec<f32> = normal(2, batch_b * w_rows * w_columns)
        .iter()
        .map(|value| value * scale)
        .collect();
    let w =
        array(&w_values, &[batch_b as i32, w_rows as i32, w_columns as i32]);
    let (w_q, scales, biases) = quantize(&w, group_size, bits);
    let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits);

    let lhs =
        lhs_indices.map(|indices| array_u32(indices, &[indices.len() as i32]));
    let rhs = array_u32(rhs_indices, &[rhs_indices.len() as i32]);

    let c2 = ops::gather_qmm(
        &x,
        &w_q,
        &scales,
        Some(&biases),
        lhs.as_ref(),
        Some(&rhs),
        transpose,
        QuantConfig::affine(group_size, bits),
        false,
    )
    .unwrap();

    // c1 = gather_mm(x, w_hat, lhs_indices, rhs_indices), on the host. The two
    // index arrays broadcast against each other, so a length-1 one repeats.
    let broadcast = |indices: &[u32], slot: usize| {
        let value = if indices.len() == 1 {
            indices[0]
        } else {
            indices[slot]
        };
        value as usize
    };
    let output_batch =
        lhs_indices.map_or(batch_a, <[u32]>::len).max(rhs_indices.len());
    let mut gathered_x = Vec::with_capacity(output_batch * m * k);
    let mut gathered_w = Vec::with_capacity(output_batch * w_rows * w_columns);
    for slot in 0..output_batch {
        let left = match lhs_indices {
            Some(indices) => broadcast(indices, slot),
            None if batch_a == 1 => 0,
            None => slot,
        };
        let right = broadcast(rhs_indices, slot);
        let x_stride = m * k;
        let w_stride = w_rows * w_columns;
        gathered_x.extend_from_slice(
            &x_values[left * x_stride..(left + 1) * x_stride],
        );
        gathered_w.extend_from_slice(
            &w_hat[right * w_stride..(right + 1) * w_stride],
        );
    }
    let c1 = matmul(&gathered_x, &gathered_w, output_batch, m, k, n, transpose);

    // allclose(c1, c2, atol=1e-4)
    let error = abs_max_diff(&c2.to_vec_f32().unwrap(), &c1);
    assert!(error < 1e-4, "{label}: {error:.3e}");
}

fn array_u32(
    values: &[u32],
    shape: &[i32],
) -> Array {
    Array::from_slice(values, shape).unwrap()
}

/// Not upstream: this crate compiles MLX itself, so it can produce a metallib
/// without the kernels under test. `affine_qmv_wide` is MLX's small-M path and
/// `*_nax` the Metal 4 tensor-op kernels, both needing a deployment target of at
/// least 26.2.
#[cfg(feature = "metal")]
#[test]
fn build_is_usable_for_measurement() {
    assert!(mlx::metal::is_available(), "no Metal device");
    assert!(!mlx::mlx_version().unwrap().is_empty(), "MLX reported no version");
    assert!(
        mlx::metal::memory_stats().unwrap().limit > 0,
        "allocator reported a zero limit"
    );

    let metallib = std::fs::read(mlx::metal::metallib_path())
        .expect("metallib should be readable at the recorded path");
    for kernel in [
        "affine_qmv_fast_bfloat16_t",
        "affine_qmv_wide_bfloat16_t",
        "affine_qmm_t_bfloat16_t",
        "affine_qmm_t_nax_bfloat16_t",
    ] {
        assert!(
            metallib
                .windows(kernel.len())
                .any(|window| window == kernel.as_bytes()),
            "metallib is missing `{kernel}`"
        );
    }
}

/// Not upstream: pins the affine packing this crate documents, which the Python
/// suite exercises only through `dequantize`. Bit widths 3, 5 and 6 use a
/// different layout and are not checkable this way.
#[test]
fn affine_dequantize_is_scale_times_code_plus_bias() {
    let (rows, columns, group_size) = (8usize, 128usize, 64i32);
    for bits in [4i32, 8] {
        let w_values = normal(0xABCD, rows * columns);
        let w = array(&w_values, &[rows as i32, columns as i32]);
        let (w_q, scales, biases) = quantize(&w, group_size, bits);
        let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits);

        // Read the packed weights as u32; f32 drops bits above 2^24.
        let packed = w_q.to_vec::<u32>().unwrap();
        let scales_host = scales.to_vec_f32().unwrap();
        let biases_host = biases.to_vec_f32().unwrap();

        let codes_per_word = (32 / bits) as usize;
        let groups_per_row = columns / group_size as usize;
        let words_per_row = columns / codes_per_word;
        let mask = (1u32 << bits) - 1;

        for row in 0..rows {
            for column in 0..columns {
                let word =
                    packed[row * words_per_row + column / codes_per_word];
                let shift = bits as usize * (column % codes_per_word);
                let code = (word >> shift) & mask;
                let group = row * groups_per_row + column / group_size as usize;
                let expected =
                    code as f32 * scales_host[group] + biases_host[group];
                let actual = w_hat[row * columns + column];
                assert!(
                    (expected - actual).abs() <= 1e-6,
                    "bits {bits}: [{row}][{column}] is {actual}, expected \
                     scale*code + bias = {expected}"
                );
            }
        }
    }
}

/// Not upstream: bf16 is the dtype this crate's callers measure, and it runs a
/// different kernel family than the f16 `test_qmm` uses.
#[test]
fn qmm_matches_dequantized_matmul_in_bf16() {
    let (m, n, k) = (32usize, 128usize, 128usize);
    let scale = 1.0 / (k as f32).sqrt();
    let x_values: Vec<f32> =
        normal(1, m * k).iter().map(|value| value * scale).collect();
    let w_values: Vec<f32> =
        normal(2, n * k).iter().map(|value| value * scale).collect();

    let x = astype(&x_values, &[m as i32, k as i32], bf16::from_f32);
    let w = astype(&w_values, &[n as i32, k as i32], bf16::from_f32);
    let (w_q, scales, biases) = quantize(&w, 64, 4);
    let w_hat = dequantize(&w_q, &scales, &biases, 64, 4);

    let y_q = ops::quantized_matmul(
        &x,
        &w_q,
        &scales,
        Some(&biases),
        true,
        QuantConfig::affine(64, 4),
    )
    .unwrap();
    let y_hat = matmul(&x.to_vec_f32().unwrap(), &w_hat, 1, m, k, n, true);

    // bf16 carries 8 mantissa bits, against f16's 10.
    let error = abs_max_diff(&y_q.to_vec_f32().unwrap(), &y_hat);
    assert!(error < 1e-2, "bf16: {error:.3e}");
}
