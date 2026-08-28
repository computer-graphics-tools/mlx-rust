//! The elementwise, shape and sort ops, and the `std::ops` impls over them.
//!
//! To be grown into a port of MLX's `python/tests/test_ops.py` as the op surface
//! fills in; for now it checks that each wrapped family computes and that the
//! generated `_device` twins agree with the defaulted forms.

use mlx::{Array, ops};
#[test]
fn newly_wrapped_ops_work() {
    let a = Array::from_slice(&[1.0f32, 4.0, 9.0, 16.0], &[4]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

    // unary, default stream + explicit twin
    assert_eq!(
        ops::sqrt(&a).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        ops::sqrt_device(&a, mlx::Stream::gpu()).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        ops::negative(&b).unwrap().to_vec_f32().unwrap(),
        vec![-1.0, -2.0, -3.0, -4.0]
    );
    assert_eq!(
        ops::square(&b).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 4.0, 9.0, 16.0]
    );

    // binary
    assert_eq!(
        ops::add(&a, &b).unwrap().to_vec_f32().unwrap(),
        vec![2.0, 6.0, 12.0, 20.0]
    );
    assert_eq!(
        ops::subtract(&a, &b).unwrap().to_vec_f32().unwrap(),
        vec![0.0, 2.0, 6.0, 12.0]
    );
    assert_eq!(
        ops::maximum(&a, &b).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 4.0, 9.0, 16.0]
    );
    assert_eq!(
        ops::power(&b, &b).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 4.0, 27.0, 256.0]
    );

    // matmul
    let m = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(
        ops::matmul(&m, &m).unwrap().to_vec_f32().unwrap(),
        vec![7.0, 10.0, 15.0, 22.0]
    );

    // shapes / sort
    let unsorted = Array::from_slice(&[3.0f32, 1.0, 2.0], &[3]).unwrap();
    assert_eq!(
        ops::sort(&unsorted).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        ops::zeros_like(&b).unwrap().to_vec_f32().unwrap(),
        vec![0.0; 4]
    );
    let wide = Array::from_slice(&[1.0f32, 2.0], &[1, 2]).unwrap();
    assert_eq!(ops::transpose(&wide).unwrap().shape(), &[2, 1]);
    assert_eq!(ops::squeeze(&wide).unwrap().shape(), &[2]);
}

#[test]
fn operators_match_the_fallible_ops() {
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let b = Array::from_slice(&[4.0f32, 3.0, 2.0, 1.0], &[4]).unwrap();

    assert_eq!((&a + &b).to_vec_f32().unwrap(), vec![5.0; 4]);
    assert_eq!((&a - &b).to_vec_f32().unwrap(), vec![-3.0, -1.0, 1.0, 3.0]);
    assert_eq!((&a * &b).to_vec_f32().unwrap(), vec![4.0, 6.0, 6.0, 4.0]);
    assert_eq!(
        (&a / &b).to_vec_f32().unwrap(),
        vec![0.25, 2.0 / 3.0, 1.5, 4.0]
    );
    assert_eq!((-&a).to_vec_f32().unwrap(), vec![-1.0, -2.0, -3.0, -4.0]);

    // Chaining borrows, as MLX's Python examples do.
    let chained = &(&a + &b) * &a;
    assert_eq!(chained.to_vec_f32().unwrap(), vec![5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn reductions_reduce_over_axes_and_wholes() {
    // [[1, 2], [3, 4]]
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(
        ops::sum(&a, None, None).unwrap().to_vec_f32().unwrap(),
        vec![10.0]
    );
    assert_eq!(
        ops::mean(&a, None, None).unwrap().to_vec_f32().unwrap(),
        vec![2.5]
    );
    assert_eq!(
        ops::prod(&a, None, None).unwrap().to_vec_f32().unwrap(),
        vec![24.0]
    );
    assert_eq!(
        ops::max(&a, None, None).unwrap().to_vec_f32().unwrap(),
        vec![4.0]
    );
    assert_eq!(
        ops::min(&a, None, None).unwrap().to_vec_f32().unwrap(),
        vec![1.0]
    );

    // Per-axis, and keepdims.
    assert_eq!(
        ops::sum(&a, &[0i32][..], None).unwrap().to_vec_f32().unwrap(),
        vec![4.0, 6.0]
    );
    assert_eq!(
        ops::sum(&a, &[1i32][..], None).unwrap().to_vec_f32().unwrap(),
        vec![3.0, 7.0]
    );
    assert_eq!(ops::sum(&a, &[1i32][..], true).unwrap().shape(), &[2, 1]);
    assert_eq!(ops::sum(&a, &[1i32][..], false).unwrap().shape(), &[2]);

    assert_eq!(
        ops::argmax(&a, None, None).unwrap().to_vec::<u32>().unwrap(),
        vec![3]
    );
    assert_eq!(
        ops::argmax(&a, 1i32, None).unwrap().to_vec::<u32>().unwrap(),
        vec![1, 1]
    );
    assert_eq!(
        ops::var(&a, None, None, None).unwrap().to_vec_f32().unwrap(),
        vec![1.25]
    );
}

#[test]
fn factory_constructors_build_without_host_data() {
    assert_eq!(
        ops::zeros(&[2, 2], None).unwrap().to_vec_f32().unwrap(),
        vec![0.0; 4]
    );
    assert_eq!(
        ops::ones(&[3], None).unwrap().to_vec_f32().unwrap(),
        vec![1.0; 3]
    );
    assert_eq!(
        ops::arange(0.0, 5.0, None, None).unwrap().to_vec_f32().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        ops::linspace(0.0, 1.0, 3i32, None).unwrap().to_vec_f32().unwrap(),
        vec![0.0, 0.5, 1.0]
    );
    assert_eq!(
        ops::eye(2, None, None, None).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(ops::identity(3, None).unwrap().shape(), &[3, 3]);
    let sevens = Array::from_slice(&[7.0f32], &[1]).unwrap();
    assert_eq!(
        ops::full(&[2], &sevens, None).unwrap().to_vec_f32().unwrap(),
        vec![7.0, 7.0]
    );
}

#[test]
fn shape_ops_rearrange() {
    let a = ops::arange(0.0, 6.0, None, None).unwrap();

    assert_eq!(ops::reshape(&a, &[2, 3]).unwrap().shape(), &[2, 3]);
    assert_eq!(ops::expand_dims(&a, &[0]).unwrap().shape(), &[1, 6]);
    assert_eq!(
        ops::flatten(ops::reshape(&a, &[2, 3]).unwrap(), None, None)
            .unwrap()
            .shape(),
        &[6]
    );
    assert_eq!(
        ops::swapaxes(ops::reshape(&a, &[2, 3]).unwrap(), 0, 1)
            .unwrap()
            .shape(),
        &[3, 2]
    );
    assert_eq!(
        ops::transpose_axes(ops::reshape(&a, &[2, 3]).unwrap(), &[1, 0])
            .unwrap()
            .shape(),
        &[3, 2]
    );
    assert_eq!(
        ops::broadcast_to(ops::ones(&[1], None).unwrap(), &[3])
            .unwrap()
            .shape(),
        &[3]
    );

    let left = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let right = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
    assert_eq!(
        ops::concatenate(&[&left, &right], None).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(ops::stack(&[&left, &right], None).unwrap().shape(), &[2, 2]);

    let parts = ops::split(&a, 3, None).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_vec_f32().unwrap(), vec![0.0, 1.0]);
}

#[test]
fn random_draws_are_reproducible_from_a_key() {
    use mlx::random;

    let key = random::key(0).unwrap();
    let first = random::normal(&[4], None, None, None, &key).unwrap();
    let again = random::normal(&[4], None, None, None, &key).unwrap();
    assert_eq!(
        first.to_vec_f32().unwrap(),
        again.to_vec_f32().unwrap(),
        "the same key must give the same draw"
    );

    // Split keys must diverge.
    let (left, right) = random::split(&key).unwrap();
    assert_ne!(
        random::normal(&[4], None, None, None, &left)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        random::normal(&[4], None, None, None, &right)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
    );

    let low = Array::from_slice(&[0.0f32], &[1]).unwrap();
    let high = Array::from_slice(&[1.0f32], &[1]).unwrap();
    let drawn = random::uniform(&low, &high, &[64], None, &key).unwrap();
    assert!(
        drawn.to_vec_f32().unwrap().iter().all(|&v| (0.0..1.0).contains(&v))
    );

    assert_eq!(random::gumbel(&[4], None, &key).unwrap().shape(), &[4]);
}

#[test]
fn newly_wrapped_families_compute() {
    use mlx::{fft, linalg};

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    // logical / selection
    let mask = Array::from_slice(&[true, false, true, false], &[2, 2]).unwrap();
    let zeros = ops::zeros(&[2, 2], None).unwrap();
    assert_eq!(
        ops::select(&mask, &a, &zeros).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 0.0, 3.0, 0.0]
    );
    assert!(
        ops::allclose(&a, &a, None, None, None)
            .unwrap()
            .to_vec::<bool>()
            .unwrap()[0]
    );

    // cumulative / ordering / manipulation
    let row = ops::arange(1.0, 5.0, None, None).unwrap();
    assert_eq!(
        ops::cumsum(&row, 0, None, None).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 3.0, 6.0, 10.0]
    );
    assert_eq!(ops::topk(&row, 2).unwrap().to_vec_f32().unwrap().len(), 2);
    assert_eq!(ops::tile(&row, &[2]).unwrap().shape(), &[8]);
    assert_eq!(ops::repeat(&row, 2).unwrap().shape(), &[8]);
    assert_eq!(
        ops::tril(&a, None).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 0.0, 3.0, 4.0]
    );
    assert_eq!(ops::round(&row, None).unwrap().shape(), &[4]);
    assert_eq!(
        ops::softmax(&row, None)
            .unwrap()
            .to_vec_f32()
            .unwrap()
            .iter()
            .sum::<f32>()
            .round(),
        1.0
    );

    // indexing
    let indices = Array::from_slice(&[0u32, 2], &[2]).unwrap();
    assert_eq!(
        ops::take(&row, &indices).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 3.0]
    );

    // linear algebra
    let identity = ops::eye(2, None, None, None).unwrap();
    assert_eq!(
        linalg::inv(&identity).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        linalg::norm_l2(&row, None, None).unwrap().to_vec_f32().unwrap().len(),
        1
    );
    let (q, r) = linalg::qr(&a).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(linalg::svd(&a, None).unwrap().len(), 3);

    // matrix products and convolution
    assert_eq!(ops::addmm(&a, &a, &a, None, None).unwrap().shape(), &[2, 2]);
    assert_eq!(
        ops::tensordot_axis(&a, &a, 1).unwrap().shape(),
        ops::matmul(&a, &a).unwrap().shape()
    );

    // fft round-trips
    let signal = ops::arange(0.0, 8.0, None, None).unwrap();
    let spectrum = fft::fft(&signal, 8, None, None).unwrap();
    let restored = fft::ifft(&spectrum, 8, None, None).unwrap();
    let restored = ops::real(&restored).unwrap().to_vec_f32().unwrap();
    for (index, value) in restored.iter().enumerate() {
        assert!((value - index as f32).abs() < 1e-4, "ifft(fft(x)) != x");
    }
}

#[test]
fn save_and_load_round_trip() {
    use std::collections::HashMap;

    let directory = std::env::temp_dir().join("mlx_rust_io_test");
    std::fs::create_dir_all(&directory).unwrap();

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let npy = directory.join("one.npy");
    mlx::io::save(&npy, &a).unwrap();
    assert_eq!(
        mlx::io::load(&npy).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0]
    );

    let mut arrays = HashMap::new();
    arrays.insert("weight".to_string(), a);
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "test".to_string());

    let safetensors = directory.join("weights.safetensors");
    mlx::io::save_safetensors(&safetensors, &arrays, &metadata).unwrap();
    let (loaded, loaded_metadata) =
        mlx::io::load_safetensors(&safetensors).unwrap();
    assert_eq!(
        loaded["weight"].to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0],
        "safetensors round-trip lost the array"
    );
    assert_eq!(loaded_metadata.get("format").map(String::as_str), Some("test"));

    std::fs::remove_dir_all(&directory).ok();
}
