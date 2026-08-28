//! Autodiff: gradients of known functions, checked against their analytic form.

use mlx::{Array, Result, ops, transforms};

/// A 0-d array. `grad` requires the differentiated output to be a scalar, which
/// means shape `[]` rather than `[1]`.
fn scalar(value: f32) -> Array {
    Array::from_slice(&[value], &[]).unwrap()
}

#[test]
fn grad_of_square_is_two_x() {
    // d/dx x^2 = 2x
    let gradient = transforms::grad(
        |inputs: &[Array]| Ok(vec![ops::square(&inputs[0])?]),
        &[0],
    )
    .unwrap();

    for x in [1.0f32, 3.0, -2.5] {
        let got = gradient(&[&scalar(x)]).unwrap();
        assert_eq!(got.len(), 1);
        let got = got[0].to_vec_f32().unwrap()[0];
        assert!((got - 2.0 * x).abs() < 1e-5, "d/dx x^2 at {x}: {got}");
    }
}

#[test]
fn value_and_grad_returns_both() {
    let transformed = transforms::value_and_grad(
        |inputs: &[Array]| {
            Ok(vec![ops::sum(&ops::square(&inputs[0])?, None, None)?])
        },
        &[0],
    )
    .unwrap();

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let (values, gradients) = transformed.apply(&[&x]).unwrap();

    // sum(x^2) = 14, d/dx = 2x
    assert_eq!(values[0].to_vec_f32().unwrap(), vec![14.0]);
    assert_eq!(gradients[0].to_vec_f32().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn grad_of_exp_is_exp() {
    let gradient = transforms::grad(
        |inputs: &[Array]| Ok(vec![ops::exp(&inputs[0])?]),
        &[0],
    )
    .unwrap();
    let got = gradient(&[&scalar(0.0)]).unwrap()[0].to_vec_f32().unwrap()[0];
    assert!((got - 1.0).abs() < 1e-6, "d/dx exp at 0: {got}");
}

#[test]
fn grad_with_respect_to_the_second_argument() {
    // f(a, b) = a * b; df/db = a
    let gradient = transforms::grad(
        |inputs: &[Array]| Ok(vec![ops::multiply(&inputs[0], &inputs[1])?]),
        &[1],
    )
    .unwrap();
    let got = gradient(&[&scalar(3.0), &scalar(5.0)]).unwrap()[0]
        .to_vec_f32()
        .unwrap()[0];
    assert!((got - 3.0).abs() < 1e-6, "df/db should be a = 3, got {got}");
}

#[test]
fn vjp_and_jvp_agree_with_the_derivative() {
    let x = scalar(4.0);
    let ones = scalar(1.0);

    // d/dx sqrt(x) = 1/(2 sqrt(x)) = 0.25 at x = 4
    let (outputs, cotangents) = transforms::vjp(
        |inputs: &[Array]| Ok(vec![ops::sqrt(&inputs[0])?]),
        &[&x],
        &[&ones],
    )
    .unwrap();
    assert!((outputs[0].to_vec_f32().unwrap()[0] - 2.0).abs() < 1e-6);
    assert!((cotangents[0].to_vec_f32().unwrap()[0] - 0.25).abs() < 1e-6);

    let (outputs, tangents) = transforms::jvp(
        |inputs: &[Array]| Ok(vec![ops::sqrt(&inputs[0])?]),
        &[&x],
        &[&ones],
    )
    .unwrap();
    assert!((outputs[0].to_vec_f32().unwrap()[0] - 2.0).abs() < 1e-6);
    assert!((tangents[0].to_vec_f32().unwrap()[0] - 0.25).abs() < 1e-6);
}

/// An error inside the closure must surface as an error, not a crash or a wrong
/// number: the trampoline returns non-zero rather than unwinding into C++.
#[test]
fn a_failing_closure_surfaces_as_an_error() {
    let gradient = transforms::grad(
        |_: &[Array]| -> Result<Vec<Array>> {
            Err(mlx::Error::Invalid("deliberate".into()))
        },
        &[0],
    )
    .unwrap();
    assert!(gradient(&[&scalar(1.0)]).is_err(), "the error was swallowed");
}

#[test]
fn compiled_functions_agree_with_the_uncompiled_ones() {
    let compiled = transforms::compile(
        |inputs: &[Array]| Ok(vec![ops::multiply(&inputs[0], &inputs[0])?]),
        false,
    )
    .unwrap();

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    // Repeat, so the cached graph is exercised as well as the first trace.
    for _ in 0..3 {
        let outputs = compiled.apply(&[&x]).unwrap();
        assert_eq!(outputs[0].to_vec_f32().unwrap(), vec![1.0, 4.0, 9.0]);
    }
}

#[test]
fn a_plain_closure_round_trips_through_mlx() {
    let closure = transforms::Closure::new(|inputs: &[Array]| {
        Ok(vec![ops::add(&inputs[0], &inputs[1])?])
    });
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&[10.0f32, 20.0], &[2]).unwrap();
    let outputs = closure.apply(&[&a, &b]).unwrap();
    assert_eq!(outputs[0].to_vec_f32().unwrap(), vec![11.0, 22.0]);
}
