//! Complex arrays, which `fft` returns and which had no host-side element type.

use approx::assert_relative_eq;
use mlx::{Array, Dtype, fft, ops};
use num_complex::Complex32;

/// The DFT of a known signal, computed on the host, against `fft`'s output read
/// back as `Complex32`. This is the whole point of the `Element` impl: before it,
/// `fft` results could not leave the device.
#[test]
fn fft_output_reads_back_as_complex_and_matches_a_host_dft() {
    let length = 8usize;
    let samples: Vec<f32> = (0..length).map(|index| index as f32).collect();
    let signal = Array::from_slice(&samples, &[length as i32]).unwrap();

    let spectrum = fft::fft(&signal, length as i32, None, None).unwrap();
    assert_eq!(spectrum.dtype().unwrap(), Dtype::Complex64);

    let got = spectrum.to_vec::<Complex32>().unwrap();
    assert_eq!(got.len(), length);

    for (bin, actual) in got.iter().enumerate() {
        let mut expected = Complex32::new(0.0, 0.0);
        for (index, &sample) in samples.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * (bin * index) as f32
                / length as f32;
            expected += Complex32::new(angle.cos(), angle.sin()) * sample;
        }
        assert_relative_eq!(actual.re, expected.re, epsilon = 1e-3);
        assert_relative_eq!(actual.im, expected.im, epsilon = 1e-3);
    }
}

#[test]
fn complex_arrays_round_trip_through_the_host() {
    let values = vec![
        Complex32::new(1.0, 2.0),
        Complex32::new(-3.0, 0.5),
        Complex32::new(0.0, -1.0),
    ];
    let array = Array::from_slice(&values, &[3]).unwrap();

    assert_eq!(array.dtype().unwrap(), Dtype::Complex64);
    assert_eq!(array.to_vec::<Complex32>().unwrap(), values);
    assert_eq!(array.as_slice::<Complex32>().unwrap(), values.as_slice());

    // `real` and `imag` must agree with the parts we put in.
    assert_eq!(
        ops::real(&array).unwrap().to_vec_f32().unwrap(),
        vec![1.0, -3.0, 0.0]
    );
    assert_eq!(
        ops::imag(&array).unwrap().to_vec_f32().unwrap(),
        vec![2.0, 0.5, -1.0]
    );
}
