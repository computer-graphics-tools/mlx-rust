//! Elementwise math, comparisons, bitwise ops and matrix products.

use super::macros::{binary_ops, unary_ops};

unary_ops! {
    /// Elementwise absolute value.
    abs_device => mlx_abs,
    /// Elementwise inverse cosine.
    arccos_device => mlx_arccos,
    /// Elementwise inverse hyperbolic cosine.
    arccosh_device => mlx_arccosh,
    /// Elementwise inverse sine.
    arcsin_device => mlx_arcsin,
    /// Elementwise inverse hyperbolic sine.
    arcsinh_device => mlx_arcsinh,
    /// Elementwise inverse tangent.
    arctan_device => mlx_arctan,
    /// Elementwise inverse hyperbolic tangent.
    arctanh_device => mlx_arctanh,
    /// Elementwise bitwise inverse.
    bitwise_invert_device => mlx_bitwise_invert,
    /// Elementwise ceiling.
    ceil_device => mlx_ceil,
    /// Elementwise complex conjugate.
    conjugate_device => mlx_conjugate,
    /// Elementwise cosine.
    cos_device => mlx_cos,
    /// Elementwise hyperbolic cosine.
    cosh_device => mlx_cosh,
    /// Convert angles from radians to degrees.
    degrees_device => mlx_degrees,
    /// Elementwise error function.
    erf_device => mlx_erf,
    /// Elementwise inverse error function.
    erfinv_device => mlx_erfinv,
    /// Elementwise exponential.
    exp_device => mlx_exp,
    /// Elementwise `exp(x) - 1`, accurate for small `x`.
    expm1_device => mlx_expm1,
    /// Elementwise floor.
    floor_device => mlx_floor,
    /// Imaginary part of a complex array.
    imag_device => mlx_imag,
    /// Elementwise finiteness test.
    isfinite_device => mlx_isfinite,
    /// Elementwise infinity test.
    isinf_device => mlx_isinf,
    /// Elementwise NaN test.
    isnan_device => mlx_isnan,
    /// Elementwise negative-infinity test.
    isneginf_device => mlx_isneginf,
    /// Elementwise positive-infinity test.
    isposinf_device => mlx_isposinf,
    /// Elementwise natural logarithm.
    log_device => mlx_log,
    /// Elementwise base-10 logarithm.
    log10_device => mlx_log10,
    /// Elementwise `log(1 + x)`, accurate for small `x`.
    log1p_device => mlx_log1p,
    /// Elementwise base-2 logarithm.
    log2_device => mlx_log2,
    /// Elementwise logical not.
    logical_not_device => mlx_logical_not,
    /// Elementwise negation.
    negative_device => mlx_negative,
    /// Convert angles from degrees to radians.
    radians_device => mlx_radians,
    /// Real part of a complex array.
    real_device => mlx_real,
    /// Elementwise reciprocal.
    reciprocal_device => mlx_reciprocal,
    /// Elementwise reciprocal square root.
    rsqrt_device => mlx_rsqrt,
    /// Elementwise logistic sigmoid.
    sigmoid_device => mlx_sigmoid,
    /// Elementwise sign.
    sign_device => mlx_sign,
    /// Elementwise sine.
    sin_device => mlx_sin,
    /// Elementwise hyperbolic sine.
    sinh_device => mlx_sinh,
    /// Elementwise square root.
    sqrt_device => mlx_sqrt,
    /// Elementwise square.
    square_device => mlx_square,
    /// Stop gradients from propagating through this array.
    stop_gradient_device => mlx_stop_gradient,
    /// Elementwise tangent.
    tan_device => mlx_tan,
    /// Elementwise hyperbolic tangent.
    tanh_device => mlx_tanh,
}

binary_ops! {
    /// Elementwise `a + b`, with broadcasting.
    add_device => mlx_add,
    /// Elementwise inverse tangent of `a / b`.
    arctan2_device => mlx_arctan2,
    /// Elementwise bitwise and.
    bitwise_and_device => mlx_bitwise_and,
    /// Elementwise bitwise or.
    bitwise_or_device => mlx_bitwise_or,
    /// Elementwise bitwise exclusive or.
    bitwise_xor_device => mlx_bitwise_xor,
    /// Elementwise `a / b`, with broadcasting.
    divide_device => mlx_divide,
    /// Elementwise `a == b`.
    equal_device => mlx_equal,
    /// Elementwise floor division.
    floor_divide_device => mlx_floor_divide,
    /// Elementwise `a > b`.
    greater_device => mlx_greater,
    /// Elementwise `a >= b`.
    greater_equal_device => mlx_greater_equal,
    /// Inner product of two arrays.
    inner_device => mlx_inner,
    /// Kronecker product of two arrays.
    kron_device => mlx_kron,
    /// Elementwise left bit shift.
    left_shift_device => mlx_left_shift,
    /// Elementwise `a < b`.
    less_device => mlx_less,
    /// Elementwise `a <= b`.
    less_equal_device => mlx_less_equal,
    /// Elementwise `log(exp(a) + exp(b))`, numerically stable.
    logaddexp_device => mlx_logaddexp,
    /// Elementwise logical and.
    logical_and_device => mlx_logical_and,
    /// Elementwise logical or.
    logical_or_device => mlx_logical_or,
    /// Matrix multiplication.
    matmul_device => mlx_matmul,
    /// Elementwise maximum.
    maximum_device => mlx_maximum,
    /// Elementwise minimum.
    minimum_device => mlx_minimum,
    /// Elementwise `a * b`, with broadcasting.
    multiply_device => mlx_multiply,
    /// Elementwise `a != b`.
    not_equal_device => mlx_not_equal,
    /// Outer product of two arrays.
    outer_device => mlx_outer,
    /// Elementwise `a` raised to the power `b`.
    power_device => mlx_power,
    /// Elementwise remainder of division.
    remainder_device => mlx_remainder,
    /// Elementwise right bit shift.
    right_shift_device => mlx_right_shift,
    /// Elementwise `a - b`, with broadcasting.
    subtract_device => mlx_subtract,
}
