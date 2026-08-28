//! Convolutions.

#![expect(clippy::too_many_arguments, reason = "mirrors mlx-c")]

use mlx_rust_macros::{default_device, generate_macro};

use crate::{array::Array, error::Result, stream::Stream};

/// 1-D convolution.
#[generate_macro]
#[default_device]
pub fn conv1d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv1d(
            result,
            input.handle,
            weight.handle,
            stride,
            padding,
            dilation,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// 2-D convolution.
#[generate_macro]
#[default_device]
pub fn conv2d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride_0: impl Into<Option<i32>>,
    #[optional] stride_1: impl Into<Option<i32>>,
    #[optional] padding_0: impl Into<Option<i32>>,
    #[optional] padding_1: impl Into<Option<i32>>,
    #[optional] dilation_0: impl Into<Option<i32>>,
    #[optional] dilation_1: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride_0 = stride_0.into().unwrap_or(1);
    let stride_1 = stride_1.into().unwrap_or(1);
    let padding_0 = padding_0.into().unwrap_or(0);
    let padding_1 = padding_1.into().unwrap_or(0);
    let dilation_0 = dilation_0.into().unwrap_or(1);
    let dilation_1 = dilation_1.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv2d(
            result,
            input.handle,
            weight.handle,
            stride_0,
            stride_1,
            padding_0,
            padding_1,
            dilation_0,
            dilation_1,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// 3-D convolution.
#[generate_macro]
#[default_device]
pub fn conv3d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride_0: impl Into<Option<i32>>,
    #[optional] stride_1: impl Into<Option<i32>>,
    #[optional] stride_2: impl Into<Option<i32>>,
    #[optional] padding_0: impl Into<Option<i32>>,
    #[optional] padding_1: impl Into<Option<i32>>,
    #[optional] padding_2: impl Into<Option<i32>>,
    #[optional] dilation_0: impl Into<Option<i32>>,
    #[optional] dilation_1: impl Into<Option<i32>>,
    #[optional] dilation_2: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride_0 = stride_0.into().unwrap_or(1);
    let stride_1 = stride_1.into().unwrap_or(1);
    let stride_2 = stride_2.into().unwrap_or(1);
    let padding_0 = padding_0.into().unwrap_or(0);
    let padding_1 = padding_1.into().unwrap_or(0);
    let padding_2 = padding_2.into().unwrap_or(0);
    let dilation_0 = dilation_0.into().unwrap_or(1);
    let dilation_1 = dilation_1.into().unwrap_or(1);
    let dilation_2 = dilation_2.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv3d(
            result,
            input.handle,
            weight.handle,
            stride_0,
            stride_1,
            stride_2,
            padding_0,
            padding_1,
            padding_2,
            dilation_0,
            dilation_1,
            dilation_2,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// 1-D transposed convolution.
#[generate_macro]
#[default_device]
pub fn conv_transpose1d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] output_padding: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let output_padding = output_padding.into().unwrap_or(0);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv_transpose1d(
            result,
            input.handle,
            weight.handle,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// 2-D transposed convolution.
#[generate_macro]
#[default_device]
pub fn conv_transpose2d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride_0: impl Into<Option<i32>>,
    #[optional] stride_1: impl Into<Option<i32>>,
    #[optional] padding_0: impl Into<Option<i32>>,
    #[optional] padding_1: impl Into<Option<i32>>,
    #[optional] dilation_0: impl Into<Option<i32>>,
    #[optional] dilation_1: impl Into<Option<i32>>,
    #[optional] output_padding_0: impl Into<Option<i32>>,
    #[optional] output_padding_1: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride_0 = stride_0.into().unwrap_or(1);
    let stride_1 = stride_1.into().unwrap_or(1);
    let padding_0 = padding_0.into().unwrap_or(0);
    let padding_1 = padding_1.into().unwrap_or(0);
    let dilation_0 = dilation_0.into().unwrap_or(1);
    let dilation_1 = dilation_1.into().unwrap_or(1);
    let output_padding_0 = output_padding_0.into().unwrap_or(0);
    let output_padding_1 = output_padding_1.into().unwrap_or(0);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv_transpose2d(
            result,
            input.handle,
            weight.handle,
            stride_0,
            stride_1,
            padding_0,
            padding_1,
            dilation_0,
            dilation_1,
            output_padding_0,
            output_padding_1,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// 3-D transposed convolution.
#[generate_macro]
#[default_device]
pub fn conv_transpose3d_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride_0: impl Into<Option<i32>>,
    #[optional] stride_1: impl Into<Option<i32>>,
    #[optional] stride_2: impl Into<Option<i32>>,
    #[optional] padding_0: impl Into<Option<i32>>,
    #[optional] padding_1: impl Into<Option<i32>>,
    #[optional] padding_2: impl Into<Option<i32>>,
    #[optional] dilation_0: impl Into<Option<i32>>,
    #[optional] dilation_1: impl Into<Option<i32>>,
    #[optional] dilation_2: impl Into<Option<i32>>,
    #[optional] output_padding_0: impl Into<Option<i32>>,
    #[optional] output_padding_1: impl Into<Option<i32>>,
    #[optional] output_padding_2: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let stride_0 = stride_0.into().unwrap_or(1);
    let stride_1 = stride_1.into().unwrap_or(1);
    let stride_2 = stride_2.into().unwrap_or(1);
    let padding_0 = padding_0.into().unwrap_or(0);
    let padding_1 = padding_1.into().unwrap_or(0);
    let padding_2 = padding_2.into().unwrap_or(0);
    let dilation_0 = dilation_0.into().unwrap_or(1);
    let dilation_1 = dilation_1.into().unwrap_or(1);
    let dilation_2 = dilation_2.into().unwrap_or(1);
    let output_padding_0 = output_padding_0.into().unwrap_or(0);
    let output_padding_1 = output_padding_1.into().unwrap_or(0);
    let output_padding_2 = output_padding_2.into().unwrap_or(0);
    let groups = groups.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv_transpose3d(
            result,
            input.handle,
            weight.handle,
            stride_0,
            stride_1,
            stride_2,
            padding_0,
            padding_1,
            padding_2,
            dilation_0,
            dilation_1,
            dilation_2,
            output_padding_0,
            output_padding_1,
            output_padding_2,
            groups,
            stream.as_ref().handle,
        )
    })
}

/// General N-D convolution.
#[generate_macro]
#[default_device]
pub fn conv_general_device(
    input: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    stride: &[i32],
    padding_lo: &[i32],
    padding_hi: &[i32],
    kernel_dilation: &[i32],
    input_dilation: &[i32],
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] flip: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let input = input.as_ref();
    let weight = weight.as_ref();
    let groups = groups.into().unwrap_or(1);
    let flip = flip.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_conv_general(
            result,
            input.handle,
            weight.handle,
            stride.as_ptr(),
            stride.len(),
            padding_lo.as_ptr(),
            padding_lo.len(),
            padding_hi.as_ptr(),
            padding_hi.len(),
            kernel_dilation.as_ptr(),
            kernel_dilation.len(),
            input_dilation.as_ptr(),
            input_dilation.len(),
            groups,
            flip,
            stream.as_ref().handle,
        )
    })
}
