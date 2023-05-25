import numpy as np
from scipy.signal import fftconvolve as _fftconvolve
from torch import Tensor
from torch.nn.functional import unfold

EPS = 1e-10


def softmax(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    exp = np.exp(X)
    logits = exp / exp.sum(axis=0, keepdims=True)
    return logits


def softmax_diff(X: np.ndarray) -> np.ndarray:
    num_features = X.shape[0]
    p = softmax(X)
    # See https://themaverickmeerkat.com/2019-10-23-Softmax/
    outer = np.einsum("ji,ki->ijk", p, p)
    diff = np.einsum("ij,jk->kji", np.eye(num_features), p)
    grads = diff - outer
    return grads


def sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(X)) + EPS)


def sigmoid_diff(X: np.ndarray) -> np.ndarray:
    s = sigmoid(np.asarray(X))
    return s * (1 - s)


def tanh(X: np.ndarray) -> np.ndarray:
    return np.tanh(X)


def tanh_diff(X: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(np.asarray(X)) ** 2


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    loss = 0.5 * (np.linalg.norm(y_hat - y, axis=0) ** 2).mean()
    if np.isnan(loss):
        raise ValueError("loss is nan!")
    return loss


def mse_loss_diff(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y.shape[1]
    return (y_hat - y) / m


def cross_entropy_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Compute the cross-entropy loss.

    The output cross-entropy of shape `NxM`.
    """
    assert y_hat.shape == y.shape
    loss = (-np.asarray(y) * np.log(np.asarray(y_hat))).sum(axis=0).mean()
    if np.isnan(loss):
        raise ValueError("loss is nan!")
    return loss


def cross_entropy_loss_diff(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the derivative of the cross-entropy loss.

    Derivative of cross-entropy of shape `NxM`.
    """
    assert y_hat.shape == y.shape
    m = y.shape[1]
    diff = -np.asarray(y) / y_hat / m
    return diff


def relu(X: np.ndarray) -> np.ndarray:
    return np.maximum(X, 0.0)


def relu_diff(X: np.ndarray) -> np.ndarray:
    return (X >= 0.0).astype(float)


def get_conv_output_size(size, kernel_size, padding_size, stride_size):
    return (size - kernel_size + 2 * padding_size) // stride_size + 1


def get_conv_input_size(size, kernel_size, padding_size, stride_size):
    return (size - 1) * stride_size + kernel_size - 2 * padding_size


def fft_tensor_convolution(
    x: np.ndarray,
    y: np.ndarray,
    padding: int,
    stride: int,
    mode: str = "valid",
    is_correlation: bool = True,
) -> np.ndarray:
    if is_correlation:
        y = y[:, :, ::-1, ::-1]

    num_data = x.shape[0]
    input_channels = x.shape[1]
    input_width = x.shape[2]
    input_height = x.shape[3]

    output_channels = y.shape[0]
    assert y.shape[1] == input_channels
    kernel_size = y.shape[2]

    output_width = get_conv_output_size(
        input_width,
        kernel_size,
        padding,
        stride,
    )
    output_height = get_conv_output_size(
        input_height,
        kernel_size,
        padding,
        stride,
    )

    output = np.zeros(
        (
            num_data,
            output_channels,
            output_width,
            output_height,
        ),
        dtype=x.dtype,
    )

    if padding:
        x = np.pad(
            x,
            [
                (0, 0),
                (0, 0),
                (padding, padding),
                (padding, padding),
            ],
            mode="constant",
        )

    for m in range(num_data):
        for i in range(input_channels):
            for j in range(output_channels):
                convolved = _fftconvolve(
                    x[m, i],
                    y[j, i],
                    mode="valid",
                )[::stride, ::stride]
                output[m, j] += convolved
    return output


def gemm_tensor_convolution(
    x: np.ndarray,
    y: np.ndarray,
    padding: int,
    stride: int,
    mode: str = "valid",  # TODO
    is_correlation: bool = True,
):
    """Convolution using GEMM.

    This method transforms the convolution into a matrix multiplication
    by "unfolding" input tensor into a matrix.

    Args:
        x (np.ndarray):
            4-D tensor, shape (M, C, H, W).
        y (np.ndarray):
            4-D tensor, shape (C', C, K, K),
            here C is the number of input channels and
            C' is the number of output channels.
        padding (int): padding size.
        stride (int): stride size.
        mode (str, optional): convolution mode. Defaults to "valid". (TODO)
        is_correlation (bool, optional): whether to perform convolution or correlation. Defaults to True.

    Returns:
        np.ndarray: 4-D tensor, shape (M, C', H', W').
    """
    if not is_correlation:
        # NOTE: If it's convolution, flip the kernel.
        y = y[:, :, ::-1, ::-1]

    num_data = x.shape[0]
    input_channels = x.shape[1]
    input_width = x.shape[2]
    input_height = x.shape[3]

    output_channels = y.shape[0]
    input_filter_channels = y.shape[1]
    kernel_size = y.shape[2]
    assert input_filter_channels == input_channels

    conv_width = get_conv_output_size(input_width, kernel_size, padding, stride)
    conv_height = get_conv_output_size(input_height, kernel_size, padding, stride)
    output_size = (num_data, output_channels, conv_width, conv_height)

    y_reshaped = y.transpose([1, 2, 3, 0]).reshape(-1, output_channels)
    x_reshaped = unfold(Tensor(x).clone(), kernel_size=kernel_size, padding=padding, stride=stride)
    return np.einsum("ijk,jl->ilk", x_reshaped, y_reshaped).reshape(output_size)
