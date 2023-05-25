from __future__ import annotations

import typing

import numpy as np


def one_hot(labels: list[int], num_classes: int) -> np.ndarray:
    """Encode a list of labels in to one-hot encoding.

    Args:
        labels (list[int]): list of labels
        num_classes (int): the number of classes

    Returns:
        np.ndarray: encoded matrix
    """
    labels = np.asarray(labels)
    encoded = np.zeros((num_classes, len(labels)), dtype=int)
    indices = [i * num_classes + label for i, label in enumerate(labels)]
    encoded.T.flat[indices] = 1
    return encoded


def _get_dft_mat(n: int) -> np.ndarray:
    cov = np.asarray(range(n)).reshape(-1, 1)
    cov = cov @ cov.T
    dft_mat = np.exp(-2.0j * np.pi / n) ** cov
    return dft_mat


def _get_dft_mat_inv(n: int) -> np.ndarray:
    # NOTE: The precision is 2 orders of magnitude lower than directly calling numpy inv. 1e-14 -> 1e-12.
    # cov = np.asarray(range(n)).reshape(-1, 1)
    # cov = cov @ cov.T
    # dft_mat_inv = 1 / n * np.exp(2.0 * np.pi * 1.0j / n) ** cov
    # return dft_mat_inv
    return np.linalg.inv(_get_dft_mat(n))


def dft(x: np.ndarray) -> np.ndarray:
    """Perform Discrete Fourier Transform to transform signal from spatial domain to frequency domain.

    The time complexity is O(N^2). Note that it's for educational purpose only, because it does not scale.
    The code below is essentially computing the product of the DFT matrix the input coefficients.
    """
    return _get_dft_mat(len(x)) @ x


def idft(x: np.ndarray) -> np.ndarray:
    return _get_dft_mat_inv(len(x)) @ x


def fft(x: np.ndarray) -> np.ndarray:
    """Perform Fast Fourier Transform to transform signal from spatial domain to frequency domain.

    The time complexity is O(N*log(N)).
    NOTE: As a simple practice, here, signal length can only be the power of two.
    TODO: Figure out the other signal case.

    See https://www.youtube.com/watch?v=h7apO7q16V0 for more explanation on this,
    however pay attention that the code is wrong in the video.

    Some more notes on performance:

        > x = np.linspace(-6.0, 6.0, 128)
        > signal = np.sin(x)
        > %timeit dft(signal)
        6.66 ms ± 954 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        > %timeit fft(signal)
        578 µs ± 4.24 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

        > %timeit scipy.fft.fft(signal)
        4.17 µs ± 26.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    """
    n = len(x)
    if n == 1:
        return x
    w = np.exp(-2.0j * np.pi / n)
    x_e = x[::2]
    x_o = x[1::2]
    y_e = fft(x_e)
    y_o = fft(x_o)
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n // 2):
        wi = w**i
        y[i] = y_e[i] + wi * y_o[i]
        y[i + n // 2] = y_e[i] - wi * y_o[i]
    return y


def _ifft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n == 1:
        return x
    w = np.exp(2.0j * np.pi / n)
    x_e = x[::2]
    x_o = x[1::2]
    y_e = _ifft(x_e)
    y_o = _ifft(x_o)
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n // 2):
        wi = w**i
        y[i] = y_e[i] + wi * y_o[i]
        y[i + n // 2] = y_e[i] - wi * y_o[i]
    return y


def ifft(x: np.ndarray) -> np.ndarray:
    """Perform Inverse Fast Fourier Transform to transform signal from frequency domain to spatial domain.

    See https://www.youtube.com/watch?v=h7apO7q16V0 for more explanation on this,
    however pay attention that the code is wrong in the video.
    """
    return _ifft(x) / len(x)


def _ft2d(x: np.ndarray, ft1d_fn: typing.Callable) -> np.ndarray:
    """Perform 2D Discrete Fourier Transform."""
    x_freq = np.zeros_like(x, dtype=np.complex128)

    for row in range(x.shape[0]):
        x_freq[row] = ft1d_fn(x[row])
    for col in range(x_freq.shape[1]):
        x_freq[:, col] = ft1d_fn(x_freq[:, col])
    return x_freq


def _ift2d(x: np.ndarray, ift1d_fn: typing.Callable) -> np.ndarray:
    x_spatial = np.zeros_like(x, dtype=np.complex128)
    for col in range(x.shape[1]):
        x_spatial[:, col] = ift1d_fn(x[:, col])
    for row in range(x_spatial.shape[0]):
        x_spatial[row] = ift1d_fn(x_spatial[row])
    return x_spatial.real


def dft2d(x: np.ndarray) -> np.ndarray:
    """Perform 2D Discrete Fourier Transform."""
    return _ft2d(x, dft)


def idft2d(x: np.ndarray) -> np.ndarray:
    """Perform 2D Inverse Discrete Fourier Transform."""
    return _ift2d(x, idft)


def fft2d(x: np.ndarray) -> np.ndarray:
    """Perform 2D Fast Fourier Transform."""
    return _ft2d(x, fft)


def ifft2d(x: np.ndarray) -> np.ndarray:
    """Perform 2D Inverse Fast Fourier Transform."""
    return _ift2d(x, ifft)


def fft_convolve(x: np.ndarray, y: np.ndarray, mode="same"):
    s1 = np.asarray(x.shape)
    s2 = np.asarray(y.shape)
    size = s1 + s2
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    offset = 0
    if mode == "same":
        offset = s2.shape[0] // 2
    elif mode == "full":
        offset = 0
    fslice = tuple([slice(offset, int(sz) - offset - 1) for sz in size])
    convolved = np.fft.ifft2(np.fft.fft2(x, fsize) * np.fft.fft2(y, fsize))
    return convolved[fslice].real.astype(x.dtype)
