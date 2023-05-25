import numpy as np
import pytest
from scipy.fft import fft, fft2, ifft, ifft2
from scipy.signal import fftconvolve

from kunet import util
from kunet.util import one_hot


def test_one_hot():
    labels = [1, 3, 0, 2, 2]
    num_classes = 4
    expected = np.zeros((4, 5))
    expected[1, 0] = 1
    expected[3, 1] = 1
    expected[0, 2] = 1
    expected[2, 3] = 1
    expected[2, 4] = 1

    got = one_hot(labels, num_classes)
    np.testing.assert_allclose(expected, got)


@pytest.mark.parametrize(
    "fun",
    [np.sin, np.cos, np.tan],
)
def test_dft(fun):
    x = np.arange(-3.0, 5.0, 0.1)
    signal = fun(x)
    # > %timeit util.dft(signal)
    # 727 µs ± 1.85 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # > %timeit fft.fft(signal)
    # 4.53 µs ± 14.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    transformed = util.dft(signal)
    np.testing.assert_allclose(transformed, fft(signal))
    np.testing.assert_allclose(util.idft(transformed), ifft(fft(signal)), atol=1e-13)


@pytest.mark.parametrize(
    "fun1, fun2",
    [
        [np.sin, np.sin],
        [np.sin, np.cos],
        [np.cos, np.cos],
        [np.sin, np.tan],
    ],
)
def test_dft2d(fun1, fun2):
    res1 = fun1(np.arange(-3.0, 5.0, 0.1).reshape(-1, 1))
    res2 = fun2(np.arange(-3.0, 5.0, 0.1).reshape(-1, 1))
    signal = res1 @ res2.T

    got = util.dft2d(signal)
    expected = fft2(signal)
    np.testing.assert_allclose(expected, got)

    got = util.idft2d(got)
    expected = ifft2(expected)
    np.testing.assert_allclose(got, signal, atol=1e-13)
    np.testing.assert_allclose(got, expected, atol=1e-13)
    np.testing.assert_allclose(expected, signal, atol=1e-14)


@pytest.mark.parametrize(
    "fun",
    [np.sin, np.cos, np.tan, np.exp, lambda x: np.sin(2 * x) + np.cos(3 * x + 1)],
)
def test_fft(fun):
    x = np.linspace(-3.0, 5.0, 128)
    signal = fun(x)
    got = util.fft(signal)
    expected = fft(signal)
    np.testing.assert_allclose(got, expected)

    got = util.ifft(got)
    expected = ifft(expected)
    np.testing.assert_allclose(got, signal)
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(expected, signal)


@pytest.mark.parametrize(
    "fun",
    [np.sin, np.cos, np.tan],
)
def test_fft_dft_cross_validation(fun):
    x = np.linspace(-3.0, 5.0, 128)
    signal = fun(x)
    got1 = util.fft(signal)
    got2 = util.dft(signal)
    np.testing.assert_allclose(got1, got2, atol=1e-13)

    got1 = util.ifft(got1)
    got2 = util.idft(got2)
    np.testing.assert_allclose(got1, got2)
    np.testing.assert_allclose(got1, signal)
    np.testing.assert_allclose(got2, signal)


@pytest.mark.parametrize(
    "fun1, fun2",
    [
        [np.sin, np.sin],
        [np.sin, np.cos],
        [np.cos, np.cos],
        [np.sin, np.tan],
    ],
)
def test_fft2d(fun1, fun2):
    # NOTE: fft can only support the signal length of power of 2 now.
    res1 = fun1(np.linspace(-3.0, 5.0, 128).reshape(-1, 1))
    res2 = fun2(np.linspace(-3.0, 5.0, 128).reshape(-1, 1))
    signal = res1 @ res2.T

    got = util.fft2d(signal)
    expected = fft2(signal)
    np.testing.assert_allclose(expected, got)

    got = util.ifft2d(got).real
    expected = ifft2(expected).real
    np.testing.assert_allclose(got, signal, atol=1e-13)
    np.testing.assert_allclose(got, expected, atol=1e-13)
    np.testing.assert_allclose(expected, signal, atol=1e-14)


def test_fft_convolve():
    res1 = np.sin(np.linspace(-3.0, 5.0, 128).reshape(-1, 1))
    res2 = np.cos(np.linspace(-3.0, 5.0, 128).reshape(-1, 1))
    image = res1 @ res2.T
    image = (image - image.min()) / image.ptp() * 255.0
    kernel_size = 3
    kernel = np.full((kernel_size, kernel_size), fill_value=1.0 / kernel_size**2)

    got = util.fft_convolve(image, kernel, mode="full")
    expected = fftconvolve(image, kernel, mode="full")
    np.testing.assert_allclose(got, expected)

    got = util.fft_convolve(image, kernel, mode="same")
    expected = fftconvolve(image, kernel, mode="same")
    np.testing.assert_allclose(got, expected)
