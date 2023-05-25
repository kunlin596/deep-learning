import numpy as np
import pytest
import sympy as sym

from kunet import functional, utils

np.random.seed(42)


def test_one_hot():
    labels = [1, 3, 0, 2, 2]
    num_classes = 4
    expected = np.zeros((4, 5))
    expected[1, 0] = 1
    expected[3, 1] = 1
    expected[0, 2] = 1
    expected[2, 3] = 1
    expected[2, 4] = 1

    got = utils.one_hot(labels, num_classes)
    np.testing.assert_allclose(expected, got)


@pytest.fixture
def nfeatures() -> int:
    return 10


@pytest.fixture
def nsamples() -> int:
    return 3


@pytest.fixture
def X(nfeatures: int, nsamples: int) -> np.ndarray:
    return np.random.random((nfeatures, nsamples))


def test_tanh():
    X = sym.var("X")
    g = sym.tanh(X)
    g_diff = g.diff()
    tanh = utils.HyperbolicTangent()

    expected = g.subs(dict(X=1.0))

    got = functional.tanh(1.0)
    np.testing.assert_almost_equal(expected, got)

    got = tanh(1.0)
    np.testing.assert_almost_equal(expected, got)

    expected = g_diff.subs(dict(X=1.0))

    got = functional.tanh_diff(1.0)
    np.testing.assert_almost_equal(expected, got)

    got = tanh.backward()
    np.testing.assert_almost_equal(expected, got)


def test_sigmoid():
    X = sym.var("X")
    g = 1 / (1 + sym.exp(-X))
    g_diff = g.diff()
    sigmoid = utils.Sigmoid()

    expected = g.subs(dict(X=1.0))
    got = functional.sigmoid(1.0)
    np.testing.assert_almost_equal(expected, got)

    got = sigmoid(1.0)
    np.testing.assert_almost_equal(expected, got)

    expected = g_diff.subs(dict(X=1.0))
    got = functional.sigmoid_diff(1.0)
    np.testing.assert_almost_equal(expected, got)

    got = sigmoid.backward()
    np.testing.assert_almost_equal(expected, got)


def _get_y_hat(n_features, n_samples):
    logits = sym.MatrixSymbol("logits", n_features, n_samples)
    logits_exp = logits.applyfunc(sym.exp).as_explicit()
    s = (sym.Matrix.ones(1, n_features) * logits_exp).applyfunc(lambda x: 1 / x).as_explicit()
    y_hat = sym.Matrix(
        [
            sym.matrix_multiply_elementwise(logits_exp.row(i), s).as_explicit()
            for i in range(n_features)
        ]
    )
    return y_hat, logits


def test_softmax_single_sample():
    n_features = 3
    y_hat, logits = _get_y_hat(n_features, 1)

    dummy_values = np.asarray(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    expected = np.asarray(y_hat.subs({logits: sym.Matrix(dummy_values)})).astype(float)
    got = functional.softmax(dummy_values)

    np.testing.assert_allclose(expected, got)
    np.testing.assert_almost_equal(np.sum(got), 1.0)

    jacobian = sym.simplify(y_hat.jacobian(logits))
    expected = jacobian.subs({logits: sym.Matrix(dummy_values)})
    expected = np.asarray(expected).astype(float)

    # NOTE: The first axis is batch number
    got = functional.softmax_diff(dummy_values)[0]
    np.testing.assert_almost_equal(expected, got)

    softmax = utils.Softmax()
    softmax.forward(dummy_values)
    prev_grads = np.random.random((n_features, 1))
    grads = softmax.backward(prev_grads)
    assert grads.shape == (n_features, 1)


def test_softmax_multiple_samples():
    n_features = 3
    n_samples = 2
    dummy_values = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    expected = np.empty((n_samples, n_features, n_features), dtype=object)
    for i in range(n_samples):
        y_hat, logits = _get_y_hat(n_features, 1)
        sample_dummy_values = dummy_values[:, i]
        jacobian = sym.simplify(y_hat.jacobian(logits))
        sample_expected = jacobian.subs({logits: sym.Matrix(sample_dummy_values)})
        sample_expected = np.asarray(sample_expected).astype(float)
        expected[i] = sample_expected

    got = functional.softmax_diff(dummy_values)
    np.testing.assert_almost_equal(expected, got)

    softmax = utils.Softmax()
    softmax.forward(dummy_values)
    prev_grads = np.random.random((n_features, n_samples))

    expected = []
    for prev_grads_sample, got_sample in zip(prev_grads.T, got):
        expected.append(got_sample @ prev_grads_sample.T)
    expected = np.asarray(expected).T

    got = softmax.backward(prev_grads)
    assert got.shape == (n_features, n_samples)
    np.testing.assert_allclose(expected, got)


def test_cross_entropy():
    y1, y2, y3 = sym.symbols("y1, y2, y3")
    yhat1, yhat2, yhat3 = sym.symbols("yhat1, yhat2, yhat3")

    g = -(y1 * sym.log(yhat1) + y2 * sym.log(yhat2) + y3 * sym.log(yhat3))
    subs = dict(
        y1=0.0,
        y2=1.0,
        y3=0.0,
        yhat1=0.2,
        yhat2=0.7,
        yhat3=0.1,
    )

    expected = g.subs(subs)

    dummy_y = np.asarray([0, 1, 0]).reshape(-1, 1)
    dummy_y_hat = np.asarray([0.2, 0.7, 0.1]).reshape(-1, 1)

    got = functional.cross_entropy_loss(dummy_y_hat, dummy_y)
    np.testing.assert_almost_equal(expected, got)

    expected = g.diff(yhat1).subs(subs)
    got = functional.cross_entropy_loss_diff(dummy_y_hat, dummy_y)[0]
    np.testing.assert_almost_equal(expected, got)

    expected = g.diff(yhat2).subs(subs)
    got = functional.cross_entropy_loss_diff(dummy_y_hat, dummy_y)[1]
    np.testing.assert_almost_equal(expected, got)

    expected = g.diff(yhat3).subs(subs)
    got = functional.cross_entropy_loss_diff(dummy_y_hat, dummy_y)[2]
    np.testing.assert_almost_equal(expected, got)
