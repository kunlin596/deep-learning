import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import sympy as sym
import torch
import torch.nn as nn
from PIL import Image

from kunet import functional, layer
from kunet.layer import (
    ConvLayer2D,
    CrossEntropyLoss,
    DenseLayer,
    HyperbolicTangent,
    MSELoss,
    ReLU,
    SequentialModel,
    Sigmoid,
    Softmax,
)

np.random.seed(42)
np.set_printoptions(suppress=True)


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
    tanh = layer.HyperbolicTangent()

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
    sigmoid = layer.Sigmoid()

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
    y_hat = sym.Matrix([sym.matrix_multiply_elementwise(logits_exp.row(i), s).as_explicit() for i in range(n_features)])
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

    softmax = layer.Softmax()
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

    softmax = layer.Softmax()
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


def fn1(X1, X2):
    return 0.1 + X1**2 + 0.2 * X2 + 0.3


def fn2(X1, X2):
    return 0.1 * X1**3 + 0.2 * X2**2 + 0.3


def univariate_fn1(X):
    return 0.2 * X**2 + 0.1 * X + 1.1


def univariate_fn2(X):
    return -0.2 * X**2 + 0.3 * X - 3.0


def univariate_fn3(X):
    return 0.01 * X**3 - 0.01 * X**2 + 0.02 * X + 2.2 + np.sin(X) * 3.0


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def num_samples() -> int:
    return 64


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "fn,num_epochs,loss_threshold,activation_fn",
    list(
        itertools.product(
            [univariate_fn1, univariate_fn2],
            [50000],
            [0.1],
            [Sigmoid, HyperbolicTangent, ReLU],
        )
    ),
)
def test_one_layer_mlp_nonlinear_regression(
    fn: Callable,
    num_epochs: int,
    loss_threshold: float,
    activation_fn: Callable,
    num_samples: int,
):
    # hyper-parameters
    hidden_size = 128
    learning_rate = 0.01

    model = SequentialModel(
        [
            DenseLayer(
                input_size=1,
                output_size=hidden_size,
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            DenseLayer(
                input_size=hidden_size,
                output_size=1,
                activation=None,
                normalize=False,
                layer_key="2",
            ),
        ]
    )
    X = np.linspace(-10, 10, num_samples).reshape(1, -1)
    y = fn(X).reshape(1, -1)
    loss_fn = MSELoss()

    all_loss = []
    for _ in range(1, num_epochs + 1):
        y_hat = model(X).reshape(1, -1)
        epoch_loss = loss_fn(y_hat, y)
        all_loss.append(epoch_loss)
        model.backward(loss_fn.backward())
        model.step(learning_rate)
        # if i % 1000 == 0:
        #     print(f"{i:6d}: loss={epoch_loss:12.5f}")
        if epoch_loss < loss_threshold:
            break

    assert epoch_loss < loss_threshold


@pytest.mark.parametrize(
    "fn,num_epochs,loss_threshold,activation_fn",
    list(
        itertools.product(
            [univariate_fn1, univariate_fn2, univariate_fn3],
            [100000],
            [0.41],
            [Sigmoid, HyperbolicTangent, ReLU],
        )
    ),
)
def test_two_layer_mlp_nonlinear_regression(
    fn: Callable,
    num_epochs: int,
    loss_threshold: float,
    activation_fn: Callable,
    num_samples: int,
):
    # hyper-parameters
    hidden_size = [64, 64]
    learning_rate = 0.01

    model = SequentialModel(
        [
            DenseLayer(
                input_size=1,
                output_size=hidden_size[0],
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            DenseLayer(
                input_size=hidden_size[0],
                output_size=hidden_size[1],
                activation=activation_fn(),
                normalize=False,
                layer_key="2",
            ),
            DenseLayer(
                input_size=hidden_size[1],
                output_size=1,
                activation=None,
                normalize=False,
                layer_key="3",
            ),
        ]
    )
    X = np.linspace(-10, 10, num_samples).reshape(1, -1)
    y = fn(X).reshape(1, -1)
    loss_fn = MSELoss()

    for i in range(1, num_epochs + 1):
        y_hat = model(X).reshape(1, -1)
        epoch_loss = loss_fn(y_hat, y)
        model.backward(loss_fn.backward())
        model.step(learning_rate)

        if i % 1000 == 0:
            print(f"{i:6d}: loss={epoch_loss:12.5f}")

        if epoch_loss < loss_threshold:
            break

    assert epoch_loss < loss_threshold


@pytest.mark.parametrize(
    "fn,num_epochs,loss_threshold,activation_fn",
    list(
        itertools.product(
            [univariate_fn1, univariate_fn2, univariate_fn3],
            [100000],
            [0.41],
            [Sigmoid, HyperbolicTangent, ReLU],
        )
    ),
)
def test_three_layer_mlp_nonlinear_regression(
    fn: Callable,
    num_epochs: int,
    loss_threshold: float,
    activation_fn: Callable,
    num_samples: int,
):
    # hyper-parameters
    hidden_size = [64, 64, 32]
    learning_rate = 0.01

    model = SequentialModel(
        [
            DenseLayer(
                input_size=1,
                output_size=hidden_size[0],
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            DenseLayer(
                input_size=hidden_size[0],
                output_size=hidden_size[1],
                activation=activation_fn(),
                normalize=False,
                layer_key="2",
            ),
            DenseLayer(
                input_size=hidden_size[1],
                output_size=hidden_size[2],
                activation=activation_fn(),
                normalize=False,
                layer_key="3",
            ),
            DenseLayer(
                input_size=hidden_size[2],
                output_size=1,
                activation=None,
                normalize=False,
                layer_key="4",
            ),
        ]
    )

    X = np.linspace(-10, 10, num_samples).reshape(1, -1)
    y = fn(X).reshape(1, -1)
    loss_fn = MSELoss()

    for i in range(1, num_epochs + 1):
        y_hat = model(X).reshape(1, -1)
        epoch_loss = loss_fn(y_hat, y)
        model.backward(loss_fn.backward())
        model.step(learning_rate)

        if i % 1000 == 0:
            print(f"{i:6d}: loss={epoch_loss:12.5f}")

        if epoch_loss < loss_threshold:
            break

    assert epoch_loss < loss_threshold


@pytest.fixture
def dummy_classification_training_data(rng: np.random.Generator):
    # Prepare dummy data
    num_samples = 256
    num_classes = 3

    X1 = rng.standard_normal((2, num_samples)) * 15.0
    X1[0, :] += 50.0
    y1 = np.zeros((num_classes, num_samples))
    y1[0, :] = 1

    X2 = rng.standard_normal((2, num_samples)) * 10.0
    y2 = np.zeros((num_classes, num_samples))
    y2[1, :] = 1

    X3 = rng.standard_normal((2, num_samples)) * 8.0 + 50.0
    y3 = np.zeros((num_classes, num_samples))
    y3[2, :] = 1
    indices = list(range(num_samples * 3))
    rng.shuffle(indices)
    X = np.hstack([X1, X2, X3])[..., indices]
    y = np.hstack([y1, y2, y3])[..., indices]
    return dict(
        X=X,
        y=y,
        num_classes=num_classes,
        num_samples=num_samples,
    )


def test_one_layer_mlp_nonlinear_classification(
    dummy_classification_training_data: dict,
):
    num_classes = dummy_classification_training_data["num_classes"]
    X = dummy_classification_training_data["X"]
    y = dummy_classification_training_data["y"]

    # hyper-parameters
    learning_rate = 0.01
    hidden_size = 64
    loss_threshold = 0.6
    num_epochs = 50000

    model = SequentialModel(
        [
            DenseLayer(
                input_size=2,
                output_size=hidden_size,
                activation=ReLU(),
                normalize=False,
                layer_key="1",
            ),
            DenseLayer(
                input_size=hidden_size,
                output_size=num_classes,
                activation=Sigmoid(),
                normalize=False,
                layer_key="2",
            ),
            Softmax(),
        ]
    )

    loss_fn = CrossEntropyLoss()

    all_loss = []
    for i in range(1, num_epochs + 1):
        y_hat = model(X).reshape(num_classes, -1)
        epoch_loss = loss_fn(y_hat, y)
        all_loss.append(epoch_loss)
        model.backward(loss_fn.backward())
        model.step(learning_rate)
        if i % 1000 == 0:
            print(f"{i:6d}: loss={epoch_loss:12.5f}")

    assert epoch_loss < loss_threshold


@pytest.fixture
def image():
    with open(Path(__file__).parent / "mario.png", "rb") as file:
        return np.transpose(np.asarray(Image.open(file)), [2, 0, 1])


@pytest.mark.parametrize(
    "output_channels, kernel_size, padding_size, stride_size, algorithm",
    list(
        itertools.product(
            [1, 2],
            [3, 7],
            [0],
            [1],  # FIXME: stride is not working now
            ["gemm", "fft"],
        )
    ),
)
def test_conv_layer_2d(
    output_channels: int,
    kernel_size: int,
    padding_size: int,
    stride_size: int,
    image: np.ndarray,
    algorithm: str,
):
    images = np.stack([image, image], axis=0)
    X = images.astype(float)

    input_channels = X.shape[1]
    input_width = X.shape[2]
    input_height = X.shape[3]

    layer = ConvLayer2D(
        input_channels=input_channels,
        input_width=input_width,
        input_height=input_height,
        output_channels=output_channels,
        kernel_size=kernel_size,
        padding_size=padding_size,
        stride_size=stride_size,
        conv2d_algorithm=algorithm,
        weight_initialization_scale=1.0,
    )

    expected_layer = nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=kernel_size,
        padding=padding_size,
        stride=stride_size,
    )

    expected_layer.weight.data = torch.Tensor(layer.W)
    expected_layer.bias.data = torch.Tensor(layer.b.reshape(-1))

    # Test forward

    y = layer(X)
    expected_y = expected_layer(torch.Tensor(X))
    np.testing.assert_allclose(
        expected_y.detach().numpy(),
        y,
        rtol=0.05,
        atol=1e-4,
    )

    # Test backward

    grads_prev = np.random.random(layer.output_size)
    dA = layer.backward(grads_prev)

    expected_dW = nn.grad.conv2d_weight(
        torch.Tensor(X),
        layer.W.shape,
        torch.Tensor(grads_prev),
        stride=stride_size,
        padding=padding_size,
    ).numpy()
    expected_dA = nn.grad.conv2d_input(
        X.shape,
        torch.Tensor(layer.W),
        torch.Tensor(grads_prev),
        stride=stride_size,
        padding=padding_size,
    ).numpy()
    expected_db = grads_prev.sum(axis=(0, 2, 3))[..., np.newaxis]

    np.testing.assert_allclose(expected_dW, layer._cache["dW"], rtol=0.001, atol=1e-10)
    np.testing.assert_allclose(expected_db, layer._cache["db"], rtol=0.001, atol=1e-10)
    np.testing.assert_allclose(expected_dA, dA, rtol=0.41, atol=1e-10)
