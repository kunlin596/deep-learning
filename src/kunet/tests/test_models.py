import itertools

import numpy as np
import pytest

from kunet import models, utils

np.set_printoptions(suppress=True)


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
def rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "fn,num_epochs,loss_threshold,activation_fn",
    list(
        itertools.product(
            [univariate_fn1, univariate_fn2],
            [50000],
            [0.1],
            [utils.Sigmoid, utils.HyperbolicTangent, utils.ReLU],
        )
    ),
)
def test_one_layer_mlp_nonlinear_regression(
    fn,
    activation_fn,
    num_epochs,
    loss_threshold,
    num_samples,
):
    # hyper-parameters
    hidden_size = 128
    learning_rate = 0.01

    model = models.SequentialModel(
        [
            models.DenseLayer(
                input_size=1,
                output_size=hidden_size,
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            models.DenseLayer(
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
    loss_fn = utils.MSELoss()

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
            [0.02],
            [utils.Sigmoid, utils.HyperbolicTangent, utils.ReLU],
        )
    ),
)
def test_two_layer_mlp_nonlinear_regression(
    fn,
    num_epochs,
    loss_threshold,
    activation_fn,
    num_samples,
):
    # hyper-parameters
    hidden_size = [64, 64]
    learning_rate = 0.01

    model = models.SequentialModel(
        [
            models.DenseLayer(
                input_size=1,
                output_size=hidden_size[0],
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            models.DenseLayer(
                input_size=hidden_size[0],
                output_size=hidden_size[1],
                activation=activation_fn(),
                normalize=False,
                layer_key="2",
            ),
            models.DenseLayer(
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
    loss_fn = utils.MSELoss()

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
            [0.022],
            [utils.Sigmoid, utils.HyperbolicTangent, utils.ReLU],
        )
    ),
)
def test_three_layer_mlp_nonlinear_regression(
    fn,
    num_epochs,
    loss_threshold,
    activation_fn,
    num_samples,
):
    # hyper-parameters
    hidden_size = [64, 64, 32]
    learning_rate = 0.01

    model = models.SequentialModel(
        [
            models.DenseLayer(
                input_size=1,
                output_size=hidden_size[0],
                activation=activation_fn(),
                normalize=False,
                layer_key="1",
            ),
            models.DenseLayer(
                input_size=hidden_size[0],
                output_size=hidden_size[1],
                activation=activation_fn(),
                normalize=False,
                layer_key="2",
            ),
            models.DenseLayer(
                input_size=hidden_size[1],
                output_size=hidden_size[2],
                activation=activation_fn(),
                normalize=False,
                layer_key="3",
            ),
            models.DenseLayer(
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
    loss_fn = utils.MSELoss()

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
def dummy_classification_training_data(rng):
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

    model = models.SequentialModel(
        [
            models.DenseLayer(
                input_size=2,
                output_size=hidden_size,
                activation=utils.ReLU(),
                normalize=False,
                layer_key="1",
            ),
            models.DenseLayer(
                input_size=hidden_size,
                output_size=num_classes,
                activation=utils.Sigmoid(),
                normalize=False,
                layer_key="2",
            ),
            utils.Softmax(),
        ]
    )

    loss_fn = utils.CrossEntropyLoss()

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
