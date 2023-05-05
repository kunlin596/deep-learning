import numpy as np

EPS = 1e-10


def softmax(X: np.ndarray) -> np.ndarray:
    """Softmax function"""
    X = np.asarray(X)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    exp = np.exp(X)
    logits = exp / exp.sum(axis=0, keepdims=1)
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
    """Sigmoid function"""
    return 1 / (1 + np.exp(-np.asarray(X)) + EPS)


def sigmoid_diff(X: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid function"""
    s = sigmoid(np.asarray(X))
    return s * (1 - s)


def tanh(X: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent function"""
    return np.tanh(X)


def tanh_diff(X: np.ndarray) -> np.ndarray:
    """Derivative of hyperbolic tangent function"""
    return 1 - np.tanh(np.asarray(X)) ** 2


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error loss function"""
    loss = 0.5 * (np.linalg.norm(y_hat - y, axis=0) ** 2).mean()
    if np.isnan(loss):
        raise ValueError("loss is nan!")
    return loss


def mse_loss_diff(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Derivative of squared error loss function"""
    m = y.shape[1]
    return (y_hat - y) / m


def cross_entropy_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy of shape `NxM`"""
    assert y_hat.shape == y.shape
    loss = (-np.asarray(y) * np.log(np.asarray(y_hat))).sum(axis=0).mean()
    if np.isnan(loss):
        raise ValueError("loss is nan!")
    return loss


def cross_entropy_loss_diff(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Derivative of cross-entropy of shape `NxM`"""
    assert y_hat.shape == y.shape
    m = y.shape[1]
    diff = -np.asarray(y) / y_hat / m
    return diff


def relu(X: np.ndarray) -> np.ndarray:
    """Rectified linear unit function"""
    return np.maximum(X, 0.0)


def relu_diff(X: np.ndarray) -> np.ndarray:
    """Derivative of rectified linear unit"""
    return (X >= 0.0).astype(float)
