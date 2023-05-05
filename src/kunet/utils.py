from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from kunet import functional


def one_hot(labels: list[int], num_classes: int) -> np.ndarray:
    labels = np.asarray(labels)
    encoded = np.zeros((num_classes, len(labels)), dtype=int)
    indices = [i * num_classes + label for i, label in enumerate(labels)]
    encoded.T.flat[indices] = 1
    return encoded


class Node(ABC):
    """Computational graph node"""

    _cache = None
    _track_grads = None
    _parameters = None
    _layer_key = None

    def __init__(self, layer_key: str = "", track_grads: bool = True):
        self._layer_key = layer_key
        self._track_grads = track_grads
        self._cache = {}
        self._parameters = {}

    def zero_grads(self):
        self._cache = {}

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Feed forward"""

    @abstractmethod
    def backward(self, grads: np.ndarray | float) -> np.ndarray:
        """Compute gradients"""

    def step(self, lr: float) -> None:
        """Update internal state"""
        return

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @property
    def cache(self) -> dict[str, np.ndarray]:
        return self._cache

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalization(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=1, keepdims=1)
        sigma = X.std(axis=1, keepdims=1)
        self._cache["sigma"] = sigma
        Z = (X - mu) / sigma
        return Z

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return grads / self._cache["sigma"]


class Sigmoid(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = functional.sigmoid(Z)
        self._cache["Z"] = Z
        return A

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return grads * functional.sigmoid_diff(self._cache["Z"])


class HyperbolicTangent(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = functional.tanh(Z)
        self._cache["Z"] = Z
        return A

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return grads * functional.tanh_diff(self._cache["Z"])


class ReLU(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = functional.relu(Z)
        self._cache["Z"] = Z
        return A

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return grads * functional.relu_diff(self._cache["Z"])


class Softmax(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = functional.softmax(Z)
        self._cache["Z"] = Z
        return A

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        curr_grads = functional.softmax_diff(self._cache["Z"])
        return np.einsum("kij,jk->ik", curr_grads, grads)


class MSELoss(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, y_hat: np.ndarray, y) -> float:
        loss = functional.mse_loss(y_hat, y)
        self._cache["y_hat"] = y_hat
        self._cache["y"] = y
        return loss

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return functional.mse_loss_diff(
            self._cache["y_hat"],
            self._cache["y"],
        )


class CrossEntropyLoss(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        loss = functional.cross_entropy_loss(y_hat, y)
        self._cache["y_hat"] = y_hat
        self._cache["y"] = y
        return loss

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return functional.cross_entropy_loss_diff(
            self._cache["y_hat"],
            self._cache["y"],
        )
