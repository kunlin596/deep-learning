from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Node(ABC):
    """Computational graph node."""

    _cache: dict[str, np.ndarray] | None = None  # type: ignore
    _track_grads: bool = False
    _parameters: dict | None = None
    _layer_key: str | None = None

    def __init__(self, layer_key: str = "", track_grads: bool = True):
        self._layer_key = layer_key
        self._track_grads = track_grads
        self._cache: dict[str, np.ndarray] = {}
        self._parameters = {}

    def zero_grads(self):
        self._cache = {}

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray | float:
        """Compute the feed forward result."""

    @abstractmethod
    def backward(self, grads: np.ndarray | float) -> np.ndarray:
        """Compute the gradients."""

    def step(self, learning_rate: float) -> None:
        """Update internal state."""
        return

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray | float:
        return self.forward(*args, **kwargs)

    @property
    def cache(self) -> dict[str, np.ndarray]:
        return self._cache

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
