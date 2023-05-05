from __future__ import annotations

import numpy as np

from kunet import utils


class DenseLayer(utils.Node):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: utils.Node | None,
        layer_key: str = "",
        normalize: bool = True,
        weight_initialization_scale: float = 1e-3,
    ) -> None:
        utils.Node.__init__(self, layer_key=layer_key)
        self._input_size = input_size
        self._output_size = output_size
        self._W = np.random.standard_normal((self._output_size, self._input_size))
        self._W *= weight_initialization_scale
        self._b = np.zeros((self._output_size, 1))
        self._activation = activation
        self._normalizer = utils.Normalization() if normalize else None

    def forward(self, X: np.array) -> np.ndarray:
        Z = self.W @ X + self.b
        if self._normalizer:
            Z = self._normalizer(Z)
        self._cache["A_prev"] = X
        A = Z
        if self._activation is not None:
            A = self._activation(Z)
        if np.isnan(A).any():
            raise ValueError()
        return A

    def backward(self, grads: np.matrix) -> np.matrix:
        activation_grads = grads
        if self._activation is not None:
            activation_grads = self._activation.backward(grads)

        normalized_grads = activation_grads
        if self._normalizer:
            normalized_grads = self._normalizer.backward(activation_grads)

        self._cache["dW"] = normalized_grads @ self._cache["A_prev"].T
        self._cache["db"] = np.sum(normalized_grads, axis=1, keepdims=1)
        return self.W.T @ normalized_grads

    def step(self, lr: float):
        self._W -= lr * self._cache["dW"]
        self._b -= lr * self._cache["db"]

    @property
    def W(self) -> np.ndarray:
        return self._W

    @W.setter
    def W(self, new_W: np.ndarray) -> None:
        self._W = new_W

    @property
    def b(self) -> np.ndarray:
        return self._b

    @b.setter
    def b(self, new_b: np.ndarray) -> None:
        self._b = new_b

    def __repr__(self) -> str:
        return (
            f"DenseLayer{self._layer_key}(input_size={self._input_size}, "
            f"output_size={self._output_size}, activation={repr(self._activation)})"
        )


class SequentialModel(utils.Node):
    def __init__(
        self,
        nodes: list[utils.Node],
        layer_key: str = "",
    ) -> None:
        utils.Node.__init__(self, layer_key)
        self._nodes = nodes

    @property
    def nodes(self) -> list[utils.Node]:
        return self._nodes

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self._nodes:
            output = layer(output)
        return output

    def backward(self, grads: np.ndarray | None = None) -> np.ndarray:
        for node in self._nodes[::-1]:
            grads = node.backward(grads)
        return grads

    def step(self, lr: float):
        for node in self._nodes:
            node.step(lr)

    def __repr__(self) -> str:
        layer_repr = "\n\t".join([f"{repr(layer)}" for layer in self._nodes])
        return f"SequentialModel(\n\t{layer_repr}\n)"


class MLPClassifier(SequentialModel):
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[utils.Node],
    ) -> None:
        nodes = []
        assert len(layer_sizes) - 1 == len(activations)
        for layer_id, (input_size, output_size, activation) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], activations)
        ):
            nodes.append(
                DenseLayer(
                    input_size=input_size,
                    output_size=output_size,
                    layer_key=layer_id + 1,
                    activation=activation,
                )
            )
        SequentialModel.__init__(self, nodes)
