from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve as _fftconvolve

from kunet import functional
from kunet.graph import Node


class BatchNormalization(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 4:
            mu = X.mean(axis=0, keepdims=True)
            sigma = X.std(axis=0, keepdims=True) + 1e-10
        else:
            mu = X.mean(axis=1, keepdims=True)
            sigma = X.std(axis=1, keepdims=True) + 1e-10
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


class Linear(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        return grads


class HyperbolicTangent(Node):
    def __init__(self, layer_key: str = "", track_grads: bool = True):
        super().__init__(layer_key, track_grads)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = functional.tanh(Z)
        self._cache["Z"] = Z
        return A

    def backward(self, grads: np.ndarray | float = 1.0) -> np.ndarray:
        new_grads = grads * functional.tanh_diff(self._cache["Z"])
        return new_grads


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
    def __init__(
        self,
        layer_key: str = "",
        track_grads: bool = True,
    ):
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


class DenseLayer(Node):
    """Densely connected layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Node | None,
        layer_key: str = "",
        normalize: bool = True,
        weight_initialization_scale: float = 1e-3,
    ) -> None:
        Node.__init__(self, layer_key=layer_key)
        self._input_size = input_size
        self._output_size = output_size
        self._W = np.random.standard_normal((self._output_size, self._input_size))
        self._W *= weight_initialization_scale
        self._b = np.zeros((self._output_size, 1))
        self._activation = activation
        self._normalizer = BatchNormalization() if normalize else None

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = self.W @ X + self.b
        if self._normalizer:
            Z = self._normalizer(Z)
        self._cache["A_prev"] = X
        A = Z
        if self._activation is not None:
            A = self._activation(Z)
        if np.isnan(A).any():
            raise ValueError("Something is wrong with the input A! It has NaN inside!")
        return A

    def backward(self, grads: np.ndarray) -> np.matrix:
        activation_grads = grads
        if self._activation is not None:
            activation_grads = self._activation.backward(grads)

        normalized_grads = activation_grads
        if self._normalizer:
            normalized_grads = self._normalizer.backward(activation_grads)

        self._cache["dW"] = normalized_grads @ self._cache["A_prev"].T
        self._cache["db"] = np.sum(normalized_grads, axis=1, keepdims=1)
        return self.W.T @ normalized_grads

    def step(self, learning_rate: float):
        self._W -= learning_rate * self._cache["dW"]
        self._b -= learning_rate * self._cache["db"]

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


class FlattenLayer(Node):
    """A flatten layer which transforms data into one vector."""

    def __init__(
        self,
        layer_key: str = "",
        track_grads: bool = True,
    ):
        Node.__init__(self, layer_key, track_grads)

    def forward(self, X: np.ndarray) -> Any:
        self._cache["X_shape"] = X.shape
        return X.reshape(X.shape[0], -1).T

    def backward(self, grads: np.ndarray | float) -> np.ndarray:
        new_grads = np.reshape(grads.T, self._cache["X_shape"])
        return new_grads

    def __repr__(self) -> str:
        return f"Flatten_{self._layer_key}()"


class ConvLayer2D(Node):
    """2D convolutional layer.

    The input tensor size should be in the order of batch, channel, width, height.
    """

    @dataclass
    class ConvSize:
        """Convolutional layer size."""

        depth: int
        width: int

    def __init__(
        self,
        input_channels: int,
        input_width: int,
        input_height: int,
        output_channels: int,
        kernel_size: int,
        padding_size: int,
        stride_size: int,
        layer_key: str = "",
        track_grads: bool = True,
        weight_initialization_scale: float = 1e-3,
        conv2d_algorithm: str = "gemm",
    ):
        super().__init__(layer_key, track_grads)
        self._kernel_size = kernel_size
        if self._kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd!")
        self._padding_size = padding_size
        self._stride_size = stride_size
        self._output_channels = output_channels

        self._input_channels = input_channels
        self._input_width = input_width
        self._input_height = input_height
        self._batch_size = 1
        self._conv2d_algorithm = conv2d_algorithm

        # Initialize the weights
        self._W = np.random.standard_normal(
            (
                # Output channels
                self._output_channels,
                # Input channels,
                self._input_channels,
                # Output width
                self._kernel_size,
                # Output height
                self._kernel_size,
            ),
        )
        self.W *= weight_initialization_scale
        self.b = np.zeros((self._output_channels, 1))

    @property
    def output_size(self) -> tuple[int, int, int, int]:
        return (
            self._batch_size,
            self._output_channels,
            functional.get_conv_output_size(
                self._input_width,
                self._kernel_size,
                self._padding_size,
                self._stride_size,
            ),
            functional.get_conv_output_size(
                self._input_height,
                self._kernel_size,
                self._padding_size,
                self._stride_size,
            ),
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._batch_size = X.shape[0]
        self._cache["A_prev"] = X
        conv_fn = None
        if self._conv2d_algorithm == "fft":
            conv_fn = functional.fft_tensor_convolution
        elif self._conv2d_algorithm == "gemm":
            conv_fn = functional.gemm_tensor_convolution
        else:
            raise ValueError(f"Not supported conv2d algorithm {self._conv2d_algorithm}!")
        output = conv_fn(
            X,
            self.W,
            self._padding_size,
            self._stride_size,
        )
        output += self.b[np.newaxis, ..., np.newaxis]
        return output

    def backward(self, grads: np.ndarray | float) -> np.ndarray:
        """Backward pass of the convolutional layer.

        See https://www.youtube.com/watch?v=Lakz2MoHy6o for detailed walkthrough.

        Args:
            grads (np.ndarray | float): gradients from the next layer.

        Raises:
            ValueError: if any shapes are not matching.

        Returns:
            np.ndarray: gradients for the previous layer.
        """
        assert np.shape(grads) == self.output_size

        # NOTE: A inner-product-like tensor convolution.
        # I am now aware of an more efficient way to do this.
        A_prev = self._cache["A_prev"]
        dW = np.zeros_like(self.W)
        dA = np.zeros_like(A_prev)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                convolved = _fftconvolve(
                    np.pad(
                        A_prev[:, j],
                        (
                            (0, 0),
                            (self._padding_size, self._padding_size),
                            (self._padding_size, self._padding_size),
                        ),
                        mode="constant",
                    ),
                    grads[:, i, ::-1, ::-1],
                    mode="valid",
                )[0]
                assert dW[i, j].shape == convolved.shape
                dW[i, j] = convolved
                for m in range(A_prev.shape[0]):
                    convolved = _fftconvolve(
                        grads[m, i],
                        self.W[i, j],
                        mode="full",
                    )
                    dA[m, j] += convolved[
                        self._padding_size : convolved.shape[0] - self._padding_size,
                        self._padding_size : convolved.shape[1] - self._padding_size,
                    ]

        db = np.sum(grads[:, :], axis=(0, 2, 3)).reshape(self.b.shape)

        assert dW.shape == self.W.shape, f"dW and W have difference shape! {dW.shape} != {self.W.shape}."
        assert (
            dA.shape == self._cache["A_prev"].shape
        ), f"dA and A_prev have difference shape! {dA.shape} != {self._cache['A_prev'].shape}."
        assert db.shape == self.b.shape, f"db and b have difference shape! {db.shape} != {self.b.shape}."

        self._cache["dW"] = dW
        self._cache["db"] = db

        return dA

    def step(self, learning_rate: float) -> None:
        self.W -= learning_rate * self._cache["dW"]
        self.b -= learning_rate * self._cache["db"]

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

    @property
    def conv2d_algorithm(self) -> str:
        return self._conv2d_algorithm

    @conv2d_algorithm.setter
    def conv2d_algorithm(self, new_conv2d_algorithm: str) -> None:
        self._conv2d_algorithm = new_conv2d_algorithm

    def __repr__(self) -> str:
        return (
            f"ConvolutionalLayer2D_{self._layer_key}("
            f"input_size=({self._input_channels}, {self._input_width}, {self._input_height}), "
            f"output_size={self.output_size[1:]})"
        )


class SequentialModel(Node):
    def __init__(
        self,
        nodes: list[Node],
        layer_key: str = "",
    ) -> None:
        Node.__init__(self, layer_key)
        self._nodes = nodes

    @property
    def nodes(self) -> list[Node]:
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
