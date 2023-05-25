from __future__ import annotations

import numpy as np

from kunet.layer import (
    BatchNormalization,
    ConvLayer2D,
    DenseLayer,
    FlattenLayer,
    HyperbolicTangent,
    Node,
    SequentialModel,
    Softmax,
)


class MLPClassifier(SequentialModel):
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[Node],
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


class CNN(SequentialModel):
    def __init__(
        self,
        input_channels: int,
        input_width: int,
        input_height: int,
        conv_sizes: list[ConvLayer2D.ConvSize],
    ) -> None:
        nodes: list[Node] = []

        # Add convolutional layers
        layer_input_channels = input_channels
        layer_input_width = input_width
        layer_input_height = input_height
        for i, conv_size in enumerate(conv_sizes):
            layer = ConvLayer2D(
                input_channels=layer_input_channels,
                input_width=layer_input_width,
                input_height=layer_input_height,
                output_channels=conv_size.depth,
                kernel_size=conv_size.width,
                stride_size=1,
                padding_size=0,
                layer_key=f"{i + 1}",
            )
            layer_input_channels = layer.output_size[1]
            layer_input_width = layer.output_size[2]
            layer_input_height = layer.output_size[3]
            nodes.append(layer)
            nodes.append(BatchNormalization())
            nodes.append(HyperbolicTangent())

        nodes.append(FlattenLayer(layer_key="1"))

        output_vector_size = np.prod(nodes[-4].output_size)

        nodes.append(
            DenseLayer(
                input_size=output_vector_size,
                output_size=100,
                activation=HyperbolicTangent(),
                normalize=False,
                layer_key="1",
            )
        )
        nodes.append(
            DenseLayer(
                input_size=100,
                output_size=10,
                activation=Softmax(),
                normalize=False,
                layer_key="2",
            )
        )

        SequentialModel.__init__(self, nodes)
