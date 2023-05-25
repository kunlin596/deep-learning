from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kunet import functional


@pytest.fixture
def image():
    with open(Path(__file__).parent / "mario.png", "rb") as file:
        return np.transpose(np.asarray(Image.open(file)), [2, 0, 1])[:3, :, :]


@pytest.mark.parametrize(
    "padding,stride",
    list(itertools.product(range(0, 4), range(1, 3))),
)
def test_tensor_convolution(image, padding, stride):
    # NOTE: dummy image
    # image = np.zeros((3, 100, 100))
    # image[:, 50:60, :] = 255
    # image[:, :, 50:60] = 255

    images = np.stack([image, image], axis=0).astype(np.float64)
    filter1 = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    filter2 = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    filter3 = np.ones((3, 3))
    filter4 = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    filters = np.asarray(
        [
            np.tile(filter1[np.newaxis, ...], [3, 1, 1]),
            np.tile(filter2[np.newaxis, ...], [3, 1, 1]),
            np.tile(filter3[np.newaxis, ...], [3, 1, 1]),
            np.tile(filter4[np.newaxis, ...], [3, 1, 1]),
        ]
    )
    filters /= 9.0

    output1 = functional.fft_tensor_convolution(images, filters, padding, stride)
    output2 = functional.gemm_tensor_convolution(images, filters, padding, stride)
    np.testing.assert_allclose(output1, output2, atol=1e-12)

    # NOTE: visualize for debugging purpose.
    # import matplotlib.pyplot as plt
    # k = 1
    # num_filters = len(filters)
    # num_cols = num_filters * 3

    # for image_id in range(len(images)):
    #     for i in range(num_filters):
    #         diff = output1[image_id][i] - output2[image_id][i]
    #         #
    #         plt.subplot(len(images), num_cols, k)
    #         k += 1
    #         plt.imshow(output1[image_id][i], cmap="gray")
    #         plt.title(f"{image_id} - filter {i}")

    #         #
    #         plt.subplot(len(images), num_cols, k)
    #         k += 1
    #         plt.imshow(output2[image_id][i], cmap="gray")
    #         plt.title(f"{image_id} - filter {i}")

    #         #
    #         plt.subplot(len(images), num_cols, k)
    #         k += 1
    #         plt.imshow(diff, cmap="gray")
    #         plt.title(f"{image_id} - diff {i} {diff.ptp(): 7.3f}")
    #         print(diff.ptp())
