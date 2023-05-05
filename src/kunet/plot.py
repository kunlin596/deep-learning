from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from kunet import utils


def plot3d(X1, X2, model, fn):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(X1, X2)
    Z = fn(X, Y)
    Z_hat = model(np.dstack([X, Y]).reshape(2, -1)).reshape(Z.shape)

    surf = ax.plot_surface(X, Y, Z, label="ground truth")
    # See https://stackoverflow.com/a/70313234
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    surf = ax.plot_surface(X, Y, Z_hat, label="prediction")
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    ax.legend()
    plt.show()


def plot2d(X, model, fn, loss):
    X = X.reshape(-1)
    plt.subplot(121)
    plt.plot(X, fn(X), label="ground truth")
    plt.scatter(X, fn(X), s=5.0)
    plt.plot(X, model(X.reshape(1, -1)).reshape(-1), label="prediction")
    plt.legend()
    plt.subplot(122)
    plt.plot(loss)
    plt.show()


def plot_classification_ground_truth_2d(
    ax: Axes,
    X: np.ndarray,
    y: np.ndarray,
    labels: list,
):
    mask = y.argmax(axis=0)
    for label in labels:
        XX = X[:, mask == label]
        ax.scatter(XX[0, :], XX[1, :], marker="x")
    ax.axis("equal")


def plot_classification_inference_result_2d(
    axes: Axes,
    model: utils.Node,
    X: np.ndarray,
    labels: list,
):
    y_hat = model(X)
    mask = y_hat.argmax(axis=0)
    for label in labels:
        XX = X[:, mask == label]
        axes.scatter(XX[0, :], XX[1, :], marker="x")
    axes.axis("equal")


def plot_classification_decision_boundary_2d(
    axes: Axes,
    model: utils.Node,
    X: np.ndarray,
    labels: list,
    padding: float = 10.0,
    num_points: int = 1000,
    alpha: float = 0.5,
):
    x_min, y_min = X.min(axis=1) - padding
    x_max, y_max = X.max(axis=1) + padding
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(x_min, x_max, num_points),
        np.linspace(y_min, y_max, num_points),
    )
    X_plot = np.dstack([mesh_x, mesh_y])
    X_plot = X_plot.reshape(-1, 2).T
    y_hat = model(X_plot)
    mask = y_hat.argmax(axis=0)
    Z_plot = mask.reshape(num_points, num_points)
    X_plot = (X_plot.T).reshape(num_points, num_points, 2)

    for label in labels:
        axes.contour(
            X_plot[:, :, 0],
            X_plot[:, :, 1],
            Z_plot == label,
            alpha=alpha,
        )
    axes.axis("equal")
