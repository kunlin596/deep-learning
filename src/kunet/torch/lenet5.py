from __future__ import annotations

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as D
from torchvision import transforms as T


class LeNet5(nn.Module):
    def __init__(self, num_classes: int) -> None:
        nn.Module.__init__(self)

        self._num_classes = num_classes

        self._feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=120,
                kernel_size=5,
                stride=1,
            ),
            nn.Tanh(),
        )

        self._classifier = nn.Sequential(
            nn.Linear(
                in_features=120,
                out_features=84,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=84,
                out_features=self._num_classes,
            ),
        )

    @property
    def feature_extractor(self) -> nn.Module:
        return self._feature_extractor

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._feature_extractor(x)
        flattened = torch.flatten(features, 1)
        logits = self._classifier(flattened)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def _run_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer | None, float]:
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    # Per-epoch
    running_loss = 0

    for _, (X, y_true) in enumerate(data_loader):
        if optimizer is not None:
            optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        if optimizer is not None:
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    return model, epoch_loss


def _get_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
):
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def _plot_losses(train_losses: list[float], valid_losses: list[float]):
    import matplotlib.pyplot as plt
    import numpy as np

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color="blue", label="Training loss")
    ax.plot(valid_losses, color="red", label="Validation loss")
    ax.set(title="Loss over epochs", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.show()


def _train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: str,
    print_every: int = 1,
):
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(epochs):
        # training
        model, train_loss = _run_epoch(
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = _run_epoch(
                data_loader=valid_loader,
                model=model,
                criterion=criterion,
                device=device,
            )
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = _get_accuracy(model, train_loader, device=device)
            valid_acc = _get_accuracy(model, valid_loader, device=device)

            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Train accuracy: {100 * train_acc:.2f}\t"
                f"Valid accuracy: {100 * valid_acc:.2f}"
            )

    _plot_losses(train_losses, valid_losses)


def _main():
    # Test run on MNIST dataset.

    # Hyper-parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    batch_size = 64
    learning_rate = 0.001
    seed = 42
    num_classes = 10
    num_epochs = 10

    torch.manual_seed(seed)

    # Prepare training data
    transforms = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    train_dataset = D.MNIST(root="/tmp/mnist_data", train=True, transform=transforms, download=True)
    valid_dataset = D.MNIST(root="/tmp/mnist_data", train=False, transform=transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    # Set up model
    model = LeNet5(num_classes=num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    _train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=num_epochs,
        device=device,
    )


if __name__ == "__main__":
    _main()
