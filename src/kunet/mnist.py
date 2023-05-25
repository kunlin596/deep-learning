from __future__ import annotations

import pathlib

import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as D
from torchvision import transforms as T

from kunet import util
from kunet.layer import ConvLayer2D, CrossEntropyLoss
from kunet.models import CNN

np.set_printoptions(suppress=True)


def main():
    # Hyper-parameters
    # model_layer_sizes = [784, 256, 64, 10]
    batch_size = 128
    num_epochs = 3
    learning_rate = 0.1

    # Set up dataset
    dataset = D.MNIST(
        root=pathlib.Path("/tmp/dataset"),
        download=True,
        transform=T.ToTensor(),
    )

    training_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.2, 0.1])

    print("training:  ", len(training_dataset))
    print("validation:", len(validation_dataset))
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = CNN(
        input_channels=1,
        input_width=28,
        input_height=28,
        conv_sizes=[
            ConvLayer2D.ConvSize(5, 5),
            ConvLayer2D.ConvSize(3, 3),
        ],
    )

    # Initialize model
    # model = MLPClassifier(
    #     layer_sizes=model_layer_sizes,
    #     activations=[
    #         functional.HyperbolicTangent(),
    #         functional.HyperbolicTangent(),
    #         functional.Softmax(),
    #     ],
    # )

    loss_fn = CrossEntropyLoss()
    print(model)

    # Train
    # num_features = 28 * 28
    num_classes = 10
    for epoch in range(num_epochs):
        print(f"--- EPOCH {epoch} ---")
        running_loss = 0.0
        current_count = 0
        for batch, (image, label) in enumerate(training_loader):
            # curr_batch_size = min(image.shape[0], batch_size)
            # X = np.asarray(image).reshape(curr_batch_size, 1, num_features)
            # X = np.transpose(X, [2, 1, 0]).squeeze()
            X = image
            y = util.one_hot(label, num_classes)

            # Forward
            y_hat = model.forward(X)
            loss = loss_fn(y_hat, y)
            model.backward(loss_fn.backward())
            model.step(learning_rate)

            predictions = y_hat.argmax(axis=0)
            current_count += np.count_nonzero(predictions == label.numpy())

            if batch % 10 == 0:
                print(f"epoch: {epoch}, batch: {batch:4d}, loss={loss:10.7f}")
        running_loss += loss
        print(f"running loss {running_loss}, accuracy={current_count/len(training_loader.dataset):7.3f}.")

        if running_loss < 1.0:
            learning_rate *= 0.1

        from IPython import embed

        embed()


if __name__ == "__main__":
    main()
