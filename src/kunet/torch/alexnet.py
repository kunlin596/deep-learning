import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Dense, Dropout, MaxPool2d, Module, ReLU, Softmax


class AlexNet(Module):
    def __init__(self) -> None:
        Module.__init__(self)
        # NOTE: input size 227x227x3

        self._feature_extractor = nn.Sequential(
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self._classifier = nn.Sequential(
            Dense(in_features=9216, out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Dense(in_features=4096, out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Dense(in_features=4096, out_features=1000),
            Softmax(dim=1),
        )

    @property
    def feature_extractor(self) -> nn.Module:
        return self._feature_extractor

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    def forward(self, x) -> tuple[Tensor, Tensor]:
        features = self._feature_extractor(x)
        flattened = torch.flatten(features, 1)
        logits = self._classifier(flattened)
        probs = F.softmax(logits, dim=1)
        return logits, probs
