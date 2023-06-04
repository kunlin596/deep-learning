import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Dense, Dropout, MaxPool2d, Module, ReLU


class VGG16(Module):
    def __init__(self) -> None:
        Module.__init__(self)
        # NOTE: input size 224x224x3
        # ~138m parameters

        self._feature_extractor = nn.Sequential(
            # conv 64 x 2
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # conv 128 x 2
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # conv 256 x 3
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # conv 512 x 3
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # conv 512 x 3
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )
        # 7 x 7 x 512
        self._classifier = nn.Sequential(
            Dense(in_features=4096, out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Dense(in_features=4096, out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Dense(in_features=4096, out_features=1000),
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
