import importlib
import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: str, output_size: int):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)

        input_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_size, output_size)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)
