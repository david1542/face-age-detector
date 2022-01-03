import importlib
import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone_name: str, output_size: int):
        super().__init__()
        model_fn = get_backbone_fn(backbone_name)
        self.backbone = model_fn(pretrained=True)

        freeze_model(self.backbone)

        # Send backbone to a generic function that patches the final layer
        patch_backbone(backbone_name=backbone_name,
                       backbone=self.backbone, output_size=output_size)

    def forward(self, images):
        return self.backbone(images)


# Utility functions
def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def create_classifier(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, output_size)
    )


def patch_backbone(backbone_name: str, backbone: str, output_size: int):
    if backbone_name == 'resnet18':
        input_size = backbone.fc.in_features
        backbone.fc = create_classifier(input_size, output_size)
    elif backbone_name == 'alexnet':
        input_size = backbone.classifier[6].in_features
        backbone.classifier[6] = create_classifier(input_size, output_size)
    elif backbone_name == 'vgg16':
        input_size = backbone.classifier[6].in_features
        backbone.classifier[6] = create_classifier(input_size, output_size)
    else:
        raise Exception(
            f'Could not find model with name {backbone_name} or it is not supported.')


def get_backbone_fn(backbone_name: str):
    if backbone_name == 'resnet18':
        return models.resnet18
    elif backbone_name == 'alexnet':
        return models.alexnet
    elif backbone_name == 'vgg16':
        return models.vgg16

    raise Exception(
        f'Could not find model with name {backbone_name} or it is not supported.')
