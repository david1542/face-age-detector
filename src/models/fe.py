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

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)


# Utility functions
def freeze_model(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def patch_backbone(backbone_name: str, backbone: str, output_size: int):
    if backbone_name == 'resnet18':
        input_size = backbone.fc.in_features
        backbone.fc = nn.Linear(input_size, output_size)
    elif backbone_name == 'alexnet':
        input_size = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Linear(input_size, output_size)
    elif backbone_name == 'vgg16':
        input_size = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Linear(input_size, output_size)
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
    # elif backbone_name == 'squeezenet':
    #     return models.squeezenet1_0
    # elif backbone_name == 'densenet':
    #     return models.densenet161
    # elif backbone_name == 'inception':
    #     return models.inception_v3
    # elif backbone_name == 'googlenet':
    #     return models.googlenet
    # elif backbone_name == 'shufflenet':
    #     return models.shufflenet_v2_x1_0
    # elif backbone_name == 'mobilenet_v2':
    #     return models.mobilenet_v2
    # elif backbone_name == 'mobilenet_v3_large':
    #     return models.mobilenet_v3_large
    # elif backbone_name == 'mobilenet_v3_small':
    #     return models.mobilenet_v3_small
    # elif backbone_name == 'resnext50_32x4d':
    #     return models.resnext50_32x4d
    # elif backbone_name == 'wide_resnet50_2':
    #     return models.wide_resnet50_2
    # elif backbone_name == 'mnasnet':
    #     return models.mnasnet1_0
    # elif backbone_name == 'efficientnet_b0':
    #     return models.efficientnet_b0
    # elif backbone_name == 'efficientnet_b1':
    #     return models.efficientnet_b1
    # elif backbone_name == 'efficientnet_b2':
    #     return models.efficientnet_b2
    # elif backbone_name == 'efficientnet_b3':
    #     return models.efficientnet_b3
    # elif backbone_name == 'efficientnet_b4':
    #     return models.efficientnet_b4
    # elif backbone_name == 'efficientnet_b5':
    #     return models.efficientnet_b5
    # elif backbone_name == 'efficientnet_b6':
    #     return models.efficientnet_b6
    # elif backbone_name == 'efficientnet_b7':
    #     return models.efficientnet_b7
    # elif backbone_name == 'regnet_y_400mf':
    #     return models.regnet_y_400mf
    # elif backbone_name == 'regnet_y_800mf':
    #     return models.regnet_y_800mf
    # elif backbone_name == 'regnet_y_1_6gf':
    #     return models.regnet_y_1_6gf
    # elif backbone_name == 'regnet_y_3_2gf':
    #     return models.regnet_y_3_2gf
    # elif backbone_name == 'regnet_y_8gf':
    #     return models.regnet_y_8gf
    # elif backbone_name == 'regnet_y_16gf':
    #     return models.regnet_y_16gf
    # elif backbone_name == 'regnet_y_32gf':
    #     return models.regnet_y_32gf
    # elif backbone_name == 'regnet_x_400mf':
    #     return models.regnet_x_400mf
    # elif backbone_name == 'regnet_x_800mf':
    #     return models.regnet_x_800mf
    # elif backbone_name == 'regnet_x_1_6gf':
    #     return models.regnet_x_1_6gf
    # elif backbone_name == 'regnet_x_3_2gf':
    #     return models.regnet_x_3_2gf
    # elif backbone_name == 'regnet_x_8gf':
    #     return models.regnet_x_8gf
    # elif backbone_name == 'regnet_x_16gf':
    #     return models.regnet_x_16gf
    # elif backbone_name == 'regnet_x_32gf':
    #     return models.regnet_x_32gf
