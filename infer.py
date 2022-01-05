import hydra
import argparse
from PIL import Image
import torch.nn.functional as F


from omegaconf import OmegaConf
import pytorch_lightning as pl
from src.models.module import PLModule
from src.datamodules.age.constants import INDEX_2_LABEL


def predict(args: argparse.Namespace):
    config = OmegaConf.load(args.config_path)

    # Load model
    backbone: pl.LightningModule = hydra.utils.instantiate(config.model)
    model = PLModule.load_from_checkpoint(args.checkpoint_path, model=backbone)

    # Load image
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule)
    image = Image.open(args.image_path)
    image = datamodule.transform(image)

    # Predict
    outputs = F.softmax(model(image.unsqueeze(0)), dim=1)
    group_idx = outputs.argmax().item()
    group_name = INDEX_2_LABEL[group_idx]

    print(f'Predicted age group is: {group_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)

    args = parser.parse_args()
    predict(args)
