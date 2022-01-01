import os
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from src.utils.transform import create_transform
from src.utils.data import get_train_valid_sets, get_image
from src.datamodules.age.utils import preprocess_metadata
from src.datamodules.age.constants import LABEL_2_INDEX


class FaceAgeDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, transform: Compose, aligned: bool = False):
        self.metadata = metadata
        self.transform = transform
        self.aligned = aligned

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        instance = self.metadata.iloc[index]

        # Get image and transform it
        image = get_image(instance, aligned=self.aligned)
        image = self.transform(image)

        # Get label and map it to its index
        label = LABEL_2_INDEX[instance['age']]

        return image, label


class AgeDataModule(pl.LightningDataModule):
    def __init__(self, fold: 0, batch_size: int, image_width: int, image_height: int):
        self.fold = fold
        self.datasets = {}
        self.batch_size = batch_size
        self.cpu_count = os.cpu_count()
        self.transform = create_transform(image_width, image_height)

    def setup(self, *args, **kwargs):
        train_set, valid_set = get_train_valid_sets(self.fold)
        train_set = preprocess_metadata(train_set)
        valid_set = preprocess_metadata(valid_set)

        self.datasets['train'] = FaceAgeDataset(
            train_set, transform=self.transform)
        self.datasets['valid'] = FaceAgeDataset(
            valid_set, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets['train'], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets['valid'], batch_size=self.batch_size)
