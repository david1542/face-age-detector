import os

import pandas as pd
import numpy as np
from PIL import Image

from constants import RAW_DATA_PATH, ALIGNED_FACES_PATH, REGULAR_FACES_PATH


def clean_dataframe(df: pd.DataFrame):
    df = df.applymap(lambda x: x if x != 'None' else np.nan)
    df = df.reset_index(drop=True)
    return df


def load_fold(fold: int, frontal: bool = False):
    file_name = f'fold_frontal_{fold}_data.txt' if frontal else f'fold_{fold}_data.txt'
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, file_name), sep='\t')
    return df


def get_train_valid_sets(valid_fold_num: int, frontal: bool = False):
    # Load train set
    train_set = pd.concat([load_fold(i, frontal)
                          for i in range(5) if i != valid_fold_num])
    train_set = clean_dataframe(train_set)

    # Load valid set
    valid_set = load_fold(valid_fold_num, frontal)
    valid_set = clean_dataframe(valid_set)

    return train_set, valid_set


def build_image_path(instance, aligned: bool = False):
    images_path = ALIGNED_FACES_PATH if aligned else REGULAR_FACES_PATH
    prefix = 'landmark_aligned_face' if aligned else 'coarse_tilt_aligned_face'
    file_name = f"{prefix}.{instance['face_id']}.{instance['original_image']}"
    file_path = os.path.join(images_path, instance['user_id'], file_name)
    return file_path


def get_image(instance, aligned: bool = False):
    return Image.open(build_image_path(instance, aligned))
