import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import get_image


def display_n_images(df: pd.DataFrame, n_rows: int, n_columns):
    width = n_columns * 4
    height = n_rows * 5

    _, ax = plt.subplots(n_rows, n_columns, figsize=(width, height))
    ax = ax.ravel()

    for i in range(25):
        rand = np.random.choice(len(df))
        instance = df.iloc[rand]
        image = get_image(instance)

        title = f"Gender: {instance['gender']}, Age: {instance['age']}"
        ax[i].imshow(image, cmap='gray')
        ax[i].set_title(title)

    plt.show()
