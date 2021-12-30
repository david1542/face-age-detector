import numpy as np
import pandas as pd

from src.datamodules.age.constants import AGE_LABELS
from src.utils.data import clean_dataframe


def _get_limits():
    limits = []
    for label in AGE_LABELS:
        low, high = label.replace('(', '').replace(')', '').split(', ')
        limits.append((int(low), int(high)))
    return limits


def find_age_group(age: int):
    limits = _get_limits()
    for i, limit in enumerate(limits):
        if age >= limit[0] and age <= limit[1]:
            return AGE_LABELS[i]

    # In case we didn't find hard coded age group, try to find
    # the nearest group by calculating distances
    midpoints = np.array([np.mean([low, high]) for low, high in limits])
    return AGE_LABELS[np.argmin(np.abs(midpoints - age))]

# Functions that preprocess age values


def map_numeric_values(df: pd.DataFrame):
    df_mapped = df.copy()

    numeric_mask = ~df_mapped['age'].str.contains('\(')
    numeric_values = df_mapped.loc[numeric_mask, 'age']
    df_mapped.loc[numeric_mask, 'age'] = numeric_values.astype(
        int).apply(find_age_group)
    return df_mapped


def map_group_values(df: pd.DataFrame):
    map = {
        '(27, 32)': '(25, 32)',
        '(38, 42)': '(38, 43)',
        '(38, 48)': '(38, 43)',
        '(8, 12)': '(8, 13)'
    }

    df_mapped = df.copy()
    df_mapped['age'].replace(map, inplace=True)
    df_mapped.drop(df_mapped.loc[df_mapped['age']
                   == '(8, 23)'].index, inplace=True)
    return df_mapped


def preprocess_metadata(df: pd.DataFrame):
    # Clean rows which have NaN in their age column
    df_processed = clean_dataframe(df)
    df_processed = df.loc[~df['age'].isnull()]

    # Preprocess the age column
    df_processed = map_group_values(df_processed)
    df_processed = map_numeric_values(df_processed)
    return df_processed
