import pandas as pd

from utils.data import clean_dataframe


def preprocess_metadata(df: pd.DataFrame):
    df_processed = df.copy()
    df_processed = clean_dataframe(df)
    df_processed = df_processed.loc[~df_processed['gender'].isnull()]

    return df_processed
