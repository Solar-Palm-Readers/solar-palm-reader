import os

import pandas as pd

data_root = r'data'


def load_saved_dataframe(dataset: str) -> pd.DataFrame:
    """
    Load a complete saved pandas dataset.
    """
    dataset = dataset.replace(' ', '_')
    dataset_path = os.path.join(data_root, f'{dataset}.dat')
    return pd.read_pickle(dataset_path)


def create_channel_columns(df_dscover: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by epoch time, extract all recording channels and insert them into new columns.
    """
    raise NotImplemented()


def create_complete_dataset() -> pd.DataFrame:
    """
    Concatenate the DSCOVR FC0 dataset with the warped Wind dataframe by cross-checking the epoch times.
    """
    raise NotImplemented()
