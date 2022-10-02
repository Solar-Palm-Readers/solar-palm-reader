import os

import pandas as pd

data_root = r'data'


def load_dataframe(dataset: str) -> pd.DataFrame:
    dataset = dataset.replace(' ', '_')
    dataset_path = os.path.join(data_root, f'{dataset}.dat')
    return pd.read_pickle(dataset_path)
